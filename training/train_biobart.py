"""
PP1 Task B: BioBART Multi-Modal Report Generator (FINAL)
================================================================
Connects the trained ConvNeXt-Base vision model to a BioBART language model.
Treats the 7x7 spatial feature map as a sequence of 49 "visual words",
allowing BioBART to attend to specific regions while writing the report.

Hardware target: RTX 4060 8GB VRAM, 32GB RAM, Ryzen 7 8845HS
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ML_DATASET_DIR = DATA_DIR / "output" / "multi_label_dataset"
IMAGE_ROOT = DATA_DIR / "images" / "images"
CV_MODEL_DIR = PROJECT_ROOT / "models" / "multilabel_convnext"
OUTPUT_DIR = PROJECT_ROOT / "models" / "biobart_report"

# Hyperparameters — tuned for RTX 4060 8GB VRAM
BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 8   # Effective batch size = 4 * 8 = 32
IMAGE_SIZE = 224
NUM_EPOCHS = 15
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
MAX_TEXT_LEN = 512             # Full findings + impression
NUM_WORKERS = 4
PATIENCE = 5                   # Early stopping patience
SEQ_LEN_VISUAL = 49            # 7x7 spatial grid from ConvNeXt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BART_MODEL_NAME = "GanjinZero/biobart-base"


# ============================================================================
# DATASET
# ============================================================================

class ReportDataset(Dataset):
    """
    Multi-modal dataset: loads (image, report_text) pairs.
    Images come from the flat directory structure.
    Report text is constructed from findings + impression columns.
    """

    def __init__(self, csv_path, image_root, split_name, tokenizer, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.split_name = split_name
        self.transform = transform
        self.tokenizer = tokenizer

        # Build image path lookup: scan positive/ and negative/ subdirs
        self.image_paths = {}
        for cls_dir in ["positive", "negative"]:
            dir_path = self.image_root / split_name / cls_dir
            if dir_path.exists():
                for fname in os.listdir(dir_path):
                    if fname.endswith(".jpg"):
                        dicom_id = fname.replace(".jpg", "")
                        self.image_paths[dicom_id] = str(dir_path / fname)

        # Filter to only rows where we have images
        valid_mask = self.df["dicom_id"].isin(self.image_paths)
        n_before = len(self.df)
        self.df = self.df[valid_mask].reset_index(drop=True)
        n_after = len(self.df)

        if n_before != n_after:
            print(f"  [!] {split_name}: {n_before - n_after} rows missing images, "
                  f"keeping {n_after}/{n_before}")

        print(f"  {split_name}: {len(self.df)} samples loaded, "
              f"{len(self.image_paths)} images found")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load image (PIL already imported at module level)
        img_path = self.image_paths[row["dicom_id"]]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 2. Build target text (clean NaN values)
        findings = str(row["findings_text"]) if pd.notna(row["findings_text"]) else ""
        impression = str(row["impression_text"]) if pd.notna(row["impression_text"]) else ""

        if findings and impression:
            text = f"FINDINGS: {findings} IMPRESSION: {impression}"
        elif impression:
            text = f"IMPRESSION: {impression}"
        elif findings:
            text = f"FINDINGS: {findings}"
        else:
            text = "No report available."

        # 3. Tokenize text
        encoded = self.tokenizer(
            text,
            max_length=MAX_TEXT_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Labels: same as input_ids but padding tokens replaced with -100
        # so cross-entropy loss ignores padding positions
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return image, input_ids, attention_mask, labels


# ============================================================================
# ARCHITECTURE — The Vision-to-Language Bridge
# ============================================================================

class CXRToTextModel(nn.Module):
    """
    Multi-modal model:
      ConvNeXt-Base (frozen) → Linear bridge → BioBART decoder

    The ConvNeXt 7x7 feature map is treated as 49 "visual tokens"
    which BioBART cross-attends to while generating text.
    """

    def __init__(self, convnext_ckpt_path):
        super().__init__()

        # ---- 1. VISUAL ENCODER (ConvNeXt-Base, frozen) ----
        print("  -> Loading ConvNeXt backbone...")
        self.vision_encoder = models.convnext_base(weights=None)

        # Remove classifier and pooling to get raw spatial features
        # Output shape: (Batch, 1024, 7, 7) for 224x224 input
        self.vision_encoder.avgpool = nn.Identity()
        self.vision_encoder.classifier = nn.Identity()

        # Load trained CXR weights from v1 best_model.pth
        if os.path.exists(convnext_ckpt_path):
            ckpt = torch.load(convnext_ckpt_path, weights_only=False)
            state_dict = ckpt["model_state_dict"]

            # Filter: only load backbone.features.* keys (skip classifier head)
            # and strip the "backbone." prefix to match vanilla convnext_base keys
            filtered_dict = {
                k.replace("backbone.", ""): v
                for k, v in state_dict.items()
                if "backbone.features" in k
            }

            missing, unexpected = self.vision_encoder.load_state_dict(
                filtered_dict, strict=False
            )
            n_loaded = len(filtered_dict)
            print(f"  -> ConvNeXt CXR weights loaded ({n_loaded} tensors, "
                  f"{len(missing)} missing, {len(unexpected)} unexpected)")
        else:
            print("  [!] WARNING: ConvNeXt checkpoint not found! Using random weights.")

        # Freeze entire vision encoder — no gradients
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.vision_encoder.eval()

        # ---- 2. THE BRIDGE (1024 → 768) ----
        # Projects ConvNeXt features to BioBART's hidden dimension
        self.bridge = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ---- 3. TEXT DECODER (BioBART) ----
        print(f"  -> Loading {BART_MODEL_NAME}...")
        self.text_model = BartForConditionalGeneration.from_pretrained(BART_MODEL_NAME)

    def _encode_images(self, images):
        """
        Convert images → visual token sequence.
        Returns: (hidden_states, attention_mask) both on same device as images.
        """
        with torch.no_grad():
            # Shape: (B, 1024, 7, 7)
            features = self.vision_encoder(images)

        B, C, H, W = features.shape
        # Reshape to (B, 49, 1024) — sequence of spatial visual tokens
        features = features.view(B, C, H * W).permute(0, 2, 1)

        # Bridge to BART dimension: (B, 49, 768)
        hidden_states = self.bridge(features)

        # All 49 visual tokens are valid — create all-ones attention mask
        # This is CRITICAL for BART cross-attention to work correctly
        attention_mask = torch.ones(B, H * W, dtype=torch.long, device=images.device)

        return hidden_states, attention_mask

    def forward(self, images, labels=None, decoder_input_ids=None):
        """Training forward pass."""
        hidden_states, enc_attention_mask = self._encode_images(images)

        encoder_outputs = BaseModelOutput(last_hidden_state=hidden_states)

        outputs = self.text_model(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_attention_mask,   # FIX: explicit encoder attention mask
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )

        return outputs.loss, outputs.logits

    @torch.no_grad()
    def generate_report(self, images, tokenizer, max_length=512):
        """Inference: autoregressively generate a report from an image."""
        hidden_states, enc_attention_mask = self._encode_images(images)

        encoder_outputs = BaseModelOutput(last_hidden_state=hidden_states)

        generated_ids = self.text_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_attention_mask,    # FIX: explicit encoder attention mask
            max_length=max_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def main():
    print("=" * 70)
    print("PP1 TASK B: BioBART Multi-Modal Report Generator (FINAL)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE} x {GRAD_ACCUMULATION_STEPS} acc = "
          f"{BATCH_SIZE * GRAD_ACCUMULATION_STEPS} effective")
    print(f"Max text length: {MAX_TEXT_LEN} tokens")
    print(f"Epochs: {NUM_EPOCHS} (patience={PATIENCE})")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Tokenizer ----
    print("[1/4] Loading BioBART tokenizer...")
    tokenizer = BartTokenizer.from_pretrained(BART_MODEL_NAME)

    # ---- 2. Datasets ----
    print("\n[2/4] Loading datasets...")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ReportDataset(
        ML_DATASET_DIR / "ml_train.csv", IMAGE_ROOT, "Train", tokenizer, transform
    )
    val_dataset = ReportDataset(
        ML_DATASET_DIR / "ml_val.csv", IMAGE_ROOT, "Val", tokenizer, transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ---- 3. Model ----
    print("\n[3/4] Building multi-modal bridge architecture...")
    model = CXRToTextModel(convnext_ckpt_path=CV_MODEL_DIR / "best_model.pth")
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable (Bridge+BART): {trainable_params:,}")
    print(f"  Frozen (ConvNeXt):    {frozen_params:,}")

    # ---- 4. Optimizer & Scheduler ----
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # ---- Training ----
    print(f"\n[4/4] Training for {NUM_EPOCHS} epochs...")
    print("-" * 70)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # ================ TRAIN ================
        model.train()
        # Keep ConvNeXt in eval mode always (BatchNorm / DropPath behavior)
        model.vision_encoder.eval()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"  Train Ep {epoch:02d}", unit="batch", leave=False)
        for i, (images, input_ids, attention_mask, labels) in enumerate(pbar):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.amp.autocast("cuda"):
                loss, _ = model(images, labels=labels)
                loss = loss / GRAD_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * GRAD_ACCUMULATION_STEPS
            pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUMULATION_STEPS:.4f}")

        # Flush any residual accumulated gradients
        # (handles case where len(train_loader) % GRAD_ACCUMULATION_STEPS != 0)
        if (i + 1) % GRAD_ACCUMULATION_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)

        # ================ VALIDATE ================
        model.eval()
        val_loss = 0.0

        pbar = tqdm(val_loader, desc=f"  Val   Ep {epoch:02d}", unit="batch", leave=False)
        with torch.no_grad():
            for images, input_ids, attention_mask, labels in pbar:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                with torch.amp.autocast("cuda"):
                    loss, _ = model(images, labels=labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"TrL: {avg_train_loss:.4f} | "
              f"VaL: {avg_val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"{epoch_time:.0f}s")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val_loss,
            }
            torch.save(checkpoint, OUTPUT_DIR / "best_model.pth")
            print(f"  --> Best model saved! (Val Loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  [!] Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

        # Generate a sample report every 3 epochs (clear VRAM first)
        if epoch % 3 == 0 or epoch == 1:
            try:
                torch.cuda.empty_cache()
                # Use a fresh image from the val set
                sample_batch = next(iter(val_loader))
                sample_img = sample_batch[0][0:1].to(DEVICE)
                sample_labels = sample_batch[3][0]

                pred_text = model.generate_report(sample_img, tokenizer, max_length=200)[0]
                true_text = tokenizer.decode(
                    sample_labels[sample_labels != -100], skip_special_tokens=True
                )
                print(f"      TRUE : {true_text[:120]}...")
                print(f"      PRED : {pred_text[:120]}...")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"      [!] Sample generation failed: {e}")

    print("-" * 70)

    # ---- Save training history ----
    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    print(f"\nHistory: {OUTPUT_DIR / 'training_history.csv'}")
    print(f"Model:   {OUTPUT_DIR / 'best_model.pth'}")

    # ---- Final: Generate sample reports from validation set ----
    print(f"\n{'=' * 70}")
    print("SAMPLE GENERATED REPORTS (from best model)")
    print(f"{'=' * 70}")

    # Load the best model
    best_ckpt = torch.load(OUTPUT_DIR / "best_model.pth", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded best model from epoch {best_ckpt['epoch']} "
          f"(Val Loss: {best_ckpt['val_loss']:.4f})")

    torch.cuda.empty_cache()
    sample_iter = iter(val_loader)
    for s in range(3):
        try:
            batch = next(sample_iter)
            img = batch[0][0:1].to(DEVICE)
            lbl = batch[3][0]

            pred = model.generate_report(img, tokenizer, max_length=256)[0]
            true = tokenizer.decode(lbl[lbl != -100], skip_special_tokens=True)

            print(f"\n  --- Sample {s + 1} ---")
            print(f"  TRUE: {true[:200]}")
            print(f"  PRED: {pred[:200]}")
        except StopIteration:
            break
        except Exception as e:
            print(f"  [!] Sample {s + 1} failed: {e}")

    print(f"\n{'=' * 70}")
    print("[DONE] BioBART training complete!")


if __name__ == "__main__":
    main()

"""
PP1 Task A v2: Multi-Label CXR Classification with ConvNeXt-Base
================================================================
Two-phase transfer learning:
  Phase 1: Freeze backbone, train classifier head only (10 epochs)
  Phase 2: Unfreeze last 2 stages, fine-tune with low LR (20 epochs)

Fixes overfitting from v1 by controlling gradient flow.
Hardware target: RTX 4060 8GB VRAM
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ML_DATASET_DIR = DATA_DIR / "output" / "multi_label_dataset"
IMAGE_ROOT = DATA_DIR / "images" / "images"
OUTPUT_DIR = PROJECT_ROOT / "models" / "multilabel_convnext"

IMAGE_SIZE = 224
NUM_WORKERS = 4

# Phase 1: Frozen backbone (only classifier trains)
PHASE1_BATCH_SIZE = 64       # Can be larger since no backbone gradients
PHASE1_EPOCHS = 10
PHASE1_LR = 5e-4             # Higher LR is OK for head-only training

# Phase 2: Unfreeze last 2 stages + fine-tune
PHASE2_BATCH_SIZE = 24       # Smaller for full gradient computation
PHASE2_EPOCHS = 20
PHASE2_LR = 5e-6             # Very low LR to prevent overwriting features
PHASE2_PATIENCE = 8

LABEL_COLS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion",
    "Lung_Opacity", "No_Finding", "Pleural_Effusion",
    "Pleural_Other", "Pneumonia", "Pneumothorax", "Support_Devices",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# DATASET
# ============================================================================

class CXRMultiLabelDataset(Dataset):
    """Multi-label CXR dataset with -1 masking for uncertain labels."""

    def __init__(self, csv_path, image_root, split_name, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.split_name = split_name
        self.transform = transform

        # Build flat image path lookup
        self.image_paths = {}
        for cls_dir in ["positive", "negative"]:
            dir_path = self.image_root / split_name / cls_dir
            if dir_path.exists():
                for fname in os.listdir(dir_path):
                    dicom_id = fname.replace(".jpg", "")
                    self.image_paths[dicom_id] = str(dir_path / fname)

        # Filter rows with images
        valid_mask = self.df["dicom_id"].isin(self.image_paths)
        n_before = len(self.df)
        self.df = self.df[valid_mask].reset_index(drop=True)
        n_after = len(self.df)

        if n_before != n_after:
            print(f"  [!] {split_name}: {n_before - n_after} rows missing images, "
                  f"keeping {n_after}/{n_before}")

        self.labels = self.df[LABEL_COLS].values.astype(np.float32)
        self.pos_weights = self._compute_pos_weights()
        print(f"  {split_name}: {len(self)} samples, {len(self.image_paths)} images")

    def _compute_pos_weights(self):
        weights = []
        for i, col in enumerate(LABEL_COLS):
            known_mask = self.labels[:, i] != -1
            known_labels = self.labels[known_mask, i]
            n_pos = (known_labels == 1).sum()
            n_neg = (known_labels == 0).sum()
            w = n_neg / n_pos if n_pos > 0 else 1.0
            weights.append(min(w, 10.0))
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        from PIL import Image
        image = Image.open(self.image_paths[row["dicom_id"]]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = self.labels[idx].copy()
        mask = (labels != -1).astype(np.float32)
        labels = np.clip(labels, 0, 1)
        return image, torch.tensor(labels), torch.tensor(mask)


# ============================================================================
# MODEL
# ============================================================================

class CXRConvNeXt(nn.Module):
    """ConvNeXt-Base with separate backbone and head for staged training."""

    def __init__(self, num_classes=14):
        super().__init__()
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        self.backbone = models.convnext_base(weights=weights)

        # Replace classifier with stronger head
        in_features = self.backbone.classifier[2].in_features  # 1024
        self.backbone.classifier[2] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def freeze_backbone(self):
        """Freeze all layers except the classifier head."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def unfreeze_last_stages(self):
        """Unfreeze stages 6 and 7 (last 2 ConvNeXt stages) + classifier."""
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze classifier (head)
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

        # Unfreeze last 2 feature stages (stages 2 and 3 = indices 4,5 in features)
        # ConvNeXt features: [0]=stem, [1]=stage1, [2]=ds1, [3]=stage2, [4]=ds2, [5]=stage3, [6]=ds3, [7]=stage4
        for i in [5, 6, 7]:  # stage3, downsample3, stage4
            if i < len(self.backbone.features):
                for param in self.backbone.features[i].parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def masked_bce_loss(logits, targets, mask, pos_weight=None):
    """Masked BCE loss - ignores uncertain labels (mask=0)."""
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(logits.device), reduction='none'
        )
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')

    loss = criterion(logits, targets) * mask
    return loss.sum() / mask.sum() if mask.sum() > 0 else loss.sum() * 0


def compute_auroc(all_targets, all_preds, all_masks):
    """Per-class AUROC, skipping classes with insufficient data."""
    aurocs = {}
    for i, col in enumerate(LABEL_COLS):
        m = all_masks[:, i] == 1
        if m.sum() < 10:
            aurocs[col] = float('nan')
            continue
        t, p = all_targets[m, i], all_preds[m, i]
        if len(np.unique(t)) < 2:
            aurocs[col] = float('nan')
            continue
        try:
            aurocs[col] = roc_auc_score(t, p)
        except Exception:
            aurocs[col] = float('nan')
    return aurocs


def train_one_epoch(model, dataloader, optimizer, pos_weight, scaler, desc="Train"):
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(dataloader, desc=f"  {desc}", unit="batch", leave=False)
    for images, labels, masks in pbar:
        images, labels, masks = images.to(DEVICE), labels.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            loss = masked_bce_loss(model(images), labels, masks, pos_weight)
        scaler.scale(loss).backward()

        # Gradient clipping to prevent instability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, dataloader, pos_weight):
    model.eval()
    total_loss, n = 0.0, 0
    all_t, all_p, all_m = [], [], []
    pbar = tqdm(dataloader, desc="  Val  ", unit="batch", leave=False)
    for images, labels, masks in pbar:
        images, labels, masks = images.to(DEVICE), labels.to(DEVICE), masks.to(DEVICE)
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = masked_bce_loss(logits, labels, masks, pos_weight)
        all_t.append(labels.cpu().numpy())
        all_p.append(torch.sigmoid(logits).cpu().numpy())
        all_m.append(masks.cpu().numpy())
        total_loss += loss.item()
        n += 1

    all_t = np.concatenate(all_t)
    all_p = np.concatenate(all_p)
    all_m = np.concatenate(all_m)
    return total_loss / max(n, 1), compute_auroc(all_t, all_p, all_m), all_t, all_p, all_m


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PP1 TASK A v2: ConvNeXt-Base (Two-Phase Transfer Learning)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Phase 1: Frozen backbone, head-only, {PHASE1_EPOCHS} epochs, LR={PHASE1_LR}")
    print(f"Phase 2: Unfreeze last stages, {PHASE2_EPOCHS} epochs, LR={PHASE2_LR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Transforms ----
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ---- Datasets ----
    print("[1/6] Loading datasets...")
    train_ds = CXRMultiLabelDataset(ML_DATASET_DIR / "ml_train.csv", IMAGE_ROOT, "Train", train_transform)
    val_ds = CXRMultiLabelDataset(ML_DATASET_DIR / "ml_val.csv", IMAGE_ROOT, "Val", val_transform)
    pos_weight = train_ds.pos_weights

    print(f"\n[2/6] Class weights:")
    for i, col in enumerate(LABEL_COLS):
        print(f"  {col:30s} {pos_weight[i]:.2f}x")

    # ---- Model ----
    print(f"\n[3/6] Building ConvNeXt-Base model...")
    model = CXRConvNeXt(num_classes=len(LABEL_COLS))
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ==================================================================
    # PHASE 1: FROZEN BACKBONE -- TRAIN HEAD ONLY
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 1: Frozen Backbone (Head-Only Training)")
    print(f"{'=' * 70}")

    model.freeze_backbone()
    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_p1:,} / {total_params:,} "
          f"({100*trainable_p1/total_params:.1f}%)")

    train_loader = DataLoader(
        train_ds, batch_size=PHASE1_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=PHASE1_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR, weight_decay=0.01
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    best_auroc = 0.0
    history = []

    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, pos_weight, scaler, "P1-Train")
        val_loss, aurocs, _, _, _ = evaluate(model, val_loader, pos_weight)
        scheduler.step()

        valid_a = [v for v in aurocs.values() if not np.isnan(v)]
        mean_a = np.mean(valid_a) if valid_a else 0.0
        pneu_a = aurocs.get("Pneumonia", float('nan'))

        print(f"  P1 Epoch {epoch:02d}/{PHASE1_EPOCHS} | "
              f"TrLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
              f"AUROC: {mean_a:.4f} | Pneumonia: {pneu_a:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | {time.time()-t0:.0f}s")

        history.append({"phase": 1, "epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "mean_auroc": mean_a, "pneumonia_auroc": pneu_a})

        if mean_a > best_auroc:
            best_auroc = mean_a
            torch.save({
                "epoch": epoch, "phase": 1,
                "model_state_dict": model.state_dict(),
                "mean_auroc": mean_a, "aurocs": aurocs,
                "label_cols": LABEL_COLS, "pos_weights": pos_weight,
            }, OUTPUT_DIR / "best_model.pth")
            print(f"  --> Best model saved! (AUROC: {mean_a:.4f})")

    # ==================================================================
    # PHASE 2: UNFREEZE LAST STAGES -- FINE-TUNE
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 2: Unfreeze Last Stages (Fine-Tuning)")
    print(f"{'=' * 70}")

    # Load best Phase 1 model
    ckpt = torch.load(OUTPUT_DIR / "best_model.pth", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded best Phase 1 model (epoch {ckpt['epoch']}, AUROC {ckpt['mean_auroc']:.4f})")

    # Unfreeze last stages
    model.unfreeze_last_stages()
    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_p2:,} / {total_params:,} "
          f"({100*trainable_p2/total_params:.1f}%)")

    # Smaller batch for larger gradient memory
    train_loader = DataLoader(
        train_ds, batch_size=PHASE2_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=PHASE2_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Different LR for backbone vs head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

    optimizer = AdamW([
        {"params": backbone_params, "lr": PHASE2_LR},          # Very low for backbone
        {"params": head_params, "lr": PHASE2_LR * 10},         # 10x higher for head
    ], weight_decay=0.02)

    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda")

    patience_counter = 0

    for epoch in range(1, PHASE2_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, pos_weight, scaler, "P2-Train")
        val_loss, aurocs, _, _, _ = evaluate(model, val_loader, pos_weight)
        scheduler.step()

        valid_a = [v for v in aurocs.values() if not np.isnan(v)]
        mean_a = np.mean(valid_a) if valid_a else 0.0
        pneu_a = aurocs.get("Pneumonia", float('nan'))

        print(f"  P2 Epoch {epoch:02d}/{PHASE2_EPOCHS} | "
              f"TrLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
              f"AUROC: {mean_a:.4f} | Pneumonia: {pneu_a:.4f} | "
              f"LR(bb): {optimizer.param_groups[0]['lr']:.2e} | {time.time()-t0:.0f}s")

        history.append({"phase": 2, "epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "mean_auroc": mean_a, "pneumonia_auroc": pneu_a})

        if mean_a > best_auroc:
            best_auroc = mean_a
            patience_counter = 0
            torch.save({
                "epoch": epoch, "phase": 2,
                "model_state_dict": model.state_dict(),
                "mean_auroc": mean_a, "aurocs": aurocs,
                "label_cols": LABEL_COLS, "pos_weights": pos_weight,
            }, OUTPUT_DIR / "best_model.pth")
            print(f"  --> Best model saved! (AUROC: {mean_a:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PHASE2_PATIENCE:
                print(f"\n  [!] Early stopping at P2 epoch {epoch}")
                break

    # ==================================================================
    # TEST EVALUATION
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("[6/6] Evaluating best model on TEST set...")
    print(f"{'=' * 70}")

    test_ds = CXRMultiLabelDataset(ML_DATASET_DIR / "ml_test.csv", IMAGE_ROOT, "Test", val_transform)
    test_loader = DataLoader(test_ds, batch_size=PHASE2_BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    ckpt = torch.load(OUTPUT_DIR / "best_model.pth", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded best model (Phase {ckpt['phase']}, Epoch {ckpt['epoch']})")

    test_loss, test_aurocs, _, _, _ = evaluate(model, test_loader, pos_weight)

    print(f"\n{'=' * 70}")
    print(f"TEST SET RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Condition':30s} {'AUROC':>8s}")
    print(f"{'-' * 42}")

    valid_test = []
    for col in LABEL_COLS:
        auc = test_aurocs.get(col, float('nan'))
        if not np.isnan(auc):
            valid_test.append(auc)
        s = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        tag = " <-- PRIMARY" if col == "Pneumonia" else ""
        print(f"  {col:30s} {s:>8s}{tag}")

    mean_test = np.mean(valid_test) if valid_test else 0.0
    print(f"{'-' * 42}")
    print(f"  {'MEAN AUROC':30s} {mean_test:.4f}")
    print(f"  {'Test Loss':30s} {test_loss:.4f}")
    print(f"{'=' * 70}")

    # Save history
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history_v2.csv", index=False)
    print(f"\nHistory: {OUTPUT_DIR / 'training_history_v2.csv'}")
    print(f"Model:   {OUTPUT_DIR / 'best_model.pth'}")
    print(f"\n[DONE] Two-phase training complete!")


if __name__ == "__main__":
    main()

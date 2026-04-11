"""
PP1 FINAL: ConvNeXt-Base Multi-Label CXR Classifier
====================================================
Root cause analysis of v1 and v2 failures:

v1 FAILURE: LR=1e-4 for ALL 87.6M params from epoch 1
  -> Backbone (pretrained) adapted TOO FAST
  -> Peak at epoch 5 (AUROC 0.862), then val_loss diverged 0.55 -> 0.68
  -> Test AUROC: 0.854 (good, but left performance on the table)

v2 FAILURE: Phase 2 LR=5e-6 for backbone
  -> Too conservative, backbone barely moved
  -> 20 epochs of Phase 2, AUROC only reached 0.836
  -> Never reached v1's peak because backbone couldn't adapt

SOLUTION (v3): Discriminative Learning Rates + OneCycleLR
  -> Early backbone layers (edges/textures): LR * 0.01
  -> Late backbone layers (high-level features): LR * 0.1
  -> Classifier head (task-specific): LR * 1.0
  -> OneCycleLR: Warmup phase prevents early overfitting
  -> Gradient clipping + proper weight decay prevent divergence

Expected: AUROC 0.87-0.90 without overfitting
Hardware: RTX 4060 8GB VRAM
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import roc_auc_score, average_precision_score
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
BATCH_SIZE = 32
NUM_EPOCHS = 25
NUM_WORKERS = 4
PATIENCE = 10

# Discriminative learning rates
HEAD_LR = 2e-4          # Classifier head: learn fast
LATE_BACKBONE_LR = 2e-5 # Late stages (5,6,7): adapt moderately
EARLY_BACKBONE_LR = 2e-6 # Early stages (0-4): barely touch

WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.02  # Soft targets: 0->0.02, 1->0.98
MAX_GRAD_NORM = 1.0

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
    """Multi-label CXR dataset with uncertain (-1) label masking."""

    def __init__(self, csv_path, image_root, split_name, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.transform = transform

        # Build flat image path lookup
        self.image_paths = {}
        for cls_dir in ["positive", "negative"]:
            dir_path = self.image_root / split_name / cls_dir
            if dir_path.exists():
                for fname in os.listdir(dir_path):
                    self.image_paths[fname.replace(".jpg", "")] = str(dir_path / fname)

        valid = self.df["dicom_id"].isin(self.image_paths)
        n_before = len(self.df)
        self.df = self.df[valid].reset_index(drop=True)
        if len(self.df) < n_before:
            print(f"  [!] {split_name}: {n_before - len(self.df)} missing images")

        self.labels = self.df[LABEL_COLS].values.astype(np.float32)
        self.pos_weights = self._compute_pos_weights()
        print(f"  {split_name}: {len(self)} samples loaded")

    def _compute_pos_weights(self):
        weights = []
        for i in range(len(LABEL_COLS)):
            known = self.labels[:, i] != -1
            kl = self.labels[known, i]
            n_pos, n_neg = (kl == 1).sum(), (kl == 0).sum()
            w = n_neg / n_pos if n_pos > 0 else 1.0
            weights.append(min(w, 10.0))
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.image_paths[self.df.iloc[idx]["dicom_id"]]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        labels = self.labels[idx].copy()
        mask = (labels != -1).astype(np.float32)

        # Label smoothing on known labels
        smooth_labels = np.where(labels == 1, 1.0 - LABEL_SMOOTHING,
                        np.where(labels == 0, LABEL_SMOOTHING, 0.0))

        return img, torch.tensor(smooth_labels, dtype=torch.float32), \
               torch.tensor(mask, dtype=torch.float32)


# ============================================================================
# MODEL
# ============================================================================

class CXRConvNeXt(nn.Module):
    """ConvNeXt-Base with proper classifier head."""

    def __init__(self, num_classes=14):
        super().__init__()
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        self.backbone = models.convnext_base(weights=weights)

        in_features = self.backbone.classifier[2].in_features  # 1024
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def get_param_groups(self):
        """
        Split parameters into 3 groups with discriminative learning rates:
          - early_backbone: features[0:5] (stem + stage1/2 + downsamples)
          - late_backbone:  features[5:8] (stage3/4 + downsamples)
          - head:           classifier
        """
        early_params = []
        late_params = []
        head_params = []

        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("classifier"):
                head_params.append(param)
            elif name.startswith("features"):
                # Extract feature index: features.X.Y.Z...
                parts = name.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    idx = int(parts[1])
                    if idx >= 5:  # stages 3, 4
                        late_params.append(param)
                    else:  # stem, stages 1, 2
                        early_params.append(param)
                else:
                    late_params.append(param)
            else:
                head_params.append(param)

        return [
            {"params": early_params, "lr": EARLY_BACKBONE_LR, "name": "early_backbone"},
            {"params": late_params, "lr": LATE_BACKBONE_LR, "name": "late_backbone"},
            {"params": head_params, "lr": HEAD_LR, "name": "head"},
        ]

    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def masked_bce_loss(logits, targets, mask, pos_weight):
    """Masked BCE with pos_weight -- ignores uncertain labels."""
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(logits.device), reduction='none'
    )
    loss = criterion(logits, targets) * mask
    return loss.sum() / mask.sum() if mask.sum() > 0 else loss.sum() * 0


def compute_metrics(all_targets, all_preds, all_masks):
    """Compute per-class AUROC and average precision."""
    aurocs, aps = {}, {}
    for i, col in enumerate(LABEL_COLS):
        m = all_masks[:, i] == 1
        if m.sum() < 10:
            aurocs[col] = float('nan')
            aps[col] = float('nan')
            continue
        # Undo label smoothing for metric computation
        t = all_targets[m, i]
        t_binary = (t > 0.5).astype(np.float32)
        p = all_preds[m, i]

        if len(np.unique(t_binary)) < 2:
            aurocs[col] = float('nan')
            aps[col] = float('nan')
            continue
        try:
            aurocs[col] = roc_auc_score(t_binary, p)
            aps[col] = average_precision_score(t_binary, p)
        except Exception:
            aurocs[col] = float('nan')
            aps[col] = float('nan')
    return aurocs, aps


def train_one_epoch(model, loader, optimizer, scheduler, pos_weight, scaler):
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc="  Train", unit="batch", leave=False)
    for images, labels, masks in pbar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            loss = masked_bce_loss(model(images), labels, masks, pos_weight)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         lr=f"{scheduler.get_last_lr()[-1]:.2e}")
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, pos_weight):
    model.eval()
    total_loss, n = 0.0, 0
    all_t, all_p, all_m = [], [], []
    pbar = tqdm(loader, desc="  Val  ", unit="batch", leave=False)
    for images, labels, masks in pbar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = masked_bce_loss(logits, labels, masks, pos_weight)

        all_t.append(labels.cpu().numpy())
        all_p.append(torch.sigmoid(logits).cpu().numpy())
        all_m.append(masks.cpu().numpy())
        total_loss += loss.item()
        n += 1

    return (total_loss / max(n, 1),
            *compute_metrics(np.concatenate(all_t), np.concatenate(all_p),
                             np.concatenate(all_m)))


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PP1 FINAL: ConvNeXt-Base Multi-Label Classifier")
    print("Discriminative LR + OneCycleLR + Label Smoothing")
    print("=" * 70)
    print(f"Device:          {DEVICE}")
    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Epochs:          {NUM_EPOCHS}")
    print(f"LR (head):       {HEAD_LR}")
    print(f"LR (late bb):    {LATE_BACKBONE_LR}")
    print(f"LR (early bb):   {EARLY_BACKBONE_LR}")
    print(f"Label smoothing: {LABEL_SMOOTHING}")
    print(f"Grad clip:       {MAX_GRAD_NORM}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Transforms ----
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ---- Datasets ----
    print("[1/5] Loading datasets...")
    train_ds = CXRMultiLabelDataset(
        ML_DATASET_DIR / "ml_train.csv", IMAGE_ROOT, "Train", train_transform)
    val_ds = CXRMultiLabelDataset(
        ML_DATASET_DIR / "ml_val.csv", IMAGE_ROOT, "Val", val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=True)
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True)

    pos_weight = train_ds.pos_weights

    print(f"\n[2/5] Class weights (pos_weight):")
    for i, col in enumerate(LABEL_COLS):
        print(f"  {col:30s} {pos_weight[i]:.2f}x")

    # ---- Model ----
    print(f"\n[3/5] Building ConvNeXt-Base...")
    model = CXRConvNeXt(num_classes=len(LABEL_COLS))
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ---- Param groups with discriminative LR ----
    param_groups = model.get_param_groups()
    for pg in param_groups:
        n_params = sum(p.numel() for p in pg["params"])
        print(f"  {pg['name']:20s}: {n_params:>12,} params, LR={pg['lr']:.2e}")

    # ---- Optimizer ----
    optimizer = AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    # OneCycleLR with per-group max_lr
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * NUM_EPOCHS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[EARLY_BACKBONE_LR, LATE_BACKBONE_LR, HEAD_LR],
        total_steps=total_steps,
        pct_start=0.1,           # 10% warmup
        anneal_strategy='cos',
        div_factor=25,           # Start LR = max_lr / 25
        final_div_factor=1000,   # End LR = max_lr / (25 * 1000)
    )

    scaler = torch.amp.GradScaler("cuda")

    print(f"\n[4/5] Training for {NUM_EPOCHS} epochs "
          f"({total_steps} steps, warmup={int(total_steps*0.1)} steps)...")
    print("-" * 70)

    best_auroc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, pos_weight, scaler)
        val_loss, aurocs, aps = evaluate(model, val_loader, pos_weight)

        valid_a = [v for v in aurocs.values() if not np.isnan(v)]
        mean_auroc = np.mean(valid_a) if valid_a else 0.0
        pneu_auroc = aurocs.get("Pneumonia", float('nan'))

        valid_ap = [v for v in aps.values() if not np.isnan(v)]
        mean_ap = np.mean(valid_ap) if valid_ap else 0.0

        # Get current LR for each group
        lr_head = optimizer.param_groups[2]['lr']
        lr_bb = optimizer.param_groups[0]['lr']

        elapsed = time.time() - t0

        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"TrL: {train_loss:.4f} | VaL: {val_loss:.4f} | "
              f"AUROC: {mean_auroc:.4f} | Pneu: {pneu_auroc:.4f} | "
              f"mAP: {mean_ap:.4f} | "
              f"LR(h/bb): {lr_head:.1e}/{lr_bb:.1e} | {elapsed:.0f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "mean_auroc": mean_auroc, "pneumonia_auroc": pneu_auroc,
            "mean_ap": mean_ap,
        })

        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "mean_auroc": mean_auroc,
                "aurocs": aurocs,
                "aps": aps,
                "label_cols": LABEL_COLS,
                "pos_weights": pos_weight,
            }, OUTPUT_DIR / "best_model.pth")
            print(f"  --> Best model saved! (AUROC: {mean_auroc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  [!] Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    print("-" * 70)

    # ---- Test evaluation ----
    print(f"\n[5/5] Evaluating best model on TEST set...")

    test_ds = CXRMultiLabelDataset(
        ML_DATASET_DIR / "ml_test.csv", IMAGE_ROOT, "Test", val_transform)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    ckpt = torch.load(OUTPUT_DIR / "best_model.pth", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded best model from epoch {ckpt['epoch']} "
          f"(val AUROC: {ckpt['mean_auroc']:.4f})")

    test_loss, test_aurocs, test_aps = evaluate(model, test_loader, pos_weight)

    print(f"\n{'=' * 70}")
    print("TEST SET RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Condition':30s} {'AUROC':>8s}  {'AP':>8s}")
    print(f"{'-' * 50}")

    valid_aurocs, valid_aps = [], []
    for col in LABEL_COLS:
        auc = test_aurocs.get(col, float('nan'))
        ap = test_aps.get(col, float('nan'))
        if not np.isnan(auc): valid_aurocs.append(auc)
        if not np.isnan(ap): valid_aps.append(ap)
        a_s = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        p_s = f"{ap:.4f}" if not np.isnan(ap) else "N/A"
        tag = " <-- PRIMARY" if col == "Pneumonia" else ""
        print(f"  {col:30s} {a_s:>8s}  {p_s:>8s}{tag}")

    mean_auc = np.mean(valid_aurocs) if valid_aurocs else 0.0
    mean_ap = np.mean(valid_aps) if valid_aps else 0.0
    print(f"{'-' * 50}")
    print(f"  {'MEAN AUROC':30s} {mean_auc:.4f}")
    print(f"  {'MEAN AP':30s} {mean_ap:.4f}")
    print(f"  {'Test Loss':30s} {test_loss:.4f}")
    print(f"{'=' * 70}")

    # Save history
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history_final.csv", index=False)
    print(f"\nHistory: {OUTPUT_DIR / 'training_history_final.csv'}")
    print(f"Model:   {OUTPUT_DIR / 'best_model.pth'}")
    print(f"\n[DONE] Training complete!")


if __name__ == "__main__":
    main()

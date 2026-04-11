"""
PP1 Task A: Multi-Label CXR Classification with ConvNeXt-Base
================================================================
Trains ConvNeXt-Base (ImageNet pretrained) on 14 CXR pathology labels
using masked BCE loss to handle uncertain (-1) labels.

Hardware target: RTX 4060 8GB VRAM, 32GB RAM
Expected training time: ~2-4 hours (30 epochs)
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ML_DATASET_DIR = DATA_DIR / "output" / "multi_label_dataset"
IMAGE_ROOT = DATA_DIR / "images" / "images"
OUTPUT_DIR = PROJECT_ROOT / "models" / "multilabel_convnext"

# Training hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4
PATIENCE = 7  # Early stopping patience

# Label columns (14 CXR conditions)
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
    """
    Multi-label CXR dataset.
    Images are stored as flat JPGs: {IMAGE_ROOT}/{Split}/{pos_or_neg}/{dicom_id}.jpg
    Labels include -1 for uncertain (masked during training).
    """
    
    def __init__(self, csv_path, image_root, split_name, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.split_name = split_name  # "Train", "Val", "Test"
        self.transform = transform
        
        # Build image path lookup: scan both positive/ and negative/ subdirs
        self.image_paths = {}
        for cls_dir in ["positive", "negative"]:
            dir_path = self.image_root / split_name / cls_dir
            if dir_path.exists():
                for fname in os.listdir(dir_path):
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
        
        # Extract labels as numpy array
        self.labels = self.df[LABEL_COLS].values.astype(np.float32)
        
        # Compute class weights (pos_weight for BCEWithLogitsLoss)
        self.pos_weights = self._compute_pos_weights()
        
        print(f"  {split_name}: {len(self)} samples loaded, "
              f"{len(self.image_paths)} images found")
    
    def _compute_pos_weights(self):
        """Compute positive class weights for each label column."""
        weights = []
        for i, col in enumerate(LABEL_COLS):
            known_mask = self.labels[:, i] != -1
            known_labels = self.labels[known_mask, i]
            n_pos = (known_labels == 1).sum()
            n_neg = (known_labels == 0).sum()
            if n_pos > 0:
                w = n_neg / n_pos
            else:
                w = 1.0
            weights.append(min(w, 10.0))  # Cap at 10x to avoid instability
        return torch.tensor(weights, dtype=torch.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dicom_id = row["dicom_id"]
        
        # Load image
        img_path = self.image_paths[dicom_id]
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # Labels and mask
        labels = self.labels[idx].copy()
        mask = (labels != -1).astype(np.float32)  # 1 where known, 0 where uncertain
        labels = np.clip(labels, 0, 1)  # Convert -1 to 0 for tensor (masked anyway)
        
        return image, torch.tensor(labels), torch.tensor(mask)


# ============================================================================
# MODEL
# ============================================================================

class CXRConvNeXt(nn.Module):
    """ConvNeXt-Base pretrained on ImageNet for 14-label multi-label classification."""
    
    def __init__(self, num_classes=14, pretrained=True):
        super().__init__()
        
        if pretrained:
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
            self.backbone = models.convnext_base(weights=weights)
        else:
            self.backbone = models.convnext_base(weights=None)
        
        # Replace classifier head
        in_features = self.backbone.classifier[2].in_features  # 1024
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def masked_bce_loss(logits, targets, mask, pos_weight=None):
    """
    BCE loss that only computes on known labels (mask=1).
    Uncertain labels (mask=0) are excluded from loss computation.
    """
    if pos_weight is not None:
        pos_weight = pos_weight.to(logits.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    loss = criterion(logits, targets)
    loss = loss * mask  # Zero out uncertain labels
    
    if mask.sum() > 0:
        return loss.sum() / mask.sum()
    else:
        return loss.sum() * 0  # No known labels in batch (unlikely)


def compute_auroc(all_targets, all_preds, all_masks):
    """Compute per-class AUROC, skipping classes with < 2 unique values."""
    aurocs = {}
    for i, col in enumerate(LABEL_COLS):
        mask = all_masks[:, i] == 1  # Only known labels
        if mask.sum() < 10:
            aurocs[col] = float('nan')
            continue
        
        targets = all_targets[mask, i]
        preds = all_preds[mask, i]
        
        unique_vals = np.unique(targets)
        if len(unique_vals) < 2:
            aurocs[col] = float('nan')
            continue
        
        try:
            aurocs[col] = roc_auc_score(targets, preds)
        except Exception:
            aurocs[col] = float('nan')
    
    return aurocs


def train_one_epoch(model, dataloader, optimizer, pos_weight, scaler):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc="  Train", unit="batch", leave=False)
    for images, labels, masks in pbar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        masks = masks.to(DEVICE)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = masked_bce_loss(logits, labels, masks, pos_weight)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, pos_weight):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    all_targets = []
    all_preds = []
    all_masks = []
    
    pbar = tqdm(dataloader, desc="  Val  ", unit="batch", leave=False)
    for images, labels, masks in pbar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        masks = masks.to(DEVICE)
        
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = masked_bce_loss(logits, labels, masks, pos_weight)
        
        probs = torch.sigmoid(logits).cpu().numpy()
        
        all_targets.append(labels.cpu().numpy())
        all_preds.append(probs)
        all_masks.append(masks.cpu().numpy())
        
        total_loss += loss.item()
        n_batches += 1
    
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    avg_loss = total_loss / max(n_batches, 1)
    aurocs = compute_auroc(all_targets, all_preds, all_masks)
    
    return avg_loss, aurocs, all_targets, all_preds, all_masks


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    print("=" * 70)
    print("PP1 TASK A: ConvNeXt-Base Multi-Label CXR Classifier")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Labels: {len(LABEL_COLS)} conditions")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ---- Data transforms ----
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # ---- Datasets ----
    print("[1/5] Loading datasets...")
    train_dataset = CXRMultiLabelDataset(
        ML_DATASET_DIR / "ml_train.csv", IMAGE_ROOT, "Train", train_transform
    )
    val_dataset = CXRMultiLabelDataset(
        ML_DATASET_DIR / "ml_val.csv", IMAGE_ROOT, "Val", val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Print class weights
    print(f"\n[2/5] Class weights (pos_weight):")
    pos_weight = train_dataset.pos_weights
    for i, col in enumerate(LABEL_COLS):
        print(f"  {col:30s} {pos_weight[i]:.2f}x")
    
    # ---- Model ----
    print(f"\n[3/5] Building ConvNeXt-Base model...")
    model = CXRConvNeXt(num_classes=len(LABEL_COLS), pretrained=True)
    model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # ---- Optimizer & Scheduler ----
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    
    # ---- Training loop ----
    print(f"\n[4/5] Training for {NUM_EPOCHS} epochs...")
    print("-" * 70)
    
    best_val_loss = float('inf')
    best_mean_auroc = 0.0
    patience_counter = 0
    history = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, pos_weight, scaler)
        
        # Validate
        val_loss, aurocs, _, _, _ = evaluate(model, val_loader, pos_weight)
        
        # Step scheduler
        scheduler.step()
        
        # Compute mean AUROC (exclude NaN)
        valid_aurocs = [v for v in aurocs.values() if not np.isnan(v)]
        mean_auroc = np.mean(valid_aurocs) if valid_aurocs else 0.0
        pneumonia_auroc = aurocs.get("Pneumonia", float('nan'))
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Mean AUROC: {mean_auroc:.4f} | "
              f"Pneumonia AUROC: {pneumonia_auroc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"{epoch_time:.0f}s")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_auroc": mean_auroc,
            "pneumonia_auroc": pneumonia_auroc,
        })
        
        # Save best model (by mean AUROC)
        if mean_auroc > best_mean_auroc:
            best_mean_auroc = mean_auroc
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "mean_auroc": mean_auroc,
                "aurocs": aurocs,
                "label_cols": LABEL_COLS,
                "pos_weights": pos_weight,
            }
            torch.save(checkpoint, OUTPUT_DIR / "best_model.pth")
            print(f"  --> New best model saved! (Mean AUROC: {mean_auroc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  [!] Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break
    
    print("-" * 70)
    
    # ---- Final evaluation on test set ----
    print(f"\n[5/5] Evaluating best model on TEST set...")
    
    test_dataset = CXRMultiLabelDataset(
        ML_DATASET_DIR / "ml_test.csv", IMAGE_ROOT, "Test", val_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Load best model
    checkpoint = torch.load(OUTPUT_DIR / "best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded best model from epoch {checkpoint['epoch']}")
    
    test_loss, test_aurocs, _, _, _ = evaluate(model, test_loader, pos_weight)
    
    print(f"\n{'=' * 70}")
    print(f"TEST SET RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Condition':30s} {'AUROC':>8s}")
    print(f"{'-' * 40}")
    
    valid_test_aurocs = []
    for col in LABEL_COLS:
        auc = test_aurocs.get(col, float('nan'))
        if not np.isnan(auc):
            valid_test_aurocs.append(auc)
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        marker = " <-- PRIMARY" if col == "Pneumonia" else ""
        print(f"  {col:30s} {auc_str:>8s}{marker}")
    
    mean_test_auroc = np.mean(valid_test_aurocs) if valid_test_aurocs else 0.0
    print(f"{'-' * 40}")
    print(f"  {'MEAN AUROC':30s} {mean_test_auroc:.4f}")
    print(f"  {'Test Loss':30s} {test_loss:.4f}")
    print(f"{'=' * 70}")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    print(f"\nTraining history saved to: {OUTPUT_DIR / 'training_history.csv'}")
    print(f"Best model saved to: {OUTPUT_DIR / 'best_model.pth'}")
    print(f"\n[DONE] Training complete!")


if __name__ == "__main__":
    main()

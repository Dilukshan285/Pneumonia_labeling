"""
Layer 3 — Self-Trained GatorTron-Base Classifier (BEST Clinical BERT)

Fine-tunes UFNLP/gatortron-base on the ultra-high-confidence seed labels
from Layer 1, then uses it as an independent binary classifier on ALL reports.

WHY GatorTron-Base is the BEST choice:
  - 345M parameters (MegatronBERT architecture)
  - Pre-trained on 90+ BILLION words of clinical text:
      * 82B words from UF Health clinical notes (real EHR data)
      * 4.5B words from PubMed abstracts
      * 0.5B words from Wikipedia
      * Includes MIMIC-III notes (same source as our reports!)
  - Understands clinical shorthand, abbreviations, report structure
  - Vastly superior to BioBERT/BiomedBERT for clinical text

GatorTron has NEVER seen the Snorkel labels, NLI scores, or keyword
patterns — it learns purely from text → label mapping. This makes it
a genuinely independent classification system.

RESUME SUPPORT:
  - If fine-tuned model exists on disk → skips training, goes to prediction
  - Prediction checkpoints every 5000 batches → resumes from last checkpoint
  - Use --retrain flag to force retraining even if model exists

Phase 1: Fine-tune on Layer 1 seeds (80/10/10 train/val/test split)
Phase 2: Predict on ALL 227,835 reports

Input:  layer1_seeds.csv, parsed_reports.csv
Output: layer3_pubmedbert.csv (study_id, l3_label, l3_confidence)

Runtime: ~90 min fine-tuning + ~30 min inference on RTX 4060
"""

import os
import sys
import gc
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_INTERMEDIATE,
    LABEL_POSITIVE, LABEL_NEGATIVE,
    RANDOM_SEED,
)

# Files
SEEDS_CSV = os.path.join(DATA_INTERMEDIATE, "layer1_seeds.csv")
PARSED_REPORTS = os.path.join(DATA_INTERMEDIATE, "parsed_reports.csv")
OUTPUT_CSV = os.path.join(DATA_INTERMEDIATE, "layer3_pubmedbert.csv")
MODEL_SAVE_DIR = os.path.join(DATA_INTERMEDIATE, "gatortron_finetuned")

# Prediction checkpoint
PRED_CHECKPOINT_FILE = os.path.join(DATA_INTERMEDIATE, "layer3_pred_checkpoint.json")
PRED_CHECKPOINT_INTERVAL = 5000  # Save every N batches

# ============================================================================
# MODEL — GatorTron-Base: THE BEST clinical BERT model
# 345M params, trained on 90B+ words of clinical text including MIMIC
# ============================================================================
MODEL_NAME = "UFNLP/gatortron-base"
MAX_LENGTH = 512          # Full context; batch 4 + gradient checkpointing fits in 8GB
BATCH_SIZE = 4            # Minimal for 8GB VRAM; gradient accumulation recovers effective batch
GRADIENT_ACCUMULATION_STEPS = 4  # effective batch = 4 × 4 = 16
LEARNING_RATE = 2e-5    # Standard for BERT fine-tuning
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
PRED_BATCH_SIZE = 32    # Inference only — no gradients, lower VRAM
CONFIDENCE_THRESHOLD = 0.70  # Below this → EXCLUDED

# L3 label encoding
L3_EXCLUDED = 99


def _set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_pred_checkpoint(batch_idx, labels, confidences):
    """Save prediction progress for resume."""
    temp = PRED_CHECKPOINT_FILE + ".tmp"
    with open(temp, 'w') as f:
        json.dump({
            "batch_idx": batch_idx,
            "labels": labels,
            "confidences": confidences,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f)
    os.replace(temp, PRED_CHECKPOINT_FILE)


def _load_pred_checkpoint():
    """Load prediction checkpoint if exists."""
    if not os.path.exists(PRED_CHECKPOINT_FILE):
        return None
    try:
        with open(PRED_CHECKPOINT_FILE, 'r') as f:
            ck = json.load(f)
        if 'batch_idx' in ck and 'labels' in ck and 'confidences' in ck:
            return ck
    except Exception:
        pass
    return None


class ReportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


class PredictionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }


def prepare_text(row):
    """Combine impression and findings into model input."""
    imp = str(row.get('impression_text', '') or '').strip()
    find = str(row.get('findings_text', '') or '').strip()
    if imp and find:
        return f"{imp} [SEP] {find}"
    return imp if imp else find


def train_model(train_texts, train_labels, val_texts, val_labels, device):
    """Fine-tune GatorTron-Base on seed labels."""
    print(f"    Loading tokenizer and model: {MODEL_NAME}")
    print(f"    (GatorTron-Base: 345M params, trained on 90B+ clinical words)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Load in FP32 for training — GradScaler requires FP32 parameters.
    # autocast() handles FP16 casting only during the forward pass (correct usage).
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
    ).to(device)

    # Gradient checkpointing: recompute activations during backward pass
    # Slashes activation memory from O(n_layers) to O(sqrt(n_layers))
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Model parameters: {n_params/1e6:.0f}M")

    # Compute class weights for imbalanced data
    n_pos = sum(1 for l in train_labels if l == 1)
    n_neg = sum(1 for l in train_labels if l == 0)
    pos_weight = n_neg / max(n_pos, 1)
    print(f"    Class weights — pos_weight: {pos_weight:.2f} (NEG: {n_neg}, POS: {n_pos})")

    train_ds = ReportDataset(train_texts, train_labels, tokenizer)
    val_ds = ReportDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    # Account for gradient accumulation in scheduler total steps
    steps_per_epoch = (len(train_loader) + GRADIENT_ACCUMULATION_STEPS - 1) // GRADIENT_ACCUMULATION_STEPS
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # FP32 weight tensor to match the FP32 model parameters
    weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    best_val_acc = 0.0
    best_epoch = 0
    patience = 2
    no_improve = 0

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"    Epoch {epoch+1}/{NUM_EPOCHS}",
                    file=sys.stdout, leave=False)

        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss_accum = loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss_accum).backward()

            # Step optimizer every GRADIENT_ACCUMULATION_STEPS or at end of epoch
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                preds = outputs.logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total

        print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}  "
              f"train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improve = 0
            # Save best model
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            model.save_pretrained(MODEL_SAVE_DIR)
            tokenizer.save_pretrained(MODEL_SAVE_DIR)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    print(f"    Best val_acc: {best_val_acc:.1f}% at epoch {best_epoch}")
    print(f"    Model saved: {MODEL_SAVE_DIR}")

    return best_val_acc


def predict_all(texts, device):
    """Run fine-tuned GatorTron on all reports with checkpoint/resume support."""
    print(f"    Loading fine-tuned model from {MODEL_SAVE_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_SAVE_DIR, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    pred_ds = PredictionDataset(texts, tokenizer)
    pred_loader = DataLoader(pred_ds, batch_size=PRED_BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    all_labels = []
    all_confidences = []
    start_batch = 0
    since_ck = 0

    # Resume from checkpoint
    ck = _load_pred_checkpoint()
    if ck:
        start_batch = ck['batch_idx']
        all_labels = ck['labels']
        all_confidences = ck['confidences']
        n_done = start_batch * PRED_BATCH_SIZE
        print(f"    RESUMING prediction from batch {start_batch:,} (~{n_done:,} reports)")

    total_batches = len(pred_loader)
    pbar = tqdm(total=total_batches, initial=start_batch,
                desc="    Predicting", file=sys.stdout, unit="batch")

    with torch.no_grad():
        for batch_idx, batch in enumerate(pred_loader):
            # Skip already-processed batches
            if batch_idx < start_batch:
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits.float(), dim=-1)
            preds = probs.argmax(dim=-1)
            confs = probs.max(dim=-1).values

            all_labels.extend(preds.cpu().numpy().tolist())
            all_confidences.extend(confs.cpu().numpy().tolist())

            pbar.update(1)
            since_ck += 1

            # Checkpoint
            if since_ck >= PRED_CHECKPOINT_INTERVAL:
                _save_pred_checkpoint(batch_idx + 1, all_labels, all_confidences)
                since_ck = 0

    pbar.close()

    # Clean up checkpoint
    if os.path.exists(PRED_CHECKPOINT_FILE):
        os.remove(PRED_CHECKPOINT_FILE)

    return all_labels, all_confidences


def _model_exists():
    """Check if fine-tuned model exists on disk."""
    config_path = os.path.join(MODEL_SAVE_DIR, "config.json")
    return os.path.exists(config_path)


def main():
    t_start = time.time()
    _set_seed()

    # Check for --retrain flag
    force_retrain = '--retrain' in sys.argv

    print("=" * 70)
    print("LAYER 3 — SELF-TRAINED GatorTron-Base CLASSIFIER (BEST CLINICAL BERT)")
    print("=" * 70)
    print()
    print(f"  Model: {MODEL_NAME}")
    print(f"  Architecture: MegatronBERT (345M params)")
    print(f"  Pre-training: 90B+ words of clinical text (incl. MIMIC-III)")
    print(f"  Batch: {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} accum = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} effective")
    print(f"  Seq length: {MAX_LENGTH} tokens | Gradient checkpointing: ON")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ---- Check if training can be skipped ----
    skip_training = _model_exists() and not force_retrain
    best_val_acc = None

    if skip_training:
        print("  *** RESUME MODE: Fine-tuned model found on disk ***")
        print(f"  *** Skipping training. Use --retrain to force retraining. ***")
        print(f"  *** Model path: {MODEL_SAVE_DIR} ***")
        print()
    else:
        # ---- Load seeds ----
        print("  Phase 1: Loading Layer 1 seeds...")
        df_seeds = pd.read_csv(SEEDS_CSV, low_memory=False)
        df_seeds['study_id'] = df_seeds['study_id'].astype(str)
        n_seeds = len(df_seeds)
        print(f"    Seeds: {n_seeds:,}")
        seed_counts = Counter(df_seeds['seed_label'].tolist())
        print(f"    POS: {seed_counts.get(LABEL_POSITIVE,0):,}  "
              f"NEG: {seed_counts.get(LABEL_NEGATIVE,0):,}")
        print()

        # Load report texts for seeds
        print("  Loading report texts for training...")
        df_reports_train = pd.read_csv(PARSED_REPORTS, low_memory=False,
                                 usecols=['study_id', 'impression_text', 'findings_text'])
        df_reports_train['study_id'] = df_reports_train['study_id'].astype(str)

        df_seed_data = df_seeds.merge(
            df_reports_train[['study_id', 'impression_text', 'findings_text']],
            on='study_id', how='inner'
        )
        print(f"    Seeds with text: {len(df_seed_data):,}")
        del df_reports_train

        # Prepare texts
        seed_texts = [prepare_text(row) for _, row in df_seed_data.iterrows()]
        seed_labels = df_seed_data['seed_label'].values.tolist()

        # ---- Train/Val/Test split (80/10/10 stratified) ----
        from sklearn.model_selection import train_test_split

        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            seed_texts, seed_labels, test_size=0.2,
            random_state=RANDOM_SEED, stratify=seed_labels
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5,
            random_state=RANDOM_SEED, stratify=temp_labels
        )

        print(f"    Train: {len(train_texts):,}  Val: {len(val_texts):,}  Test: {len(test_texts):,}")
        print()

        # ---- Phase 1: Fine-tune ----
        print(f"  Phase 1: Fine-tuning GatorTron-Base (345M params)...")
        _clear_gpu()

        best_val_acc = train_model(
            train_texts, train_labels, val_texts, val_labels, device
        )
        print()

        # Clean up
        _clear_gpu()

    # ---- Phase 2: Predict on ALL reports ----
    print("  Phase 2: Predicting on ALL reports...")

    df_reports = pd.read_csv(PARSED_REPORTS, low_memory=False,
                             usecols=['study_id', 'subject_id',
                                      'impression_text', 'findings_text'])
    df_reports['study_id'] = df_reports['study_id'].astype(str)

    all_texts = [prepare_text(row) for _, row in df_reports.iterrows()]
    n_total = len(all_texts)

    # Replace empty texts with a neutral placeholder
    all_texts = [t if t.strip() else "normal chest" for t in all_texts]

    pred_labels, pred_confidences = predict_all(all_texts, device)

    # Clean up
    _clear_gpu()

    # Apply confidence threshold
    l3_labels = []
    for label, conf in zip(pred_labels, pred_confidences):
        if conf >= CONFIDENCE_THRESHOLD:
            l3_labels.append(label)
        else:
            l3_labels.append(L3_EXCLUDED)

    # ---- Save ----
    df_output = pd.DataFrame({
        'study_id': df_reports['study_id'].values,
        'subject_id': df_reports['subject_id'].values,
        'l3_label': l3_labels,
        'l3_raw_label': pred_labels,
        'l3_confidence': pred_confidences,
    })

    df_output.to_csv(OUTPUT_CSV, index=False)
    file_size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)

    # Stats
    l3_counts = Counter(l3_labels)
    n_pos = l3_counts.get(LABEL_POSITIVE, 0)
    n_neg = l3_counts.get(LABEL_NEGATIVE, 0)
    n_exc = l3_counts.get(L3_EXCLUDED, 0)

    t_total = time.time() - t_start

    print()
    print("=" * 70)
    print("LAYER 3 COMPLETE — GatorTron-Base (BEST CLINICAL BERT)")
    print("=" * 70)
    print()
    print(f"  Model: {MODEL_NAME}")
    if best_val_acc is not None:
        print(f"  Best val accuracy: {best_val_acc:.1f}%")
    else:
        print(f"  Training: SKIPPED (resumed from saved model)")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Classification results (all {n_total:,} reports):")
    print(f"    POSITIVE (conf >= {CONFIDENCE_THRESHOLD}): {n_pos:>8,} ({100*n_pos/n_total:.1f}%)")
    print(f"    NEGATIVE (conf >= {CONFIDENCE_THRESHOLD}): {n_neg:>8,} ({100*n_neg/n_total:.1f}%)")
    print(f"    EXCLUDED (low confidence):    {n_exc:>8,} ({100*n_exc/n_total:.1f}%)")
    print()
    print(f"  File: {OUTPUT_CSV}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Runtime: {t_total/60:.1f} min")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
================================================================================
STEP 1 — CORRECT MISLABELED REPORTS
================================================================================
Uses audit_results_full.csv to fix labels in final_pneumonia_labels.csv.
Creates a NEW corrected file — original CSV is NEVER modified.

Corrections applied:
  - FALSE_POSITIVE  (pipeline=1, audit=NEG)  → label=0, soft_score=0.05
  - FALSE_NEGATIVE  (pipeline=0, audit=POS)  → label=1, soft_score=0.95
  - POS_BUT_UNCERTAIN                        → REMOVED from training
  - NEG_BUT_UNCERTAIN                        → REMOVED from training
  - All others                               → keep original label

Output: final_pneumonia_labels_corrected.csv
================================================================================
"""

import pandas as pd
import os
import time

start = time.time()

# === Paths ===
LABELS_CSV = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\final_pneumonia_labels.csv"
AUDIT_CSV  = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\audit_results_full.csv"
OUTPUT_DIR = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output"

print("=" * 70)
print("STEP 1 — CORRECTING MISLABELED REPORTS")
print("=" * 70)

# === Load original labels ===
print("\nLoading original labels (will NOT be modified) ...")
df_labels = pd.read_csv(LABELS_CSV)
df_labels['subject_id'] = df_labels['subject_id'].astype(str)
df_labels['study_id'] = df_labels['study_id'].astype(str)
print(f"  Original total: {len(df_labels):,}")
print(f"  Original POS:   {(df_labels['label']==1).sum():,}")
print(f"  Original NEG:   {(df_labels['label']==0).sum():,}")

# === Load audit results ===
print("\nLoading audit results ...")
df_audit = pd.read_csv(AUDIT_CSV)
df_audit['subject_id'] = df_audit['subject_id'].astype(str)
df_audit['study_id'] = df_audit['study_id'].astype(str)
print(f"  Audit rows: {len(df_audit):,}")

# === Merge ===
print("\nMerging labels with audit results ...")
df = df_labels.merge(
    df_audit[['subject_id', 'study_id', 'audit_label', 'audit_reason', 'audit_detail', 'mismatch', 'mismatch_type']],
    on=['subject_id', 'study_id'],
    how='left'
)
print(f"  Merged rows: {len(df):,}")

# === Apply corrections ===
print("\nApplying corrections ...")

# Track changes
corrections = {
    'false_pos_flipped': 0,
    'false_neg_flipped': 0,
    'pos_uncertain_removed': 0,
    'neg_uncertain_removed': 0,
    'kept_original': 0,
}

# Create new columns
df['corrected_label'] = df['label'].copy()
df['corrected_soft_score'] = df['soft_score'].copy()
df['corrected_label_source'] = df['label_source'].copy()

# FALSE POSITIVES: pipeline=1, audit=NEGATIVE → flip to 0
mask_fp = df['mismatch_type'] == 'FALSE_POSITIVE'
df.loc[mask_fp, 'corrected_label'] = 0
df.loc[mask_fp, 'corrected_soft_score'] = 0.05
df.loc[mask_fp, 'corrected_label_source'] = 'audit_correction'
corrections['false_pos_flipped'] = mask_fp.sum()

# FALSE NEGATIVES: pipeline=0, audit=POSITIVE → flip to 1
mask_fn = df['mismatch_type'] == 'FALSE_NEGATIVE'
df.loc[mask_fn, 'corrected_label'] = 1
df.loc[mask_fn, 'corrected_soft_score'] = 0.95
df.loc[mask_fn, 'corrected_label_source'] = 'audit_correction'
corrections['false_neg_flipped'] = mask_fn.sum()

# UNCERTAIN: remove from training set
mask_pos_unc = df['mismatch_type'] == 'POS_BUT_UNCERTAIN'
mask_neg_unc = df['mismatch_type'] == 'NEG_BUT_UNCERTAIN'
mask_uncertain = mask_pos_unc | mask_neg_unc
corrections['pos_uncertain_removed'] = mask_pos_unc.sum()
corrections['neg_uncertain_removed'] = mask_neg_unc.sum()
corrections['kept_original'] = len(df) - mask_fp.sum() - mask_fn.sum() - mask_uncertain.sum()

print(f"\n  Corrections applied:")
print(f"    False positives flipped (1→0):  {corrections['false_pos_flipped']:>8,}")
print(f"    False negatives flipped (0→1):  {corrections['false_neg_flipped']:>8,}")
print(f"    POS→Uncertain removed:          {corrections['pos_uncertain_removed']:>8,}")
print(f"    NEG→Uncertain removed:          {corrections['neg_uncertain_removed']:>8,}")
print(f"    Kept original label:            {corrections['kept_original']:>8,}")

# === Remove uncertain reports from training set ===
df_corrected = df[~mask_uncertain].copy()

# === Build final corrected CSV ===
df_final = df_corrected[['subject_id', 'study_id', 'corrected_label', 'corrected_soft_score', 'corrected_label_source']].copy()
df_final.columns = ['subject_id', 'study_id', 'label', 'soft_score', 'label_source']

# === Save ===
corrected_path = os.path.join(OUTPUT_DIR, "final_pneumonia_labels_corrected.csv")
df_final.to_csv(corrected_path, index=False)

# === Statistics ===
total_corrected = len(df_final)
pos_corrected = (df_final['label'] == 1).sum()
neg_corrected = (df_final['label'] == 0).sum()
ratio = max(pos_corrected, neg_corrected) / min(pos_corrected, neg_corrected) if min(pos_corrected, neg_corrected) > 0 else float('inf')
majority = "NEGATIVE" if neg_corrected > pos_corrected else "POSITIVE"

print(f"\n{'='*70}")
print(f"CORRECTED LABEL DISTRIBUTION")
print(f"{'='*70}")
print(f"  Total training labels:  {total_corrected:>10,}  (was {len(df_labels):,})")
print(f"  Removed (uncertain):    {len(df_labels) - total_corrected:>10,}")
print(f"  POSITIVE (1):           {pos_corrected:>10,}  (was {(df_labels['label']==1).sum():,})")
print(f"  NEGATIVE (0):           {neg_corrected:>10,}  (was {(df_labels['label']==0).sum():,})")
print(f"  Class ratio:            {ratio:.2f}:1  ({majority} heavy)")
print(f"\n  Saved to: {corrected_path}")

# === Imbalance note ===
if ratio > 3.0:
    imbalance_note = os.path.join(OUTPUT_DIR, "class_imbalance_note.txt")
    with open(imbalance_note, 'w') as f:
        f.write(f"CLASS IMBALANCE WARNING\n")
        f.write(f"{'='*50}\n")
        f.write(f"POSITIVE: {pos_corrected:,}\n")
        f.write(f"NEGATIVE: {neg_corrected:,}\n")
        f.write(f"Ratio: {ratio:.2f}:1 ({majority} heavy)\n\n")
        f.write(f"Recommendation: Use class-weighted loss or\n")
        f.write(f"minority oversampling during model training.\n")
    print(f"  Imbalance warning saved: {imbalance_note}")

# === Also save the full audit-merged table for reference ===
reference_path = os.path.join(OUTPUT_DIR, "labels_correction_reference.csv")
df[['subject_id', 'study_id', 'label', 'soft_score', 'label_source',
    'audit_label', 'audit_reason', 'mismatch_type',
    'corrected_label', 'corrected_soft_score', 'corrected_label_source']].to_csv(reference_path, index=False)
print(f"  Full reference table:   {reference_path}")

elapsed = time.time() - start
print(f"\nCompleted in {elapsed:.1f}s")
print("=" * 70)

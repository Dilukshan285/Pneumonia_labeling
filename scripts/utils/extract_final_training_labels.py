"""
===============================================================================
FINAL TRAINING LABEL EXTRACTION — Snorkel Soft Scores + CheXpert Disease Filter
===============================================================================

This script takes the COMPLETED Snorkel pipeline output (snorkel_soft_scores.csv)
and produces the final binary training label set for PP1 and PP2.

Pipeline:
  1. Load Snorkel soft scores (computed from 6-LF agreement via LabelModel)
  2. Apply confidence thresholds:
       soft_score >= 0.75  →  POSITIVE (high confidence)
       soft_score <= 0.25  →  NEGATIVE (high confidence)
       0.25 < soft_score < 0.75  →  EXCLUDED (uncertain)
  3. Apply CheXpert 8-disease filter to NEGATIVE candidates:
       Exclude studies with diseases that mimic pneumonia on CXR
       KEEP: Cardiomegaly, Fracture, Enlarged Cardiomediastinum, Support Devices
       (these do NOT look like pneumonia and are valid negative examples)
  4. Output: final_training_labels.csv

Why NOT 12 diseases:
  - Cardiomegaly       → cardiac, not pulmonary; doesn't mimic pneumonia
  - Fracture           → skeletal, not pulmonary opacity
  - Enlarged Cardiomed → mediastinal, not lung parenchyma
  - Support Devices    → lines/tubes, not opacity confusion

Why NOT the refined_binary_labeling.py approach:
  - That script uses a single keyword classifier (essentially LF1 alone)
  - Snorkel aggregates 6 independent voters and learns their reliability
  - Snorkel's soft_score is a real probability, not a hardcoded number

Output: final_training_labels.csv with columns:
  subject_id, study_id, label, soft_score, label_source
===============================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import logging

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_DIR = r"C:\Users\dviya\Desktop\Pneumonia_labeling"
SNORKEL_SCORES_CSV = os.path.join(PROJECT_DIR, "data", "intermediate", "snorkel_soft_scores.csv")
CHEXPERT_CSV = os.path.join(PROJECT_DIR, "data", "raw", "mimic-cxr-2.0.0-chexpert.csv")
OUTPUT_CSV = os.path.join(PROJECT_DIR, "data", "output", "final_training_labels.csv")
LOG_FILE = os.path.join(PROJECT_DIR, "logs", "extract_final_training_labels.log")

# Confidence thresholds (from Stage 3 implementation plan)
THRESHOLD_POSITIVE = 0.75
THRESHOLD_NEGATIVE = 0.25

# CheXpert diseases to EXCLUDE from NEGATIVE set
# Only diseases that produce opacities/patterns mimicking pneumonia on CXR
# Cardiomegaly, Fracture, Enlarged Cardiomediastinum, Support Devices are
# EXCLUDED from this list because they do NOT look like pneumonia
DISEASES_MIMICKING_PNEUMONIA = [
    'Atelectasis',         # Lung collapse → opacity confusion
    'Consolidation',       # Direct opacity overlap with pneumonia
    'Edema',               # Pulmonary edema → bilateral opacities
    'Lung Lesion',         # Nodules/masses → focal opacity
    'Lung Opacity',        # Generic opacity → direct overlap
    'Pleural Effusion',    # Fluid → basilar opacity mimics
    'Pleural Other',       # Pleural abnormality → opacity confusion
    'Pneumothorax',        # Air leak → altered lung appearance
]

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


def main():
    start_time = time.time()
    log.info("=" * 70)
    log.info("FINAL TRAINING LABEL EXTRACTION")
    log.info("Snorkel Soft Scores + CheXpert 8-Disease Filter")
    log.info("=" * 70)
    log.info(f"  Positive threshold: soft_score >= {THRESHOLD_POSITIVE}")
    log.info(f"  Negative threshold: soft_score <= {THRESHOLD_NEGATIVE}")
    log.info(f"  Disease exclusions: {len(DISEASES_MIMICKING_PNEUMONIA)} diseases")
    log.info("")

    # ===================================================================
    # STEP 1: Load Snorkel soft scores
    # ===================================================================
    log.info("STEP 1: Loading Snorkel soft scores...")

    if not os.path.exists(SNORKEL_SCORES_CSV):
        log.error(f"Snorkel scores not found: {SNORKEL_SCORES_CSV}")
        log.error("Run the full Snorkel pipeline (Stages 1-3) first.")
        return 1

    df = pd.read_csv(SNORKEL_SCORES_CSV)
    df['study_id'] = df['study_id'].astype(str).str.strip()
    df['subject_id'] = df['subject_id'].astype(str).str.strip()

    log.info(f"  Total reports: {len(df):,}")
    log.info(f"  study_id format: {df['study_id'].iloc[:3].tolist()}")
    log.info(f"  soft_score range: [{df['soft_score'].min():.6f}, {df['soft_score'].max():.6f}]")
    log.info(f"  soft_score mean:  {df['soft_score'].mean():.4f}")
    log.info(f"  Label sources: {df['label_source'].value_counts().to_dict()}")
    log.info("")

    # ===================================================================
    # STEP 2: Apply confidence thresholds (vectorized)
    # ===================================================================
    log.info("STEP 2: Applying confidence thresholds...")

    # Vectorized threshold application — no iterrows needed
    df['threshold_label'] = np.where(
        df['soft_score'] >= THRESHOLD_POSITIVE, 'POSITIVE',
        np.where(
            df['soft_score'] <= THRESHOLD_NEGATIVE, 'NEGATIVE',
            'UNCERTAIN'
        )
    )

    threshold_counts = df['threshold_label'].value_counts()
    for label_name in ['POSITIVE', 'NEGATIVE', 'UNCERTAIN']:
        count = threshold_counts.get(label_name, 0)
        pct = 100 * count / len(df)
        log.info(f"  {label_name:>12s}: {count:>8,}  ({pct:.1f}%)")
    log.info("")

    # Split into pools
    positives = df[df['threshold_label'] == 'POSITIVE'].copy()
    neg_candidates = df[df['threshold_label'] == 'NEGATIVE'].copy()
    uncertain = df[df['threshold_label'] == 'UNCERTAIN'].copy()

    log.info(f"  POSITIVE pool:   {len(positives):,}")
    log.info(f"  NEGATIVE candidates: {len(neg_candidates):,}")
    log.info(f"  UNCERTAIN (excluded): {len(uncertain):,}")
    log.info("")

    # ===================================================================
    # STEP 3: Load CheXpert and build 8-disease filter
    # ===================================================================
    log.info("STEP 3: Loading CheXpert for 8-disease clean-negative filter...")

    chx = pd.read_csv(CHEXPERT_CSV)
    # FIX: Add 's' prefix to match Snorkel study_id format
    chx['study_id'] = 's' + chx['study_id'].astype(str).str.strip()

    log.info(f"  CheXpert entries: {len(chx):,}")
    log.info(f"  study_id format: {chx['study_id'].iloc[:3].tolist()}")

    # Verify join
    overlap = set(df['study_id'].head(500)) & set(chx['study_id'].head(500))
    log.info(f"  Join check (500-sample overlap): {len(overlap)}")
    if len(overlap) == 0:
        log.error("  ZERO overlap! study_id format mismatch. Aborting.")
        return 1
    log.info("")

    # Build clean-negative mask: exclude only 8 pneumonia-mimicking diseases
    # A study is "clean" if none of the 8 diseases has value 1.0 or -1.0
    log.info(f"  Applying 8-disease filter (excluding mimicking diseases):")
    clean_mask = pd.Series(True, index=chx.index)
    for disease in DISEASES_MIMICKING_PNEUMONIA:
        disease_has_finding = chx[disease].isin([1.0, -1.0])
        n_affected = disease_has_finding.sum()
        clean_mask = clean_mask & ~disease_has_finding
        log.info(f"    {disease:<25s}: {n_affected:>7,} studies with finding")

    clean_study_ids = set(chx.loc[clean_mask, 'study_id'].values)
    excluded_study_ids = set(chx.loc[~clean_mask, 'study_id'].values)

    log.info("")
    log.info(f"  Clean studies (no mimicking diseases): {len(clean_study_ids):,}")
    log.info(f"  Excluded studies:                      {len(excluded_study_ids):,}")
    log.info("")

    # Count how many would be excluded by the OLD 12-disease filter for comparison
    old_12_diseases = DISEASES_MIMICKING_PNEUMONIA + [
        'Cardiomegaly', 'Fracture', 'Enlarged Cardiomediastinum', 'Support Devices'
    ]
    old_clean_mask = pd.Series(True, index=chx.index)
    for disease in old_12_diseases:
        old_clean_mask = old_clean_mask & ~chx[disease].isin([1.0, -1.0])
    old_clean_count = old_clean_mask.sum()
    log.info(f"  [COMPARISON] Old 12-disease filter would keep: {old_clean_count:,}")
    log.info(f"  [COMPARISON] New 8-disease filter keeps:       {len(clean_study_ids):,}")
    log.info(f"  [COMPARISON] Additional negatives retained:    {len(clean_study_ids) - old_clean_count:,}")
    log.info("")

    # ===================================================================
    # STEP 4: Apply disease filter to negative candidates
    # ===================================================================
    log.info("STEP 4: Filtering negative candidates...")

    neg_before = len(neg_candidates)
    neg_candidates['is_clean'] = neg_candidates['study_id'].isin(clean_study_ids)

    # Studies not in CheXpert at all — treat as clean (pre-filtered reports)
    not_in_chexpert = ~neg_candidates['study_id'].isin(set(chx['study_id'].values))
    n_not_in_chx = not_in_chexpert.sum()
    log.info(f"  Negative candidates not in CheXpert: {n_not_in_chx:,}")
    log.info(f"  (These are pre-filter/all-abstain reports — treated as clean)")

    # Clean negatives: either clean in CheXpert OR not in CheXpert
    clean_negatives = neg_candidates[neg_candidates['is_clean'] | not_in_chexpert].copy()
    dirty_negatives = neg_candidates[~(neg_candidates['is_clean'] | not_in_chexpert)]

    log.info(f"  Negative candidates before filter: {neg_before:,}")
    log.info(f"  Clean negatives after filter:      {len(clean_negatives):,}")
    log.info(f"  Excluded by disease filter:         {len(dirty_negatives):,}")
    log.info("")

    # ===================================================================
    # STEP 5: Combine POSITIVE + clean NEGATIVE
    # ===================================================================
    log.info("STEP 5: Building final training label set...")

    # Assign binary labels
    positives['label'] = 1
    positives['label_source_final'] = positives['label_source']
    clean_negatives['label'] = 0
    clean_negatives['label_source_final'] = clean_negatives['label_source']

    final = pd.concat([
        positives[['subject_id', 'study_id', 'label', 'soft_score', 'label_source_final']],
        clean_negatives[['subject_id', 'study_id', 'label', 'soft_score', 'label_source_final']],
    ], ignore_index=True)

    final.columns = ['subject_id', 'study_id', 'label', 'soft_score', 'label_source']

    # Sort for deterministic output
    final = final.sort_values('study_id').reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final.to_csv(OUTPUT_CSV, index=False)
    log.info(f"  Saved: {OUTPUT_CSV}")
    log.info("")

    # ===================================================================
    # STEP 6: Summary statistics
    # ===================================================================
    pos_count = (final['label'] == 1).sum()
    neg_count = (final['label'] == 0).sum()
    total = len(final)
    ratio = neg_count / pos_count if pos_count > 0 else float('inf')

    log.info("=" * 70)
    log.info("FINAL TRAINING LABEL SUMMARY")
    log.info("=" * 70)
    log.info(f"  POSITIVE:  {pos_count:>8,}  ({100*pos_count/total:.1f}%)")
    log.info(f"  NEGATIVE:  {neg_count:>8,}  ({100*neg_count/total:.1f}%)")
    log.info(f"  TOTAL:     {total:>8,}")
    log.info(f"  NEG:POS ratio: {ratio:.2f}:1")
    log.info("")

    # Label source breakdown
    log.info(f"  Label source breakdown:")
    for src, cnt in final['label_source'].value_counts().items():
        log.info(f"    {src}: {cnt:,}")
    log.info("")

    # Soft score distribution by label
    log.info(f"  Soft score statistics:")
    for lbl, name in [(1, 'POSITIVE'), (0, 'NEGATIVE')]:
        subset = final[final['label'] == lbl]['soft_score']
        log.info(f"    {name}: mean={subset.mean():.4f}  "
                 f"std={subset.std():.4f}  "
                 f"min={subset.min():.4f}  max={subset.max():.4f}")
    log.info("")

    # Exclusion accounting
    total_reports = len(df)
    log.info(f"  Report accounting:")
    log.info(f"    Total reports in dataset:     {total_reports:>8,}")
    log.info(f"    Excluded (uncertain scores):  {len(uncertain):>8,}")
    log.info(f"    Excluded (disease filter):    {len(dirty_negatives):>8,}")
    log.info(f"    Final training labels:        {total:>8,}")
    log.info(f"    Coverage:                     {100*total/total_reports:.1f}%")
    log.info("")

    # Class imbalance check
    if ratio > 3.0:
        imbalance_note = os.path.join(os.path.dirname(OUTPUT_CSV), "class_imbalance_note.txt")
        majority = "NEGATIVE" if neg_count > pos_count else "POSITIVE"
        with open(imbalance_note, 'w') as f:
            f.write(f"CLASS IMBALANCE WARNING\n")
            f.write(f"{'='*50}\n")
            f.write(f"POSITIVE: {pos_count:,}\n")
            f.write(f"NEGATIVE: {neg_count:,}\n")
            f.write(f"Ratio: {ratio:.2f}:1 ({majority} heavy)\n\n")
            f.write(f"Recommendation: Use class-weighted loss or\n")
            f.write(f"minority oversampling during model training.\n")
        log.info(f"  ⚠ Class imbalance detected ({ratio:.2f}:1)")
        log.info(f"    Imbalance note saved: {imbalance_note}")
        log.info("")

    # ===================================================================
    # STEP 7: Sample verification (vectorized lookup)
    # ===================================================================
    log.info("=" * 70)
    log.info("SAMPLE VERIFICATION — 5 POSITIVE, 5 NEGATIVE")
    log.info("=" * 70)

    # Load parsed reports for verification text
    parsed_path = os.path.join(PROJECT_DIR, "data", "intermediate", "parsed_reports.csv")
    if os.path.exists(parsed_path):
        parsed = pd.read_csv(parsed_path, usecols=['study_id', 'impression_text', 'findings_text'])
        parsed['study_id'] = parsed['study_id'].astype(str).str.strip()
        parsed['impression_text'] = parsed['impression_text'].fillna('')

        for lbl, name in [(1, 'POSITIVE'), (0, 'NEGATIVE')]:
            subset = final[final['label'] == lbl]
            samples = subset.sample(n=min(5, len(subset)), random_state=42)
            log.info(f"\n  --- {name} samples ---")
            # Vectorized merge instead of per-row lookup
            sample_with_text = samples.merge(parsed[['study_id', 'impression_text']], on='study_id', how='left')
            for _, row in sample_with_text.iterrows():
                imp = str(row.get('impression_text', ''))[:180]
                log.info(f"  [{row['study_id']}] score={row['soft_score']:.4f} src={row['label_source']}")
                log.info(f"    IMPRESSION: {imp}")
                log.info("")
    else:
        log.info("  (parsed_reports.csv not found — skipping text verification)")

    elapsed = time.time() - start_time
    log.info(f"Completed in {elapsed:.1f} seconds")
    log.info(f"Output: {OUTPUT_CSV}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

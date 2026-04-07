"""
Stage 4 — Steps 4.4 and 4.5: Merge Manual Labels & Create Final Label Set

After completing manual labeling of the 200 active learning reports (Step 4.3),
this script:
  1. Reads the manually labeled active_learning_queue.csv
  2. Validates all 200 manual_label entries
  3. Merges manual labels into the confident pool (overriding pipeline labels)
  4. Assigns fixed soft_scores for manual labels (0.95/0.05/0.50)
  5. Combines all label sources into one final DataFrame
  6. Removes UNCERTAIN labels (excluded from training)
  7. Saves final_pneumonia_labels.csv — the complete label reference for PP1/PP2

Input:   active_learning_queue.csv  (with manual_label column filled in)
         confident_pool.csv         (from Stage 3)
         prefilter_negatives.csv    (from Step 2.0)
Output:  final_pneumonia_labels.csv (POSITIVE + NEGATIVE only — training ready)

Estimated runtime: < 30 seconds
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    ACTIVE_LEARNING_QUEUE_CSV,
    CONFIDENT_POOL_CSV,
    FINAL_LABELS_CSV,
    DATA_INTERMEDIATE,
    DATA_OUTPUT,
    LABEL_POSITIVE,
    LABEL_NEGATIVE,
    LABEL_UNCERTAIN,
    MANUAL_POSITIVE_SCORE,
    MANUAL_NEGATIVE_SCORE,
    MANUAL_UNCERTAIN_SCORE,
    RANDOM_SEED,
)


LABEL_NAMES = {
    LABEL_POSITIVE: "POSITIVE",
    LABEL_NEGATIVE: "NEGATIVE",
    LABEL_UNCERTAIN: "UNCERTAIN",
}

# Reverse mapping: string → integer
LABEL_ENCODING = {
    "POSITIVE": LABEL_POSITIVE,
    "NEGATIVE": LABEL_NEGATIVE,
    "UNCERTAIN": LABEL_UNCERTAIN,
}

# Fixed soft_score overrides for manual labels (per spec)
MANUAL_SCORE_MAP = {
    LABEL_POSITIVE: MANUAL_POSITIVE_SCORE,    # 0.95
    LABEL_NEGATIVE: MANUAL_NEGATIVE_SCORE,    # 0.05
    LABEL_UNCERTAIN: MANUAL_UNCERTAIN_SCORE,  # 0.50
}


def main():
    t_start = time.time()

    print("=" * 70)
    print("STAGE 4 — STEPS 4.4 & 4.5: MERGE MANUAL LABELS & FINAL LABEL SET")
    print("(Manual Labels → Override Pipeline → Remove UNCERTAIN → Final CSV)")
    print("=" * 70)
    print()
    print(f"  MANUAL_POSITIVE_SCORE: {MANUAL_POSITIVE_SCORE}")
    print(f"  MANUAL_NEGATIVE_SCORE: {MANUAL_NEGATIVE_SCORE}")
    print(f"  MANUAL_UNCERTAIN_SCORE: {MANUAL_UNCERTAIN_SCORE}")
    print(f"  RANDOM_SEED:           {RANDOM_SEED}")
    print()

    # =====================================================================
    # LOAD ACTIVE LEARNING QUEUE (with manual labels)
    # =====================================================================
    if not os.path.exists(ACTIVE_LEARNING_QUEUE_CSV):
        print(f"ERROR: active_learning_queue.csv not found at: {ACTIVE_LEARNING_QUEUE_CSV}")
        print("  Run Steps 4.1-4.2 first.")
        return 1

    print(f"Loading manually labeled queue from {ACTIVE_LEARNING_QUEUE_CSV}...")
    df_queue = pd.read_csv(ACTIVE_LEARNING_QUEUE_CSV, low_memory=False)
    n_queue = len(df_queue)
    print(f"  Reports in queue: {n_queue:,}")
    print()

    # Validate manual_label column exists
    if 'manual_label' not in df_queue.columns:
        print("ERROR: 'manual_label' column not found in active_learning_queue.csv!")
        print("  This column must be filled in during Step 4.3.")
        return 1

    # Check for empty/missing labels
    df_queue['manual_label'] = df_queue['manual_label'].astype(str).str.strip().str.upper()
    empty_mask = (df_queue['manual_label'] == '') | (df_queue['manual_label'] == 'NAN')
    n_empty = int(empty_mask.sum())

    if n_empty > 0:
        print(f"  WARNING: {n_empty} reports have empty/missing manual_label values!")
        print(f"  These reports have not been labeled yet.")
        print()

        if n_empty == n_queue:
            print("  ERROR: ALL reports are unlabeled. Complete Step 4.3 first.")
            print("  Open the CSV, label each report in the 'manual_label' column,")
            print("  save, and re-run this script.")
            return 1
        else:
            print(f"  Proceeding with {n_queue - n_empty} labeled reports.")
            print(f"  The {n_empty} unlabeled reports will be excluded.")
            df_queue = df_queue[~empty_mask].copy()
            n_queue = len(df_queue)
    print()

    # Validate label values
    valid_labels = set(LABEL_ENCODING.keys())
    invalid_mask = ~df_queue['manual_label'].isin(valid_labels)
    n_invalid = int(invalid_mask.sum())

    if n_invalid > 0:
        invalid_vals = df_queue[invalid_mask]['manual_label'].unique().tolist()
        print(f"  ERROR: {n_invalid} reports have invalid manual_label values: {invalid_vals}")
        print(f"  Valid values are: {sorted(valid_labels)}")
        return 1

    print(f"  ✓ All {n_queue} manual labels are valid.")
    print()

    # Convert string labels to integer encoding
    df_queue['manual_label_int'] = df_queue['manual_label'].map(LABEL_ENCODING)

    # Assign fixed soft_scores per spec
    df_queue['manual_soft_score'] = df_queue['manual_label_int'].map(MANUAL_SCORE_MAP)

    # Manual label distribution
    manual_counts = Counter(df_queue['manual_label'].tolist())
    print(f"  Manual label distribution ({n_queue} reports):")
    for label_name in ['POSITIVE', 'NEGATIVE', 'UNCERTAIN']:
        cnt = manual_counts.get(label_name, 0)
        score = MANUAL_SCORE_MAP.get(LABEL_ENCODING[label_name], '?')
        print(f"    {label_name:>12s}: {cnt:>4}  ({100*cnt/n_queue:.1f}%)  → soft_score = {score}")
    print()

    n_manual_uncertain = manual_counts.get('UNCERTAIN', 0)
    n_manual_training = n_queue - n_manual_uncertain
    print(f"  Manual labels entering training:    {n_manual_training}")
    print(f"  Manual labels excluded (UNCERTAIN): {n_manual_uncertain}")
    print()

    sys.stdout.flush()

    # =====================================================================
    # STEP 4.4 — MERGE MANUAL LABELS INTO CONFIDENT POOL
    # =====================================================================
    print("=" * 70)
    print("STEP 4.4 — MERGE MANUAL LABELS INTO CONFIDENT POOL")
    print("=" * 70)
    print()

    # Load confident pool from Stage 3
    if not os.path.exists(CONFIDENT_POOL_CSV):
        print(f"ERROR: confident_pool.csv not found at: {CONFIDENT_POOL_CSV}")
        print("  Run Stage 3 first.")
        return 1

    print(f"Loading confident pool from {CONFIDENT_POOL_CSV}...")
    df_confident = pd.read_csv(CONFIDENT_POOL_CSV, low_memory=False)
    n_confident = len(df_confident)
    print(f"  Confident pool loaded: {n_confident:,}")
    print()

    # Ensure study_id types match
    df_confident['study_id'] = df_confident['study_id'].astype(str)
    df_queue['study_id'] = df_queue['study_id'].astype(str)

    # Check for overlap: manual labels should NOT already be in the confident pool
    # (they came from the uncertain pool), but verify just in case
    manual_study_ids = set(df_queue['study_id'].tolist())
    overlap = manual_study_ids & set(df_confident['study_id'].tolist())
    if len(overlap) > 0:
        print(f"  NOTE: {len(overlap)} manually labeled reports found in confident pool.")
        print(f"  These will be removed from the confident pool and replaced with manual labels.")
        df_confident = df_confident[~df_confident['study_id'].isin(manual_study_ids)].copy()
        print(f"  Confident pool after removal: {len(df_confident):,}")
    else:
        print(f"  ✓ No overlap between manual labels and confident pool.")
    print()

    # Build the manual labels DataFrame with the same column structure
    # as the confident pool for clean concatenation
    df_manual = pd.DataFrame({
        'subject_id': np.nan,  # Will be filled below
        'study_id': df_queue['study_id'].values,
        'soft_score': df_queue['manual_soft_score'].values,
        'label': df_queue['manual_label_int'].values,
        'label_name': df_queue['manual_label'].values,
        'label_source': 'manual',
    })

    # Try to fill subject_id from the queue or the LF results
    if 'subject_id' in df_queue.columns:
        df_manual['subject_id'] = df_queue['subject_id'].values
    else:
        # Look up subject_id from confident pool or source data
        lf_results_csv = os.path.join(DATA_INTERMEDIATE, "lf1_to_lf6_results.csv")
        if os.path.exists(lf_results_csv):
            df_lf = pd.read_csv(lf_results_csv, usecols=['study_id', 'subject_id'], low_memory=False)
            df_lf['study_id'] = df_lf['study_id'].astype(str)
            sid_map = df_lf.set_index('study_id')['subject_id'].to_dict()
            df_manual['subject_id'] = df_manual['study_id'].map(sid_map)

    print(f"  Manual labels prepared: {len(df_manual):,}")
    print(f"    POSITIVE: {(df_manual['label'] == LABEL_POSITIVE).sum()}")
    print(f"    NEGATIVE: {(df_manual['label'] == LABEL_NEGATIVE).sum()}")
    print(f"    UNCERTAIN: {(df_manual['label'] == LABEL_UNCERTAIN).sum()}")
    print()

    sys.stdout.flush()

    # =====================================================================
    # STEP 4.5 — FINAL COMBINED LABEL SET
    # =====================================================================
    print("=" * 70)
    print("STEP 4.5 — FINAL COMBINED LABEL SET")
    print("=" * 70)
    print()

    # Prepare confident pool for merging
    # Map assigned_label to the standard 'label' column
    df_confident_out = pd.DataFrame({
        'subject_id': df_confident['subject_id'].values,
        'study_id': df_confident['study_id'].values,
        'soft_score': df_confident['soft_score'].values,
        'label': df_confident['assigned_label'].values,
        'label_name': df_confident['assigned_label_name'].values,
        'label_source': df_confident['label_source'].values,
    })

    # Concatenate: confident pool (pipeline + pre_filter + all_abstain) + manual
    df_combined = pd.concat([df_confident_out, df_manual], ignore_index=True)
    n_combined = len(df_combined)
    print(f"  Combined label set (before UNCERTAIN removal): {n_combined:,}")
    print()

    # Source breakdown before filtering
    source_counts = Counter(df_combined['label_source'].tolist())
    print(f"  By label source:")
    for source, cnt in source_counts.most_common():
        print(f"    {source:20s}: {cnt:>8,}  ({100*cnt/n_combined:.1f}%)")
    print()

    # Label breakdown before filtering
    label_counts = Counter(df_combined['label'].tolist())
    print(f"  By label (before UNCERTAIN removal):")
    for code in [LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN]:
        cnt = label_counts.get(code, 0)
        name = LABEL_NAMES[code]
        print(f"    {name:>12s}: {cnt:>8,}  ({100*cnt/n_combined:.1f}%)")
    print()

    # Remove UNCERTAIN labels (they do NOT enter training)
    n_uncertain_removed = int((df_combined['label'] == LABEL_UNCERTAIN).sum())
    df_final = df_combined[df_combined['label'] != LABEL_UNCERTAIN].copy()
    n_final = len(df_final)

    print(f"  UNCERTAIN labels removed: {n_uncertain_removed:,}")
    print(f"  Final label set size:     {n_final:,}")
    print()

    # Verify no duplicates
    n_unique = df_final['study_id'].nunique()
    if n_unique != n_final:
        n_dups = n_final - n_unique
        print(f"  WARNING: {n_dups} duplicate study_ids found!")
        # Keep manual labels over pipeline labels (manual appears last in concat)
        df_final = df_final.drop_duplicates(subset='study_id', keep='last')
        n_final = len(df_final)
        print(f"  After deduplication: {n_final:,}")
    else:
        print(f"  ✓ No duplicate study_ids. All {n_final:,} are unique.")
    print()

    # Final label distribution
    final_label_counts = Counter(df_final['label'].tolist())
    n_final_pos = final_label_counts.get(LABEL_POSITIVE, 0)
    n_final_neg = final_label_counts.get(LABEL_NEGATIVE, 0)

    print(f"  FINAL LABEL DISTRIBUTION:")
    print(f"    POSITIVE: {n_final_pos:>8,}  ({100*n_final_pos/n_final:.1f}%)")
    print(f"    NEGATIVE: {n_final_neg:>8,}  ({100*n_final_neg/n_final:.1f}%)")
    print()

    if n_final_neg > 0:
        ratio = n_final_pos / n_final_neg
        print(f"  Class ratio (POS/NEG): {ratio:.4f}")
        print(f"  Ratio (NEG/POS):       {1/ratio:.2f}:1")
    print()

    # Final source breakdown
    final_source_counts = Counter(df_final['label_source'].tolist())
    print(f"  By label source:")
    for source, cnt in final_source_counts.most_common():
        print(f"    {source:20s}: {cnt:>8,}  ({100*cnt/n_final:.1f}%)")
    print()

    # Soft score statistics by label
    print(f"  Soft score statistics by label:")
    for code, name in [(LABEL_POSITIVE, "POSITIVE"), (LABEL_NEGATIVE, "NEGATIVE")]:
        scores = df_final[df_final['label'] == code]['soft_score'].values
        if len(scores) > 0:
            print(f"    {name:>12s}: mean={scores.mean():.4f}  median={np.median(scores):.4f}  "
                  f"std={scores.std():.4f}  [{scores.min():.4f}, {scores.max():.4f}]")
    print()

    # Save final labels
    output_columns = ['subject_id', 'study_id', 'label', 'soft_score', 'label_source']
    os.makedirs(os.path.dirname(FINAL_LABELS_CSV), exist_ok=True)
    df_final[output_columns].to_csv(FINAL_LABELS_CSV, index=False)
    file_size_mb = os.path.getsize(FINAL_LABELS_CSV) / (1024 * 1024)

    print(f"  Saved: {FINAL_LABELS_CSV}")
    print(f"    Rows:    {n_final:,}")
    print(f"    Size:    {file_size_mb:.1f} MB")
    print(f"    Columns: {output_columns}")
    print()

    sys.stdout.flush()

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    t_total = time.time() - t_start

    print("=" * 70)
    print("STAGE 4 COMPLETE — FINAL PNEUMONIA LABEL SET CREATED")
    print("=" * 70)
    print()
    print(f"  Total labeled reports: {n_final:,}")
    print(f"    POSITIVE: {n_final_pos:,}  ({100*n_final_pos/n_final:.1f}%)")
    print(f"    NEGATIVE: {n_final_neg:,}  ({100*n_final_neg/n_final:.1f}%)")
    print()
    print(f"  Label sources:")
    for source, cnt in final_source_counts.most_common():
        print(f"    {source:20s}: {cnt:>8,}")
    print()
    print(f"  Output: {FINAL_LABELS_CSV}")
    print(f"  Runtime: {t_total:.1f}s")
    print()
    print(f"  Next: Stage 5 — Validation (Cohen's Kappa on {300} reports)")
    print(f"        Stage 6 — Image Linking (connect labels to CXR images)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Stage 5 — Step 5.1: Sample 300 Random Reports for Validation

Samples 300 reports from the confident pool for independent manual validation.
Explicitly excludes all 200 study_ids that were manually reviewed in Stage 4
to prevent overlap. The export file contains only study_id, impression_text,
and findings_text — NO pipeline labels are included to prevent anchoring bias
during manual labeling in Step 5.2.

Input:   confident_pool.csv          (from Stage 3)
         active_learning_queue.csv   (from Stage 4 — for exclusion)
         parsed_reports.csv          (from Stage 1 — for text columns)
Output:  validation_sample_300.csv   (for blind manual labeling)

Estimated runtime: < 30 seconds
"""

import os
import sys
import time

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CONFIDENT_POOL_CSV,
    ACTIVE_LEARNING_QUEUE_CSV,
    PARSED_REPORTS_CSV,
    DATA_OUTPUT,
    RANDOM_SEED,
    VALIDATION_SAMPLE_SIZE,
)


def main():
    t_start = time.time()

    print("=" * 70)
    print("STAGE 5 — STEP 5.1: SAMPLE 300 REPORTS FOR VALIDATION")
    print("(Confident Pool → Exclude Stage 4 IDs → Random Sample → Blind Export)")
    print("=" * 70)
    print()
    print(f"  VALIDATION_SAMPLE_SIZE: {VALIDATION_SAMPLE_SIZE}")
    print(f"  RANDOM_SEED:           {RANDOM_SEED}")
    print()

    # =====================================================================
    # LOAD CONFIDENT POOL
    # =====================================================================
    if not os.path.exists(CONFIDENT_POOL_CSV):
        print(f"ERROR: confident_pool.csv not found at: {CONFIDENT_POOL_CSV}")
        print("  Run Stage 3 first.")
        return 1

    print(f"Loading confident pool from {CONFIDENT_POOL_CSV}...")
    df_confident = pd.read_csv(CONFIDENT_POOL_CSV, low_memory=False)
    df_confident['study_id'] = df_confident['study_id'].astype(str)
    n_confident = len(df_confident)
    print(f"  Confident pool loaded: {n_confident:,} reports")
    print()

    # =====================================================================
    # LOAD STAGE 4 STUDY_IDS FOR EXCLUSION
    # =====================================================================
    if not os.path.exists(ACTIVE_LEARNING_QUEUE_CSV):
        print(f"WARNING: active_learning_queue.csv not found at: {ACTIVE_LEARNING_QUEUE_CSV}")
        print("  Proceeding without exclusion (no Stage 4 overlap to remove).")
        stage4_ids = set()
    else:
        print(f"Loading Stage 4 study_ids for exclusion from {ACTIVE_LEARNING_QUEUE_CSV}...")
        df_queue = pd.read_csv(ACTIVE_LEARNING_QUEUE_CSV, usecols=['study_id'], low_memory=False)
        df_queue['study_id'] = df_queue['study_id'].astype(str)
        stage4_ids = set(df_queue['study_id'].tolist())
        print(f"  Stage 4 study_ids to exclude: {len(stage4_ids)}")
    print()

    # =====================================================================
    # EXCLUDE STAGE 4 REPORTS FROM CONFIDENT POOL
    # =====================================================================
    overlap_mask = df_confident['study_id'].isin(stage4_ids)
    n_overlap = int(overlap_mask.sum())
    print(f"  Overlap found in confident pool: {n_overlap}")

    df_eligible = df_confident[~overlap_mask].copy()
    n_eligible = len(df_eligible)
    print(f"  Eligible reports after exclusion: {n_eligible:,}")
    print()

    if n_eligible < VALIDATION_SAMPLE_SIZE:
        print(f"  ERROR: Only {n_eligible} eligible reports available, "
              f"but {VALIDATION_SAMPLE_SIZE} required.")
        print("  Reduce VALIDATION_SAMPLE_SIZE or check data integrity.")
        return 1

    # =====================================================================
    # RANDOM SAMPLE OF 300 REPORTS
    # =====================================================================
    print(f"  Sampling {VALIDATION_SAMPLE_SIZE} reports with seed={RANDOM_SEED}...")
    df_sample = df_eligible.sample(
        n=VALIDATION_SAMPLE_SIZE,
        random_state=RANDOM_SEED,
        replace=False,
    )
    print(f"  ✓ Sampled {len(df_sample):,} reports")
    print()

    # Record pipeline label distribution of the sample (for internal reference only)
    if 'assigned_label_name' in df_sample.columns:
        sample_label_dist = df_sample['assigned_label_name'].value_counts()
        print(f"  Pipeline label distribution of sample (internal reference only):")
        for lbl, cnt in sample_label_dist.items():
            print(f"    {lbl:>12s}: {cnt:>4}  ({100*cnt/len(df_sample):.1f}%)")
        print()

    # =====================================================================
    # LOAD PARSED REPORTS FOR TEXT COLUMNS
    # =====================================================================
    print(f"Loading parsed reports for text columns from {PARSED_REPORTS_CSV}...")
    df_parsed = pd.read_csv(
        PARSED_REPORTS_CSV,
        usecols=['study_id', 'impression_text', 'findings_text'],
        low_memory=False,
    )
    df_parsed['study_id'] = df_parsed['study_id'].astype(str)
    print(f"  Parsed reports loaded: {len(df_parsed):,}")
    print()

    # =====================================================================
    # MERGE TEXT COLUMNS INTO SAMPLE
    # =====================================================================
    sample_ids = set(df_sample['study_id'].tolist())
    df_text = df_parsed[df_parsed['study_id'].isin(sample_ids)].copy()
    print(f"  Text records matched: {len(df_text):,}")

    # Some confident pool reports may be pre_filter/all_abstain with no parsed text
    # For those, we'll use empty strings
    df_export = df_sample[['study_id']].merge(df_text, on='study_id', how='left')
    df_export['impression_text'] = df_export['impression_text'].fillna('')
    df_export['findings_text'] = df_export['findings_text'].fillna('')

    # Add empty manual_label column for Step 5.2
    df_export['manual_label'] = ''

    # =====================================================================
    # EXPORT — BLIND (NO PIPELINE LABELS)
    # =====================================================================
    output_path = os.path.join(DATA_OUTPUT, "validation_sample_300.csv")
    export_columns = ['study_id', 'impression_text', 'findings_text', 'manual_label']
    df_export[export_columns].to_csv(output_path, index=False)
    file_size_kb = os.path.getsize(output_path) / 1024

    print()
    print(f"  ✓ Validation sample exported: {output_path}")
    print(f"    Rows:    {len(df_export):,}")
    print(f"    Size:    {file_size_kb:.1f} KB")
    print(f"    Columns: {export_columns}")
    print()

    # Also save the internal reference with pipeline labels (NOT for manual use)
    # This is used by Step 5.3 to retrieve pipeline labels for Kappa calculation
    ref_path = os.path.join(DATA_OUTPUT, "validation_sample_300_reference.csv")
    ref_cols = ['study_id']
    if 'assigned_label' in df_sample.columns:
        ref_cols.append('assigned_label')
    if 'assigned_label_name' in df_sample.columns:
        ref_cols.append('assigned_label_name')
    if 'soft_score' in df_sample.columns:
        ref_cols.append('soft_score')
    if 'label_source' in df_sample.columns:
        ref_cols.append('label_source')

    df_sample[ref_cols].to_csv(ref_path, index=False)
    print(f"  ✓ Internal reference saved: {ref_path}")
    print(f"    (Contains pipeline labels — DO NOT open during manual labeling)")
    print()

    # Check for reports with both impression and findings empty
    n_both_empty = int(((df_export['impression_text'] == '') & 
                        (df_export['findings_text'] == '')).sum())
    if n_both_empty > 0:
        print(f"  ⚠ WARNING: {n_both_empty} reports have BOTH impression and findings empty.")
        print(f"    These may be pre-filter negatives. Label them based on any available text.")
    print()

    # =====================================================================
    # SUMMARY
    # =====================================================================
    t_total = time.time() - t_start

    print("=" * 70)
    print("STEP 5.1 COMPLETE — VALIDATION SAMPLE READY")
    print("=" * 70)
    print()
    print(f"  Sampled: {VALIDATION_SAMPLE_SIZE} reports from confident pool")
    print(f"  Excluded: {len(stage4_ids)} Stage 4 study_ids")
    print(f"  Output:  {output_path}")
    print(f"  Runtime: {t_total:.1f}s")
    print()
    print(f"  NEXT STEP:")
    print(f"    1. Open: {output_path}")
    print(f"    2. Read each report's impression_text and findings_text")
    print(f"    3. Fill the 'manual_label' column with POSITIVE, NEGATIVE, or UNCERTAIN")
    print(f"    4. Save the file")
    print(f"    5. Run Step 5.3 to calculate Cohen's Kappa")
    print()
    print(f"  ⚠ DO NOT view pipeline labels or soft_scores during manual labeling!")
    print(f"  ⚠ DO NOT open validation_sample_300_reference.csv until Step 5.3!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

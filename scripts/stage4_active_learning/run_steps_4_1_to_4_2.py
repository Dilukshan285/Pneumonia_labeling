"""
Stage 4 — Steps 4.1 and 4.2: Active Learning Queue Generation

Ranks all reports in the uncertain pool by confusion score (absolute distance
from 0.5) and selects the top 200 most ambiguous reports where the six labeling
functions disagreed most strongly. Exports these as active_learning_queue.csv
with full report text and all LF votes for manual review.

Input:   uncertain_pool.csv        (from Stage 3)
         lf1_to_lf6_results.csv    (from Stage 2, contains report text + LF votes)
Output:  active_learning_queue.csv (200 most uncertain reports for manual labeling)

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
    UNCERTAIN_POOL_CSV,
    DATA_INTERMEDIATE,
    DATA_OUTPUT,
    ACTIVE_LEARNING_QUEUE_CSV,
    ACTIVE_LEARNING_COUNT,
    LABEL_POSITIVE,
    LABEL_NEGATIVE,
    LABEL_UNCERTAIN,
    LABEL_ABSTAIN,
    RANDOM_SEED,
)


LABEL_NAMES = {
    LABEL_POSITIVE: "POSITIVE",
    LABEL_NEGATIVE: "NEGATIVE",
    LABEL_UNCERTAIN: "UNCERTAIN",
    LABEL_ABSTAIN: "ABSTAIN",
}

LF_COLUMNS = ['lf1_label', 'lf2_label', 'lf3_label', 'lf4_label', 'lf5_label', 'lf6_label']
LF_NAMES = [
    'LF1 (Keywords)',
    'LF2 (NegEx)',
    'LF3 (CheXpert)',
    'LF4 (Section Weight)',
    'LF5 (NLI Zero-Shot)',
    'LF6 (Uncertainty)',
]

# Input: LF1-LF6 results (contains report_text, impression_text, findings_text, and LF votes)
LF1_TO_LF6_RESULTS_CSV = os.path.join(DATA_INTERMEDIATE, "lf1_to_lf6_results.csv")


def main():
    t_start = time.time()

    print("=" * 70)
    print("STAGE 4 — STEPS 4.1 & 4.2: ACTIVE LEARNING QUEUE GENERATION")
    print("(Rank Uncertain Reports → Select Top 200 → Export for Manual Review)")
    print("=" * 70)
    print()
    print(f"  ACTIVE_LEARNING_COUNT: {ACTIVE_LEARNING_COUNT}")
    print(f"  RANDOM_SEED:           {RANDOM_SEED}")
    print()

    # =====================================================================
    # LOAD UNCERTAIN POOL
    # =====================================================================
    if not os.path.exists(UNCERTAIN_POOL_CSV):
        print(f"ERROR: uncertain_pool.csv not found at: {UNCERTAIN_POOL_CSV}")
        print("  Run Stage 3 first.")
        return 1

    print(f"Loading uncertain pool from {UNCERTAIN_POOL_CSV}...")
    df_uncertain = pd.read_csv(UNCERTAIN_POOL_CSV, low_memory=False)
    n_uncertain = len(df_uncertain)
    print(f"  Uncertain reports loaded: {n_uncertain:,}")
    print()

    if n_uncertain == 0:
        print("ERROR: Uncertain pool is empty. No reports to select.")
        return 1

    if n_uncertain < ACTIVE_LEARNING_COUNT:
        print(f"  WARNING: Uncertain pool ({n_uncertain}) is smaller than")
        print(f"  ACTIVE_LEARNING_COUNT ({ACTIVE_LEARNING_COUNT}).")
        print(f"  Will select all {n_uncertain} reports instead.")
        n_select = n_uncertain
    else:
        n_select = ACTIVE_LEARNING_COUNT

    # =====================================================================
    # LOAD LF1-LF6 RESULTS (for report text and LF votes)
    # =====================================================================
    if not os.path.exists(LF1_TO_LF6_RESULTS_CSV):
        print(f"ERROR: lf1_to_lf6_results.csv not found at: {LF1_TO_LF6_RESULTS_CSV}")
        print("  This file contains report_text and LF votes needed for the queue.")
        return 1

    print(f"Loading LF1-LF6 results for report text and LF votes...")
    df_lf = pd.read_csv(LF1_TO_LF6_RESULTS_CSV, low_memory=False)
    print(f"  LF results loaded: {len(df_lf):,}")
    print()

    # Verify required columns in LF results
    required_lf_cols = ['study_id', 'report_text', 'impression_text', 'findings_text'] + LF_COLUMNS
    missing = [c for c in required_lf_cols if c not in df_lf.columns]
    if missing:
        print(f"ERROR: Missing columns in LF results: {missing}")
        return 1
    print(f"  All required columns verified in LF results.")
    print()

    sys.stdout.flush()

    # =====================================================================
    # STEP 4.1 — RANK UNCERTAIN REPORTS BY CONFUSION SCORE
    # =====================================================================
    print("=" * 70)
    print("STEP 4.1 — RANK UNCERTAIN REPORTS BY CONFUSION SCORE")
    print("=" * 70)
    print()

    # Confusion score = |soft_score - 0.5|
    # Score of 0.0 = maximum disagreement (soft_score exactly 0.5)
    # Score closer to 0.25 = less disagreement (soft_score near boundaries)
    df_uncertain = df_uncertain.copy()

    # The uncertain pool from Stage 3 already has boundary_distance column
    # but we recalculate to ensure consistency and name it confusion_score
    df_uncertain['confusion_score'] = np.abs(df_uncertain['soft_score'].values - 0.5)

    # Sort ascending: most confused (closest to 0.5) first
    df_uncertain = df_uncertain.sort_values('confusion_score', ascending=True).reset_index(drop=True)

    print(f"  Confusion score (|soft_score - 0.5|) calculated for {n_uncertain:,} reports.")
    print()
    print(f"  Confusion score distribution:")
    print(f"    Min (most confused):  {df_uncertain['confusion_score'].min():.4f}")
    print(f"    Max (least confused): {df_uncertain['confusion_score'].max():.4f}")
    print(f"    Mean:                 {df_uncertain['confusion_score'].mean():.4f}")
    print(f"    Median:               {df_uncertain['confusion_score'].median():.4f}")
    print()

    # Show the score distribution in bins
    bins = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50]
    bin_labels_hist = ["0.00-0.01", "0.01-0.02", "0.02-0.05", "0.05-0.10",
                       "0.10-0.15", "0.15-0.20", "0.20-0.25", "0.25+"]
    hist, _ = np.histogram(df_uncertain['confusion_score'].values, bins=bins)
    max_cnt = max(hist) if max(hist) > 0 else 1
    print(f"  Confusion score histogram:")
    for lbl, cnt in zip(bin_labels_hist, hist):
        bar = "█" * min(40, int(40 * cnt / max_cnt))
        print(f"    {lbl}: {cnt:>6,}  {bar}")
    print()

    sys.stdout.flush()

    # =====================================================================
    # STEP 4.2 — SELECT TOP 200 MOST UNCERTAIN REPORTS
    # =====================================================================
    print("=" * 70)
    print(f"STEP 4.2 — SELECT TOP {n_select} MOST UNCERTAIN REPORTS")
    print("=" * 70)
    print()

    # Select the top N most confused reports
    df_top = df_uncertain.head(n_select).copy()
    selected_study_ids = set(df_top['study_id'].tolist())

    print(f"  Selected {n_select} reports with lowest confusion scores.")
    print(f"    Confusion score range of selected: "
          f"[{df_top['confusion_score'].min():.4f}, {df_top['confusion_score'].max():.4f}]")
    print(f"    Soft score range of selected: "
          f"[{df_top['soft_score'].min():.4f}, {df_top['soft_score'].max():.4f}]")
    print()

    # Join with LF results to get report text and LF votes
    # Use study_id as the join key
    print(f"  Joining with LF results for report text and LF votes...")

    # Ensure study_id types match for join
    df_top['study_id'] = df_top['study_id'].astype(str)
    df_lf['study_id'] = df_lf['study_id'].astype(str)

    # Select only needed columns from LF results to avoid column conflicts
    lf_join_cols = ['study_id', 'report_text', 'impression_text', 'findings_text'] + LF_COLUMNS
    df_lf_subset = df_lf[lf_join_cols].drop_duplicates(subset='study_id')

    # Merge: left join to preserve all selected reports
    df_queue = df_top[['study_id', 'soft_score', 'confusion_score']].merge(
        df_lf_subset,
        on='study_id',
        how='left',
    )

    # Verify join completeness
    n_matched = df_queue['report_text'].notna().sum()
    n_missing = n_select - n_matched
    print(f"    Matched: {n_matched:,} / {n_select} reports")
    if n_missing > 0:
        print(f"    WARNING: {n_missing} reports could not be joined with LF results!")
        print(f"    These reports will have empty text and LF columns.")
    else:
        print(f"    ✓ All {n_select} reports matched successfully.")
    print()

    # Sort by confusion_score (most confused first) — redundant but explicit
    df_queue = df_queue.sort_values('confusion_score', ascending=True).reset_index(drop=True)

    # Add a blank manual_label column for the human annotator
    # This is where the manual labels will be entered in Step 4.3
    df_queue['manual_label'] = ''

    # Convert LF labels to readable names for easier human review
    # Create readable vote columns alongside the integer columns
    for lf_col in LF_COLUMNS:
        readable_col = lf_col.replace('_label', '_vote')
        if lf_col in df_queue.columns:
            df_queue[readable_col] = df_queue[lf_col].map(LABEL_NAMES).fillna('UNKNOWN')

    # Define the output column order per spec
    # Spec requires: study_id, report_text, impression_text, findings_text,
    #                soft_score, LF1_vote through LF6_vote
    # We add: confusion_score, lf1-lf6 integer labels, manual_label
    output_columns = [
        'study_id',
        'report_text',
        'impression_text',
        'findings_text',
        'soft_score',
        'confusion_score',
        'lf1_vote', 'lf2_vote', 'lf3_vote', 'lf4_vote', 'lf5_vote', 'lf6_vote',
        'lf1_label', 'lf2_label', 'lf3_label', 'lf4_label', 'lf5_label', 'lf6_label',
        'manual_label',
    ]

    # Verify all output columns exist
    available_cols = [c for c in output_columns if c in df_queue.columns]
    missing_out_cols = [c for c in output_columns if c not in df_queue.columns]
    if missing_out_cols:
        print(f"  WARNING: Missing output columns (will be omitted): {missing_out_cols}")
        output_columns = available_cols

    # Save the active learning queue
    os.makedirs(os.path.dirname(ACTIVE_LEARNING_QUEUE_CSV), exist_ok=True)
    df_queue[output_columns].to_csv(ACTIVE_LEARNING_QUEUE_CSV, index=False)
    queue_size_mb = os.path.getsize(ACTIVE_LEARNING_QUEUE_CSV) / (1024 * 1024)

    print(f"  Saved: {ACTIVE_LEARNING_QUEUE_CSV}")
    print(f"    Rows: {len(df_queue):,}")
    print(f"    Size: {queue_size_mb:.2f} MB")
    print(f"    Columns: {output_columns}")
    print()

    # LF vote distribution within selected reports
    print(f"  LF vote distribution within the {n_select} selected reports:")
    for lf_col, lf_name in zip(LF_COLUMNS, LF_NAMES):
        if lf_col in df_queue.columns:
            vals = df_queue[lf_col].values
            pos = int(np.sum(vals == LABEL_POSITIVE))
            neg = int(np.sum(vals == LABEL_NEGATIVE))
            unc = int(np.sum(vals == LABEL_UNCERTAIN))
            abst = int(np.sum(vals == LABEL_ABSTAIN))
            print(f"    {lf_name:25s}  POS={pos:>4}  NEG={neg:>4}  UNC={unc:>4}  ABSTAIN={abst:>4}")
    print()

    # Preview first 5 reports (truncated text)
    print(f"  Preview — Top 5 most uncertain reports:")
    print(f"  {'-'*70}")
    for i, (_, row) in enumerate(df_queue.head(5).iterrows()):
        imp_text = str(row.get('impression_text', ''))[:120]
        find_text = str(row.get('findings_text', ''))[:120]
        votes = []
        for lf_col in LF_COLUMNS:
            v = row.get(lf_col, -1)
            votes.append(LABEL_NAMES.get(int(v), '?'))
        vote_str = ', '.join(votes)

        print(f"  [{i+1}] study_id={row['study_id']}  score={row['soft_score']:.4f}  confusion={row['confusion_score']:.4f}")
        print(f"      IMPRESSION: \"{imp_text}\"")
        print(f"      FINDINGS:   \"{find_text}\"")
        print(f"      LF VOTES:   [{vote_str}]")
        print()
    print()

    sys.stdout.flush()

    # =====================================================================
    # INSTRUCTIONS FOR STEP 4.3
    # =====================================================================
    print("=" * 70)
    print("STEP 4.3 — MANUAL LABELING INSTRUCTIONS")
    print("=" * 70)
    print()
    print(f"  The file {ACTIVE_LEARNING_QUEUE_CSV}")
    print(f"  contains {n_select} reports ready for manual labeling.")
    print()
    print(f"  INSTRUCTIONS:")
    print(f"    1. Open the CSV file in a spreadsheet editor (Excel, LibreOffice).")
    print(f"    2. For each row, read the impression_text and findings_text columns.")
    print(f"    3. In the 'manual_label' column, enter one of:")
    print(f"         POSITIVE  — if pneumonia IS present per the radiologist's conclusion")
    print(f"         NEGATIVE  — if pneumonia is NOT present")
    print(f"         UNCERTAIN — ONLY if the text is genuinely hedged and unresolvable")
    print(f"    4. Do NOT look at soft_score or LF vote columns during labeling")
    print(f"       to avoid anchoring bias.")
    print(f"    5. Save the file when done.")
    print(f"    6. Run run_steps_4_4_to_4_5.py to merge labels and create final dataset.")
    print()
    print(f"  Estimated time: 3-5 hours for {n_select} reports.")
    print()
    print(f"  CLINICAL NOTE:")
    print(f"    If a clinical collaborator or practicing radiologist is available,")
    print(f"    their review of ambiguous cases is strongly recommended.")
    print(f"    Document reviewer credentials in the research paper.")
    print()

    t_total = time.time() - t_start

    print("=" * 70)
    print(f"STEPS 4.1-4.2 COMPLETE — ACTIVE LEARNING QUEUE GENERATED")
    print(f"  Output: {ACTIVE_LEARNING_QUEUE_CSV}")
    print(f"  Reports queued: {n_select}")
    print(f"  Runtime: {t_total:.1f}s")
    print(f"  Next: Complete manual labeling (Step 4.3), then run Steps 4.4-4.5")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Stage 2 — Run Steps 2.17 through 2.19
(Snorkel Label Model: Build Matrix → Train → Generate Soft Scores)

Executes:
  Step 2.17 — Build the Snorkel label matrix from all 6 LF outputs
  Step 2.18 — Train the Snorkel LabelModel (cardinality=3, CPU-only)
  Step 2.19 — Generate soft label probabilities and merge with pre-filter

This script:
  1. Loads LF1-LF6 results from lf1_to_lf6_results.csv
  2. Builds the integer label matrix (163,499 rows × 6 columns)
  3. Validates no all-ABSTAIN rows exist
  4. Trains the Snorkel LabelModel with cardinality=3
  5. Generates predict_proba soft scores (POSITIVE = column index 1)
  6. Saves snorkel_soft_scores.csv
  7. Merges with prefilter_negatives.csv for complete coverage

CRITICAL: Snorkel 0.9.8 has a known incompatibility with Python 3.11+
    collections.Hashable was removed from the collections module in Python 3.10+
    and moved to collections.abc. This script applies the compatibility fix
    BEFORE importing any Snorkel module. This MUST be the first thing that runs.

Estimated runtime: ~2-5 minutes on CPU (32GB RAM, Ryzen 7 8845HS)
"""

# ============================================================================
# PYTHON 3.11+ COMPATIBILITY FIX FOR SNORKEL 0.9.8
# This MUST execute before ANY Snorkel import. Do NOT move this block.
# ============================================================================
import collections.abc

# Restore removed collections attributes that Snorkel 0.9.8 depends on
_COMPAT_ATTRS = [
    'Hashable', 'Mapping', 'MutableMapping', 'MutableSet',
    'Callable', 'Iterable', 'Iterator', 'Sequence',
]
for _attr in _COMPAT_ATTRS:
    if not hasattr(collections, _attr):
        setattr(collections, _attr, getattr(collections.abc, _attr))
# ============================================================================

import os
import sys
import time

import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_INTERMEDIATE,
    PREFILTER_NEGATIVES_CSV,
    SNORKEL_SOFT_SCORES_CSV,
    SNORKEL_CARDINALITY,
    SNORKEL_EPOCHS,
    SNORKEL_LR,
    SNORKEL_OPTIMIZER,
    RANDOM_SEED,
    PREFILTER_NEGATIVE_SCORE,
    LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN, LABEL_ABSTAIN,
)
from stage2_labeling.keywords import KEYWORD_LIST_VERSION


LABEL_NAMES = {
    LABEL_POSITIVE: "POSITIVE",
    LABEL_NEGATIVE: "NEGATIVE",
    LABEL_UNCERTAIN: "UNCERTAIN",
    LABEL_ABSTAIN: "ABSTAIN",
}

# Input: LF1-LF6 results (saved by run_steps_2_15_to_2_16.py)
LF1_TO_LF6_RESULTS_CSV = os.path.join(DATA_INTERMEDIATE, "lf1_to_lf6_results.csv")

# The 6 LF label columns in fixed order
LF_COLUMNS = ['lf1_label', 'lf2_label', 'lf3_label', 'lf4_label', 'lf5_label', 'lf6_label']
LF_NAMES = [
    'LF1 (Keywords)',
    'LF2 (NegEx)',
    'LF3 (CheXpert)',
    'LF4 (Section Weight)',
    'LF5 (NLI Zero-Shot)',
    'LF6 (Uncertainty)',
]


def _print_distribution(label_list, total_n, title=""):
    """Helper to print label distribution."""
    counts = Counter(label_list)
    if title:
        print(f"  {title}:")
    for code in [LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN, LABEL_ABSTAIN]:
        cnt = counts.get(code, 0)
        name = LABEL_NAMES[code]
        print(f"    {name:>12s}:  {cnt:>8,}  ({100*cnt/total_n:.1f}%)")
    coverage = total_n - counts.get(LABEL_ABSTAIN, 0)
    print(f"    {'Coverage':>12s}:  {coverage:>8,}  ({100*coverage/total_n:.1f}%)")
    print()


def main():
    print("=" * 70)
    print("STAGE 2 -- STEPS 2.17 THROUGH 2.19 EXECUTION")
    print("(Snorkel Label Model: Build Matrix → Train → Soft Scores)")
    print("=" * 70)
    print()
    print(f"  KEYWORD_LIST_VERSION:  {KEYWORD_LIST_VERSION}")
    print(f"  SNORKEL_CARDINALITY:   {SNORKEL_CARDINALITY}")
    print(f"  SNORKEL_EPOCHS:        {SNORKEL_EPOCHS}")
    print(f"  SNORKEL_LR:            {SNORKEL_LR}")
    print(f"  SNORKEL_OPTIMIZER:     {SNORKEL_OPTIMIZER}")
    print(f"  RANDOM_SEED:           {RANDOM_SEED}")
    print(flush=True)

    # =====================================================================
    # LOAD INPUT DATA
    # =====================================================================
    if not os.path.exists(LF1_TO_LF6_RESULTS_CSV):
        print(f"ERROR: lf1_to_lf6_results.csv not found at: {LF1_TO_LF6_RESULTS_CSV}")
        print("  Run run_steps_2_15_to_2_16.py first.")
        return 1

    print(f"Loading LF1-LF6 results from {LF1_TO_LF6_RESULTS_CSV}...")
    df = pd.read_csv(LF1_TO_LF6_RESULTS_CSV, low_memory=False)

    n_total = len(df)
    print(f"  Reports loaded: {n_total:,}")
    print(flush=True)

    # Verify all 6 LF columns exist
    missing_cols = [c for c in LF_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing LF columns: {missing_cols}")
        return 1
    print(f"  All 6 LF columns verified present.")
    print(flush=True)

    # =====================================================================
    # STEP 2.17 — BUILD THE SNORKEL LABEL MATRIX
    # =====================================================================
    print("=" * 70)
    print("STEP 2.17 — BUILD THE SNORKEL LABEL MATRIX")
    print("=" * 70)
    print()

    # Extract the label matrix as a numpy integer array
    # Shape: (n_reports, 6) — one row per report, one column per LF
    # Values: POSITIVE=1, NEGATIVE=0, UNCERTAIN=2, ABSTAIN=-1
    L = df[LF_COLUMNS].values.astype(int)
    print(f"  Label matrix shape: {L.shape}")
    print(f"  Label matrix dtype: {L.dtype}")
    print()

    # Verify encoding values
    unique_vals = np.unique(L)
    expected_vals = {LABEL_ABSTAIN, LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_UNCERTAIN}
    actual_vals = set(unique_vals.tolist())
    print(f"  Unique values in matrix: {sorted(actual_vals)}")
    print(f"  Expected values:         {sorted(expected_vals)}")

    unexpected = actual_vals - expected_vals
    if unexpected:
        print(f"  ERROR: Unexpected values found: {unexpected}")
        return 1
    print(f"  ✓ All values match expected encoding.")
    print()

    # CRITICAL SAFEGUARD: Check for rows with ALL 6 columns set to ABSTAIN (-1)
    # These occur when a keyword appears in history_text/report_text (passing
    # the pre-filter) but NOT in impression_text or findings_text (where all
    # 6 LFs operate). These rows cause undefined/numerically unstable behavior
    # in the Snorkel LabelModel and MUST be excluded from the label matrix.
    #
    # Clinically, a report with pneumonia terms ONLY in clinical history but
    # NOT in the diagnostic sections (impression/findings) does not describe
    # a current pneumonia finding. These are assigned NEGATIVE with a low
    # soft score, similar to pre-filter negatives.
    all_abstain_mask = np.all(L == LABEL_ABSTAIN, axis=1)
    n_all_abstain = int(all_abstain_mask.sum())
    print(f"  All-ABSTAIN row check:")
    print(f"    Rows with ALL 6 LFs = ABSTAIN: {n_all_abstain}")

    if n_all_abstain > 0:
        print(f"    These {n_all_abstain:,} reports have pneumonia/lung terms in")
        print(f"    history_text or report_text (passing pre-filter) but NOT in")
        print(f"    impression_text or findings_text (where all 6 LFs operate).")
        print(f"    → Excluding from Snorkel matrix.")
        print(f"    → Assigning NEGATIVE with soft_score={PREFILTER_NEGATIVE_SCORE}")
        print()

        # Separate all-ABSTAIN rows
        df_all_abstain = df[all_abstain_mask].copy()
        df_valid = df[~all_abstain_mask].copy()
        L_valid = L[~all_abstain_mask]

        # Show a few examples for transparency
        sample_indices = np.where(all_abstain_mask)[0][:3]
        print(f"    Sample all-ABSTAIN reports:")
        for idx in sample_indices:
            sid = df.iloc[idx]['study_id']
            imp = str(df.iloc[idx].get('impression_text', ''))[:80]
            find = str(df.iloc[idx].get('findings_text', ''))[:80]
            hist = str(df.iloc[idx].get('history_text', ''))[:80]
            print(f"      study_id={sid}")
            print(f"        IMPRESSION: \"{imp}\"")
            print(f"        FINDINGS:   \"{find}\"")
            print(f"        HISTORY:    \"{hist}\"")
        print()
    else:
        print(f"    ✓ No all-ABSTAIN rows found. Pre-filter integrity confirmed.")
        df_all_abstain = pd.DataFrame()
        df_valid = df
        L_valid = L

    n_valid = len(df_valid)
    print(f"  Label matrix for Snorkel training: {L_valid.shape}")
    print(f"    Valid rows (≥1 non-ABSTAIN vote): {n_valid:,}")
    print(f"    Excluded all-ABSTAIN rows:        {n_all_abstain:,}")
    print()

    # Per-LF coverage summary (on valid rows only)
    print(f"  Per-LF vote distribution in label matrix (valid rows):")
    for i, (col, name) in enumerate(zip(LF_COLUMNS, LF_NAMES)):
        lf_vals = L_valid[:, i]
        pos = np.sum(lf_vals == LABEL_POSITIVE)
        neg = np.sum(lf_vals == LABEL_NEGATIVE)
        unc = np.sum(lf_vals == LABEL_UNCERTAIN)
        abst = np.sum(lf_vals == LABEL_ABSTAIN)
        cov = n_valid - abst
        print(f"    {name:25s}  POS={pos:>7,}  NEG={neg:>7,}  "
              f"UNC={unc:>7,}  ABSTAIN={abst:>7,}  COV={100*cov/n_valid:.1f}%")
    print()

    # Non-ABSTAIN votes per row distribution
    n_votes_per_row = np.sum(L_valid != LABEL_ABSTAIN, axis=1)
    print(f"  Non-ABSTAIN votes per row distribution (valid rows):")
    for n_votes in range(7):
        cnt = np.sum(n_votes_per_row == n_votes)
        if cnt > 0:
            print(f"    {n_votes} votes: {cnt:>8,}  ({100*cnt/n_valid:.1f}%)")
    print(f"    Mean: {n_votes_per_row.mean():.2f} votes/row")
    print(f"    Min:  {n_votes_per_row.min()} votes/row")
    print()

    sys.stdout.flush()

    # =====================================================================
    # STEP 2.18 — TRAIN THE SNORKEL LABEL MODEL
    # =====================================================================
    print("=" * 70)
    print("STEP 2.18 — TRAIN THE SNORKEL LABEL MODEL")
    print("=" * 70)
    print()
    print(f"  Initializing LabelModel with:")
    print(f"    cardinality = {SNORKEL_CARDINALITY} (NEGATIVE=0, POSITIVE=1, UNCERTAIN=2)")
    print(f"    n_epochs    = {SNORKEL_EPOCHS}")
    print(f"    lr          = {SNORKEL_LR}")
    print(f"    optimizer   = {SNORKEL_OPTIMIZER}")
    print(f"    seed        = {RANDOM_SEED}")
    print()
    print(f"  CRITICAL: cardinality=3 is MANDATORY for this 3-class pipeline.")
    print(f"  Omitting it would default to cardinality=2 (binary), producing")
    print(f"  incorrect probability distributions and corrupted soft scores.")
    print()
    print(f"  Training runs on CPU, estimated RAM usage: 8-10 GB")
    print(flush=True)

    # Import Snorkel LabelModel (compatibility fix already applied at top)
    from snorkel.labeling.model import LabelModel

    # Initialize with cardinality=3 — this is the single most critical parameter
    label_model = LabelModel(
        cardinality=SNORKEL_CARDINALITY,
        verbose=True,
    )

    print(f"\n  Training started...")
    t_start = time.time()

    # Train the label model
    # The model learns the accuracy and correlation structure of each LF
    # without requiring any manually labeled examples
    label_model.fit(
        L_train=L_valid,
        n_epochs=SNORKEL_EPOCHS,
        lr=SNORKEL_LR,
        optimizer=SNORKEL_OPTIMIZER,
        seed=RANDOM_SEED,
        log_freq=100,  # Print training progress every 100 epochs
    )

    t_train = time.time() - t_start
    print(f"\n  Training completed in {t_train:.1f}s ({t_train/60:.1f} min)")
    print()

    # Display learned LF weights (accuracy estimates)
    print(f"  Learned LF accuracy estimates:")
    try:
        # Snorkel LabelModel stores estimated accuracies
        # Access the model's internal parameters
        lf_summary = label_model.get_weights()
        for i, (name, weight) in enumerate(zip(LF_NAMES, lf_summary)):
            print(f"    {name:25s}  weight = {weight:.4f}")
    except Exception as e:
        print(f"    (Could not extract weights: {e})")
    print()

    sys.stdout.flush()

    # =====================================================================
    # STEP 2.19 — GENERATE SOFT LABEL PROBABILITIES
    # =====================================================================
    print("=" * 70)
    print("STEP 2.19 — GENERATE SOFT LABEL PROBABILITIES")
    print("=" * 70)
    print()

    # Generate probability distributions across all 3 classes for each report
    # predict_proba returns shape (n_reports, 3):
    #   Column 0 = P(NEGATIVE)
    #   Column 1 = P(POSITIVE)
    #   Column 2 = P(UNCERTAIN)
    print(f"  Running predict_proba on label matrix ({n_valid:,} reports)...")
    t_start = time.time()
    proba = label_model.predict_proba(L_valid)
    t_predict = time.time() - t_start

    print(f"  predict_proba completed in {t_predict:.1f}s")
    print(f"  Output shape: {proba.shape}")
    print(f"  Output dtype: {proba.dtype}")
    print()

    # Verify output dimensions
    assert proba.shape == (n_valid, SNORKEL_CARDINALITY), \
        f"predict_proba shape mismatch: {proba.shape} != ({n_valid}, {SNORKEL_CARDINALITY})"

    # Verify probabilities sum to ~1.0 for each row
    row_sums = proba.sum(axis=1)
    sum_ok = np.allclose(row_sums, 1.0, atol=1e-4)
    print(f"  Probability sum verification:")
    print(f"    Mean row sum: {row_sums.mean():.6f}")
    print(f"    Min row sum:  {row_sums.min():.6f}")
    print(f"    Max row sum:  {row_sums.max():.6f}")
    print(f"    All rows sum to ~1.0: {sum_ok}")
    if not sum_ok:
        print(f"    WARNING: Some rows do not sum to 1.0! Check LabelModel output.")
    print()

    # Extract soft_score = P(POSITIVE) = predict_proba[:, 1]
    # This is column index 1 because POSITIVE = 1 in our encoding
    soft_scores = proba[:, 1]

    print(f"  Soft score (P(POSITIVE)) distribution:")
    print(f"    Mean:   {soft_scores.mean():.4f}")
    print(f"    Median: {np.median(soft_scores):.4f}")
    print(f"    Std:    {soft_scores.std():.4f}")
    print(f"    Min:    {soft_scores.min():.4f}")
    print(f"    Max:    {soft_scores.max():.4f}")
    print()

    # Score histogram
    bins = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
    bin_labels = ["0.00-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50",
                  "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]
    hist, _ = np.histogram(soft_scores, bins=bins)
    max_cnt = max(hist) if max(hist) > 0 else 1
    print(f"  Soft score histogram:")
    for lbl, cnt in zip(bin_labels, hist):
        bar = "#" * min(50, int(50 * cnt / max_cnt))
        print(f"    {lbl}: {cnt:>8,}  {bar}")
    print()

    # Class probability breakdown
    p_neg = proba[:, 0]   # P(NEGATIVE)
    p_pos = proba[:, 1]   # P(POSITIVE)
    p_unc = proba[:, 2]   # P(UNCERTAIN)

    print(f"  Per-class probability statistics:")
    for class_name, class_proba in [("P(NEGATIVE)", p_neg), ("P(POSITIVE)", p_pos), ("P(UNCERTAIN)", p_unc)]:
        print(f"    {class_name:15s}  mean={class_proba.mean():.4f}  "
              f"median={np.median(class_proba):.4f}  "
              f"std={class_proba.std():.4f}")
    print()

    # Dominant class assignment (argmax)
    predicted_labels = np.argmax(proba, axis=1)
    pred_counts = Counter(predicted_labels.tolist())
    print(f"  Predicted label distribution (argmax):")
    for code in [LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_UNCERTAIN]:
        cnt = pred_counts.get(code, 0)
        name = LABEL_NAMES[code]
        print(f"    {name:>12s}: {cnt:>8,}  ({100*cnt/n_valid:.1f}%)")
    print()

    sys.stdout.flush()

    # =====================================================================
    # SAVE SNORKEL SOFT SCORES
    # =====================================================================
    print("-" * 70)
    print("SAVING SNORKEL SOFT SCORES")
    print("-" * 70)
    print()

    # Build the Snorkel output DataFrame (valid rows only)
    df_snorkel = pd.DataFrame({
        'subject_id': df_valid['subject_id'].values,
        'study_id': df_valid['study_id'].values,
        'soft_score': soft_scores,
        'p_negative': p_neg,
        'p_positive': p_pos,
        'p_uncertain': p_unc,
        'predicted_label': predicted_labels,
        'label_source': 'snorkel_lm',
    })

    # Save Snorkel-only scores (reports that went through the LF pipeline)
    snorkel_only_csv = os.path.join(DATA_INTERMEDIATE, "snorkel_only_scores.csv")
    df_snorkel.to_csv(snorkel_only_csv, index=False)
    print(f"  Saved Snorkel-only scores: {snorkel_only_csv}")
    print(f"    Rows: {len(df_snorkel):,}")
    print()

    # =====================================================================
    # MERGE WITH PRE-FILTER NEGATIVES AND ALL-ABSTAIN FOR COMPLETE COVERAGE
    # =====================================================================
    print(f"  Merging with pre-filter negatives and all-ABSTAIN rows...")

    # Build list of DataFrames to concatenate
    # Order matters: Snorkel LM first, then all-ABSTAIN, then pre-filter.
    # This ordering is used by drop_duplicates(keep='first') to preserve
    # Snorkel predictions when duplicates exist.
    merge_parts = [df_snorkel]

    # Add all-ABSTAIN rows (assigned NEGATIVE with low soft score)
    # These are reports that passed the pre-filter (have pneumonia/lung terms
    # somewhere in report_text) but all 6 LFs returned ABSTAIN because the
    # terms only appeared in history_text, not in impression_text or findings_text.
    # Clinically: pneumonia terms in history but not in diagnostic sections
    # means the current study does not describe an active pneumonia finding.
    n_abstain_added = 0
    if len(df_all_abstain) > 0:
        df_abstain_out = pd.DataFrame({
            'subject_id': df_all_abstain['subject_id'].values,
            'study_id': df_all_abstain['study_id'].values,
            'soft_score': PREFILTER_NEGATIVE_SCORE,
            'p_negative': 1.0 - PREFILTER_NEGATIVE_SCORE,
            'p_positive': PREFILTER_NEGATIVE_SCORE,
            'p_uncertain': 0.0,
            'predicted_label': LABEL_NEGATIVE,
            'label_source': 'all_abstain',
        })
        merge_parts.append(df_abstain_out)
        n_abstain_added = len(df_abstain_out)
        print(f"    All-ABSTAIN rows added: {n_abstain_added:,}")

    # Add pre-filter negatives (reports with zero pneumonia/lung keywords)
    n_prefilter = 0
    if not os.path.exists(PREFILTER_NEGATIVES_CSV):
        print(f"  WARNING: prefilter_negatives.csv not found at: {PREFILTER_NEGATIVES_CSV}")
        print(f"  Proceeding without pre-filter negatives.")
    else:
        df_prefilter = pd.read_csv(PREFILTER_NEGATIVES_CSV, low_memory=False)
        n_prefilter = len(df_prefilter)
        print(f"    Pre-filter negatives loaded: {n_prefilter:,}")

        # Build pre-filter rows with the same column structure
        df_prefilter_out = pd.DataFrame({
            'subject_id': df_prefilter['subject_id'].values,
            'study_id': df_prefilter['study_id'].values,
            'soft_score': PREFILTER_NEGATIVE_SCORE,
            'p_negative': 1.0 - PREFILTER_NEGATIVE_SCORE,
            'p_positive': PREFILTER_NEGATIVE_SCORE,
            'p_uncertain': 0.0,
            'predicted_label': LABEL_NEGATIVE,
            'label_source': 'pre_filter',
        })
        merge_parts.append(df_prefilter_out)

    # Concatenate ALL parts (Snorkel + all-ABSTAIN + pre-filter)
    df_combined = pd.concat(merge_parts, ignore_index=True)
    n_combined = len(df_combined)

    print()
    print(f"    Combined total: {n_combined:,}")
    print(f"      Snorkel LM:    {len(df_snorkel):,}")
    print(f"      All-ABSTAIN:   {n_abstain_added:,}")
    print(f"      Pre-filter:    {n_prefilter:,}")
    print()

    # Verify no duplicate study_ids
    n_unique_ids = df_combined['study_id'].nunique()
    if n_unique_ids != n_combined:
        n_dups = n_combined - n_unique_ids
        print(f"    WARNING: {n_dups} duplicate study_ids detected!")
        # Deduplicate keeping Snorkel LM predictions over all-ABSTAIN/pre-filter
        df_combined = df_combined.drop_duplicates(subset='study_id', keep='first')
        print(f"    Deduplicated: {len(df_combined):,} rows")
    else:
        print(f"    ✓ No duplicate study_ids. All {n_unique_ids:,} are unique.")
    print()

    # Save combined file
    df_combined.to_csv(SNORKEL_SOFT_SCORES_CSV, index=False)

    file_size_mb = os.path.getsize(SNORKEL_SOFT_SCORES_CSV) / (1024 * 1024)
    df_final = pd.read_csv(SNORKEL_SOFT_SCORES_CSV, nrows=1)
    print(f"  Saved: {SNORKEL_SOFT_SCORES_CSV}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Columns: {list(df_final.columns)}")
    print()

    # =====================================================================
    # FINAL COMBINED SUMMARY
    # =====================================================================
    print("=" * 70)
    print("COMBINED SOFT SCORE SUMMARY")
    print("=" * 70)
    print()

    df_final_full = pd.read_csv(SNORKEL_SOFT_SCORES_CSV, low_memory=False)
    n_final = len(df_final_full)

    # By label source
    source_counts = Counter(df_final_full['label_source'].tolist())
    print(f"  Total reports: {n_final:,}")
    for source, cnt in source_counts.most_common():
        print(f"    {source:20s}: {cnt:>8,}  ({100*cnt/n_final:.1f}%)")
    print()

    # Soft score distribution across all reports
    all_scores = df_final_full['soft_score'].values
    print(f"  Combined soft_score distribution:")
    print(f"    Mean:   {all_scores.mean():.4f}")
    print(f"    Median: {np.median(all_scores):.4f}")
    print(f"    Std:    {all_scores.std():.4f}")
    print()

    # Predicted label distribution
    pred_combined = Counter(df_final_full['predicted_label'].tolist())
    print(f"  Combined predicted label distribution:")
    for code in [LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_UNCERTAIN]:
        cnt = pred_combined.get(code, 0)
        name = LABEL_NAMES[code]
        print(f"    {name:>12s}: {cnt:>8,}  ({100*cnt/n_final:.1f}%)")
    print()

    # High-confidence positive/negative counts
    from config import POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD
    n_high_pos = np.sum(all_scores >= POSITIVE_THRESHOLD)
    n_high_neg = np.sum(all_scores <= NEGATIVE_THRESHOLD)
    n_uncertain_zone = n_final - n_high_pos - n_high_neg
    print(f"  Confidence zone distribution (for Stage 3):")
    print(f"    High confidence POSITIVE (>= {POSITIVE_THRESHOLD}): {n_high_pos:>8,}  ({100*n_high_pos/n_final:.1f}%)")
    print(f"    High confidence NEGATIVE (<= {NEGATIVE_THRESHOLD}): {n_high_neg:>8,}  ({100*n_high_neg/n_final:.1f}%)")
    print(f"    Uncertain zone ({NEGATIVE_THRESHOLD} < x < {POSITIVE_THRESHOLD}):   {n_uncertain_zone:>8,}  ({100*n_uncertain_zone/n_final:.1f}%)")
    print()

    sys.stdout.flush()

    # =====================================================================
    # DONE
    # =====================================================================
    print("=" * 70)
    print("STEPS 2.17-2.19 COMPLETE — SNORKEL LABEL MODEL TRAINED")
    print(f"  Output: {SNORKEL_SOFT_SCORES_CSV}")
    print(f"  Total reports covered: {n_final:,}")
    print("  Next: Stage 3 — Threshold conversion (Confident/Uncertain pool splits)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

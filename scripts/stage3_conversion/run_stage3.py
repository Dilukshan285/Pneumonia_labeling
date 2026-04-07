"""
Stage 3 — Threshold Conversion: Soft Scores → Hard Labels + Pool Splitting

Converts the continuous soft_score (P(POSITIVE)) from the Snorkel LabelModel
into discrete hard labels using configurable confidence thresholds, then splits
the full dataset into two pools:

  1. CONFIDENT POOL — Reports where the Snorkel model has high confidence:
       - soft_score >= POSITIVE_THRESHOLD (0.75) → assigned_label = POSITIVE
       - soft_score <= NEGATIVE_THRESHOLD (0.25) → assigned_label = NEGATIVE
     These labels enter PP1 and PP2 training directly.

  2. UNCERTAIN POOL — Reports in the ambiguous zone:
       - NEGATIVE_THRESHOLD < soft_score < POSITIVE_THRESHOLD
     These reports are excluded from training and queued for Stage 4
     (Active Learning) where the 200 most informative are manually labeled.

Additionally produces a THRESHOLD SENSITIVITY ANALYSIS that evaluates how
different threshold pairs affect pool sizes, class balance, and the width of
the uncertain zone — enabling informed threshold tuning if needed.

Input:   snorkel_soft_scores.csv  (from Steps 2.17-2.19)
Outputs: confident_pool.csv      (high-confidence POSITIVE/NEGATIVE labels)
         uncertain_pool.csv      (ambiguous zone for Stage 4 review)
         threshold_sensitivity.csv (analysis across threshold grid)

Estimated runtime: < 30 seconds on Ryzen 7 8845HS with 32GB RAM
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    SNORKEL_SOFT_SCORES_CSV,
    CONFIDENT_POOL_CSV,
    UNCERTAIN_POOL_CSV,
    THRESHOLD_SENSITIVITY_CSV,
    DATA_INTERMEDIATE,
    LOGS_DIR,
    POSITIVE_THRESHOLD,
    NEGATIVE_THRESHOLD,
    LABEL_POSITIVE,
    LABEL_NEGATIVE,
    LABEL_UNCERTAIN,
    RANDOM_SEED,
)


# ============================================================================
# LABEL NAMES FOR DISPLAY
# ============================================================================
LABEL_NAMES = {
    LABEL_POSITIVE: "POSITIVE",
    LABEL_NEGATIVE: "NEGATIVE",
    LABEL_UNCERTAIN: "UNCERTAIN",
}

# ============================================================================
# THRESHOLD SENSITIVITY GRID
# ============================================================================
# Evaluate a range of threshold pairs to understand the impact of different
# cutoff choices on pool sizes and class balance. This grid covers:
#   - Tight thresholds (0.40/0.60) — aggressive labeling, small uncertain zone
#   - Default thresholds (0.25/0.75) — balanced approach
#   - Wide thresholds (0.10/0.90) — conservative, large uncertain zone
#
# Each pair is (NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD).
# A report is NEGATIVE if soft_score <= neg_thresh, POSITIVE if >= pos_thresh.
SENSITIVITY_GRID = [
    (0.05, 0.95),
    (0.10, 0.90),
    (0.15, 0.85),
    (0.20, 0.80),
    (0.25, 0.75),  # ← default thresholds from config
    (0.30, 0.70),
    (0.35, 0.65),
    (0.40, 0.60),
    (0.45, 0.55),
]


def apply_thresholds(soft_scores, neg_threshold, pos_threshold):
    """
    Convert soft scores to hard labels using the given thresholds.

    Args:
        soft_scores: numpy array of P(POSITIVE) values in [0, 1]
        neg_threshold: scores <= this → NEGATIVE
        pos_threshold: scores >= this → POSITIVE
        Scores between (neg_threshold, pos_threshold) → UNCERTAIN

    Returns:
        numpy array of integer labels (LABEL_NEGATIVE=0, LABEL_POSITIVE=1, LABEL_UNCERTAIN=2)
    """
    labels = np.full(len(soft_scores), LABEL_UNCERTAIN, dtype=int)
    labels[soft_scores >= pos_threshold] = LABEL_POSITIVE
    labels[soft_scores <= neg_threshold] = LABEL_NEGATIVE
    return labels


def run_threshold_sensitivity(soft_scores, n_total):
    """
    Run threshold sensitivity analysis across the predefined grid.

    For each threshold pair, computes:
      - Number and percentage of POSITIVE, NEGATIVE, UNCERTAIN assignments
      - Class balance ratio (POSITIVE / NEGATIVE)
      - Uncertain zone width
      - Confident pool size (POSITIVE + NEGATIVE)

    Returns a DataFrame with one row per threshold pair.
    """
    rows = []
    for neg_thresh, pos_thresh in SENSITIVITY_GRID:
        labels = apply_thresholds(soft_scores, neg_thresh, pos_thresh)
        counts = Counter(labels.tolist())

        n_pos = counts.get(LABEL_POSITIVE, 0)
        n_neg = counts.get(LABEL_NEGATIVE, 0)
        n_unc = counts.get(LABEL_UNCERTAIN, 0)
        n_confident = n_pos + n_neg
        zone_width = pos_thresh - neg_thresh

        # Class balance ratio: POSITIVE / NEGATIVE (avoid division by zero)
        balance_ratio = n_pos / n_neg if n_neg > 0 else float('inf')

        # Mark the default threshold pair
        is_default = (neg_thresh == NEGATIVE_THRESHOLD and pos_thresh == POSITIVE_THRESHOLD)

        rows.append({
            'neg_threshold': neg_thresh,
            'pos_threshold': pos_thresh,
            'zone_width': zone_width,
            'n_positive': n_pos,
            'n_negative': n_neg,
            'n_uncertain': n_unc,
            'n_confident': n_confident,
            'pct_positive': 100 * n_pos / n_total,
            'pct_negative': 100 * n_neg / n_total,
            'pct_uncertain': 100 * n_unc / n_total,
            'pct_confident': 100 * n_confident / n_total,
            'balance_ratio': balance_ratio,
            'is_default': is_default,
        })

    return pd.DataFrame(rows)


def main():
    t_start_total = time.time()

    print("=" * 70)
    print("STAGE 3 — THRESHOLD CONVERSION & POOL SPLITTING")
    print("(Soft Scores → Hard Labels → Confident/Uncertain Pools)")
    print("=" * 70)
    print()
    print(f"  POSITIVE_THRESHOLD:  {POSITIVE_THRESHOLD}  (soft_score >= this → POSITIVE)")
    print(f"  NEGATIVE_THRESHOLD:  {NEGATIVE_THRESHOLD}  (soft_score <= this → NEGATIVE)")
    print(f"  UNCERTAIN_ZONE:      ({NEGATIVE_THRESHOLD}, {POSITIVE_THRESHOLD})")
    print(f"  RANDOM_SEED:         {RANDOM_SEED}")
    print()

    # =====================================================================
    # LOAD SNORKEL SOFT SCORES
    # =====================================================================
    if not os.path.exists(SNORKEL_SOFT_SCORES_CSV):
        print(f"ERROR: snorkel_soft_scores.csv not found at: {SNORKEL_SOFT_SCORES_CSV}")
        print("  Run Steps 2.17-2.19 first.")
        return 1

    print(f"Loading soft scores from {SNORKEL_SOFT_SCORES_CSV}...")
    df = pd.read_csv(SNORKEL_SOFT_SCORES_CSV, low_memory=False)
    n_total = len(df)
    print(f"  Total reports loaded: {n_total:,}")
    print()

    # Verify required columns
    required_cols = ['subject_id', 'study_id', 'soft_score', 'p_negative',
                     'p_positive', 'p_uncertain', 'predicted_label', 'label_source']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return 1
    print(f"  All required columns verified present.")

    # Verify soft_score range
    soft_scores = df['soft_score'].values
    score_min = soft_scores.min()
    score_max = soft_scores.max()
    print(f"  Soft score range: [{score_min:.6f}, {score_max:.6f}]")

    if np.any(np.isnan(soft_scores)):
        n_nan = int(np.isnan(soft_scores).sum())
        print(f"  ERROR: {n_nan} NaN values found in soft_score column!")
        return 1
    print(f"  ✓ No NaN values in soft_score.")

    if score_min < 0.0 or score_max > 1.0:
        print(f"  ERROR: soft_score values outside [0, 1] range!")
        return 1
    print(f"  ✓ All soft_score values in valid [0, 1] range.")
    print()

    # Label source breakdown
    source_counts = Counter(df['label_source'].tolist())
    print(f"  Input by label source:")
    for source, cnt in source_counts.most_common():
        print(f"    {source:20s}: {cnt:>8,}  ({100*cnt/n_total:.1f}%)")
    print()

    sys.stdout.flush()

    # =====================================================================
    # STEP 3.1 — APPLY THRESHOLD CONVERSION
    # =====================================================================
    print("=" * 70)
    print("STEP 3.1 — APPLY THRESHOLD CONVERSION")
    print("=" * 70)
    print()

    # Convert soft scores to hard labels
    assigned_labels = apply_thresholds(soft_scores, NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD)
    df['assigned_label'] = assigned_labels

    # Map integer labels to string names for readability
    df['assigned_label_name'] = df['assigned_label'].map(LABEL_NAMES)

    # Distribution of assigned labels
    label_counts = Counter(assigned_labels.tolist())
    n_pos = label_counts.get(LABEL_POSITIVE, 0)
    n_neg = label_counts.get(LABEL_NEGATIVE, 0)
    n_unc = label_counts.get(LABEL_UNCERTAIN, 0)
    n_confident = n_pos + n_neg

    print(f"  Threshold conversion results:")
    print(f"    POSITIVE (soft_score >= {POSITIVE_THRESHOLD}):        {n_pos:>8,}  ({100*n_pos/n_total:.1f}%)")
    print(f"    NEGATIVE (soft_score <= {NEGATIVE_THRESHOLD}):        {n_neg:>8,}  ({100*n_neg/n_total:.1f}%)")
    print(f"    UNCERTAIN ({NEGATIVE_THRESHOLD} < score < {POSITIVE_THRESHOLD}):  {n_unc:>8,}  ({100*n_unc/n_total:.1f}%)")
    print()
    print(f"    Confident pool total:  {n_confident:>8,}  ({100*n_confident/n_total:.1f}%)")
    print(f"    Uncertain pool total:  {n_unc:>8,}  ({100*n_unc/n_total:.1f}%)")
    print()

    # =====================================================================
    # STEP 3.4 — CHECK CLASS BALANCE
    # =====================================================================
    # Per spec: if majority/minority ratio exceeds 3:1 in either direction,
    # record this imbalance explicitly in a separate note file for use during
    # model training. Imbalance is handled via class-weighted loss or
    # oversampling during training, NOT during the labeling pipeline.
    print(f"  Class balance (confident pool):")
    if n_neg > 0 and n_pos > 0:
        majority_count = max(n_pos, n_neg)
        minority_count = min(n_pos, n_neg)
        majority_class = "NEGATIVE" if n_neg > n_pos else "POSITIVE"
        minority_class = "POSITIVE" if n_neg > n_pos else "NEGATIVE"
        imbalance_ratio = majority_count / minority_count

        print(f"    POSITIVE:  {n_pos:>8,}  ({100*n_pos/n_confident:.1f}% of confident)")
        print(f"    NEGATIVE:  {n_neg:>8,}  ({100*n_neg/n_confident:.1f}% of confident)")
        print(f"    Majority class: {majority_class} ({majority_count:,})")
        print(f"    Minority class: {minority_class} ({minority_count:,})")
        print(f"    Majority/Minority ratio: {imbalance_ratio:.2f}:1")
        print()

        # Save imbalance note file if ratio > 3:1
        if imbalance_ratio > 3.0:
            print(f"    ⚠ CLASS IMBALANCE EXCEEDS 3:1 THRESHOLD")
            print(f"      Ratio: {imbalance_ratio:.2f}:1 ({majority_class}-heavy)")
            print(f"      This is expected for MIMIC-CXR — pneumonia is less prevalent")
            print(f"      than normal findings in the clinical population.")
            print(f"      Imbalance will be handled during model training via:")
            print(f"        - Class-weighted loss functions")
            print(f"        - Minority class oversampling")
            print(f"      NOT corrected during the labeling pipeline.")
            print()

            # Write separate note file per spec
            os.makedirs(LOGS_DIR, exist_ok=True)
            note_path = os.path.join(LOGS_DIR, "class_imbalance_note.txt")
            with open(note_path, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("CLASS IMBALANCE NOTE — STAGE 3 CONFIDENT POOL\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Generated by: Stage 3 — Threshold Conversion\n")
                f.write(f"Thresholds: POSITIVE >= {POSITIVE_THRESHOLD}, NEGATIVE <= {NEGATIVE_THRESHOLD}\n\n")
                f.write(f"Confident Pool Size: {n_confident:,}\n")
                f.write(f"  POSITIVE: {n_pos:,} ({100*n_pos/n_confident:.1f}%)\n")
                f.write(f"  NEGATIVE: {n_neg:,} ({100*n_neg/n_confident:.1f}%)\n\n")
                f.write(f"Majority Class: {majority_class} ({majority_count:,})\n")
                f.write(f"Minority Class: {minority_class} ({minority_count:,})\n")
                f.write(f"Imbalance Ratio: {imbalance_ratio:.2f}:1\n\n")
                f.write("EXPECTED BEHAVIOR:\n")
                f.write("  A NEGATIVE-heavy imbalance is expected in MIMIC-CXR because\n")
                f.write("  pneumonia is less prevalent than normal chest findings in the\n")
                f.write("  clinical population. This realistic imbalance must be preserved\n")
                f.write("  in the label distribution.\n\n")
                f.write("RECOMMENDED HANDLING DURING MODEL TRAINING:\n")
                f.write("  1. Use class-weighted loss functions (e.g., pos_weight in BCEWithLogitsLoss)\n")
                f.write(f"     Suggested pos_weight = {imbalance_ratio:.2f}\n")
                f.write("  2. OR apply minority class oversampling via WeightedRandomSampler\n")
                f.write("  3. Do NOT correct imbalance during the labeling pipeline\n")
            print(f"      Saved note file: {note_path}")
        else:
            print(f"    ✓ Class balance ratio ({imbalance_ratio:.2f}:1) is within 3:1 threshold.")
    elif n_neg == 0:
        print(f"    WARNING: Zero NEGATIVE reports in confident pool!")
    elif n_pos == 0:
        print(f"    WARNING: Zero POSITIVE reports in confident pool!")
    print()

    # Cross-tabulation: assigned_label × label_source
    print(f"  Assigned label × Label source cross-tabulation:")
    ct = pd.crosstab(df['assigned_label_name'], df['label_source'], margins=True)
    # Reorder rows for consistent display
    row_order = [r for r in ['POSITIVE', 'NEGATIVE', 'UNCERTAIN', 'All'] if r in ct.index]
    ct = ct.reindex(row_order)
    print(ct.to_string())
    print()

    sys.stdout.flush()

    # =====================================================================
    # STEP 3.2 — SPLIT INTO CONFIDENT AND UNCERTAIN POOLS
    # =====================================================================
    print("=" * 70)
    print("STEP 3.2 — SPLIT INTO CONFIDENT AND UNCERTAIN POOLS")
    print("=" * 70)
    print()

    # Confident pool: POSITIVE or NEGATIVE (enters PP1/PP2 training)
    mask_confident = df['assigned_label'].isin([LABEL_POSITIVE, LABEL_NEGATIVE])
    df_confident = df[mask_confident].copy()

    # Uncertain pool: UNCERTAIN (queued for Stage 4 active learning)
    mask_uncertain = df['assigned_label'] == LABEL_UNCERTAIN
    df_uncertain = df[mask_uncertain].copy()

    # Integrity check: confident + uncertain must equal total
    assert len(df_confident) + len(df_uncertain) == n_total, \
        f"Pool split integrity failure: {len(df_confident)} + {len(df_uncertain)} != {n_total}"
    print(f"  ✓ Pool split integrity verified: {len(df_confident):,} + {len(df_uncertain):,} = {n_total:,}")
    print()

    # ---- Confident Pool Statistics ----
    print(f"  CONFIDENT POOL ({len(df_confident):,} reports):")
    print(f"    These reports receive hard labels and enter PP1/PP2 training.")
    print()

    # Soft score distribution within confident pool by class
    df_conf_pos = df_confident[df_confident['assigned_label'] == LABEL_POSITIVE]
    df_conf_neg = df_confident[df_confident['assigned_label'] == LABEL_NEGATIVE]

    if len(df_conf_pos) > 0:
        pos_scores = df_conf_pos['soft_score'].values
        print(f"    POSITIVE subset ({len(df_conf_pos):,} reports):")
        print(f"      Soft score range: [{pos_scores.min():.4f}, {pos_scores.max():.4f}]")
        print(f"      Mean: {pos_scores.mean():.4f}  Median: {np.median(pos_scores):.4f}  Std: {pos_scores.std():.4f}")
    if len(df_conf_neg) > 0:
        neg_scores = df_conf_neg['soft_score'].values
        print(f"    NEGATIVE subset ({len(df_conf_neg):,} reports):")
        print(f"      Soft score range: [{neg_scores.min():.4f}, {neg_scores.max():.4f}]")
        print(f"      Mean: {neg_scores.mean():.4f}  Median: {np.median(neg_scores):.4f}  Std: {neg_scores.std():.4f}")
    print()

    # Source breakdown within confident pool
    print(f"    By label source:")
    conf_sources = Counter(df_confident['label_source'].tolist())
    for source, cnt in conf_sources.most_common():
        print(f"      {source:20s}: {cnt:>8,}  ({100*cnt/len(df_confident):.1f}%)")
    print()

    # ---- Uncertain Pool Statistics ----
    print(f"  UNCERTAIN POOL ({len(df_uncertain):,} reports):")
    print(f"    These reports are excluded from training and queued for")
    print(f"    Stage 4 (Active Learning) manual review.")
    print()

    if len(df_uncertain) > 0:
        unc_scores = df_uncertain['soft_score'].values
        print(f"    Soft score range: [{unc_scores.min():.4f}, {unc_scores.max():.4f}]")
        print(f"    Mean: {unc_scores.mean():.4f}  Median: {np.median(unc_scores):.4f}  Std: {unc_scores.std():.4f}")
        print()

        # Uncertainty sub-distribution: how many are close to each boundary?
        near_neg = np.sum(unc_scores < 0.40)
        mid_zone = np.sum((unc_scores >= 0.40) & (unc_scores <= 0.60))
        near_pos = np.sum(unc_scores > 0.60)
        print(f"    Uncertainty sub-distribution:")
        print(f"      Near NEGATIVE boundary ({NEGATIVE_THRESHOLD}-0.40): {near_neg:>6,}  ({100*near_neg/len(df_uncertain):.1f}%)")
        print(f"      Mid-zone (0.40-0.60):                   {mid_zone:>6,}  ({100*mid_zone/len(df_uncertain):.1f}%)")
        print(f"      Near POSITIVE boundary (0.60-{POSITIVE_THRESHOLD}): {near_pos:>6,}  ({100*near_pos/len(df_uncertain):.1f}%)")
        print()

        # Predicted label breakdown in uncertain pool (Snorkel's argmax prediction)
        unc_pred = Counter(df_uncertain['predicted_label'].tolist())
        print(f"    Snorkel argmax predictions within uncertain pool:")
        for code in [LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_UNCERTAIN]:
            cnt = unc_pred.get(code, 0)
            name = LABEL_NAMES.get(code, str(code))
            print(f"      {name:>12s}: {cnt:>6,}  ({100*cnt/len(df_uncertain):.1f}%)")
        print()

        # Source breakdown within uncertain pool
        print(f"    By label source:")
        unc_sources = Counter(df_uncertain['label_source'].tolist())
        for source, cnt in unc_sources.most_common():
            print(f"      {source:20s}: {cnt:>8,}  ({100*cnt/len(df_uncertain):.1f}%)")
    print()

    sys.stdout.flush()

    # =====================================================================
    # STEP 3.3 — SAVE CONFIDENT AND UNCERTAIN POOLS
    # =====================================================================
    print("=" * 70)
    print("STEP 3.3 — SAVE POOL FILES")
    print("=" * 70)
    print()

    # Define columns to save in each pool file
    # Include all probability columns for downstream audit/analysis
    output_columns = [
        'subject_id', 'study_id', 'soft_score',
        'p_negative', 'p_positive', 'p_uncertain',
        'predicted_label', 'label_source',
        'assigned_label', 'assigned_label_name',
    ]

    # Save confident pool
    os.makedirs(os.path.dirname(CONFIDENT_POOL_CSV), exist_ok=True)
    df_confident[output_columns].to_csv(CONFIDENT_POOL_CSV, index=False)
    conf_size_mb = os.path.getsize(CONFIDENT_POOL_CSV) / (1024 * 1024)
    print(f"  Saved: {CONFIDENT_POOL_CSV}")
    print(f"    Rows: {len(df_confident):,}")
    print(f"    Size: {conf_size_mb:.1f} MB")
    print(f"    Columns: {output_columns}")
    print()

    # Save uncertain pool
    os.makedirs(os.path.dirname(UNCERTAIN_POOL_CSV), exist_ok=True)

    # Sort uncertain pool by distance from decision boundary (0.5)
    # Reports closest to 0.5 are the most ambiguous and most valuable for
    # Stage 4 active learning. Sorting here makes it easy to pick the top-N.
    df_uncertain = df_uncertain.copy()
    df_uncertain['boundary_distance'] = np.abs(df_uncertain['soft_score'].values - 0.5)
    df_uncertain = df_uncertain.sort_values('boundary_distance', ascending=True)

    uncertain_output_columns = output_columns + ['boundary_distance']
    df_uncertain[uncertain_output_columns].to_csv(UNCERTAIN_POOL_CSV, index=False)
    unc_size_mb = os.path.getsize(UNCERTAIN_POOL_CSV) / (1024 * 1024)
    print(f"  Saved: {UNCERTAIN_POOL_CSV}")
    print(f"    Rows: {len(df_uncertain):,}")
    print(f"    Size: {unc_size_mb:.1f} MB")
    print(f"    Sorted by: boundary_distance (ascending — most ambiguous first)")
    print(f"    Columns: {uncertain_output_columns}")
    print()

    # Preview most ambiguous reports (top 10)
    if len(df_uncertain) > 0:
        print(f"  Top 10 most ambiguous reports (closest to 0.5 boundary):")
        top10 = df_uncertain.head(10)[['study_id', 'soft_score', 'boundary_distance',
                                       'predicted_label', 'label_source']].copy()
        top10['predicted_label'] = top10['predicted_label'].map(LABEL_NAMES)
        for _, row in top10.iterrows():
            print(f"    {row['study_id']:>15s}  score={row['soft_score']:.4f}  "
                  f"dist={row['boundary_distance']:.4f}  "
                  f"pred={row['predicted_label']:>12s}  src={row['label_source']}")
    print()

    sys.stdout.flush()

    # =====================================================================
    # STEP 3.5 — THRESHOLD SENSITIVITY CHECK
    # =====================================================================
    # Per spec: test two alternative threshold pairs (0.30/0.70 and 0.20/0.80)
    # and compare against the default 0.25/0.75. Check if:
    #   - Default confident pool < half the size of 0.30/0.70 pair → FAIL
    #   - Default confident pool > double the size of 0.20/0.80 pair → FAIL
    # If either condition triggers, thresholds must be reviewed and adjusted.
    # Save all three comparison results as threshold_sensitivity.csv.
    print("=" * 70)
    print("STEP 3.5 — THRESHOLD SENSITIVITY CHECK")
    print("=" * 70)
    print()

    # The three required threshold pairs per spec
    sensitivity_pairs = [
        (0.30, 0.70, "Alternative (wider)"),
        (0.25, 0.75, "Default"),
        (0.20, 0.80, "Alternative (tighter)"),
    ]

    sensitivity_rows = []
    for neg_t, pos_t, pair_name in sensitivity_pairs:
        labels_s = apply_thresholds(soft_scores, neg_t, pos_t)
        counts_s = Counter(labels_s.tolist())
        s_pos = counts_s.get(LABEL_POSITIVE, 0)
        s_neg = counts_s.get(LABEL_NEGATIVE, 0)
        s_unc = counts_s.get(LABEL_UNCERTAIN, 0)
        s_conf = s_pos + s_neg
        bal = s_pos / s_neg if s_neg > 0 else float('inf')

        sensitivity_rows.append({
            'pair_name': pair_name,
            'neg_threshold': neg_t,
            'pos_threshold': pos_t,
            'n_positive': s_pos,
            'n_negative': s_neg,
            'n_uncertain': s_unc,
            'n_confident': s_conf,
            'pct_confident': 100 * s_conf / n_total,
            'pct_uncertain': 100 * s_unc / n_total,
            'balance_ratio': bal,
        })

    df_sensitivity = pd.DataFrame(sensitivity_rows)

    # Display the sensitivity table
    print(f"  {'PAIR NAME':>22s}  {'NEG':>5s}  {'POS':>5s}  "
          f"{'N_POS':>8s}  {'N_NEG':>8s}  {'N_UNC':>8s}  "
          f"{'N_CONF':>8s}  {'%CONF':>6s}  {'%UNC':>6s}  {'BAL':>6s}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*5}  "
          f"{'-'*8}  {'-'*8}  {'-'*8}  "
          f"{'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}")

    for _, row in df_sensitivity.iterrows():
        marker = "  ←" if row['pair_name'] == 'Default' else ""
        bal_str = f"{row['balance_ratio']:.3f}" if row['balance_ratio'] != float('inf') else "inf"
        print(f"  {row['pair_name']:>22s}  {row['neg_threshold']:>5.2f}  {row['pos_threshold']:>5.2f}  "
              f"{row['n_positive']:>8,}  {row['n_negative']:>8,}  {row['n_uncertain']:>8,}  "
              f"{row['n_confident']:>8,}  {row['pct_confident']:>5.1f}%  {row['pct_uncertain']:>5.1f}%  "
              f"{bal_str:>6s}{marker}")
    print()

    # Perform the spec's required comparison checks
    conf_default = df_sensitivity[df_sensitivity['pair_name'] == 'Default'].iloc[0]['n_confident']
    conf_wider = df_sensitivity[df_sensitivity['pair_name'] == 'Alternative (wider)'].iloc[0]['n_confident']
    conf_tighter = df_sensitivity[df_sensitivity['pair_name'] == 'Alternative (tighter)'].iloc[0]['n_confident']

    print(f"  Threshold validation checks (per spec):")
    print(f"    Default confident pool:               {conf_default:>8,}")
    print(f"    Alternative wider (0.30/0.70) pool:    {conf_wider:>8,}")
    print(f"    Alternative tighter (0.20/0.80) pool:  {conf_tighter:>8,}")
    print()

    # Check 1: Default < half of wider?
    half_wider = conf_wider / 2
    check1_pass = conf_default >= half_wider
    print(f"    Check 1: Default ({conf_default:,}) >= half of wider ({half_wider:,.0f})?")
    if check1_pass:
        print(f"      ✓ PASS — Default is {conf_default/conf_wider:.2%} of wider pool.")
    else:
        print(f"      ✗ FAIL — Default confident pool is less than half the size of")
        print(f"        the 0.30/0.70 pair. Thresholds should be reviewed.")
    print()

    # Check 2: Default > double of tighter?
    double_tighter = conf_tighter * 2
    check2_pass = conf_default <= double_tighter
    print(f"    Check 2: Default ({conf_default:,}) <= double of tighter ({double_tighter:,})?")
    if check2_pass:
        print(f"      ✓ PASS — Default is {conf_default/conf_tighter:.2%} of tighter pool.")
    else:
        print(f"      ✗ FAIL — Default confident pool is more than double the size of")
        print(f"        the 0.20/0.80 pair. Thresholds should be reviewed.")
    print()

    if check1_pass and check2_pass:
        print(f"    ✓ Both checks passed. Thresholds {NEGATIVE_THRESHOLD}/{POSITIVE_THRESHOLD} are validated.")
    else:
        print(f"    ⚠ WARNING: One or both checks failed. Review threshold selection.")
    print()

    # Save sensitivity analysis (all three rows for research paper methodology)
    os.makedirs(os.path.dirname(THRESHOLD_SENSITIVITY_CSV), exist_ok=True)
    df_sensitivity.to_csv(THRESHOLD_SENSITIVITY_CSV, index=False)
    sens_size_kb = os.path.getsize(THRESHOLD_SENSITIVITY_CSV) / 1024
    print(f"  Saved: {THRESHOLD_SENSITIVITY_CSV}")
    print(f"    Rows: {len(df_sensitivity)} threshold pairs")
    print(f"    Size: {sens_size_kb:.1f} KB")
    print(f"    (Include this table in research paper methodology section)")
    print()

    sys.stdout.flush()

    # =====================================================================
    # CONFIDENT POOL DETAILED BREAKDOWN
    # =====================================================================
    print("=" * 70)
    print("CONFIDENT POOL DETAILED BREAKDOWN")
    print("=" * 70)
    print()

    # Score histogram for confident pool only (finer granularity)
    print(f"  Confident pool soft_score histogram:")
    conf_scores = df_confident['soft_score'].values
    bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
    bin_labels = [
        "0.00-0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20-0.25",
        "0.75-0.80", "0.80-0.85", "0.85-0.90", "0.90-0.95", "0.95-1.00",
    ]
    hist, _ = np.histogram(conf_scores, bins=bins)
    max_cnt = max(hist) if max(hist) > 0 else 1
    for lbl, cnt in zip(bin_labels, hist):
        bar = "█" * min(40, int(40 * cnt / max_cnt))
        print(f"    {lbl}: {cnt:>8,}  {bar}")
    print()

    # Reports at the boundary edges (within 0.02 of thresholds)
    edge_margin = 0.02
    near_neg_edge = np.sum((conf_scores >= NEGATIVE_THRESHOLD - edge_margin) &
                           (conf_scores <= NEGATIVE_THRESHOLD))
    near_pos_edge = np.sum((conf_scores >= POSITIVE_THRESHOLD) &
                           (conf_scores <= POSITIVE_THRESHOLD + edge_margin))
    print(f"  Boundary edge analysis (within {edge_margin} of thresholds):")
    print(f"    Near NEGATIVE edge ({NEGATIVE_THRESHOLD-edge_margin:.2f}-{NEGATIVE_THRESHOLD:.2f}): {near_neg_edge:,} reports")
    print(f"    Near POSITIVE edge ({POSITIVE_THRESHOLD:.2f}-{POSITIVE_THRESHOLD+edge_margin:.2f}): {near_pos_edge:,} reports")
    print(f"    These are the most fragile confident labels — small threshold")
    print(f"    changes would reclassify them to the uncertain pool.")
    print()

    sys.stdout.flush()

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    t_total = time.time() - t_start_total

    print("=" * 70)
    print("STAGE 3 COMPLETE — THRESHOLD CONVERSION & POOL SPLITTING")
    print("=" * 70)
    print()
    print(f"  Total reports processed: {n_total:,}")
    print()
    print(f"  CONFIDENT POOL:  {len(df_confident):,}  ({100*len(df_confident)/n_total:.1f}%)")
    print(f"    POSITIVE:      {n_pos:,}  ({100*n_pos/n_total:.1f}%)")
    print(f"    NEGATIVE:      {n_neg:,}  ({100*n_neg/n_total:.1f}%)")
    print(f"  UNCERTAIN POOL:  {len(df_uncertain):,}  ({100*len(df_uncertain)/n_total:.1f}%)")
    print()
    print(f"  Outputs:")
    print(f"    {CONFIDENT_POOL_CSV}")
    print(f"    {UNCERTAIN_POOL_CSV}")
    print(f"    {THRESHOLD_SENSITIVITY_CSV}")
    print()
    print(f"  Runtime: {t_total:.1f}s")
    print()
    print(f"  Next: Stage 4 — Active Learning")
    print(f"    → Select top {200} most informative reports from uncertain pool")
    print(f"    → Manual expert labeling of those reports")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

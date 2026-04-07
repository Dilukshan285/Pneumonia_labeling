"""
Stage 5 — Steps 5.3 & 5.4: Calculate Cohen's Kappa and Record Results

After completing manual labeling of the 300 validation reports (Step 5.2),
this script:
  1. Loads manual_validation_labels.csv (study_id + manual_label)
  2. Joins with final_pneumonia_labels.csv to get pipeline labels
  3. Computes Cohen's Kappa:
     a. Full 3-class (POSITIVE/NEGATIVE/UNCERTAIN) — as specified
     b. Binary (POSITIVE/NEGATIVE only, excluding manual UNCERTAIN) — practical
  4. Records label distribution, source breakdown, and Kappa scores
  5. Saves cohen_kappa_validation.csv — evidence of label quality

Interpretation:
  Kappa >= 0.80  → Strong agreement, research-grade labels
  0.60 <= Kappa < 0.80  → Moderate agreement, review thresholds
  Kappa < 0.60  → Pipeline needs structural adjustment

The binary Kappa is the primary metric because UNCERTAIN labels are excluded
from model training. The 3-class Kappa is reported for completeness.

Input:   manual_validation_labels.csv  (from Step 5.2)
         final_pneumonia_labels.csv    (from Stage 4)
Output:  cohen_kappa_validation.csv    (Kappa score + statistics)

Estimated runtime: < 15 seconds
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MANUAL_VALIDATION_CSV,
    FINAL_LABELS_CSV,
    COHEN_KAPPA_CSV,
    DATA_OUTPUT,
    LABEL_POSITIVE,
    LABEL_NEGATIVE,
    LABEL_UNCERTAIN,
)


# String to integer encoding for manual labels
LABEL_ENCODING = {
    "POSITIVE": LABEL_POSITIVE,     # 1
    "NEGATIVE": LABEL_NEGATIVE,     # 0
    "UNCERTAIN": LABEL_UNCERTAIN,   # 2
}

LABEL_NAMES = {
    LABEL_POSITIVE: "POSITIVE",
    LABEL_NEGATIVE: "NEGATIVE",
    LABEL_UNCERTAIN: "UNCERTAIN",
}


def interpret_kappa(kappa):
    """Return interpretation string for a Kappa score."""
    if kappa >= 0.80:
        return "STRONG AGREEMENT - Research-grade labels confirmed"
    elif kappa >= 0.60:
        return "MODERATE AGREEMENT - Review threshold settings in Stage 3"
    else:
        return "INSUFFICIENT AGREEMENT - Pipeline needs structural adjustment"


def print_confusion_matrix(y_true, y_pred, labels, label_names):
    """Pretty-print a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = "              " + "  ".join(f"{l:>10s}" for l in label_names)
    print(f"    {header}")
    for i, row_label in enumerate(label_names):
        row_vals = "  ".join(f"{v:>10d}" for v in cm[i])
        print(f"    {row_label:>12s}  {row_vals}")
    return cm


def main():
    t_start = time.time()

    print("=" * 70)
    print("STAGE 5 -- STEPS 5.3 & 5.4: COHEN'S KAPPA VALIDATION")
    print("(Manual Labels vs Pipeline Labels -- Inter-Annotator Agreement)")
    print("=" * 70)
    print()

    # =====================================================================
    # LOAD MANUAL VALIDATION LABELS
    # =====================================================================
    if not os.path.exists(MANUAL_VALIDATION_CSV):
        print(f"ERROR: {MANUAL_VALIDATION_CSV} not found.")
        print("  Complete Step 5.2 (manual labeling of 300 reports) first.")
        return 1

    print(f"Loading manual validation labels from {MANUAL_VALIDATION_CSV}...")
    df_manual = pd.read_csv(MANUAL_VALIDATION_CSV, low_memory=False)
    df_manual['study_id'] = df_manual['study_id'].astype(str)
    df_manual['manual_label'] = df_manual['manual_label'].astype(str).str.strip().str.upper()
    n_manual = len(df_manual)
    print(f"  Manual labels loaded: {n_manual:,}")
    print()

    # Validate manual labels
    valid_labels = set(LABEL_ENCODING.keys())
    invalid_mask = ~df_manual['manual_label'].isin(valid_labels)
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        invalid_vals = df_manual[invalid_mask]['manual_label'].unique().tolist()
        print(f"  WARNING: {n_invalid} invalid manual labels: {invalid_vals}")
        print(f"  These will be excluded from Kappa calculation.")
        df_manual = df_manual[~invalid_mask].copy()
        n_manual = len(df_manual)
        print(f"  Valid labels remaining: {n_manual}")
    print()

    # Manual label distribution
    manual_counts = Counter(df_manual['manual_label'].tolist())
    print(f"  Manual label distribution ({n_manual} reports):")
    for lbl in ['POSITIVE', 'NEGATIVE', 'UNCERTAIN']:
        cnt = manual_counts.get(lbl, 0)
        print(f"    {lbl:>12s}: {cnt:>4}  ({100*cnt/n_manual:.1f}%)")
    print()

    # Convert manual labels to integer encoding
    df_manual['manual_label_int'] = df_manual['manual_label'].map(LABEL_ENCODING)

    # =====================================================================
    # LOAD PIPELINE LABELS (from final_pneumonia_labels.csv)
    # =====================================================================
    if not os.path.exists(FINAL_LABELS_CSV):
        print(f"ERROR: {FINAL_LABELS_CSV} not found.")
        print("  Complete Stage 4 first.")
        return 1

    print(f"Loading pipeline labels from {FINAL_LABELS_CSV}...")
    df_pipeline = pd.read_csv(FINAL_LABELS_CSV, low_memory=False)
    df_pipeline['study_id'] = df_pipeline['study_id'].astype(str)
    n_pipeline = len(df_pipeline)
    print(f"  Pipeline labels loaded: {n_pipeline:,}")
    print()

    # =====================================================================
    # JOIN MANUAL AND PIPELINE LABELS ON study_id
    # =====================================================================
    print(f"  Joining manual and pipeline labels on study_id...")
    df_merged = df_manual.merge(
        df_pipeline[['study_id', 'label', 'soft_score', 'label_source']],
        on='study_id',
        how='inner',
        suffixes=('_manual', '_pipeline'),
    )
    n_matched = len(df_merged)
    n_missing = n_manual - n_matched
    print(f"  Matched: {n_matched:,} / {n_manual:,}")
    if n_missing > 0:
        print(f"  WARNING: {n_missing} study_ids not found in final_pneumonia_labels.csv")
    print()

    if n_matched == 0:
        print("  ERROR: No matching study_ids found. Cannot compute Kappa.")
        return 1

    # Pipeline label distribution for matched reports
    pipeline_counts = Counter(df_merged['label'].tolist())
    print(f"  Pipeline label distribution (matched {n_matched} reports):")
    for code in [LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN]:
        cnt = pipeline_counts.get(code, 0)
        name = LABEL_NAMES.get(code, f"CODE_{code}")
        print(f"    {name:>12s}: {cnt:>4}  ({100*cnt/n_matched:.1f}%)")
    print()

    # Pipeline label source breakdown
    source_counts = Counter(df_merged['label_source'].tolist())
    print(f"  Pipeline label sources (matched reports):")
    for source, cnt in source_counts.most_common():
        print(f"    {source:>20s}: {cnt:>4}  ({100*cnt/n_matched:.1f}%)")
    print()

    sys.stdout.flush()

    # =====================================================================
    # STEP 5.3A -- FULL 3-CLASS COHEN'S KAPPA
    # =====================================================================
    print("=" * 70)
    print("STEP 5.3A -- COHEN'S KAPPA (FULL 3-CLASS)")
    print("=" * 70)
    print()

    y_manual_full = df_merged['manual_label_int'].values
    y_pipeline_full = df_merged['label'].values

    all_labels = sorted(set(y_manual_full.tolist() + y_pipeline_full.tolist()))
    label_str = [LABEL_NAMES.get(l, f"CODE_{l}") for l in all_labels]

    kappa_full = cohen_kappa_score(y_pipeline_full, y_manual_full, labels=all_labels)

    print(f"  Labels in comparison: {label_str}")
    print(f"  Cohen's Kappa (3-class): {kappa_full:.4f}")
    print(f"  Interpretation: {interpret_kappa(kappa_full)}")
    print()

    n_agree_full = int((y_manual_full == y_pipeline_full).sum())
    agree_pct_full = 100 * n_agree_full / n_matched
    print(f"  Agreement: {n_agree_full}/{n_matched} ({agree_pct_full:.1f}%)")
    print()

    print(f"  Confusion Matrix (rows=Pipeline, cols=Manual):")
    print_confusion_matrix(y_pipeline_full, y_manual_full, all_labels, label_str)
    print()

    sys.stdout.flush()

    # =====================================================================
    # STEP 5.3B -- BINARY COHEN'S KAPPA (PRIMARY METRIC)
    # =====================================================================
    print("=" * 70)
    print("STEP 5.3B -- COHEN'S KAPPA (BINARY: POSITIVE vs NEGATIVE)")
    print("  This is the PRIMARY metric because UNCERTAIN labels are")
    print("  excluded from model training. Only POSITIVE/NEGATIVE matter.")
    print("=" * 70)
    print()

    # Filter to only reports where manual label is POSITIVE or NEGATIVE
    binary_mask = df_merged['manual_label_int'].isin([LABEL_POSITIVE, LABEL_NEGATIVE])
    df_binary = df_merged[binary_mask].copy()
    n_binary = len(df_binary)
    n_excluded = n_matched - n_binary

    print(f"  Reports with binary manual labels: {n_binary}")
    print(f"  Reports excluded (manual=UNCERTAIN): {n_excluded}")
    print()

    if n_binary > 0:
        y_manual_bin = df_binary['manual_label_int'].values
        y_pipeline_bin = df_binary['label'].values

        binary_labels = [LABEL_NEGATIVE, LABEL_POSITIVE]
        binary_names = ['NEGATIVE', 'POSITIVE']

        kappa_binary = cohen_kappa_score(y_pipeline_bin, y_manual_bin, labels=binary_labels)

        print(f"  Cohen's Kappa (binary): {kappa_binary:.4f}")
        print(f"  Interpretation: {interpret_kappa(kappa_binary)}")
        print()

        n_agree_bin = int((y_manual_bin == y_pipeline_bin).sum())
        agree_pct_bin = 100 * n_agree_bin / n_binary
        print(f"  Agreement: {n_agree_bin}/{n_binary} ({agree_pct_bin:.1f}%)")
        print()

        print(f"  Confusion Matrix (rows=Pipeline, cols=Manual):")
        print_confusion_matrix(y_pipeline_bin, y_manual_bin, binary_labels, binary_names)
        print()

        # Detailed classification report
        print(f"  Classification Report:")
        report = classification_report(
            y_pipeline_bin, y_manual_bin,
            labels=binary_labels,
            target_names=binary_names,
            zero_division=0,
        )
        for line in report.split('\n'):
            print(f"    {line}")
        print()
    else:
        kappa_binary = float('nan')
        n_agree_bin = 0
        agree_pct_bin = 0.0
        print("  ERROR: No binary labels to compare.")
        print()

    sys.stdout.flush()

    # =====================================================================
    # DISAGREEMENT ANALYSIS
    # =====================================================================
    print("=" * 70)
    print("DISAGREEMENT ANALYSIS")
    print("=" * 70)
    print()

    disagree_mask = y_manual_full != y_pipeline_full
    n_disagree = int(disagree_mask.sum())
    df_disagree = df_merged[disagree_mask].copy()

    if n_disagree > 0:
        # Categorize disagreements
        types = []
        for _, row in df_disagree.iterrows():
            pl = LABEL_NAMES.get(int(row['label']), '?')
            ml = row['manual_label']
            types.append(f"{pl} -> {ml}")

        type_counts = Counter(types)
        print(f"  Total disagreements: {n_disagree}")
        print(f"  By type:")
        for t, cnt in type_counts.most_common():
            print(f"    {t:>30s}: {cnt:>4}  ({100*cnt/n_disagree:.1f}%)")
        print()

        # Show first few disagreements
        print(f"  Sample disagreements (first 15):")
        for idx, (_, row) in enumerate(df_disagree.head(15).iterrows()):
            pl = LABEL_NAMES.get(int(row['label']), '?')
            ml = row['manual_label']
            print(f"    {row['study_id']}  pipeline={pl:>10s}  manual={ml:>10s}  "
                  f"soft_score={row['soft_score']:.4f}  source={row['label_source']}")
        if n_disagree > 15:
            print(f"    ... and {n_disagree - 15} more")
    else:
        print(f"  No disagreements! Perfect agreement.")
    print()

    sys.stdout.flush()

    # =====================================================================
    # STEP 5.4 -- RECORD AND REPORT KAPPA SCORE
    # =====================================================================
    print("=" * 70)
    print("STEP 5.4 -- RECORD AND REPORT KAPPA SCORE")
    print("=" * 70)
    print()

    # Gather all statistics for the final labels dataset
    print(f"Loading full final_pneumonia_labels.csv for distribution statistics...")
    df_full = pd.read_csv(FINAL_LABELS_CSV, low_memory=False)
    n_full = len(df_full)

    # Class distribution in final labels
    full_label_counts = Counter(df_full['label'].tolist())
    n_pos = full_label_counts.get(LABEL_POSITIVE, 0)
    n_neg = full_label_counts.get(LABEL_NEGATIVE, 0)
    n_unc = full_label_counts.get(LABEL_UNCERTAIN, 0)

    print(f"  Final label set: {n_full:,} reports")
    print(f"    POSITIVE:  {n_pos:>8,}  ({100*n_pos/n_full:.1f}%)")
    print(f"    NEGATIVE:  {n_neg:>8,}  ({100*n_neg/n_full:.1f}%)")
    if n_unc > 0:
        print(f"    UNCERTAIN: {n_unc:>8,}  ({100*n_unc/n_full:.1f}%)")
    print()

    # Label source breakdown
    full_source_counts = Counter(df_full['label_source'].tolist())
    print(f"  By label source:")
    for source, cnt in full_source_counts.most_common():
        print(f"    {source:>20s}: {cnt:>8,}  ({100*cnt/n_full:.1f}%)")
    print()

    # =====================================================================
    # BUILD AND SAVE COHEN'S KAPPA VALIDATION CSV
    # =====================================================================
    rows = []

    # Primary metric
    rows.append({
        'Metric': "Cohen's Kappa (Binary - PRIMARY)",
        'Value': f'{kappa_binary:.4f}' if not np.isnan(kappa_binary) else 'N/A',
        'Details': interpret_kappa(kappa_binary) if not np.isnan(kappa_binary) else 'No binary labels',
    })

    # Full 3-class metric
    rows.append({
        'Metric': "Cohen's Kappa (3-Class)",
        'Value': f'{kappa_full:.4f}',
        'Details': interpret_kappa(kappa_full),
    })

    # Binary comparison details
    rows.append({
        'Metric': 'Binary Reports Compared',
        'Value': str(n_binary),
        'Details': f'{n_agree_bin} agreed ({agree_pct_bin:.1f}%), {n_binary - n_agree_bin} disagreed',
    })

    # Full comparison details
    rows.append({
        'Metric': 'Full Reports Compared',
        'Value': str(n_matched),
        'Details': f'{n_agree_full} agreed ({agree_pct_full:.1f}%), {n_matched - n_agree_full} disagreed',
    })

    # Manual labels excluded as UNCERTAIN
    rows.append({
        'Metric': 'Manual UNCERTAIN (excluded from binary)',
        'Value': str(n_excluded),
        'Details': f'{100*n_excluded/n_matched:.1f}% of validation sample',
    })

    # Final label distribution
    rows.append({
        'Metric': 'Final Labels - POSITIVE',
        'Value': str(n_pos),
        'Details': f'{100*n_pos/n_full:.1f}% of total',
    })
    rows.append({
        'Metric': 'Final Labels - NEGATIVE',
        'Value': str(n_neg),
        'Details': f'{100*n_neg/n_full:.1f}% of total',
    })
    if n_unc > 0:
        rows.append({
            'Metric': 'Final Labels - UNCERTAIN',
            'Value': str(n_unc),
            'Details': f'{100*n_unc/n_full:.1f}% of total (excluded from training)',
        })

    rows.append({
        'Metric': 'Total Final Labels',
        'Value': str(n_full),
        'Details': 'POSITIVE + NEGATIVE only in training set',
    })

    # Label source breakdown
    for source, cnt in full_source_counts.most_common():
        rows.append({
            'Metric': f'Label Source - {source}',
            'Value': str(cnt),
            'Details': f'{100*cnt/n_full:.1f}% of total',
        })

    # Validation manual distribution
    for lbl in ['POSITIVE', 'NEGATIVE', 'UNCERTAIN']:
        cnt = manual_counts.get(lbl, 0)
        rows.append({
            'Metric': f'Validation Manual - {lbl}',
            'Value': str(cnt),
            'Details': f'{100*cnt/n_manual:.1f}% of {n_manual} validation reports',
        })

    # Class ratio
    if n_neg > 0:
        ratio_pn = n_pos / n_neg
        rows.append({
            'Metric': 'Class Ratio (POS/NEG)',
            'Value': f'{ratio_pn:.4f}',
            'Details': f'NEG/POS = {1/ratio_pn:.2f}:1',
        })

    df_summary = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(COHEN_KAPPA_CSV), exist_ok=True)
    df_summary.to_csv(COHEN_KAPPA_CSV, index=False)
    file_size_kb = os.path.getsize(COHEN_KAPPA_CSV) / 1024

    print(f"  Saved: {COHEN_KAPPA_CSV}")
    print(f"    Rows:    {len(df_summary)}")
    print(f"    Size:    {file_size_kb:.1f} KB")
    print()

    # Print the full summary table
    print(f"  COHEN'S KAPPA VALIDATION SUMMARY")
    print(f"  " + "-" * 66)
    for _, row in df_summary.iterrows():
        metric = str(row['Metric'])[:40]
        value = str(row['Value'])[:12]
        print(f"  | {metric:<40s} | {value:>12s} |")
    print(f"  " + "-" * 66)
    print()

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    t_total = time.time() - t_start

    print("=" * 70)
    print("STAGE 5 COMPLETE -- VALIDATION RESULTS")
    print("=" * 70)
    print()
    kappa_primary = kappa_binary if not np.isnan(kappa_binary) else kappa_full
    print(f"  PRIMARY Cohen's Kappa (binary):  {kappa_binary:.4f}")
    print(f"  Full Cohen's Kappa (3-class):    {kappa_full:.4f}")
    print(f"  Interpretation:        {interpret_kappa(kappa_primary)}")
    print()
    print(f"  Binary comparison:     {n_binary} reports ({n_agree_bin} agreed, {agree_pct_bin:.1f}%)")
    print(f"  Full comparison:       {n_matched} reports ({n_agree_full} agreed, {agree_pct_full:.1f}%)")
    print()
    print(f"  Total final labels:    {n_full:,}")
    print(f"  POSITIVE:              {n_pos:,} ({100*n_pos/n_full:.1f}%)")
    print(f"  NEGATIVE:              {n_neg:,} ({100*n_neg/n_full:.1f}%)")
    print()
    print(f"  Output: {COHEN_KAPPA_CSV}")
    print(f"  Runtime: {t_total:.1f}s")
    print()

    if kappa_primary >= 0.80:
        print(f"  PASS -- PIPELINE VALIDATED. Proceed to Stage 6 (Image Linking)")
    elif kappa_primary >= 0.60:
        print(f"  CAUTION -- MODERATE AGREEMENT. Consider reviewing Stage 3 thresholds")
        print(f"    before proceeding to Stage 6.")
    else:
        print(f"  FAIL -- INSUFFICIENT AGREEMENT. Do NOT proceed to model training.")
        print(f"    Review pipeline structure and re-run validation.")
    print()
    print(f"  FOR RESEARCH PAPER METHODOLOGY SECTION:")
    print(f"    - Cohen's Kappa (binary): {kappa_binary:.4f}")
    print(f"    - Cohen's Kappa (3-class): {kappa_full:.4f}")
    print(f"    - Validation sample size: {n_matched} (from confident pool)")
    print(f"    - Binary comparison size: {n_binary} (excluding UNCERTAIN manual labels)")
    print(f"    - Pipeline class distribution: {n_pos} POS ({100*n_pos/n_full:.1f}%), "
          f"{n_neg} NEG ({100*n_neg/n_full:.1f}%)")
    print(f"    - Label sources: {dict(full_source_counts)}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

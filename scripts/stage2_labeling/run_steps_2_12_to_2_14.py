"""
Stage 2 — Run and Validate Steps 2.12 through 2.14
(Labeling Function 5: NLI Zero-Shot Classification)

Executes:
  Step 2.12 — Load facebook/bart-large-mnli on RTX 4060 GPU
  Step 2.13 — Run zero-shot classification on all report texts
  Step 2.14 — Convert NLI scores to Snorkel labels with 0.40 threshold

This script:
  1. Loads LF1-LF4 results from lf1_lf2_lf3_lf4_results.csv
  2. Runs the full LF5 NLI pipeline (load -> classify -> convert -> unload)
  3. Prints detailed statistics, cross-tabulations, and agreement analysis
  4. Saves intermediate results with LF1 through LF5 columns

Resume support:
  If the script is interrupted, simply re-run it. It will automatically
  resume from the last checkpoint. Checkpoints are saved every ~4000 texts.

Estimated runtime on RTX 4060 8GB with batch_size=8:
  ~163K reports x ~30ms per report = ~1.5 hours
"""

import os
import sys
import warnings
import logging

# Suppress HuggingFace/transformers warnings that PowerShell treats as errors
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import pandas as pd
from collections import Counter
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_INTERMEDIATE,
    NLI_MODEL_NAME,
    NLI_BATCH_SIZE,
    NLI_CONFIDENCE_THRESHOLD,
    LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN, LABEL_ABSTAIN,
)
from stage2_labeling.keywords import KEYWORD_LIST_VERSION
from stage2_labeling.lf5_nli import run_lf5_full


LABEL_NAMES = {
    LABEL_POSITIVE: "POSITIVE",
    LABEL_NEGATIVE: "NEGATIVE",
    LABEL_UNCERTAIN: "UNCERTAIN",
    LABEL_ABSTAIN: "ABSTAIN",
}

# Input: LF1-LF4 results (saved by run_steps_2_8_to_2_11.py)
LF1_TO_LF4_RESULTS_CSV = os.path.join(DATA_INTERMEDIATE, "lf1_lf2_lf3_lf4_results.csv")

# Output: LF1-LF5 combined results
LF1_TO_LF5_RESULTS_CSV = os.path.join(DATA_INTERMEDIATE, "lf1_to_lf5_results.csv")


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


def _print_cross_tab(df, col1, col2, name1, name2):
    """Print a cross-tabulation of two label columns."""
    ct = pd.crosstab(
        df[col1].map(lambda x: LABEL_NAMES.get(x, str(x))),
        df[col2].map(lambda x: LABEL_NAMES.get(x, str(x))),
        margins=True,
    )
    order = ["POSITIVE", "NEGATIVE", "UNCERTAIN", "ABSTAIN", "All"]
    rows = [r for r in order if r in ct.index]
    cols = [c for c in order if c in ct.columns]
    ct = ct.loc[rows, cols]
    print(f"  Cross-tabulation: {name1} (rows) x {name2} (cols)")
    print()
    print(ct.to_string())
    print()


def main():
    print("=" * 70)
    print(f"STAGE 2 -- STEPS 2.12 THROUGH 2.14 EXECUTION")
    print("(LF5: NLI Zero-Shot Classification -- facebook/bart-large-mnli)")
    print("=" * 70)
    print()
    print(f"  KEYWORD_LIST_VERSION:    {KEYWORD_LIST_VERSION}")
    print(f"  NLI_MODEL:               {NLI_MODEL_NAME}")
    print(f"  NLI_BATCH_SIZE:          {NLI_BATCH_SIZE}")
    print(f"  NLI_CONFIDENCE_THRESH:   {NLI_CONFIDENCE_THRESHOLD}")
    print(flush=True)

    # =====================================================================
    # LOAD INPUT DATA
    # =====================================================================
    if not os.path.exists(LF1_TO_LF4_RESULTS_CSV):
        print(f"ERROR: lf1_lf2_lf3_lf4_results.csv not found at: {LF1_TO_LF4_RESULTS_CSV}")
        print("  Run run_steps_2_8_to_2_11.py first.")
        return 1

    print(f"Loading LF1-LF4 results from {LF1_TO_LF4_RESULTS_CSV}...")
    df = pd.read_csv(LF1_TO_LF4_RESULTS_CSV, low_memory=False)

    for col in ['impression_text', 'findings_text', 'history_text', 'report_text']:
        if col in df.columns:
            df[col] = df[col].fillna('')

    n_total = len(df)
    print(f"  Reports loaded: {n_total:,}")
    print(flush=True)

    # Show LF1-LF4 baseline
    print("-" * 70)
    print("BASELINE -- LF1 through LF4 (from previous steps)")
    print("-" * 70)
    for lf_col, lf_name in [
        ('lf1_label', 'LF1 (Keywords)'),
        ('lf2_label', 'LF2 (NegEx)'),
        ('lf3_label', 'LF3 (CheXpert)'),
        ('lf4_label', 'LF4 (Section Weight)'),
    ]:
        _print_distribution(df[lf_col].tolist(), n_total, lf_name)

    sys.stdout.flush()

    # =====================================================================
    # STEPS 2.12-2.14 — RUN FULL LF5 PIPELINE
    # =====================================================================
    print("=" * 70)
    print("STEPS 2.12-2.14 -- LF5 NLI ZERO-SHOT CLASSIFICATION")
    print("=" * 70)
    print(flush=True)

    t_overall_start = time.time()
    lf5_labels, lf5_scores, lf5_top_labels = run_lf5_full(df, resume=True)
    t_overall = time.time() - t_overall_start

    df['lf5_label'] = lf5_labels
    df['lf5_score'] = lf5_scores
    df['lf5_top_label'] = lf5_top_labels

    print(f"  Total LF5 pipeline time: {t_overall:.1f}s ({t_overall/60:.1f} min, "
          f"{t_overall/3600:.1f} hours)")
    print(flush=True)

    # =====================================================================
    # LF5 RESULTS
    # =====================================================================
    print("-" * 70)
    print("LF5 RESULTS SUMMARY")
    print("-" * 70)
    _print_distribution(lf5_labels, n_total, "LF5 (NLI Zero-Shot)")

    # Score distribution
    import numpy as np
    scores_arr = np.array(lf5_scores)
    non_zero_scores = scores_arr[scores_arr > 0]
    if len(non_zero_scores) > 0:
        print(f"  NLI confidence score distribution (non-zero):")
        print(f"    Mean:   {non_zero_scores.mean():.3f}")
        print(f"    Median: {np.median(non_zero_scores):.3f}")
        print(f"    Min:    {non_zero_scores.min():.3f}")
        print(f"    Max:    {non_zero_scores.max():.3f}")
        print(f"    Std:    {non_zero_scores.std():.3f}")
        print()

        # Score histogram bins
        bins = [0.0, 0.20, 0.40, 0.60, 0.80, 1.01]
        bin_labels_text = ["0.00-0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", "0.80-1.00"]
        hist, _ = np.histogram(non_zero_scores, bins=bins)
        print(f"  Score distribution bins:")
        for bl, cnt in zip(bin_labels_text, hist):
            bar = "#" * min(50, int(50 * cnt / max(hist)))
            print(f"    {bl}: {cnt:>8,}  {bar}")
        print()

    # Low-confidence ABSTAIN analysis
    lf5_counts = Counter(lf5_labels)
    n_abstain = lf5_counts.get(LABEL_ABSTAIN, 0)
    low_conf_mask = (df['lf5_label'] == LABEL_ABSTAIN) & (df['lf5_score'] > 0)
    n_low_conf = low_conf_mask.sum()
    n_empty = n_abstain - n_low_conf
    print(f"  ABSTAIN breakdown:")
    print(f"    Empty text (no input):        {n_empty:>8,}")
    print(f"    Low confidence (< {NLI_CONFIDENCE_THRESHOLD}):     {n_low_conf:>8,}")
    print(f"    Total ABSTAIN:                {n_abstain:>8,}")
    print()

    # Top NLI label distribution
    top_label_counts = Counter(lf5_top_labels)
    print(f"  Top candidate label distribution:")
    for label, cnt in top_label_counts.most_common():
        print(f"    {label:30s}  {cnt:>8,}")
    print()

    sys.stdout.flush()

    # =====================================================================
    # CROSS-TABULATIONS
    # =====================================================================
    print("-" * 70)
    print("CROSS-TABULATION: LF1 x LF5")
    print("-" * 70)
    _print_cross_tab(df, 'lf1_label', 'lf5_label', 'LF1 (Keywords)', 'LF5 (NLI)')

    # Agreement: LF1 vs LF5
    mask_both = (df['lf1_label'] != LABEL_ABSTAIN) & (df['lf5_label'] != LABEL_ABSTAIN)
    n_both = mask_both.sum()
    if n_both > 0:
        n_agree = (df.loc[mask_both, 'lf1_label'] == df.loc[mask_both, 'lf5_label']).sum()
        print(f"  LF1 vs LF5 agreement (where both vote):")
        print(f"    Both voted:  {n_both:>8,}")
        print(f"    Agree:       {n_agree:>8,}  ({100*n_agree/n_both:.1f}%)")
        print(f"    Disagree:    {n_both - n_agree:>8,}  ({100*(n_both - n_agree)/n_both:.1f}%)")
        print()

    # Agreement: LF3 (CheXpert) vs LF5 (NLI)
    print("-" * 70)
    print("CROSS-TABULATION: LF3 x LF5")
    print("-" * 70)
    _print_cross_tab(df, 'lf3_label', 'lf5_label', 'LF3 (CheXpert)', 'LF5 (NLI)')

    mask_both = (df['lf3_label'] != LABEL_ABSTAIN) & (df['lf5_label'] != LABEL_ABSTAIN)
    n_both = mask_both.sum()
    if n_both > 0:
        n_agree = (df.loc[mask_both, 'lf3_label'] == df.loc[mask_both, 'lf5_label']).sum()
        print(f"  LF3 vs LF5 agreement (where both vote):")
        print(f"    Both voted:  {n_both:>8,}")
        print(f"    Agree:       {n_agree:>8,}  ({100*n_agree/n_both:.1f}%)")
        print(f"    Disagree:    {n_both - n_agree:>8,}  ({100*(n_both - n_agree)/n_both:.1f}%)")
        print()

    sys.stdout.flush()

    # =====================================================================
    # SAMPLE OUTPUTS
    # =====================================================================
    for label_code, label_name in [
        (LABEL_POSITIVE, "POSITIVE"),
        (LABEL_NEGATIVE, "NEGATIVE"),
        (LABEL_UNCERTAIN, "UNCERTAIN"),
    ]:
        subset = df[df['lf5_label'] == label_code]
        if len(subset) > 0:
            print(f"  SAMPLE -- LF5 = {label_name} (3 examples):")
            print(f"  {'-'*66}")
            samples = subset.sample(min(3, len(subset)), random_state=42)
            for _, row in samples.iterrows():
                imp = str(row['impression_text'])[:150]
                print(f"    study_id: {row['study_id']}")
                print(f"    LF5: {label_name} (score={row['lf5_score']:.3f}, "
                      f"top={row['lf5_top_label']})")
                print(f"    IMPRESSION: \"{imp}\"")
                print(f"    LF1={LABEL_NAMES.get(row['lf1_label'],'?')}, "
                      f"LF3={LABEL_NAMES.get(row['lf3_label'],'?')}, "
                      f"LF4={LABEL_NAMES.get(row['lf4_label'],'?')}")
                print()

    # =====================================================================
    # 5-LF SUMMARY
    # =====================================================================
    print("=" * 70)
    print("5-LF SUMMARY -- Current Label Matrix Status")
    print("=" * 70)
    print()

    for lf_col, lf_name in [
        ('lf1_label', 'LF1 (Keywords v2.5)'),
        ('lf2_label', 'LF2 (NegEx Clinical)'),
        ('lf3_label', 'LF3 (CheXpert Ref)'),
        ('lf4_label', 'LF4 (Section Weight)'),
        ('lf5_label', 'LF5 (NLI Zero-Shot)'),
    ]:
        counts = Counter(df[lf_col])
        coverage = n_total - counts.get(LABEL_ABSTAIN, 0)
        pos = counts.get(LABEL_POSITIVE, 0)
        neg = counts.get(LABEL_NEGATIVE, 0)
        unc = counts.get(LABEL_UNCERTAIN, 0)
        print(f"  {lf_name:30s}  POS={pos:>7,}  NEG={neg:>7,}  "
              f"UNC={unc:>7,}  COV={coverage:>7,} ({100*coverage/n_total:.1f}%)")

    print()

    # Unanimous agreement (all 5 non-ABSTAIN LFs agree)
    lf_cols = ['lf1_label', 'lf2_label', 'lf3_label', 'lf4_label', 'lf5_label']
    non_abstain_mask = pd.Series([True] * n_total, index=df.index)
    for col in lf_cols:
        non_abstain_mask = non_abstain_mask & (df[col] != LABEL_ABSTAIN)

    n_all_vote = non_abstain_mask.sum()
    if n_all_vote > 0:
        all_agree_mask = non_abstain_mask.copy()
        for i in range(1, len(lf_cols)):
            all_agree_mask = all_agree_mask & (df[lf_cols[0]] == df[lf_cols[i]])
        n_all_agree = all_agree_mask.sum()
        print(f"  Reports where ALL 5 LFs vote:    {n_all_vote:>8,}")
        print(f"  Reports where ALL 5 LFs agree:   {n_all_agree:>8,}  "
              f"({100*n_all_agree/n_all_vote:.1f}%)")
        print()

    sys.stdout.flush()

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    print("-" * 70)
    print("SAVING INTERMEDIATE RESULTS")
    print("-" * 70)
    print()

    os.makedirs(DATA_INTERMEDIATE, exist_ok=True)

    save_cols = ['subject_id', 'study_id', 'report_text',
                 'impression_text', 'findings_text', 'history_text',
                 'lf1_label', 'lf1_matched_keyword', 'lf1_match_stage',
                 'lf2_label', 'lf3_label', 'lf4_label',
                 'lf5_label', 'lf5_score', 'lf5_top_label']
    df_save = df[save_cols].copy()
    df_save.to_csv(LF1_TO_LF5_RESULTS_CSV, index=False)

    file_size_mb = os.path.getsize(LF1_TO_LF5_RESULTS_CSV) / (1024 * 1024)
    print(f"  Saved: {LF1_TO_LF5_RESULTS_CSV}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Rows: {len(df_save):,}")
    print()

    # =====================================================================
    # DONE
    # =====================================================================
    print("=" * 70)
    print("STEPS 2.12-2.14 COMPLETE")
    print(f"  Output: {LF1_TO_LF5_RESULTS_CSV}")
    print("  Next: LF6 (Pattern Heuristics), then Snorkel LabelModel")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Stage 2 — Run and Validate Steps 2.8 through 2.11
(Labeling Functions 3 & 4: CheXpert Reference + Section Weight Priority)

Executes:
  Step 2.8  — Load original CheXpert labels
  Step 2.9  — Assign LF3 labels (CheXpert as one vote among six)
  Step 2.10 — Apply section weight logic for LF4
  Step 2.11 — Assign LF4 labels (section-weighted priority vote)

This script:
  1. Loads LF1+LF2 results from lf1_lf2_results.csv
  2. Loads CheXpert labels and merges as LF3
  3. Applies LF4 section weight priority classification
  4. Prints detailed statistics, cross-tabulations, and sample outputs
  5. Saves intermediate results with LF1, LF2, LF3, and LF4 columns
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
from collections import Counter
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_INTERMEDIATE,
    LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN, LABEL_ABSTAIN,
)
from stage2_labeling.keywords import KEYWORD_LIST_VERSION
from stage2_labeling.lf3_chexpert import load_chexpert_labels, merge_lf3_labels
from stage2_labeling.lf4_section_weight import lf4_section_weight, lf4_section_weight_debug


LABEL_NAMES = {
    LABEL_POSITIVE: "POSITIVE",
    LABEL_NEGATIVE: "NEGATIVE",
    LABEL_UNCERTAIN: "UNCERTAIN",
    LABEL_ABSTAIN: "ABSTAIN",
}

# Input: LF1+LF2 results (saved by run_steps_2_5_to_2_7.py)
LF1_LF2_RESULTS_CSV = os.path.join(DATA_INTERMEDIATE, "lf1_lf2_results.csv")

# Output: LF1+LF2+LF3+LF4 combined results
LF1_TO_LF4_RESULTS_CSV = os.path.join(DATA_INTERMEDIATE, "lf1_lf2_lf3_lf4_results.csv")


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
    # Reorder rows/cols for consistent display
    order = ["POSITIVE", "NEGATIVE", "UNCERTAIN", "ABSTAIN", "All"]
    rows = [r for r in order if r in ct.index]
    cols = [c for c in order if c in ct.columns]
    ct = ct.loc[rows, cols]
    print(f"  Cross-tabulation: {name1} (rows) × {name2} (cols)")
    print()
    print(ct.to_string())
    print()


def main():
    print("=" * 70)
    print(f"STAGE 2 — STEPS 2.8 THROUGH 2.11 EXECUTION ({KEYWORD_LIST_VERSION})")
    print("(LF3: CheXpert Reference + LF4: Section Weight Priority)")
    print("=" * 70)
    print()
    print(f"  KEYWORD_LIST_VERSION: {KEYWORD_LIST_VERSION}")
    print()

    # =====================================================================
    # LOAD INPUT DATA
    # =====================================================================
    if not os.path.exists(LF1_LF2_RESULTS_CSV):
        print(f"ERROR: lf1_lf2_results.csv not found at: {LF1_LF2_RESULTS_CSV}")
        print("  Run run_steps_2_5_to_2_7.py first.")
        return 1

    print(f"Loading LF1+LF2 results from {LF1_LF2_RESULTS_CSV}...")
    df = pd.read_csv(LF1_LF2_RESULTS_CSV, low_memory=False)

    for col in ['impression_text', 'findings_text', 'history_text', 'report_text']:
        df[col] = df[col].fillna('')

    n_total = len(df)
    print(f"  Reports loaded: {n_total:,}")
    print()

    # Show LF1+LF2 baseline
    print("-" * 70)
    print("BASELINE — LF1 & LF2 (from previous steps)")
    print("-" * 70)
    _print_distribution(df['lf1_label'].tolist(), n_total, "LF1 (Keywords)")
    _print_distribution(df['lf2_label'].tolist(), n_total, "LF2 (NegEx)")

    # =====================================================================
    # STEP 2.8 — LOAD CHEXPERT LABELS
    # =====================================================================
    print("-" * 70)
    print("STEP 2.8 — LOADING ORIGINAL CHEXPERT LABELS")
    print("-" * 70)
    print()

    t_start = time.time()
    try:
        df_cx = load_chexpert_labels()
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return 1
    t_load = time.time() - t_start

    n_cx = len(df_cx)
    print(f"  Loaded {n_cx:,} unique CheXpert studies in {t_load:.1f}s")
    print()

    # CheXpert label distribution
    cx_counts = Counter(df_cx['lf3_label'])
    print(f"  CheXpert Pneumonia label distribution:")
    for code in [LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN, LABEL_ABSTAIN]:
        cnt = cx_counts.get(code, 0)
        name = LABEL_NAMES[code]
        print(f"    {name:>12s}:  {cnt:>8,}  ({100*cnt/n_cx:.1f}%)")
    print()

    # =====================================================================
    # STEP 2.9 — MERGE LF3 LABELS
    # =====================================================================
    print("-" * 70)
    print("STEP 2.9 — MERGING LF3 LABELS INTO PIPELINE")
    print("-" * 70)
    print()

    lf3_labels = merge_lf3_labels(df, df_cx)
    df['lf3_label'] = lf3_labels

    n_matched = (df['lf3_label'] != LABEL_ABSTAIN).sum()
    n_unmatched = (df['lf3_label'] == LABEL_ABSTAIN).sum()
    print(f"  Reports matched to CheXpert:      {n_matched:>8,}  ({100*n_matched/n_total:.1f}%)")
    print(f"  Reports NOT in CheXpert (ABSTAIN): {n_unmatched:>8,}  ({100*n_unmatched/n_total:.1f}%)")
    print()

    _print_distribution(df['lf3_label'].tolist(), n_total, "LF3 (CheXpert Reference)")

    # Cross-tabulation: LF1 × LF3
    print("-" * 70)
    print("CROSS-TABULATION: LF1 × LF3")
    print("-" * 70)
    _print_cross_tab(df, 'lf1_label', 'lf3_label', 'LF1 (Keywords)', 'LF3 (CheXpert)')

    # Agreement analysis: LF1 vs LF3 (where both are non-ABSTAIN)
    mask_both = (df['lf1_label'] != LABEL_ABSTAIN) & (df['lf3_label'] != LABEL_ABSTAIN)
    n_both = mask_both.sum()
    if n_both > 0:
        n_agree = (df.loc[mask_both, 'lf1_label'] == df.loc[mask_both, 'lf3_label']).sum()
        n_disagree = n_both - n_agree
        print(f"  LF1 vs LF3 agreement (where both vote):")
        print(f"    Both voted:    {n_both:>8,}")
        print(f"    Agree:         {n_agree:>8,}  ({100*n_agree/n_both:.1f}%)")
        print(f"    Disagree:      {n_disagree:>8,}  ({100*n_disagree/n_both:.1f}%)")
        print()

        # Breakdown of disagreements
        if n_disagree > 0:
            disagree_mask = mask_both & (df['lf1_label'] != df['lf3_label'])
            disagree_df = df[disagree_mask]

            print(f"  Disagreement breakdown:")
            # Count specific disagreement patterns
            combo_counts = Counter()
            for _, row in disagree_df[['lf1_label', 'lf3_label']].iterrows():
                lf1_name = LABEL_NAMES.get(row['lf1_label'], '?')
                lf3_name = LABEL_NAMES.get(row['lf3_label'], '?')
                combo_counts[f"LF1={lf1_name} vs LF3={lf3_name}"] += 1

            for combo, cnt in combo_counts.most_common(10):
                print(f"    {combo:45s}  {cnt:>6,}")
            print()

    # =====================================================================
    # STEPS 2.10-2.11 — LF4 SECTION WEIGHT PRIORITY
    # =====================================================================
    print("-" * 70)
    print("STEPS 2.10-2.11 — APPLYING LF4 (SECTION WEIGHT PRIORITY)")
    print("-" * 70)
    print()
    print(f"  Processing {n_total:,} reports with section weight logic...")
    print(f"  Priority: IMPRESSION (weight=3) > FINDINGS (weight=2)")
    print(f"  If impression conflicts with findings → ALWAYS use impression")
    print()

    lf4_labels = []
    lf4_debug_infos = []
    n_conflicts = 0

    t_start = time.time()
    for idx, row in tqdm(df.iterrows(), total=n_total,
                         desc="  LF4 section weight", unit="reports"):
        label, debug_info = lf4_section_weight_debug(row)
        lf4_labels.append(label)
        lf4_debug_infos.append(debug_info)
        if debug_info.get('conflict'):
            n_conflicts += 1

    t_elapsed = time.time() - t_start
    df['lf4_label'] = lf4_labels

    print()
    print(f"  LF4 processing time: {t_elapsed:.1f}s ({t_elapsed/60:.1f} min)")
    print()

    # LF4 results
    _print_distribution(df['lf4_label'].tolist(), n_total, "LF4 (Section Weight Priority)")

    # Section conflict analysis
    print("-" * 70)
    print("SECTION CONFLICT ANALYSIS")
    print("-" * 70)
    print()
    print(f"  Total section conflicts (impression ≠ findings): {n_conflicts:,}")
    if n_conflicts > 0:
        print(f"    → All resolved by prioritizing IMPRESSION section")
        print()

        # Show conflict examples
        conflict_indices = [i for i, info in enumerate(lf4_debug_infos) if info.get('conflict')]
        import random
        random.seed(42)
        sample_indices = random.sample(conflict_indices, min(5, len(conflict_indices)))

        print(f"  Sample section conflicts (up to 5):")
        print(f"  {'='*68}")
        for ci in sample_indices:
            info = lf4_debug_infos[ci]
            row = df.iloc[ci]
            print(f"    study_id: {row['study_id']}")
            print(f"    Impression label: {info['impression_label']}")
            print(f"    Findings label:   {info['findings_label']}")
            print(f"    Final (LF4):      {info['final_label']} (IMPRESSION wins)")
            print(f"    Impression text:  \"{info['impression_text_truncated'][:120]}\"")
            print(f"    Findings text:    \"{info['findings_text_truncated'][:120]}\"")
            print()

    # Source section breakdown
    source_counts = Counter(info.get('source_section') for info in lf4_debug_infos)
    print(f"  LF4 source section breakdown:")
    for src in ["IMPRESSION", "FINDINGS", None]:
        cnt = source_counts.get(src, 0)
        display = src if src else "NONE (ABSTAIN)"
        print(f"    {display:>20s}:  {cnt:>8,}  ({100*cnt/n_total:.1f}%)")
    print()

    # Cross-tabulation: LF1 × LF4
    print("-" * 70)
    print("CROSS-TABULATION: LF1 × LF4")
    print("-" * 70)
    _print_cross_tab(df, 'lf1_label', 'lf4_label', 'LF1 (Keywords)', 'LF4 (Section Weight)')

    # Agreement: LF1 vs LF4
    mask_both = (df['lf1_label'] != LABEL_ABSTAIN) & (df['lf4_label'] != LABEL_ABSTAIN)
    n_both = mask_both.sum()
    if n_both > 0:
        n_agree = (df.loc[mask_both, 'lf1_label'] == df.loc[mask_both, 'lf4_label']).sum()
        print(f"  LF1 vs LF4 agreement (where both vote):")
        print(f"    Both voted:  {n_both:>8,}")
        print(f"    Agree:       {n_agree:>8,}  ({100*n_agree/n_both:.1f}%)")
        print(f"    Disagree:    {n_both - n_agree:>8,}  ({100*(n_both - n_agree)/n_both:.1f}%)")
        print()

    # =====================================================================
    # OVERALL SUMMARY — All 4 LFs so far
    # =====================================================================
    print("=" * 70)
    print("4-LF SUMMARY — Current Label Matrix Status")
    print("=" * 70)
    print()

    for lf_col, lf_name in [
        ('lf1_label', 'LF1 (Keywords v2.5)'),
        ('lf2_label', 'LF2 (NegEx Clinical)'),
        ('lf3_label', 'LF3 (CheXpert Ref)'),
        ('lf4_label', 'LF4 (Section Weight)'),
    ]:
        counts = Counter(df[lf_col])
        coverage = n_total - counts.get(LABEL_ABSTAIN, 0)
        pos = counts.get(LABEL_POSITIVE, 0)
        neg = counts.get(LABEL_NEGATIVE, 0)
        unc = counts.get(LABEL_UNCERTAIN, 0)
        print(f"  {lf_name:30s}  POS={pos:>7,}  NEG={neg:>7,}  "
              f"UNC={unc:>7,}  COV={coverage:>7,} ({100*coverage/n_total:.1f}%)")

    print()

    # Unanimous agreement analysis (all non-ABSTAIN LFs agree)
    lf_cols = ['lf1_label', 'lf2_label', 'lf3_label', 'lf4_label']
    non_abstain_mask = True
    for col in lf_cols:
        non_abstain_mask = non_abstain_mask & (df[col] != LABEL_ABSTAIN)

    n_all_vote = non_abstain_mask.sum()
    if n_all_vote > 0:
        all_agree_mask = non_abstain_mask.copy()
        for i in range(1, len(lf_cols)):
            all_agree_mask = all_agree_mask & (df[lf_cols[0]] == df[lf_cols[i]])
        n_all_agree = all_agree_mask.sum()
        print(f"  Reports where ALL 4 LFs vote:    {n_all_vote:>8,}")
        print(f"  Reports where ALL 4 LFs agree:   {n_all_agree:>8,}  ({100*n_all_agree/n_all_vote:.1f}%)")
        print()

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
                 'lf2_label', 'lf3_label', 'lf4_label']
    df_save = df[save_cols].copy()
    df_save.to_csv(LF1_TO_LF4_RESULTS_CSV, index=False)

    file_size_mb = os.path.getsize(LF1_TO_LF4_RESULTS_CSV) / (1024 * 1024)
    print(f"  Saved: {LF1_TO_LF4_RESULTS_CSV}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Rows: {len(df_save):,}")
    print(f"  Columns: {save_cols}")
    print()

    # =====================================================================
    # DONE
    # =====================================================================
    print("=" * 70)
    print("STEPS 2.8-2.11 COMPLETE")
    print(f"  Output: {LF1_TO_LF4_RESULTS_CSV}")
    print("  Next: LF5 (NLI Zero-Shot), LF6 (Pattern Heuristics), "
          "then Snorkel LabelModel")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Stage 2 — Run and Validate Steps 2.15 through 2.16
(Labeling Function 6: Uncertainty Phrase Detector)

Executes:
  Step 2.15 — Build uncertainty detector with hedging phrase patterns
  Step 2.16 — Apply detector: UNCERTAIN if any phrase matches, else ABSTAIN

This script:
  1. Loads LF1-LF5 results from lf1_to_lf5_results.csv
  2. Runs LF6 uncertainty phrase detection on all reports
  3. Prints detailed statistics, cross-tabulations, and agreement analysis
  4. Saves final 6-LF intermediate results for the Snorkel LabelModel

Estimated runtime: ~30 seconds (regex matching only, no GPU required)
"""

import os
import sys
import time

import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_INTERMEDIATE,
    LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN, LABEL_ABSTAIN,
)
from stage2_labeling.keywords import KEYWORD_LIST_VERSION
from stage2_labeling.lf6_uncertainty import (
    lf6_uncertainty_debug,
    UNCERTAINTY_PHRASES,
    _UNCERTAINTY_PHRASE_PATTERN,
)


LABEL_NAMES = {
    LABEL_POSITIVE: "POSITIVE",
    LABEL_NEGATIVE: "NEGATIVE",
    LABEL_UNCERTAIN: "UNCERTAIN",
    LABEL_ABSTAIN: "ABSTAIN",
}

# Input: LF1-LF5 results (saved by run_steps_2_12_to_2_14.py)
LF1_TO_LF5_RESULTS_CSV = os.path.join(DATA_INTERMEDIATE, "lf1_to_lf5_results.csv")

# Output: LF1-LF6 combined results (final label matrix for Snorkel)
LF1_TO_LF6_RESULTS_CSV = os.path.join(DATA_INTERMEDIATE, "lf1_to_lf6_results.csv")


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
    print(f"STAGE 2 -- STEPS 2.15 THROUGH 2.16 EXECUTION")
    print("(LF6: Uncertainty Phrase Detector)")
    print("=" * 70)
    print()
    print(f"  KEYWORD_LIST_VERSION:       {KEYWORD_LIST_VERSION}")
    print(f"  UNCERTAINTY_PHRASES count:  {len(UNCERTAINTY_PHRASES)}")
    print(f"  LF6 output classes:         UNCERTAIN or ABSTAIN only")
    print(flush=True)

    # =====================================================================
    # LOAD INPUT DATA
    # =====================================================================
    if not os.path.exists(LF1_TO_LF5_RESULTS_CSV):
        print(f"ERROR: lf1_to_lf5_results.csv not found at: {LF1_TO_LF5_RESULTS_CSV}")
        print("  Run run_steps_2_12_to_2_14.py first.")
        return 1

    print(f"Loading LF1-LF5 results from {LF1_TO_LF5_RESULTS_CSV}...")
    df = pd.read_csv(LF1_TO_LF5_RESULTS_CSV, low_memory=False)

    for col in ['impression_text', 'findings_text', 'history_text', 'report_text']:
        if col in df.columns:
            df[col] = df[col].fillna('')

    n_total = len(df)
    print(f"  Reports loaded: {n_total:,}")
    print(flush=True)

    # Show LF1-LF5 baseline
    print("-" * 70)
    print("BASELINE -- LF1 through LF5 (from previous steps)")
    print("-" * 70)
    for lf_col, lf_name in [
        ('lf1_label', 'LF1 (Keywords)'),
        ('lf2_label', 'LF2 (NegEx)'),
        ('lf3_label', 'LF3 (CheXpert)'),
        ('lf4_label', 'LF4 (Section Weight)'),
        ('lf5_label', 'LF5 (NLI Zero-Shot)'),
    ]:
        if lf_col in df.columns:
            _print_distribution(df[lf_col].tolist(), n_total, lf_name)

    sys.stdout.flush()

    # =====================================================================
    # STEPS 2.15-2.16 — RUN LF6 UNCERTAINTY PHRASE DETECTOR
    # =====================================================================
    print("=" * 70)
    print("STEPS 2.15-2.16 -- LF6 UNCERTAINTY PHRASE DETECTOR")
    print("=" * 70)
    print()
    print("  Step 2.15 — Uncertainty detector built (compiled regex patterns)")
    print(f"    Phrases loaded: {len(UNCERTAINTY_PHRASES)}")
    print()
    print("  Step 2.16 — Applying uncertainty phrase detection to all reports...")
    print(flush=True)

    t_start = time.time()

    # --- VECTORIZED LF6 APPLICATION ---
    # Step 1: Build target_text column (impression first, findings fallback)
    impression = df['impression_text'].fillna('').astype(str).str.strip()
    findings = df['findings_text'].fillna('').astype(str).str.strip()
    target_text = impression.where(impression != '', findings)

    # Step 2: Vectorized regex match using pandas .str.contains()
    # This is ~50-100x faster than iterrows() on ~163K rows
    has_uncertainty = target_text.str.contains(
        _UNCERTAINTY_PHRASE_PATTERN, na=False
    )

    # Step 3: Assign labels vectorized
    df['lf6_label'] = np.where(
        target_text == '',
        LABEL_ABSTAIN,
        np.where(has_uncertainty, LABEL_UNCERTAIN, LABEL_ABSTAIN)
    )

    # Step 4: Extract matched phrase and source section for analysis
    # Use .str.extract() for the matched phrase (vectorized regex extraction)
    # Build a capturing-group version of the pattern for extraction
    extracted = target_text.str.extract(
        '(' + _UNCERTAINTY_PHRASE_PATTERN.pattern[3:-1] + ')',  # unwrap (?:...)
        flags=_UNCERTAINTY_PHRASE_PATTERN.flags,
    )
    df['lf6_matched_phrase'] = extracted[0].fillna('')

    # Source section: IMPRESSION if impression non-empty, else FINDINGS
    df['lf6_source_section'] = np.where(
        df['lf6_label'] == LABEL_ABSTAIN,
        '',
        np.where(impression != '', 'IMPRESSION', 'FINDINGS')
    )

    t_elapsed = time.time() - t_start

    print(f"    Detection completed in {t_elapsed:.1f}s")
    print(f"    Average per report: {1000*t_elapsed/n_total:.4f}ms")
    print(flush=True)

    # =====================================================================
    # LF6 RESULTS SUMMARY
    # =====================================================================
    print("-" * 70)
    print("LF6 RESULTS SUMMARY")
    print("-" * 70)
    _print_distribution(df['lf6_label'].tolist(), n_total, "LF6 (Uncertainty Phrases)")

    # Phrase frequency breakdown
    lf6_counts = Counter(df['lf6_label'].tolist())
    n_uncertain = lf6_counts.get(LABEL_UNCERTAIN, 0)
    n_abstain = lf6_counts.get(LABEL_ABSTAIN, 0)

    print(f"  LF6 outcome:")
    print(f"    UNCERTAIN (hedge detected):  {n_uncertain:>8,}  ({100*n_uncertain/n_total:.1f}%)")
    print(f"    ABSTAIN   (no hedge):        {n_abstain:>8,}  ({100*n_abstain/n_total:.1f}%)")
    print()

    # Source section breakdown
    section_counts = Counter(df['lf6_source_section'].tolist())
    if n_uncertain > 0:
        print(f"  Source section for UNCERTAIN detections:")
        for sec in ["IMPRESSION", "FINDINGS"]:
            cnt = section_counts.get(sec, 0)
            print(f"    {sec:>12s}:  {cnt:>8,}  ({100*cnt/n_uncertain:.1f}%)")
        print()

    # Matched phrase frequency
    phrase_counts = Counter(p for p in df['lf6_matched_phrase'].tolist() if p)
    if phrase_counts:
        print(f"  Top matched uncertainty phrases:")
        for phrase, cnt in phrase_counts.most_common(20):
            print(f"    {phrase:40s}  {cnt:>8,}")
        print()

    sys.stdout.flush()

    # =====================================================================
    # CROSS-TABULATIONS
    # =====================================================================

    # LF1 x LF6: How does LF6 interact with keyword-based labels?
    print("-" * 70)
    print("CROSS-TABULATION: LF1 x LF6")
    print("-" * 70)
    _print_cross_tab(df, 'lf1_label', 'lf6_label', 'LF1 (Keywords)', 'LF6 (Uncertainty)')

    # Key insight: LF6 UNCERTAIN on reports where LF1 said POSITIVE
    # These are cases where LF1 found a positive keyword but LF6 detects
    # hedging language — the Snorkel model will weigh these competing signals.
    lf1_pos_lf6_unc = ((df['lf1_label'] == LABEL_POSITIVE) &
                       (df['lf6_label'] == LABEL_UNCERTAIN)).sum()
    lf1_neg_lf6_unc = ((df['lf1_label'] == LABEL_NEGATIVE) &
                       (df['lf6_label'] == LABEL_UNCERTAIN)).sum()
    lf1_unc_lf6_unc = ((df['lf1_label'] == LABEL_UNCERTAIN) &
                       (df['lf6_label'] == LABEL_UNCERTAIN)).sum()
    lf1_abs_lf6_unc = ((df['lf1_label'] == LABEL_ABSTAIN) &
                       (df['lf6_label'] == LABEL_UNCERTAIN)).sum()

    print(f"  LF6 UNCERTAIN breakdown by LF1 label:")
    print(f"    LF1=POSITIVE  + LF6=UNCERTAIN:  {lf1_pos_lf6_unc:>8,}  (hedge overrides positive)")
    print(f"    LF1=NEGATIVE  + LF6=UNCERTAIN:  {lf1_neg_lf6_unc:>8,}  (redundant uncertainty)")
    print(f"    LF1=UNCERTAIN + LF6=UNCERTAIN:  {lf1_unc_lf6_unc:>8,}  (reinforcing uncertainty)")
    print(f"    LF1=ABSTAIN   + LF6=UNCERTAIN:  {lf1_abs_lf6_unc:>8,}  (new uncertainty signal)")
    print()

    # LF5 (NLI) x LF6
    print("-" * 70)
    print("CROSS-TABULATION: LF5 x LF6")
    print("-" * 70)
    _print_cross_tab(df, 'lf5_label', 'lf6_label', 'LF5 (NLI)', 'LF6 (Uncertainty)')

    sys.stdout.flush()

    # =====================================================================
    # SAMPLE OUTPUTS
    # =====================================================================
    # Show examples where LF6=UNCERTAIN and LF1=POSITIVE (most interesting)
    conflict_mask = (df['lf6_label'] == LABEL_UNCERTAIN) & (df['lf1_label'] == LABEL_POSITIVE)
    conflict_df = df[conflict_mask]

    if len(conflict_df) > 0:
        print("-" * 70)
        print(f"SAMPLE — LF1=POSITIVE but LF6=UNCERTAIN ({len(conflict_df):,} reports)")
        print("-" * 70)
        samples = conflict_df.sample(min(5, len(conflict_df)), random_state=42)
        for _, row in samples.iterrows():
            imp = str(row['impression_text'])[:200]
            print(f"  study_id: {row['study_id']}")
            print(f"    LF6 phrase: \"{row['lf6_matched_phrase']}\" (in {row['lf6_source_section']})")
            print(f"    IMPRESSION: \"{imp}\"")
            print(f"    LF1={LABEL_NAMES.get(row['lf1_label'],'?')}, "
                  f"LF2={LABEL_NAMES.get(row['lf2_label'],'?')}, "
                  f"LF3={LABEL_NAMES.get(row['lf3_label'],'?')}, "
                  f"LF4={LABEL_NAMES.get(row['lf4_label'],'?')}, "
                  f"LF5={LABEL_NAMES.get(row['lf5_label'],'?')}")
            print()

    # Show examples where LF6=UNCERTAIN and all other LFs agreed POSITIVE
    if 'lf5_label' in df.columns:
        all_pos_lf6_unc = (
            (df['lf1_label'] == LABEL_POSITIVE) &
            (df['lf4_label'] == LABEL_POSITIVE) &
            (df['lf5_label'] == LABEL_POSITIVE) &
            (df['lf6_label'] == LABEL_UNCERTAIN)
        )
        n_all_pos_unc = all_pos_lf6_unc.sum()
        if n_all_pos_unc > 0:
            print(f"  NOTE: {n_all_pos_unc:,} reports where LF1+LF4+LF5 all say POSITIVE "
                  f"but LF6 says UNCERTAIN.")
            print(f"  The Snorkel LabelModel will weigh LF6's uncertainty signal against")
            print(f"  the majority POSITIVE votes from the other functions.")
            print()

    sys.stdout.flush()

    # =====================================================================
    # 6-LF COMPLETE SUMMARY
    # =====================================================================
    print("=" * 70)
    print("6-LF COMPLETE SUMMARY — Full Label Matrix Status")
    print("=" * 70)
    print()

    for lf_col, lf_name in [
        ('lf1_label', 'LF1 (Keywords v2.5)'),
        ('lf2_label', 'LF2 (NegEx Clinical)'),
        ('lf3_label', 'LF3 (CheXpert Ref)'),
        ('lf4_label', 'LF4 (Section Weight)'),
        ('lf5_label', 'LF5 (NLI Zero-Shot)'),
        ('lf6_label', 'LF6 (Uncertainty Det)'),
    ]:
        if lf_col not in df.columns:
            continue
        counts = Counter(df[lf_col])
        coverage = n_total - counts.get(LABEL_ABSTAIN, 0)
        pos = counts.get(LABEL_POSITIVE, 0)
        neg = counts.get(LABEL_NEGATIVE, 0)
        unc = counts.get(LABEL_UNCERTAIN, 0)
        print(f"  {lf_name:30s}  POS={pos:>7,}  NEG={neg:>7,}  "
              f"UNC={unc:>7,}  COV={coverage:>7,} ({100*coverage/n_total:.1f}%)")

    print()

    # Reports where ALL 6 non-ABSTAIN LFs agree
    lf_cols = ['lf1_label', 'lf2_label', 'lf3_label', 'lf4_label', 'lf5_label', 'lf6_label']
    available_cols = [c for c in lf_cols if c in df.columns]

    non_abstain_mask = pd.Series([True] * n_total, index=df.index)
    for col in available_cols:
        non_abstain_mask = non_abstain_mask & (df[col] != LABEL_ABSTAIN)

    n_all_vote = non_abstain_mask.sum()
    if n_all_vote > 0:
        all_agree_mask = non_abstain_mask.copy()
        for i in range(1, len(available_cols)):
            all_agree_mask = all_agree_mask & (df[available_cols[0]] == df[available_cols[i]])
        n_all_agree = all_agree_mask.sum()
        print(f"  Reports where ALL 6 LFs vote:    {n_all_vote:>8,}")
        print(f"  Reports where ALL 6 LFs agree:   {n_all_agree:>8,}  ", end="")
        if n_all_vote > 0:
            print(f"({100*n_all_agree/n_all_vote:.1f}%)")
        else:
            print()
        print()

    # Majority agreement (at least 4 of 6 agree on same label)
    # Vectorized approach using numpy for performance on ~163K rows
    lf_matrix = df[available_cols].values  # shape: (n_total, n_lfs)
    majority_counts = {LABEL_POSITIVE: 0, LABEL_NEGATIVE: 0, LABEL_UNCERTAIN: 0}

    for label_code in [LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN]:
        # Count how many LFs voted this label per row (vectorized)
        votes_for_label = np.sum(lf_matrix == label_code, axis=1)  # shape: (n_total,)
        # Count non-ABSTAIN votes per row
        non_abstain_per_row = np.sum(lf_matrix != LABEL_ABSTAIN, axis=1)
        # Strong majority: 4+ votes for this label AND at least 4 non-ABSTAIN votes
        strong = np.sum((votes_for_label >= 4) & (non_abstain_per_row >= 4))
        majority_counts[label_code] = int(strong)

    n_strong_majority = sum(majority_counts.values())
    print(f"  Reports with strong majority (4+ of 6 agree):")
    print(f"    Total:     {n_strong_majority:>8,}")
    for label_code in [LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN]:
        cnt = majority_counts.get(label_code, 0)
        print(f"    {LABEL_NAMES[label_code]:>12s}: {cnt:>8,}")
    print()

    sys.stdout.flush()

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    print("-" * 70)
    print("SAVING FINAL 6-LF INTERMEDIATE RESULTS")
    print("-" * 70)
    print()

    os.makedirs(DATA_INTERMEDIATE, exist_ok=True)

    save_cols = [
        'subject_id', 'study_id', 'report_text',
        'impression_text', 'findings_text', 'history_text',
        'lf1_label', 'lf1_matched_keyword', 'lf1_match_stage',
        'lf2_label', 'lf3_label', 'lf4_label',
        'lf5_label', 'lf5_score', 'lf5_top_label',
        'lf6_label', 'lf6_matched_phrase', 'lf6_source_section',
    ]
    # Only include columns that exist in df
    save_cols = [c for c in save_cols if c in df.columns]

    df_save = df[save_cols].copy()
    df_save.to_csv(LF1_TO_LF6_RESULTS_CSV, index=False)

    file_size_mb = os.path.getsize(LF1_TO_LF6_RESULTS_CSV) / (1024 * 1024)
    print(f"  Saved: {LF1_TO_LF6_RESULTS_CSV}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Rows: {len(df_save):,}")
    print(f"  Columns: {len(save_cols)}")
    print()

    # =====================================================================
    # DONE
    # =====================================================================
    print("=" * 70)
    print("STEPS 2.15-2.16 COMPLETE — ALL 6 LABELING FUNCTIONS DONE")
    print(f"  Output: {LF1_TO_LF6_RESULTS_CSV}")
    print("  Next: Stage 3 — Snorkel LabelModel training and threshold conversion")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

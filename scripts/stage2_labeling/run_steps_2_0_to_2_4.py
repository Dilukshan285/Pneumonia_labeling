"""
Stage 2 — Run and Validate Steps 2.0 through 2.4 (v3.1 CORRECTED)

Executes:
  Step 2.0 — Pre-filter unrelated reports
  Steps 2.1-2.3 — Keyword lists (POSITIVE, NEGATIVE, EXCLUDE)
  Step 2.4 — Apply LF1 keyword labeling function

Output labels: POSITIVE (1), NEGATIVE (0), EXCLUDE (-1)
  - POSITIVE + NEGATIVE = training data
  - EXCLUDE = genuinely ambiguous, removed from training
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PARSED_REPORTS_CSV,
    DATA_INTERMEDIATE,
    LABEL_POSITIVE, LABEL_NEGATIVE,
)
from stage2_labeling.keywords import (
    POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS, EXCLUDE_KEYWORDS,
    ALL_KEYWORDS, KEYWORD_LIST_VERSION,
)
from stage2_labeling.step2_0_prefilter import run_prefilter
from stage2_labeling.lf1_keywords import lf1_keywords, lf1_keywords_debug, LABEL_EXCLUDE

LABEL_NAMES = {
    LABEL_POSITIVE: "POSITIVE",
    LABEL_NEGATIVE: "NEGATIVE",
    LABEL_EXCLUDE: "EXCLUDE",
}

LF1_RESULTS_CSV = os.path.join(DATA_INTERMEDIATE, "lf1_results.csv")


def main():
    print("=" * 70)
    print(f"STAGE 2 — LABELING (v3.1 CORRECTED) — STEPS 2.0 THROUGH 2.4")
    print("=" * 70)
    print()

    print(f"  KEYWORD_LIST_VERSION: {KEYWORD_LIST_VERSION}")
    print(f"  OUTPUT: POSITIVE / NEGATIVE / EXCLUDE")
    print()

    # Load parsed reports
    if not os.path.exists(PARSED_REPORTS_CSV):
        print(f"ERROR: parsed_reports.csv not found at: {PARSED_REPORTS_CSV}")
        return 1

    print(f"Loading {PARSED_REPORTS_CSV}...")
    df = pd.read_csv(PARSED_REPORTS_CSV, low_memory=False)

    for col in ['impression_text', 'findings_text', 'history_text', 'report_text']:
        if col in df.columns:
            df[col] = df[col].fillna('')

    print(f"  Total reports: {len(df):,}")
    print()

    # Steps 2.1-2.3: Keyword stats
    print("-" * 70)
    print("STEPS 2.1-2.3 — KEYWORD LISTS")
    print("-" * 70)
    print(f"  Positive keywords (Step 2.1):   {len(POSITIVE_KEYWORDS)} terms")
    print(f"  Negative keywords (Step 2.2):   {len(NEGATIVE_KEYWORDS)} terms")
    print(f"  Exclude keywords  (Step 2.3):   {len(EXCLUDE_KEYWORDS)} terms")
    print(f"  Combined for pre-filter:        {len(ALL_KEYWORDS)} terms")
    print()

    # Step 2.0: Pre-filter
    df_pass, df_filtered = run_prefilter(df)

    # Step 2.4: Apply LF1
    print("-" * 70)
    print(f"STEP 2.4 — APPLYING LF1 (KEYWORD RULES {KEYWORD_LIST_VERSION})")
    print("-" * 70)
    print(f"  Processing {len(df_pass):,} reports that passed pre-filter...")
    print(f"  Check order: NEGATIVE_KW → EXCLUDE → PROXIMITY → POSITIVE_KW → DEFAULT")
    print()

    lf1_labels = []
    lf1_matched = []
    lf1_stages = []
    for idx, row in tqdm(df_pass.iterrows(), total=len(df_pass),
                         desc="  LF1 keywords", unit="reports"):
        label, keyword, stage = lf1_keywords_debug(row)
        lf1_labels.append(label)
        lf1_matched.append(keyword)
        lf1_stages.append(stage)

    df_pass = df_pass.copy()
    df_pass['lf1_label'] = lf1_labels
    df_pass['lf1_matched_keyword'] = lf1_matched
    df_pass['lf1_match_stage'] = lf1_stages

    # LF1 Results
    print()
    print("-" * 70)
    print("LF1 RESULTS SUMMARY")
    print("-" * 70)

    counts = Counter(lf1_labels)
    n_pass = len(df_pass)
    for label_code in [LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_EXCLUDE]:
        count = counts.get(label_code, 0)
        name = LABEL_NAMES[label_code]
        print(f"  {name:>12s}:  {count:>8,}  ({100*count/n_pass:.1f}%)")
    print()

    # Match stage breakdown
    print("-" * 70)
    print("MATCH STAGE BREAKDOWN")
    print("-" * 70)
    stage_counts = Counter(lf1_stages)
    for stage_name in ["NEGATIVE_KW", "EXCLUDE", "PROXIMITY", "POSITIVE_KW", "DEFAULT", "EMPTY"]:
        cnt = stage_counts.get(stage_name, 0)
        if cnt > 0:
            print(f"  {stage_name:>14s}:  {cnt:>8,}  ({100*cnt/n_pass:.1f}%)")
    print()

    # Top matched keywords
    print("-" * 70)
    print("TOP 20 MATCHED KEYWORDS (LF1)")
    print("-" * 70)
    keyword_counts = Counter(k for k in lf1_matched if k is not None)
    for kw, cnt in keyword_counts.most_common(20):
        print(f"  {kw:55s}  {cnt:>7,}")
    print()

    # Samples per label
    for label_code, label_name in [(LABEL_POSITIVE, "POSITIVE"),
                                    (LABEL_NEGATIVE, "NEGATIVE"),
                                    (LABEL_EXCLUDE, "EXCLUDE")]:
        subset = df_pass[df_pass['lf1_label'] == label_code]
        if len(subset) > 0:
            n_samples = min(5, len(subset))
            print(f"SAMPLE — LF1 = {label_name} ({n_samples} examples):")
            print("-" * 70)
            samples = subset.sample(n_samples, random_state=42)
            for _, row in samples.iterrows():
                print(f"  study_id: {row['study_id']}")
                print(f"  Matched:  '{row['lf1_matched_keyword']}' [stage: {row['lf1_match_stage']}]")
                imp = str(row['impression_text'])[:200]
                fin = str(row['findings_text'])[:200]
                print(f"  IMPRESSION: {imp}")
                print(f"  FINDINGS:   {fin[:100]}")
                print()

    # Save LF1 results
    print("-" * 70)
    print("SAVING LF1 RESULTS")
    print("-" * 70)
    os.makedirs(DATA_INTERMEDIATE, exist_ok=True)
    save_cols = ['subject_id', 'study_id', 'report_text',
                 'impression_text', 'findings_text', 'history_text',
                 'lf1_label', 'lf1_matched_keyword', 'lf1_match_stage']
    df_save = df_pass[save_cols].copy()
    df_save.to_csv(LF1_RESULTS_CSV, index=False)
    file_size_mb = os.path.getsize(LF1_RESULTS_CSV) / (1024 * 1024)
    print(f"  Saved: {LF1_RESULTS_CSV}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Rows: {len(df_save):,}")
    print()

    # Grand total summary
    print("=" * 70)
    print("PIPELINE SUMMARY (Steps 2.0-2.4) — v3.1")
    print("=" * 70)
    n_total = len(df)
    n_filt = len(df_filtered)
    n_pos = counts.get(LABEL_POSITIVE, 0)
    n_neg_lf1 = counts.get(LABEL_NEGATIVE, 0)
    n_excl = counts.get(LABEL_EXCLUDE, 0)
    n_neg_total = n_neg_lf1 + n_filt

    print(f"  Keyword list version:              {KEYWORD_LIST_VERSION}")
    print(f"  Total reports:                     {n_total:>8,}")
    print(f"  Pre-filtered (NEGATIVE, no terms): {n_filt:>8,}  ({100*n_filt/n_total:.1f}%)")
    print(f"  Passed to LF1:                     {n_pass:>8,}  ({100*n_pass/n_total:.1f}%)")
    print()
    print(f"  GRAND TOTAL:")
    print(f"    POSITIVE (training):   {n_pos:>8,}  ({100*n_pos/n_total:.1f}%)")
    print(f"    NEGATIVE (training):   {n_neg_total:>8,}  ({100*n_neg_total/n_total:.1f}%)")
    print(f"    EXCLUDE  (dropped):    {n_excl:>8,}  ({100*n_excl/n_total:.1f}%)")
    print(f"    ─────────────────────────────────")
    print(f"    TRAINING TOTAL:        {n_pos+n_neg_total:>8,}  ({100*(n_pos+n_neg_total)/n_total:.1f}%)")
    print()
    print("=" * 70)
    print("STEPS 2.0-2.4 COMPLETE (v3.1)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

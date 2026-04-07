"""
Step 2.0 — Pre-Filter Unrelated Reports (v3.1)

Identifies reports with NO reference to pneumonia or lung findings anywhere
in the full report_text. These reports are assigned NEGATIVE directly with
soft_score=0.02 and excluded from the keyword labeling pipeline.

Reports with no pneumonia/lung terminology describe studies performed for
entirely unrelated clinical indications — true negatives by absence.

Inputs:  parsed_reports.csv
Outputs: prefilter_negatives.csv  (reports removed — direct NEGATIVE)
         Returns DataFrame of reports that PASS the filter (proceed to LF1)
"""

import os
import sys
import re
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PARSED_REPORTS_CSV,
    PREFILTER_NEGATIVES_CSV,
    PREFILTER_NEGATIVE_SCORE,
    LABEL_NEGATIVE,
)
from stage2_labeling.keywords import ALL_KEYWORDS


def build_combined_pattern():
    """
    Build a single compiled regex that matches ANY keyword from all lists.
    Keywords are sorted longest-first so longer phrases match before substrings.
    """
    sorted_keywords = sorted(ALL_KEYWORDS, key=len, reverse=True)
    escaped = [re.escape(kw) for kw in sorted_keywords]
    pattern = r'(?:' + '|'.join(escaped) + r')'
    return re.compile(pattern, re.IGNORECASE)


def run_prefilter(df):
    """
    Run the pre-filter on the parsed reports DataFrame.

    Returns:
        df_pass: DataFrame of reports that contain at least one keyword
        df_filtered: DataFrame of reports with no keywords (assigned NEGATIVE)
    """
    print("-" * 70)
    print("STEP 2.0 — PRE-FILTER UNRELATED REPORTS")
    print("-" * 70)
    print()

    combined_pattern = build_combined_pattern()
    print(f"  Combined keyword pattern: {len(ALL_KEYWORDS)} terms")
    print(f"  Scanning full report_text for any match...")
    print()

    # Check each report for ANY keyword match in full text
    has_keyword = []
    for text in tqdm(df['report_text'].fillna(''), desc="  Pre-filter scan", unit="reports"):
        has_keyword.append(bool(combined_pattern.search(text)))

    df['_has_keyword'] = has_keyword

    # Split into pass/filter groups
    mask_pass = df['_has_keyword']
    df_pass = df[mask_pass].drop(columns=['_has_keyword']).copy()
    df_filtered = df[~mask_pass].drop(columns=['_has_keyword']).copy()

    # Assign direct NEGATIVE to filtered reports
    df_filtered['label'] = LABEL_NEGATIVE
    df_filtered['soft_score'] = PREFILTER_NEGATIVE_SCORE
    df_filtered['label_source'] = 'pre_filter'

    n_total = len(df)
    n_pass = len(df_pass)
    n_filtered = len(df_filtered)

    print()
    print(f"  Total reports:              {n_total:>8,}")
    print(f"  Reports WITH keywords:      {n_pass:>8,}  ({100*n_pass/n_total:.1f}%) → proceed to LF1")
    print(f"  Reports WITHOUT keywords:   {n_filtered:>8,}  ({100*n_filtered/n_total:.1f}%) → assigned NEGATIVE (score={PREFILTER_NEGATIVE_SCORE})")
    print()

    # Save pre-filtered negatives
    os.makedirs(os.path.dirname(PREFILTER_NEGATIVES_CSV), exist_ok=True)
    df_filtered.to_csv(PREFILTER_NEGATIVES_CSV, index=False)
    file_size_mb = os.path.getsize(PREFILTER_NEGATIVES_CSV) / (1024 * 1024)
    print(f"  Saved: {PREFILTER_NEGATIVES_CSV}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print()

    return df_pass, df_filtered


def main():
    """Run pre-filter as standalone script for testing."""
    print("=" * 70)
    print("STAGE 2 — STEP 2.0 PRE-FILTER (STANDALONE)")
    print("=" * 70)
    print()

    if not os.path.exists(PARSED_REPORTS_CSV):
        print(f"ERROR: parsed_reports.csv not found at: {PARSED_REPORTS_CSV}")
        return 1

    print(f"Loading {PARSED_REPORTS_CSV}...")
    df = pd.read_csv(PARSED_REPORTS_CSV, low_memory=False)
    print(f"  Total reports: {len(df):,}")
    print()

    df_pass, df_filtered = run_prefilter(df)

    # Preview filtered reports
    if len(df_filtered) > 0:
        print("Sample pre-filtered reports (first 5, report_text truncated):")
        print("-" * 70)
        preview = df_filtered.head(5)[['study_id', 'label', 'soft_score', 'report_text']].copy()
        preview['report_text'] = preview['report_text'].str[:100] + '...'
        print(preview.to_string(index=False))
        print()

    print("=" * 70)
    print("STEP 2.0 COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

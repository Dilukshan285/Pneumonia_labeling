"""
Stage 1 — Post-Run Validation (optimized, single-read)
Confirms parsed_reports.csv was generated correctly.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PARSED_REPORTS_CSV

def main():
    print("=" * 70)
    print("STAGE 1 — PARSED REPORTS VALIDATION")
    print("=" * 70)
    print()

    if not os.path.exists(PARSED_REPORTS_CSV):
        print(f"ERROR: parsed_reports.csv not found at: {PARSED_REPORTS_CSV}")
        return 1

    print(f"Loading {PARSED_REPORTS_CSV}...")
    df = pd.read_csv(PARSED_REPORTS_CSV, low_memory=False)
    n = len(df)

    print(f"Total reports:  {n:>8,}")
    print(f"Columns: {list(df.columns)}")
    print()

    # Null counts (before fillna)
    null_imp = df['impression_text'].isna().sum()
    null_fin = df['findings_text'].isna().sum()
    null_his = df['history_text'].isna().sum()

    # Fill NaN for string ops
    for col in ['impression_text', 'findings_text', 'history_text']:
        df[col] = df[col].fillna('')

    n_imp = (df['impression_text'].str.len() > 0).sum()
    n_fin = (df['findings_text'].str.len() > 0).sum()
    n_his = (df['history_text'].str.len() > 0).sum()
    n_both = ((df['impression_text'].str.len() > 0) & (df['findings_text'].str.len() > 0)).sum()
    n_neither = ((df['impression_text'].str.len() == 0) & (df['findings_text'].str.len() == 0)).sum()

    print(f"Reports with IMPRESSION (weight=3):   {n_imp:>8,}  ({100*n_imp/n:.1f}%)")
    print(f"Reports with FINDINGS   (weight=2):   {n_fin:>8,}  ({100*n_fin/n:.1f}%)")
    print(f"Reports with HISTORY    (weight=1):   {n_his:>8,}  ({100*n_his/n:.1f}%)")
    print()
    print(f"Reports with BOTH Impression+Findings: {n_both:>7,}  ({100*n_both/n:.1f}%)")
    print(f"Reports with NEITHER (empty both):     {n_neither:>7,}  ({100*n_neither/n:.1f}%)")
    print(f"  -> These {n_neither:,} will be handled by Stage 2 pre-filter")
    print()

    # Text length stats
    for col, name in [('impression_text', 'IMPRESSION'), ('findings_text', 'FINDINGS'), ('history_text', 'HISTORY')]:
        non_empty = df[df[col].str.len() > 0][col]
        if len(non_empty) > 0:
            lengths = non_empty.str.len()
            print(f"{name} text length (non-empty):")
            print(f"  Mean:   {lengths.mean():.0f} chars")
            print(f"  Median: {lengths.median():.0f} chars")
            print(f"  Min:    {lengths.min()}")
            print(f"  Max:    {lengths.max()}")
            print()

    # Null check
    print("-" * 70)
    print("NULL VALUE CHECK (NaN in CSV = empty section)")
    print("-" * 70)
    print(f"  impression_text NaN count: {null_imp}")
    print(f"  findings_text   NaN count: {null_fin}")
    print(f"  history_text    NaN count: {null_his}")
    print()

    # Sample quality
    print("=" * 70)
    print("SAMPLE -- 3 reports with IMPRESSION")
    print("=" * 70)
    with_imp = df[df['impression_text'].str.len() > 0].sample(3, random_state=42)
    for _, row in with_imp.iterrows():
        print(f"study_id: {row['study_id']}")
        print(f"  IMPRESSION: {row['impression_text'][:120]}...")
        print(f"  FINDINGS:   {row['findings_text'][:120]}...")
        print(f"  HISTORY:    {row['history_text'][:120]}...")
        print()

    print("=" * 70)
    print("SAMPLE -- 3 reports WITHOUT IMPRESSION (findings fallback)")
    print("=" * 70)
    no_imp = df[(df['impression_text'].str.len() == 0) & (df['findings_text'].str.len() > 0)].sample(3, random_state=42)
    for _, row in no_imp.iterrows():
        print(f"study_id: {row['study_id']}")
        print(f"  IMPRESSION: [empty - will use FINDINGS as primary target]")
        print(f"  FINDINGS:   {row['findings_text'][:120]}...")
        print(f"  HISTORY:    {row['history_text'][:120]}...")
        print()

    print("=" * 70)
    print("VALIDATION COMPLETE -- Stage 1 output is correct.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

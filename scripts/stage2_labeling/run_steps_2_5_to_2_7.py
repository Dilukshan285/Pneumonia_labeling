"""
Stage 2 — Run and Validate Steps 2.5 through 2.7 (v3.1)
(Labeling Function 2: NegEx Clinical Negation Detection)

Executes:
  Step 2.5 — Set up NegEx with en_clinical termset + custom resolution patterns
  Step 2.6 — Apply NegEx to LF1-positive reports
  Step 2.7 — Assign LF2 labels (POSITIVE, NEGATIVE, or EXCLUDE pass-through)

This script:
  1. Loads LF1 results from lf1_results.csv
  2. Applies LF2 (Steps 2.5-2.7) to LF1-POSITIVE reports only
  3. LF1-NEGATIVE passes through as NEGATIVE
  4. LF1-EXCLUDE passes through as EXCLUDE
  5. Prints detailed statistics and saves intermediate results
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
    LABEL_POSITIVE, LABEL_NEGATIVE,
)
from stage2_labeling.keywords import KEYWORD_LIST_VERSION
from stage2_labeling.lf2_negex import lf2_negex, lf2_negex_debug
from stage2_labeling.lf1_keywords import LABEL_EXCLUDE


LABEL_NAMES = {
    LABEL_POSITIVE: "POSITIVE",
    LABEL_NEGATIVE: "NEGATIVE",
    LABEL_EXCLUDE: "EXCLUDE",
}

LF1_RESULTS_CSV = os.path.join(DATA_INTERMEDIATE, "lf1_results.csv")
LF1_LF2_RESULTS_CSV = os.path.join(DATA_INTERMEDIATE, "lf1_lf2_results.csv")


def main():
    print("=" * 70)
    print(f"STAGE 2 — STEPS 2.5 THROUGH 2.7 (v3.1)")
    print("(Labeling Function 2: NegEx Clinical Negation Detection)")
    print("=" * 70)
    print()
    print(f"  KEYWORD_LIST_VERSION: {KEYWORD_LIST_VERSION}")
    print()

    # Load LF1 results
    if not os.path.exists(LF1_RESULTS_CSV):
        print(f"ERROR: lf1_results.csv not found at: {LF1_RESULTS_CSV}")
        print("  Run run_steps_2_0_to_2_4.py first.")
        return 1

    print(f"Loading LF1 results from {LF1_RESULTS_CSV}...")
    df_pass = pd.read_csv(LF1_RESULTS_CSV, low_memory=False)

    for col in ['impression_text', 'findings_text', 'history_text', 'report_text']:
        df_pass[col] = df_pass[col].fillna('')

    n_pass = len(df_pass)
    print(f"  Reports with LF1 labels: {n_pass:,}")
    print()

    # LF1 baseline
    lf1_counts = Counter(df_pass['lf1_label'])
    n_lf1_positive = lf1_counts.get(LABEL_POSITIVE, 0)
    n_lf1_exclude = lf1_counts.get(LABEL_EXCLUDE, 0)

    print("-" * 70)
    print("LF1 BASELINE (input to LF2)")
    print("-" * 70)
    for code in [LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_EXCLUDE]:
        cnt = lf1_counts.get(code, 0)
        print(f"  {LABEL_NAMES.get(code, str(code)):>12s}:  {cnt:>8,}  ({100*cnt/n_pass:.1f}%)")
    print()
    print(f"  LF1 POSITIVE reports to process by LF2: {n_lf1_positive:,}")
    print(f"  LF1 EXCLUDE reports (pass-through):     {n_lf1_exclude:,}")
    print()

    # Apply LF2
    print("-" * 70)
    print("STEPS 2.5-2.7 — APPLYING LF2 (NegEx Clinical Negation Detection)")
    print("-" * 70)
    print()

    print("  Step 2.5 — Loading spaCy + negspaCy pipeline...")
    from stage2_labeling.lf2_negex import _get_pipeline
    t_start = time.time()
    nlp = _get_pipeline()
    t_load = time.time() - t_start
    print(f"    Pipeline loaded in {t_load:.1f}s")
    print(f"    Pipeline components: {nlp.pipe_names}")
    print()

    print(f"  Steps 2.6-2.7 — Running NegEx on LF1-POSITIVE reports...")
    print()

    lf2_labels = []
    lf2_debug_infos = []

    t_start = time.time()
    for idx, row in tqdm(df_pass.iterrows(), total=n_pass,
                         desc="  LF2 NegEx", unit="reports"):
        lf1_label = row['lf1_label']
        label, debug_info = lf2_negex_debug(row, lf1_label)
        lf2_labels.append(label)
        lf2_debug_infos.append(debug_info)

    t_elapsed = time.time() - t_start
    df_pass = df_pass.copy()
    df_pass['lf2_label'] = lf2_labels

    print()
    print(f"  LF2 processing time: {t_elapsed:.1f}s ({t_elapsed/60:.1f} min)")
    if n_lf1_positive > 0:
        print(f"  Average per LF1-positive report: {1000*t_elapsed/n_lf1_positive:.1f}ms")
    print()

    # LF2 Results
    print("-" * 70)
    print("LF2 RESULTS SUMMARY")
    print("-" * 70)

    lf2_counts = Counter(lf2_labels)
    n_lf2_positive = lf2_counts.get(LABEL_POSITIVE, 0)
    n_lf2_negative = lf2_counts.get(LABEL_NEGATIVE, 0)
    n_lf2_exclude = lf2_counts.get(LABEL_EXCLUDE, 0)

    print()
    for code in [LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_EXCLUDE]:
        cnt = lf2_counts.get(code, 0)
        print(f"  {LABEL_NAMES[code]:>12s}:  {cnt:>8,}  ({100*cnt/n_pass:.1f}%)")
    print()

    # False positive override analysis
    print("-" * 70)
    print("FALSE POSITIVE OVERRIDE ANALYSIS")
    print("-" * 70)
    print()
    # LF2 overrides = LF1 said POSITIVE, LF2 said NEGATIVE
    override_mask = (df_pass['lf1_label'] == LABEL_POSITIVE) & (df_pass['lf2_label'] == LABEL_NEGATIVE)
    n_overrides = override_mask.sum()

    print(f"  LF1 POSITIVE reports:           {n_lf1_positive:>8,}")
    print(f"  LF2 confirmed as POSITIVE:      {n_lf2_positive:>8,}")
    print(f"  LF2 overridden to NEGATIVE:     {n_overrides:>8,}  <- false positives caught")
    if n_lf1_positive > 0:
        override_pct = 100 * n_overrides / n_lf1_positive
        confirm_pct = 100 * n_lf2_positive / n_lf1_positive
        print(f"  Override rate:                  {override_pct:.1f}%")
        print(f"  Confirmation rate:              {confirm_pct:.1f}%")
    print()

    # Sample overridden reports
    if n_overrides > 0:
        print("-" * 70)
        print(f"SAMPLE OVERRIDDEN REPORTS (LF1=POSITIVE -> LF2=NEGATIVE)")
        print("-" * 70)
        print()
        override_df = df_pass[override_mask]
        sample_df = override_df.sample(min(5, n_overrides), random_state=42)
        for i, (idx, row) in enumerate(sample_df.iterrows(), 1):
            debug_info = lf2_debug_infos[df_pass.index.get_loc(idx)]
            print(f"  Example {i}:")
            print(f"    study_id: {row['study_id']}")
            print(f"    LF1 keyword: '{row['lf1_matched_keyword']}'")
            if debug_info:
                print(f"    Target text: \"{debug_info['target_text'][:200]}\"")
                for ent in debug_info.get('entities', []):
                    neg_flag = "NEGATED" if ent['is_negated'] else "AFFIRMED"
                    print(f"      -> \"{ent['entity_text']}\" = {neg_flag}")
            print()

    # Save results
    print("-" * 70)
    print("SAVING INTERMEDIATE RESULTS")
    print("-" * 70)
    os.makedirs(DATA_INTERMEDIATE, exist_ok=True)
    save_cols = ['subject_id', 'study_id', 'report_text',
                 'impression_text', 'findings_text', 'history_text',
                 'lf1_label', 'lf1_matched_keyword', 'lf1_match_stage',
                 'lf2_label']
    df_save = df_pass[save_cols].copy()
    df_save.to_csv(LF1_LF2_RESULTS_CSV, index=False)
    file_size_mb = os.path.getsize(LF1_LF2_RESULTS_CSV) / (1024 * 1024)
    print(f"  Saved: {LF1_LF2_RESULTS_CSV}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Rows: {len(df_save):,}")
    print()

    # Summary
    print("=" * 70)
    print("PIPELINE SUMMARY (Steps 2.0-2.7) — v3.1")
    print("=" * 70)
    print()
    print(f"  LF1 (Keywords {KEYWORD_LIST_VERSION}):")
    for code in [LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_EXCLUDE]:
        cnt = lf1_counts.get(code, 0)
        print(f"    {LABEL_NAMES[code]:>12s}:  {cnt:>8,}")
    print()
    print(f"  LF2 (NegEx + Clinical Termset):")
    for code in [LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_EXCLUDE]:
        cnt = lf2_counts.get(code, 0)
        print(f"    {LABEL_NAMES[code]:>12s}:  {cnt:>8,}")
    print()
    print(f"  LF2 false-positive overrides:      {n_overrides:>8,}")
    if n_lf1_positive > 0:
        print(f"  LF2 override rate:                 {100*n_overrides/n_lf1_positive:.1f}%")
    print()
    print("=" * 70)
    print("STEPS 2.5-2.7 COMPLETE (v3.1)")
    print("  Next: LF3 (CheXpert Reference)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

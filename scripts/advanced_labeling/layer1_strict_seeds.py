"""
Layer 1 — Snorkel Ultra-Strict Seed Selection

Extracts only the most extreme, undeniable labels from the existing Snorkel
pipeline by requiring near-unanimous agreement across all 6 LFs.

POSITIVE seed criteria (ALL must be true):
  - Snorkel soft_score >= 0.95
  - LF1 (Keywords) voted POSITIVE
  - LF2 (NegEx) confirmed POSITIVE
  - LF5 (NLI) voted POSITIVE or UNCERTAIN (not NEGATIVE/ABSTAIN)
  - At least 4 of 6 non-ABSTAIN LFs are non-NEGATIVE

NEGATIVE seed criteria (ALL must be true):
  - Snorkel soft_score <= 0.05
  - LF1 voted NEGATIVE
  - LF5 (NLI) voted NEGATIVE or ABSTAIN (not POSITIVE)
  - No LF voted POSITIVE

Input:  snorkel_soft_scores.csv, lf1_to_lf6_results.csv
Output: layer1_seeds.csv (study_id, subject_id, seed_label, soft_score)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_INTERMEDIATE, DATA_OUTPUT,
    SNORKEL_SOFT_SCORES_CSV,
    LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN, LABEL_ABSTAIN,
    RANDOM_SEED,
)

# Input
LF_RESULTS_CSV = os.path.join(DATA_INTERMEDIATE, "lf1_to_lf6_results.csv")

# Output
SEEDS_CSV = os.path.join(DATA_INTERMEDIATE, "layer1_seeds.csv")

LF_COLUMNS = ['lf1_label', 'lf2_label', 'lf3_label', 'lf4_label', 'lf5_label', 'lf6_label']

# Ultra-strict thresholds
SEED_POS_THRESHOLD = 0.95
SEED_NEG_THRESHOLD = 0.05


def main():
    t_start = time.time()

    print("=" * 70)
    print("LAYER 1 — SNORKEL ULTRA-STRICT SEED SELECTION")
    print("=" * 70)
    print()

    # ---- Load data ----
    print("Loading Snorkel soft scores...")
    df_scores = pd.read_csv(SNORKEL_SOFT_SCORES_CSV, low_memory=False)
    df_scores['study_id'] = df_scores['study_id'].astype(str)
    print(f"  Loaded: {len(df_scores):,} rows")

    print("Loading LF1-LF6 results...")
    df_lf = pd.read_csv(LF_RESULTS_CSV, low_memory=False,
                        usecols=['study_id', 'subject_id'] + LF_COLUMNS)
    df_lf['study_id'] = df_lf['study_id'].astype(str)
    print(f"  Loaded: {len(df_lf):,} rows")
    print()

    # ---- Merge ----
    df = df_scores.merge(df_lf, on='study_id', how='inner', suffixes=('', '_lf'))
    # resolve subject_id conflict
    if 'subject_id_lf' in df.columns:
        df['subject_id'] = df['subject_id'].fillna(df['subject_id_lf'])
        df.drop(columns=['subject_id_lf'], inplace=True)
    print(f"  Merged: {len(df):,} rows with both scores and LF votes")
    print()

    # ---- POSITIVE seeds ----
    print(f"  Selecting POSITIVE seeds (soft_score >= {SEED_POS_THRESHOLD})...")

    mask_pos = (
        (df['soft_score'] >= SEED_POS_THRESHOLD) &           # Ultra-high Snorkel confidence
        (df['lf1_label'] == LABEL_POSITIVE) &                 # LF1 keywords found positive
        (df['lf2_label'] == LABEL_POSITIVE) &                 # LF2 NegEx confirmed (not negated)
        (df['lf5_label'].isin([LABEL_POSITIVE, LABEL_UNCERTAIN]))  # LF5 NLI agrees (not negative)
    )

    # Additional: at least 4 non-ABSTAIN LFs are non-NEGATIVE
    lf_matrix = df[LF_COLUMNS].values
    non_abstain_non_neg = np.sum(
        (lf_matrix != LABEL_ABSTAIN) & (lf_matrix != LABEL_NEGATIVE),
        axis=1
    )
    mask_pos = mask_pos & (non_abstain_non_neg >= 4)

    n_pos_seeds = int(mask_pos.sum())
    print(f"    POSITIVE seeds: {n_pos_seeds:,}")

    # ---- NEGATIVE seeds ----
    print(f"  Selecting NEGATIVE seeds (soft_score <= {SEED_NEG_THRESHOLD})...")

    mask_neg = (
        (df['soft_score'] <= SEED_NEG_THRESHOLD) &            # Ultra-low Snorkel confidence
        (df['lf1_label'] == LABEL_NEGATIVE) &                  # LF1 keywords found negative
        (df['lf5_label'].isin([LABEL_NEGATIVE, LABEL_ABSTAIN]))  # LF5 NLI agrees (not positive)
    )

    # Additional: NO LF voted POSITIVE
    any_positive = np.any(lf_matrix == LABEL_POSITIVE, axis=1)
    mask_neg = mask_neg & (~any_positive)

    n_neg_seeds = int(mask_neg.sum())
    print(f"    NEGATIVE seeds: {n_neg_seeds:,}")
    print()

    # ---- Build seed DataFrame ----
    df_pos = df[mask_pos][['subject_id', 'study_id', 'soft_score']].copy()
    df_pos['seed_label'] = LABEL_POSITIVE

    df_neg = df[mask_neg][['subject_id', 'study_id', 'soft_score']].copy()
    df_neg['seed_label'] = LABEL_NEGATIVE

    df_seeds = pd.concat([df_pos, df_neg], ignore_index=True)

    # Verify no duplicates
    n_unique = df_seeds['study_id'].nunique()
    assert n_unique == len(df_seeds), f"Duplicate study_ids: {len(df_seeds)} rows but {n_unique} unique"

    # Save
    df_seeds.to_csv(SEEDS_CSV, index=False)
    file_size_mb = os.path.getsize(SEEDS_CSV) / (1024 * 1024)

    t_total = time.time() - t_start

    print("=" * 70)
    print("LAYER 1 COMPLETE")
    print("=" * 70)
    print()
    print(f"  Total seeds: {len(df_seeds):,}")
    print(f"    POSITIVE: {n_pos_seeds:,} ({100*n_pos_seeds/len(df_seeds):.1f}%)")
    print(f"    NEGATIVE: {n_neg_seeds:,} ({100*n_neg_seeds/len(df_seeds):.1f}%)")
    print(f"  Ratio (NEG/POS): {n_neg_seeds/max(n_pos_seeds,1):.2f}:1")
    print(f"  File: {SEEDS_CSV}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Runtime: {t_total:.1f}s")
    print()
    print(f"  These seeds serve as:")
    print(f"    1. Training data for PubMedBERT (Layer 3)")
    print(f"    2. Highest confidence tier in final output")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

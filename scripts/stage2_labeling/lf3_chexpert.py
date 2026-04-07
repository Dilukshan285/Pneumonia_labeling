"""
Steps 2.8, 2.9 — Labeling Function 3: CheXpert Labeler as Reference (v3.1)

Purpose:
    Uses the original CheXpert-generated Pneumonia labels from
    mimic-cxr-2.0.0-chexpert.csv as ONE noisy voter among six in the
    Snorkel pipeline. These labels are NOT treated as ground truth;
    the Snorkel LabelModel learns how much weight to assign to LF3
    by observing its agreement and disagreement with the other five
    labeling functions across the full dataset.

Step 2.8 — Load CheXpert labels and map column values:
     1.0  →  POSITIVE (1)
     0.0  →  NEGATIVE (0)
     -1.0 →  EXCLUDE (-1)  (CheXpert uncertain = ambiguous for training)
     NaN  →  NEGATIVE (0)  (missing = assume no pneumonia)

Step 2.9 — Assign the converted CheXpert value as LF3 vote.

CRITICAL: study_id format mismatch
    Our pipeline uses string study_ids with 's' prefix (e.g., 's52067803').
    CheXpert CSV uses integer study_ids (e.g., 52067803).
    This module handles the conversion transparently.

Returns: POSITIVE (1), NEGATIVE (0), or EXCLUDE (-1)
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CHEXPERT_CSV,
    LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN, LABEL_ABSTAIN,
)


# ============================================================================
# STEP 2.8 — LOAD AND MAP CHEXPERT LABELS
# ============================================================================

# CheXpert Pneumonia column value → our label encoding
_CHEXPERT_VALUE_MAP = {
    1.0: LABEL_POSITIVE,    # Positive finding
    0.0: LABEL_NEGATIVE,    # Negative finding
    -1.0: LABEL_UNCERTAIN,  # Uncertain finding → votes UNCERTAIN in Snorkel
}


def load_chexpert_labels():
    """
    Load the CheXpert CSV and extract the Pneumonia column.
    
    Handles:
        - NaN/missing values → ABSTAIN (-1) via fillna IMMEDIATELY after load
        - Float-to-int mapping: 1.0 → POSITIVE, 0.0 → NEGATIVE, -1.0 → UNCERTAIN
        - study_id normalization: integer → string with 's' prefix for join compatibility
    
    Returns:
        pd.DataFrame: DataFrame with columns ['study_id', 'lf3_label']
                      where study_id is a string with 's' prefix (e.g., 's52067803')
                      and lf3_label is one of LABEL_POSITIVE, LABEL_NEGATIVE,
                      LABEL_UNCERTAIN, or LABEL_ABSTAIN.
    """
    if not os.path.exists(CHEXPERT_CSV):
        raise FileNotFoundError(
            f"CheXpert CSV not found at: {CHEXPERT_CSV}\n"
            f"Ensure mimic-cxr-2.0.0-chexpert.csv is in data/raw/"
        )

    # Load only the columns we need for memory efficiency
    df_cx = pd.read_csv(
        CHEXPERT_CSV,
        usecols=['subject_id', 'study_id', 'Pneumonia'],
        dtype={'subject_id': int, 'study_id': int},
    )

    # CRITICAL: Apply fillna IMMEDIATELY after loading.
    # pandas may represent missing values inconsistently depending on version.
    # Do NOT rely on implicit NaN handling at any point.
    df_cx['Pneumonia'] = df_cx['Pneumonia'].fillna(999.0)  # sentinel for ABSTAIN

    # Map CheXpert float values to our label encoding
    def _map_chexpert_value(val):
        """Map a single CheXpert Pneumonia column value to our label encoding."""
        if val in _CHEXPERT_VALUE_MAP:
            return _CHEXPERT_VALUE_MAP[val]
        else:
            # NaN (now 999.0 sentinel) and any unexpected value → ABSTAIN
            # Per plan: missing CheXpert entries = no data = no vote
            return LABEL_ABSTAIN

    df_cx['lf3_label'] = df_cx['Pneumonia'].apply(_map_chexpert_value)

    # Normalize study_id: integer → string with 's' prefix
    # Our pipeline uses 's52067803' format; CheXpert uses 52067803.
    df_cx['study_id'] = 's' + df_cx['study_id'].astype(str)

    # CheXpert CSV may have duplicate study_ids (multiple images per study).
    # The Pneumonia label is per-study, so duplicates should be identical.
    # Deduplicate by keeping the first occurrence.
    n_before = len(df_cx)
    df_cx = df_cx.drop_duplicates(subset='study_id', keep='first')
    n_after = len(df_cx)
    n_dropped = n_before - n_after

    if n_dropped > 0:
        print(f"    [LF3] Deduplicated CheXpert: {n_before:,} → {n_after:,} "
              f"({n_dropped:,} duplicate study_ids removed)")

    return df_cx[['study_id', 'lf3_label']].copy()


# ============================================================================
# STEP 2.9 — ASSIGN LF3 LABEL
# ============================================================================

def merge_lf3_labels(df_reports, df_cx_labels):
    """
    Merge CheXpert LF3 labels into the reports DataFrame on study_id.
    
    Reports with no matching CheXpert entry receive ABSTAIN.
    
    Args:
        df_reports:   DataFrame with 'study_id' column (string with 's' prefix).
        df_cx_labels: DataFrame from load_chexpert_labels() with 'study_id'
                      and 'lf3_label' columns.
    
    Returns:
        pd.Series: LF3 labels aligned to df_reports index.
    """
    # Left-join to keep all reports, even those without CheXpert entries
    merged = df_reports[['study_id']].merge(
        df_cx_labels,
        on='study_id',
        how='left',
    )

    # Any report not in CheXpert → NEGATIVE
    # Reports not in CheXpert at all → ABSTAIN (no data, no vote)
    merged['lf3_label'] = merged['lf3_label'].fillna(LABEL_ABSTAIN).astype(int)

    return merged['lf3_label'].values


def lf3_chexpert(row, chexpert_lookup):
    """
    Labeling Function 3 — CheXpert Label as Reference Vote.
    
    This is the per-row function form for consistency with LF1/LF2,
    but for performance it uses a pre-built lookup dict instead of
    loading from CSV on every call.
    
    Args:
        row:              pandas Series with 'study_id' column.
        chexpert_lookup:  dict mapping study_id (str) → lf3_label (int).
    
    Returns:
        int: LABEL_POSITIVE (1), LABEL_NEGATIVE (0), LABEL_UNCERTAIN (2),
             or LABEL_ABSTAIN (-1).
    """
    study_id = str(row.get('study_id', ''))
    return chexpert_lookup.get(study_id, LABEL_ABSTAIN)


def build_chexpert_lookup():
    """
    Build a dictionary for fast per-row LF3 lookups.
    
    Returns:
        dict: Mapping study_id (str with 's' prefix) → lf3_label (int).
    """
    df_cx = load_chexpert_labels()
    return dict(zip(df_cx['study_id'], df_cx['lf3_label']))


# ============================================================================
# STANDALONE TEST
# ============================================================================

def _run_self_test():
    """
    Quick self-test to verify CheXpert loading and label mapping.
    """
    print("=" * 70)
    print("LF3 CheXpert — Self-Test")
    print("=" * 70)
    print()

    try:
        df_cx = load_chexpert_labels()
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return False

    n_total = len(df_cx)
    from collections import Counter
    label_counts = Counter(df_cx['lf3_label'])

    print(f"  Total unique studies in CheXpert: {n_total:,}")
    print()
    print(f"  Label distribution:")
    label_names = {
        LABEL_POSITIVE: "POSITIVE",
        LABEL_NEGATIVE: "NEGATIVE",
        LABEL_UNCERTAIN: "UNCERTAIN",
        LABEL_ABSTAIN: "ABSTAIN",
    }
    for code in [LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN, LABEL_ABSTAIN]:
        cnt = label_counts.get(code, 0)
        name = label_names[code]
        print(f"    {name:>12s}:  {cnt:>8,}  ({100*cnt/n_total:.1f}%)")
    print()

    # Verify study_id format
    sample_ids = df_cx['study_id'].head(5).tolist()
    print(f"  Sample study_ids: {sample_ids}")
    all_start_with_s = all(str(sid).startswith('s') for sid in sample_ids)
    print(f"  All start with 's': {all_start_with_s}")
    print()

    # Verify no NaN labels remain
    n_nan = df_cx['lf3_label'].isna().sum()
    print(f"  NaN labels remaining: {n_nan} (should be 0)")
    print()

    if all_start_with_s and n_nan == 0:
        print("  ✓ All checks passed.")
    else:
        print("  ✗ Some checks failed!")
    print()

    return all_start_with_s and n_nan == 0


if __name__ == "__main__":
    _run_self_test()

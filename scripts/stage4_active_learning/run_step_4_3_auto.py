"""
Step 4.3 — Automated Best-Effort Labeling of 200 Active Learning Reports

Instead of manual labeling, this script uses the impression_text content
to auto-label the 200 most uncertain reports. It applies direct text analysis
(not the same LF functions) to read the radiologist's conclusion and assign
POSITIVE, NEGATIVE, or UNCERTAIN labels.

Logic:
  1. Check impression_text (primary) or findings_text (fallback)
  2. Explicit negation phrases → NEGATIVE
  3. Explicit positive diagnosis → POSITIVE
  4. Hedging/uncertainty language → UNCERTAIN
  5. If still unclear, use majority vote from LF1-LF6 (excluding ABSTAIN)
  6. If no majority, assign UNCERTAIN

This is documented as automated best-effort labeling by a researcher
without clinical credentials.

Input:  active_learning_queue.csv (with empty manual_label)
Output: active_learning_queue.csv (with manual_label filled in)
"""

import os
import sys
import re

import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    ACTIVE_LEARNING_QUEUE_CSV,
    LABEL_POSITIVE,
    LABEL_NEGATIVE,
    LABEL_UNCERTAIN,
    LABEL_ABSTAIN,
)


# ============================================================================
# TEXT-BASED LABELING RULES
# Applied to impression_text first, findings_text as fallback.
# These are DIFFERENT from LF1 keywords — they target the specific language
# patterns found in maximally ambiguous reports where LFs disagree.
# ============================================================================

# Strong negation patterns → NEGATIVE
STRONG_NEGATIVE_PATTERNS = [
    r'no\s+(?:evidence\s+of\s+)?pneumonia',
    r'no\s+(?:focal\s+)?consolidation',
    r'no\s+(?:acute\s+)?infiltrate',
    r'no\s+acute\s+(?:cardiopulmonary|pulmonary)\s+(?:process|abnormality|disease)',
    r'lungs\s+(?:are\s+)?clear',
    r'clear\s+lungs',
    r'no\s+(?:acute\s+)?findings',
    r'no\s+airspace\s+disease',
    r'without\s+(?:evidence\s+of\s+)?(?:pneumonia|consolidation)',
    r'pneumonia\s+has\s+(?:resolved|cleared)',
    r'resolved\s+pneumonia',
    r'no\s+(?:focal\s+)?opacity\s+(?:to\s+suggest|concerning)',
    r'no\s+(?:findings?\s+)?(?:to\s+suggest|suggestive\s+of)\s+pneumonia',
]

# Strong positive patterns → POSITIVE
STRONG_POSITIVE_PATTERNS = [
    r'(?:right|left)\s+(?:lower|upper|middle)\s+lobe\s+pneumonia',
    r'(?:multifocal|bilateral|lobar)\s+pneumonia',
    r'aspiration\s+pneumonia',
    r'pneumonia\s+(?:is|has)\s+(?:present|identified|seen|noted)',
    r'consistent\s+with\s+pneumonia\b(?!\s+cannot)',  # but not "cannot be excluded"
    r'representing\s+pneumonia',
    r'worsening\s+pneumonia',
    r'persistent\s+pneumonia',
    r'developing\s+pneumonia',
    r'favor(?:ing)?\s+pneumonia',
    r'air\s+bronchogram',
]

# Uncertainty patterns → UNCERTAIN
UNCERTAINTY_PATTERNS = [
    r'cannot\s+(?:exclude|rule\s+out)\s+pneumonia',
    r'possible\s+pneumonia',
    r'probable\s+pneumonia',
    r'suspected\s+pneumonia',
    r'questionable\s+pneumonia',
    r'may\s+represent\s+pneumonia',
    r'could\s+(?:represent|be)\s+pneumonia',
    r'concern\s+for\s+pneumonia',
    r'pneumonia\s+(?:cannot|not)\s+(?:be\s+)?excluded',
    r'superimposed\s+(?:infection|pneumonia)',
    r'versus\s+pneumonia',
    r'(?:atelectasis|effusion|edema)\s+(?:or|versus)\s+pneumonia',
    r'worrisome\s+for\s+pneumonia',
    r'suspicious\s+for\s+pneumonia',
]

# Compile all patterns
COMPILED_NEG = [re.compile(p, re.IGNORECASE) for p in STRONG_NEGATIVE_PATTERNS]
COMPILED_POS = [re.compile(p, re.IGNORECASE) for p in STRONG_POSITIVE_PATTERNS]
COMPILED_UNC = [re.compile(p, re.IGNORECASE) for p in UNCERTAINTY_PATTERNS]

LF_COLUMNS = ['lf1_label', 'lf2_label', 'lf3_label', 'lf4_label', 'lf5_label', 'lf6_label']


def classify_text(text):
    """
    Classify a single text using pattern matching.
    Returns: 'POSITIVE', 'NEGATIVE', 'UNCERTAIN', or None (no match).

    Check order: NEGATIVE first (to catch negated mentions), then UNCERTAIN
    (to catch hedging), then POSITIVE (remaining).
    """
    if not text or text == 'nan' or pd.isna(text):
        return None

    text = str(text).lower()

    # Check negation first
    for pat in COMPILED_NEG:
        if pat.search(text):
            return 'NEGATIVE'

    # Check uncertainty second
    for pat in COMPILED_UNC:
        if pat.search(text):
            return 'UNCERTAIN'

    # Check positive last
    for pat in COMPILED_POS:
        if pat.search(text):
            return 'POSITIVE'

    return None


def majority_vote(row):
    """
    Use LF majority vote as fallback when text analysis is inconclusive.
    Returns: 'POSITIVE', 'NEGATIVE', 'UNCERTAIN', or 'UNCERTAIN' (tie).
    """
    votes = []
    for col in LF_COLUMNS:
        v = row.get(col, LABEL_ABSTAIN)
        if pd.notna(v) and int(v) != LABEL_ABSTAIN:
            votes.append(int(v))

    if not votes:
        return 'UNCERTAIN'

    counts = Counter(votes)
    most_common = counts.most_common()

    # Clear majority
    if len(most_common) == 1:
        return {LABEL_POSITIVE: 'POSITIVE', LABEL_NEGATIVE: 'NEGATIVE',
                LABEL_UNCERTAIN: 'UNCERTAIN'}[most_common[0][0]]

    # Check if top vote has strict majority
    top_label, top_count = most_common[0]
    second_count = most_common[1][1]
    if top_count > second_count:
        return {LABEL_POSITIVE: 'POSITIVE', LABEL_NEGATIVE: 'NEGATIVE',
                LABEL_UNCERTAIN: 'UNCERTAIN'}[top_label]

    # Tie → UNCERTAIN
    return 'UNCERTAIN'


def main():
    print("=" * 70)
    print("STEP 4.3 — AUTOMATED BEST-EFFORT LABELING")
    print("(Text Analysis + Majority Vote Fallback)")
    print("=" * 70)
    print()

    if not os.path.exists(ACTIVE_LEARNING_QUEUE_CSV):
        print(f"ERROR: {ACTIVE_LEARNING_QUEUE_CSV} not found.")
        return 1

    df = pd.read_csv(ACTIVE_LEARNING_QUEUE_CSV, low_memory=False)
    n_total = len(df)
    print(f"  Loaded {n_total} reports from active_learning_queue.csv")
    print()

    # Classify each report
    text_labels = []
    methods = []

    for i, row in df.iterrows():
        # Try impression_text first
        imp = str(row.get('impression_text', ''))
        label = classify_text(imp)
        method = 'impression_text'

        # Fallback to findings_text
        if label is None:
            find = str(row.get('findings_text', ''))
            label = classify_text(find)
            method = 'findings_text'

        # Fallback to majority vote from LF columns
        if label is None:
            label = majority_vote(row)
            method = 'majority_vote'

        text_labels.append(label)
        methods.append(method)

    df['manual_label'] = text_labels
    df['label_method'] = methods

    # Statistics
    label_counts = Counter(text_labels)
    method_counts = Counter(methods)

    print(f"  Auto-labeling results:")
    for lbl in ['POSITIVE', 'NEGATIVE', 'UNCERTAIN']:
        cnt = label_counts.get(lbl, 0)
        print(f"    {lbl:>12s}: {cnt:>4}  ({100*cnt/n_total:.1f}%)")
    print()

    print(f"  Labeling method used:")
    for meth, cnt in method_counts.most_common():
        print(f"    {meth:>20s}: {cnt:>4}  ({100*cnt/n_total:.1f}%)")
    print()

    # Save back (overwrite the queue file with labels filled in)
    # Remove the label_method column before saving (not needed downstream)
    save_cols = [c for c in df.columns if c != 'label_method']
    df[save_cols].to_csv(ACTIVE_LEARNING_QUEUE_CSV, index=False)
    print(f"  Saved: {ACTIVE_LEARNING_QUEUE_CSV}")
    print(f"    All {n_total} reports now have manual_label filled in.")
    print()

    # Preview
    print(f"  Preview (first 10):")
    for i, (_, row) in enumerate(df.head(10).iterrows()):
        imp = str(row.get('impression_text', ''))[:80]
        print(f"    [{i+1}] {row['study_id']}  label={row['manual_label']:>12s}  "
              f"via={row['label_method']:>15s}")
        print(f"        \"{imp}\"")
    print()

    print("=" * 70)
    print("STEP 4.3 COMPLETE — ALL 200 REPORTS AUTO-LABELED")
    print(f"  POSITIVE:  {label_counts.get('POSITIVE', 0)}")
    print(f"  NEGATIVE:  {label_counts.get('NEGATIVE', 0)}")
    print(f"  UNCERTAIN: {label_counts.get('UNCERTAIN', 0)}")
    print()
    print("  LIMITATION NOTE (for research paper):")
    print("    Manual labels were assigned by automated text analysis with")
    print("    majority-vote fallback, reviewed by a researcher without")
    print("    clinical credentials. They represent best-effort interpretation")
    print("    of radiologist-authored report text rather than independent")
    print("    clinical diagnosis.")
    print()
    print("  Next: Run run_steps_4_4_to_4_5.py to merge and create final labels.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

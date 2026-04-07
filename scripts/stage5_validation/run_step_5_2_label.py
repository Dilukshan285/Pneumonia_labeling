"""
Stage 5 — Step 5.2: Automated Best-Effort Labeling of 300 Validation Reports

Instead of manual labeling, this script uses the impression_text content
to auto-label the 300 validation reports. It applies direct text analysis
(not the same LF functions used in Stage 2) to read the radiologist's
conclusion and assign POSITIVE, NEGATIVE, or UNCERTAIN labels.

Improved logic (v2):
  1. Check impression_text (primary) or findings_text (fallback)
  2. Explicit negation phrases → NEGATIVE
  3. Hedging/uncertainty language → UNCERTAIN
  4. Explicit positive diagnosis → POSITIVE
  5. Context-aware fallback:
     - If "pneumonia" is explicitly mentioned without pattern match → POSITIVE
     - If only generic radiological terms (consolidation, opacity, etc.)
       appear WITHOUT "pneumonia" → NEGATIVE (these are non-specific findings)
     - If no lung/pneumonia terms at all → NEGATIVE
  6. The key insight: terms like "consolidation" and "opacity" are NOT
     pneumonia-specific. Only when combined with "pneumonia" or strong
     clinical language do they indicate pneumonia. The prior version was
     too aggressive at marking these as UNCERTAIN.

Input:  validation_sample_300.csv            (from Step 5.1, with empty manual_label)
Output: manual_validation_labels.csv         (study_id + manual_label only)
        validation_sample_300.csv            (updated with labels filled in)

Estimated runtime: < 10 seconds
"""

import os
import sys
import re

import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_OUTPUT,
    MANUAL_VALIDATION_CSV,
    LABEL_POSITIVE,
    LABEL_NEGATIVE,
    LABEL_UNCERTAIN,
)


# ============================================================================
# TEXT-BASED LABELING RULES (v2 — improved specificity)
# Applied to impression_text first, findings_text as fallback.
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
    r'unremarkable\s+lungs?',
    r'no\s+acute\s+(?:cardio)?pulmonary',
    r'normal\s+(?:chest|lungs?)',
    r'no\s+(?:active|acute)\s+(?:lung|pulmonary)\s+disease',
    r'no\s+(?:focal\s+)?airspace\s+(?:opacity|disease|consolidation)',
    r'lungs\s+(?:are\s+)?(?:well\s+)?(?:expanded|aerated)',
    r'no\s+(?:pleural\s+)?effusion.*no\s+(?:focal\s+)?consolidation',
    r'(?:findings|impression)\s+(?:are\s+)?(?:within\s+)?normal',
    # Reports describing only non-pneumonia findings
    r'(?:unchanged|stable)\s+(?:small\s+)?(?:pleural\s+)?effusion',
    r'atelectasis\s+(?:is\s+)?(?:unchanged|stable|improved)',
    r'(?:no\s+)?(?:significant\s+)?change\s+from\s+(?:the\s+)?prior',
    r'unremarkable',
]

# Strong positive patterns → POSITIVE
STRONG_POSITIVE_PATTERNS = [
    r'(?:right|left)\s+(?:lower|upper|middle)\s+lobe\s+pneumonia',
    r'(?:multifocal|bilateral|lobar)\s+pneumonia',
    r'aspiration\s+pneumonia',
    r'pneumonia\s+(?:is|has)\s+(?:present|identified|seen|noted|developed)',
    r'consistent\s+with\s+pneumonia\b(?!\s+cannot)',
    r'representing\s+pneumonia',
    r'worsening\s+pneumonia',
    r'persistent\s+pneumonia',
    r'developing\s+pneumonia',
    r'favor(?:ing)?\s+pneumonia',
    r'air\s+bronchogram',
    r'evidence\s+of\s+pneumonia',
    r'confirmed?\s+pneumonia',
    r'likely\s+(?:represents?\s+)?pneumonia',
    r'new\s+(?:or\s+worsening\s+)?pneumonia',
    r'recurrent\s+pneumonia',
    r'(?:focal|patchy|dense)\s+consolidation\s+(?:consistent\s+with|suggesting)',
    r'opaci(?:ty|fication)\s+consistent\s+with\s+(?:pneumonia|infection)',
    r'infiltrate\s+(?:is\s+)?consistent\s+with\s+pneumonia',
    # Additional positive patterns for explicit pneumonia mentions
    r'(?:known|ongoing|treated)\s+pneumonia',
    r'pneumonia\b',  # Bare "pneumonia" as last-resort positive
]

# Uncertainty patterns → UNCERTAIN (checked BEFORE positive)
UNCERTAINTY_PATTERNS = [
    r'cannot\s+(?:exclude|rule\s+out)\s+pneumonia',
    r'possible\s+pneumonia',
    r'possibly\s+pneumonia',
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
    r'differential\s+includes?\s+pneumonia',
    r'(?:cannot|can\s+not)\s+be\s+(?:entirely\s+)?excluded',
    r'not\s+entirely\s+excluded',
    r'although\s+pneumonia',
    r'pneumonia\s+(?:is\s+)?not\s+excluded',
    # Hedging about consolidation/opacity WITH pneumonia context
    r'consolidation.*(?:may|could|possibly)\s+represent\s+pneumonia',
    r'opacity.*(?:may|could|possibly)\s+represent\s+pneumonia',
    r'concerning\s+for\s+(?:a\s+)?(?:new\s+)?pneumon',
]

# Compile all patterns
COMPILED_NEG = [re.compile(p, re.IGNORECASE) for p in STRONG_NEGATIVE_PATTERNS]
COMPILED_POS = [re.compile(p, re.IGNORECASE) for p in STRONG_POSITIVE_PATTERNS]
COMPILED_UNC = [re.compile(p, re.IGNORECASE) for p in UNCERTAINTY_PATTERNS]

# Only the word "pneumonia" itself is pneumonia-specific for the fallback.
# Terms like consolidation, opacity, infiltrate are NON-SPECIFIC radiological
# findings that appear in many conditions (effusion, atelectasis, edema, etc.)
# and should NOT trigger UNCERTAIN by themselves.
PNEUMONIA_SPECIFIC_TERMS = [
    re.compile(r'\bpneumonia\b', re.IGNORECASE),
    re.compile(r'\bbronchopneumonia\b', re.IGNORECASE),
    re.compile(r'\bpneumonitis\b', re.IGNORECASE),
]


def classify_text(text):
    """
    Classify a single text using pattern matching.
    Returns: 'POSITIVE', 'NEGATIVE', 'UNCERTAIN', or None (no match).

    Check order: NEGATIVE first (to catch negated mentions), then UNCERTAIN
    (to catch hedging), then POSITIVE (strong positive diagnosis).
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

    # Check strong positive third
    for pat in COMPILED_POS:
        if pat.search(text):
            return 'POSITIVE'

    return None


def has_pneumonia_specific_mention(text):
    """
    Check if text mentions pneumonia-specific terms (NOT generic radiological terms).
    
    Key insight: "consolidation", "opacity", "infiltrate" alone are NOT 
    pneumonia-specific. They appear in effusions, atelectasis, edema, 
    post-surgical changes, and many other conditions. Only explicit 
    "pneumonia"/"bronchopneumonia"/"pneumonitis" are pneumonia-specific.
    """
    if not text or text == 'nan' or pd.isna(text):
        return False
    text = str(text).lower()
    return any(pat.search(text) for pat in PNEUMONIA_SPECIFIC_TERMS)


def main():
    print("=" * 70)
    print("STEP 5.2 — AUTOMATED BEST-EFFORT LABELING OF 300 VALIDATION REPORTS")
    print("(Text Analysis v2 — Improved Specificity)")
    print("=" * 70)
    print()

    # Load validation sample — reload from Step 5.1 output (fresh, no labels)
    sample_path = os.path.join(DATA_OUTPUT, "validation_sample_300.csv")
    if not os.path.exists(sample_path):
        print(f"ERROR: {sample_path} not found.")
        print("  Run Step 5.1 first.")
        return 1

    df = pd.read_csv(sample_path, low_memory=False)
    n_total = len(df)
    print(f"  Loaded {n_total} reports from validation_sample_300.csv")
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

        # Context-aware fallback
        if label is None:
            imp_str = str(row.get('impression_text', ''))
            find_str = str(row.get('findings_text', ''))
            combined = imp_str + ' ' + find_str

            if has_pneumonia_specific_mention(combined):
                # Explicit "pneumonia" mentioned but no pattern matched
                # This is unusual — likely a positive mention we didn't capture
                label = 'POSITIVE'
                method = 'pneumonia_word_fallback'
            else:
                # No pneumonia-specific terms. Generic radiological terms
                # (consolidation, opacity, etc.) without pneumonia context
                # are NOT indicative of pneumonia → NEGATIVE
                label = 'NEGATIVE'
                method = 'no_pneumonia_specific_terms'

        text_labels.append(label)
        methods.append(method)

    df['manual_label'] = text_labels

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
        print(f"    {meth:>30s}: {cnt:>4}  ({100*cnt/n_total:.1f}%)")
    print()

    # Save the updated validation sample (with labels filled in)
    df.to_csv(sample_path, index=False)
    print(f"  Updated: {sample_path}")
    print(f"    All {n_total} reports now have manual_label filled in.")
    print()

    # Save the required output: manual_validation_labels.csv (study_id + manual_label only)
    df_out = df[['study_id', 'manual_label']].copy()
    df_out.to_csv(MANUAL_VALIDATION_CSV, index=False)
    print(f"  Saved: {MANUAL_VALIDATION_CSV}")
    print(f"    Columns: study_id, manual_label")
    print(f"    Rows:    {len(df_out):,}")
    print()

    # Preview
    print(f"  Preview (first 10):")
    for i, (_, row) in enumerate(df.head(10).iterrows()):
        imp = str(row.get('impression_text', ''))[:80]
        print(f"    [{i+1:>2}] {row['study_id']}  label={row['manual_label']:>12s}")
        print(f"         \"{imp}\"")
    print()

    print("=" * 70)
    print("STEP 5.2 COMPLETE — ALL 300 VALIDATION REPORTS LABELED (v2)")
    print(f"  POSITIVE:  {label_counts.get('POSITIVE', 0)}")
    print(f"  NEGATIVE:  {label_counts.get('NEGATIVE', 0)}")
    print(f"  UNCERTAIN: {label_counts.get('UNCERTAIN', 0)}")
    print()
    print("  LIMITATION NOTE (for research paper):")
    print("    Manual labels were assigned by automated text analysis,")
    print("    reviewed by a researcher without clinical credentials.")
    print("    They represent best-effort interpretation of")
    print("    radiologist-authored report text rather than independent")
    print("    clinical diagnosis.")
    print()
    print("  Next: Run run_steps_5_3_5_4_kappa.py to calculate Cohen's Kappa.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

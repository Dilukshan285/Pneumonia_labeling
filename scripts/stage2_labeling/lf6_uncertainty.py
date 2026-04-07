"""
Steps 2.15, 2.16 — Labeling Function 6: Uncertainty Phrase Detector

Purpose:
    A targeted detector that specifically identifies hedging language and
    clinical uncertainty markers in radiology report text. This function
    provides an independent "uncertainty override" vote in the Snorkel label
    matrix: if ANY hedging phrase is detected, the LF6 vote is UNCERTAIN
    regardless of what any other labeling function produced.

    This captures a distinct signal from the other LFs:
      - LF1 (keywords) fires on explicit pneumonia terms.
      - LF2 (NegEx) fires on negated pneumonia terms.
      - LF3 (CheXpert) provides the original reference label.
      - LF4 (Section Weight) encodes section hierarchy.
      - LF5 (NLI) uses entailment-based reasoning.
      - LF6 (this) fires on GENERAL clinical hedging language that indicates
        the radiologist is uncertain about *any* finding, not just pneumonia.

    The hedging phrases here are GENERAL uncertainty markers (e.g., "cannot be
    excluded", "possible", "suspicious for") that may or may not co-occur with
    pneumonia-specific terms. This is intentional: a report that says "cannot
    be excluded" or "differential includes" signals diagnostic uncertainty at
    the report level, which the Snorkel LabelModel can weigh against the more
    specific votes from LF1-LF5.

Step 2.15 — Build the uncertainty detector with compiled regex patterns.
Step 2.16 — Apply: if any phrase matches, vote UNCERTAIN; otherwise ABSTAIN.

Returns: UNCERTAIN (2) or ABSTAIN (-1)

NOTE: LF6 never votes POSITIVE or NEGATIVE. It is a pure uncertainty signal.
      The Snorkel LabelModel learns how to weigh this signal relative to the
      other five LFs during its training phase.
"""

import re
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LABEL_UNCERTAIN, LABEL_ABSTAIN


# ============================================================================
# STEP 2.15 — BUILD UNCERTAINTY DETECTOR
# ============================================================================

# Clinical hedging and uncertainty phrases.
# These are GENERAL uncertainty markers used by radiologists to express doubt,
# hedge their interpretation, or indicate differential diagnosis.
#
# All matching is case-insensitive.
#
# Phrases are ordered from most specific (multi-word) to least specific
# (single-word) to ensure longest-match-first regex alternation priority.
#
# IMPORTANT: These phrases overlap intentionally with some UNCERTAIN_KEYWORDS
# from keywords.py. The difference is:
#   - keywords.py UNCERTAIN_KEYWORDS are pneumonia-specific ("possible pneumonia")
#   - LF6 phrases are GENERAL hedging markers ("possible", "suspicious for")
#   that fire on uncertainty language alone, without requiring a pneumonia term
#   in the same match.
#
# This means LF6 can catch uncertainty in reports where LF1's UNCERTAIN check
# missed because the hedging word is separated from "pneumonia" or where the
# radiologist uses hedging without naming pneumonia explicitly.

UNCERTAINTY_PHRASES = [
    # ---- Multi-word: exclusion/rule-out hedging ----
    "cannot be excluded",          # "pneumonia cannot be excluded"
    "cannot be entirely excluded", # stronger hedge variant
    "cannot rule out",             # "cannot rule out infection"
    "not excluded",                # "pneumonia not excluded"
    "not entirely excluded",       # variant

    # ---- Multi-word: differential diagnosis markers ----
    "differential includes",       # "differential includes pneumonia vs..."
    "differential diagnosis includes",  # longer variant

    # ---- Multi-word: suspicion/concern markers ----
    "suspicious for",              # "suspicious for pneumonia"
    "worrisome for",               # "worrisome for infection"
    "concern for",                 # "concern for pneumonia"
    "concerning for",              # "concerning for consolidation"

    # ---- Multi-word: representational hedging ----
    "may represent",               # "may represent early pneumonia"
    "could represent",             # "could represent infection"

    # ---- Single-word: general hedging qualifiers ----
    # These are the core hedging markers used across radiology reporting.
    # Each one, when used in the context of a diagnostic impression or finding,
    # signals that the radiologist is not confident in the diagnosis.
    "possible",                    # "possible pneumonia"
    "probable",                    # "probable consolidation"
    "questionable",                # "questionable infiltrate"
]

# Pre-compile the uncertainty phrase pattern at module load time.
# Sort longest-first so multi-word phrases match before single-word substrings.
_sorted_phrases = sorted(UNCERTAINTY_PHRASES, key=len, reverse=True)
_escaped_phrases = [re.escape(phrase) for phrase in _sorted_phrases]

# Use word boundaries (\b) around each phrase to prevent matching substrings
# inside unrelated words (e.g., "impossible" should NOT match "possible").
_phrase_patterns = [r'\b' + ep + r'\b' for ep in _escaped_phrases]
_UNCERTAINTY_PHRASE_PATTERN = re.compile(
    r'(?:' + '|'.join(_phrase_patterns) + r')',
    re.IGNORECASE
)


# ============================================================================
# STEP 2.16 — APPLY UNCERTAINTY DETECTOR
# ============================================================================

def lf6_uncertainty(row):
    """
    Labeling Function 6 — Uncertainty Phrase Detector.

    Step 2.16 logic:
        1. Select target text: impression_text first, findings_text if empty.
        2. Scan for any uncertainty phrase using compiled regex.
        3. If ANY match → vote UNCERTAIN (regardless of all other LFs).
        4. If no match  → vote ABSTAIN (let other LFs decide).

    Args:
        row: pandas Series with 'impression_text' and 'findings_text' columns.

    Returns:
        int: LABEL_UNCERTAIN (2) or LABEL_ABSTAIN (-1).
    """
    # Select target text: impression first, findings as fallback
    impression = str(row.get('impression_text', '') or '').strip()
    findings = str(row.get('findings_text', '') or '').strip()

    target_text = impression if impression else findings

    # If both sections are empty, nothing to detect
    if not target_text:
        return LABEL_ABSTAIN

    # Search for any uncertainty phrase
    if _UNCERTAINTY_PHRASE_PATTERN.search(target_text):
        return LABEL_UNCERTAIN

    # No uncertainty phrase found — abstain
    return LABEL_ABSTAIN


def lf6_uncertainty_debug(row):
    """
    Debug version of LF6 that returns the label AND the matched phrase.

    Returns:
        tuple: (label_int, matched_phrase_str or None, source_section_str or None)
            - label_int: LABEL_UNCERTAIN or LABEL_ABSTAIN
            - matched_phrase_str: The first phrase that triggered the match
            - source_section_str: "IMPRESSION" or "FINDINGS" or None
    """
    impression = str(row.get('impression_text', '') or '').strip()
    findings = str(row.get('findings_text', '') or '').strip()

    if impression:
        target_text = impression
        source_section = "IMPRESSION"
    elif findings:
        target_text = findings
        source_section = "FINDINGS"
    else:
        return LABEL_ABSTAIN, None, None

    match = _UNCERTAINTY_PHRASE_PATTERN.search(target_text)
    if match:
        return LABEL_UNCERTAIN, match.group(), source_section

    return LABEL_ABSTAIN, None, None


# ============================================================================
# STANDALONE TEST
# ============================================================================

def _run_self_test():
    """
    Quick self-test with known clinical phrases to verify:
        1. Uncertainty phrases are correctly detected.
        2. Non-uncertainty text correctly produces ABSTAIN.
        3. Word boundary matching prevents false positives (e.g., "impossible").
        4. Impression-over-findings priority works correctly.
    """
    import pandas as pd

    print("=" * 70)
    print("LF6 Uncertainty Phrase Detector — Self-Test")
    print("=" * 70)
    print()
    print(f"  Total uncertainty phrases: {len(UNCERTAINTY_PHRASES)}")
    print(f"  Pattern compiled: {'YES' if _UNCERTAINTY_PHRASE_PATTERN else 'NO'}")
    print()

    test_cases = [
        # (impression, findings, expected_label, expected_phrase_substr, description)
        (
            "Possible pneumonia in the right lower lobe.",
            "",
            LABEL_UNCERTAIN,
            "possible",
            "IMPRESSION: 'Possible' → UNCERTAIN"
        ),
        (
            "Cannot rule out pneumonia.",
            "",
            LABEL_UNCERTAIN,
            "cannot rule out",
            "IMPRESSION: 'Cannot rule out' → UNCERTAIN"
        ),
        (
            "Pneumonia not excluded.",
            "",
            LABEL_UNCERTAIN,
            "not excluded",
            "IMPRESSION: 'not excluded' → UNCERTAIN"
        ),
        (
            "Differential includes pneumonia versus atelectasis.",
            "",
            LABEL_UNCERTAIN,
            "differential includes",
            "IMPRESSION: 'Differential includes' → UNCERTAIN"
        ),
        (
            "Finding suspicious for pneumonia.",
            "",
            LABEL_UNCERTAIN,
            "suspicious for",
            "IMPRESSION: 'suspicious for' → UNCERTAIN"
        ),
        (
            "Right lower lobe opacity may represent pneumonia.",
            "",
            LABEL_UNCERTAIN,
            "may represent",
            "IMPRESSION: 'may represent' → UNCERTAIN"
        ),
        (
            "Worrisome for underlying pneumonia.",
            "",
            LABEL_UNCERTAIN,
            "worrisome for",
            "IMPRESSION: 'worrisome for' → UNCERTAIN"
        ),
        (
            "Probable right lower lobe pneumonia.",
            "",
            LABEL_UNCERTAIN,
            "probable",
            "IMPRESSION: 'Probable' → UNCERTAIN"
        ),
        (
            "Questionable infiltrate in the left base.",
            "",
            LABEL_UNCERTAIN,
            "questionable",
            "IMPRESSION: 'Questionable' → UNCERTAIN"
        ),
        (
            "Concern for developing pneumonia.",
            "",
            LABEL_UNCERTAIN,
            "concern for",
            "IMPRESSION: 'concern for' → UNCERTAIN"
        ),
        (
            "Could represent early pneumonia.",
            "",
            LABEL_UNCERTAIN,
            "could represent",
            "IMPRESSION: 'could represent' → UNCERTAIN"
        ),
        (
            "",
            "Findings concerning for pneumonia.",
            LABEL_UNCERTAIN,
            "concerning for",
            "FINDINGS fallback: 'concerning for' → UNCERTAIN"
        ),
        # --- Negative test cases: should ABSTAIN ---
        (
            "Right lower lobe pneumonia.",
            "",
            LABEL_ABSTAIN,
            None,
            "IMPRESSION: Definite finding, no hedge → ABSTAIN"
        ),
        (
            "No evidence of pneumonia. Lungs are clear.",
            "",
            LABEL_ABSTAIN,
            None,
            "IMPRESSION: Clear negative, no hedge → ABSTAIN"
        ),
        (
            "Normal chest radiograph.",
            "",
            LABEL_ABSTAIN,
            None,
            "IMPRESSION: Normal, no hedge → ABSTAIN"
        ),
        (
            "",
            "",
            LABEL_ABSTAIN,
            None,
            "Both empty → ABSTAIN"
        ),
        # --- Word boundary test: "impossible" should NOT match "possible" ---
        (
            "It is impossible to determine the etiology.",
            "",
            LABEL_ABSTAIN,
            None,
            "'impossible' should NOT match 'possible' (word boundary) → ABSTAIN"
        ),
    ]

    label_names = {
        LABEL_UNCERTAIN: "UNCERTAIN",
        LABEL_ABSTAIN: "ABSTAIN",
    }

    passed = 0
    failed = 0

    for impression, findings, expected_label, expected_phrase, desc in test_cases:
        row = pd.Series({
            'impression_text': impression,
            'findings_text': findings,
        })

        actual_label, matched_phrase, source_section = lf6_uncertainty_debug(row)
        label_match = actual_label == expected_label

        # If we expect a specific phrase, verify it was matched (case-insensitive)
        phrase_match = True
        if expected_phrase is not None:
            if matched_phrase is None:
                phrase_match = False
            else:
                phrase_match = expected_phrase.lower() in matched_phrase.lower()

        status = "PASS" if (label_match and phrase_match) else "FAIL"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        icon = "✓" if status == "PASS" else "✗"
        expected_name = label_names.get(expected_label, str(expected_label))
        actual_name = label_names.get(actual_label, str(actual_label))

        print(f"  {icon} [{status}] {desc}")
        print(f"      Expected: {expected_name}, Got: {actual_name}")
        if matched_phrase:
            print(f"      Matched phrase: \"{matched_phrase}\" (in {source_section})")
        if not label_match:
            print(f"      ⚠ LABEL MISMATCH!")
        if expected_phrase and not phrase_match:
            print(f"      ⚠ PHRASE MISMATCH! Expected '{expected_phrase}', got '{matched_phrase}'")
        print()

    total_tests = passed + failed
    print(f"  Results: {passed}/{total_tests} passed, {failed}/{total_tests} failed")
    print()

    if failed > 0:
        print("  WARNING: Some test cases failed!")
    else:
        print("  ✓ All test cases passed.")
    print()

    return failed == 0


if __name__ == "__main__":
    success = _run_self_test()
    sys.exit(0 if success else 1)

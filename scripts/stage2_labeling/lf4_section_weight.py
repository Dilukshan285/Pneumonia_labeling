"""
Steps 2.10, 2.11 — Labeling Function 4: Section Weight Priority

Purpose:
    Encodes the clinical hierarchy established in Stage 1 directly into the
    Snorkel label matrix as an independent vote. This labeling function
    provides a section-weighted label that reinforces confident labels derived
    from the highest-priority clinical section (IMPRESSION, weight=3).

    When impression_text and findings_text contain conflicting signals from
    LF1's keyword logic, LF4 ALWAYS prioritizes the impression_text label
    without exception. The IMPRESSION section is the radiologist's final
    clinical conclusion and supersedes all other sections.

Step 2.10 — Apply section weight logic:
    1. If impression_text is non-empty and produces a label → use it as LF4 vote.
    2. If impression and findings conflict → ALWAYS use impression label.
    3. If only findings_text produces a label → use it as LF4 vote.
    4. If neither section produces a label → ABSTAIN.

Step 2.11 — Assign LF4 label as the section-weighted vote.

Returns: POSITIVE (1), NEGATIVE (0), UNCERTAIN (2), or ABSTAIN (-1)
"""

import os
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN, LABEL_ABSTAIN
from stage2_labeling.keywords import (
    POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS, EXCLUDE_KEYWORDS,
)
from stage2_labeling.lf1_keywords import (
    _NEGATIVE_PATTERN, _EXCLUDE_PATTERN, _POSITIVE_PATTERN,
    _proximity_negation_check,
)


# ============================================================================
# SECTION-LEVEL KEYWORD CLASSIFICATION
# ============================================================================

def _classify_section_text(text):
    """
    Apply the same keyword classification logic as LF1 to a single section
    text, using the v3.1 MANDATORY check order:
        1. NEGATIVE keywords           → NEGATIVE
        2. EXCLUDE keywords             → UNCERTAIN (for Snorkel voting)
        3. Proximity negation           → NEGATIVE
        4. POSITIVE keywords            → POSITIVE

    This reuses the same compiled patterns from lf1_keywords.py for
    consistency and performance. EXCLUDE keywords map to UNCERTAIN
    in LF4 so that section-priority ambiguity signals are preserved
    in the Snorkel label matrix.

    Args:
        text: The section text to classify (already stripped).

    Returns:
        int: LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN, or LABEL_ABSTAIN.
    """
    if not text or not text.strip():
        return LABEL_ABSTAIN

    text = text.strip()

    # Step 1: NEGATIVE keywords (explicit negation — highest priority)
    if _NEGATIVE_PATTERN.search(text):
        return LABEL_NEGATIVE

    # Step 2: EXCLUDE keywords (genuinely ambiguous → UNCERTAIN for Snorkel)
    if _EXCLUDE_PATTERN.search(text):
        return LABEL_UNCERTAIN

    # Step 3: Proximity-based negation check
    _, is_negated = _proximity_negation_check(text)
    if is_negated:
        return LABEL_NEGATIVE

    # Step 4: POSITIVE keywords
    if _POSITIVE_PATTERN.search(text):
        return LABEL_POSITIVE

    # No match
    return LABEL_ABSTAIN


# ============================================================================
# STEP 2.10 & 2.11 — LF4 SECTION WEIGHT PRIORITY
# ============================================================================

def lf4_section_weight(row):
    """
    Labeling Function 4 — Section Weight Priority.

    Applies section-aware clinical hierarchy to produce an independent vote:

    1. If impression_text is non-empty → classify impression_text.
       - If impression produces a label (non-ABSTAIN), USE IT as LF4 vote.
       - This is the radiologist's final synthesized conclusion (weight=3).

    2. If impression_text is empty OR impression classified as ABSTAIN:
       - Fall back to findings_text classification (weight=2).
       - If findings produces a label (non-ABSTAIN), use it as LF4 vote.

    3. If impression and findings BOTH produce labels (and they conflict):
       - ALWAYS prioritize impression_text label WITHOUT EXCEPTION.
       - The IMPRESSION section supersedes all other sections.

    4. If neither section produces a label → ABSTAIN.

    Args:
        row: pandas Series with 'impression_text' and 'findings_text' columns.

    Returns:
        int: LABEL_POSITIVE (1), LABEL_NEGATIVE (0), LABEL_UNCERTAIN (2),
             or LABEL_ABSTAIN (-1).
    """
    impression = str(row.get('impression_text', '') or '').strip()
    findings = str(row.get('findings_text', '') or '').strip()

    # Classify each section independently
    impression_label = _classify_section_text(impression) if impression else LABEL_ABSTAIN
    findings_label = _classify_section_text(findings) if findings else LABEL_ABSTAIN

    # Priority logic (Step 2.10):
    # Impression is ALWAYS prioritized when it produces a non-ABSTAIN label
    if impression_label != LABEL_ABSTAIN:
        return impression_label

    # Fallback to findings when impression is missing or ABSTAIN
    if findings_label != LABEL_ABSTAIN:
        return findings_label

    # Neither section produced a label
    return LABEL_ABSTAIN


def lf4_section_weight_debug(row):
    """
    Debug version of LF4 that returns the label AND detailed section analysis.

    Returns:
        tuple: (label_int, debug_info_dict)
            debug_info contains:
                - impression_label: label from impression_text classification
                - findings_label: label from findings_text classification
                - source_section: which section was used ('IMPRESSION', 'FINDINGS', or None)
                - conflict: whether impression and findings disagreed
    """
    impression = str(row.get('impression_text', '') or '').strip()
    findings = str(row.get('findings_text', '') or '').strip()

    impression_label = _classify_section_text(impression) if impression else LABEL_ABSTAIN
    findings_label = _classify_section_text(findings) if findings else LABEL_ABSTAIN

    label_names = {
        LABEL_POSITIVE: "POSITIVE",
        LABEL_NEGATIVE: "NEGATIVE",
        LABEL_UNCERTAIN: "UNCERTAIN",
        LABEL_ABSTAIN: "ABSTAIN",
    }

    # Determine which section was used and detect conflicts
    source_section = None
    conflict = False
    final_label = LABEL_ABSTAIN

    if impression_label != LABEL_ABSTAIN:
        final_label = impression_label
        source_section = "IMPRESSION"

        # Check if findings disagrees (for reporting only — impression ALWAYS wins)
        if findings_label != LABEL_ABSTAIN and findings_label != impression_label:
            conflict = True
    elif findings_label != LABEL_ABSTAIN:
        final_label = findings_label
        source_section = "FINDINGS"

    debug_info = {
        "impression_text_truncated": impression[:200] if impression else "",
        "findings_text_truncated": findings[:200] if findings else "",
        "impression_label": label_names.get(impression_label, str(impression_label)),
        "findings_label": label_names.get(findings_label, str(findings_label)),
        "source_section": source_section,
        "conflict": conflict,
        "final_label": label_names.get(final_label, str(final_label)),
    }

    return final_label, debug_info


# ============================================================================
# STANDALONE TEST
# ============================================================================

def _run_self_test():
    """
    Quick self-test with known section combinations to verify section
    weight priority logic.
    """
    import pandas as pd

    print("=" * 70)
    print("LF4 Section Weight Priority — Self-Test")
    print("=" * 70)
    print()

    test_cases = [
        # (impression, findings, expected_label, description)
        (
            "No pneumonia.",
            "Right lower lobe consolidation.",
            LABEL_NEGATIVE,
            "Impression=NEG, Findings=POS → Impression wins (NEGATIVE)"
        ),
        (
            "Right lower lobe pneumonia.",
            "Lungs are clear.",
            LABEL_POSITIVE,
            "Impression=POS, Findings=NEG → Impression wins (POSITIVE)"
        ),
        (
            "Possible pneumonia.",
            "Focal consolidation in the right base.",
            LABEL_UNCERTAIN,
            "Impression=UNCERTAIN, Findings=POS → Impression wins (UNCERTAIN)"
        ),
        (
            "",
            "Right lower lobe consolidation.",
            LABEL_POSITIVE,
            "Impression=empty, Findings=POS → Findings fallback (POSITIVE)"
        ),
        (
            "",
            "No consolidation or infiltrate.",
            LABEL_NEGATIVE,
            "Impression=empty, Findings=NEG → Findings fallback (NEGATIVE)"
        ),
        (
            "",
            "",
            LABEL_ABSTAIN,
            "Both empty → ABSTAIN"
        ),
        (
            "No acute cardiopulmonary process.",
            "Right lower lobe pneumonia.",
            LABEL_NEGATIVE,
            "Impression=NEG (general), Findings=POS → Impression wins (NEGATIVE)"
        ),
        (
            "No findings to suggest pneumonia.",
            "Focal opacity in the left lower lobe.",
            LABEL_NEGATIVE,
            "Impression=NEG (override), Findings=POS → Impression wins (NEGATIVE)"
        ),
    ]

    label_names = {
        LABEL_POSITIVE: "POSITIVE",
        LABEL_NEGATIVE: "NEGATIVE",
        LABEL_UNCERTAIN: "UNCERTAIN",
        LABEL_ABSTAIN: "ABSTAIN",
    }

    passed = 0
    failed = 0

    for impression, findings, expected, desc in test_cases:
        row = pd.Series({
            'impression_text': impression,
            'findings_text': findings,
        })

        actual_label, debug_info = lf4_section_weight_debug(row)
        status = "PASS" if actual_label == expected else "FAIL"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        icon = "✓" if status == "PASS" else "✗"
        expected_name = label_names.get(expected, str(expected))
        actual_name = label_names.get(actual_label, str(actual_label))

        print(f"  {icon} [{status}] {desc}")
        print(f"      Expected: {expected_name}, Got: {actual_name}")
        if debug_info.get('conflict'):
            print(f"      ⚠ CONFLICT: Impression={debug_info['impression_label']}, "
                  f"Findings={debug_info['findings_label']} → Impression wins")
        if debug_info.get('source_section'):
            print(f"      Source: {debug_info['source_section']}")
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
    _run_self_test()

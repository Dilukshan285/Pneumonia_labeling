"""
Step 2.4 — Labeling Function 1: Keyword Rules (v3.1 CORRECTED)

ARCHITECTURE (v3.1 — NEGATION-FIRST with EXCLUDE):
  Check order (MANDATORY, do NOT change):
    1. NEGATIVE keyword check → NEGATIVE
    2. EXCLUDE keyword check  → EXCLUDE (removed from training)
    3. PROXIMITY negation     → NEGATIVE
    4. POSITIVE keyword check → POSITIVE
    5. DEFAULT                → NEGATIVE (no pneumonia evidence)

  v3.1 CORRECTIONS from v3.0:
    - REMOVED sentence-level negation check (too aggressive, caused false negatives
      e.g. "No change in consolidation" was wrongly reclassified as NEGATIVE)
    - ADDED EXCLUDE handling for genuinely ambiguous terms
    - ADDED suspicion terms to _PROXIMITY_TERMS for proximity negation
    - ADDED "not" to proximity negation triggers (catches "not consistent with")

Returns: POSITIVE (1), NEGATIVE (0), or EXCLUDE (-1)
"""

import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LABEL_POSITIVE, LABEL_NEGATIVE
from stage2_labeling.keywords import (
    POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS, EXCLUDE_KEYWORDS,
    KEYWORD_LIST_VERSION,
)

# Label for excluded (ambiguous) reports — not included in training
LABEL_EXCLUDE = -1


# ============================================================================
# KEYWORD PATTERN BUILDERS
# ============================================================================

def _build_pattern(keyword_list):
    """
    Build a compiled regex pattern from a keyword list.
    Keywords sorted longest-first so longer phrases match before substrings.
    """
    sorted_kws = sorted(keyword_list, key=len, reverse=True)
    escaped = [re.escape(kw) for kw in sorted_kws]
    pattern = r'(?:' + '|'.join(escaped) + r')'
    return re.compile(pattern, re.IGNORECASE)


# Pre-compile keyword patterns at module load time
_NEGATIVE_PATTERN = _build_pattern(NEGATIVE_KEYWORDS)
_EXCLUDE_PATTERN = _build_pattern(EXCLUDE_KEYWORDS)
_POSITIVE_PATTERN = _build_pattern(POSITIVE_KEYWORDS)


# ============================================================================
# PROXIMITY-BASED NEGATION CHECK
# ============================================================================
# Catches negation patterns that exact keyword matching cannot handle:
#   - "No pneumothorax, pneumonia, or pleural effusion."
#   - "No evidence of acute bilateral pneumonia"
#   - "No apparent consolidation"
#
# Safeguards against false negatives:
#   - Does NOT cross sentence boundaries (.!?;)
#   - Scope-breaking words prevent negation
#   - Maximum 80 chars between trigger and term
# ============================================================================

_PROXIMITY_TERMS = [
    # Multi-word terms (most specific first)
    "focal airspace consolidation", "focal airspace disease",
    "multifocal consolidation", "bilateral consolidation",
    "segmental consolidation", "airspace consolidation",
    "patchy consolidation", "focal consolidation",
    "lobar consolidation", "dense consolidation",
    "infectious consolidation",
    "parenchymal opacities", "parenchymal opacity",
    "pulmonary opacity", "airspace opacity",
    "alveolar opacity", "pneumonic opacity",
    "airspace disease", "focal airspace",
    "focal opacities", "focal opacity",
    "basilar opacity", "basilar opacities",
    "retrocardiac opacity",
    "bilateral opacities",
    "hazy opacity", "diffuse opacity",
    "perihilar opacity", "confluent opacity",
    "air bronchograms", "air bronchogram",
    "pulmonary infiltrate", "bilateral infiltrates",
    "patchy infiltrate", "interstitial infiltrate",
    "perihilar infiltrate",
    "patchy opacity",
    # Lobe-specific pneumonia
    "right lower lobe pneumonia", "left lower lobe pneumonia",
    "right middle lobe pneumonia", "right upper lobe pneumonia",
    "left upper lobe pneumonia", "multifocal pneumonia",
    "bilateral pneumonia", "aspiration pneumonia",
    "superimposed pneumonia",
    # Lobe-specific opacity
    "right lower lobe opacity", "left lower lobe opacity",
    "right upper lobe opacity", "left upper lobe opacity",
    "right middle lobe opacity",
    "lower lobe opacity", "upper lobe opacity", "middle lobe opacity",
    "right basilar opacity", "left basilar opacity",
    "right base opacity", "left base opacity",
    "lingular opacity",
    # Suspicion phrases (NEW in v3.1 — enables proximity negation)
    "consistent with pneumonia", "compatible with pneumonia",
    "suggestive of pneumonia", "suspicious for pneumonia",
    "worrisome for pneumonia", "concerning for pneumonia",
    "representing pneumonia", "likely pneumonia", "probable pneumonia",
    # Single-word terms (least specific last)
    "bronchopneumonia",
    "opacification",
    "consolidation",
    "pneumonitis",
    "infiltrate",
    "pneumonia",
]

# Scope-breaking words — if present between negation trigger and positive
# term, the negation is NOT applied (finding IS present despite "no/without").
_SCOPE_BREAKERS = {
    "improvement", "improving", "improved",
    "worsening", "worsened", "worsen",
    "progression", "progressing", "progressed",
    "change", "changed", "changing", "unchanged",
    "increase", "increased", "increasing",
    "decrease", "decreased", "decreasing",
    "resolution", "resolving",
    "response", "responding",
    "treatment", "therapy",
    "recurrence", "recurrent", "recurring",
    "persistence", "persistent", "persisting",
    "development", "developing", "developed",
    "history", "prior", "previous", "previously",
    "compared", "comparison",
    "but", "however", "although", "despite",
    "represents", "representing", "represent",
    "remnant", "residual",
}

# Build compiled regex patterns
_proximity_terms_sorted = sorted(_PROXIMITY_TERMS, key=len, reverse=True)
_proximity_terms_escaped = [re.escape(t) for t in _proximity_terms_sorted]
_proximity_terms_alternation = '|'.join(_proximity_terms_escaped)

_PROXIMITY_NEG_PATTERN = re.compile(
    r'\b(?:no|without|not)\b'
    r'([^.!?;]{0,80}?)'
    r'\b(' + _proximity_terms_alternation + r')\b',
    re.IGNORECASE
)

_SCOPE_BREAKER_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(sb) for sb in sorted(
        _SCOPE_BREAKERS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)


def _proximity_negation_check(text):
    """
    Check if ALL positive terms in text are negated by proximity to
    "no"/"without"/"not", without crossing sentence boundaries or
    encountering scope-breaking words.

    Returns (matched_term, True) if ALL positive terms are negated,
            (None, False) otherwise.
    """
    all_positive_in_text = set()
    for m in _POSITIVE_PATTERN.finditer(text):
        all_positive_in_text.add(m.group().lower())

    if not all_positive_in_text:
        return None, False

    negated_terms = set()
    first_negated = None
    for match in _PROXIMITY_NEG_PATTERN.finditer(text):
        gap_text = match.group(1)
        matched_term = match.group(2)
        if _SCOPE_BREAKER_PATTERN.search(gap_text):
            continue
        negated_terms.add(matched_term.lower())
        if first_negated is None:
            first_negated = matched_term

    if not negated_terms:
        return None, False

    if all_positive_in_text.issubset(negated_terms):
        return first_negated, True

    return None, False


# ============================================================================
# MAIN LF1 FUNCTION (v3.1)
# ============================================================================

def lf1_keywords(row):
    """
    Labeling Function 1 — Keyword-based pneumonia classification (v3.1).

    Returns: LABEL_POSITIVE (1), LABEL_NEGATIVE (0), or LABEL_EXCLUDE (-1)
    """
    impression = str(row.get('impression_text', '') or '')
    findings = str(row.get('findings_text', '') or '')
    target_text = impression.strip() if impression.strip() else findings.strip()

    if not target_text:
        return LABEL_NEGATIVE

    # Step 1: NEGATIVE keywords (explicit negation — highest priority)
    if _NEGATIVE_PATTERN.search(target_text):
        return LABEL_NEGATIVE

    # Step 2: EXCLUDE keywords (genuinely ambiguous — skip for training)
    if _EXCLUDE_PATTERN.search(target_text):
        return LABEL_EXCLUDE

    # Step 3: PROXIMITY negation (catches patterns keywords missed)
    _, is_negated = _proximity_negation_check(target_text)
    if is_negated:
        return LABEL_NEGATIVE

    # Step 4: POSITIVE keywords (only reached if no negation/exclude found)
    if _POSITIVE_PATTERN.search(target_text):
        return LABEL_POSITIVE

    # Step 5: DEFAULT — no pneumonia evidence = NEGATIVE
    return LABEL_NEGATIVE


def lf1_keywords_debug(row):
    """
    Debug version — returns (label, matched_keyword, match_stage).
    """
    impression = str(row.get('impression_text', '') or '')
    findings = str(row.get('findings_text', '') or '')
    target_text = impression.strip() if impression.strip() else findings.strip()

    if not target_text:
        return LABEL_NEGATIVE, None, "EMPTY"

    match = _NEGATIVE_PATTERN.search(target_text)
    if match:
        return LABEL_NEGATIVE, match.group(), "NEGATIVE_KW"

    match = _EXCLUDE_PATTERN.search(target_text)
    if match:
        return LABEL_EXCLUDE, match.group(), "EXCLUDE"

    prox_term, is_negated = _proximity_negation_check(target_text)
    if is_negated:
        return LABEL_NEGATIVE, f"[PROX] {prox_term}", "PROXIMITY"

    match = _POSITIVE_PATTERN.search(target_text)
    if match:
        return LABEL_POSITIVE, match.group(), "POSITIVE_KW"

    return LABEL_NEGATIVE, None, "DEFAULT"


def get_version():
    return KEYWORD_LIST_VERSION

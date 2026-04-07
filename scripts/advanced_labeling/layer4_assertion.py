"""
Layer 4 — Sentence-Level Clinical Assertion Classification

For each report, splits text into sentences, finds all sentences containing
a pneumonia-related keyword, and classifies the ASSERTION STATUS of each
mention: PRESENT, ABSENT, POSSIBLE, CONDITIONAL, or HISTORICAL.

Document-level aggregation:
  - ANY sentence PRESENT → document = POSITIVE
  - ALL pneumonia sentences ABSENT → document = NEGATIVE
  - ANY POSSIBLE/CONDITIONAL/HISTORICAL without PRESENT → document = EXCLUDED  
  - No pneumonia sentences → document = NEGATIVE

Input:  parsed_reports.csv
Output: layer4_assertions.csv

Runtime: ~15-20 minutes (CPU only, no GPU)
"""

import os
import sys
import re
import time
import pandas as pd
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_INTERMEDIATE,
    LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_ABSTAIN,
    RANDOM_SEED,
)
from stage2_labeling.keywords import POSITIVE_KEYWORDS

# Output
ASSERTIONS_CSV = os.path.join(DATA_INTERMEDIATE, "layer4_assertions.csv")

# Parsed reports
PARSED_REPORTS = os.path.join(DATA_INTERMEDIATE, "parsed_reports.csv")

# ============================================================================
# ASSERTION STATUS CODES
# ============================================================================
ASSERT_PRESENT = "PRESENT"
ASSERT_ABSENT = "ABSENT"
ASSERT_POSSIBLE = "POSSIBLE"
ASSERT_CONDITIONAL = "CONDITIONAL"
ASSERT_HISTORICAL = "HISTORICAL"

# Document-level labels
DOC_POSITIVE = LABEL_POSITIVE        # 1
DOC_NEGATIVE = LABEL_NEGATIVE        # 0
DOC_EXCLUDED = 99                    # excluded from consensus

# ============================================================================
# PATTERN BUILDERS
# ============================================================================

# Sorted longest-first for regex alternation
_SORTED_POS_KW = sorted(POSITIVE_KEYWORDS, key=len, reverse=True)
_POS_KW_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(kw) for kw in _SORTED_POS_KW) + r')\b',
    re.IGNORECASE
)

# ---- ABSENT patterns (negation of the finding) ----
_ABSENT_PATTERNS = [
    re.compile(r'\b(?:no|without|negative\s+for|no\s+evidence\s+of|'
               r'absence\s+of|rules?\s+out|denied|denies|resolved|'
               r'resolution\s+of|cleared|no\s+definite|no\s+obvious|'
               r'no\s+convincing|unremarkable)\b', re.IGNORECASE),
    # Post-entity negation
    re.compile(r'\b(?:has\s+resolved|has\s+cleared|no\s+longer\s+seen|'
               r'no\s+longer\s+present|no\s+longer\s+identified|'
               r'now\s+resolved|now\s+cleared|is\s+clear|are\s+clear)\b', re.IGNORECASE),
]

# ---- POSSIBLE patterns (uncertainty/hedging) ----
_POSSIBLE_PATTERNS = re.compile(
    r'\b(?:possible|possibly|probable|probably|questionable|suspected|'
    r'suspicious\s+for|worrisome\s+for|concern\s+for|concerning\s+for|'
    r'may\s+represent|could\s+represent|might\s+represent|'
    r'cannot\s+(?:be\s+)?(?:entirely\s+)?exclude[d]?|'
    r'cannot\s+rule\s+out|not\s+(?:entirely\s+)?excluded|'
    r'differential\s+includes|versus|vs\.?)\b',
    re.IGNORECASE
)

# ---- CONDITIONAL patterns (hypothetical/future) ----
_CONDITIONAL_PATTERNS = re.compile(
    r'\b(?:if\s+(?:clinical|there\s+is)|should\s+(?:be|have)|'
    r'would\s+(?:suggest|recommend)|recommend(?:ed|s)?\s+(?:clinical|follow)|'
    r'clinical\s+correlation\s+(?:is\s+)?(?:recommended|suggested|advised)|'
    r'further\s+(?:evaluation|workup|assessment)\s+(?:is\s+)?(?:recommended|suggested))\b',
    re.IGNORECASE
)

# ---- HISTORICAL patterns (past finding) ----
_HISTORICAL_PATTERNS = re.compile(
    r'\b(?:history\s+of|h/o|prior|previous(?:ly)?|known|'
    r'remote\s+history|past\s+medical\s+history|'
    r'prior\s+episode|previous\s+episode|'
    r'old|chronic|longstanding|former)\b',
    re.IGNORECASE
)


def _split_sentences(text):
    """Split text into sentences using common delimiters."""
    if not text or not text.strip():
        return []
    # Split on period, question mark, exclamation, semicolon, or newline
    # But preserve common abbreviations
    text = re.sub(r'(?<!\b[A-Z])\.(?=\s+[A-Z])', '.\n', text)
    text = re.sub(r'[;!?]\s+', '.\n', text)
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    # Further split on periods if sentence is very long
    result = []
    for s in sentences:
        if len(s) > 300 and '. ' in s:
            parts = s.split('. ')
            result.extend([p.strip() for p in parts if p.strip()])
        else:
            result.append(s)
    return result


def classify_assertion(sentence, keyword_match):
    """
    Classify the assertion status of a pneumonia keyword within a sentence.
    Returns one of: PRESENT, ABSENT, POSSIBLE, CONDITIONAL, HISTORICAL
    """
    s_lower = sentence.lower()

    # Check ABSENT first (explicit negation)
    for pattern in _ABSENT_PATTERNS:
        if pattern.search(sentence):
            return ASSERT_ABSENT

    # Check POSSIBLE (hedging/uncertainty)
    if _POSSIBLE_PATTERNS.search(sentence):
        return ASSERT_POSSIBLE

    # Check CONDITIONAL (hypothetical)
    if _CONDITIONAL_PATTERNS.search(sentence):
        return ASSERT_CONDITIONAL

    # Check HISTORICAL (past finding)
    if _HISTORICAL_PATTERNS.search(sentence):
        return ASSERT_HISTORICAL

    # Default: PRESENT (finding is affirmed)
    return ASSERT_PRESENT


def classify_document(text):
    """
    Classify a document by analyzing assertion status of each pneumonia mention.

    Returns: (doc_label, assertion_counts_dict, n_pneumonia_sentences)
    """
    if not text or not text.strip():
        return DOC_NEGATIVE, {}, 0

    sentences = _split_sentences(text)
    assertion_counts = Counter()
    n_pneumonia_sentences = 0

    for sentence in sentences:
        # Find all pneumonia keywords in this sentence
        matches = list(_POS_KW_PATTERN.finditer(sentence))
        if not matches:
            continue

        n_pneumonia_sentences += 1

        # Classify assertion for this sentence (based on first match)
        assertion = classify_assertion(sentence, matches[0].group())
        assertion_counts[assertion] += 1

    if n_pneumonia_sentences == 0:
        return DOC_NEGATIVE, assertion_counts, 0

    # Aggregation logic
    if assertion_counts.get(ASSERT_PRESENT, 0) > 0:
        return DOC_POSITIVE, assertion_counts, n_pneumonia_sentences

    if all(assertion_counts.get(a, 0) == 0 for a in
           [ASSERT_PRESENT, ASSERT_POSSIBLE, ASSERT_CONDITIONAL, ASSERT_HISTORICAL]):
        # Only ABSENT assertions
        return DOC_NEGATIVE, assertion_counts, n_pneumonia_sentences

    if assertion_counts.get(ASSERT_ABSENT, 0) == n_pneumonia_sentences:
        # ALL sentences are negated
        return DOC_NEGATIVE, assertion_counts, n_pneumonia_sentences

    # Has POSSIBLE/CONDITIONAL/HISTORICAL without PRESENT → exclude
    return DOC_EXCLUDED, assertion_counts, n_pneumonia_sentences


def main():
    t_start = time.time()

    print("=" * 70)
    print("LAYER 4 — SENTENCE-LEVEL CLINICAL ASSERTION CLASSIFICATION")
    print("=" * 70)
    print()

    # ---- Load parsed reports ----
    print(f"Loading parsed reports from {PARSED_REPORTS}...")
    df = pd.read_csv(PARSED_REPORTS, low_memory=False,
                     usecols=['study_id', 'subject_id', 'impression_text', 'findings_text'])
    df['study_id'] = df['study_id'].astype(str)
    n_total = len(df)
    print(f"  Loaded: {n_total:,} reports")
    print()

    # ---- Process each report ----
    print("  Classifying assertion status for each report...")
    print("  (impression_text first, findings_text as fallback)")
    print()

    results = []
    for idx in range(n_total):
        if idx % 25000 == 0 and idx > 0:
            elapsed = time.time() - t_start
            rate = idx / elapsed
            remaining = (n_total - idx) / rate
            print(f"    Progress: {idx:,}/{n_total:,} ({100*idx/n_total:.1f}%) "
                  f"- {remaining:.0f}s remaining", flush=True)

        row = df.iloc[idx]
        impression = str(row.get('impression_text', '') or '').strip()
        findings = str(row.get('findings_text', '') or '').strip()
        target_text = impression if impression else findings

        doc_label, assertions, n_pneu_sents = classify_document(target_text)

        results.append({
            'study_id': row['study_id'],
            'subject_id': row['subject_id'],
            'l4_label': doc_label,
            'n_pneumonia_sentences': n_pneu_sents,
            'n_present': assertions.get(ASSERT_PRESENT, 0),
            'n_absent': assertions.get(ASSERT_ABSENT, 0),
            'n_possible': assertions.get(ASSERT_POSSIBLE, 0),
            'n_conditional': assertions.get(ASSERT_CONDITIONAL, 0),
            'n_historical': assertions.get(ASSERT_HISTORICAL, 0),
            'dominant_assertion': (
                assertions.most_common(1)[0][0] if assertions else "NONE"
            ),
        })

    df_results = pd.DataFrame(results)

    # ---- Statistics ----
    label_counts = Counter(df_results['l4_label'].tolist())
    n_pos = label_counts.get(DOC_POSITIVE, 0)
    n_neg = label_counts.get(DOC_NEGATIVE, 0)
    n_exc = label_counts.get(DOC_EXCLUDED, 0)

    assertion_totals = {
        ASSERT_PRESENT: int(df_results['n_present'].sum()),
        ASSERT_ABSENT: int(df_results['n_absent'].sum()),
        ASSERT_POSSIBLE: int(df_results['n_possible'].sum()),
        ASSERT_CONDITIONAL: int(df_results['n_conditional'].sum()),
        ASSERT_HISTORICAL: int(df_results['n_historical'].sum()),
    }

    # ---- Save ----
    df_results.to_csv(ASSERTIONS_CSV, index=False)
    file_size_mb = os.path.getsize(ASSERTIONS_CSV) / (1024 * 1024)

    t_total = time.time() - t_start

    print()
    print("=" * 70)
    print("LAYER 4 COMPLETE")
    print("=" * 70)
    print()
    print(f"  Document-level label distribution:")
    print(f"    POSITIVE (has PRESENT assertion):  {n_pos:>8,} ({100*n_pos/n_total:.1f}%)")
    print(f"    NEGATIVE (all ABSENT / no terms):  {n_neg:>8,} ({100*n_neg/n_total:.1f}%)")
    print(f"    EXCLUDED (POSSIBLE/COND/HIST):     {n_exc:>8,} ({100*n_exc/n_total:.1f}%)")
    print()
    print(f"  Sentence-level assertion counts:")
    for assertion, count in sorted(assertion_totals.items(), key=lambda x: -x[1]):
        print(f"    {assertion:>15s}: {count:>8,}")
    print()
    print(f"  File: {ASSERTIONS_CSV}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Runtime: {t_total:.1f}s")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

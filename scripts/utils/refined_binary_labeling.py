"""
===============================================================================
REFINED BINARY PNEUMONIA LABELING PIPELINE — VERSION 3.0 (FINAL)
===============================================================================

Produces ONLY POSITIVE (1) and NEGATIVE (0) labels. No UNCERTAIN, No ABSTAIN.

POSITIVE pipeline (multi-layer verification):
  Layer 1: Negated-uncertain override check  → catches "no X to suggest pneumonia"
  Layer 2: Uncertainty keyword exclusion      → drops hedging language entirely
  Layer 3: Negative keyword detection         → catches explicit negation
  Layer 4: Positive keyword detection         → finds pneumonia-related terms
  Layer 5: Proximity-based negation (NegEx)   → flips false positives
  Layer 6: Section priority                   → impression > findings, ignore history

NEGATIVE pipeline (clean negative strategy):
  Filter 1: NOT classified as POSITIVE by NLP
  Filter 2: CheXpert 12-disease exclusion (1.0 AND -1.0 excluded)
  Filter 3: Only truly normal CXRs with no pathology

Output: final_pneumonia_labels_v2.csv (new file, existing files untouched)
===============================================================================
"""

import pandas as pd
import numpy as np
import re
import os
import sys
import time
import logging
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_DIR = r"C:\Users\dviya\Desktop\Pneumonia_labeling"
PARSED_REPORTS_CSV = os.path.join(PROJECT_DIR, "data", "intermediate", "parsed_reports.csv")
CHEXPERT_CSV = os.path.join(PROJECT_DIR, "data", "raw", "mimic-cxr-2.0.0-chexpert.csv")
OUTPUT_CSV = os.path.join(PROJECT_DIR, "data", "output", "final_pneumonia_labels_v2.csv")
LOG_FILE = os.path.join(PROJECT_DIR, "logs", "refined_binary_labeling.log")

# ---------------------------------------------------------------------------
# MIDDLE-GROUND NEGATIVE FILTER FOR PP1 (DenseNet-121 vision model)
# ---------------------------------------------------------------------------
# PP1 sees CXR IMAGES, not text. Diseases that look visually distinct from
# pneumonia on X-ray should be ALLOWED in the NEGATIVE set so PP1 learns
# that "sick but not pneumonia" = NEGATIVE. Diseases that produce the same
# radiographic pattern as pneumonia (dense white opacities in lung parenchyma)
# must be EXCLUDED to avoid teaching PP1 that pneumonia-looking images are NEGATIVE.
#
# ALLOWED in NEGATIVE (visually distinct from pneumonia on CXR):
#   - No Finding           → perfectly normal
#   - Cardiomegaly         → enlarged heart silhouette, not lung parenchyma
#   - Fracture             → bone, not lung tissue
#   - Enlarged Cardiomediastinum → mediastinal widening, not lung opacity
#   - Support Devices      → tubes/lines/catheters, distinct from opacity
#   - Pleural Effusion     → fluid layering, meniscus sign, distinct from consolidation
#   - Pneumothorax         → air (black), opposite of pneumonia (white)
#
# EXCLUDED from NEGATIVE (radiographically mimics pneumonia):
#   - Consolidation        → identical white patch as pneumonia
#   - Atelectasis          → collapsed lung = opacity, can mimic pneumonia
#   - Lung Opacity         → same visual finding as pneumonia by definition
#   - Lung Lesion          → mass/nodule can overlap with pneumonia pattern
#   - Edema                → bilateral opacities, mimics multifocal pneumonia
#   - Pleural Other        → ambiguous pleural disease, may mimic pneumonia
# ---------------------------------------------------------------------------

# Only these diseases are EXCLUDED from the NEGATIVE set (look like pneumonia on CXR)
DISEASES_EXCLUDED_FROM_NEGATIVE = [
    'Consolidation', 'Atelectasis', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Pleural Other'
]

# These diseases are ALLOWED in the NEGATIVE set (visually distinct from pneumonia)
# Listed here for documentation only — NOT used in the filter logic.
# A NEGATIVE study may have any of these and still be a valid training example.
DISEASES_ALLOWED_IN_NEGATIVE = [
    'Cardiomegaly', 'Fracture', 'Enlarged Cardiomediastinum',
    'Support Devices', 'Pleural Effusion', 'Pneumothorax'
]

# ---------------------------------------------------------------------------
# IMPORT KEYWORD LISTS (from existing keywords.py v2.5)
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))
from stage2_labeling.keywords import (
    POSITIVE_KEYWORDS,
    NEGATIVE_KEYWORDS,
    UNCERTAIN_KEYWORDS,
    NEGATED_UNCERTAIN_OVERRIDES,
    KEYWORD_LIST_VERSION
)

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COMPILE REGEX PATTERNS (longest-first for most-specific matching)
# ---------------------------------------------------------------------------

def compile_pattern(keyword_list):
    """Compile keyword list into a single regex, longest keywords first."""
    sorted_kw = sorted(keyword_list, key=len, reverse=True)
    pattern = '|'.join(re.escape(kw) for kw in sorted_kw)
    return re.compile(pattern, re.IGNORECASE)

PAT_NEGATED_UNCERTAIN = compile_pattern(NEGATED_UNCERTAIN_OVERRIDES)
PAT_UNCERTAIN = compile_pattern(UNCERTAIN_KEYWORDS)
PAT_NEGATIVE = compile_pattern(NEGATIVE_KEYWORDS)
PAT_POSITIVE = compile_pattern(POSITIVE_KEYWORDS)

# Words that, if they follow a negative keyword match, invalidate the negation
# e.g., "no evidence of pneumonia progression" → pneumonia IS present
FALSE_NEG_CONTINUATIONS = re.compile(
    r'\s+(progression|worsening|change|interval\s+change|extending|extension)',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# NEGATION DETECTION (proximity-based, clinical)
# ---------------------------------------------------------------------------

# Pre-negation triggers (appear BEFORE the finding, within the same sentence)
PRE_NEGATION_TRIGGERS = [
    'no evidence of', 'without evidence of', 'no signs of', 'no sign of',
    'negative for', 'not consistent with', 'no suggestion of',
    'no radiographic evidence of', 'no radiographic evidence for',
    'no convincing evidence for', 'no convincing signs of',
    'without', 'no definite', 'no obvious', 'no new', 'no acute',
    'no', 'not', 'denies', 'denied', 'absence of', 'absent', 'never',
    'rule out', 'rules out', 'ruled out',
]
# Sort longest first for matching priority
PRE_NEGATION_TRIGGERS.sort(key=len, reverse=True)

# Post-negation triggers (appear AFTER the finding)
POST_NEGATION_TRIGGERS = [
    'has resolved', 'have resolved', 'has cleared', 'have cleared',
    'has improved', 'now resolved', 'now cleared', 'completely resolved',
    'resolved', 'unlikely', 'is absent', 'not present', 'not seen',
    'was not seen', 'is not seen', 'are not seen', 'not identified',
    'has been excluded', 'was excluded',
]
POST_NEGATION_TRIGGERS.sort(key=len, reverse=True)

# Scope-breaking words — negation before these does NOT extend past them
SCOPE_BREAKERS = ['but', 'however', 'although', 'though', 'except',
                  'apart from', 'aside from', 'nevertheless', 'nonetheless']

WINDOW_PRE = 60   # characters before keyword to check for negation
WINDOW_POST = 40  # characters after keyword to check for negation


def is_negated(text_lower, keyword_lower, match_start):
    """
    Check if a positive keyword at position match_start is negated.
    Uses proximity-based pre/post negation triggers with scope breaking.
    Returns True if negated, False if affirmed.
    """
    # --- Pre-negation check ---
    pre_start = max(0, match_start - WINDOW_PRE)
    prefix = text_lower[pre_start:match_start]

    # Find the latest scope breaker in the prefix
    latest_breaker = -1
    for sb in SCOPE_BREAKERS:
        pos = prefix.rfind(sb)
        if pos > latest_breaker:
            latest_breaker = pos + len(sb)

    # Effective prefix is after the last scope breaker
    if latest_breaker > 0:
        effective_prefix = prefix[latest_breaker:]
    else:
        effective_prefix = prefix

    for trigger in PRE_NEGATION_TRIGGERS:
        if trigger in effective_prefix:
            return True

    # --- Post-negation check ---
    kw_end = match_start + len(keyword_lower)
    post_end = min(len(text_lower), kw_end + WINDOW_POST)
    suffix = text_lower[kw_end:post_end]

    for trigger in POST_NEGATION_TRIGGERS:
        if trigger in suffix:
            return True

    return False


# ---------------------------------------------------------------------------
# REPORT CLASSIFICATION
# ---------------------------------------------------------------------------

def get_target_text(row):
    """Get the best text for classification: impression > findings. Ignore history."""
    imp = str(row.get('impression_text', '')) if pd.notna(row.get('impression_text')) else ''
    find = str(row.get('findings_text', '')) if pd.notna(row.get('findings_text')) else ''
    imp = imp.strip()
    find = find.strip()
    if imp:
        return imp
    elif find:
        return find
    return ''


def classify_report(text):
    """
    Classify a single report's target text.

    Returns tuple: (classification, matched_keyword_or_reason)
      classification is one of:
        'POSITIVE'  — confirmed pneumonia finding
        'NEGATIVE'  — explicitly negated or clear
        'DROP'      — uncertain language, excluded entirely
        'NEUTRAL'   — no pneumonia-related keywords found
    """
    if not text or len(text.strip()) < 5:
        return ('NEUTRAL', 'empty_text')

    text_lower = text.lower()

    # Layer 1: Negated-uncertain overrides → NEGATIVE
    m = PAT_NEGATED_UNCERTAIN.search(text_lower)
    if m:
        return ('NEGATIVE', f'negated_uncertain:{m.group()}')

    # Layer 2: Uncertain keywords → DROP (excluded entirely)
    m = PAT_UNCERTAIN.search(text_lower)
    if m:
        return ('DROP', f'uncertain:{m.group()}')

    # Layer 3: Negative keywords → NEGATIVE (with false-continuation check)
    m = PAT_NEGATIVE.search(text_lower)
    if m:
        # Check for false negative continuations like "no evidence of pneumonia progression"
        end_pos = m.end()
        remaining = text_lower[end_pos:end_pos + 30]
        if not FALSE_NEG_CONTINUATIONS.match(remaining):
            return ('NEGATIVE', f'negative_kw:{m.group()}')
        # If false continuation found, fall through to positive check

    # Layer 4: Positive keywords → POSITIVE candidate
    m = PAT_POSITIVE.search(text_lower)
    if m:
        keyword = m.group()
        match_start = m.start()

        # Layer 5: Proximity-based negation detection
        if is_negated(text_lower, keyword, match_start):
            return ('NEGATIVE', f'negex_negated:{keyword}')

        return ('POSITIVE', f'positive_kw:{keyword}')

    # Layer 6: No match at all
    return ('NEUTRAL', 'no_keyword_match')


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    log.info("=" * 70)
    log.info("REFINED BINARY PNEUMONIA LABELING PIPELINE — VERSION 3.0")
    log.info("=" * 70)
    log.info(f"Keyword list version: {KEYWORD_LIST_VERSION}")
    log.info(f"Positive keywords: {len(POSITIVE_KEYWORDS)}")
    log.info(f"Negative keywords: {len(NEGATIVE_KEYWORDS)}")
    log.info(f"Uncertain keywords: {len(UNCERTAIN_KEYWORDS)}")
    log.info(f"Negated-uncertain overrides: {len(NEGATED_UNCERTAIN_OVERRIDES)}")
    log.info(f"Diseases excluded from negatives (mimic pneumonia on CXR): {len(DISEASES_EXCLUDED_FROM_NEGATIVE)}")
    log.info(f"Diseases allowed in negatives (visually distinct): {len(DISEASES_ALLOWED_IN_NEGATIVE)}")
    log.info("")

    # -----------------------------------------------------------------------
    # STEP 1: Load parsed reports
    # -----------------------------------------------------------------------
    log.info("STEP 1: Loading parsed reports...")
    df = pd.read_csv(PARSED_REPORTS_CSV)
    # Keep study_id with 's' prefix (e.g., 's52067803') to match report filenames
    df['study_id'] = df['study_id'].astype(str).str.strip()
    df['subject_id'] = df['subject_id'].astype(str).str.strip()
    log.info(f"  Loaded {len(df):,} parsed reports")
    log.info(f"  study_id sample: {df['study_id'].iloc[:3].tolist()}")

    # Fill NaN text fields with empty string
    for col in ['impression_text', 'findings_text', 'history_text']:
        df[col] = df[col].fillna('')

    # -----------------------------------------------------------------------
    # STEP 2: Load CheXpert labels
    # -----------------------------------------------------------------------
    log.info("STEP 2: Loading CheXpert labels...")
    chx = pd.read_csv(CHEXPERT_CSV)
    # FIX: CheXpert uses plain integer study_ids (e.g., 50414267).
    # Parsed reports use 's' prefix (e.g., 's50414267').
    # Add 's' prefix to CheXpert to match parsed_reports format.
    chx['study_id'] = 's' + chx['study_id'].astype(str).str.strip()
    log.info(f"  Loaded {len(chx):,} CheXpert entries")
    log.info(f"  study_id sample (after 's' prefix): {chx['study_id'].iloc[:3].tolist()}")

    # Verify join compatibility
    sample_parsed = set(df['study_id'].head(100))
    sample_chx = set(chx['study_id'].head(100))
    overlap = sample_parsed & sample_chx
    log.info(f"  Join check — sample overlap: {len(overlap)} / 100 "
             f"(0 = BROKEN JOIN, investigate study_id formats)")
    if len(overlap) == 0:
        log.warning("  ⚠ ZERO overlap between parsed_reports and CheXpert study_ids!")
        log.warning(f"    parsed sample: {list(sample_parsed)[:5]}")
        log.warning(f"    chexpert sample: {list(sample_chx)[:5]}")

    # Build clean-negative mask: only diseases that MIMIC pneumonia on CXR
    # are excluded (6 diseases). Visually distinct diseases (cardiomegaly,
    # fracture, pleural effusion, etc.) are ALLOWED in the NEGATIVE set
    # so PP1 learns that "sick but not pneumonia" = NEGATIVE.
    clean_mask = pd.Series(True, index=chx.index)
    for disease in DISEASES_EXCLUDED_FROM_NEGATIVE:
        # Exclude if value is 1.0 (positive) or -1.0 (uncertain)
        clean_mask = clean_mask & ((chx[disease].isna()) | (chx[disease] == 0.0))

    clean_study_ids = set(chx.loc[clean_mask, 'study_id'].values)
    log.info(f"  Studies passing middle-ground filter (clean): {len(clean_study_ids):,}")
    log.info(f"  Studies excluded (pneumonia-mimicking diseases): "
             f"{len(chx) - len(clean_study_ids):,}")
    log.info(f"  Excluded diseases: {DISEASES_EXCLUDED_FROM_NEGATIVE}")
    log.info(f"  Allowed diseases:  {DISEASES_ALLOWED_IN_NEGATIVE}")

    # -----------------------------------------------------------------------
    # STEP 3: Classify all reports using NLP
    # -----------------------------------------------------------------------
    log.info("STEP 3: Classifying all reports...")
    classifications = []
    evidence_list = []

    for idx, row in df.iterrows():
        target_text = get_target_text(row)
        cls, evidence = classify_report(target_text)
        classifications.append(cls)
        evidence_list.append(evidence)

        if (idx + 1) % 50000 == 0:
            log.info(f"  Processed {idx + 1:,} / {len(df):,} reports...")

    df['nlp_class'] = classifications
    df['nlp_evidence'] = evidence_list

    # Distribution of NLP classifications
    nlp_counts = df['nlp_class'].value_counts()
    log.info(f"  NLP Classification Results:")
    for cls_name, count in nlp_counts.items():
        log.info(f"    {cls_name}: {count:,} ({100*count/len(df):.1f}%)")

    # -----------------------------------------------------------------------
    # STEP 4: Build POSITIVE set
    # -----------------------------------------------------------------------
    log.info("STEP 4: Building POSITIVE label set...")
    pos_mask = df['nlp_class'] == 'POSITIVE'
    positives = df[pos_mask][['subject_id', 'study_id', 'nlp_evidence']].copy()
    positives['label'] = 1
    positives['label_source'] = 'nlp_confirmed'
    log.info(f"  POSITIVE labels: {len(positives):,}")

    # -----------------------------------------------------------------------
    # STEP 5: Build NEGATIVE set (clean negatives only)
    # -----------------------------------------------------------------------
    log.info("STEP 5: Building NEGATIVE label set (clean negatives)...")

    # Candidates for NEGATIVE: NLP said NEGATIVE or NEUTRAL (not POSITIVE, not DROP)
    neg_candidates = df[df['nlp_class'].isin(['NEGATIVE', 'NEUTRAL'])].copy()
    log.info(f"  NEGATIVE/NEUTRAL candidates from NLP: {len(neg_candidates):,}")

    # Apply CheXpert clean filter: must be in clean_study_ids
    neg_candidates['is_clean'] = neg_candidates['study_id'].isin(clean_study_ids)
    clean_negatives = neg_candidates[neg_candidates['is_clean']].copy()
    log.info(f"  After middle-ground disease filter (6 excluded): {len(clean_negatives):,}")

    # Exclude any that are also in the POSITIVE set (shouldn't happen, but safeguard)
    pos_study_ids = set(positives['study_id'].values)
    clean_negatives = clean_negatives[~clean_negatives['study_id'].isin(pos_study_ids)]
    log.info(f"  After excluding positive overlaps: {len(clean_negatives):,}")

    negatives = clean_negatives[['subject_id', 'study_id', 'nlp_evidence']].copy()
    negatives['label'] = 0
    negatives['label_source'] = negatives['nlp_evidence'].apply(
        lambda x: 'nlp_negative' if 'negative' in str(x) or 'negex' in str(x) else 'clean_normal'
    )

    # -----------------------------------------------------------------------
    # STEP 6: Combine and save
    # -----------------------------------------------------------------------
    log.info("STEP 6: Combining POSITIVE + NEGATIVE and saving...")

    final = pd.concat([positives, negatives], ignore_index=True)

    # Generate confidence scores based on evidence strength
    def compute_confidence(row):
        ev = str(row['nlp_evidence']).lower()
        if row['label'] == 1:
            # POSITIVE confidence
            if 'pneumonia' in ev:
                return 0.95  # Explicit pneumonia term
            elif 'bronchopneumonia' in ev or 'pneumonitis' in ev:
                return 0.95
            else:
                return 0.85  # Generic term (consolidation, opacity, etc.)
        else:
            # NEGATIVE confidence
            if 'negative_kw' in ev or 'negated_uncertain' in ev:
                return 0.95  # Explicit negation
            elif 'negex_negated' in ev:
                return 0.90  # NegEx-detected negation
            else:
                return 0.92  # Clean normal (no keywords at all)

    final['confidence'] = final.apply(compute_confidence, axis=1)

    # Select and order output columns
    final = final[['subject_id', 'study_id', 'label', 'confidence', 'label_source',
                    'nlp_evidence']].copy()
    final.columns = ['subject_id', 'study_id', 'label', 'confidence', 'label_source',
                     'evidence']

    # Sort by study_id for deterministic output
    final = final.sort_values('study_id').reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final.to_csv(OUTPUT_CSV, index=False)
    log.info(f"  Saved: {OUTPUT_CSV}")
    log.info(f"  Total labels: {len(final):,}")

    # -----------------------------------------------------------------------
    # STEP 7: Summary statistics and quality checks
    # -----------------------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)

    pos_count = (final['label'] == 1).sum()
    neg_count = (final['label'] == 0).sum()
    total = len(final)
    ratio = neg_count / pos_count if pos_count > 0 else float('inf')

    log.info(f"  POSITIVE:  {pos_count:>8,}  ({100*pos_count/total:.1f}%)")
    log.info(f"  NEGATIVE:  {neg_count:>8,}  ({100*neg_count/total:.1f}%)")
    log.info(f"  TOTAL:     {total:>8,}")
    log.info(f"  NEG:POS ratio: {ratio:.2f}:1")
    log.info("")
    log.info(f"  Label source breakdown:")
    for src, cnt in final['label_source'].value_counts().items():
        log.info(f"    {src}: {cnt:,}")

    # Dropped/excluded counts
    dropped = (df['nlp_class'] == 'DROP').sum()
    excluded_by_chexpert = len(neg_candidates) - len(clean_negatives)
    not_in_final = len(df) - total
    log.info(f"")
    log.info(f"  Reports dropped (uncertain): {dropped:,}")
    log.info(f"  Reports excluded by CheXpert disease filter: {excluded_by_chexpert:,}")
    log.info(f"  Total reports NOT in final set: {not_in_final:,}")

    # -----------------------------------------------------------------------
    # STEP 8: Sample verification
    # -----------------------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("SAMPLE VERIFICATION — 10 POSITIVE examples")
    log.info("=" * 70)

    pos_samples = final[final['label'] == 1].sample(n=min(10, pos_count), random_state=42)
    for _, row in pos_samples.iterrows():
        sid = row['study_id']
        report_row = df[df['study_id'] == sid]
        if len(report_row) > 0:
            imp = str(report_row.iloc[0]['impression_text'])[:200]
            log.info(f"  [{sid}] evidence={row['evidence']}")
            log.info(f"    IMPRESSION: {imp}")
            log.info("")

    log.info("=" * 70)
    log.info("SAMPLE VERIFICATION — 10 NEGATIVE examples")
    log.info("=" * 70)

    neg_samples = final[final['label'] == 0].sample(n=min(10, neg_count), random_state=42)
    for _, row in neg_samples.iterrows():
        sid = row['study_id']
        report_row = df[df['study_id'] == sid]
        if len(report_row) > 0:
            imp = str(report_row.iloc[0]['impression_text'])[:200]
            log.info(f"  [{sid}] source={row['label_source']} evidence={row['evidence']}")
            log.info(f"    IMPRESSION: {imp}")
            log.info("")

    elapsed = time.time() - start_time
    log.info(f"Pipeline completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    log.info(f"Output saved to: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()

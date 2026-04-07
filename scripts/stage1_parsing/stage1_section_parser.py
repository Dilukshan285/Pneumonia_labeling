"""
Stage 1 — Section-Aware Report Parsing

Parses each radiology report into clinical sections (IMPRESSION, FINDINGS,
HISTORY, etc.) and stores them as separate columns. This enables downstream
labeling functions to prioritize high-value sections and avoid false positives
from low-value sections like CLINICAL HISTORY.

Steps implemented:
  1.1 — Define section header patterns
  1.2 — Split each report into sections via regex
  1.3 — Assign clinical weights (IMPRESSION=3, FINDINGS=2, HISTORY=1, TECHNIQUE=0)
  1.4 — Store parsed sections as parsed_reports.csv
"""

import os
import sys
import re
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MASTER_REPORTS_CSV, PARSED_REPORTS_CSV

NUM_WORKERS = 8

# ============================================================================
# STEP 1.1 — SECTION HEADER PATTERNS
# ============================================================================
# Each pattern is (compiled_regex, normalized_section_name)
# Patterns are ordered by specificity — more specific patterns first to
# prevent shorter patterns from consuming text meant for longer headers.
# All patterns are case-insensitive and handle trailing colons, irregular
# spacing, and punctuation variations.

SECTION_HEADER_PATTERNS = [
    # Most specific multi-word headers first
    (re.compile(r'(?i)\n\s*CLINICAL\s+HISTORY\s*[:\-]?\s*'), 'CLINICAL_HISTORY'),
    (re.compile(r'(?i)\n\s*CLINICAL\s+INFORMATION\s*[:\-]?\s*'), 'CLINICAL_HISTORY'),
    (re.compile(r'(?i)\n\s*CLINICAL\s+INDICATION\s*[:\-]?\s*'), 'INDICATION'),
    (re.compile(r'(?i)\n\s*REASON\s+FOR\s+EXAMINATION\s*[:\-]?\s*'), 'INDICATION'),
    (re.compile(r'(?i)\n\s*REASON\s+FOR\s+EXAM\s*[:\-]?\s*'), 'INDICATION'),
    (re.compile(r'(?i)\n\s*TYPE\s+OF\s+EXAMINATION\s*[:\-]?\s*'), 'EXAMINATION'),
    (re.compile(r'(?i)\n\s*FINAL\s+REPORT\s*'), 'FINAL_REPORT'),
    (re.compile(r'(?i)\n\s*WET\s+READ\s*[:\-]?\s*'), 'WET_READ'),
    (re.compile(r'(?i)\n\s*CHEST\s+RADIOGRAPH[S]?\s*'), 'EXAMINATION'),
    (re.compile(r'(?i)\n\s*PORTABLE\s+CHEST\s*'), 'EXAMINATION'),

    # Single-word headers
    (re.compile(r'(?i)\n\s*IMPRESSION\s*[:\-]?\s*'), 'IMPRESSION'),
    (re.compile(r'(?i)\n\s*FINDINGS?\s*[:\-]?\s*'), 'FINDINGS'),
    (re.compile(r'(?i)\n\s*INDICATION\s*[:\-]?\s*'), 'INDICATION'),
    (re.compile(r'(?i)\n\s*HISTORY\s*[:\-]?\s*'), 'HISTORY'),
    (re.compile(r'(?i)\n\s*TECHNIQUE\s*[:\-]?\s*'), 'TECHNIQUE'),
    (re.compile(r'(?i)\n\s*COMPARISONS?\s*[:\-]?\s*'), 'COMPARISON'),
    (re.compile(r'(?i)\n\s*CONCLUSION\s*[:\-]?\s*'), 'CONCLUSION'),
    (re.compile(r'(?i)\n\s*RECOMMENDATIONS?\s*[:\-]?\s*'), 'RECOMMENDATION'),
    (re.compile(r'(?i)\n\s*EXAMINATION\s*[:\-]?\s*'), 'EXAMINATION'),
    (re.compile(r'(?i)\n\s*ADDENDUM\s*[:\-]?\s*'), 'ADDENDUM'),
    (re.compile(r'(?i)\n\s*STUDY\s*[:\-]?\s*'), 'EXAMINATION'),
    (re.compile(r'(?i)\n\s*EXAM\s*[:\-]?\s*'), 'EXAMINATION'),
    (re.compile(r'(?i)\n\s*NOTIFICATION\s*[:\-]?\s*'), 'NOTIFICATION'),
]

# ============================================================================
# STEP 1.3 — CLINICAL WEIGHTS
# ============================================================================
# Weight 3 = highest priority (radiologist's final conclusion)
# Weight 2 = direct observational findings
# Weight 1 = clinical context (reason for imaging)
# Weight 0 = not used for labeling under any circumstance

SECTION_WEIGHTS = {
    'IMPRESSION': 3,
    'CONCLUSION': 3,
    'FINDINGS': 2,
    'CLINICAL_HISTORY': 1,
    'HISTORY': 1,
    'INDICATION': 1,
    'TECHNIQUE': 0,
    'COMPARISON': 0,
    'EXAMINATION': 0,
    'FINAL_REPORT': 0,
    'WET_READ': 0,
    'RECOMMENDATION': 0,
    'ADDENDUM': 0,
    'NOTIFICATION': 0,
}


# ============================================================================
# STEP 1.2 — SPLIT REPORT INTO SECTIONS
# ============================================================================

def parse_report_sections(report_text):
    """
    Parse a single report into a dictionary of section_name -> section_text.

    Uses regex to find section headers and split the text at header boundaries.
    Returns a dict with normalized uppercase section names as keys.
    Sections not found in the report have empty string values.
    """
    # Prepend a newline so all patterns (which expect \n prefix) can match
    # the very first line of the report
    text = '\n' + report_text

    # Find all section header positions
    header_positions = []
    for pattern, section_name in SECTION_HEADER_PATTERNS:
        for match in pattern.finditer(text):
            header_positions.append((match.start(), match.end(), section_name))

    # If no headers found, return everything as unstructured text
    if not header_positions:
        return {
            'IMPRESSION': '',
            'FINDINGS': '',
            'CLINICAL_HISTORY': '',
            'HISTORY': '',
            'INDICATION': '',
            'TECHNIQUE': '',
            'COMPARISON': '',
            'EXAMINATION': '',
            'CONCLUSION': '',
            'RECOMMENDATION': '',
            '_FULL_TEXT': report_text.strip(),
        }

    # Sort by position in text
    header_positions.sort(key=lambda x: x[0])

    # Remove overlapping headers — keep the earliest match at each position
    filtered = []
    last_end = -1
    for start, end, name in header_positions:
        if start >= last_end:
            filtered.append((start, end, name))
            last_end = end
    header_positions = filtered

    # Extract section text between consecutive headers
    sections = {}
    for i, (start, end, name) in enumerate(header_positions):
        # Section text runs from end of this header to start of next header
        if i + 1 < len(header_positions):
            next_start = header_positions[i + 1][0]
            section_text = text[end:next_start].strip()
        else:
            section_text = text[end:].strip()

        # If section already exists (duplicate header), append text
        if name in sections and sections[name]:
            sections[name] = sections[name] + ' ' + section_text
        else:
            sections[name] = section_text

    # Ensure all expected sections exist with empty string default
    all_section_names = [
        'IMPRESSION', 'FINDINGS', 'CLINICAL_HISTORY', 'HISTORY',
        'INDICATION', 'TECHNIQUE', 'COMPARISON', 'EXAMINATION',
        'CONCLUSION', 'RECOMMENDATION',
    ]
    for name in all_section_names:
        if name not in sections:
            sections[name] = ''

    sections['_FULL_TEXT'] = report_text.strip()

    return sections


def extract_section_columns(row):
    """
    Extract the three diagnostically relevant section columns from a report.
    Returns (impression_text, findings_text, history_text).
    """
    text = row['report_text']
    if not isinstance(text, str) or not text.strip():
        return '', '', ''

    sections = parse_report_sections(text)

    # IMPRESSION: check IMPRESSION first, then CONCLUSION as fallback
    impression = sections.get('IMPRESSION', '')
    if not impression:
        impression = sections.get('CONCLUSION', '')

    # FINDINGS
    findings = sections.get('FINDINGS', '')

    # HISTORY: merge CLINICAL_HISTORY, HISTORY, and INDICATION
    history_parts = []
    for key in ['CLINICAL_HISTORY', 'HISTORY', 'INDICATION']:
        val = sections.get(key, '')
        if val:
            history_parts.append(val)
    history = ' '.join(history_parts)

    return impression.strip(), findings.strip(), history.strip()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("STAGE 1 — SECTION-AWARE REPORT PARSING")
    print("=" * 70)
    print()

    # Load master reports
    if not os.path.exists(MASTER_REPORTS_CSV):
        print(f"ERROR: master_reports.csv not found at: {MASTER_REPORTS_CSV}")
        print("       Run p3_load_reports.py first.")
        return 1

    print(f"Loading {MASTER_REPORTS_CSV}...")
    df = pd.read_csv(MASTER_REPORTS_CSV)
    print(f"  Total reports: {len(df):,}")
    print()

    # -----------------------------------------------------------------------
    # Step 1.2 + 1.4 — Parse sections and store as columns
    # -----------------------------------------------------------------------
    print("Parsing sections for all reports...")
    print("  (Using regex-based section splitter with clinical weight priorities)")
    print()

    # Process reports with progress bar
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Parsing", unit="reports"):
        impression, findings, history = extract_section_columns(row)
        results.append((impression, findings, history))

    # Add columns to DataFrame
    df['impression_text'] = [r[0] for r in results]
    df['findings_text'] = [r[1] for r in results]
    df['history_text'] = [r[2] for r in results]

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print()
    print("-" * 70)
    print("SECTION PARSING SUMMARY")
    print("-" * 70)

    n_total = len(df)
    n_impression = (df['impression_text'].str.len() > 0).sum()
    n_findings = (df['findings_text'].str.len() > 0).sum()
    n_history = (df['history_text'].str.len() > 0).sum()
    n_both = ((df['impression_text'].str.len() > 0) & (df['findings_text'].str.len() > 0)).sum()
    n_neither = ((df['impression_text'].str.len() == 0) & (df['findings_text'].str.len() == 0)).sum()

    print(f"  Total reports:                        {n_total:>8,}")
    print()
    print(f"  Reports with IMPRESSION (weight=3):   {n_impression:>8,}  ({100*n_impression/n_total:.1f}%)")
    print(f"  Reports with FINDINGS   (weight=2):   {n_findings:>8,}  ({100*n_findings/n_total:.1f}%)")
    print(f"  Reports with HISTORY    (weight=1):   {n_history:>8,}  ({100*n_history/n_total:.1f}%)")
    print()
    print(f"  Reports with BOTH Impression+Findings: {n_both:>7,}  ({100*n_both/n_total:.1f}%)")
    print(f"  Reports with NEITHER (empty both):     {n_neither:>7,}  ({100*n_neither/n_total:.1f}%)")
    print(f"    → These {n_neither:,} will be handled by Stage 2 pre-filter")
    print()

    # Text length stats for each section
    for col_name, display_name in [('impression_text', 'IMPRESSION'), ('findings_text', 'FINDINGS'), ('history_text', 'HISTORY')]:
        non_empty = df[df[col_name].str.len() > 0][col_name]
        if len(non_empty) > 0:
            print(f"  {display_name} text length (non-empty):")
            print(f"    Mean:   {non_empty.str.len().mean():.0f} chars")
            print(f"    Median: {non_empty.str.len().median():.0f} chars")
            print(f"    Min:    {non_empty.str.len().min()}")
            print(f"    Max:    {non_empty.str.len().max()}")
            print()

    # Show clinical weight reference
    print("-" * 70)
    print("CLINICAL WEIGHT REFERENCE (Step 1.3)")
    print("-" * 70)
    print("  IMPRESSION / CONCLUSION   → weight 3  (final diagnosis)")
    print("  FINDINGS                  → weight 2  (direct observations)")
    print("  HISTORY / INDICATION      → weight 1  (clinical context)")
    print("  TECHNIQUE / COMPARISON    → weight 0  (never used for labeling)")
    print()

    # -----------------------------------------------------------------------
    # Step 1.4 — Save parsed_reports.csv
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(PARSED_REPORTS_CSV), exist_ok=True)
    print(f"Saving to: {PARSED_REPORTS_CSV}")
    df.to_csv(PARSED_REPORTS_CSV, index=False)
    file_size_mb = os.path.getsize(PARSED_REPORTS_CSV) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")
    print()

    # Preview
    print("First 3 rows (text columns truncated to 60 chars):")
    print("-" * 70)
    preview = df.head(3)[['study_id', 'impression_text', 'findings_text', 'history_text']].copy()
    for col in ['impression_text', 'findings_text', 'history_text']:
        preview[col] = preview[col].str[:60].fillna('') + '...'
    print(preview.to_string(index=False))
    print()

    print("=" * 70)
    print("STAGE 1 COMPLETE — parsed_reports.csv saved.")
    print(f"  Output: {PARSED_REPORTS_CSV}")
    print(f"  Columns: subject_id, study_id, report_text, impression_text, findings_text, history_text")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

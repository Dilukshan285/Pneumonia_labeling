"""
Step P4 — Verify Report Contents
Randomly samples 30 reports from master_reports.csv, inspects their structure,
and documents which sections are present in each. This validates the section
parsing rules for Stage 1.
"""

import os
import sys
import re
import pandas as pd
import random

# Add parent directory to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MASTER_REPORTS_CSV, P4_INSPECTION_REPORT, RANDOM_SEED

SAMPLE_SIZE = 30

# Known section header patterns (case-insensitive)
SECTION_PATTERNS = [
    (r'(?i)\bFINDINGS?\s*:', 'FINDINGS'),
    (r'(?i)\bIMPRESSION\s*:', 'IMPRESSION'),
    (r'(?i)\bCLINICAL\s+HISTORY\s*:', 'CLINICAL HISTORY'),
    (r'(?i)\bINDICATION\s*:', 'INDICATION'),
    (r'(?i)\bHISTORY\s*:', 'HISTORY'),
    (r'(?i)\bTECHNIQUE\s*:', 'TECHNIQUE'),
    (r'(?i)\bCOMPARISON\s*:', 'COMPARISON'),
    (r'(?i)\bCONCLUSION\s*:', 'CONCLUSION'),
    (r'(?i)\bRECOMMENDATION\s*:', 'RECOMMENDATION'),
    (r'(?i)\bEXAMINATION\s*:', 'EXAMINATION'),
    (r'(?i)\bFINAL\s+REPORT', 'FINAL REPORT'),
    (r'(?i)\bWET\s+READ', 'WET READ'),
    (r'(?i)\bADDENDUM\s*:', 'ADDENDUM'),
]

# Chest/lung terminology for identifying unrelated studies
CHEST_TERMS = [
    'chest', 'lung', 'lungs', 'pulmonary', 'pneumonia', 'pleural',
    'cardiopulmonary', 'cardiac', 'heart', 'mediastin', 'hilar',
    'bronch', 'trachea', 'diaphragm', 'rib', 'thorax', 'thoracic',
    'consolidation', 'infiltrate', 'opacity', 'effusion', 'pneumothorax',
    'atelectasis', 'cardiomegaly', 'edema'
]


def detect_sections(report_text):
    """Detect which sections are present in a report."""
    found_sections = []
    for pattern, name in SECTION_PATTERNS:
        if re.search(pattern, report_text):
            found_sections.append(name)
    return found_sections


def has_chest_terminology(report_text):
    """Check if the report contains any chest/lung terms."""
    text_lower = report_text.lower()
    for term in CHEST_TERMS:
        if term in text_lower:
            return True
    return False


def main():
    print("=" * 70)
    print("STEP P4 — VERIFY REPORT CONTENTS")
    print("=" * 70)
    print()

    if not os.path.exists(MASTER_REPORTS_CSV):
        print(f"ERROR: master_reports.csv not found at: {MASTER_REPORTS_CSV}")
        print("       Run p3_load_reports.py first.")
        return 1

    # Load master reports
    print(f"Loading {MASTER_REPORTS_CSV}...")
    df = pd.read_csv(MASTER_REPORTS_CSV)
    print(f"  Total reports: {len(df):,}")
    print()

    # Random sample
    random.seed(RANDOM_SEED)
    sample_indices = random.sample(range(len(df)), min(SAMPLE_SIZE, len(df)))
    sample_df = df.iloc[sample_indices].reset_index(drop=True)

    # Analyze each report
    lines = []
    lines.append("=" * 80)
    lines.append("STEP P4 — REPORT INSPECTION RESULTS")
    lines.append(f"Sample size: {len(sample_df)} reports (random seed: {RANDOM_SEED})")
    lines.append("=" * 80)
    lines.append("")

    section_counts = {}
    chest_related = 0
    non_chest = 0

    for idx, row in sample_df.iterrows():
        study_id = row['study_id']
        subject_id = row['subject_id']
        text = row['report_text']

        sections = detect_sections(text)
        has_chest = has_chest_terminology(text)

        if has_chest:
            chest_related += 1
        else:
            non_chest += 1

        for s in sections:
            section_counts[s] = section_counts.get(s, 0) + 1

        lines.append(f"--- Report {idx + 1}/{len(sample_df)} ---")
        lines.append(f"  Subject ID:  {subject_id}")
        lines.append(f"  Study ID:    {study_id}")
        lines.append(f"  Text length: {len(text)} chars")
        lines.append(f"  Sections:    {', '.join(sections) if sections else 'NONE DETECTED'}")
        lines.append(f"  Chest terms: {'YES' if has_chest else 'NO — potentially unrelated study'}")
        lines.append(f"  Text preview (first 300 chars):")
        lines.append(f"    {text[:300].replace(chr(10), ' | ')}")
        lines.append("")

    # Summary statistics
    lines.append("")
    lines.append("=" * 80)
    lines.append("SECTION FREQUENCY SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(sample_df)
        bar = '#' * int(pct / 2)
        lines.append(f"  {section:<20s}  {count:>3d}/{len(sample_df)}  ({pct:5.1f}%)  {bar}")

    lines.append("")
    lines.append(f"  Reports WITH chest/lung terminology:     {chest_related}/{len(sample_df)} ({100*chest_related/len(sample_df):.1f}%)")
    lines.append(f"  Reports WITHOUT chest/lung terminology:  {non_chest}/{len(sample_df)} ({100*non_chest/len(sample_df):.1f}%)")
    lines.append("")

    # Key observations
    lines.append("=" * 80)
    lines.append("KEY OBSERVATIONS FOR STAGE 1 SECTION PARSING")
    lines.append("=" * 80)
    lines.append("")

    has_impression = section_counts.get('IMPRESSION', 0)
    has_findings = section_counts.get('FINDINGS', 0)

    lines.append(f"  1. IMPRESSION section present in {has_impression}/{len(sample_df)} reports ({100*has_impression/len(sample_df):.1f}%)")
    lines.append(f"     → This is the highest-priority section for labeling (weight=3)")
    lines.append("")
    lines.append(f"  2. FINDINGS section present in {has_findings}/{len(sample_df)} reports ({100*has_findings/len(sample_df):.1f}%)")
    lines.append(f"     → Fallback target when IMPRESSION is missing (weight=2)")
    lines.append("")

    both = sum(1 for i, row in sample_df.iterrows()
               if 'IMPRESSION' in detect_sections(row['report_text'])
               and 'FINDINGS' in detect_sections(row['report_text']))
    neither = sum(1 for i, row in sample_df.iterrows()
                  if 'IMPRESSION' not in detect_sections(row['report_text'])
                  and 'FINDINGS' not in detect_sections(row['report_text']))

    lines.append(f"  3. Reports with BOTH Impression + Findings: {both}/{len(sample_df)}")
    lines.append(f"     Reports with NEITHER:                    {neither}/{len(sample_df)}")
    lines.append("")
    lines.append(f"  4. Non-chest studies detected: {non_chest}/{len(sample_df)}")
    lines.append(f"     → These will be handled by the pre-filter in Stage 2 Step 2.0")
    lines.append("")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(P4_INSPECTION_REPORT), exist_ok=True)

    # Write to file
    report_text = '\n'.join(lines)
    with open(P4_INSPECTION_REPORT, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Print to console
    print(report_text)

    print()
    print(f"Full inspection report saved to: {P4_INSPECTION_REPORT}")
    print()
    print("=" * 70)
    print("STEP P4 COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

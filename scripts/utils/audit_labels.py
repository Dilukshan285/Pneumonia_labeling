"""
AUDIT: Estimate accuracy of final_pneumonia_labels.csv
Checks a random sample of POSITIVE and NEGATIVE labeled reports
using simple negation pattern matching to detect obvious errors.
"""

import pandas as pd
import os
import re
import random

random.seed(42)

# === Paths ===
LABELS_CSV = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\final_pneumonia_labels.csv"
REPORTS_DIR = r"c:\Users\dviya\Downloads\mimic-cxr-reports\files"

# === Load labels ===
df = pd.read_csv(LABELS_CSV)
df['subject_id'] = df['subject_id'].astype(str)
df['study_id'] = df['study_id'].astype(str)

def get_report_path(subject_id, study_id):
    prefix = "p" + subject_id[:2]
    return os.path.join(REPORTS_DIR, prefix, f"p{subject_id}", f"{study_id}.txt")

def read_report(subject_id, study_id):
    path = get_report_path(subject_id, study_id)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    return None

# ============================================================================
# NEGATION PATTERNS that indicate NEGATIVE for pneumonia
# If any of these appear in IMPRESSION or FINDINGS, the report is likely NEGATIVE
# ============================================================================
CLEAR_NEGATIVE_PATTERNS = [
    r'no\s+(new\s+)?focal\s+opacit(y|ies)\s+to\s+suggest\s+pneumonia',
    r'no\s+(evidence\s+of\s+)?pneumonia',
    r'no\s+acute\s+infiltrate',
    r'no\s+focal\s+consolidation',
    r'no\s+consolidation',
    r'no\s+infiltrate',
    r'lungs\s+are\s+clear',
    r'clear\s+lungs',
    r'no\s+acute\s+cardiopulmonary',
    r'no\s+acute\s+findings',
    r'no\s+acute\s+process',
    r'no\s+airspace\s+disease',
    r'no\s+focal\s+airspace',
    r'no\s+new\s+consolidation',
    r'no\s+new\s+infiltrate',
    r'without\s+pneumonia',
    r'without\s+consolidation',
    r'without\s+infiltrate',
    r'no\s+evidence\s+of\s+consolidation',
    r'no\s+evidence\s+of\s+infiltrate',
    r'no\s+evidence\s+of\s+acute\s+pneumonia',
    r'no\s+focal\s+opacity',
    r'unremarkable\s+lungs',
    r'lungs\s+are\s+well\s+expanded',
    r'negative\s+for\s+pneumonia',
    r'rules?\s+out\s+pneumonia',
    r'clear\s+of\s+infiltrate',
    r'no\s+suspicious\s+.*pneumonia',
    r'no\s+definite.*pneumonia',
    r'no\s+focal\s+infiltrate',
]

# POSITIVE patterns that indicate POSITIVE (true pneumonia finding)
# Using simple patterns without lookbehinds - negation is handled separately
CLEAR_POSITIVE_PATTERNS = [
    r'pneumonia\s+is\s+(identified|seen|present|noted)',
    r'consolidation\s+(consistent|compatible)\s+with\s+pneumonia',
    r'findings\s+(consistent|compatible|suggestive)\s+(with|of)\s+pneumonia',
    r'representing\s+pneumonia',
    r'likely\s+(represents?\s+)?pneumonia',
    r'air\s+bronchogram',
    r'lobar\s+consolidation',
    r'focal\s+consolidation',
    r'right\s+(lower|upper|middle)\s+lobe\s+(pneumonia|consolidation)',
    r'left\s+(lower|upper)\s+lobe\s+(pneumonia|consolidation)',
    r'bilateral\s+pneumonia',
    r'developing\s+pneumonia',
    r'infectious\s+(process|consolidation)',
    r'aspiration\s+pneumonia',
    r'multifocal\s+pneumonia',
]

def extract_impression(text):
    """Extract IMPRESSION section from report."""
    text_lower = text.lower()
    # Try to find IMPRESSION section
    patterns = [r'impression:?\s*\n(.*?)(?=\n\s*[A-Z]{3,}|\Z)',
                r'impression:?\s*(.*?)(?=\n\s*[A-Z]{3,}|\Z)']
    for pat in patterns:
        match = re.search(pat, text_lower, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None

def extract_findings(text):
    """Extract FINDINGS section from report."""
    text_lower = text.lower()
    match = re.search(r'findings:?\s*\n(.*?)(?=impression|recommendation|\Z)', text_lower, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def check_for_negation(text):
    """Check if text contains clear negative pneumonia patterns."""
    text_lower = text.lower()
    for pattern in CLEAR_NEGATIVE_PATTERNS:
        if re.search(pattern, text_lower):
            return True, pattern
    return False, None

def check_for_positive(text):
    """Check if text contains clear positive pneumonia patterns."""
    text_lower = text.lower()
    for pattern in CLEAR_POSITIVE_PATTERNS:
        if re.search(pattern, text_lower):
            return True, pattern
    return False, None

# ============================================================================
# AUDIT 1: Check POSITIVE labels for false positives (negation errors)
# ============================================================================
print("=" * 70)
print("AUDIT 1: Checking POSITIVE-labeled reports for false positives")
print("=" * 70)

positives = df[df['label'] == 1]
pos_sample_indices = random.sample(range(len(positives)), min(1000, len(positives)))
pos_sample = positives.iloc[pos_sample_indices]

false_positives = []
true_positives = []
unclear_positives = []

for _, row in pos_sample.iterrows():
    text = read_report(row['subject_id'], row['study_id'])
    if text is None:
        continue
    
    impression = extract_impression(text)
    findings = extract_findings(text)
    target = impression if impression else (findings if findings else text)
    
    has_neg, neg_pattern = check_for_negation(target)
    has_pos, pos_pattern = check_for_positive(target)
    
    if has_neg and not has_pos:
        false_positives.append({
            'subject_id': row['subject_id'],
            'study_id': row['study_id'],
            'soft_score': row['soft_score'],
            'neg_pattern': neg_pattern,
            'impression_snippet': (impression or "")[:200]
        })
    elif has_pos:
        true_positives.append(row['study_id'])
    else:
        unclear_positives.append(row['study_id'])

total_checked_pos = len(true_positives) + len(false_positives) + len(unclear_positives)
print(f"\nPositive-labeled reports checked: {total_checked_pos}")
print(f"  Likely TRUE positives:   {len(true_positives)} ({100*len(true_positives)/total_checked_pos:.1f}%)")
print(f"  Likely FALSE positives:  {len(false_positives)} ({100*len(false_positives)/total_checked_pos:.1f}%)")
print(f"  Ambiguous/unclear:       {len(unclear_positives)} ({100*len(unclear_positives)/total_checked_pos:.1f}%)")

print(f"\n--- Sample FALSE POSITIVES (first 10) ---")
for i, fp in enumerate(false_positives[:10]):
    print(f"\n  [{i+1}] p{fp['subject_id']}_{fp['study_id']}")
    print(f"      Soft Score: {fp['soft_score']:.6f}")
    print(f"      Matched negation: {fp['neg_pattern']}")
    print(f"      Impression: {fp['impression_snippet'][:150]}...")

# ============================================================================
# AUDIT 2: Check NEGATIVE labels for false negatives
# ============================================================================
print(f"\n{'='*70}")
print("AUDIT 2: Checking NEGATIVE-labeled reports for false negatives")
print("=" * 70)

negatives = df[df['label'] == 0]
neg_sample_indices = random.sample(range(len(negatives)), min(1000, len(negatives)))
neg_sample = negatives.iloc[neg_sample_indices]

false_negatives = []
true_negatives = []
unclear_negatives = []

for _, row in neg_sample.iterrows():
    text = read_report(row['subject_id'], row['study_id'])
    if text is None:
        continue
    
    impression = extract_impression(text)
    findings = extract_findings(text)
    target = impression if impression else (findings if findings else text)
    
    has_neg, neg_pattern = check_for_negation(target)
    has_pos, pos_pattern = check_for_positive(target)
    
    if has_pos and not has_neg:
        false_negatives.append({
            'subject_id': row['subject_id'],
            'study_id': row['study_id'],
            'soft_score': row['soft_score'],
            'pos_pattern': pos_pattern,
            'impression_snippet': (impression or "")[:200]
        })
    elif has_neg:
        true_negatives.append(row['study_id'])
    else:
        unclear_negatives.append(row['study_id'])

total_checked_neg = len(true_negatives) + len(false_negatives) + len(unclear_negatives)
print(f"\nNegative-labeled reports checked: {total_checked_neg}")
print(f"  Likely TRUE negatives:   {len(true_negatives)} ({100*len(true_negatives)/total_checked_neg:.1f}%)")
print(f"  Likely FALSE negatives:  {len(false_negatives)} ({100*len(false_negatives)/total_checked_neg:.1f}%)")
print(f"  Ambiguous/unclear:       {len(unclear_negatives)} ({100*len(unclear_negatives)/total_checked_neg:.1f}%)")

print(f"\n--- Sample FALSE NEGATIVES (first 10) ---")
for i, fn in enumerate(false_negatives[:10]):
    print(f"\n  [{i+1}] p{fn['subject_id']}_{fn['study_id']}")
    print(f"      Soft Score: {fn['soft_score']:.6f}")
    print(f"      Matched positive: {fn['pos_pattern']}")
    print(f"      Impression: {fn['impression_snippet'][:150]}...")

# ============================================================================
# AUDIT 3: Check ALL positives (full sweep for false positive rate)
# ============================================================================
print(f"\n{'='*70}")
print("AUDIT 3: FULL SWEEP of ALL positive-labeled reports")
print("=" * 70)

all_fp_count = 0
all_tp_count = 0
all_unclear_count = 0
all_fp_examples = []

for _, row in positives.iterrows():
    text = read_report(row['subject_id'], row['study_id'])
    if text is None:
        continue
    
    impression = extract_impression(text)
    findings = extract_findings(text)
    target = impression if impression else (findings if findings else text)
    
    has_neg, neg_pattern = check_for_negation(target)
    has_pos, pos_pattern = check_for_positive(target)
    
    if has_neg and not has_pos:
        all_fp_count += 1
        if len(all_fp_examples) < 5:
            all_fp_examples.append({
                'subject_id': row['subject_id'],
                'study_id': row['study_id'],
                'soft_score': row['soft_score'],
                'neg_pattern': neg_pattern,
            })
    elif has_pos:
        all_tp_count += 1
    else:
        all_unclear_count += 1

total_all_pos = all_tp_count + all_fp_count + all_unclear_count
print(f"\nTotal positive-labeled reports scanned: {total_all_pos}")
print(f"  Likely TRUE positives:   {all_tp_count} ({100*all_tp_count/total_all_pos:.1f}%)")
print(f"  Likely FALSE positives:  {all_fp_count} ({100*all_fp_count/total_all_pos:.1f}%)")
print(f"  Ambiguous/unclear:       {all_unclear_count} ({100*all_unclear_count/total_all_pos:.1f}%)")

# ============================================================================
# OVERALL ESTIMATED ACCURACY
# ============================================================================
print(f"\n{'='*70}")
print("OVERALL ESTIMATED LABEL ACCURACY")
print("=" * 70)

# Conservative estimate: ambiguous cases could go either way
# Best case: all ambiguous are correct
# Worst case: all ambiguous are wrong
total_reports = len(df)
total_pos = len(positives)
total_neg = len(negatives)

# Extrapolate false positive rate from full sweep
fp_rate = all_fp_count / total_all_pos if total_all_pos > 0 else 0
# Extrapolate false negative rate from sample
fn_rate = len(false_negatives) / total_checked_neg if total_checked_neg > 0 else 0

estimated_wrong_pos = int(total_pos * fp_rate)
estimated_wrong_neg = int(total_neg * fn_rate)
estimated_total_wrong = estimated_wrong_pos + estimated_wrong_neg
estimated_accuracy = 100 * (total_reports - estimated_total_wrong) / total_reports

print(f"\nDataset size:             {total_reports}")
print(f"Positive labels:          {total_pos}")
print(f"Negative labels:          {total_neg}")
print(f"\nFalse positive rate:      {100*fp_rate:.2f}%  (~{estimated_wrong_pos} reports)")
print(f"False negative rate:      {100*fn_rate:.2f}%  (~{estimated_wrong_neg} reports)")
print(f"Estimated total errors:   ~{estimated_total_wrong}")
print(f"\nESTIMATED OVERALL ACCURACY: {estimated_accuracy:.2f}%")
print(f"{'='*70}")

# Save detailed results
results_df = pd.DataFrame(false_positives)
if len(results_df) > 0:
    results_df.to_csv(r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\audit_false_positives.csv", index=False)
    print(f"\nFalse positive details saved to audit_false_positives.csv")

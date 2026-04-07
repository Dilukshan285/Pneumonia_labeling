"""
================================================================================
STEP 4 — RE-AUDIT CORRECTED LABELS TO VERIFY IMPROVEMENT
================================================================================
Quick verification audit on the corrected CSV to confirm error rate dropped.
Uses same classification logic as the original audit + multiprocessing.
================================================================================
"""

import pandas as pd
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# === Paths ===
CORRECTED_CSV = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\final_pneumonia_labels_corrected.csv"
REPORTS_DIR   = r"c:\Users\dviya\Downloads\mimic-cxr-reports\files"
OUTPUT_DIR    = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output"
NUM_WORKERS   = 12

# === Same keyword/phrase lists as original audit ===
DEFINITE_NEGATIVE_PHRASES = [
    "no pneumonia", "no evidence of pneumonia", "no acute pneumonia",
    "no evidence of acute pneumonia", "no focal pneumonia",
    "no consolidation", "no evidence of consolidation", "no focal consolidation",
    "no new consolidation", "no infiltrate", "no infiltrates",
    "no focal infiltrate", "no new infiltrate", "no new infiltrates",
    "no airspace disease", "no evidence of airspace disease",
    "no focal opacity", "no focal opacities", "no new focal opacity",
    "no new focal opacities", "no airspace opacity",
    "no acute cardiopulmonary process", "no acute cardiopulmonary abnormality",
    "no acute process", "no acute findings", "no acute abnormality",
    "lungs are clear", "clear lungs", "lungs are well aerated",
    "lungs remain clear", "the lungs are clear", "both lungs are clear",
    "unremarkable lungs", "no acute pulmonary disease",
    "without consolidation", "without pneumonia", "without infiltrate",
    "no superimposed pneumonia", "no superimposed consolidation",
    "no pulmonary opacity", "no pulmonary opacities",
]

NEGATION_CONTEXT_PATTERNS = [
    r"no\s+(?:new\s+)?(?:focal\s+)?(?:airspace\s+)?(?:opacity|opacities)\s+(?:to\s+)?suggest(?:ive of)?\s+pneumonia",
    r"no\s+(?:convincing|definite|definitive)\s+(?:evidence\s+(?:of|for)\s+)?pneumonia",
    r"(?:does|do)\s+not\s+(?:suggest|indicate|represent|demonstrate)\s+pneumonia",
    r"not\s+(?:consistent|compatible)\s+with\s+pneumonia",
    r"no\s+(?:findings|evidence)\s+(?:of|for|to\s+suggest)\s+(?:acute\s+)?(?:pneumonia|infection)",
    r"(?:resolved|resolving|improving|improved|cleared)\s+pneumonia",
    r"pneumonia\s+(?:has\s+)?(?:resolved|cleared|improved)",
]

DEFINITE_POSITIVE_PHRASES = [
    "consistent with pneumonia", "compatible with pneumonia",
    "suggestive of pneumonia", "suggesting pneumonia",
    "represents pneumonia", "representing pneumonia",
    "likely pneumonia", "likely represents pneumonia",
    "evidence of pneumonia", "new pneumonia",
    "developing pneumonia", "worsening pneumonia",
    "multifocal pneumonia", "bilateral pneumonia",
    "aspiration pneumonia", "lobar pneumonia",
    "pneumonia is seen", "pneumonia is present",
    "pneumonia is identified", "pneumonia is noted",
    "infectious process", "infectious consolidation",
    "new consolidation", "worsening consolidation",
    "focal consolidation", "lobar consolidation",
    "new infiltrate", "new infiltrates",
    "new airspace disease", "new airspace opacity",
    "air bronchograms", "air bronchogram",
]

UNCERTAINTY_PHRASES = [
    "possible pneumonia", "cannot exclude pneumonia",
    "cannot rule out pneumonia", "questionable pneumonia",
    "suspected pneumonia", "may represent pneumonia",
    "could represent pneumonia", "concern for pneumonia",
    "pneumonia not excluded", "possible consolidation",
    "cannot exclude consolidation", "possible infiltrate",
]

PNEUMONIA_POSITIVE_TERMS = [
    "pneumonia", "bronchopneumonia", "pneumonitis", "consolidation",
    "infiltrate", "infiltrates", "air bronchogram", "airspace disease",
    "airspace opacity", "focal opacity", "patchy opacity", "opacification",
]


def extract_sections(text):
    text_clean = text.replace('\r\n', '\n').replace('\r', '\n')
    impression = ""
    findings = ""
    imp_match = re.search(
        r'(?:IMPRESSION|Impression)\s*:?\s*\n(.*?)(?=\n\s*(?:RECOMMENDATION|NOTIFICATION|ALERT|ATTESTATION|CLINICAL|INDICATION|TECHNIQUE|COMPARISON|HISTORY|WET\s+READ)|$)',
        text_clean, re.DOTALL | re.IGNORECASE
    )
    if imp_match:
        impression = imp_match.group(1).strip()
    find_match = re.search(
        r'(?:FINDINGS|Findings)\s*:?\s*\n(.*?)(?=\n\s*(?:IMPRESSION|Impression|RECOMMENDATION|NOTIFICATION)|$)',
        text_clean, re.DOTALL | re.IGNORECASE
    )
    if find_match:
        findings = find_match.group(1).strip()
    return {'impression': impression, 'findings': findings, 'full': text_clean.lower()}


def classify_report(text):
    sections = extract_sections(text)
    primary = sections['impression'] if sections['impression'] else sections['findings']
    if not primary:
        primary = text
    primary_lower = primary.lower()
    full_lower = sections['full']

    neg_matches = [p for p in DEFINITE_NEGATIVE_PHRASES if p in primary_lower]
    neg_regex = [p for p in NEGATION_CONTEXT_PATTERNS if re.search(p, primary_lower)]
    pos_matches = [p for p in DEFINITE_POSITIVE_PHRASES if p in primary_lower]
    unc_matches = [p for p in UNCERTAINTY_PHRASES if p in primary_lower]
    has_term = any(t in primary_lower for t in PNEUMONIA_POSITIVE_TERMS)
    if not has_term:
        has_term = any(t in full_lower for t in PNEUMONIA_POSITIVE_TERMS)

    if unc_matches and not neg_matches and not pos_matches:
        return "AUDIT_UNCERTAIN"
    if neg_matches and pos_matches:
        if len(neg_matches) >= len(pos_matches):
            return "AUDIT_NEGATIVE"
        return "AUDIT_POSITIVE"
    if unc_matches and neg_matches:
        return "AUDIT_NEGATIVE"
    if unc_matches and pos_matches:
        return "AUDIT_UNCERTAIN"
    if neg_matches:
        return "AUDIT_NEGATIVE"
    if neg_regex:
        return "AUDIT_NEGATIVE"
    if pos_matches:
        return "AUDIT_POSITIVE"
    if unc_matches:
        return "AUDIT_UNCERTAIN"
    if has_term:
        # Sentence-level check
        text_lower = primary_lower
        for term in PNEUMONIA_POSITIVE_TERMS:
            if term in text_lower:
                idx = text_lower.find(term)
                prefix = text_lower[max(0, idx-60):idx]
                neg_words = ['no ', 'not ', 'without ', 'negative ', 'denies ',
                             'absent ', 'free of ', 'rather than ', 'cleared ',
                             'resolved ', 'does not ', 'is not ', 'unlikely ']
                if any(nw in prefix for nw in neg_words):
                    return "AUDIT_NEGATIVE"
                else:
                    return "AUDIT_POSITIVE"
    if not has_term:
        return "AUDIT_NEGATIVE"
    return "AUDIT_UNCLEAR"


def process_batch(batch_data):
    results = []
    for subject_id, study_id, label, soft_score, label_source in batch_data:
        sid = str(subject_id)
        stid = str(study_id)
        prefix = "p" + sid[:2]
        path = os.path.join(REPORTS_DIR, prefix, f"p{sid}", f"{stid}.txt")

        if not os.path.exists(path):
            results.append((sid, stid, label, 'MISSING', False, ''))
            continue
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except:
            results.append((sid, stid, label, 'ERROR', False, ''))
            continue

        audit = classify_report(text)
        pipeline_class = "POSITIVE" if label == 1 else "NEGATIVE"
        mismatch = False
        mtype = ""
        if pipeline_class == "POSITIVE" and audit == "AUDIT_NEGATIVE":
            mismatch = True
            mtype = "FALSE_POSITIVE"
        elif pipeline_class == "NEGATIVE" and audit == "AUDIT_POSITIVE":
            mismatch = True
            mtype = "FALSE_NEGATIVE"
        elif pipeline_class == "POSITIVE" and audit == "AUDIT_UNCERTAIN":
            mismatch = True
            mtype = "POS_UNCERTAIN"
        elif pipeline_class == "NEGATIVE" and audit == "AUDIT_UNCERTAIN":
            mismatch = True
            mtype = "NEG_UNCERTAIN"

        results.append((sid, stid, label, audit, mismatch, mtype))
    return results


def main():
    start = time.time()
    print("=" * 70)
    print("STEP 4 — VERIFICATION AUDIT ON CORRECTED LABELS")
    print("=" * 70)

    df = pd.read_csv(CORRECTED_CSV)
    total = len(df)
    pos_c = (df['label'] == 1).sum()
    neg_c = (df['label'] == 0).sum()
    print(f"\n  Corrected CSV: {total:,} reports ({pos_c:,} POS, {neg_c:,} NEG)")

    data = list(zip(df['subject_id'].values, df['study_id'].values,
                     df['label'].values, df['soft_score'].values, df['label_source'].values))

    batch_size = 500
    batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    print(f"  Processing {len(batches)} batches with {NUM_WORKERS} workers ...\n")

    all_results = []
    done = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_batch, b): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            all_results.extend(future.result())
            done += 1
            if done % 40 == 0 or done == len(batches):
                print(f"  {done}/{len(batches)} batches ({100*done/len(batches):.0f}%)")

    # Analyze
    fp = sum(1 for r in all_results if r[5] == 'FALSE_POSITIVE')
    fn = sum(1 for r in all_results if r[5] == 'FALSE_NEGATIVE')
    pu = sum(1 for r in all_results if r[5] == 'POS_UNCERTAIN')
    nu = sum(1 for r in all_results if r[5] == 'NEG_UNCERTAIN')
    mismatches = fp + fn + pu + nu
    
    audit_pos = sum(1 for r in all_results if r[3] == 'AUDIT_POSITIVE')
    audit_neg = sum(1 for r in all_results if r[3] == 'AUDIT_NEGATIVE')
    audit_unc = sum(1 for r in all_results if r[3] == 'AUDIT_UNCERTAIN')
    audit_unk = sum(1 for r in all_results if r[3] == 'AUDIT_UNCLEAR')
    
    confirmed_correct_pos = sum(1 for r in all_results if r[2] == 1 and r[3] == 'AUDIT_POSITIVE')
    confirmed_correct_neg = sum(1 for r in all_results if r[2] == 0 and r[3] == 'AUDIT_NEGATIVE')
    confirmed_correct = confirmed_correct_pos + confirmed_correct_neg
    confirmed_wrong = fp + fn
    confirmable = confirmed_correct + confirmed_wrong
    accuracy = confirmed_correct / confirmable * 100 if confirmable > 0 else 0
    
    fp_rate = fp / pos_c * 100 if pos_c > 0 else 0
    fn_rate = fn / neg_c * 100 if neg_c > 0 else 0

    elapsed = time.time() - start

    report = []
    report.append("=" * 70)
    report.append("VERIFICATION AUDIT — CORRECTED LABELS")
    report.append("=" * 70)
    report.append(f"")
    report.append(f"Total reports:             {total:>10,}")
    report.append(f"Pipeline POSITIVE:         {pos_c:>10,}")
    report.append(f"Pipeline NEGATIVE:         {neg_c:>10,}")
    report.append(f"")
    report.append(f"--- Audit Classification ---")
    report.append(f"AUDIT_POSITIVE:            {audit_pos:>10,}")
    report.append(f"AUDIT_NEGATIVE:            {audit_neg:>10,}")
    report.append(f"AUDIT_UNCERTAIN:           {audit_unc:>10,}")
    report.append(f"AUDIT_UNCLEAR:             {audit_unk:>10,}")
    report.append(f"")
    report.append(f"--- Mismatches ---")
    report.append(f"Total mismatches:          {mismatches:>10,}  ({mismatches/total*100:.2f}%)")
    report.append(f"  FALSE_POSITIVE:          {fp:>10,}  ({fp_rate:.2f}% of POS)")
    report.append(f"  FALSE_NEGATIVE:          {fn:>10,}  ({fn_rate:.2f}% of NEG)")
    report.append(f"  POS_UNCERTAIN:           {pu:>10,}")
    report.append(f"  NEG_UNCERTAIN:           {nu:>10,}")
    report.append(f"")
    report.append(f"--- Accuracy ---")
    report.append(f"Confirmed correct:         {confirmed_correct:>10,}")
    report.append(f"  Correct POS:             {confirmed_correct_pos:>10,}")
    report.append(f"  Correct NEG:             {confirmed_correct_neg:>10,}")
    report.append(f"Confirmed wrong:           {confirmed_wrong:>10,}")
    report.append(f"Confirmable total:         {confirmable:>10,}")
    report.append(f"")
    report.append(f"ACCURACY (confirmed):      {accuracy:>10.2f}%")
    report.append(f"")
    report.append(f"Time: {elapsed:.1f}s")
    report.append("=" * 70)

    report_text = "\n".join(report)
    print(f"\n{report_text}")

    verify_path = os.path.join(OUTPUT_DIR, "verification_audit_summary.txt")
    with open(verify_path, 'w') as f:
        f.write(report_text)
    print(f"\nSaved to: {verify_path}")


if __name__ == "__main__":
    main()

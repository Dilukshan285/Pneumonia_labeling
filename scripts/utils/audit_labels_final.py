"""
================================================================================
FINAL COMPREHENSIVE AUDIT — Pneumonia Label Accuracy
================================================================================
Scans ALL 203,081 reports in final_pneumonia_labels.csv.
Uses multiprocessing for speed on Ryzen 7 8845HS (16 threads).

For each report:
  1. Reads the original radiology report text
  2. Extracts IMPRESSION and FINDINGS sections
  3. Splits into sentences
  4. Checks each sentence for pneumonia-related terms
  5. Determines if those terms are negated within the sentence
  6. Classifies the report as AUDIT_POSITIVE, AUDIT_NEGATIVE, or AUDIT_UNCLEAR
  7. Compares against the pipeline label to find mismatches

Output:
  - audit_results_full.csv          (all reports with audit classification)
  - audit_false_positives.csv       (labeled POS but report says NEG)
  - audit_false_negatives.csv       (labeled NEG but report says POS)
  - audit_summary.txt               (statistics)
================================================================================
"""

import pandas as pd
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================
LABELS_CSV = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\final_pneumonia_labels.csv"
REPORTS_DIR = r"c:\Users\dviya\Downloads\mimic-cxr-reports\files"
OUTPUT_DIR  = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output"
NUM_WORKERS = 12  # Use 12 of 16 threads, leave 4 for system

# ============================================================================
# PNEUMONIA KEYWORD SETS (case-insensitive matching)
# ============================================================================

# --- Terms that indicate pneumonia IS present ---
PNEUMONIA_POSITIVE_TERMS = [
    "pneumonia",
    "bronchopneumonia",
    "pneumonitis",
    "consolidation",
    "infiltrate",
    "infiltrates",
    "air bronchogram",
    "air bronchograms",
    "airspace disease",
    "airspace opacity",
    "airspace opacities",
    "alveolar opacity",
    "alveolar opacities",
    "pulmonary opacity",
    "pulmonary opacities",
    "focal opacity",
    "focal opacities",
    "patchy opacity",
    "patchy opacities",
    "opacification",
    "perihilar infiltrate",
    "perihilar infiltrates",
]

# --- Phrases that DEFINITIVELY negate pneumonia ---
DEFINITE_NEGATIVE_PHRASES = [
    "no pneumonia",
    "no evidence of pneumonia",
    "no acute pneumonia",
    "no evidence of acute pneumonia",
    "no focal pneumonia",
    "no evidence of focal pneumonia",
    "no consolidation",
    "no evidence of consolidation",
    "no focal consolidation",
    "no evidence of focal consolidation",
    "no new consolidation",
    "no infiltrate",
    "no infiltrates",
    "no focal infiltrate",
    "no focal infiltrates",
    "no new infiltrate",
    "no new infiltrates",
    "no evidence of infiltrate",
    "no airspace disease",
    "no evidence of airspace disease",
    "no focal airspace disease",
    "no acute airspace disease",
    "no focal opacity",
    "no focal opacities",
    "no new focal opacity",
    "no new focal opacities",
    "no airspace opacity",
    "no airspace opacities",
    "no acute cardiopulmonary process",
    "no acute cardiopulmonary abnormality",
    "no acute cardiopulmonary disease",
    "no acute process",
    "no acute findings",
    "no acute abnormality",
    "lungs are clear",
    "clear lungs",
    "lungs are well aerated",
    "lungs are well expanded",
    "lungs remain clear",
    "lung fields are clear",
    "the lungs are clear",
    "both lungs are clear",
    "unremarkable lungs",
    "no acute pulmonary disease",
    "no acute pulmonary process",
    "no acute lung disease",
    "without consolidation",
    "without pneumonia",
    "without infiltrate",
    "without focal consolidation",
    "without evidence of pneumonia",
    "no opacification",
    "no new opacification",
    "no evidence of opacification",
    "negative for pneumonia",
    "rules out pneumonia",
    "ruled out pneumonia",
    "rule out pneumonia",
    "no superimposed pneumonia",
    "no superimposed consolidation",
    "no superimposed infiltrate",
    "no superimposed infection",
    "no pulmonary opacity",
    "no pulmonary opacities",
    "no new pulmonary opacity",
    "no patchy opacity",
    "no patchy opacities",
    "no air bronchogram",
    "no air bronchograms",
]

# --- Phrases that SUGGEST pneumonia but with negation context ---
# These appear in sentences that mention pneumonia but negate it
NEGATION_CONTEXT_PATTERNS = [
    r"no\s+(?:new\s+)?(?:focal\s+)?(?:airspace\s+)?(?:opacity|opacities)\s+(?:to\s+)?suggest(?:ive of)?\s+pneumonia",
    r"no\s+(?:new\s+)?(?:focal\s+)?(?:airspace\s+)?(?:opacity|opacities)\s+(?:to\s+)?(?:indicate|represent)\s+pneumonia",
    r"no\s+(?:convincing|definite|definitive|definitive)\s+(?:evidence\s+(?:of|for)\s+)?pneumonia",
    r"no\s+(?:convincing|definite|definitive)\s+(?:evidence\s+(?:of|for)\s+)?consolidation",
    r"no\s+(?:convincing|definite|definitive)\s+(?:evidence\s+(?:of|for)\s+)?infiltrate",
    r"(?:unlikely|doubtful)\s+(?:to\s+represent\s+)?pneumonia",
    r"pneumonia\s+(?:is\s+)?(?:unlikely|doubtful|excluded|ruled\s+out)",
    r"(?:does|do)\s+not\s+(?:suggest|indicate|represent|demonstrate)\s+pneumonia",
    r"(?:does|do)\s+not\s+(?:suggest|indicate|represent|demonstrate)\s+consolidation",
    r"not\s+(?:consistent|compatible)\s+with\s+pneumonia",
    r"not\s+(?:consistent|compatible)\s+with\s+consolidation",
    r"(?:findings|appearance)\s+(?:are|is)\s+not\s+(?:consistent|compatible|suggestive)\s+(?:with|of)\s+pneumonia",
    r"no\s+(?:findings|evidence)\s+(?:of|for|to\s+suggest)\s+(?:acute\s+)?(?:pneumonia|infection|infectious\s+process)",
    r"(?:resolved|resolving|improving|improved|cleared)\s+pneumonia",
    r"pneumonia\s+(?:has\s+)?(?:resolved|cleared|improved)",
    r"(?:previous|prior|known)\s+pneumonia\s+(?:has\s+)?(?:resolved|cleared|improved)",
    r"no\s+(?:remaining|residual)\s+(?:pneumonia|consolidation|infiltrate)",
]

# --- Phrases that indicate TRUE POSITIVE pneumonia ---
DEFINITE_POSITIVE_PHRASES = [
    "consistent with pneumonia",
    "compatible with pneumonia",
    "suggestive of pneumonia",
    "suggesting pneumonia",
    "represents pneumonia",
    "representing pneumonia",
    "likely pneumonia",
    "likely represents pneumonia",
    "likely representing pneumonia",
    "findings of pneumonia",
    "evidence of pneumonia",
    "diagnosed with pneumonia",
    "confirms pneumonia",
    "confirmed pneumonia",
    "new pneumonia",
    "developing pneumonia",
    "worsening pneumonia",
    "progressive pneumonia",
    "multifocal pneumonia",
    "bilateral pneumonia",
    "aspiration pneumonia",
    "community-acquired pneumonia",
    "hospital-acquired pneumonia",
    "right lower lobe pneumonia",
    "left lower lobe pneumonia",
    "right upper lobe pneumonia",
    "left upper lobe pneumonia",
    "right middle lobe pneumonia",
    "basilar pneumonia",
    "lobar pneumonia",
    "lingular pneumonia",
    "pneumonia is seen",
    "pneumonia is present",
    "pneumonia is identified",
    "pneumonia is noted",
    "pneumonia is demonstrated",
    "pneumonia is suspected",
    "infectious process",
    "infectious consolidation",
    "findings consistent with consolidation",
    "consolidation consistent with infection",
    "consolidation representing infection",
    "new consolidation",
    "worsening consolidation",
    "increasing consolidation",
    "focal consolidation",
    "lobar consolidation",
    "new infiltrate",
    "new infiltrates",
    "worsening infiltrate",
    "new airspace disease",
    "new airspace opacity",
    "air bronchograms",
    "air bronchogram",
]

# --- UNCERTAINTY phrases (report truly hedges) ---
UNCERTAINTY_PHRASES = [
    "possible pneumonia",
    "possibly pneumonia",
    "cannot exclude pneumonia",
    "cannot rule out pneumonia",
    "cannot be excluded",
    "questionable pneumonia",
    "suspected pneumonia",
    "may represent pneumonia",
    "could represent pneumonia",
    "concern for pneumonia",
    "pneumonia not excluded",
    "differential includes pneumonia",
    "versus pneumonia",
    "vs pneumonia",
    "atelectasis versus pneumonia",
    "atelectasis vs pneumonia",
    "superimposed pneumonia cannot be excluded",
    "superimposed infection cannot be excluded",
    "possible superimposed pneumonia",
    "possible superimposed infection",
    "may represent infection",
    "could represent infection",
    "possible consolidation",
    "possible infiltrate",
    "cannot exclude consolidation",
    "cannot rule out consolidation",
    "infection cannot be excluded",
    "infectious process cannot be excluded",
    "pneumonia is not excluded",
    "consolidation is not excluded",
]


# ============================================================================
# TEXT PROCESSING FUNCTIONS
# ============================================================================

def extract_sections(text):
    """
    Extract IMPRESSION, FINDINGS, and FULL text from a report.
    Returns dict with keys: impression, findings, full
    """
    text_clean = text.replace('\r\n', '\n').replace('\r', '\n')
    
    impression = ""
    findings = ""
    
    # Extract IMPRESSION
    imp_match = re.search(
        r'(?:IMPRESSION|Impression)\s*:?\s*\n(.*?)(?=\n\s*(?:RECOMMENDATION|NOTIFICATION|ALERT|ATTESTATION|CLINICAL|INDICATION|TECHNIQUE|COMPARISON|HISTORY|WET\s+READ)|$)',
        text_clean, re.DOTALL | re.IGNORECASE
    )
    if imp_match:
        impression = imp_match.group(1).strip()
    
    # Extract FINDINGS
    find_match = re.search(
        r'(?:FINDINGS|Findings)\s*:?\s*\n(.*?)(?=\n\s*(?:IMPRESSION|Impression|RECOMMENDATION|NOTIFICATION)|$)',
        text_clean, re.DOTALL | re.IGNORECASE
    )
    if find_match:
        findings = find_match.group(1).strip()
    
    return {
        'impression': impression,
        'findings': findings,
        'full': text_clean.lower()
    }


def split_sentences(text):
    """Split text into sentences."""
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Split on period followed by space+capital or double space or numbered list
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])|(?<=[.!?])\s{2,}|\n\s*\d+\.', text)
    # Also split on newlines that look like separate items
    result = []
    for s in sentences:
        parts = s.strip().split('\n')
        for p in parts:
            p = p.strip()
            if len(p) > 5:  # Skip tiny fragments
                result.append(p)
    return result


def classify_report(text):
    """
    Classify a single report as POSITIVE, NEGATIVE, UNCERTAIN, or UNCLEAR.
    
    Strategy:
    1. Extract IMPRESSION (highest priority) and FINDINGS
    2. Check for definite negative phrases first (entire impression/findings)
    3. Check for definite positive phrases
    4. Check for uncertainty phrases
    5. Check for negation context patterns (regex-based)
    6. Sentence-level analysis for remaining cases
    
    Returns: (classification, reason, matched_pattern)
    """
    sections = extract_sections(text)
    
    # Use impression if available, otherwise findings, otherwise full text
    primary = sections['impression'] if sections['impression'] else sections['findings']
    if not primary:
        primary = text
    
    primary_lower = primary.lower()
    full_lower = sections['full']
    
    # --- Step 1: Check definite negative phrases in primary section ---
    neg_matches = []
    for phrase in DEFINITE_NEGATIVE_PHRASES:
        if phrase in primary_lower:
            neg_matches.append(phrase)
    
    # --- Step 2: Check negation context patterns (regex) ---
    neg_regex_matches = []
    for pattern in NEGATION_CONTEXT_PATTERNS:
        if re.search(pattern, primary_lower):
            neg_regex_matches.append(pattern)
    
    # --- Step 3: Check definite positive phrases ---
    pos_matches = []
    for phrase in DEFINITE_POSITIVE_PHRASES:
        if phrase in primary_lower:
            pos_matches.append(phrase)
    
    # --- Step 4: Check uncertainty phrases ---
    unc_matches = []
    for phrase in UNCERTAINTY_PHRASES:
        if phrase in primary_lower:
            unc_matches.append(phrase)
    
    # --- Step 5: Check if any pneumonia-related term exists at all ---
    has_pneumonia_term = False
    for term in PNEUMONIA_POSITIVE_TERMS:
        if term in primary_lower:
            has_pneumonia_term = True
            break
    
    if not has_pneumonia_term:
        # Also check full text
        for term in PNEUMONIA_POSITIVE_TERMS:
            if term in full_lower:
                has_pneumonia_term = True
                break
    
    # =========================================================================
    # CLASSIFICATION LOGIC (priority order)
    # =========================================================================
    
    # CASE 1: Has uncertainty phrases — classify as UNCERTAIN
    if unc_matches and not neg_matches and not pos_matches:
        return "AUDIT_UNCERTAIN", "uncertainty_phrase", "; ".join(unc_matches[:3])
    
    # CASE 2: Has BOTH negative AND positive phrases — need to resolve conflict
    if neg_matches and pos_matches:
        # If impression has negative conclusion, trust it
        # Check last sentence of impression for the final word
        sentences = split_sentences(primary)
        if sentences:
            last_sentence = sentences[-1].lower()
            # Check if last sentence is negative
            for phrase in DEFINITE_NEGATIVE_PHRASES:
                if phrase in last_sentence:
                    return "AUDIT_NEGATIVE", "neg_overrides_pos_last_sentence", f"NEG:[{neg_matches[0]}] vs POS:[{pos_matches[0]}]"
            # Check if last sentence is positive
            for phrase in DEFINITE_POSITIVE_PHRASES:
                if phrase in last_sentence:
                    return "AUDIT_POSITIVE", "pos_overrides_neg_last_sentence", f"POS:[{pos_matches[0]}] vs NEG:[{neg_matches[0]}]"
        
        # If negation phrases outnumber positive, lean negative
        if len(neg_matches) > len(pos_matches):
            return "AUDIT_NEGATIVE", "more_neg_than_pos", f"NEG({len(neg_matches)}) vs POS({len(pos_matches)})"
        elif len(pos_matches) > len(neg_matches):
            return "AUDIT_POSITIVE", "more_pos_than_neg", f"POS({len(pos_matches)}) vs NEG({len(neg_matches)})"
        else:
            return "AUDIT_UNCERTAIN", "equal_neg_pos_conflict", f"NEG:[{neg_matches[0]}] vs POS:[{pos_matches[0]}]"
    
    # CASE 3: Has uncertainty + negative → lean negative
    if unc_matches and neg_matches:
        return "AUDIT_NEGATIVE", "neg_with_uncertainty", f"NEG:[{neg_matches[0]}] UNC:[{unc_matches[0]}]"
    
    # CASE 4: Has uncertainty + positive → lean uncertain
    if unc_matches and pos_matches:
        return "AUDIT_UNCERTAIN", "pos_with_uncertainty", f"POS:[{pos_matches[0]}] UNC:[{unc_matches[0]}]"
    
    # CASE 5: Only negative phrases found
    if neg_matches:
        return "AUDIT_NEGATIVE", "definite_negative_phrase", neg_matches[0]
    
    # CASE 6: Only negative regex patterns found
    if neg_regex_matches:
        return "AUDIT_NEGATIVE", "negation_context_pattern", neg_regex_matches[0][:80]
    
    # CASE 7: Only positive phrases found
    if pos_matches:
        return "AUDIT_POSITIVE", "definite_positive_phrase", pos_matches[0]
    
    # CASE 8: Only uncertainty phrases found
    if unc_matches:
        return "AUDIT_UNCERTAIN", "uncertainty_only", unc_matches[0]
    
    # CASE 9: Has pneumonia term but no clear classification
    # Do sentence-level negation check
    if has_pneumonia_term:
        sentences = split_sentences(primary)
        for sent in sentences:
            sent_lower = sent.lower()
            for term in PNEUMONIA_POSITIVE_TERMS:
                if term in sent_lower:
                    # Check if the sentence negates the term
                    # Look for "no", "without", "not", "negative" before the term
                    term_pos = sent_lower.find(term)
                    prefix = sent_lower[:term_pos]
                    
                    # Check for negation words in the prefix (within ~60 chars)
                    prefix_check = prefix[-60:] if len(prefix) > 60 else prefix
                    negation_words = ['no ', 'not ', 'without ', 'negative ', 'denies ',
                                      'absent ', 'free of ', 'lack of ', 'rather than ',
                                      'cleared ', 'resolved ', 'no evidence ',
                                      'does not ', 'do not ', 'did not ',
                                      'is not ', 'are not ', 'was not ', 'were not ',
                                      'unlikely ', 'doubtful ']
                    
                    is_negated = any(nw in prefix_check for nw in negation_words)
                    
                    # Also check for "to suggest" / "to indicate" patterns after "no...opacity"
                    if not is_negated:
                        # Check broader sentence for "no...to suggest [term]"
                        suggest_pattern = re.search(
                            r'no\s+.{0,40}?\s+(?:to\s+)?(?:suggest|indicate|represent)',
                            sent_lower[:term_pos]
                        )
                        if suggest_pattern:
                            is_negated = True
                    
                    if is_negated:
                        return "AUDIT_NEGATIVE", "sentence_negation", f"'{sent[:100]}...'"
                    else:
                        # Has positive term without negation
                        return "AUDIT_POSITIVE", "sentence_positive_term", f"term='{term}' in '{sent[:100]}...'"
    
    # CASE 10: No pneumonia-related terms at all
    if not has_pneumonia_term:
        return "AUDIT_NEGATIVE", "no_pneumonia_terms", "no relevant terms found"
    
    # CASE 11: Fallthrough — genuinely unclear
    return "AUDIT_UNCLEAR", "unresolved", "could not classify"


# ============================================================================
# WORKER FUNCTION (runs in separate process)
# ============================================================================

def process_batch(batch_data):
    """
    Process a batch of (subject_id, study_id, label, soft_score, label_source) tuples.
    Returns list of result dicts.
    """
    results = []
    
    for subject_id, study_id, label, soft_score, label_source in batch_data:
        sid = str(subject_id)
        stid = str(study_id)
        
        # Build report path
        prefix = "p" + sid[:2]
        report_path = os.path.join(REPORTS_DIR, prefix, f"p{sid}", f"{stid}.txt")
        
        if not os.path.exists(report_path):
            results.append({
                'subject_id': sid,
                'study_id': stid,
                'pipeline_label': label,
                'soft_score': soft_score,
                'label_source': label_source,
                'audit_label': 'MISSING',
                'audit_reason': 'report_file_not_found',
                'audit_detail': report_path,
                'mismatch': False,
                'mismatch_type': '',
            })
            continue
        
        # Read report
        try:
            with open(report_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except Exception as e:
            results.append({
                'subject_id': sid,
                'study_id': stid,
                'pipeline_label': label,
                'soft_score': soft_score,
                'label_source': label_source,
                'audit_label': 'ERROR',
                'audit_reason': 'read_error',
                'audit_detail': str(e),
                'mismatch': False,
                'mismatch_type': '',
            })
            continue
        
        # Classify
        audit_label, audit_reason, audit_detail = classify_report(text)
        
        # Determine mismatch
        pipeline_class = "POSITIVE" if label == 1 else "NEGATIVE"
        mismatch = False
        mismatch_type = ""
        
        if pipeline_class == "POSITIVE" and audit_label == "AUDIT_NEGATIVE":
            mismatch = True
            mismatch_type = "FALSE_POSITIVE"
        elif pipeline_class == "NEGATIVE" and audit_label == "AUDIT_POSITIVE":
            mismatch = True
            mismatch_type = "FALSE_NEGATIVE"
        elif pipeline_class == "POSITIVE" and audit_label == "AUDIT_UNCERTAIN":
            mismatch = True
            mismatch_type = "POS_BUT_UNCERTAIN"
        elif pipeline_class == "NEGATIVE" and audit_label == "AUDIT_UNCERTAIN":
            mismatch = True
            mismatch_type = "NEG_BUT_UNCERTAIN"
        
        results.append({
            'subject_id': sid,
            'study_id': stid,
            'pipeline_label': label,
            'soft_score': soft_score,
            'label_source': label_source,
            'audit_label': audit_label,
            'audit_reason': audit_reason,
            'audit_detail': str(audit_detail)[:200],
            'mismatch': mismatch,
            'mismatch_type': mismatch_type,
        })
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = time.time()
    
    print("=" * 70)
    print("FINAL COMPREHENSIVE LABEL AUDIT")
    print(f"Workers: {NUM_WORKERS}")
    print("=" * 70)
    
    # Load labels
    print("\n[1/5] Loading final_pneumonia_labels.csv ...")
    df = pd.read_csv(LABELS_CSV)
    total = len(df)
    pos_count = (df['label'] == 1).sum()
    neg_count = (df['label'] == 0).sum()
    print(f"  Total reports:  {total:,}")
    print(f"  Positive (1):   {pos_count:,}")
    print(f"  Negative (0):   {neg_count:,}")
    
    # Prepare data tuples
    data_tuples = list(zip(
        df['subject_id'].values,
        df['study_id'].values,
        df['label'].values,
        df['soft_score'].values,
        df['label_source'].values,
    ))
    
    # Split into batches
    batch_size = 500
    batches = [data_tuples[i:i+batch_size] for i in range(0, len(data_tuples), batch_size)]
    print(f"\n[2/5] Processing {len(batches)} batches of ~{batch_size} reports each ...")
    print(f"  Using {NUM_WORKERS} parallel workers\n")
    
    # Process with multiprocessing
    all_results = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
        
        for future in as_completed(futures):
            batch_results = future.result()
            all_results.extend(batch_results)
            completed += 1
            
            if completed % 20 == 0 or completed == len(batches):
                elapsed = time.time() - start_time
                pct = 100 * completed / len(batches)
                reports_done = len(all_results)
                rate = reports_done / elapsed if elapsed > 0 else 0
                eta = (total - reports_done) / rate if rate > 0 else 0
                print(f"  Progress: {completed}/{len(batches)} batches "
                      f"({pct:.1f}%) | {reports_done:,} reports | "
                      f"{rate:.0f} reports/sec | ETA: {eta:.0f}s")
    
    # Convert to DataFrame
    print(f"\n[3/5] Building results DataFrame ...")
    results_df = pd.DataFrame(all_results)
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    print(f"\n[4/5] Computing statistics ...\n")
    
    total_processed = len(results_df)
    missing = (results_df['audit_label'] == 'MISSING').sum()
    errors = (results_df['audit_label'] == 'ERROR').sum()
    valid = total_processed - missing - errors
    
    # Audit classifications
    audit_counts = results_df['audit_label'].value_counts()
    
    # Mismatch analysis
    mismatches = results_df[results_df['mismatch'] == True]
    mismatch_types = mismatches['mismatch_type'].value_counts()
    
    false_positives = results_df[results_df['mismatch_type'] == 'FALSE_POSITIVE']
    false_negatives = results_df[results_df['mismatch_type'] == 'FALSE_NEGATIVE']
    pos_uncertain  = results_df[results_df['mismatch_type'] == 'POS_BUT_UNCERTAIN']
    neg_uncertain  = results_df[results_df['mismatch_type'] == 'NEG_BUT_UNCERTAIN']
    
    # Among pipeline positives
    pipeline_pos = results_df[results_df['pipeline_label'] == 1]
    pipeline_neg = results_df[results_df['pipeline_label'] == 0]
    
    fp_rate = len(false_positives) / len(pipeline_pos) * 100 if len(pipeline_pos) > 0 else 0
    fn_rate = len(false_negatives) / len(pipeline_neg) * 100 if len(pipeline_neg) > 0 else 0
    pos_unc_rate = len(pos_uncertain) / len(pipeline_pos) * 100 if len(pipeline_pos) > 0 else 0
    neg_unc_rate = len(neg_uncertain) / len(pipeline_neg) * 100 if len(pipeline_neg) > 0 else 0
    
    # Overall accuracy (excluding UNCLEAR and UNCERTAIN audit results)
    confirmed_correct_pos = len(pipeline_pos[pipeline_pos['audit_label'] == 'AUDIT_POSITIVE'])
    confirmed_correct_neg = len(pipeline_neg[pipeline_neg['audit_label'] == 'AUDIT_NEGATIVE'])
    confirmed_correct = confirmed_correct_pos + confirmed_correct_neg
    confirmed_wrong = len(false_positives) + len(false_negatives)
    confirmable = confirmed_correct + confirmed_wrong
    
    accuracy_confirmable = confirmed_correct / confirmable * 100 if confirmable > 0 else 0
    
    # Build summary text
    summary = []
    summary.append("=" * 70)
    summary.append("COMPREHENSIVE LABEL AUDIT — FINAL RESULTS")
    summary.append("=" * 70)
    summary.append(f"")
    summary.append(f"Total reports in CSV:       {total:>10,}")
    summary.append(f"Reports processed:          {total_processed:>10,}")
    summary.append(f"Missing report files:       {missing:>10,}")
    summary.append(f"Read errors:                {errors:>10,}")
    summary.append(f"Valid reports analyzed:      {valid:>10,}")
    summary.append(f"")
    summary.append("-" * 70)
    summary.append("PIPELINE LABEL DISTRIBUTION")
    summary.append("-" * 70)
    summary.append(f"Pipeline POSITIVE (1):      {pos_count:>10,}")
    summary.append(f"Pipeline NEGATIVE (0):      {neg_count:>10,}")
    summary.append(f"")
    summary.append("-" * 70)
    summary.append("AUDIT CLASSIFICATION DISTRIBUTION")
    summary.append("-" * 70)
    for label, count in audit_counts.items():
        pct = count / total_processed * 100
        summary.append(f"  {label:<25s} {count:>10,}  ({pct:.2f}%)")
    summary.append(f"")
    summary.append("-" * 70)
    summary.append("MISMATCH ANALYSIS (Pipeline vs Audit)")
    summary.append("-" * 70)
    summary.append(f"Total mismatches:           {len(mismatches):>10,}  ({len(mismatches)/valid*100:.2f}% of valid)")
    summary.append(f"")
    for mtype, count in mismatch_types.items():
        summary.append(f"  {mtype:<25s} {count:>10,}")
    summary.append(f"")
    summary.append("-" * 70)
    summary.append("FALSE POSITIVE ANALYSIS (Labeled POS, Report says NEG)")
    summary.append("-" * 70)
    summary.append(f"False positives found:      {len(false_positives):>10,}")
    summary.append(f"False positive rate:        {fp_rate:>10.2f}%  (of {len(pipeline_pos):,} pipeline positives)")
    summary.append(f"")
    if len(false_positives) > 0:
        fp_reasons = false_positives['audit_reason'].value_counts()
        summary.append(f"  Breakdown by reason:")
        for reason, count in fp_reasons.items():
            summary.append(f"    {reason:<40s} {count:>6,}")
    summary.append(f"")
    summary.append("-" * 70)
    summary.append("FALSE NEGATIVE ANALYSIS (Labeled NEG, Report says POS)")
    summary.append("-" * 70)
    summary.append(f"False negatives found:      {len(false_negatives):>10,}")
    summary.append(f"False negative rate:        {fn_rate:>10.2f}%  (of {len(pipeline_neg):,} pipeline negatives)")
    summary.append(f"")
    if len(false_negatives) > 0:
        fn_reasons = false_negatives['audit_reason'].value_counts()
        summary.append(f"  Breakdown by reason:")
        for reason, count in fn_reasons.items():
            summary.append(f"    {reason:<40s} {count:>6,}")
    summary.append(f"")
    summary.append("-" * 70)
    summary.append("UNCERTAIN MISMATCHES")
    summary.append("-" * 70)
    summary.append(f"Pipeline POS → Audit UNCERTAIN: {len(pos_uncertain):>8,}  ({pos_unc_rate:.2f}%)")
    summary.append(f"Pipeline NEG → Audit UNCERTAIN: {len(neg_uncertain):>8,}  ({neg_unc_rate:.2f}%)")
    summary.append(f"")
    summary.append("-" * 70)
    summary.append("ACCURACY ESTIMATE")
    summary.append("-" * 70)
    summary.append(f"Confirmed correct labels:   {confirmed_correct:>10,}")
    summary.append(f"  - Correct positives:      {confirmed_correct_pos:>10,}")
    summary.append(f"  - Correct negatives:      {confirmed_correct_neg:>10,}")
    summary.append(f"Confirmed wrong labels:     {confirmed_wrong:>10,}")
    summary.append(f"  - False positives:        {len(false_positives):>10,}")
    summary.append(f"  - False negatives:        {len(false_negatives):>10,}")
    summary.append(f"Confirmable total:          {confirmable:>10,}")
    summary.append(f"")
    summary.append(f"ACCURACY (confirmed):       {accuracy_confirmable:>10.2f}%")
    summary.append(f"")
    summary.append(f"  Note: {valid - confirmable:,} reports could not be confirmed")
    summary.append(f"  either way (AUDIT_UNCLEAR, AUDIT_UNCERTAIN, or no pneumonia terms).")
    summary.append(f"  The true accuracy is likely between {accuracy_confirmable:.1f}% and")
    summary.append(f"  {(confirmed_correct + (valid - confirmable)) / valid * 100:.1f}% depending on how those resolve.")
    summary.append(f"")
    elapsed_total = time.time() - start_time
    summary.append(f"Audit completed in {elapsed_total:.1f} seconds ({elapsed_total/60:.1f} minutes)")
    summary.append("=" * 70)
    
    # Print summary
    summary_text = "\n".join(summary)
    print(summary_text)
    
    # ========================================================================
    # SAVE FILES
    # ========================================================================
    print(f"\n[5/5] Saving output files ...")
    
    # Full results
    full_path = os.path.join(OUTPUT_DIR, "audit_results_full.csv")
    results_df.to_csv(full_path, index=False)
    print(f"  Full results:          {full_path}")
    
    # False positives only
    fp_path = os.path.join(OUTPUT_DIR, "audit_false_positives.csv")
    false_positives.to_csv(fp_path, index=False)
    print(f"  False positives:       {fp_path} ({len(false_positives):,} rows)")
    
    # False negatives only
    fn_path = os.path.join(OUTPUT_DIR, "audit_false_negatives.csv")
    false_negatives.to_csv(fn_path, index=False)
    print(f"  False negatives:       {fn_path} ({len(false_negatives):,} rows)")
    
    # Summary text
    summary_path = os.path.join(OUTPUT_DIR, "audit_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"  Summary:               {summary_path}")
    
    # Print sample false positives
    if len(false_positives) > 0:
        print(f"\n{'='*70}")
        print(f"SAMPLE FALSE POSITIVES (first 15)")
        print(f"{'='*70}")
        for _, row in false_positives.head(15).iterrows():
            print(f"\n  p{row['subject_id']}_{row['study_id']}")
            print(f"    Pipeline: POSITIVE (soft_score={row['soft_score']:.6f})")
            print(f"    Audit:    NEGATIVE ({row['audit_reason']})")
            print(f"    Detail:   {row['audit_detail'][:120]}")
    
    # Print sample false negatives
    if len(false_negatives) > 0:
        print(f"\n{'='*70}")
        print(f"SAMPLE FALSE NEGATIVES (first 15)")
        print(f"{'='*70}")
        for _, row in false_negatives.head(15).iterrows():
            print(f"\n  p{row['subject_id']}_{row['study_id']}")
            print(f"    Pipeline: NEGATIVE (soft_score={row['soft_score']:.6f})")
            print(f"    Audit:    POSITIVE ({row['audit_reason']})")
            print(f"    Detail:   {row['audit_detail'][:120]}")
    
    print(f"\n{'='*70}")
    print(f"AUDIT COMPLETE — Total time: {elapsed_total:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

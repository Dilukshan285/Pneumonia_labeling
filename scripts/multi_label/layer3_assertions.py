"""
Layer 3: Sentence-Level Assertion Classification
===================================================
Splits each report into individual sentences and classifies each
mention of a pathology keyword as PRESENT, ABSENT, UNCERTAIN, 
or HYPOTHETICAL using regex-based assertion patterns.

This captures nuances that keyword-level and document-level NLI miss:
  - "pneumonia has resolved" → ABSENT (temporal change)
  - "possible atelectasis" → UNCERTAIN 
  - "if clinical concern, consider pneumonia" → HYPOTHETICAL → UNCERTAIN
  - "worsening consolidation" → PRESENT (active finding)
"""

import re

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from multi_label_config import (
    POSITIVE_KEYWORDS, PATHOLOGY_CLASSES,
    ASSERTION_PRESENT_PATTERNS, ASSERTION_ABSENT_PATTERNS,
    ASSERTION_UNCERTAIN_PATTERNS,
    LABEL_PRESENT, LABEL_ABSENT, LABEL_UNCERTAIN,
)


def _split_sentences(text):
    """
    Split radiology report text into sentences.
    Handles common radiology formatting (numbered lists, abbreviations).
    """
    if not text or str(text).strip() in ("", "nan", "None"):
        return []
    
    text = str(text).strip()
    
    # Normalize whitespace (reports have odd line breaks)
    text = re.sub(r'\s+', ' ', text)
    
    # Protect common abbreviations from sentence splitting
    protected = text
    abbrevs = ["Dr.", "Mr.", "Mrs.", "Ms.", "vs.", "etc.", "i.e.", "e.g.", 
               "a.m.", "p.m.", "cm.", "mm."]
    for abbr in abbrevs:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    
    # Also split on numbered lists (e.g., "1. Finding one 2. Finding two")
    expanded = []
    for sent in sentences:
        parts = re.split(r'(?:^|\s)(\d+\.)\s+', sent)
        for p in parts:
            p = p.strip()
            if p and not re.match(r'^\d+\.$', p):
                expanded.append(p.replace("<DOT>", "."))
    
    return [s for s in expanded if len(s) > 5]


def _find_keywords_in_sentence(sentence, keywords):
    """Find which keywords from a list appear in the sentence."""
    sent_lower = sentence.lower()
    found = []
    for kw in keywords:
        if kw.lower() in sent_lower:
            found.append(kw)
    return found


def _classify_assertion(sentence, keyword):
    """
    Classify the assertion status of a keyword within a sentence.
    
    Returns: "PRESENT", "ABSENT", "UNCERTAIN"
    """
    sent_lower = sentence.lower()
    kw_lower = keyword.lower()
    
    # Use a representative short keyword for pattern matching
    # (some keywords are multi-word; use the most specific part)
    kw_pattern = re.escape(kw_lower)
    
    # --- Check UNCERTAIN patterns first (takes priority) ---
    for pattern_template in ASSERTION_UNCERTAIN_PATTERNS:
        pattern = pattern_template.replace(r"\b{kw}\b", r"\b" + kw_pattern + r"\b")
        try:
            if re.search(pattern, sent_lower):
                return "UNCERTAIN"
        except re.error:
            continue
    
    # --- Check ABSENT patterns ---
    for pattern_template in ASSERTION_ABSENT_PATTERNS:
        pattern = pattern_template.replace(r"\b{kw}\b", r"\b" + kw_pattern + r"\b")
        try:
            if re.search(pattern, sent_lower):
                return "ABSENT"
        except re.error:
            continue
    
    # Additional explicit negation checks
    negation_phrases = [
        f"no {kw_lower}", f"no evidence of {kw_lower}",
        f"without {kw_lower}", f"not {kw_lower}",
        f"negative for {kw_lower}", f"rules out {kw_lower}",
        f"ruled out {kw_lower}",
    ]
    for phrase in negation_phrases:
        if phrase in sent_lower:
            return "ABSENT"
    
    # Check for resolved/historical
    resolved_patterns = [
        f"{kw_lower}.*(?:has |have )?(?:resolved|cleared|improved|removed)",
        f"(?:resolved|cleared|improved|prior|old|previous|remote|healed).*{kw_lower}",
    ]
    for pat in resolved_patterns:
        try:
            if re.search(pat, sent_lower):
                return "ABSENT"
        except re.error:
            continue
    
    # --- Check PRESENT patterns ---
    for pattern_template in ASSERTION_PRESENT_PATTERNS:
        pattern = pattern_template.replace(r"\b{kw}\b", r"\b" + kw_pattern + r"\b")
        try:
            if re.search(pattern, sent_lower):
                return "PRESENT"
        except re.error:
            continue
    
    # Default: if the keyword is mentioned without negation or uncertainty,
    # assume PRESENT (the simplest reading)
    if kw_lower in sent_lower:
        return "PRESENT"
    
    return "ABSENT"


def classify_report_assertions(impression_text, findings_text):
    """
    Classify a single report using sentence-level assertion detection.
    
    Args:
        impression_text: Impression section text
        findings_text: Findings section text
    
    Returns:
        dict: {pathology: label} where label is 1/0/-1
        dict: {pathology: {"present": n, "absent": n, "uncertain": n, "sentences": [...]}}
    """
    results = {}
    details = {}
    
    # Split into sentences (impression gets higher weight)
    imp_sentences = _split_sentences(impression_text)
    find_sentences = _split_sentences(findings_text)
    
    # Weight: impression sentences count 1.5x
    weighted_sentences = [(s, 1.5) for s in imp_sentences] + [(s, 1.0) for s in find_sentences]
    
    if not weighted_sentences:
        for cls in PATHOLOGY_CLASSES:
            results[cls] = LABEL_UNCERTAIN
            details[cls] = {"present": 0, "absent": 0, "uncertain": 0, "sentences": []}
        return results, details
    
    for cls in PATHOLOGY_CLASSES:
        keywords = POSITIVE_KEYWORDS.get(cls, [])
        
        present_score = 0.0
        absent_score = 0.0
        uncertain_score = 0.0
        matched_sentences = []
        
        for sentence, weight in weighted_sentences:
            found_keywords = _find_keywords_in_sentence(sentence, keywords)
            
            for kw in found_keywords:
                assertion = _classify_assertion(sentence, kw)
                matched_sentences.append({
                    "sentence": sentence[:100],
                    "keyword": kw,
                    "assertion": assertion,
                })
                
                if assertion == "PRESENT":
                    present_score += weight
                elif assertion == "ABSENT":
                    absent_score += weight
                elif assertion == "UNCERTAIN":
                    uncertain_score += weight
        
        # Decision logic
        if cls == "No_Finding":
            # Special: "No Finding" keywords mean normal study
            if present_score > 0:
                results[cls] = LABEL_PRESENT
            else:
                results[cls] = LABEL_ABSENT
        else:
            if present_score > 0 and absent_score == 0 and uncertain_score == 0:
                results[cls] = LABEL_PRESENT
            elif absent_score > 0 and present_score == 0:
                results[cls] = LABEL_ABSENT
            elif present_score > absent_score and present_score > uncertain_score:
                results[cls] = LABEL_PRESENT
            elif absent_score > present_score:
                results[cls] = LABEL_ABSENT
            elif uncertain_score > 0:
                results[cls] = LABEL_UNCERTAIN
            else:
                # No keywords found
                results[cls] = LABEL_ABSENT
        
        details[cls] = {
            "present": present_score,
            "absent": absent_score,
            "uncertain": uncertain_score,
            "sentences": matched_sentences,
        }
    
    return results, details


def run_layer3(df, text_col_impression="impression_text", text_col_findings="findings_text"):
    """
    Run Layer 3 assertion classification on an entire DataFrame.
    
    Args:
        df: DataFrame with text columns
    
    Returns:
        dict: {pathology: [labels_per_row]}
    """
    from tqdm import tqdm
    
    all_labels = {cls: [] for cls in PATHOLOGY_CLASSES}
    total = len(df)
    
    for idx, row in tqdm(df.iterrows(), total=total, desc="  Layer 3 (Assertions)", unit="report"):
        imp = row.get(text_col_impression, "")
        find = row.get(text_col_findings, "")
        labels, _ = classify_report_assertions(imp, find)
        
        for cls in PATHOLOGY_CLASSES:
            all_labels[cls].append(labels[cls])
    
    print(f"  [Layer 3] Complete -- {total} reports processed.")
    return all_labels


if __name__ == "__main__":
    # Quick self-test
    test_cases = [
        {
            "imp": "No acute cardiopulmonary process.",
            "find": "The lungs are clear without focal consolidation, effusion, or pneumothorax.",
            "desc": "Normal study"
        },
        {
            "imp": "Worsening multifocal pneumonia. Small bilateral effusions.",
            "find": "Multifocal consolidations worse in the right lung and left lower lobe.",
            "desc": "Active pneumonia"
        },
        {
            "imp": "Possible atelectasis at the right base.",
            "find": "There is a small right pleural effusion. Heart size is mildly enlarged.",
            "desc": "Uncertain findings"
        },
    ]
    
    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {tc['desc']}")
        print(f"Impression: {tc['imp']}")
        labels, details = classify_report_assertions(tc["imp"], tc["find"])
        for cls in PATHOLOGY_CLASSES:
            if labels[cls] != 0 or details[cls]["present"] > 0 or details[cls]["uncertain"] > 0:
                status = {1: "PRESENT", 0: "ABSENT", -1: "UNCERTAIN"}[labels[cls]]
                print(f"  {cls:30s} -> {status}")

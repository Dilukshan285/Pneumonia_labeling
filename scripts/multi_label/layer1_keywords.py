"""
Layer 1: Negation-Aware Clinical Keyword Extraction
=====================================================
Uses negspaCy + spaCy to detect clinical keywords for 14 pathologies,
then classifies each mention as PRESENT, ABSENT, or UNCERTAIN based on
negation and uncertainty cue detection.

For each report (impression + findings), produces a dict:
  { "Atelectasis": 1, "Cardiomegaly": 0, "Pneumonia": -1, ... }
  where 1=PRESENT, 0=ABSENT, -1=UNCERTAIN
"""

import re
import spacy
from negspacy.negation import Negex

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from multi_label_config import (
    POSITIVE_KEYWORDS, NEGATION_CUES, UNCERTAINTY_CUES,
    PATHOLOGY_CLASSES, LABEL_PRESENT, LABEL_ABSENT, LABEL_UNCERTAIN,
)


def _build_nlp():
    """Build the spaCy pipeline with negspaCy negation detection."""
    from negspacy.termsets import termset
    
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    
    # Get clinical negation patterns as a dict
    ts = termset("en_clinical")
    clinical_patterns = ts.get_patterns()
    
    # Add extra clinical negation cues
    extra_preceding = [
        "no evidence of", "absence of", "negative for",
        "ruled out", "rules out", "without evidence of",
    ]
    for term in extra_preceding:
        if term not in clinical_patterns.get("preceding_negations", []):
            clinical_patterns.setdefault("preceding_negations", []).append(term)
    
    # Add negex with clinical termset dict
    nlp.add_pipe(
        "negex",
        config={
            "neg_termset": clinical_patterns,
            "chunk_prefix": ["no", "not", "without", "absence of",
                             "negative for", "ruled out", "deny",
                             "denies", "denied"],
        },
        last=True,
    )
    return nlp


# Module-level singleton (created on first call)
_NLP_INSTANCE = None

def get_nlp():
    global _NLP_INSTANCE
    if _NLP_INSTANCE is None:
        _NLP_INSTANCE = _build_nlp()
    return _NLP_INSTANCE


def _check_uncertainty(text_lower, keyword):
    """Check if a keyword mention is wrapped in uncertainty language."""
    # Find the keyword position
    kw_pos = text_lower.find(keyword.lower())
    if kw_pos == -1:
        return False
    
    # Check a window of 60 chars before and 30 chars after the keyword
    window_start = max(0, kw_pos - 60)
    window_end = min(len(text_lower), kw_pos + len(keyword) + 30)
    window = text_lower[window_start:window_end]
    
    for cue in UNCERTAINTY_CUES:
        if cue in window:
            return True
    return False


def _check_explicit_negation(text_lower, keyword):
    """
    Check for explicit negation patterns that negspaCy might miss.
    Returns True if the keyword is negated by surrounding context.
    """
    kw_pos = text_lower.find(keyword.lower())
    if kw_pos == -1:
        return False
    
    # Check 50 chars before the keyword
    prefix_start = max(0, kw_pos - 50)
    prefix = text_lower[prefix_start:kw_pos]
    
    # Check 40 chars after the keyword
    suffix_end = min(len(text_lower), kw_pos + len(keyword) + 40)
    suffix = text_lower[kw_pos + len(keyword):suffix_end]
    
    # Prefix-based negation
    negation_prefixes = [
        "no ", "no evidence of", "without ", "not ",
        "negative for ", "ruled out ", "absence of ",
        "has resolved", "has cleared",
    ]
    for neg in negation_prefixes:
        if neg in prefix:
            return True
    
    # Suffix-based negation
    negation_suffixes = [
        "has resolved", "has cleared", "has improved",
        "resolved", "removed", "was removed",
        "is not seen", "not seen", "not identified",
        "not present", "not demonstrated",
    ]
    for neg in negation_suffixes:
        if neg in suffix:
            return True
    
    return False


def classify_report_keywords(impression_text, findings_text):
    """
    Classify a single report using negation-aware keyword extraction.
    
    Args:
        impression_text: Impression section text (may be empty/NaN)
        findings_text: Findings section text (may be empty/NaN)
    
    Returns:
        dict: {pathology: label} where label is 1/0/-1
        dict: {pathology: {"present": n, "absent": n, "uncertain": n}}
    """
    nlp = get_nlp()
    results = {}
    details = {}
    
    # Clean text inputs
    impression = str(impression_text).strip() if impression_text and str(impression_text).strip() not in ("", "nan", "None") else ""
    findings = str(findings_text).strip() if findings_text and str(findings_text).strip() not in ("", "nan", "None") else ""
    
    # Combine with impression weighted higher (processed first, gets priority)
    combined_text = impression + " " + findings
    combined_lower = combined_text.lower()
    
    if not combined_text.strip():
        # Empty report — all UNCERTAIN
        for cls in PATHOLOGY_CLASSES:
            results[cls] = LABEL_UNCERTAIN
            details[cls] = {"present": 0, "absent": 0, "uncertain": 0}
        return results, details
    
    # Process with spaCy + negspaCy
    doc = nlp(combined_text[:10000])  # Cap length for spaCy
    
    for cls in PATHOLOGY_CLASSES:
        keywords = POSITIVE_KEYWORDS.get(cls, [])
        present_count = 0
        absent_count = 0
        uncertain_count = 0
        
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in combined_lower:
                continue
            
            # Found a keyword match — determine its assertion status
            is_uncertain = _check_uncertainty(combined_lower, kw_lower)
            is_negated = _check_explicit_negation(combined_lower, kw_lower)
            
            # Also check negspaCy negation on entity spans
            negex_negated = False
            for ent in doc.ents:
                if kw_lower in ent.text.lower():
                    if hasattr(ent._, 'negex') and ent._.negex:
                        negex_negated = True
                        break
            
            # Also check via sentence-level token matching
            if not negex_negated:
                for sent in doc.sents:
                    sent_lower = sent.text.lower()
                    if kw_lower in sent_lower:
                        # Check for negation markers in the sentence
                        for token in sent:
                            if token.dep_ == "neg" and kw_lower in sent_lower:
                                negex_negated = True
                                break
                    if negex_negated:
                        break
            
            if is_uncertain:
                uncertain_count += 1
            elif is_negated or negex_negated:
                absent_count += 1
            else:
                present_count += 1
        
        # Special handling for "No_Finding"
        if cls == "No_Finding":
            # "No Finding" is special — keywords ARE the positive assertion
            # If "no acute cardiopulmonary process" is found non-negated, 
            # it means the study IS normal
            if present_count > 0:
                results[cls] = LABEL_PRESENT
            elif absent_count > 0:
                results[cls] = LABEL_ABSENT
            else:
                results[cls] = LABEL_ABSENT  # If no normal-finding keywords matched, assume abnormal
        else:
            # Standard pathology
            if present_count > 0 and absent_count == 0:
                results[cls] = LABEL_PRESENT
            elif absent_count > 0 and present_count == 0:
                results[cls] = LABEL_ABSENT
            elif present_count > 0 and absent_count > 0:
                # Conflicting signals — mark uncertain
                results[cls] = LABEL_UNCERTAIN
            elif uncertain_count > 0:
                results[cls] = LABEL_UNCERTAIN
            else:
                # No keywords found at all — absent
                results[cls] = LABEL_ABSENT
        
        details[cls] = {
            "present": present_count,
            "absent": absent_count, 
            "uncertain": uncertain_count,
        }
    
    return results, details


def run_layer1(df, text_col_impression="impression_text", text_col_findings="findings_text"):
    """
    Run Layer 1 keyword extraction on an entire DataFrame.
    
    Args:
        df: DataFrame with impression and findings text columns
    
    Returns:
        dict of {pathology: [labels_per_row]}
    """
    from tqdm import tqdm
    
    all_labels = {cls: [] for cls in PATHOLOGY_CLASSES}
    total = len(df)
    
    for idx, row in tqdm(df.iterrows(), total=total, desc="  Layer 1 (Keywords)", unit="report"):
        imp = row.get(text_col_impression, "")
        find = row.get(text_col_findings, "")
        labels, _ = classify_report_keywords(imp, find)
        
        for cls in PATHOLOGY_CLASSES:
            all_labels[cls].append(labels[cls])
    
    print(f"  [Layer 1] Complete -- {total} reports processed.")
    return all_labels


if __name__ == "__main__":
    # Quick self-test
    test_imp = "No acute cardiopulmonary process."
    test_find = "The lungs are clear without focal consolidation, effusion, or pneumothorax. The cardiomediastinal silhouette is within normal limits."
    labels, details = classify_report_keywords(test_imp, test_find)
    print("Test report labels:")
    for cls, label in labels.items():
        status = {1: "PRESENT", 0: "ABSENT", -1: "UNCERTAIN"}[label]
        print(f"  {cls:30s} -> {status:10s}  (P:{details[cls]['present']} A:{details[cls]['absent']} U:{details[cls]['uncertain']})")

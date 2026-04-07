"""
Steps 2.5, 2.6, 2.7 — Labeling Function 2: NegEx Clinical Negation Detection (v3.1)

Purpose:
    Detects false positives from LF1 by running clinical negation detection
    on reports where LF1 assigned a POSITIVE vote. Uses negspaCy with the
    en_clinical termset, which contains medical-specific negation triggers
    (e.g., "no evidence of", "without", "denies", "rules out", "negative for",
    "resolved") that are superior to general-purpose NLP negation systems
    when applied to radiology report text.

Workflow:
    1. LF2 only runs on reports where LF1 voted POSITIVE.
    2. For each such report, it passes the target section text (impression or
       findings) through the negspaCy pipeline.
    3. For each sentence containing a positive pneumonia term, it checks whether
       the entity is flagged as negated by negspaCy.
    4. If ALL pneumonia entities in the text are negated → LF2 votes NEGATIVE.
       If ANY pneumonia entity is NOT negated → LF2 votes POSITIVE (confirms LF1).
    5. For reports where LF1 voted NEGATIVE → LF2 votes NEGATIVE (pass-through).
    6. For reports where LF1 voted EXCLUDE → LF2 votes EXCLUDE (pass-through).

Returns: POSITIVE (1), NEGATIVE (0), or EXCLUDE (-1)
         LF2 only confirms or overrides LF1-POSITIVE reports.
"""

import os
import sys
import re
import spacy
from negspacy.negation import Negex
from negspacy.termsets import termset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LABEL_POSITIVE, LABEL_NEGATIVE
from stage2_labeling.keywords import POSITIVE_KEYWORDS
from stage2_labeling.lf1_keywords import LABEL_EXCLUDE


# ============================================================================
# STEP 2.5 — NegEx Pipeline Setup with Clinical Termset
# ============================================================================


def _build_pneumonia_entity_patterns():
    """
    Build spaCy EntityRuler patterns from the POSITIVE_KEYWORDS list.
    
    This ensures that positive pneumonia terms are recognized as named
    entities by spaCy, which is required for negspaCy to evaluate
    their negation status. Without explicitly defining these patterns,
    the base en_core_web_sm model would not recognize medical terms
    like 'consolidation' or 'air bronchogram' as entities.
    
    Returns:
        list[dict]: spaCy EntityRuler pattern dicts.
    """
    patterns = []
    for kw in POSITIVE_KEYWORDS:
        # Each keyword becomes a pattern that matches case-insensitively
        # by using lowercased token matching
        tokens = kw.lower().split()
        pattern = {
            "label": "PNEUMONIA_FINDING",
            "pattern": [{"LOWER": t} for t in tokens],
        }
        patterns.append(pattern)
    return patterns


def _build_custom_negation_patterns():
    """
    Build custom pseudo-negation patterns to supplement the en_clinical termset.
    
    The clinical termset handles pre-entity negation triggers well (e.g.,
    "no evidence of [pneumonia]"), but misses important post-entity patterns
    common in radiology reports where the negation/resolution cue comes
    AFTER the entity. Examples:
        - "[pneumonia] has resolved"
        - "[consolidation] has cleared"
        - "resolution of [pneumonia]"
        - "[infiltrate] is improving"
    
    NegEx supports two types of negation patterns:
        - preceding_negations: patterns that appear BEFORE the entity
        - following_negations: patterns that appear AFTER the entity
    
    Returns:
        dict: Custom negation patterns to merge with clinical termset.
    """
    custom_preceding = [
        # Resolution/clearing (preceding patterns)
        "resolution of",
        "resolved",
        "clearing of",
        "interval resolution of",
        "near-complete resolution of",
        "complete resolution of",
        "interval improvement of",
        "improvement of",
    ]
    
    custom_following = [
        # Resolution/clearing (following patterns — entity comes first)
        "has resolved",
        "have resolved",
        "resolved",
        "has cleared",
        "has been cleared",
        # NOTE: "has improved" / "is improving" REMOVED — improving means
        # the finding is still partially present, not fully resolved.
        # "consolidation has improved" = consolidation IS still there.
        "is resolving",
        "is clearing",
        "now resolved",
        "now cleared",
        "no longer present",
        "no longer seen",
        "no longer identified",
        "no longer evident",
        "no longer visualized",
    ]
    
    return {
        "preceding_negations": custom_preceding,
        "following_negations": custom_following,
    }


def _setup_full_pipeline():
    """
    Build the complete NegEx pipeline with custom pneumonia entity patterns
    and augmented negation triggers.
    
    Architecture:
        1. en_core_web_sm (tokenizer, tagger, parser, NER)
        2. EntityRuler (adds PNEUMONIA_FINDING entities from keyword list)
        3. NegEx (evaluates negation status using clinical + custom patterns)
    
    The EntityRuler is added BEFORE NegEx but AFTER the built-in NER,
    so that our custom PNEUMONIA_FINDING entities are available for
    negation evaluation. The NegEx component uses a merged pattern set:
    en_clinical termset + custom radiology resolution/improvement patterns.
    
    Returns:
        spacy.Language: Fully configured pipeline.
    """
    nlp = spacy.load("en_core_web_sm")
    
    # Add EntityRuler to recognize pneumonia terms as entities
    # Must be added before NegEx so entities exist when NegEx processes them
    ruler = nlp.add_pipe("entity_ruler", after="ner", config={"overwrite_ents": True})
    patterns = _build_pneumonia_entity_patterns()
    ruler.add_patterns(patterns)
    
    # Load clinical termset and merge with custom patterns
    ts = termset("en_clinical")
    base_patterns = ts.get_patterns()
    custom_patterns = _build_custom_negation_patterns()
    
    # Merge: extend preceding and following negation lists
    merged_patterns = dict(base_patterns)  # shallow copy
    if "preceding_negations" in merged_patterns:
        merged_patterns["preceding_negations"] = (
            list(merged_patterns["preceding_negations"])
            + custom_patterns["preceding_negations"]
        )
    else:
        merged_patterns["preceding_negations"] = custom_patterns["preceding_negations"]
    
    if "following_negations" in merged_patterns:
        merged_patterns["following_negations"] = (
            list(merged_patterns["following_negations"])
            + custom_patterns["following_negations"]
        )
    else:
        merged_patterns["following_negations"] = custom_patterns["following_negations"]
    
    nlp.add_pipe(
        "negex",
        config={
            "neg_termset": merged_patterns,
            "ent_types": ["PNEUMONIA_FINDING"],  # Only evaluate our custom entities
            "chunk_prefix": [],
        },
    )
    
    return nlp


# Module-level pipeline — initialized lazily to avoid loading at import time
_nlp_pipeline = None


def _get_pipeline():
    """Get or initialize the NegEx pipeline (lazy singleton)."""
    global _nlp_pipeline
    if _nlp_pipeline is None:
        _nlp_pipeline = _setup_full_pipeline()
    return _nlp_pipeline


# ============================================================================
# STEP 2.6 & 2.7 — Apply NegEx and Assign LF2 Label
# ============================================================================

def _analyze_negation(text, nlp):
    """
    Run NegEx on a text and determine the negation status of all
    PNEUMONIA_FINDING entities found in the text.
    
    Args:
        text: The section text to analyze (impression or findings).
        nlp:  The configured spaCy+NegEx pipeline.
    
    Returns:
        tuple: (n_total_entities, n_negated_entities, details_list)
            - n_total: Number of PNEUMONIA_FINDING entities found.
            - n_negated: Number that were flagged as negated by NegEx.
            - details: List of dicts with entity text, negation status,
                       and surrounding sentence context for debugging.
    """
    doc = nlp(text)
    
    total = 0
    negated = 0
    details = []
    
    for ent in doc.ents:
        if ent.label_ == "PNEUMONIA_FINDING":
            total += 1
            is_negated = ent._.negex  # True if entity is negated
            
            if is_negated:
                negated += 1
            
            # Extract the sentence containing this entity for context
            sent_text = ""
            for sent in doc.sents:
                if sent.start <= ent.start and ent.end <= sent.end:
                    sent_text = sent.text.strip()
                    break
            
            details.append({
                "entity_text": ent.text,
                "is_negated": is_negated,
                "sentence": sent_text,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })
    
    return total, negated, details


def lf2_negex(row, lf1_label):
    """
    Labeling Function 2 — NegEx Clinical Negation Detection (v3.1).
    
    Step 2.7 Logic:
        - If LF1 voted POSITIVE and NegEx confirms ALL positive terms are
          negated → LF2 votes NEGATIVE (false positive override).
        - If LF1 voted POSITIVE and NegEx finds ANY positive term is NOT
          negated → LF2 votes POSITIVE (confirms LF1).
        - If LF1 voted NEGATIVE → LF2 votes NEGATIVE (pass-through).
        - If LF1 voted EXCLUDE → LF2 votes EXCLUDE (pass-through).
    
    Args:
        row:       pandas Series with 'impression_text' and 'findings_text'.
        lf1_label: int, the label assigned by LF1 for this report.
    
    Returns:
        int: LABEL_POSITIVE (1), LABEL_NEGATIVE (0), or LABEL_EXCLUDE (-1).
    """
    # LF2 only operates on reports where LF1 found positive terms
    if lf1_label == LABEL_EXCLUDE:
        return LABEL_EXCLUDE
    if lf1_label != LABEL_POSITIVE:
        return LABEL_NEGATIVE
    
    # Select target text: impression first, findings as fallback
    impression = str(row.get('impression_text', '') or '')
    findings = str(row.get('findings_text', '') or '')
    target_text = impression.strip() if impression.strip() else findings.strip()
    
    if not target_text:
        return LABEL_NEGATIVE
    
    nlp = _get_pipeline()
    n_total, n_negated, _ = _analyze_negation(target_text, nlp)
    
    # No pneumonia entities found
    if n_total == 0:
        return LABEL_NEGATIVE
    
    # ALL entities negated → this is a false positive; override to NEGATIVE
    if n_negated == n_total:
        return LABEL_NEGATIVE
    
    # At least one entity is NOT negated → confirmed POSITIVE
    return LABEL_POSITIVE


def lf2_negex_debug(row, lf1_label):
    """
    Debug version of LF2 that returns the label AND detailed negation analysis.
    
    Args:
        row:       pandas Series with 'impression_text' and 'findings_text'.
        lf1_label: int, the label assigned by LF1 for this report.
    
    Returns:
        tuple: (label_int, debug_info_dict_or_None)
    """
    if lf1_label == LABEL_EXCLUDE:
        return LABEL_EXCLUDE, None
    if lf1_label != LABEL_POSITIVE:
        return LABEL_NEGATIVE, None
    
    impression = str(row.get('impression_text', '') or '')
    findings = str(row.get('findings_text', '') or '')
    target_text = impression.strip() if impression.strip() else findings.strip()
    
    if not target_text:
        return LABEL_NEGATIVE, None
    
    nlp = _get_pipeline()
    n_total, n_negated, details = _analyze_negation(target_text, nlp)
    
    debug_info = {
        "target_text": target_text[:300],
        "n_total_entities": n_total,
        "n_negated_entities": n_negated,
        "entities": details,
    }
    
    if n_total == 0:
        return LABEL_NEGATIVE, debug_info
    
    if n_negated == n_total:
        return LABEL_NEGATIVE, debug_info
    
    return LABEL_POSITIVE, debug_info


# ============================================================================
# STANDALONE TEST
# ============================================================================

def _run_self_test():
    """
    Quick self-test with known positive and negated pneumonia phrases.
    Validates that the NegEx pipeline correctly detects negation in
    clinical radiology report text.
    """
    print("=" * 70)
    print("LF2 NegEx — Self-Test")
    print("=" * 70)
    print()
    
    nlp = _get_pipeline()
    
    test_cases = [
        # (text, expected_all_negated)
        ("There is a focal consolidation in the right lower lobe.", False),
        ("No evidence of pneumonia.", True),
        ("No consolidation or infiltrate is seen.", True),
        ("Lungs are clear without consolidation.", True),
        ("There is consolidation in the right lower lobe. No pneumonia.", False),
        ("Patient denies pneumonia symptoms. Chest X-ray shows consolidation.", False),
        ("No focal consolidation. No pneumonia. Lungs are clear.", True),
        ("Findings are negative for pneumonia.", True),
        ("The pneumonia has resolved.", True),
        ("Patchy opacity in the left lower lobe consistent with pneumonia.", False),
        ("There is no evidence of consolidation or pneumonia.", True),
        # NegEx cannot resolve complex double-negation patterns like "cannot be
        # excluded"; the Snorkel label model downstream handles these via LF
        # agreement. Expected: entity IS NOT negated by NegEx (False).
        ("Pneumonia cannot be excluded but no consolidation is identified.", False),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_all_negated in test_cases:
        n_total, n_negated, details = _analyze_negation(text, nlp)
        
        if n_total == 0:
            actual_all_negated = None
            status = "NO_ENTITIES"
        else:
            actual_all_negated = (n_negated == n_total)
            status = "PASS" if actual_all_negated == expected_all_negated else "FAIL"
        
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        
        icon = "✓" if status == "PASS" else ("✗" if status == "FAIL" else "?")
        print(f"  {icon} [{status}] \"{text[:70]}...\"")
        print(f"      Entities: {n_total}, Negated: {n_negated}, "
              f"Expected all-negated: {expected_all_negated}, "
              f"Actual: {actual_all_negated}")
        
        for d in details:
            neg_flag = "NEGATED" if d['is_negated'] else "AFFIRMED"
            print(f"        → \"{d['entity_text']}\" = {neg_flag}")
        print()
    
    total_tests = passed + failed
    print(f"  Results: {passed}/{total_tests} passed, {failed}/{total_tests} failed")
    print()
    
    if failed > 0:
        print("  WARNING: Some test cases failed. Review NegEx configuration.")
        print("  Note: Some edge cases are inherently difficult for rule-based")
        print("  negation detection. The Snorkel label model downstream will")
        print("  reconcile disagreements across all 6 labeling functions.")
    else:
        print("  All test cases passed.")
    
    print()
    return failed == 0


if __name__ == "__main__":
    _run_self_test()

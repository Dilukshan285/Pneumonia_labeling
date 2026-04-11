"""
Consensus Layer: Weighted Majority Voting
==========================================
Combines votes from Layer 1 (keywords), Layer 2 (NLI), and Layer 3 (assertions)
using weighted majority voting to produce final multi-label vectors.

Weights:
  - Layer 1 (keywords):    1.0  (basic, but fast and reliable for clear cases)
  - Layer 2 (NLI):         1.5  (semantic understanding, highest weight)
  - Layer 3 (assertions):  1.2  (sentence-level context)

Final label:
  - 1  (PRESENT):   weighted vote for PRESENT wins
  - 0  (ABSENT):    weighted vote for ABSENT wins  
  - -1 (UNCERTAIN): no clear majority, or majority votes uncertain
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from multi_label_config import (
    PATHOLOGY_CLASSES, LAYER_WEIGHTS,
    LABEL_PRESENT, LABEL_ABSENT, LABEL_UNCERTAIN,
)


def compute_consensus(layer1_labels, layer2_labels, layer3_labels):
    """
    Compute consensus labels from three layers for the entire dataset.
    
    Args:
        layer1_labels: dict {pathology: [labels]} from keyword extraction
        layer2_labels: dict {pathology: [labels]} from NLI classification
        layer3_labels: dict {pathology: [labels]} from assertion detection
    
    Returns:
        dict: {pathology: [final_labels]}
        dict: {pathology: [confidence_scores]}  (0.0 to 1.0)
    """
    w1 = LAYER_WEIGHTS["layer1_keywords"]
    w2 = LAYER_WEIGHTS["layer2_nli"]
    w3 = LAYER_WEIGHTS["layer3_assertions"]
    total_weight = w1 + w2 + w3
    
    final_labels = {cls: [] for cls in PATHOLOGY_CLASSES}
    confidence_scores = {cls: [] for cls in PATHOLOGY_CLASSES}
    
    # All layers should have same number of samples
    n_samples = len(layer1_labels[PATHOLOGY_CLASSES[0]])
    
    for cls in PATHOLOGY_CLASSES:
        l1 = layer1_labels[cls]
        l2 = layer2_labels[cls]
        l3 = layer3_labels[cls]
        
        for i in range(n_samples):
            v1, v2, v3 = l1[i], l2[i], l3[i]
            
            # Weighted scoring
            present_score = 0.0
            absent_score = 0.0
            uncertain_score = 0.0
            
            for vote, weight in [(v1, w1), (v2, w2), (v3, w3)]:
                if vote == LABEL_PRESENT:
                    present_score += weight
                elif vote == LABEL_ABSENT:
                    absent_score += weight
                elif vote == LABEL_UNCERTAIN:
                    uncertain_score += weight
            
            # Determine winner
            max_score = max(present_score, absent_score, uncertain_score)
            
            if max_score == 0:
                # All layers returned no signal (shouldn't happen often)
                final_labels[cls].append(LABEL_ABSENT)
                confidence_scores[cls].append(0.0)
            elif present_score == max_score and present_score > absent_score:
                final_labels[cls].append(LABEL_PRESENT)
                confidence_scores[cls].append(present_score / total_weight)
            elif absent_score == max_score and absent_score > present_score:
                final_labels[cls].append(LABEL_ABSENT)
                confidence_scores[cls].append(absent_score / total_weight)
            elif uncertain_score == max_score:
                final_labels[cls].append(LABEL_UNCERTAIN)
                confidence_scores[cls].append(uncertain_score / total_weight)
            elif present_score == absent_score:
                # Tie between present and absent → uncertain
                final_labels[cls].append(LABEL_UNCERTAIN)
                confidence_scores[cls].append(0.5)
            else:
                # Fallback
                final_labels[cls].append(LABEL_UNCERTAIN)
                confidence_scores[cls].append(0.33)
    
    return final_labels, confidence_scores


def apply_no_finding_logic(final_labels, n_samples):
    """
    Apply mutual exclusivity constraint for 'No_Finding':
    If ANY pathology (other than Support_Devices) is PRESENT,
    then No_Finding should be ABSENT.
    
    Conversely, if No_Finding is PRESENT and no other pathology 
    is PRESENT, keep it as PRESENT.
    """
    pathology_conditions = [
        cls for cls in PATHOLOGY_CLASSES 
        if cls not in ("No_Finding", "Support_Devices")
    ]
    
    for i in range(n_samples):
        any_pathology_present = any(
            final_labels[cls][i] == LABEL_PRESENT 
            for cls in pathology_conditions
        )
        
        if any_pathology_present:
            final_labels["No_Finding"][i] = LABEL_ABSENT
        elif not any_pathology_present and final_labels["No_Finding"][i] != LABEL_PRESENT:
            # If no pathology is flagged and No_Finding wasn't detected,
            # check if everything is absent — if so, set No_Finding = PRESENT
            all_absent = all(
                final_labels[cls][i] == LABEL_ABSENT 
                for cls in pathology_conditions
            )
            if all_absent:
                final_labels["No_Finding"][i] = LABEL_PRESENT
    
    return final_labels


def print_consensus_summary(final_labels, n_samples):
    """Print a summary of consensus label distributions."""
    print("\n" + "=" * 70)
    print("CONSENSUS LABEL DISTRIBUTION")
    print("=" * 70)
    print(f"{'Pathology':30s} {'PRESENT':>8s} {'ABSENT':>8s} {'UNCERTAIN':>10s} {'Prev%':>7s}")
    print("-" * 70)
    
    for cls in PATHOLOGY_CLASSES:
        n_present = sum(1 for l in final_labels[cls] if l == LABEL_PRESENT)
        n_absent = sum(1 for l in final_labels[cls] if l == LABEL_ABSENT)
        n_uncertain = sum(1 for l in final_labels[cls] if l == LABEL_UNCERTAIN)
        prevalence = n_present / n_samples * 100 if n_samples > 0 else 0
        
        print(f"{cls:30s} {n_present:8d} {n_absent:8d} {n_uncertain:10d} {prevalence:6.1f}%")
    
    print("-" * 70)
    print(f"Total reports: {n_samples}")
    print("=" * 70)

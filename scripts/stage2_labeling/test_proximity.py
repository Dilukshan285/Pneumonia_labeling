# -*- coding: utf-8 -*-
"""Edge case validation - failures only"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stage2_labeling.lf1_keywords import _proximity_negation_check

tests = [
    ("No pneumothorax, pneumonia, or effusion.", True, "comma-sep"),
    ("No vascular congestion, pleural effusion, or pneumonia.", True, "comma end"),
    ("No focal consolidation, effusion or pneumothorax.", True, "multi-word comma"),
    ("No apparent consolidation.", True, "apparent"),
    ("No definite signs of pneumonia though post diuresis.", True, "definite signs"),
    ("without vascular congestion, pleural effusion, or acute focal pneumonia.", True, "without comma"),
    ("No focal lung consolidation.", True, "no focal lung"),
    ("without acute pneumonia.", True, "without acute"),
    ("No CHF or pneumonia.", True, "no X or Y"),
    ("No evidence of left lower lobe pneumonia on current examination.", True, "evidence lobe"),
    ("without evidence of acute bilateral pneumonia.", True, "without evidence"),
    ("No significant consolidation.", True, "significant"),
    ("Previously severe consolidation has improved.", False, "improved"),
    ("Complete resolution of left upper lobe pneumonia.", False, "resolution"),
    ("No improvement of pneumonia.", False, "improvement"),
    ("No change in consolidation.", False, "change"),
    ("No history of pneumonia.", False, "history"),
    ("No resolution of pneumonia.", False, "resolution2"),
    ("No worsening of pneumonia but new effusion.", False, "worsening"),
    ("Pneumonia is present. No pleural effusion.", False, "diff sentence"),
    ("Worsening pneumonia. No pleural effusion.", False, "no trigger"),
    ("Right lower lobe consolidation.", False, "no neg trigger"),
    ("Consolidation in right lower lobe is worsening.", False, "no neg2"),
    ("No chest radiographic findings to suggest pneumonitis, but this diagnosis is more readily made by CT.", True, "no findings to suggest = negation"),
    ("the preexisting left parenchymal opacity has almost completely resolved.", False, "no trigger3"),
    ("without definite atelectatic change or consolidation.", False, "change scope"),
    # --- Multi-term tests (PROX bug fix validation) ---
    # Mixed: one negated + one affirmed → must NOT fire (the core bug fix)
    ("Equivocal retrocardiac opacity.  Otherwise, no focal infiltrate.", False, "multi-term mixed"),
    ("Right lower lobe consolidation.  No pneumonia.", False, "multi-term affirmed first"),
    ("No consolidation.  Patchy opacity in left lower lobe.", False, "multi-term negated first"),
    ("Dense consolidation in right lung.  No focal infiltrate seen.", False, "multi-term dense+neg"),
    # All terms negated with SEPARATE triggers → should fire
    ("No pneumonia.  No consolidation.", True, "multi-term all negated separate triggers"),
    # Comma-list without separate triggers: the proximity regex only captures
    # the first term. "or infiltrate"/"or pneumonia" lack their own trigger.
    # In the real pipeline these are caught by Step 2 NEGATIVE keywords
    # ("no consolidation", "without focal consolidation") BEFORE reaching Step 2.5.
    ("No consolidation or infiltrate.", False, "comma-list: only first term has trigger"),
    ("without focal consolidation or pneumonia.", False, "comma-list: only first term has trigger 2"),
]

failed_list = []
for i, (text, should_negate, desc) in enumerate(tests):
    term, is_neg = _proximity_negation_check(text)
    if is_neg != should_negate:
        failed_list.append((i, text, should_negate, is_neg, term, desc))

print(f"PASSED: {len(tests)-len(failed_list)}/{len(tests)}")
print(f"FAILED: {len(failed_list)}/{len(tests)}")
print()
for i, text, should, actual, term, desc in failed_list:
    print(f"TEST {i} FAILED [{desc}]:")
    print(f"  Text: {text}")
    print(f"  Expected negated={should}, Got negated={actual}, term={term}")
    print()

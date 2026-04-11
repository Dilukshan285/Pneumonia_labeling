"""
Layer 2: DeBERTa-v3 Zero-Shot Multi-Label Classification
==========================================================
Uses MoritzLaurer/deberta-v3-large-mnli-fever-anli-ling-wanli for
14 independent entailment checks per report.

This model is trained on 5 NLI datasets (MNLI + FEVER + ANLI + LingNLI + WANLI),
achieving 92.7% accuracy vs BART-large-mnli's 90.1%.

Label mapping for this model:
  0 = entailment, 1 = neutral, 2 = contradiction

GPU-accelerated with batched processing + tqdm progress bars.
"""

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from multi_label_config import (
    NLI_MODEL_NAME, NLI_BATCH_SIZE, NLI_MAX_TOKENS,
    NLI_HYPOTHESES, NLI_CONTRADICTION_HYPOTHESES,
    PATHOLOGY_CLASSES, LABEL_PRESENT, LABEL_ABSENT, LABEL_UNCERTAIN,
)


class NLIClassifier:
    """
    Wraps DeBERTa-v3-large NLI for efficient multi-label zero-shot classification.
    """
    
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  [Layer 2] Loading {NLI_MODEL_NAME} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        self.model.to(self.device)
        self.model.half()  # fp16 for efficiency on RTX 4060
        self.model.eval()
        
        # Auto-detect label mapping from model config
        id2label = self.model.config.id2label
        self.entailment_idx = None
        self.contradiction_idx = None
        self.neutral_idx = None
        for idx, label in id2label.items():
            label_lower = label.lower()
            if "entail" in label_lower:
                self.entailment_idx = int(idx)
            elif "contra" in label_lower:
                self.contradiction_idx = int(idx)
            elif "neutral" in label_lower:
                self.neutral_idx = int(idx)
        
        print(f"  [Layer 2] Model loaded. Label map: entail={self.entailment_idx}, "
              f"contra={self.contradiction_idx}, neutral={self.neutral_idx}")
    
    def _get_report_text(self, impression, findings):
        """Combine impression and findings into a single premise string."""
        imp = str(impression).strip() if impression and str(impression).strip() not in ("", "nan", "None") else ""
        find = str(findings).strip() if findings and str(findings).strip() not in ("", "nan", "None") else ""
        
        # Prioritize impression, append findings for context
        if imp and find:
            text = f"IMPRESSION: {imp} FINDINGS: {find}"
        elif imp:
            text = imp
        elif find:
            text = find
        else:
            text = ""
        
        return text[:2000]  # Cap length
    
    @torch.no_grad()
    def classify_batch(self, premises, hypothesis):
        """
        Run NLI on a batch of premises against a single hypothesis.
        Returns arrays of entailment and contradiction probabilities.
        """
        hypotheses = [hypothesis] * len(premises)
        
        inputs = self.tokenizer(
            premises, hypotheses,
            return_tensors="pt",
            max_length=NLI_MAX_TOKENS,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        logits = self.model(**inputs).logits.float()  # back to fp32 for softmax
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        entailment_probs = probs[:, self.entailment_idx]
        contradiction_probs = probs[:, self.contradiction_idx]
        
        return entailment_probs, contradiction_probs
    
    @torch.no_grad()
    def classify_single(self, premise, hypothesis):
        """Run NLI on a single premise-hypothesis pair (OOM fallback)."""
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            max_length=NLI_MAX_TOKENS,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        logits = self.model(**inputs).logits.float()
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        
        return probs[self.entailment_idx], probs[self.contradiction_idx]
    
    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


def run_layer2(df, text_col_impression="impression_text", text_col_findings="findings_text"):
    """
    Run Layer 2 NLI classification on an entire DataFrame.
    
    Processes one pathology at a time across all reports in batches.
    Uses tqdm progress bars for real-time monitoring.
    
    Args:
        df: DataFrame with text columns
    
    Returns:
        dict: {pathology: [labels_per_row]}
        dict: {pathology: [probs_per_row]}
    """
    classifier = NLIClassifier()
    
    # Prepare all premise texts
    premises = []
    for _, row in df.iterrows():
        imp = row.get(text_col_impression, "")
        find = row.get(text_col_findings, "")
        premises.append(classifier._get_report_text(imp, find))
    
    total = len(premises)
    all_labels = {cls: [] for cls in PATHOLOGY_CLASSES}
    all_probs = {cls: [] for cls in PATHOLOGY_CLASSES}
    
    # Process each pathology across all reports in batches
    for cls_idx, cls in enumerate(tqdm(PATHOLOGY_CLASSES, desc="  Layer 2 (NLI)", unit="condition")):
        hypothesis = NLI_HYPOTHESES[cls]
        
        cls_ent_probs = []
        cls_contra_probs = []
        
        # Batch progress within each condition
        n_batches = (total + NLI_BATCH_SIZE - 1) // NLI_BATCH_SIZE
        batch_iter = range(0, total, NLI_BATCH_SIZE)
        
        for batch_start in tqdm(batch_iter, desc=f"    {cls}", unit="batch",
                                leave=False, total=n_batches):
            batch_end = min(batch_start + NLI_BATCH_SIZE, total)
            batch_premises = premises[batch_start:batch_end]
            
            # Filter out empty premises
            valid_indices = []
            valid_premises = []
            for i, p in enumerate(batch_premises):
                if p.strip():
                    valid_indices.append(i)
                    valid_premises.append(p)
            
            batch_ent = np.full(len(batch_premises), 0.5)
            batch_contra = np.full(len(batch_premises), 0.5)
            
            if valid_premises:
                try:
                    ent_p, contra_p = classifier.classify_batch(valid_premises, hypothesis)
                    for local_idx, orig_idx in enumerate(valid_indices):
                        batch_ent[orig_idx] = ent_p[local_idx]
                        batch_contra[orig_idx] = contra_p[local_idx]
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        for local_idx, premise in zip(valid_indices, valid_premises):
                            ep, cp = classifier.classify_single(premise, hypothesis)
                            batch_ent[local_idx] = ep
                            batch_contra[local_idx] = cp
                    else:
                        raise
            
            cls_ent_probs.extend(batch_ent.tolist())
            cls_contra_probs.extend(batch_contra.tolist())
        
        # Convert probabilities to labels
        n_present = 0
        n_absent = 0
        n_uncertain = 0
        for i in range(total):
            ent_p = cls_ent_probs[i]
            contra_p = cls_contra_probs[i]
            
            all_probs[cls].append(ent_p)
            
            if ent_p > 0.5 and ent_p > contra_p:
                all_labels[cls].append(LABEL_PRESENT)
                n_present += 1
            elif contra_p > 0.5 and contra_p > ent_p:
                all_labels[cls].append(LABEL_ABSENT)
                n_absent += 1
            else:
                all_labels[cls].append(LABEL_UNCERTAIN)
                n_uncertain += 1
        
        tqdm.write(f"    -> {cls}: P={n_present}, A={n_absent}, U={n_uncertain}")
    
    # Cleanup GPU
    classifier.cleanup()
    
    print(f"  [Layer 2] Complete -- {total} reports x {len(PATHOLOGY_CLASSES)} conditions.")
    return all_labels, all_probs


if __name__ == "__main__":
    # Quick self-test
    clf = NLIClassifier()
    
    test_cases = [
        ("Worsening multifocal pneumonia",
         "Multifocal consolidations worse in right lung. Small bilateral effusions.",
         "Pneumonia case"),
        ("No acute cardiopulmonary process.",
         "Lungs are clear. No effusion or pneumothorax.",
         "Normal case"),
    ]
    
    for imp, find, desc in test_cases:
        premise = clf._get_report_text(imp, find)
        print(f"\nTest: {desc}")
        for cls in PATHOLOGY_CLASSES:
            hyp = NLI_HYPOTHESES[cls]
            ent_p, contra_p = clf.classify_single(premise, hyp)
            if ent_p > 0.5:
                status = "PRESENT"
            elif contra_p > 0.5:
                status = "ABSENT"
            else:
                status = "UNCERTAIN"
            if status != "ABSENT" or ent_p > 0.3:
                print(f"  {cls:30s} -> {status:10s}  (ent={ent_p:.4f}, contra={contra_p:.4f})")
    
    clf.cleanup()

"""
Layer 2 — Triple-Model NLI Ensemble (BEST-IN-CLASS)

Three state-of-the-art NLI architectures independently classify each report
via zero-shot entailment scoring. Only reports that passed pre-filter are
processed (163,608 reports). Pre-filter negatives are handled in Layer 5.

  Model A: facebook/bart-large-mnli
           - REUSE existing LF5 scores (already computed)
           - BART encoder-decoder, 407M params
           - Trained on MNLI (433K pairs)

  Model B: MoritzLaurer/deberta-v3-large-zeroshot-v2.0
           - THE BEST zero-shot classifier available (435M params)
           - Trained on broad NLI + classification data
           - Binary NLI (entailment / not_entailment)
           - ~870 MB in fp16

  Model C: MoritzLaurer/deberta-v3-large-mnli-fever-anli-ling-wanli
           - THE BEST traditional NLI model (435M params)
           - Trained on 5 NLI datasets incl. adversarial ANLI
           - 3-class NLI (entailment / neutral / contradiction)
           - ~870 MB in fp16

Ensemble Logic (strict majority vote):
  - POSITIVE: ≥2 of 3 models vote POSITIVE
  - NEGATIVE: ≥2 of 3 models vote NEGATIVE
  - EXCLUDED: No majority (disagreement = exclusion)

Input:  parsed_reports.csv, lf1_to_lf6_results.csv (for Model A), prefilter_negatives.csv
Output: layer2_nli_ensemble.csv

Estimated runtime: ~10-14 hours total for Models B+C on RTX 4060 8GB
                   (Model A already computed — zero additional cost)
                   Run overnight.
"""

import os
import sys
import gc
import time
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_INTERMEDIATE,
    NLI_MAX_TOKENS,
    LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN, LABEL_ABSTAIN,
    RANDOM_SEED,
)

# Input files
PARSED_REPORTS = os.path.join(DATA_INTERMEDIATE, "parsed_reports.csv")
LF_RESULTS_CSV = os.path.join(DATA_INTERMEDIATE, "lf1_to_lf6_results.csv")
PREFILTER_CSV = os.path.join(DATA_INTERMEDIATE, "prefilter_negatives.csv")

# Output
ENSEMBLE_CSV = os.path.join(DATA_INTERMEDIATE, "layer2_nli_ensemble.csv")

# ============================================================================
# MODEL CONFIGURATIONS — THE BEST AVAILABLE
# ============================================================================

MODEL_B = {
    'name': 'MoritzLaurer/deberta-v3-large-zeroshot-v2.0',
    'desc': 'DeBERTa-v3-Large ZeroShot v2.0 (BEST zero-shot)',
    'batch_size': 16,
    'confidence': 0.40,
    'max_length': 512,
    'checkpoint': os.path.join(DATA_INTERMEDIATE, "layer2_ckpt_model_b.json"),
}

MODEL_C = {
    'name': 'MoritzLaurer/deberta-v3-large-mnli-fever-anli-ling-wanli',
    'desc': 'DeBERTa-v3-Large MNLI+FEVER+ANLI (BEST NLI)',
    'batch_size': 16,
    'confidence': 0.40,
    'max_length': 512,
    'checkpoint': os.path.join(DATA_INTERMEDIATE, "layer2_ckpt_model_c.json"),
}

# Candidate labels — same for all models for vote consistency
CANDIDATE_LABELS = ["pneumonia present", "pneumonia absent", "pneumonia uncertain"]

_CANDIDATE_TO_LABEL = {
    "pneumonia present": LABEL_POSITIVE,
    "pneumonia absent": LABEL_NEGATIVE,
    "pneumonia uncertain": LABEL_UNCERTAIN,
}

_MAX_CHAR_LENGTH = 3000  # ~500 tokens, leaves room for hypothesis in 512-token models

CHECKPOINT_INTERVAL = 5000


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _gpu_mem():
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "free_mb": 0}
    alloc = torch.cuda.memory_allocated(0) / (1024**2)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    return {"allocated_mb": round(alloc, 1), "free_mb": round(total - alloc, 1)}


def _prepare_texts(df):
    """Extract impression (preferred) or findings text, truncated."""
    impression = df['impression_text'].fillna('').astype(str).str.strip()
    findings = df['findings_text'].fillna('').astype(str).str.strip()
    texts = impression.where(impression != '', findings)
    texts = texts.str[:_MAX_CHAR_LENGTH]
    return texts.tolist()


def _save_checkpoint(idx, labels, scores, path):
    temp = path + ".tmp"
    with open(temp, 'w') as f:
        json.dump({"idx": idx, "labels": labels, "scores": scores,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
    os.replace(temp, path)


def _load_checkpoint(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            ck = json.load(f)
        if 'idx' in ck and 'labels' in ck and 'scores' in ck:
            return ck
    except Exception:
        pass
    return None


# ============================================================================
# GENERIC NLI MODEL RUNNER
# ============================================================================

def run_nli_model(model_config, texts):
    """
    Run a single NLI model on all texts with checkpointing and OOM handling.
    Returns (labels_list, scores_list).
    """
    from transformers import pipeline as hf_pipeline

    model_name = model_config['name']
    batch_size = model_config['batch_size']
    confidence = model_config['confidence']
    max_length = model_config['max_length']
    ckpt_path = model_config['checkpoint']
    desc = model_config['desc']

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Layer 2 NLI inference")

    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    _clear_gpu()

    print(f"    Loading {model_name}...")
    torch.backends.cudnn.benchmark = True

    t0 = time.time()
    classifier = hf_pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0,
        torch_dtype=torch.float16,
    )
    load_time = time.time() - t0
    print(f"    Loaded in {load_time:.1f}s")

    try:
        mem = _gpu_mem()
        print(f"    VRAM: {mem['allocated_mb']}MB used / {mem['free_mb']}MB free")
    except Exception:
        pass
    print()

    n_total = len(texts)
    all_labels = []
    all_scores = []
    start_idx = 0
    current_batch = batch_size

    # Resume from checkpoint
    ck = _load_checkpoint(ckpt_path)
    if ck:
        start_idx = ck['idx']
        all_labels = ck['labels']
        all_scores = ck['scores']
        print(f"    RESUMING from checkpoint: {start_idx:,}/{n_total:,}")

    pbar = tqdm(total=n_total, initial=start_idx, desc=f"    {desc[:40]}",
                unit="rpt", file=sys.stdout)

    i = start_idx
    since_ck = 0

    while i < n_total:
        batch_end = min(i + current_batch, n_total)
        batch = texts[i:batch_end]
        blen = len(batch)

        b_labels = [LABEL_ABSTAIN] * blen
        b_scores = [0.0] * blen

        non_empty = [(j, t) for j, t in enumerate(batch) if t.strip()]

        if non_empty:
            ne_idx, ne_texts = zip(*non_empty)
            try:
                outputs = classifier(
                    list(ne_texts),
                    candidate_labels=CANDIDATE_LABELS,
                    batch_size=current_batch,
                    truncation=True,
                )
                if isinstance(outputs, dict):
                    outputs = [outputs]

                for k, idx in enumerate(ne_idx):
                    top_label = outputs[k]['labels'][0]
                    top_score = outputs[k]['scores'][0]
                    if top_score >= confidence:
                        b_labels[idx] = _CANDIDATE_TO_LABEL.get(top_label, LABEL_ABSTAIN)
                    b_scores[idx] = top_score

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    _clear_gpu()
                    old = current_batch
                    current_batch = max(1, current_batch // 2)
                    print(f"\n    OOM! Batch size: {old} -> {current_batch}")
                    continue
                raise

        all_labels.extend(b_labels)
        all_scores.extend(b_scores)
        pbar.update(blen)
        i = batch_end
        since_ck += blen

        if since_ck >= CHECKPOINT_INTERVAL:
            _save_checkpoint(i, all_labels, all_scores, ckpt_path)
            since_ck = 0

    pbar.close()

    # Cleanup
    del classifier
    _clear_gpu()

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return all_labels, all_scores


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_start = time.time()

    print("=" * 70)
    print("LAYER 2 — TRIPLE-MODEL NLI ENSEMBLE (BEST-IN-CLASS)")
    print("=" * 70)
    print()
    print("  Models:")
    print("    A: facebook/bart-large-mnli (REUSE from LF5)")
    print(f"    B: {MODEL_B['name']}")
    print(f"    C: {MODEL_C['name']}")
    print()

    # ---- Load Model A scores (from existing LF5) ----
    print("  Loading Model A scores (LF5 / BART-large-MNLI)...")
    lf_cols = ['study_id', 'lf5_label']
    # Check if lf5_score column exists
    _sample = pd.read_csv(LF_RESULTS_CSV, nrows=2)
    if 'lf5_score' in _sample.columns:
        lf_cols.append('lf5_score')
    del _sample

    df_lf = pd.read_csv(LF_RESULTS_CSV, low_memory=False, usecols=lf_cols)
    df_lf['study_id'] = df_lf['study_id'].astype(str)
    print(f"    LF5 data: {len(df_lf):,} reports")

    model_a_counts = Counter(df_lf['lf5_label'].tolist())
    for code, name in [(LABEL_POSITIVE, "POS"), (LABEL_NEGATIVE, "NEG"),
                       (LABEL_UNCERTAIN, "UNC"), (LABEL_ABSTAIN, "ABS")]:
        print(f"      {name}: {model_a_counts.get(code, 0):>8,}")
    print()

    # ---- Load reports (post-prefilter only for efficiency) ----
    print("  Loading parsed reports...")
    df_reports = pd.read_csv(PARSED_REPORTS, low_memory=False,
                             usecols=['study_id', 'subject_id',
                                      'impression_text', 'findings_text'])
    df_reports['study_id'] = df_reports['study_id'].astype(str)

    # Filter out pre-filter negatives (no pneumonia terms = no need for NLI)
    prefilter_ids = set()
    if os.path.exists(PREFILTER_CSV):
        df_pf = pd.read_csv(PREFILTER_CSV, usecols=['study_id'], low_memory=False)
        df_pf['study_id'] = df_pf['study_id'].astype(str)
        prefilter_ids = set(df_pf['study_id'].tolist())
        del df_pf

    df_process = df_reports[~df_reports['study_id'].isin(prefilter_ids)].copy()
    n_process = len(df_process)
    n_skipped = len(prefilter_ids)
    print(f"    Total reports: {len(df_reports):,}")
    print(f"    Pre-filter (skipped): {n_skipped:,}")
    print(f"    To process with NLI: {n_process:,}")
    print()

    texts = _prepare_texts(df_process)
    study_ids = df_process['study_id'].tolist()

    # ---- Run Model B ----
    print("=" * 70)
    print(f"  MODEL B: {MODEL_B['desc']}")
    print(f"    Batch size: {MODEL_B['batch_size']}")
    print(f"    Confidence threshold: {MODEL_B['confidence']}")
    print("=" * 70)
    print()

    t_b = time.time()
    model_b_labels, model_b_scores = run_nli_model(MODEL_B, texts)
    t_b_elapsed = time.time() - t_b

    print()
    print(f"  Model B complete in {t_b_elapsed/3600:.1f} hours")
    b_counts = Counter(model_b_labels)
    for code, name in [(LABEL_POSITIVE, "POS"), (LABEL_NEGATIVE, "NEG"),
                       (LABEL_UNCERTAIN, "UNC"), (LABEL_ABSTAIN, "ABS")]:
        print(f"    {name}: {b_counts.get(code, 0):>8,}")
    print()

    # ---- Run Model C ----
    print("=" * 70)
    print(f"  MODEL C: {MODEL_C['desc']}")
    print(f"    Batch size: {MODEL_C['batch_size']}")
    print(f"    Confidence threshold: {MODEL_C['confidence']}")
    print("=" * 70)
    print()

    t_c = time.time()
    model_c_labels, model_c_scores = run_nli_model(MODEL_C, texts)
    t_c_elapsed = time.time() - t_c

    print()
    print(f"  Model C complete in {t_c_elapsed/3600:.1f} hours")
    c_counts = Counter(model_c_labels)
    for code, name in [(LABEL_POSITIVE, "POS"), (LABEL_NEGATIVE, "NEG"),
                       (LABEL_UNCERTAIN, "UNC"), (LABEL_ABSTAIN, "ABS")]:
        print(f"    {name}: {c_counts.get(code, 0):>8,}")
    print()

    # ---- Build ensemble DataFrame ----
    print("  Building 3-model NLI ensemble...")

    df_nli = pd.DataFrame({
        'study_id': study_ids,
        'model_b_label': model_b_labels,
        'model_b_score': model_b_scores,
        'model_c_label': model_c_labels,
        'model_c_score': model_c_scores,
    })

    # Merge Model A (LF5)
    df_nli = df_nli.merge(df_lf[['study_id', 'lf5_label']], on='study_id', how='left')
    df_nli['model_a_label'] = df_nli['lf5_label'].fillna(LABEL_ABSTAIN).astype(int)
    df_nli.drop(columns=['lf5_label'], inplace=True)

    # ---- 3-Way Majority Vote ----
    def majority_vote(row):
        a = int(row['model_a_label'])
        b = int(row['model_b_label'])
        c = int(row['model_c_label'])
        votes = [a, b, c]

        pos = sum(1 for v in votes if v == LABEL_POSITIVE)
        neg = sum(1 for v in votes if v == LABEL_NEGATIVE)

        # Strict majority: ≥2 of 3 must agree
        if pos >= 2:
            return LABEL_POSITIVE
        if neg >= 2:
            return LABEL_NEGATIVE
        # No majority = excluded
        return 99  # EXCLUDED

    df_nli['l2_label'] = df_nli.apply(majority_vote, axis=1)

    # Merge back with full report list (including pre-filter)
    df_ensemble = df_reports[['study_id', 'subject_id']].merge(
        df_nli[['study_id', 'l2_label', 'model_a_label',
                'model_b_label', 'model_b_score',
                'model_c_label', 'model_c_score']],
        on='study_id', how='left'
    )
    # Pre-filter reports: set l2_label = NEGATIVE (no pneumonia terms)
    df_ensemble['l2_label'] = df_ensemble['l2_label'].fillna(LABEL_NEGATIVE).astype(int)
    df_ensemble['model_b_score'] = df_ensemble['model_b_score'].fillna(0.0)
    df_ensemble['model_c_score'] = df_ensemble['model_c_score'].fillna(0.0)

    # ---- Stats ----
    l2_counts = Counter(df_ensemble['l2_label'].tolist())
    n_total = len(df_ensemble)
    n_pos = l2_counts.get(LABEL_POSITIVE, 0)
    n_neg = l2_counts.get(LABEL_NEGATIVE, 0)
    n_exc = l2_counts.get(99, 0)

    # ---- Save ----
    output_cols = ['study_id', 'subject_id', 'l2_label',
                   'model_a_label', 'model_b_label', 'model_b_score',
                   'model_c_label', 'model_c_score']
    df_ensemble[output_cols].to_csv(ENSEMBLE_CSV, index=False)
    file_size_mb = os.path.getsize(ENSEMBLE_CSV) / (1024 * 1024)
    t_total = time.time() - t_start

    print()
    print("=" * 70)
    print("LAYER 2 COMPLETE — TRIPLE-MODEL NLI ENSEMBLE")
    print("=" * 70)
    print()
    print(f"  Models used:")
    print(f"    A: facebook/bart-large-mnli (reused LF5)")
    print(f"    B: {MODEL_B['name']}")
    print(f"    C: {MODEL_C['name']}")
    print()
    print(f"  Ensemble results ({n_total:,} reports):")
    print(f"    POSITIVE (≥2/3 agree POS):  {n_pos:>8,} ({100*n_pos/n_total:.1f}%)")
    print(f"    NEGATIVE (≥2/3 agree NEG):  {n_neg:>8,} ({100*n_neg/n_total:.1f}%)")
    print(f"    EXCLUDED (no majority):     {n_exc:>8,} ({100*n_exc/n_total:.1f}%)")
    print()
    print(f"  File: {ENSEMBLE_CSV}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Runtime: {t_total/3600:.1f} hours")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

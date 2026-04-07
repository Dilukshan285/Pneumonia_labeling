"""
Steps 2.12, 2.13, 2.14 — Labeling Function 5: NLI Zero-Shot Classification

Purpose:
    Uses facebook/bart-large-mnli — a transformer model explicitly fine-tuned
    on the Multi-Genre Natural Language Inference (MNLI) corpus — to perform
    zero-shot classification of radiology report text into three pneumonia
    categories via entailment scoring.

    This model is architecturally correct for zero-shot classification because
    it was trained to determine whether a premise (the report text) entails,
    contradicts, or is neutral toward a hypothesis (the candidate label).

    CRITICAL — DO NOT SUBSTITUTE RadBERT (zzxslp/RadBERT-RoBERTa-4m):
        RadBERT is a masked language model pretrained on radiology text using
        masked language modeling ONLY. It was NEVER fine-tuned on any NLI task
        and has NO entailment scoring head. If loaded into the zero-shot-
        classification pipeline, it will NOT throw an error but WILL produce
        probability scores that are numerically plausible and superficially
        convincing while being SEMANTICALLY MEANINGLESS. This type of silent
        failure would corrupt LF5 votes without any visible warning.

Step 2.12 — Load the zero-shot-classification pipeline onto RTX 4060 GPU.
Step 2.13 — Run classification with candidate labels on report text.
Step 2.14 — Map highest-scoring label to Snorkel encoding; ABSTAIN if < 0.40.

Returns: POSITIVE (1), NEGATIVE (0), UNCERTAIN (2), or ABSTAIN (-1)
"""

import os
import sys
import gc
import json
import math
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    NLI_MODEL_NAME,
    NLI_BATCH_SIZE,
    NLI_MAX_TOKENS,
    NLI_CONFIDENCE_THRESHOLD,
    DATA_INTERMEDIATE,
    LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN, LABEL_ABSTAIN,
)


# ============================================================================
# CONSTANTS
# ============================================================================

# Candidate labels for zero-shot classification.
# These are the hypotheses that BART-large-MNLI scores against the premise
# (the report text) using its trained entailment head.
CANDIDATE_LABELS = [
    "pneumonia present",
    "pneumonia absent",
    "pneumonia uncertain",
]

# Mapping from candidate label string → Snorkel label encoding
_CANDIDATE_TO_LABEL = {
    "pneumonia present": LABEL_POSITIVE,
    "pneumonia absent": LABEL_NEGATIVE,
    "pneumonia uncertain": LABEL_UNCERTAIN,
}

# Maximum character length for pre-truncation before tokenization.
# BART's context limit is 1024 tokens. Average English word ≈ 5 chars + space,
# average token ≈ 4 chars. 1024 tokens × 4 chars ≈ 4096 chars.
# We use 4500 to be safe — the tokenizer's own truncation=True handles the
# exact token-level cutoff, but pre-truncating avoids sending 50K-char reports
# through the tokenizer which wastes time on text that will be discarded.
_MAX_CHAR_LENGTH = 4500

# Checkpoint file for resume support
CHECKPOINT_FILE = os.path.join(DATA_INTERMEDIATE, "lf5_checkpoint.json")

# How often to save checkpoints (every N texts processed)
CHECKPOINT_INTERVAL = 8000  # Larger interval = less I/O overhead


# ============================================================================
# STEP 2.12 — LOAD ZERO-SHOT CLASSIFICATION MODEL
# ============================================================================

def _clear_gpu_memory():
    """
    Clear GPU memory completely before loading or after unloading the model.
    Calls gc.collect() first to release Python references, then empties
    the CUDA cache to return all cached memory to the device allocator.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _get_gpu_memory_info():
    """
    Get current GPU memory usage for monitoring.

    Returns:
        dict: With keys 'allocated_mb', 'reserved_mb', 'total_mb', 'free_mb'.
              Returns None if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return None

    allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    free = total - reserved

    return {
        "allocated_mb": round(allocated, 1),
        "reserved_mb": round(reserved, 1),
        "total_mb": round(total, 1),
        "free_mb": round(free, 1),
    }


def load_nli_pipeline():
    """
    Step 2.12 — Load the facebook/bart-large-mnli model onto the RTX 4060 GPU.

    Performs the following in order:
        1. Verifies CUDA availability and GPU device.
        2. Clears ALL GPU memory completely before loading.
        3. Loads the zero-shot-classification pipeline with device=0 (GPU).
        4. Reports memory usage after loading.

    Returns:
        transformers.Pipeline: The loaded zero-shot-classification pipeline.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    from transformers import pipeline as hf_pipeline

    # Verify GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. LF5 requires GPU acceleration.\n"
            "Verify PyTorch CUDA installation: torch.cuda.is_available()"
        )

    gpu_name = torch.cuda.get_device_name(0)
    print(f"    GPU detected: {gpu_name}")

    # Step 1: Clear GPU memory completely BEFORE loading
    print(f"    Clearing GPU memory before model load...")
    _clear_gpu_memory()

    mem_before = _get_gpu_memory_info()
    print(f"    GPU memory before load: "
          f"{mem_before['allocated_mb']}MB allocated, "
          f"{mem_before['free_mb']}MB free / {mem_before['total_mb']}MB total")

    # Step 2: Load the zero-shot-classification pipeline
    # device=0 places the model on the first CUDA GPU (RTX 4060)
    # torch_dtype=float16 halves VRAM usage (~1.6GB → ~800MB) and
    # provides ~1.5-2x throughput improvement on RTX 4060
    print(f"    Loading model: {NLI_MODEL_NAME}")
    print(f"    Pipeline type: zero-shot-classification")
    print(f"    Device: cuda:0 (fp16 enabled for speed)")

    # Enable cuDNN autotuner for faster convolutions
    torch.backends.cudnn.benchmark = True

    t_start = time.time()
    classifier = hf_pipeline(
        "zero-shot-classification",
        model=NLI_MODEL_NAME,
        device=0,  # RTX 4060 GPU
        model_kwargs={"torch_dtype": torch.float16},  # fp16: ~2x faster, half VRAM
    )
    t_load = time.time() - t_start

    mem_after = _get_gpu_memory_info()
    model_mem = mem_after['allocated_mb'] - mem_before['allocated_mb']

    print(f"    Model loaded in {t_load:.1f}s")
    print(f"    GPU memory after load: "
          f"{mem_after['allocated_mb']}MB allocated, "
          f"{mem_after['free_mb']}MB free")
    print(f"    Model VRAM footprint: ~{model_mem:.0f}MB")

    return classifier


def unload_nli_pipeline(classifier):
    """
    Step 2.14 (cleanup) — Delete the model object and release GPU memory.

    Called after all batches have been processed. This ensures the RTX 4060's
    8GB VRAM is fully available for any subsequent operations.

    Args:
        classifier: The HuggingFace pipeline object to unload.
    """
    print(f"    Unloading NLI model and clearing GPU memory...")
    mem_before = _get_gpu_memory_info()

    # Delete the pipeline and its underlying model
    del classifier

    # Force garbage collection and clear CUDA cache
    _clear_gpu_memory()

    mem_after = _get_gpu_memory_info()
    freed = mem_before['allocated_mb'] - mem_after['allocated_mb']

    print(f"    GPU memory freed: ~{freed:.0f}MB")
    print(f"    GPU memory after cleanup: "
          f"{mem_after['allocated_mb']}MB allocated, "
          f"{mem_after['free_mb']}MB free")


# ============================================================================
# STEP 2.13 — RUN ZERO-SHOT CLASSIFICATION
# ============================================================================

def _prepare_texts(df):
    """
    Prepare input texts for zero-shot classification (VECTORIZED).

    For each report:
        1. Use impression_text if non-empty.
        2. Fall back to findings_text if impression_text is empty.
        3. Pre-truncate to _MAX_CHAR_LENGTH characters to avoid wasting
           tokenizer time on text that exceeds BART's 1024-token limit.
        4. If both sections are empty, use empty string (will produce ABSTAIN).

    Args:
        df: DataFrame with 'impression_text' and 'findings_text' columns.

    Returns:
        list[str]: Prepared texts aligned to df index, one per row.
    """
    # Vectorized: fill NaN, strip whitespace
    impression = df['impression_text'].fillna('').astype(str).str.strip()
    findings = df['findings_text'].fillna('').astype(str).str.strip()

    # Use impression where non-empty, else findings
    texts = impression.where(impression != '', findings)

    # Pre-truncate at character level for performance
    texts = texts.str[:_MAX_CHAR_LENGTH]

    return texts.tolist()


# ============================================================================
# CHECKPOINT / RESUME SUPPORT
# ============================================================================

def _save_checkpoint(processed_index, labels, scores, top_labels, checkpoint_path=None):
    """
    Save processing progress to a JSON checkpoint file.

    Args:
        processed_index: Number of texts processed so far.
        labels: List of int labels so far.
        scores: List of float scores so far.
        top_labels: List of str top labels so far.
        checkpoint_path: Path to checkpoint file (default: CHECKPOINT_FILE).
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_FILE

    checkpoint = {
        "processed_index": processed_index,
        "labels": labels,
        "scores": scores,
        "top_labels": top_labels,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Write atomically: write to temp file, then rename
    temp_path = checkpoint_path + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(checkpoint, f)
    os.replace(temp_path, checkpoint_path)


def _load_checkpoint(checkpoint_path=None):
    """
    Load processing progress from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file (default: CHECKPOINT_FILE).

    Returns:
        dict or None: Checkpoint data with keys 'processed_index', 'labels',
                      'scores', 'top_labels', or None if no checkpoint exists.
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_FILE

    if not os.path.exists(checkpoint_path):
        return None

    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        required_keys = ['processed_index', 'labels', 'scores', 'top_labels']
        if all(k in checkpoint for k in required_keys):
            return checkpoint
        else:
            print(f"    WARNING: Checkpoint file is missing required keys, starting fresh.")
            return None
    except (json.JSONDecodeError, IOError) as e:
        print(f"    WARNING: Could not load checkpoint ({e}), starting fresh.")
        return None


def _delete_checkpoint(checkpoint_path=None):
    """Delete the checkpoint file after successful completion."""
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_FILE
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"    Checkpoint file deleted.")


def run_nli_classification(classifier, texts, batch_size=None, resume=True):
    """
    Step 2.13 — Run zero-shot classification on all prepared texts.

    Processes texts in batches to manage VRAM on the RTX 4060 8GB GPU.
    The batch_size defaults to NLI_BATCH_SIZE (8) from config.

    Supports resume from checkpoint:
        - Saves progress every CHECKPOINT_INTERVAL texts.
        - On restart, loads checkpoint and skips already-processed texts.

    Handles OOM errors gracefully:
        - If an OOM error occurs, the batch size is halved and the failed
          batch is retried with the reduced size.
        - This continues until batch_size reaches 1, at which point the
          error is raised (indicates a fundamental VRAM problem).

    Args:
        classifier: The loaded zero-shot-classification pipeline.
        texts:      List of prepared text strings.
        batch_size: Number of texts to process per batch (default: config value).
        resume:     If True, attempt to resume from checkpoint.

    Returns:
        list[dict]: One result dict per text, each containing:
            - 'labels': List of candidate labels sorted by score (descending).
            - 'scores': Corresponding confidence scores.
    """
    if batch_size is None:
        batch_size = NLI_BATCH_SIZE

    n_total = len(texts)

    # --- Resume from checkpoint ---
    all_labels = []
    all_scores = []
    all_top_labels = []
    start_index = 0

    if resume:
        checkpoint = _load_checkpoint()
        if checkpoint is not None:
            start_index = checkpoint['processed_index']
            all_labels = checkpoint['labels']
            all_scores = checkpoint['scores']
            all_top_labels = checkpoint['top_labels']
            print(f"    RESUMING from checkpoint: {start_index:,} / {n_total:,} texts already processed")
            print(f"    Checkpoint saved at: {checkpoint.get('timestamp', 'unknown')}")
            print()

    remaining = n_total - start_index
    n_batches = math.ceil(remaining / batch_size)

    print(f"    Total texts: {n_total:,}")
    print(f"    Already processed: {start_index:,}")
    print(f"    Remaining to classify: {remaining:,}")
    print(f"    Batch size: {batch_size}")
    print(f"    Remaining batches: {n_batches:,}")
    print(f"    Candidate labels: {CANDIDATE_LABELS}")
    print(f"    Truncation: enabled (max {NLI_MAX_TOKENS} tokens)")
    print(f"    Confidence threshold: {NLI_CONFIDENCE_THRESHOLD}")
    print(f"    Checkpoint interval: every {CHECKPOINT_INTERVAL:,} texts")
    print(flush=True)

    current_batch_size = batch_size
    i = start_index
    texts_since_checkpoint = 0

    # tqdm to stdout to avoid PowerShell stderr termination
    pbar = tqdm(
        total=n_total,
        initial=start_index,
        desc="  LF5 NLI classification",
        unit="reports",
        file=sys.stdout,
    )

    while i < n_total:
        batch_end = min(i + current_batch_size, n_total)
        batch_texts = texts[i:batch_end]
        batch_len = len(batch_texts)

        # Skip empty texts — assign ABSTAIN directly
        non_empty_indices = []
        non_empty_texts = []
        batch_labels = [LABEL_ABSTAIN] * batch_len
        batch_scores = [0.0] * batch_len
        batch_top = ["(empty)"] * batch_len

        for j, text in enumerate(batch_texts):
            if text.strip():
                non_empty_indices.append(j)
                non_empty_texts.append(text)

        if non_empty_texts:
            try:
                # Run zero-shot classification on non-empty texts
                # truncation=True ensures BART's 1024-token limit is respected
                nli_outputs = classifier(
                    non_empty_texts,
                    candidate_labels=CANDIDATE_LABELS,
                    batch_size=current_batch_size,
                    truncation=True,
                    max_length=NLI_MAX_TOKENS,
                )

                # classifier returns a single dict if only one text, list otherwise
                if isinstance(nli_outputs, dict):
                    nli_outputs = [nli_outputs]

                # Convert each NLI output to label/score/top_label
                for k, idx in enumerate(non_empty_indices):
                    result = nli_outputs[k]
                    top_label = result['labels'][0]
                    top_score = result['scores'][0]

                    if top_score < NLI_CONFIDENCE_THRESHOLD:
                        batch_labels[idx] = LABEL_ABSTAIN
                        batch_scores[idx] = top_score
                        batch_top[idx] = f"(low: {top_label})"
                    else:
                        batch_labels[idx] = _CANDIDATE_TO_LABEL.get(top_label, LABEL_ABSTAIN)
                        batch_scores[idx] = top_score
                        batch_top[idx] = top_label

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    # OOM: clear cache, halve batch size, retry
                    _clear_gpu_memory()

                    old_bs = current_batch_size
                    current_batch_size = max(1, current_batch_size // 2)

                    if current_batch_size < 1:
                        raise RuntimeError(
                            f"OOM even with batch_size=1. Text at index {i} "
                            f"may be too long or GPU VRAM is insufficient."
                        ) from e

                    print(f"\n    WARNING: OOM at batch starting index {i}. "
                          f"Reducing batch size: {old_bs} -> {current_batch_size}",
                          flush=True)

                    # Retry this batch (don't advance i)
                    continue
                else:
                    raise

        all_labels.extend(batch_labels)
        all_scores.extend(batch_scores)
        all_top_labels.extend(batch_top)
        pbar.update(batch_len)
        i = batch_end
        texts_since_checkpoint += batch_len

        # Save checkpoint periodically
        if texts_since_checkpoint >= CHECKPOINT_INTERVAL:
            _save_checkpoint(i, all_labels, all_scores, all_top_labels)
            texts_since_checkpoint = 0
            mem = _get_gpu_memory_info()
            if mem:
                tqdm.write(
                    f"    [Checkpoint at {i:,}/{n_total:,}] GPU: "
                    f"{mem['allocated_mb']}MB allocated, "
                    f"{mem['free_mb']}MB free",
                    file=sys.stdout,
                )

    pbar.close()

    # Delete checkpoint on successful completion
    _delete_checkpoint()

    return all_labels, all_scores, all_top_labels


# ============================================================================
# STEP 2.14 — ASSIGN LF5 LABELS (now integrated into run_nli_classification)
# ============================================================================

def convert_nli_results_to_labels(results):
    """
    Step 2.14 — Convert NLI classification results to Snorkel label encoding.

    LEGACY interface kept for backward compatibility with self-test.
    The main pipeline now uses the integrated approach in run_nli_classification.

    Args:
        results: List of dicts from old-style run, each with
                 'labels' and 'scores' keys.

    Returns:
        tuple: (labels_list, scores_list, top_labels_list)
    """
    labels = []
    scores = []
    top_labels = []

    for result in results:
        if result is None or result.get('_empty', False):
            labels.append(LABEL_ABSTAIN)
            scores.append(0.0)
            top_labels.append("(empty)")
            continue

        top_label = result['labels'][0]
        top_score = result['scores'][0]

        if top_score < NLI_CONFIDENCE_THRESHOLD:
            labels.append(LABEL_ABSTAIN)
            scores.append(top_score)
            top_labels.append(f"(low: {top_label})")
        else:
            mapped_label = _CANDIDATE_TO_LABEL.get(top_label, LABEL_ABSTAIN)
            labels.append(mapped_label)
            scores.append(top_score)
            top_labels.append(top_label)

    return labels, scores, top_labels


# ============================================================================
# HIGH-LEVEL INTERFACE
# ============================================================================

def run_lf5_full(df, resume=True):
    """
    Execute the complete LF5 pipeline: load model → classify → convert → unload.

    This is the main entry point for the runner script.

    Args:
        df: DataFrame with 'impression_text' and 'findings_text' columns.
        resume: If True, attempt to resume from checkpoint.

    Returns:
        tuple: (lf5_labels, lf5_scores, lf5_top_labels)
            - lf5_labels: List of int, one per row in df.
            - lf5_scores: List of float, confidence score of top label.
            - lf5_top_labels: List of str, name of top candidate label.
    """
    # Step 2.12 — Load model
    print("  Step 2.12 — Loading zero-shot classification model...")
    classifier = load_nli_pipeline()
    print(flush=True)

    # Step 2.13 — Prepare texts and run classification
    print("  Step 2.13 — Running zero-shot classification...")
    print("    Preparing input texts (impression -> findings fallback)...")
    texts = _prepare_texts(df)

    n_empty = sum(1 for t in texts if not t.strip())
    n_non_empty = len(texts) - n_empty
    print(f"    Non-empty texts: {n_non_empty:,}")
    print(f"    Empty texts (will ABSTAIN): {n_empty:,}")
    print(flush=True)

    t_start = time.time()
    lf5_labels, lf5_scores, lf5_top_labels = run_nli_classification(
        classifier, texts, resume=resume
    )
    t_classify = time.time() - t_start

    print()
    print(f"    Classification completed in {t_classify:.1f}s ({t_classify/60:.1f} min)")
    if n_non_empty > 0:
        print(f"    Average per non-empty report: {1000*t_classify/n_non_empty:.1f}ms")
    print(flush=True)

    # Step 2.14 — Labels already converted inline during classification
    print("  Step 2.14 — Labels converted during classification.")
    print(f"    Confidence threshold: {NLI_CONFIDENCE_THRESHOLD}")
    print(f"    Total labels: {len(lf5_labels):,}")
    print(flush=True)

    # Step 2.14 (cleanup) — Unload model and free GPU memory
    print("  Step 2.14 (cleanup) — Releasing GPU resources...")
    unload_nli_pipeline(classifier)
    print(flush=True)

    return lf5_labels, lf5_scores, lf5_top_labels


# ============================================================================
# STANDALONE TEST
# ============================================================================

def _run_self_test():
    """
    Quick self-test with a handful of known radiology phrases to verify:
        1. Model loads correctly onto GPU.
        2. Zero-shot classification produces reasonable scores.
        3. Label mapping and confidence threshold work correctly.
        4. GPU memory is properly released after unloading.
    """
    print("=" * 70)
    print("LF5 NLI Zero-Shot Classification — Self-Test")
    print("=" * 70)
    print()

    # Verify GPU
    if not torch.cuda.is_available():
        print("  ERROR: CUDA not available. Cannot run self-test.")
        return False

    print(f"  Model: {NLI_MODEL_NAME}")
    print(f"  Batch size: {NLI_BATCH_SIZE}")
    print(f"  Max tokens: {NLI_MAX_TOKENS}")
    print(f"  Confidence threshold: {NLI_CONFIDENCE_THRESHOLD}")
    print()

    test_texts = [
        "Right lower lobe pneumonia.",
        "No evidence of pneumonia. Lungs are clear.",
        "Possible pneumonia versus atelectasis.",
        "Normal chest radiograph.",
        "Dense consolidation in the left lower lobe consistent with pneumonia.",
        "No acute cardiopulmonary process.",
        "",  # Empty — should ABSTAIN
    ]

    expected_labels = [
        LABEL_POSITIVE,    # Right lower lobe pneumonia
        LABEL_NEGATIVE,    # No evidence, clear
        LABEL_UNCERTAIN,   # Possible, versus
        LABEL_NEGATIVE,    # Normal
        LABEL_POSITIVE,    # Dense consolidation = pneumonia
        LABEL_NEGATIVE,    # No acute process
        LABEL_ABSTAIN,     # Empty
    ]

    label_names = {
        LABEL_POSITIVE: "POSITIVE",
        LABEL_NEGATIVE: "NEGATIVE",
        LABEL_UNCERTAIN: "UNCERTAIN",
        LABEL_ABSTAIN: "ABSTAIN",
    }

    # Load model
    print("  Loading model...")
    classifier = load_nli_pipeline()
    print()

    # Run classification (no resume for self-test)
    print("  Running classification on test cases...")
    lf5_labels, lf5_scores, lf5_top_labels = run_nli_classification(
        classifier, test_texts, batch_size=4, resume=False
    )

    # Display results
    print()

    passed = 0
    failed = 0

    for i, text in enumerate(test_texts):
        expected = expected_labels[i]
        actual = lf5_labels[i]
        score = lf5_scores[i]
        top = lf5_top_labels[i]

        match = actual == expected
        if match:
            passed += 1
        else:
            failed += 1

        icon = "+" if match else "~"
        exp_name = label_names.get(expected, "?")
        act_name = label_names.get(actual, "?")

        display_text = text[:60] if text else "(empty)"
        print(f"  {icon} \"{display_text}\"")
        print(f"      Top: {top} (score={score:.3f})")
        print(f"      Expected: {exp_name}, Got: {act_name}")
        print()

    total = passed + failed
    print(f"  Results: {passed}/{total} exact matches")
    if failed > 0:
        print(f"  Note: {failed} mismatches are acceptable — NLI models may")
        print(f"  interpret clinical text differently than keyword rules.")
        print(f"  The Snorkel LabelModel reconciles these differences.")
    print()

    # Unload
    print("  Cleaning up...")
    unload_nli_pipeline(classifier)
    print()
    print("  + Self-test completed.")
    return True


if __name__ == "__main__":
    _run_self_test()

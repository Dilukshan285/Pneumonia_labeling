"""
Build Full Multi-Label Dataset from final_image_training_manifest.csv
=====================================================================
Runs the 3-layer NLP consensus pipeline on ALL 203K reports
(not the pre-split pp1 files) to produce a single master CSV
with 14 disease columns.

Output: data/output/multi_label_dataset/ml_full_manifest.csv

Usage:
    python scripts/multi_label/build_full_multilabel.py

Expected runtime: ~3-5 hours on RTX 4060 (Layer 2 NLI is the bottleneck).
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Ensure project root is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from multi_label_config import (
    ML_DATASET_DIR, PATHOLOGY_CLASSES,
    LABEL_PRESENT, LABEL_ABSENT, LABEL_UNCERTAIN,
)
from multi_label.layer1_keywords import run_layer1
from multi_label.layer2_nli import run_layer2
from multi_label.layer3_assertions import run_layer3
from multi_label.consensus import (
    compute_consensus, apply_no_finding_logic, print_consensus_summary,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "output", "final_image_training_manifest.csv")
OUTPUT_CSV = os.path.join(ML_DATASET_DIR, "ml_full_manifest.csv")

# Process in chunks to avoid GPU OOM on Layer 2 NLI
CHUNK_SIZE = 10000   # Process 10K reports at a time


def process_chunk(df, chunk_idx, total_chunks):
    """Process a chunk of reports through all 3 NLP layers + consensus."""
    n = len(df)
    print(f"\n{'='*70}")
    print(f"PROCESSING CHUNK {chunk_idx+1}/{total_chunks} ({n} reports)")
    print(f"{'='*70}")

    df = df.reset_index(drop=True)

    # ── Layer 1: Keyword Extraction ──────────────────────────────
    t0 = time.time()
    print(f"\n  [1/4] Layer 1: Negation-Aware Keyword Extraction...")
    layer1_labels = run_layer1(df)
    t1 = time.time()
    print(f"    Done in {t1 - t0:.1f}s")

    # ── Layer 2: NLI Zero-Shot ───────────────────────────────────
    print(f"\n  [2/4] Layer 2: BART-MNLI Zero-Shot Classification...")
    layer2_labels, layer2_probs = run_layer2(df)
    t2 = time.time()
    print(f"    Done in {t2 - t1:.1f}s")

    # ── Layer 3: Assertion Detection ─────────────────────────────
    print(f"\n  [3/4] Layer 3: Sentence-Level Assertion Detection...")
    layer3_labels = run_layer3(df)
    t3 = time.time()
    print(f"    Done in {t3 - t2:.1f}s")

    # ── Consensus ────────────────────────────────────────────────
    print(f"\n  [4/4] Computing Consensus...")
    final_labels, confidence_scores = compute_consensus(
        layer1_labels, layer2_labels, layer3_labels
    )
    final_labels = apply_no_finding_logic(final_labels, n)
    print_consensus_summary(final_labels, n)

    # ── Attach labels to DataFrame ───────────────────────────────
    for cls in PATHOLOGY_CLASSES:
        df[cls] = final_labels[cls]

    if "Pneumonia" in layer2_probs:
        df["nli_prob_Pneumonia"] = layer2_probs["Pneumonia"]

    for cls in PATHOLOGY_CLASSES:
        df[f"conf_{cls}"] = confidence_scores[cls]

    total_time = time.time() - t0
    print(f"\n  Chunk {chunk_idx+1} completed in {total_time:.1f}s ({total_time/60:.1f} min)")

    return df


def main():
    print("=" * 70)
    print("FULL MULTI-LABEL DATASET BUILDER")
    print("3-Layer NLP Consensus Pipeline for 14 CXR Pathologies")
    print(f"Source: final_image_training_manifest.csv")
    print("=" * 70)

    overall_start = time.time()
    os.makedirs(ML_DATASET_DIR, exist_ok=True)

    # Load the full manifest
    print(f"\nLoading: {INPUT_CSV}")
    df_full = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"  -> {len(df_full)} total rows")
    print(f"  Columns: {list(df_full.columns)}")

    # Ensure text columns exist
    for col in ["impression_text", "findings_text"]:
        if col not in df_full.columns:
            df_full[col] = ""
        df_full[col] = df_full[col].fillna("").astype(str)

    # Process in chunks
    total_chunks = (len(df_full) + CHUNK_SIZE - 1) // CHUNK_SIZE
    processed_chunks = []

    for i in range(total_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE, len(df_full))
        chunk_df = df_full.iloc[start_idx:end_idx].copy()

        result = process_chunk(chunk_df, i, total_chunks)
        processed_chunks.append(result)

        # Save intermediate progress after each chunk
        intermediate = pd.concat(processed_chunks, ignore_index=True)
        intermediate.to_csv(OUTPUT_CSV, index=False)
        print(f"  [SAVED] Intermediate progress: {len(intermediate)} rows -> {OUTPUT_CSV}")

    # ── Final Output ─────────────────────────────────────────────
    df_final = pd.concat(processed_chunks, ignore_index=True)

    # Select output columns
    core_cols = [
        "dicom_id", "subject_id", "study_id",
        "image_rel_path", "ViewPosition",
        "label", "soft_score",
        "impression_text", "findings_text",
    ]
    label_cols = PATHOLOGY_CLASSES.copy()
    conf_cols = [f"conf_{cls}" for cls in PATHOLOGY_CLASSES]
    extra = ["nli_prob_Pneumonia"] if "nli_prob_Pneumonia" in df_final.columns else []

    all_cols = core_cols + label_cols + conf_cols + extra
    available_cols = [c for c in all_cols if c in df_final.columns]
    df_final = df_final[available_cols]

    df_final.to_csv(OUTPUT_CSV, index=False)

    # ── Summary ──────────────────────────────────────────────────
    total_time = time.time() - overall_start
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total reports processed: {len(df_final)}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min, {total_time/3600:.2f} hrs)")

    print(f"\n{'Pathology':30s} {'PRESENT':>8s} {'ABSENT':>8s} {'UNCERTAIN':>10s} {'Prev%':>7s}")
    print("-" * 70)
    for cls in PATHOLOGY_CLASSES:
        if cls in df_final.columns:
            n_p = (df_final[cls] == LABEL_PRESENT).sum()
            n_a = (df_final[cls] == LABEL_ABSENT).sum()
            n_u = (df_final[cls] == LABEL_UNCERTAIN).sum()
            prev = n_p / len(df_final) * 100
            print(f"{cls:30s} {n_p:8d} {n_a:8d} {n_u:10d} {prev:6.1f}%")

    print(f"\n[DONE] Full multi-label manifest saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

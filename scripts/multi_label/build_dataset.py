"""
Build Multi-Label Dataset — Master Orchestrator
=================================================
Loads existing pp1_train/val/test splits, runs all 3 NLP layers,
applies consensus voting, and produces final multi-label CSV files
in data/output/multi_label_dataset/.

Usage:
    python scripts/multi_label/build_dataset.py

Expected runtime: ~2-3 hours on RTX 4060 (Layer 2 NLI is the bottleneck).
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
    PP1_TRAIN_CSV, PP1_VAL_CSV, PP1_TEST_CSV,
    ML_DATASET_DIR, ML_TRAIN_CSV, ML_VAL_CSV, ML_TEST_CSV,
    PATHOLOGY_CLASSES, LABEL_PRESENT, LABEL_ABSENT, LABEL_UNCERTAIN,
)
from multi_label.layer1_keywords import run_layer1
from multi_label.layer2_nli import run_layer2
from multi_label.layer3_assertions import run_layer3
from multi_label.consensus import (
    compute_consensus, apply_no_finding_logic, print_consensus_summary,
)


def load_split(csv_path, split_name):
    """Load a pp1 split CSV and prepare it for processing."""
    print(f"\nLoading {split_name}: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  -> {len(df)} rows, columns: {list(df.columns)}")
    
    # Ensure text columns exist and handle NaN
    for col in ["impression_text", "findings_text"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    
    return df


def process_split(df, split_name):
    """
    Process a single split through all 3 layers + consensus.
    
    Returns DataFrame with multi-label columns appended.
    """
    n = len(df)
    print(f"\n{'='*70}")
    print(f"PROCESSING {split_name.upper()} SPLIT ({n} reports)")
    print(f"{'='*70}")
    
    # Reset index for consistent iteration
    df = df.reset_index(drop=True)
    
    # ── Layer 1: Keyword Extraction ──────────────────────────────
    t0 = time.time()
    print(f"\n[1/4] Running Layer 1: Negation-Aware Keyword Extraction...")
    layer1_labels = run_layer1(df)
    t1 = time.time()
    print(f"  Layer 1 took {t1 - t0:.1f}s")
    
    # ── Layer 2: NLI Zero-Shot ───────────────────────────────────
    print(f"\n[2/4] Running Layer 2: BART-MNLI Zero-Shot Classification...")
    layer2_labels, layer2_probs = run_layer2(df)
    t2 = time.time()
    print(f"  Layer 2 took {t2 - t1:.1f}s")
    
    # ── Layer 3: Assertion Detection ─────────────────────────────
    print(f"\n[3/4] Running Layer 3: Sentence-Level Assertion Detection...")
    layer3_labels = run_layer3(df)
    t3 = time.time()
    print(f"  Layer 3 took {t3 - t2:.1f}s")
    
    # ── Consensus ────────────────────────────────────────────────
    print(f"\n[4/4] Computing Consensus...")
    final_labels, confidence_scores = compute_consensus(
        layer1_labels, layer2_labels, layer3_labels
    )
    
    # Apply No_Finding mutual exclusivity
    final_labels = apply_no_finding_logic(final_labels, n)
    
    print_consensus_summary(final_labels, n)
    
    # ── Attach labels to DataFrame ───────────────────────────────
    for cls in PATHOLOGY_CLASSES:
        df[cls] = final_labels[cls]
    
    # Also attach NLI probabilities for Pneumonia (useful for compatibility)
    if "Pneumonia" in layer2_probs:
        df["nli_prob_Pneumonia"] = layer2_probs["Pneumonia"]
    
    # Add per-condition confidence
    for cls in PATHOLOGY_CLASSES:
        df[f"conf_{cls}"] = confidence_scores[cls]
    
    total_time = time.time() - t0
    print(f"\n  Total processing time for {split_name}: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    return df


def select_output_columns(df, split_name):
    """Select and order the final output columns."""
    # Core columns from original data
    core_cols = [
        "dicom_id", "subject_id", "study_id", 
        "image_rel_path", "ViewPosition",
        "impression_text", "findings_text",
    ]
    
    # Preserve original pneumonia label for comparison
    if "label" in df.columns:
        core_cols.append("label")
    if "soft_score" in df.columns:
        core_cols.append("soft_score")
    
    # Multi-label columns
    label_cols = PATHOLOGY_CLASSES.copy()
    
    # Confidence columns
    conf_cols = [f"conf_{cls}" for cls in PATHOLOGY_CLASSES]
    
    # NLI probability for pneumonia
    extra = []
    if "nli_prob_Pneumonia" in df.columns:
        extra.append("nli_prob_Pneumonia")
    
    # Split indicator
    df["split"] = split_name
    
    # Filter to only columns that exist
    all_cols = core_cols + label_cols + conf_cols + extra + ["split"]
    available_cols = [c for c in all_cols if c in df.columns]
    
    return df[available_cols]


def verify_output(df, split_name):
    """Run basic verification checks on the output DataFrame."""
    print(f"\n  Verification for {split_name}:")
    
    # Check no NaN in label columns
    for cls in PATHOLOGY_CLASSES:
        if cls in df.columns:
            nan_count = df[cls].isna().sum()
            if nan_count > 0:
                print(f"    [!] {cls} has {nan_count} NaN values!")
            else:
                print(f"    [OK] {cls}: no NaN")
    
    # Check label value range
    for cls in PATHOLOGY_CLASSES:
        if cls in df.columns:
            unique_vals = set(df[cls].unique())
            valid_vals = {LABEL_PRESENT, LABEL_ABSENT, LABEL_UNCERTAIN}
            if not unique_vals.issubset(valid_vals):
                print(f"    [!] {cls} has unexpected values: {unique_vals - valid_vals}")
    
    # Check image paths are present
    if "image_rel_path" in df.columns:
        empty_paths = df["image_rel_path"].isna().sum() + (df["image_rel_path"] == "").sum()
        print(f"    Image paths: {len(df) - empty_paths}/{len(df)} present")
    
    # Compare with original pneumonia labels if available
    if "label" in df.columns and "Pneumonia" in df.columns:
        orig = df["label"].values
        new = df["Pneumonia"].values
        # Compare only where new label is not uncertain
        mask = new != LABEL_UNCERTAIN
        if mask.sum() > 0:
            agreement = np.mean(orig[mask] == new[mask])
            print(f"    Pneumonia agreement with original pipeline: {agreement:.3f} "
                  f"({mask.sum()} non-uncertain samples)")
    
    print(f"    Total rows: {len(df)}")


def main():
    """Main orchestrator — process all splits and save results."""
    print("=" * 70)
    print("MULTI-LABEL DATASET BUILDER")
    print("3-Layer NLP Consensus Pipeline for 14 CXR Pathologies")
    print("=" * 70)
    
    overall_start = time.time()
    
    # Create output directory
    os.makedirs(ML_DATASET_DIR, exist_ok=True)
    print(f"\nOutput directory: {ML_DATASET_DIR}")
    
    # Process each split
    splits = [
        ("train", PP1_TRAIN_CSV, ML_TRAIN_CSV),
        ("val",   PP1_VAL_CSV,   ML_VAL_CSV),
        ("test",  PP1_TEST_CSV,  ML_TEST_CSV),
    ]
    
    all_dfs = []
    
    for split_name, input_csv, output_csv in splits:
        # Load
        df = load_split(input_csv, split_name)
        
        # Process through 3-layer pipeline
        df = process_split(df, split_name)
        
        # Select output columns
        df = select_output_columns(df, split_name)
        
        # Verify
        verify_output(df, split_name)
        
        # Save
        df.to_csv(output_csv, index=False)
        print(f"\n  [OK] Saved: {output_csv} ({len(df)} rows)")
        
        all_dfs.append(df)
    
    # ── Final Summary ────────────────────────────────────────────
    total_time = time.time() - overall_start
    combined = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total reports processed: {len(combined)}")
    print(f"  Train: {sum(1 for d in all_dfs[0]['split'] if d == 'train')}")
    print(f"  Val:   {sum(1 for d in all_dfs[1]['split'] if d == 'val')}")
    print(f"  Test:  {sum(1 for d in all_dfs[2]['split'] if d == 'test')}")
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min, {total_time/3600:.2f} hrs)")
    
    # Label prevalence across entire dataset
    print(f"\n{'Pathology':30s} {'PRESENT':>8s} {'ABSENT':>8s} {'UNCERTAIN':>10s} {'Prev%':>7s}")
    print("-" * 70)
    for cls in PATHOLOGY_CLASSES:
        if cls in combined.columns:
            n_p = (combined[cls] == LABEL_PRESENT).sum()
            n_a = (combined[cls] == LABEL_ABSENT).sum()
            n_u = (combined[cls] == LABEL_UNCERTAIN).sum()
            prev = n_p / len(combined) * 100
            print(f"{cls:30s} {n_p:8d} {n_a:8d} {n_u:10d} {prev:6.1f}%")
    
    # Patient-level split integrity check
    print(f"\n--- Patient-Level Split Integrity ---")
    if "subject_id" in combined.columns:
        for i, (name_i, _, _) in enumerate(splits):
            for j, (name_j, _, _) in enumerate(splits):
                if i >= j:
                    continue
                ids_i = set(all_dfs[i]["subject_id"].unique())
                ids_j = set(all_dfs[j]["subject_id"].unique())
                overlap = ids_i & ids_j
                if overlap:
                    print(f"  [!] {name_i} <-> {name_j}: {len(overlap)} overlapping patients!")
                else:
                    print(f"  [OK] {name_i} <-> {name_j}: no patient overlap")
    
    print(f"\n[DONE] Multi-label dataset saved to: {ML_DATASET_DIR}")
    print(f"   Files: ml_train.csv, ml_val.csv, ml_test.csv")


if __name__ == "__main__":
    main()

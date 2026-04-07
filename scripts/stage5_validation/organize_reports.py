"""
Organize MIMIC-CXR reports into positive/ and negative/ folders
based on final_pneumonia_labels.csv for manual reading.

Copies report files (does NOT move or delete originals).
Keeps CSV file intact.

Input:  final_pneumonia_labels.csv
        MIMIC-CXR reports at files/p{xx}/p{subject_id}/s{study_id}.txt
Output: organized_reports/positive/  (42,928 reports)
        organized_reports/negative/  (160,153 reports)
"""

import os
import sys
import time
import shutil

import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    FINAL_LABELS_CSV,
    REPORTS_DIR,
    PROJECT_DIR,
)


def find_report_path(reports_dir, subject_id, study_id):
    """
    Build the path to a report file.
    Structure: files/p{first2digits}/p{subject_id}/s{study_id}.txt
    """
    subj_str = str(int(subject_id))
    study_str = str(study_id)

    # Remove 's' prefix from study_id if present
    if study_str.startswith('s'):
        study_str = study_str[1:]

    # Prefix folder: first 2 digits of subject_id → p10, p11, etc.
    prefix = "p" + subj_str[:2]

    # Full path
    path = os.path.join(reports_dir, prefix, f"p{subj_str}", f"s{study_str}.txt")
    return path


def main():
    t_start = time.time()

    print("=" * 70)
    print("ORGANIZE REPORTS INTO POSITIVE / NEGATIVE FOLDERS")
    print("(Copy reports based on final_pneumonia_labels.csv)")
    print("=" * 70)
    print()

    # Load labels
    if not os.path.exists(FINAL_LABELS_CSV):
        print(f"ERROR: {FINAL_LABELS_CSV} not found.")
        return 1

    print(f"Loading labels from {FINAL_LABELS_CSV}...")
    df = pd.read_csv(FINAL_LABELS_CSV)
    n_total = len(df)
    n_pos = (df['label'] == 1).sum()
    n_neg = (df['label'] == 0).sum()
    print(f"  Total: {n_total:,}")
    print(f"  POSITIVE (label=1): {n_pos:,}")
    print(f"  NEGATIVE (label=0): {n_neg:,}")
    print()

    # Create output directories
    output_base = os.path.join(PROJECT_DIR, "organized_reports")
    pos_dir = os.path.join(output_base, "positive")
    neg_dir = os.path.join(output_base, "negative")

    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)
    print(f"  Output: {output_base}")
    print(f"    positive/ → {pos_dir}")
    print(f"    negative/ → {neg_dir}")
    print()

    # Copy the CSV into the output folder too
    csv_copy_path = os.path.join(output_base, "final_pneumonia_labels.csv")
    shutil.copy2(FINAL_LABELS_CSV, csv_copy_path)
    print(f"  CSV copied to: {csv_copy_path}")
    print()

    # Process reports
    stats = {"copied_pos": 0, "copied_neg": 0, "not_found": 0, "errors": 0}
    not_found_examples = []

    print(f"Copying reports...")
    sys.stdout.flush()

    for idx, row in df.iterrows():
        subject_id = row['subject_id']
        study_id = row['study_id']
        label = int(row['label'])

        # Find source path
        src_path = find_report_path(REPORTS_DIR, subject_id, study_id)

        if not os.path.exists(src_path):
            stats["not_found"] += 1
            if len(not_found_examples) < 10:
                not_found_examples.append(src_path)
            continue

        # Destination: use subject_study naming to keep unique
        study_str = str(study_id)
        if study_str.startswith('s'):
            study_str = study_str[1:]
        filename = f"p{int(subject_id)}_s{study_str}.txt"

        if label == 1:
            dst_path = os.path.join(pos_dir, filename)
            stats["copied_pos"] += 1
        else:
            dst_path = os.path.join(neg_dir, filename)
            stats["copied_neg"] += 1

        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 5:
                print(f"  ERROR copying {src_path}: {e}")

        # Progress
        if (idx + 1) % 20000 == 0:
            elapsed = time.time() - t_start
            pct = 100 * (idx + 1) / n_total
            print(f"  Progress: {idx+1:,}/{n_total:,} ({pct:.1f}%) — "
                  f"{stats['copied_pos']:,} pos, {stats['copied_neg']:,} neg — "
                  f"{elapsed:.0f}s elapsed")
            sys.stdout.flush()

    t_total = time.time() - t_start

    print()
    print("=" * 70)
    print("COMPLETE — REPORTS ORGANIZED")
    print("=" * 70)
    print()
    print(f"  POSITIVE copied: {stats['copied_pos']:,}")
    print(f"  NEGATIVE copied: {stats['copied_neg']:,}")
    print(f"  Not found:       {stats['not_found']:,}")
    print(f"  Errors:          {stats['errors']:,}")
    print()

    if not_found_examples:
        print(f"  Not-found examples (first {len(not_found_examples)}):")
        for p in not_found_examples:
            print(f"    {p}")
        print()

    print(f"  Output folder:   {output_base}")
    print(f"    positive/:     {stats['copied_pos']:,} files")
    print(f"    negative/:     {stats['copied_neg']:,} files")
    print(f"    CSV:           {csv_copy_path}")
    print(f"  Runtime:         {t_total:.1f}s")
    print()
    print(f"  Original reports in {REPORTS_DIR} are UNTOUCHED.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

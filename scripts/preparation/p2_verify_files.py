"""
Step P2 — Verify MIMIC-CXR Files
Checks that all required MIMIC-CXR files are present, non-empty, and accessible.
"""

import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import REPORTS_DIR, SPLIT_CSV, CHEXPERT_CSV, METADATA_CSV


def check_file(path, description):
    """Check if a file exists and is non-empty."""
    if not os.path.exists(path):
        print(f"  [MISSING]  {description}")
        print(f"             Expected at: {path}")
        return False

    size = os.path.getsize(path)
    if size == 0:
        print(f"  [EMPTY]    {description}")
        print(f"             File exists but is 0 bytes: {path}")
        return False

    size_mb = size / (1024 * 1024)
    print(f"  [OK]       {description}")
    print(f"             Path: {path}")
    print(f"             Size: {size_mb:.1f} MB")
    return True


def check_reports_dir(path):
    """Check if the reports directory exists and contains expected subfolders."""
    if not os.path.exists(path):
        print(f"  [MISSING]  Reports directory")
        print(f"             Expected at: {path}")
        return False

    if not os.path.isdir(path):
        print(f"  [ERROR]    Reports path exists but is not a directory: {path}")
        return False

    # Check for expected subfolders (p10, p11, ..., p19)
    subfolders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    expected = [f"p{i}" for i in range(10, 20)]
    found = [d for d in expected if d in subfolders]

    # Count total report files recursively
    report_count = 0
    patient_count = 0
    for subfolder in subfolders:
        subfolder_path = os.path.join(path, subfolder)
        for patient_dir in os.listdir(subfolder_path):
            patient_path = os.path.join(subfolder_path, patient_dir)
            if os.path.isdir(patient_path):
                patient_count += 1
                for f in os.listdir(patient_path):
                    if f.endswith('.txt'):
                        report_count += 1

    print(f"  [OK]       Reports directory")
    print(f"             Path: {path}")
    print(f"             Top-level subfolders found: {len(subfolders)} ({', '.join(sorted(found))})")
    print(f"             Total patient directories: {patient_count:,}")
    print(f"             Total report files (.txt): {report_count:,}")
    return True


def main():
    print("=" * 70)
    print("STEP P2 — VERIFY MIMIC-CXR FILES")
    print("=" * 70)
    print()

    all_ok = True

    # 1. Check reports directory
    print("1. Reports Directory")
    if not check_reports_dir(REPORTS_DIR):
        all_ok = False
    print()

    # 2. Check split CSV
    print("2. Split Assignments (mimic-cxr-2.0.0-split.csv)")
    if not check_file(SPLIT_CSV, "mimic-cxr-2.0.0-split.csv"):
        all_ok = False
    print()

    # 3. Check CheXpert CSV
    print("3. CheXpert Labels (mimic-cxr-2.0.0-chexpert.csv)")
    if not check_file(CHEXPERT_CSV, "mimic-cxr-2.0.0-chexpert.csv"):
        all_ok = False
    print()

    # 4. Check metadata CSV
    print("4. Image Metadata (mimic-cxr-2.0.0-metadata.csv)")
    if not check_file(METADATA_CSV, "mimic-cxr-2.0.0-metadata.csv"):
        all_ok = False
    print()

    # Summary
    print("=" * 70)
    if all_ok:
        print("RESULT: All required MIMIC-CXR files are present and non-empty.")
        print("        You may proceed to Step P3.")
    else:
        print("RESULT: One or more required files are MISSING or EMPTY.")
        print("        Please download all required files before proceeding.")
        print()
        print("Required files:")
        print(f"  1. Reports folder:  {REPORTS_DIR}")
        print(f"  2. Split CSV:       {SPLIT_CSV}")
        print(f"  3. CheXpert CSV:    {CHEXPERT_CSV}")
        print(f"  4. Metadata CSV:    {METADATA_CSV}")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

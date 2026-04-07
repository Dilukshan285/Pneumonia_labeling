"""
Step P3 — Load All Reports into a Single DataFrame (FAST — multi-threaded)
Reads all free-text report files from the MIMIC-CXR reports folder and creates
a master DataFrame with columns: subject_id, study_id, report_text.
Saves as master_reports.csv.

Uses ThreadPoolExecutor for parallel file I/O across patient directories.
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import REPORTS_DIR, MASTER_REPORTS_CSV

# Number of parallel threads for file I/O
NUM_WORKERS = 16


def load_patient_reports(patient_path, subject_id):
    """
    Load all reports from a single patient directory.
    Returns list of dicts with subject_id, study_id, report_text.
    """
    records = []
    total = 0
    skipped = 0

    try:
        report_files = [f for f in os.listdir(patient_path) if f.endswith('.txt')]
    except OSError:
        return records, 0, 1

    for report_file in report_files:
        total += 1
        report_path = os.path.join(patient_path, report_file)
        study_id = os.path.splitext(report_file)[0]

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_text = f.read()
        except UnicodeDecodeError:
            try:
                with open(report_path, 'r', encoding='latin-1') as f:
                    report_text = f.read()
            except Exception:
                skipped += 1
                continue

        report_text = report_text.strip()
        if not report_text:
            skipped += 1
            continue

        records.append({
            'subject_id': str(subject_id),
            'study_id': str(study_id),
            'report_text': report_text
        })

    return records, total, skipped


def load_top_folder(top_path, top_folder):
    """
    Load all patient directories within a single top-level folder (e.g., p10/).
    Uses ThreadPoolExecutor for parallel I/O.
    """
    patient_dirs = sorted([
        d for d in os.listdir(top_path)
        if os.path.isdir(os.path.join(top_path, d)) and d.startswith('p')
    ])

    all_records = []
    total_files = 0
    skipped_files = 0

    # Build list of (patient_path, subject_id) tasks
    tasks = []
    for patient_dir in patient_dirs:
        patient_path = os.path.join(top_path, patient_dir)
        subject_id = patient_dir[1:]  # Remove leading 'p'
        tasks.append((patient_path, subject_id))

    # Process patients in parallel using threads
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(load_patient_reports, path, sid): sid
            for path, sid in tasks
        }

        with tqdm(total=len(tasks), desc=f"  {top_folder}", unit="patients") as pbar:
            for future in as_completed(futures):
                records, total, skipped = future.result()
                all_records.extend(records)
                total_files += total
                skipped_files += skipped
                pbar.update(1)

    return all_records, total_files, skipped_files, len(patient_dirs)


def main():
    print("=" * 70)
    print("STEP P3 — LOAD ALL REPORTS (MULTI-THREADED)")
    print(f"  Workers: {NUM_WORKERS}")
    print("=" * 70)
    print()

    if not os.path.exists(REPORTS_DIR):
        print(f"ERROR: Reports directory not found: {REPORTS_DIR}")
        print("       Run p2_verify_files.py first to check file locations.")
        return 1

    # Ensure output directory exists
    os.makedirs(os.path.dirname(MASTER_REPORTS_CSV), exist_ok=True)

    # Get all top-level subfolders (p10, p11, ..., p19)
    top_folders = sorted([
        d for d in os.listdir(REPORTS_DIR)
        if os.path.isdir(os.path.join(REPORTS_DIR, d)) and d.startswith('p')
    ])

    print(f"Found {len(top_folders)} top-level folders: {', '.join(top_folders)}")
    print()

    all_records = []
    grand_total_files = 0
    grand_skipped_files = 0

    for top_folder in top_folders:
        top_path = os.path.join(REPORTS_DIR, top_folder)
        records, total, skipped, n_patients = load_top_folder(top_path, top_folder)

        print(f"    {top_folder}: {n_patients:,} patients, {total:,} files, {len(records):,} loaded, {skipped:,} skipped")

        all_records.extend(records)
        grand_total_files += total
        grand_skipped_files += skipped

    print()

    # Create DataFrame
    print("Creating master DataFrame...")
    df = pd.DataFrame(all_records)

    # Ensure string types for IDs
    df['subject_id'] = df['subject_id'].astype(str)
    df['study_id'] = df['study_id'].astype(str)

    # Verify no duplicates on study_id
    n_dupes = df['study_id'].duplicated().sum()

    print()
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"  Total .txt files found:      {grand_total_files:,}")
    print(f"  Files skipped (empty/error):  {grand_skipped_files:,}")
    print(f"  Reports loaded:               {len(df):,}")
    print(f"  Unique subject_ids:           {df['subject_id'].nunique():,}")
    print(f"  Unique study_ids:             {df['study_id'].nunique():,}")
    print(f"  Duplicate study_ids:          {n_dupes:,}")
    print()

    # Report text statistics
    df['_text_len'] = df['report_text'].str.len()
    print(f"  Report text length (chars):")
    print(f"    Mean:     {df['_text_len'].mean():.0f}")
    print(f"    Median:   {df['_text_len'].median():.0f}")
    print(f"    Min:      {df['_text_len'].min()}")
    print(f"    Max:      {df['_text_len'].max()}")
    df.drop(columns=['_text_len'], inplace=True)
    print()

    # Save to CSV
    print(f"Saving to: {MASTER_REPORTS_CSV}")
    df.to_csv(MASTER_REPORTS_CSV, index=False)
    file_size_mb = os.path.getsize(MASTER_REPORTS_CSV) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")
    print()

    # Show first few rows
    print("First 5 rows (report_text truncated to 80 chars):")
    print("-" * 70)
    preview = df.head(5).copy()
    preview['report_text'] = preview['report_text'].str[:80] + '...'
    print(preview.to_string(index=False))

    print()
    print("=" * 70)
    print("STEP P3 COMPLETE — master_reports.csv saved successfully.")
    print(f"  Output: {MASTER_REPORTS_CSV}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

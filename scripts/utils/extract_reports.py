"""
Extract 500 sample pneumonia reports (250 positive, 250 negative) from MIMIC-CXR
based on the final pipeline labels. Copies report text files into a new
pneumonia_reports folder organized by label class.
"""

import pandas as pd
import os
import shutil

# === Paths ===
LABELS_CSV = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\final_pneumonia_labels.csv"
REPORTS_DIR = r"c:\Users\dviya\Downloads\mimic-cxr-reports\files"
OUTPUT_DIR = r"c:\Users\dviya\Desktop\Pneumonia_labeling\pneumonia_reports"

# === Load labels ===
print("Loading final_pneumonia_labels.csv ...")
df = pd.read_csv(LABELS_CSV)
print(f"Total labeled reports: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Convert subject_id and study_id to strings for path construction
df['subject_id'] = df['subject_id'].astype(str)
df['study_id'] = df['study_id'].astype(str)

# === Separate positive and negative ===
positives = df[df['label'] == 1].copy()
negatives = df[df['label'] == 0].copy()
print(f"\nPositive reports available: {len(positives)}")
print(f"Negative reports available: {len(negatives)}")

# === Sample 250 of each (or all if fewer than 250) ===
n_pos = min(250, len(positives))
n_neg = min(250, len(negatives))
pos_sample = positives.sample(n=n_pos, random_state=42)
neg_sample = negatives.sample(n=n_neg, random_state=42)

print(f"\nSampled {n_pos} positive and {n_neg} negative reports")

# === Create output directories ===
pos_dir = os.path.join(OUTPUT_DIR, "positive")
neg_dir = os.path.join(OUTPUT_DIR, "negative")
os.makedirs(pos_dir, exist_ok=True)
os.makedirs(neg_dir, exist_ok=True)

def find_report_file(subject_id, study_id):
    """
    Build the path to a report file.
    MIMIC-CXR reports are organized as:
      files/p{first2digits}/p{subject_id}/s{study_id}.txt
    
    The top-level folders are p10, p11, ..., p19 based on the first 2 digits
    after 'p' in the subject_id (which is 8 digits like 10001217).
    """
    # subject_id is like "10001217" -> first 2 digits = "10" -> folder = "p10"
    prefix = "p" + subject_id[:2]
    report_path = os.path.join(
        REPORTS_DIR, prefix, f"p{subject_id}", f"{study_id}.txt"
    )
    return report_path

def copy_reports(sample_df, dest_dir, label_name):
    """Copy report files to destination directory with informative names."""
    copied = 0
    missing = 0
    
    for _, row in sample_df.iterrows():
        sid = row['subject_id']
        stid = row['study_id']
        soft = row['soft_score']
        src = row.get('label_source', 'unknown')
        
        report_path = find_report_file(sid, stid)
        
        if os.path.exists(report_path):
            # Name format: {label}_{subject_id}_{study_id}.txt
            dest_name = f"{label_name}_{sid}_{stid}.txt"
            dest_path = os.path.join(dest_dir, dest_name)
            
            # Read original report and prepend metadata header
            with open(report_path, 'r', encoding='utf-8', errors='replace') as f:
                report_text = f.read()
            
            header = (
                f"{'='*70}\n"
                f"PNEUMONIA LABEL: {label_name.upper()}\n"
                f"Subject ID: {sid}\n"
                f"Study ID: {stid}\n"
                f"Soft Score: {soft:.6f}\n"
                f"Label Source: {src}\n"
                f"{'='*70}\n\n"
            )
            
            with open(dest_path, 'w', encoding='utf-8') as f:
                f.write(header + report_text)
            
            copied += 1
        else:
            missing += 1
    
    return copied, missing

# === Copy positive reports ===
print(f"\nCopying positive reports to {pos_dir} ...")
pos_copied, pos_missing = copy_reports(pos_sample, pos_dir, "positive")
print(f"  Copied: {pos_copied}, Missing: {pos_missing}")

# === Copy negative reports ===
print(f"Copying negative reports to {neg_dir} ...")
neg_copied, neg_missing = copy_reports(neg_sample, neg_dir, "negative")
print(f"  Copied: {neg_copied}, Missing: {neg_missing}")

# === Save a summary CSV ===
combined = pd.concat([pos_sample, neg_sample])
combined['label_text'] = combined['label'].map({1: 'POSITIVE', 0: 'NEGATIVE'})
summary_path = os.path.join(OUTPUT_DIR, "extracted_reports_summary.csv")
combined.to_csv(summary_path, index=False)
print(f"\nSummary CSV saved to: {summary_path}")

print(f"\n{'='*70}")
print(f"EXTRACTION COMPLETE")
print(f"{'='*70}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"  positive/  -> {pos_copied} reports")
print(f"  negative/  -> {neg_copied} reports")
print(f"  Total:        {pos_copied + neg_copied} reports")
print(f"{'='*70}")

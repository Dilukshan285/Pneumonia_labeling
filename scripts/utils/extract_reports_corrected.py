"""
================================================================================
STEP 3 — RE-EXTRACT 500 SAMPLE REPORTS FROM CORRECTED LABELS
================================================================================
Pulls 250 positive + 250 negative from final_pneumonia_labels_corrected.csv
and saves to pneumonia_reports_corrected/ with correction metadata.
================================================================================
"""

import pandas as pd
import os
import time

start = time.time()

# === Paths ===
CORRECTED_CSV = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\final_pneumonia_labels_corrected.csv"
REFERENCE_CSV = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\labels_correction_reference.csv"
REPORTS_DIR   = r"c:\Users\dviya\Downloads\mimic-cxr-reports\files"
OUTPUT_DIR    = r"c:\Users\dviya\Desktop\Pneumonia_labeling\pneumonia_reports_corrected"

print("=" * 70)
print("STEP 3 — EXTRACTING SAMPLE REPORTS FROM CORRECTED LABELS")
print("=" * 70)

# === Load corrected labels ===
df = pd.read_csv(CORRECTED_CSV)
df['subject_id'] = df['subject_id'].astype(str)
df['study_id'] = df['study_id'].astype(str)
print(f"Corrected labels loaded: {len(df):,}")

# === Load reference for correction metadata ===
df_ref = pd.read_csv(REFERENCE_CSV)
df_ref['subject_id'] = df_ref['subject_id'].astype(str)
df_ref['study_id'] = df_ref['study_id'].astype(str)

# === Sample ===
positives = df[df['label'] == 1]
negatives = df[df['label'] == 0]

n_pos = min(250, len(positives))
n_neg = min(250, len(negatives))
pos_sample = positives.sample(n=n_pos, random_state=42)
neg_sample = negatives.sample(n=n_neg, random_state=42)
print(f"Sampled {n_pos} positive + {n_neg} negative")

# === Create output dirs ===
pos_dir = os.path.join(OUTPUT_DIR, "positive")
neg_dir = os.path.join(OUTPUT_DIR, "negative")
os.makedirs(pos_dir, exist_ok=True)
os.makedirs(neg_dir, exist_ok=True)

def get_report_path(subject_id, study_id):
    prefix = "p" + subject_id[:2]
    return os.path.join(REPORTS_DIR, prefix, f"p{subject_id}", f"{study_id}.txt")

def copy_reports(sample_df, dest_dir, label_name):
    copied = 0
    for _, row in sample_df.iterrows():
        sid = row['subject_id']
        stid = row['study_id']
        
        report_path = get_report_path(sid, stid)
        if not os.path.exists(report_path):
            continue
        
        with open(report_path, 'r', encoding='utf-8', errors='replace') as f:
            report_text = f.read()
        
        # Get correction metadata
        ref_row = df_ref[(df_ref['subject_id'] == sid) & (df_ref['study_id'] == stid)]
        original_label = ""
        audit_label = ""
        mismatch_type = ""
        audit_reason = ""
        if len(ref_row) > 0:
            ref = ref_row.iloc[0]
            original_label = f"{'POSITIVE' if ref['label']==1 else 'NEGATIVE'}"
            audit_label = str(ref.get('audit_label', ''))
            mismatch_type = str(ref.get('mismatch_type', ''))
            audit_reason = str(ref.get('audit_reason', ''))
        
        was_corrected = "YES" if row['label_source'] == 'audit_correction' else "NO"
        
        header = (
            f"{'='*70}\n"
            f"CORRECTED LABEL: {label_name.upper()}\n"
            f"Subject ID: {sid}\n"
            f"Study ID: {stid}\n"
            f"Soft Score: {row['soft_score']:.6f}\n"
            f"Label Source: {row['label_source']}\n"
            f"Was Corrected: {was_corrected}\n"
            f"Original Pipeline Label: {original_label}\n"
            f"Audit Classification: {audit_label}\n"
            f"Mismatch Type: {mismatch_type}\n"
            f"Audit Reason: {audit_reason}\n"
            f"{'='*70}\n\n"
        )
        
        dest_name = f"{label_name}_{sid}_{stid}.txt"
        dest_path = os.path.join(dest_dir, dest_name)
        
        with open(dest_path, 'w', encoding='utf-8') as f:
            f.write(header + report_text)
        
        copied += 1
    
    return copied

print(f"\nCopying positive reports ...")
pos_copied = copy_reports(pos_sample, pos_dir, "positive")
print(f"  Copied: {pos_copied}")

print(f"Copying negative reports ...")
neg_copied = copy_reports(neg_sample, neg_dir, "negative")
print(f"  Copied: {neg_copied}")

# Save summary
combined = pd.concat([pos_sample, neg_sample])
combined['label_text'] = combined['label'].map({1: 'POSITIVE', 0: 'NEGATIVE'})
summary_path = os.path.join(OUTPUT_DIR, "extracted_reports_summary.csv")
combined.to_csv(summary_path, index=False)

elapsed = time.time() - start
print(f"\n{'='*70}")
print(f"EXTRACTION COMPLETE")
print(f"{'='*70}")
print(f"  positive/  → {pos_copied} reports")
print(f"  negative/  → {neg_copied} reports")
print(f"  Total:       {pos_copied + neg_copied} reports")
print(f"  Summary:     {summary_path}")
print(f"  Time:        {elapsed:.1f}s")
print(f"{'='*70}")

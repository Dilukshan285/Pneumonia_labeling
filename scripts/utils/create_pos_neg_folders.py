"""
Create final_pos_neg folder with 250 positive and 250 negative report txt files
based on final_training_labels.csv (sorted by highest/lowest soft_score for best quality).
"""
import csv
import os
import shutil

# Paths
CSV_PATH = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\final_training_labels.csv"
REPORTS_ROOT = r"c:\Users\dviya\Downloads\mimic-cxr-reports\files"
OUTPUT_DIR = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\final_pos_neg"

MAX_POS = 250
MAX_NEG = 250

# Read all rows
positives = []
negatives = []

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = int(row["label"].strip())
        soft_score = float(row["soft_score"].strip())
        if label == 1:
            positives.append((row["subject_id"].strip(), row["study_id"].strip(), soft_score))
        else:
            negatives.append((row["subject_id"].strip(), row["study_id"].strip(), soft_score))

# Sort: positives by highest soft_score, negatives by lowest soft_score (most confident)
positives.sort(key=lambda x: x[2], reverse=True)
negatives.sort(key=lambda x: x[2])

print(f"Total positives in CSV: {len(positives)}")
print(f"Total negatives in CSV: {len(negatives)}")
print(f"Selecting top {MAX_POS} positive and top {MAX_NEG} negative by confidence...\n")

# Create output directories
pos_dir = os.path.join(OUTPUT_DIR, "positive")
neg_dir = os.path.join(OUTPUT_DIR, "negative")
os.makedirs(pos_dir, exist_ok=True)
os.makedirs(neg_dir, exist_ok=True)

def copy_reports(entries, dest_dir, max_count, label_name):
    copied = 0
    not_found = 0
    for subject_id, study_id, score in entries:
        if copied >= max_count:
            break
        prefix = f"p{subject_id[:2]}"
        patient_folder = f"p{subject_id}"
        report_filename = f"{study_id}.txt"
        src_path = os.path.join(REPORTS_ROOT, prefix, patient_folder, report_filename)

        if not os.path.isfile(src_path):
            not_found += 1
            continue

        dest_filename = f"{subject_id}_{study_id}.txt"
        shutil.copy2(src_path, os.path.join(dest_dir, dest_filename))
        copied += 1

    print(f"  {label_name}: copied {copied}, skipped {not_found} (not found)")
    return copied

copied_pos = copy_reports(positives, pos_dir, MAX_POS, "Positive")
copied_neg = copy_reports(negatives, neg_dir, MAX_NEG, "Negative")

print(f"\n{'='*50}")
print(f"DONE!")
print(f"{'='*50}")
print(f"Output: {OUTPUT_DIR}")
print(f"  positive/ → {copied_pos} reports")
print(f"  negative/ → {copied_neg} reports")

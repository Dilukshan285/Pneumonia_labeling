import os
import shutil
import pandas as pd

# Paths
REPORTS_DIR = r"c:\Users\dviya\Downloads\mimic-cxr-reports\files"
LABELS_CSV = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\training_ready_labels.csv"
OUTPUT_DIR = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\advace_pne"

def main():
    print("Loading labels from training_ready_labels.csv...")
    df = pd.read_csv(LABELS_CSV)

    # Apply the requested filters (focusing on the newly discussed thresholds)
    # Let's extract the "Hard" boundary cases specifically so you can see them!
    pos_df = df[(df['label'] == 1) & (df['soft_score'] >= 0.75) & (df['soft_score'] < 0.95)]
    neg_df = df[(df['label'] == 0) & (df['soft_score'] > 0.05) & (df['soft_score'] <= 0.25)]

    print(f"Found {len(pos_df):,} 'Hard Positive' reports (0.75 to 0.95)")
    print(f"Found {len(neg_df):,} 'Hard Negative' reports (0.05 to 0.25)")

    # Sample size for manual verification
    # We sample 100 of each to avoid copying 190,000 files which would freeze your computer
    SAMPLE_SIZE = 100
    print(f"\nSampling {SAMPLE_SIZE} reports from each class for manual verification...")
    
    pos_sample = pos_df.sample(min(SAMPLE_SIZE, len(pos_df)), random_state=42)
    neg_sample = neg_df.sample(min(SAMPLE_SIZE, len(neg_df)), random_state=42)

    # Create directories
    pos_dir = os.path.join(OUTPUT_DIR, "hard_positive")
    neg_dir = os.path.join(OUTPUT_DIR, "hard_negative")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    # Function to construct the original report path
    def get_report_path(study_id, subject_id):
        subject_str = str(subject_id)
        # e.g. p10 folder for subject 10000032
        p_folder = f"p{subject_str[:2]}" 
        # original txt files are named like s50414267.txt
        return os.path.join(REPORTS_DIR, p_folder, f"p{subject_str}", f"{study_id}.txt")

    # Copy Positive Reports
    print("\nCopying POSITIVE reports...")
    pos_copied = 0
    for _, row in pos_sample.iterrows():
        src = get_report_path(row['study_id'], row['subject_id'])
        if os.path.exists(src):
            # Include the soft_score in the filename so you can see it easily
            dst = os.path.join(pos_dir, f"{row['study_id']}_score_{row['soft_score']:.4f}.txt")
            shutil.copy2(src, dst)
            pos_copied += 1
        else:
            print(f"Warning: Could not find report text for {row['study_id']}")

    print(f"Successfully copied {pos_copied} POSITIVE reports to {pos_dir}")

    # Copy Negative Reports
    print("\nCopying NEGATIVE reports...")
    neg_copied = 0
    for _, row in neg_sample.iterrows():
        src = get_report_path(row['study_id'], row['subject_id'])
        if os.path.exists(src):
            dst = os.path.join(neg_dir, f"{row['study_id']}_score_{row['soft_score']:.4f}.txt")
            shutil.copy2(src, dst)
            neg_copied += 1
        else:
            print(f"Warning: Could not find report text for {row['study_id']}")
            
    print(f"Successfully copied {neg_copied} NEGATIVE reports to {neg_dir}")
    print("\nExtraction complete! You can now manually check the reports.")

if __name__ == "__main__":
    main()

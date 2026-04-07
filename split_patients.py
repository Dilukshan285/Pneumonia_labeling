import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

input_path = r'c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\pp1_balanced_dataset_20k.csv'

print(f"Loading balanced dataset: {input_path}")
df = pd.read_csv(input_path)

# We use GroupShuffleSplit to strictly isolate patients (subject_id)
# Step 1: Split 80% Train, 20% Temp
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, temp_idx = next(gss1.split(df, groups=df['subject_id']))

train_df = df.iloc[train_idx]
temp_df = df.iloc[temp_idx]

# Step 2: Split the 20% Temp into 10% Validation, 10% Testing
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df['subject_id']))

val_df = temp_df.iloc[val_idx]
test_df = temp_df.iloc[test_idx]

# --- VERIFICATION STEP (The Firewall) ---
train_patients = set(train_df['subject_id'])
val_patients = set(val_df['subject_id'])
test_patients = set(test_df['subject_id'])

# Asserts will hard-crash the script if ANY patient crosses boundaries
assert len(train_patients.intersection(val_patients)) == 0, "ERROR: Patient leakage between Train and Val!"
assert len(train_patients.intersection(test_patients)) == 0, "ERROR: Patient leakage between Train and Test!"
assert len(val_patients.intersection(test_patients)) == 0, "ERROR: Patient leakage between Val and Test!"

def print_stats(name, d):
    pos = len(d[d['label'] == 1])
    neg = len(d[d['label'] == 0])
    ratio = pos / (pos + neg)
    print(f"{name:5s} | Total Imgs: {len(d):5d} | Unique Patients: {d['subject_id'].nunique():4d} | POS: {pos:4d} | NEG: {neg:4d} | Pos Ratio: {ratio*100:.1f}%")

print("="*80)
print("PATIENT ISOLATION SPLIT RESULTS (0% Leakage Verified)")
print("-" * 80)
print_stats("TRAIN", train_df)
print_stats("VAL", val_df)
print_stats("TEST", test_df)
print("="*80)

# Save the final split files
train_df.to_csv(r'c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\pp1_train.csv', index=False)
val_df.to_csv(r'c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\pp1_val.csv', index=False)
test_df.to_csv(r'c:\Users\dviya\Desktop\Pneumonia_labeling\data\output\pp1_test.csv', index=False)

print("\nSaved files:")
print(" - data/output/pp1_train.csv")
print(" - data/output/pp1_val.csv")
print(" - data/output/pp1_test.csv")

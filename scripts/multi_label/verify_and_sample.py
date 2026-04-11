"""
Rule Verification + Sample Report Reader
"""
import pandas as pd

base = "data/output/multi_label_dataset"
train = pd.read_csv(f"{base}/ml_train.csv", low_memory=False)
val = pd.read_csv(f"{base}/ml_val.csv", low_memory=False)
test = pd.read_csv(f"{base}/ml_test.csv", low_memory=False)
combined = pd.concat([train, val, test], ignore_index=True)

print("=" * 70)
print("  RULE VERIFICATION")
print("=" * 70)

# RULE 1: patient_id — all rows with same patient must go to same split
print("\n  RULE 1: All rows with same patient_id in same split")
train_p = set(train["subject_id"])
val_p = set(val["subject_id"])
test_p = set(test["subject_id"])
tv = len(train_p & val_p)
tt = len(train_p & test_p)
vt = len(val_p & test_p)
r1 = tv == 0 and tt == 0 and vt == 0
print(f"    Train<->Val overlap:  {tv}")
print(f"    Train<->Test overlap: {tt}")
print(f"    Val<->Test overlap:   {vt}")
print(f"    RESULT: {'PASS' if r1 else 'FAIL'}")

# RULE 2: image_path (dicom_id) — must be unique across entire dataset
print("\n  RULE 2: dicom_id unique across entire dataset")
dup_img = combined["dicom_id"].duplicated().sum()
r2 = dup_img == 0
print(f"    Total images: {len(combined)}")
print(f"    Unique images: {combined['dicom_id'].nunique()}")
print(f"    Duplicates: {dup_img}")
print(f"    RESULT: {'PASS' if r2 else 'FAIL'}")

# RULE 3: report_id (study_id) — must be unique across entire dataset
print("\n  RULE 3: study_id unique across entire dataset")
dup_study = combined["study_id"].duplicated().sum()
r3 = dup_study == 0
print(f"    Total reports: {len(combined)}")
print(f"    Unique reports: {combined['study_id'].nunique()}")
print(f"    Duplicates: {dup_study}")
print(f"    RESULT: {'PASS' if r3 else 'FAIL'}")

# RULE 4: findings + impression — one text per image, never shared
print("\n  RULE 4: Report text coverage — every row has text")
has_either = (
    (combined["findings_text"].fillna("").str.strip() != "") |
    (combined["impression_text"].fillna("").str.strip() != "")
)
empty = (~has_either).sum()
r4 = empty == 0
print(f"    Rows with report text: {has_either.sum()}")
print(f"    Empty rows: {empty}")
print(f"    RESULT: {'PASS' if r4 else 'FAIL'}")

# RULE 5: Pneumonia labels — only 1 and 0 (no uncertain)
print("\n  RULE 5: Pneumonia labels — no uncertain (-1)")
pneu_unc = (combined["Pneumonia"] == -1).sum()
r5 = pneu_unc == 0
print(f"    Pneumonia = 1:  {(combined['Pneumonia']==1).sum()}")
print(f"    Pneumonia = 0:  {(combined['Pneumonia']==0).sum()}")
print(f"    Pneumonia = -1: {pneu_unc}")
print(f"    RESULT: {'PASS' if r5 else 'FAIL'}")

all_pass = r1 and r2 and r3 and r4 and r5
print(f"\n  {'='*50}")
print(f"  ALL 5 RULES: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
print(f"  {'='*50}")

# =====================================================================
# SAMPLE REPORTS
# =====================================================================
print("\n\n" + "=" * 70)
print("  10 PNEUMONIA POSITIVE REPORTS (from ml_train.csv)")
print("=" * 70)

pos = train[train["Pneumonia"] == 1].sample(10, random_state=42)
for i, (_, row) in enumerate(pos.iterrows()):
    imp = str(row["impression_text"])[:300] if pd.notna(row["impression_text"]) else "[NO IMPRESSION]"
    find = str(row["findings_text"])[:300] if pd.notna(row["findings_text"]) else "[NO FINDINGS]"
    diseases = [d for d in ["Atelectasis","Cardiomegaly","Consolidation","Edema","Lung_Opacity","No_Finding","Pleural_Effusion","Pneumothorax"] if row[d] == 1]
    print(f"\n--- POSITIVE #{i+1} (dicom={row['dicom_id']}) ---")
    print(f"  IMPRESSION: {imp}")
    print(f"  FINDINGS:   {find}")
    print(f"  OTHER DISEASES: {', '.join(diseases) if diseases else 'None'}")

print("\n\n" + "=" * 70)
print("  10 PNEUMONIA NEGATIVE REPORTS (from ml_train.csv)")
print("=" * 70)

neg = train[train["Pneumonia"] == 0].sample(10, random_state=42)
for i, (_, row) in enumerate(neg.iterrows()):
    imp = str(row["impression_text"])[:300] if pd.notna(row["impression_text"]) else "[NO IMPRESSION]"
    find = str(row["findings_text"])[:300] if pd.notna(row["findings_text"]) else "[NO FINDINGS]"
    diseases = [d for d in ["Atelectasis","Cardiomegaly","Consolidation","Edema","Lung_Opacity","No_Finding","Pleural_Effusion","Pneumothorax"] if row[d] == 1]
    print(f"\n--- NEGATIVE #{i+1} (dicom={row['dicom_id']}) ---")
    print(f"  IMPRESSION: {imp}")
    print(f"  FINDINGS:   {find}")
    print(f"  OTHER DISEASES: {', '.join(diseases) if diseases else 'None'}")

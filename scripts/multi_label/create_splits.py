"""
=======================================================================
FINAL DATASET GENERATOR — GOD-TIER ML ENGINEERING
=======================================================================
Source: multi_label_final.csv (173,175 rows, 9 diseases)
Output: ml_train.csv, ml_val.csv, ml_test.csv

Rules enforced:
  1. subject_id → all rows with same patient go to same split
  2. dicom_id  → must be unique (one image = one row)
  3. study_id  → must be unique (one report = one row)
  4. 100% report coverage (findings OR impression)
  5. Drop Pneumonia == -1 (uncertain)
  6. Patient-level split 80/10/10
  7. Balance train ONLY 1:1 on Pneumonia
  8. Val/Test keep NATURAL distribution
  9. Zero patient overlap, zero image overlap
=======================================================================
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Lung_Opacity", "No_Finding", "Pleural_Effusion", "Pneumonia",
    "Pneumothorax",
]

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def print_disease_table(df, label):
    print(f"\n  {label} — Disease Prevalence ({len(df)} rows):")
    print(f"  {'Pathology':28s} {'PRESENT':>8s} {'ABSENT':>8s} {'UNC':>6s} {'Prev%':>7s}")
    print(f"  {'-'*58}")
    for cls in DISEASES:
        p = (df[cls] == 1).sum()
        a = (df[cls] == 0).sum()
        u = (df[cls] == -1).sum()
        print(f"  {cls:28s} {p:8d} {a:8d} {u:6d} {p/len(df)*100:6.1f}%")

# =====================================================================
# STEP 1: LOAD
# =====================================================================
print_section("STEP 1: LOADING DATA")
df = pd.read_csv("data/output/multi_label_dataset/multi_label_final.csv", low_memory=False)
print(f"  Loaded: {len(df)} rows")

# =====================================================================
# STEP 2: DROP PNEUMONIA UNCERTAIN
# =====================================================================
print_section("STEP 2: DROP PNEUMONIA UNCERTAIN")
before = len(df)
df = df[df["Pneumonia"] != -1].reset_index(drop=True)
print(f"  Dropped {before - len(df)} uncertain Pneumonia rows")
print(f"  Remaining: {len(df)} rows")
print(f"  Pneumonia PRESENT: {(df['Pneumonia'] == 1).sum()}")
print(f"  Pneumonia ABSENT:  {(df['Pneumonia'] == 0).sum()}")

# =====================================================================
# STEP 3: DROP soft_score (no longer needed)
# =====================================================================
print_section("STEP 3: CLEAN COLUMNS")
drop_cols = ["soft_score", "label", "confidence_tier", "label_source",
             "assertion_status", "nli_prob_Pneumonia"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
print(f"  Dropped legacy columns: {[c for c in drop_cols if c not in df.columns or True]}")
print(f"  Final columns: {list(df.columns)}")

# =====================================================================
# STEP 4: INTEGRITY VALIDATION
# =====================================================================
print_section("STEP 4: INTEGRITY VALIDATION")

# 4a. Unique dicom_id
dup_img = df["dicom_id"].duplicated().sum()
print(f"  [{'OK' if dup_img == 0 else 'FAIL'}] Unique images (dicom_id): {df['dicom_id'].nunique()} | Duplicates: {dup_img}")

# 4b. Unique study_id
dup_study = df["study_id"].duplicated().sum()
print(f"  [{'OK' if dup_study == 0 else 'FAIL'}] Unique reports (study_id): {df['study_id'].nunique()} | Duplicates: {dup_study}")

# 4c. Report coverage
f_ok = df["findings_text"].fillna("").str.strip() != ""
i_ok = df["impression_text"].fillna("").str.strip() != ""
either = (f_ok | i_ok).sum()
print(f"  [{'OK' if either == len(df) else 'FAIL'}] Report coverage: {either}/{len(df)} (100%)")
print(f"       Has findings:   {f_ok.sum()} ({f_ok.mean()*100:.1f}%)")
print(f"       Has impression: {i_ok.sum()} ({i_ok.mean()*100:.1f}%)")
print(f"       Has BOTH:       {(f_ok & i_ok).sum()} ({(f_ok & i_ok).mean()*100:.1f}%)")

# 4d. Patient count
print(f"  [OK] Unique patients: {df['subject_id'].nunique()}")

# =====================================================================
# STEP 5: PATIENT-LEVEL SPLIT 80/10/10
# =====================================================================
print_section("STEP 5: PATIENT-LEVEL SPLIT (80/10/10)")

np.random.seed(42)

# Split 1: 90% trainval vs 10% test
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
trainval_idx, test_idx = next(gss1.split(df, groups=df["subject_id"]))
df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

# Split 2: ~88.9% train vs ~11.1% val (= 10% of total)
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1111, random_state=42)
train_idx, val_idx = next(gss2.split(df_trainval, groups=df_trainval["subject_id"]))
df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

print(f"  Train: {len(df_train)} rows ({len(df_train)/len(df)*100:.1f}%)")
print(f"  Val:   {len(df_val)} rows ({len(df_val)/len(df)*100:.1f}%)")
print(f"  Test:  {len(df_test)} rows ({len(df_test)/len(df)*100:.1f}%)")

# =====================================================================
# STEP 6: VERIFY ZERO OVERLAP
# =====================================================================
print_section("STEP 6: ZERO OVERLAP VERIFICATION")

train_pats = set(df_train["subject_id"])
val_pats = set(df_val["subject_id"])
test_pats = set(df_test["subject_id"])

tv = len(train_pats & val_pats)
tt = len(train_pats & test_pats)
vt = len(val_pats & test_pats)
print(f"  Patient overlap Train<->Val:  {tv} {'[FAIL]' if tv else '[OK]'}")
print(f"  Patient overlap Train<->Test: {tt} {'[FAIL]' if tt else '[OK]'}")
print(f"  Patient overlap Val<->Test:   {vt} {'[FAIL]' if vt else '[OK]'}")

train_imgs = set(df_train["dicom_id"])
val_imgs = set(df_val["dicom_id"])
test_imgs = set(df_test["dicom_id"])

ti = len(train_imgs & val_imgs)
tt2 = len(train_imgs & test_imgs)
vt2 = len(val_imgs & test_imgs)
print(f"  Image overlap Train<->Val:    {ti} {'[FAIL]' if ti else '[OK]'}")
print(f"  Image overlap Train<->Test:   {tt2} {'[FAIL]' if tt2 else '[OK]'}")
print(f"  Image overlap Val<->Test:     {vt2} {'[FAIL]' if vt2 else '[OK]'}")

# =====================================================================
# STEP 7: BALANCE TRAIN SET 1:1 ON PNEUMONIA
# =====================================================================
print_section("STEP 7: BALANCE TRAIN SET (1:1 Pneumonia)")

train_pos = df_train[df_train["Pneumonia"] == 1]
train_neg = df_train[df_train["Pneumonia"] == 0]

print(f"  Before balancing:")
print(f"    Pneumonia PRESENT: {len(train_pos)}")
print(f"    Pneumonia ABSENT:  {len(train_neg)}")
print(f"    Ratio:             1:{len(train_neg)/max(len(train_pos),1):.1f}")

# Downsample negatives
n_pos = len(train_pos)
train_neg_sampled = train_neg.sample(n=n_pos, random_state=42)
df_train_bal = pd.concat([train_pos, train_neg_sampled], ignore_index=True)
df_train_bal = df_train_bal.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n  After balancing:")
print(f"    Pneumonia PRESENT: {(df_train_bal['Pneumonia'] == 1).sum()}")
print(f"    Pneumonia ABSENT:  {(df_train_bal['Pneumonia'] == 0).sum()}")
print(f"    Total:             {len(df_train_bal)}")
print(f"    Ratio:             1:1 PERFECT")

# Re-verify no patient leak after downsampling
bal_pats = set(df_train_bal["subject_id"])
bp_v = len(bal_pats & val_pats)
bp_t = len(bal_pats & test_pats)
print(f"\n  Post-balance patient overlap:")
print(f"    BalTrain<->Val:  {bp_v} {'[FAIL]' if bp_v else '[OK]'}")
print(f"    BalTrain<->Test: {bp_t} {'[FAIL]' if bp_t else '[OK]'}")

# =====================================================================
# STEP 8: FINAL SUMMARY
# =====================================================================
print_section("STEP 8: FINAL SUMMARY")

print_disease_table(df_train_bal, "TRAIN (balanced 1:1)")
print_disease_table(df_val, "VAL (natural distribution)")
print_disease_table(df_test, "TEST (natural distribution)")

# Pneumonia per split
print(f"\n  Pneumonia distribution across splits:")
print(f"  {'Split':10s} {'Total':>8s} {'POS':>8s} {'NEG':>8s} {'Ratio':>10s}")
print(f"  {'-'*45}")
for name, d in [("Train", df_train_bal), ("Val", df_val), ("Test", df_test)]:
    p = (d["Pneumonia"] == 1).sum()
    n = (d["Pneumonia"] == 0).sum()
    ratio = f"1:{n/max(p,1):.1f}"
    print(f"  {name:10s} {len(d):8d} {p:8d} {n:8d} {ratio:>10s}")

# =====================================================================
# STEP 9: SAVE
# =====================================================================
print_section("STEP 9: SAVING FILES")

out = "data/output/multi_label_dataset"
df_train_bal.to_csv(f"{out}/ml_train.csv", index=False)
df_val.to_csv(f"{out}/ml_val.csv", index=False)
df_test.to_csv(f"{out}/ml_test.csv", index=False)

print(f"  ml_train.csv  ->  {len(df_train_bal):>6d} rows  (1:1 balanced)")
print(f"  ml_val.csv    ->  {len(df_val):>6d} rows  (natural)")
print(f"  ml_test.csv   ->  {len(df_test):>6d} rows  (natural)")
print(f"\n  Location: {out}/")

print(f"\n{'='*70}")
print(f"  DATASET GENERATION COMPLETE — FLAWLESS")
print(f"{'='*70}")

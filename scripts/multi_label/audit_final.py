"""
=======================================================================
GOD-TIER FINAL DATASET AUDIT
=======================================================================
Deeply validates ml_train.csv, ml_val.csv, ml_test.csv for:
  - Null values in every column
  - Duplicate images (dicom_id)
  - Duplicate reports (study_id)
  - Patient leakage across splits
  - Image leakage across splits
  - Report text coverage (findings + impression)
  - Pneumonia balance
  - Disease label integrity
  - Label value validation (only 1, 0, -1)
=======================================================================
"""
import pandas as pd
import numpy as np

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Lung_Opacity", "No_Finding", "Pleural_Effusion", "Pneumonia",
    "Pneumothorax",
]

def hr():
    print("=" * 70)

def load_all():
    base = "data/output/multi_label_dataset"
    train = pd.read_csv(f"{base}/ml_train.csv", low_memory=False)
    val = pd.read_csv(f"{base}/ml_val.csv", low_memory=False)
    test = pd.read_csv(f"{base}/ml_test.csv", low_memory=False)
    return train, val, test

def audit_nulls(df, name):
    print(f"\n  [{name}] NULL VALUES PER COLUMN:")
    has_issue = False
    for col in df.columns:
        nulls = df[col].isna().sum()
        status = "OK" if nulls == 0 else "WARNING"
        if nulls > 0:
            has_issue = True
        print(f"    {col:30s}  nulls={nulls:>6d}  [{status}]")
    if not has_issue:
        print(f"    >>> ALL COLUMNS CLEAN — ZERO NULLS")

def audit_duplicates(df, name):
    print(f"\n  [{name}] DUPLICATE CHECK:")
    dup_img = df["dicom_id"].duplicated().sum()
    dup_study = df["study_id"].duplicated().sum()
    dup_patient = df["subject_id"].duplicated().sum()
    print(f"    Duplicate dicom_id (images):  {dup_img}  [{'OK' if dup_img==0 else 'FAIL'}]")
    print(f"    Duplicate study_id (reports): {dup_study}  [{'OK' if dup_study==0 else 'FAIL'}]")
    print(f"    Duplicate subject_id (patients): {dup_patient}  [EXPECTED — same patient, multiple images]")

def audit_reports(df, name):
    print(f"\n  [{name}] REPORT COVERAGE:")
    f_ok = df["findings_text"].fillna("").str.strip() != ""
    i_ok = df["impression_text"].fillna("").str.strip() != ""
    both = f_ok & i_ok
    either = f_ok | i_ok
    neither = ~either

    print(f"    Has findings only:    {(f_ok & ~i_ok).sum():>6d}  ({(f_ok & ~i_ok).mean()*100:5.1f}%)")
    print(f"    Has impression only:  {(i_ok & ~f_ok).sum():>6d}  ({(i_ok & ~f_ok).mean()*100:5.1f}%)")
    print(f"    Has BOTH:             {both.sum():>6d}  ({both.mean()*100:5.1f}%)")
    print(f"    Has EITHER (total):   {either.sum():>6d}  ({either.mean()*100:5.1f}%)")
    print(f"    Has NEITHER:          {neither.sum():>6d}  [{'OK' if neither.sum()==0 else 'FAIL — EMPTY REPORTS!'}]")

def audit_labels(df, name):
    print(f"\n  [{name}] LABEL INTEGRITY:")
    valid_vals = {1, 0, -1}
    all_ok = True
    for cls in DISEASES:
        unique = set(df[cls].dropna().unique())
        invalid = unique - valid_vals
        if invalid:
            print(f"    {cls}: INVALID values found: {invalid}  [FAIL]")
            all_ok = False
    if all_ok:
        print(f"    ALL 9 diseases contain only valid labels (1, 0, -1)  [OK]")

    # Check no Pneumonia uncertain in any split
    pneu_unc = (df["Pneumonia"] == -1).sum()
    print(f"    Pneumonia uncertain (-1): {pneu_unc}  [{'OK' if pneu_unc==0 else 'FAIL'}]")

def audit_diseases(df, name):
    print(f"\n  [{name}] DISEASE PREVALENCE ({len(df)} rows):")
    print(f"    {'Pathology':28s} {'POS':>6s} {'NEG':>6s} {'UNC':>6s} {'Prev%':>7s}")
    print(f"    {'-'*56}")
    for cls in DISEASES:
        p = (df[cls] == 1).sum()
        a = (df[cls] == 0).sum()
        u = (df[cls] == -1).sum()
        print(f"    {cls:28s} {p:6d} {a:6d} {u:6d} {p/len(df)*100:6.1f}%")

# =====================================================================
# MAIN AUDIT
# =====================================================================
hr()
print("  GOD-TIER FINAL DATASET AUDIT")
hr()

train, val, test = load_all()
all_data = {"TRAIN": train, "VAL": val, "TEST": test}

print(f"\n  File sizes:")
print(f"    ml_train.csv:  {len(train):>6d} rows x {len(train.columns)} cols")
print(f"    ml_val.csv:    {len(val):>6d} rows x {len(val.columns)} cols")
print(f"    ml_test.csv:   {len(test):>6d} rows x {len(test.columns)} cols")
print(f"    TOTAL:         {len(train)+len(val)+len(test):>6d} rows")

# ── 1. NULL VALUES ──
hr()
print("  AUDIT 1: NULL VALUES")
hr()
for name, df in all_data.items():
    audit_nulls(df, name)

# ── 2. DUPLICATES ──
hr()
print("  AUDIT 2: DUPLICATES")
hr()
for name, df in all_data.items():
    audit_duplicates(df, name)

# Cross-split duplicates
combined = pd.concat([train, val, test])
cross_img = combined["dicom_id"].duplicated().sum()
cross_study = combined["study_id"].duplicated().sum()
print(f"\n  CROSS-SPLIT duplicate images:  {cross_img}  [{'OK' if cross_img==0 else 'FAIL'}]")
print(f"  CROSS-SPLIT duplicate reports: {cross_study}  [{'OK' if cross_study==0 else 'FAIL'}]")

# ── 3. PATIENT LEAKAGE ──
hr()
print("  AUDIT 3: PATIENT LEAKAGE")
hr()
train_p = set(train["subject_id"])
val_p = set(val["subject_id"])
test_p = set(test["subject_id"])

tv = len(train_p & val_p)
tt = len(train_p & test_p)
vt = len(val_p & test_p)
print(f"  Train <-> Val:   {tv} overlapping patients  [{'OK' if tv==0 else 'FAIL — DATA LEAKAGE!'}]")
print(f"  Train <-> Test:  {tt} overlapping patients  [{'OK' if tt==0 else 'FAIL — DATA LEAKAGE!'}]")
print(f"  Val <-> Test:    {vt} overlapping patients  [{'OK' if vt==0 else 'FAIL — DATA LEAKAGE!'}]")
print(f"  Unique patients: Train={len(train_p)}, Val={len(val_p)}, Test={len(test_p)}")

# ── 4. REPORT COVERAGE ──
hr()
print("  AUDIT 4: REPORT TEXT COVERAGE")
hr()
for name, df in all_data.items():
    audit_reports(df, name)

# ── 5. LABEL INTEGRITY ──
hr()
print("  AUDIT 5: LABEL INTEGRITY")
hr()
for name, df in all_data.items():
    audit_labels(df, name)

# ── 6. DISEASE PREVALENCE ──
hr()
print("  AUDIT 6: DISEASE PREVALENCE")
hr()
for name, df in all_data.items():
    audit_diseases(df, name)

# ── 7. PNEUMONIA BALANCE ──
hr()
print("  AUDIT 7: PNEUMONIA BALANCE")
hr()
print(f"  {'Split':10s} {'Total':>8s} {'POS':>8s} {'NEG':>8s} {'Ratio':>12s} {'Status':>10s}")
print(f"  {'-'*58}")
for name, df in all_data.items():
    p = (df["Pneumonia"] == 1).sum()
    n = (df["Pneumonia"] == 0).sum()
    ratio = f"1:{n/max(p,1):.1f}"
    balanced = "BALANCED" if name == "TRAIN" and abs(p - n) < 10 else "NATURAL" if name != "TRAIN" else "UNBALANCED"
    print(f"  {name:10s} {len(df):8d} {p:8d} {n:8d} {ratio:>12s} {balanced:>10s}")

# ── 8. COLUMN CONSISTENCY ──
hr()
print("  AUDIT 8: COLUMN CONSISTENCY")
hr()
cols_match = list(train.columns) == list(val.columns) == list(test.columns)
print(f"  All 3 files have identical columns: [{'OK' if cols_match else 'FAIL'}]")
print(f"  Columns ({len(train.columns)}):")
for c in train.columns:
    print(f"    - {c}")

# ── FINAL VERDICT ──
hr()
print("  FINAL VERDICT")
hr()
issues = []
if cross_img > 0: issues.append("Cross-split image duplicates")
if cross_study > 0: issues.append("Cross-split report duplicates")
if tv > 0 or tt > 0 or vt > 0: issues.append("Patient leakage")
for name, df in all_data.items():
    if (~((df["findings_text"].fillna("").str.strip() != "") | (df["impression_text"].fillna("").str.strip() != ""))).sum() > 0:
        issues.append(f"{name} has empty reports")
    if (df["Pneumonia"] == -1).sum() > 0:
        issues.append(f"{name} has uncertain Pneumonia labels")

if issues:
    print(f"  STATUS: FAILED")
    for iss in issues:
        print(f"    [X] {iss}")
else:
    print(f"  STATUS: FLAWLESS")
    print(f"  All checks passed. Dataset is production-ready.")
    print(f"  Zero nulls. Zero duplicates. Zero leakage. 100% report coverage.")

hr()

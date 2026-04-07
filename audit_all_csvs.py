"""
COMPREHENSIVE AUDIT — Validate every CSV in data/output/
Checks:
  1. No missing impression_text or findings_text in split files
  2. No patient leakage across train/val/test
  3. Frontal-only views (AP/PA)
  4. Label balance (1:1 POS:NEG)
  5. No ambiguous assertions (POSSIBLE/HISTORICAL/CONDITIONAL)
  6. No soft_score in dead zone (0.25 < score < 0.75)
  7. Column integrity (all required columns present)
  8. No duplicate rows
  9. Study-level consistency (same study_id → same label)
  10. Cross-file consistency (split studies are subset of manifest)
"""
import pandas as pd
import numpy as np
import os

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

issues = []

def check(condition, msg):
    if condition:
        print(f"  {PASS} {msg}")
    else:
        print(f"  {FAIL} {msg}")
        issues.append(msg)

def warn(condition, msg):
    if condition:
        print(f"  {PASS} {msg}")
    else:
        print(f"  {WARN} {msg}")

print("=" * 70)
print("  COMPREHENSIVE CSV AUDIT")
print("=" * 70)

# ─── 1. Load all files ───
files = {}
for fname in ['pp1_train.csv', 'pp1_val.csv', 'pp1_test.csv',
              'final_image_training_manifest.csv', 'advanced_final_labels.csv',
              'training_ready_labels.csv']:
    path = f'data/output/{fname}'
    if os.path.exists(path):
        files[fname] = pd.read_csv(path)
        print(f"\n{'─'*70}")
        print(f"  📄 {fname}  ({len(files[fname]):,} rows)")
        print(f"{'─'*70}")
    else:
        print(f"\n  {FAIL} {fname} — FILE NOT FOUND")
        issues.append(f"{fname} not found")
        continue

    df = files[fname]

    # ── Column check ──
    required = ['study_id', 'subject_id', 'label', 'soft_score']
    missing_cols = [c for c in required if c not in df.columns]
    check(len(missing_cols) == 0, f"Required columns present (missing: {missing_cols})" if missing_cols else "Required columns present")

    # ── Duplicate rows ──
    if 'dicom_id' in df.columns:
        dups = df.duplicated(subset='dicom_id').sum()
        check(dups == 0, f"No duplicate dicom_ids (found: {dups})")

    # ── Label distribution ──
    pos = (df['label'] == 1).sum()
    neg = (df['label'] == 0).sum()
    other = len(df) - pos - neg
    ratio = neg / pos if pos > 0 else float('inf')
    print(f"  📊 POS: {pos:,}  |  NEG: {neg:,}  |  Other: {other}  |  Ratio 1:{ratio:.2f}")
    check(other == 0, "No labels outside {0, 1}")

    # ── Soft score ranges ──
    pos_scores = df[df['label'] == 1]['soft_score']
    neg_scores = df[df['label'] == 0]['soft_score']
    if len(pos_scores) > 0:
        pos_in_dead = ((pos_scores > 0.25) & (pos_scores < 0.75)).sum()
        check(pos_in_dead == 0, f"No POS in dead zone (0.25-0.75): {pos_in_dead} found")
        print(f"       POS soft_score range: [{pos_scores.min():.4f}, {pos_scores.max():.4f}]")
    if len(neg_scores) > 0:
        neg_in_dead = ((neg_scores > 0.25) & (neg_scores < 0.75)).sum()
        check(neg_in_dead == 0, f"No NEG in dead zone (0.25-0.75): {neg_in_dead} found")
        print(f"       NEG soft_score range: [{neg_scores.min():.4f}, {neg_scores.max():.4f}]")

    # ── Assertion status ──
    if 'assertion_status' in df.columns:
        ambig = df['assertion_status'].isin(['POSSIBLE', 'HISTORICAL', 'CONDITIONAL']).sum()
        check(ambig == 0, f"No ambiguous assertions: {ambig} found")
        print(f"       Assertions: {dict(df['assertion_status'].value_counts())}")

    # ── View position (if image-level) ──
    if 'ViewPosition' in df.columns:
        views = set(df['ViewPosition'].dropna().unique())
        non_frontal = views - {'AP', 'PA'}
        check(len(non_frontal) == 0, f"Frontal-only views: {views}")

    # ── Text fields ──
    if 'impression_text' in df.columns:
        imp_missing = df['impression_text'].isna().sum()
        find_missing = df['findings_text'].isna().sum() if 'findings_text' in df.columns else -1
        
        if fname.startswith('pp1_'):
            # Split files MUST have both
            check(imp_missing == 0, f"impression_text: {imp_missing:,} missing (must be 0 for splits)")
            check(find_missing == 0, f"findings_text: {find_missing:,} missing (must be 0 for splits)")
        else:
            print(f"  📝 impression_text missing: {imp_missing:,}/{len(df):,} ({100*imp_missing/len(df):.1f}%)")
            if find_missing >= 0:
                print(f"  📝 findings_text missing: {find_missing:,}/{len(df):,} ({100*find_missing/len(df):.1f}%)")

    # ── Study-level label consistency ──
    if 'label' in df.columns and 'study_id' in df.columns:
        label_per_study = df.groupby('study_id')['label'].nunique()
        inconsistent = (label_per_study > 1).sum()
        check(inconsistent == 0, f"Study-level label consistency: {inconsistent} studies with mixed labels")


# ─── 2. Patient leakage across splits ───
print(f"\n{'='*70}")
print("  PATIENT LEAKAGE CHECK")
print(f"{'='*70}")

if all(f in files for f in ['pp1_train.csv', 'pp1_val.csv', 'pp1_test.csv']):
    train_pts = set(files['pp1_train.csv']['subject_id'].unique())
    val_pts = set(files['pp1_val.csv']['subject_id'].unique())
    test_pts = set(files['pp1_test.csv']['subject_id'].unique())

    tv = train_pts & val_pts
    tt = train_pts & test_pts
    vt = val_pts & test_pts

    check(len(tv) == 0, f"Train ∩ Val: {len(tv)} patients")
    check(len(tt) == 0, f"Train ∩ Test: {len(tt)} patients")
    check(len(vt) == 0, f"Val ∩ Test: {len(vt)} patients")

# ─── 3. Study-level balance per split ───
print(f"\n{'='*70}")
print("  STUDY-LEVEL BALANCE PER SPLIT")
print(f"{'='*70}")

for fname in ['pp1_train.csv', 'pp1_val.csv', 'pp1_test.csv']:
    if fname in files:
        df = files[fname]
        pos_studies = df[df['label'] == 1]['study_id'].nunique()
        neg_studies = df[df['label'] == 0]['study_id'].nunique()
        print(f"  {fname}: POS studies={pos_studies:,}  NEG studies={neg_studies:,}  (1:{neg_studies/pos_studies:.2f})")

# ─── 4. Cross-file consistency ───
print(f"\n{'='*70}")
print("  CROSS-FILE CONSISTENCY")
print(f"{'='*70}")

if 'final_image_training_manifest.csv' in files:
    manifest_studies = set(files['final_image_training_manifest.csv']['study_id'].unique())
    for fname in ['pp1_train.csv', 'pp1_val.csv', 'pp1_test.csv']:
        if fname in files:
            split_studies = set(files[fname]['study_id'].unique())
            outside = split_studies - manifest_studies
            check(len(outside) == 0, f"{fname} studies ⊂ manifest: {len(outside)} orphans")

# ─── 5. Total study/image counts across splits ───
print(f"\n{'='*70}")
print("  AGGREGATE TOTALS ACROSS SPLITS")
print(f"{'='*70}")

if all(f in files for f in ['pp1_train.csv', 'pp1_val.csv', 'pp1_test.csv']):
    combined = pd.concat([files['pp1_train.csv'], files['pp1_val.csv'], files['pp1_test.csv']])
    total_images = len(combined)
    total_studies = combined['study_id'].nunique()
    total_patients = combined['subject_id'].nunique()
    total_pos = combined[combined['label']==1]['study_id'].nunique()
    total_neg = combined[combined['label']==0]['study_id'].nunique()
    print(f"  Total images: {total_images:,}")
    print(f"  Total studies: {total_studies:,} (target: 20,000)")
    print(f"  Total patients: {total_patients:,}")
    print(f"  POS studies: {total_pos:,} (target: 10,000)")
    print(f"  NEG studies: {total_neg:,} (target: 10,000)")
    check(total_pos == 10000, f"POS studies == 10,000: got {total_pos:,}")
    check(total_neg == 10000, f"NEG studies == 10,000: got {total_neg:,}")

# ─── FINAL VERDICT ───
print(f"\n{'█'*70}")
if len(issues) == 0:
    print(f"  ✅ ALL CHECKS PASSED — ZERO ISSUES FOUND")
else:
    print(f"  ❌ {len(issues)} ISSUE(S) FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. {issue}")
print(f"{'█'*70}")

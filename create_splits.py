"""
Build train / val / test CSVs for the PP1 Multimodal Model.

Requirements:
  - 10,000 POSITIVE studies + 10,000 NEGATIVE studies (20K total studies)
  - Frontal views only (AP/PA) — already filtered in manifest
  - Each row = 1 physical JPG image with its report text
  - Patient-level split (no subject appears in multiple splits) to prevent data leakage
  - Split ratio: 80% train / 10% val / 10% test
  - Must have BOTH impression_text AND findings_text (zero missing fields)
  - impression_text + findings_text for BioBART report generation

Output columns:
  dicom_id, subject_id, study_id, image_rel_path, ViewPosition,
  label, soft_score, impression_text, findings_text, split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)

# ─── Load cleaned manifest (frontal-only, no POSSIBLE/HISTORICAL/CONDITIONAL) ───
print("Loading final_image_training_manifest.csv...")
df = pd.read_csv('data/output/final_image_training_manifest.csv')
print(f"  Total images: {len(df):,}")
print(f"  Total studies: {df['study_id'].nunique():,}")
print(f"  ViewPositions: {dict(df['ViewPosition'].value_counts())}")

# ─── Step 1: Require BOTH impression_text AND findings_text ───
df_with_text = df[df['impression_text'].notna() & df['findings_text'].notna()].copy()
print(f"\n  After requiring BOTH impression + findings: {len(df_with_text):,} images, {df_with_text['study_id'].nunique():,} studies")

pos_studies = df_with_text[df_with_text['label'] == 1]['study_id'].unique()
neg_studies = df_with_text[df_with_text['label'] == 0]['study_id'].unique()
print(f"  Available POS studies with text: {len(pos_studies):,}")
print(f"  Available NEG studies with text: {len(neg_studies):,}")

# ─── Step 2: Build study-level dataframe (one row per study) ───
# Pick the best image per study for sampling (prefer PA over AP)
study_df = df_with_text.groupby('study_id').agg({
    'subject_id': 'first',
    'label': 'first',
    'soft_score': 'first',
    'confidence_tier': 'first',
    'assertion_status': 'first',
    'impression_text': 'first',
    'findings_text': 'first'
}).reset_index()

# ─── Step 3: Get unique patients per class ───
pos_study_df = study_df[study_df['label'] == 1]
neg_study_df = study_df[study_df['label'] == 0]

# Get unique patients for each class
pos_patients = pos_study_df['subject_id'].unique()
neg_patients = neg_study_df['subject_id'].unique()

# Some patients might have BOTH pos and neg studies — assign them to pos class for splitting
mixed_patients = set(pos_patients) & set(neg_patients)
pure_neg_patients = np.array([p for p in neg_patients if p not in mixed_patients])

print(f"\n  POS-class patients: {len(pos_patients):,}")
print(f"  NEG-class patients (pure): {len(pure_neg_patients):,}")
print(f"  Mixed patients (have both POS+NEG studies): {len(mixed_patients):,}")

# ─── Step 4: Patient-level stratified split (80/10/10) ───
# Split POS patients
pos_train_pts, pos_temp_pts = train_test_split(pos_patients, test_size=0.20, random_state=42)
pos_val_pts, pos_test_pts = train_test_split(pos_temp_pts, test_size=0.50, random_state=42)

# Split NEG patients (pure negatives only)
neg_train_pts, neg_temp_pts = train_test_split(pure_neg_patients, test_size=0.20, random_state=42)
neg_val_pts, neg_test_pts = train_test_split(neg_temp_pts, test_size=0.50, random_state=42)

print(f"\n  Patient-level split:")
print(f"    POS patients — Train: {len(pos_train_pts)}, Val: {len(pos_val_pts)}, Test: {len(pos_test_pts)}")
print(f"    NEG patients — Train: {len(neg_train_pts)}, Val: {len(neg_val_pts)}, Test: {len(neg_test_pts)}")

# ─── Step 5: Sample 10K POS + 10K NEG studies from the split patients ───
# Target per split: Train 8K, Val 1K, Test 1K (for each class)
TARGET_POS = {'train': 8000, 'val': 1000, 'test': 1000}
TARGET_NEG = {'train': 8000, 'val': 1000, 'test': 1000}

def sample_studies(study_df, patient_ids, target_n, label_name, split_name):
    """Sample target_n studies from patients in patient_ids."""
    pool = study_df[study_df['subject_id'].isin(patient_ids)]
    if len(pool) < target_n:
        print(f"    WARNING: Only {len(pool)} {label_name} studies in {split_name} (need {target_n}), using all")
        return pool['study_id'].values
    # Prefer higher confidence (sort by soft_score distance from 0.5)
    pool = pool.copy()
    pool['confidence'] = abs(pool['soft_score'] - 0.5)
    pool = pool.sort_values('confidence', ascending=False)
    return pool.head(target_n)['study_id'].values

selected_studies = {}
for split_name, pos_pts, neg_pts in [
    ('train', pos_train_pts, neg_train_pts),
    ('val', pos_val_pts, neg_val_pts),
    ('test', pos_test_pts, neg_test_pts)
]:
    pos_ids = sample_studies(pos_study_df, pos_pts, TARGET_POS[split_name], 'POS', split_name)
    neg_ids = sample_studies(neg_study_df, neg_pts, TARGET_NEG[split_name], 'NEG', split_name)
    selected_studies[split_name] = np.concatenate([pos_ids, neg_ids])

# ─── Step 6: Expand back to image level (one row per physical JPG) ───
output_columns = [
    'dicom_id', 'subject_id', 'study_id', 'image_rel_path', 'ViewPosition',
    'label', 'soft_score', 'confidence_tier', 'assertion_status',
    'impression_text', 'findings_text'
]

print(f"\n{'='*60}")
print(f"  FINAL SPLITS (image-level)")
print(f"{'='*60}")

for split_name in ['train', 'val', 'test']:
    study_ids = selected_studies[split_name]
    split_df = df_with_text[df_with_text['study_id'].isin(study_ids)][output_columns].copy()
    split_df['split'] = split_name
    
    # Shuffle
    split_df = split_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Stats
    n_images = len(split_df)
    n_studies = split_df['study_id'].nunique()
    n_patients = split_df['subject_id'].nunique()
    n_pos = (split_df['label'] == 1).sum()
    n_neg = (split_df['label'] == 0).sum()
    pos_studies = split_df[split_df['label']==1]['study_id'].nunique()
    neg_studies = split_df[split_df['label']==0]['study_id'].nunique()
    has_impression = split_df['impression_text'].notna().sum()
    has_findings = split_df['findings_text'].notna().sum()
    
    fname = f'data/output/pp1_{split_name}.csv'
    split_df.to_csv(fname, index=False)
    
    print(f"\n  {split_name.upper()}: {fname}")
    print(f"    Images: {n_images:,}  |  Studies: {n_studies:,}  |  Patients: {n_patients:,}")
    print(f"    POS images: {n_pos:,} ({pos_studies:,} studies)  |  NEG images: {n_neg:,} ({neg_studies:,} studies)")
    print(f"    Ratio: 1:{n_neg/n_pos:.2f}")
    print(f"    impression_text: {has_impression:,}/{n_images:,} ({100*has_impression/n_images:.1f}%)")
    print(f"    findings_text: {has_findings:,}/{n_images:,} ({100*has_findings/n_images:.1f}%)")
    views = dict(split_df['ViewPosition'].value_counts())
    print(f"    ViewPositions: {views}")

# ─── Step 7: Verify no patient leakage across splits ───
print(f"\n{'='*60}")
print(f"  DATA LEAKAGE CHECK")
print(f"{'='*60}")

train_pts = set(pd.read_csv('data/output/pp1_train.csv')['subject_id'].unique())
val_pts = set(pd.read_csv('data/output/pp1_val.csv')['subject_id'].unique())
test_pts = set(pd.read_csv('data/output/pp1_test.csv')['subject_id'].unique())

tv = train_pts & val_pts
tt = train_pts & test_pts
vt = val_pts & test_pts

print(f"  Train ∩ Val patients:  {len(tv)} {'✅ CLEAN' if len(tv)==0 else '❌ LEAKAGE!'}")
print(f"  Train ∩ Test patients: {len(tt)} {'✅ CLEAN' if len(tt)==0 else '❌ LEAKAGE!'}")
print(f"  Val ∩ Test patients:   {len(vt)} {'✅ CLEAN' if len(vt)==0 else '❌ LEAKAGE!'}")

# ─── Step 8: Verify frontal-only ───
print(f"\n{'='*60}")
print(f"  FRONTAL-ONLY CHECK")
print(f"{'='*60}")
for fname in ['pp1_train.csv', 'pp1_val.csv', 'pp1_test.csv']:
    d = pd.read_csv(f'data/output/{fname}')
    views = set(d['ViewPosition'].unique())
    lateral = views - {'AP', 'PA'}
    if lateral:
        print(f"  {fname}: ❌ Contains non-frontal views: {lateral}")
    else:
        print(f"  {fname}: ✅ Frontal only ({views})")

print(f"\n{'='*60}")
print(f"  ✅ ALL DONE — 3 files saved to data/output/")
print(f"{'='*60}")

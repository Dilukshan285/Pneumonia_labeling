import pandas as pd

# 1. Check metadata
print("=" * 60)
print("METADATA")
meta = pd.read_csv('data/raw/mimic-cxr-2.0.0-metadata.csv')
print(f"Shape: {meta.shape}")
print(f"Columns: {list(meta.columns)}")
print(meta.head(3).to_string())
vp = meta['ViewPosition'].value_counts()
print(f"\nViewPosition distribution:")
for k, v in vp.items():
    print(f"  {k}: {v}")

# 2. Check manifest
print("\n" + "=" * 60)
print("FINAL IMAGE TRAINING MANIFEST (cleaned)")
mf = pd.read_csv('data/output/final_image_training_manifest.csv')
print(f"Shape: {mf.shape}")
print(f"Columns: {list(mf.columns)}")
print(mf.head(3).to_string())
vp2 = mf['ViewPosition'].value_counts()
print(f"\nViewPosition distribution:")
for k, v in vp2.items():
    print(f"  {k}: {v}")
print(f"\nUnique study_ids: {mf['study_id'].nunique()}")
print(f"Unique subject_ids: {mf['subject_id'].nunique()}")
pos = mf[mf['label'] == 1]
neg = mf[mf['label'] == 0]
print(f"POS studies: {pos['study_id'].nunique()}")
print(f"NEG studies: {neg['study_id'].nunique()}")
print(f"\nSample image_rel_path values:")
for p in mf['image_rel_path'].head(5).values:
    print(f"  {p}")

# 3. Check training_ready_labels for text
print("\n" + "=" * 60)
print("TRAINING READY LABELS (cleaned)")
tr = pd.read_csv('data/output/training_ready_labels.csv')
print(f"Shape: {tr.shape}")
print(f"Columns: {list(tr.columns)}")
has_impression = tr['impression_text'].notna().sum()
has_findings = tr['findings_text'].notna().sum()
print(f"Has impression_text: {has_impression}/{len(tr)} ({100*has_impression/len(tr):.1f}%)")
print(f"Has findings_text: {has_findings}/{len(tr)} ({100*has_findings/len(tr):.1f}%)")

# 4. Check MIMIC split file
print("\n" + "=" * 60)
print("MIMIC OFFICIAL SPLIT")
sp = pd.read_csv('data/raw/mimic-cxr-2.0.0-split.csv')
print(f"Shape: {sp.shape}")
print(f"Columns: {list(sp.columns)}")
print(sp.head(3).to_string())
split_dist = sp['split'].value_counts()
print(f"\nSplit distribution:")
for k, v in split_dist.items():
    print(f"  {k}: {v}")

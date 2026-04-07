"""Analyze disagreement patterns between pipeline and manual labels."""
import pandas as pd

df_manual = pd.read_csv('data/output/manual_validation_labels.csv')
df_manual['study_id'] = df_manual['study_id'].astype(str)
df_pipeline = pd.read_csv('data/output/final_pneumonia_labels.csv')
df_pipeline['study_id'] = df_pipeline['study_id'].astype(str)
df_sample = pd.read_csv('data/output/validation_sample_300.csv')
df_sample['study_id'] = df_sample['study_id'].astype(str)

df = df_manual.merge(df_pipeline[['study_id','label','soft_score']], on='study_id')
df = df.merge(df_sample[['study_id','impression_text','findings_text']], on='study_id', suffixes=('','_dup'))

LABEL_MAP = {0: 'NEGATIVE', 1: 'POSITIVE', 2: 'UNCERTAIN'}

print("=" * 80)
print("DISAGREEMENT ANALYSIS")
print("=" * 80)

# Pipeline=POSITIVE, Manual=UNCERTAIN (35 cases)
mask = (df['label']==1) & (df['manual_label']=='UNCERTAIN')
print(f"\n--- Pipeline POSITIVE -> Manual UNCERTAIN: {mask.sum()} cases ---")
for _, r in df[mask].head(8).iterrows():
    imp = str(r['impression_text'])[:200]
    find = str(r['findings_text'])[:200]
    print(f"  {r['study_id']}  soft={r['soft_score']:.4f}")
    print(f"    IMP: {imp}")
    if find and find != 'nan':
        print(f"    FIND: {find[:100]}")
    print()

# Pipeline=NEGATIVE, Manual=UNCERTAIN (28 cases)
mask2 = (df['label']==0) & (df['manual_label']=='UNCERTAIN')
print(f"\n--- Pipeline NEGATIVE -> Manual UNCERTAIN: {mask2.sum()} cases ---")
for _, r in df[mask2].head(8).iterrows():
    imp = str(r['impression_text'])[:200]
    find = str(r['findings_text'])[:200]
    print(f"  {r['study_id']}  soft={r['soft_score']:.4f}")
    print(f"    IMP: {imp}")
    if find and find != 'nan':
        print(f"    FIND: {find[:100]}")
    print()

# Pipeline=POSITIVE, Manual=NEGATIVE (20 cases)
mask3 = (df['label']==1) & (df['manual_label']=='NEGATIVE')
print(f"\n--- Pipeline POSITIVE -> Manual NEGATIVE: {mask3.sum()} cases ---")
for _, r in df[mask3].head(8).iterrows():
    imp = str(r['impression_text'])[:200]
    find = str(r['findings_text'])[:200]
    print(f"  {r['study_id']}  soft={r['soft_score']:.4f}")
    print(f"    IMP: {imp}")
    if find and find != 'nan':
        print(f"    FIND: {find[:100]}")
    print()

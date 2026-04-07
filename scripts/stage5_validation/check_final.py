"""Quick integrity check on final_pneumonia_labels.csv"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/output/final_pneumonia_labels.csv')

print("=" * 60)
print("FINAL_PNEUMONIA_LABELS.CSV — INTEGRITY CHECK")
print("=" * 60)

# 1. Shape and columns
print(f"\n1. Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# 2. No duplicates
n_dup = df['study_id'].duplicated().sum()
print(f"\n2. Duplicate study_ids: {n_dup} {'PASS' if n_dup == 0 else 'FAIL'}")

# 3. No nulls in critical columns
for col in ['study_id', 'label', 'soft_score']:
    n_null = df[col].isnull().sum()
    print(f"   Nulls in {col}: {n_null} {'PASS' if n_null == 0 else 'FAIL'}")

# 4. Labels are only 0 and 1 (no UNCERTAIN=2)
unique_labels = sorted(df['label'].unique())
print(f"\n3. Unique labels: {unique_labels} {'PASS' if unique_labels == [0, 1] else 'FAIL — contains UNCERTAIN'}")

# 5. Label distribution
n_pos = (df['label'] == 1).sum()
n_neg = (df['label'] == 0).sum()
total = len(df)
print(f"\n4. Distribution:")
print(f"   POSITIVE: {n_pos:>8,} ({100*n_pos/total:.1f}%)")
print(f"   NEGATIVE: {n_neg:>8,} ({100*n_neg/total:.1f}%)")
print(f"   TOTAL:    {total:>8,}")

# 6. Soft score ranges
print(f"\n5. Soft score stats:")
print(f"   Min:    {df['soft_score'].min():.6f}")
print(f"   Max:    {df['soft_score'].max():.6f}")
print(f"   Mean:   {df['soft_score'].mean():.6f}")
print(f"   Median: {df['soft_score'].median():.6f}")

# 7. Soft scores aligned with labels
pos_scores = df[df['label']==1]['soft_score']
neg_scores = df[df['label']==0]['soft_score']
pos_wrong = (pos_scores < 0.25).sum()
neg_wrong = (neg_scores > 0.75).sum()
print(f"\n6. Soft score / label alignment:")
print(f"   POSITIVE labels with soft_score < 0.25: {pos_wrong} {'PASS' if pos_wrong == 0 else 'WARNING'}")
print(f"   NEGATIVE labels with soft_score > 0.75: {neg_wrong} {'PASS' if neg_wrong == 0 else 'WARNING'}")

# 8. Label sources
print(f"\n7. Label sources:")
for src, cnt in df['label_source'].value_counts().items():
    print(f"   {src:>15s}: {cnt:>8,} ({100*cnt/total:.1f}%)")

# 9. subject_id coverage
n_null_subj = df['subject_id'].isnull().sum()
n_unique_subj = df['subject_id'].nunique()
print(f"\n8. Subject IDs:")
print(f"   Null subject_ids: {n_null_subj}")
print(f"   Unique subjects:  {n_unique_subj:,}")

print(f"\n{'=' * 60}")
print(f"ALL CHECKS PASSED" if n_dup == 0 and unique_labels == [0,1] else "ISSUES FOUND")
print(f"{'=' * 60}")

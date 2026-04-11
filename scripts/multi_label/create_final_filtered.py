"""
Filter ml_full_manifest.csv by soft_score and ensure 100% report coverage.
Output: multi_label_final.csv
"""
import pandas as pd

# Load
df = pd.read_csv("data/output/multi_label_dataset/ml_full_manifest.csv", low_memory=False)
print(f"Full manifest: {len(df)} rows")

# Filter by soft_score
pos = df[df["soft_score"] >= 0.75]
neg = df[df["soft_score"] <= 0.25]
filtered = pd.concat([pos, neg], ignore_index=True)
print(f"After soft_score filter (>=0.75 or <=0.25): {len(filtered)} rows")
print(f"  Positive (>=0.75): {len(pos)}")
print(f"  Negative (<=0.25): {len(neg)}")

# Ensure 100% report coverage
has_report = (
    (filtered["findings_text"].fillna("").str.strip() != "") |
    (filtered["impression_text"].fillna("").str.strip() != "")
)
no_report = (~has_report).sum()
print(f"\nReport coverage check:")
print(f"  With report: {has_report.sum()}")
print(f"  Without report: {no_report}")

filtered = filtered[has_report].reset_index(drop=True)
print(f"  After dropping empty reports: {len(filtered)} rows")

# Pneumonia breakdown
print(f"\n=== PNEUMONIA BREAKDOWN ===")
print(f"  Pneumonia label=1 (PRESENT):    {(filtered['Pneumonia'] == 1).sum()}")
print(f"  Pneumonia label=0 (ABSENT):     {(filtered['Pneumonia'] == 0).sum()}")
print(f"  Pneumonia label=-1 (UNCERTAIN): {(filtered['Pneumonia'] == -1).sum()}")

# By soft_score group
pos_f = filtered[filtered["soft_score"] >= 0.75]
neg_f = filtered[filtered["soft_score"] <= 0.25]
print(f"\n=== BY SOFT_SCORE GROUP ===")
print(f"  Positive group (soft_score >= 0.75): {len(pos_f)}")
print(f"    Pneumonia PRESENT:   {(pos_f['Pneumonia'] == 1).sum()}")
print(f"    Pneumonia ABSENT:    {(pos_f['Pneumonia'] == 0).sum()}")
print(f"    Pneumonia UNCERTAIN: {(pos_f['Pneumonia'] == -1).sum()}")
print(f"  Negative group (soft_score <= 0.25): {len(neg_f)}")
print(f"    Pneumonia PRESENT:   {(neg_f['Pneumonia'] == 1).sum()}")
print(f"    Pneumonia ABSENT:    {(neg_f['Pneumonia'] == 0).sum()}")
print(f"    Pneumonia UNCERTAIN: {(neg_f['Pneumonia'] == -1).sum()}")

# All 14 diseases
diseases = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion", "Lung_Opacity",
    "No_Finding", "Pleural_Effusion", "Pleural_Other", "Pneumonia",
    "Pneumothorax", "Support_Devices",
]
print(f"\n=== ALL 14 DISEASES IN FILTERED DATA ===")
print(f"{'Pathology':30s} {'PRESENT':>8s} {'ABSENT':>8s} {'UNCERTAIN':>10s} {'Prev%':>7s}")
print("-" * 70)
for cls in diseases:
    n_p = (filtered[cls] == 1).sum()
    n_a = (filtered[cls] == 0).sum()
    n_u = (filtered[cls] == -1).sum()
    prev = n_p / len(filtered) * 100
    print(f"{cls:30s} {n_p:8d} {n_a:8d} {n_u:10d} {prev:6.1f}%")

# Save
filtered.to_csv("data/output/multi_label_dataset/multi_label_final.csv", index=False)
print(f"\n[DONE] Saved: data/output/multi_label_dataset/multi_label_final.csv ({len(filtered)} rows)")

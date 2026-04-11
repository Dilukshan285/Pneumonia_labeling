import pandas as pd

df = pd.read_csv("data/output/multi_label_dataset/ml_full_manifest.csv", low_memory=False)
print(f"Loaded: {len(df)} rows")

# Dedup by study_id (prefer PA > AP)
view_priority = {"PA": 0, "AP": 1}
df["_vr"] = df["ViewPosition"].map(view_priority).fillna(2)
df = df.sort_values("_vr").drop_duplicates(subset="study_id", keep="first")
df = df.drop(columns=["_vr"]).reset_index(drop=True)
print(f"After study_id dedup: {len(df)} rows")

# 100% report coverage
has_report = (df["findings_text"].fillna("").str.strip() != "") | (df["impression_text"].fillna("").str.strip() != "")
df = df[has_report].reset_index(drop=True)
print(f"After report filter: {len(df)} rows (100% coverage)")

# Drop 5 unwanted columns
drop_diseases = ["Support_Devices", "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion", "Pleural_Other"]
drop_cols = drop_diseases + ["conf_" + d for d in drop_diseases]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Summary
diseases = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Lung_Opacity",
            "No_Finding", "Pleural_Effusion", "Pneumonia", "Pneumothorax"]

print(f"\n=== FINAL 9 DISEASE CLASSES ===")
print(f"{'Pathology':30s} {'PRESENT':>8s} {'ABSENT':>8s} {'UNC':>6s} {'Prev%':>7s}")
print("-" * 65)
for cls in diseases:
    n_p = (df[cls] == 1).sum()
    n_a = (df[cls] == 0).sum()
    n_u = (df[cls] == -1).sum()
    prev = n_p / len(df) * 100
    print(f"{cls:30s} {n_p:8d} {n_a:8d} {n_u:6d} {prev:6.1f}%")

print(f"\n=== PNEUMONIA SOFT_SCORE ===")
pos = (df["soft_score"] >= 0.75).sum()
neg = (df["soft_score"] <= 0.25).sum()
print(f"  Positive (>=0.75): {pos}")
print(f"  Negative (<=0.25): {neg}")
print(f"  Pneumonia PRESENT: {(df['Pneumonia'] == 1).sum()}")
print(f"  Pneumonia ABSENT:  {(df['Pneumonia'] == 0).sum()}")

# Save
df.to_csv("data/output/multi_label_dataset/multi_label_final.csv", index=False)
print(f"\n[DONE] multi_label_final.csv saved ({len(df)} rows, {len(df.columns)} columns)")
print(f"Columns: {list(df.columns)}")

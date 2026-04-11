import pandas as pd

df = pd.read_csv("data/output/multi_label_dataset/multi_label_final.csv", low_memory=False)

# Create comparison
df["seven_layer"] = (df["soft_score"] >= 0.75).astype(int)
df["three_layer"] = (df["Pneumonia"] == 1).astype(int)

print("=" * 70)
print("DEEP ANALYSIS: 7-Layer vs 3-Layer Pneumonia Labeling")
print("=" * 70)

both_yes = ((df.seven_layer == 1) & (df.three_layer == 1)).sum()
both_no = ((df.seven_layer == 0) & (df.three_layer == 0)).sum()
seven_yes_three_no = ((df.seven_layer == 1) & (df.three_layer == 0)).sum()
seven_no_three_yes = ((df.seven_layer == 0) & (df.three_layer == 1)).sum()
seven_yes_three_unc = ((df.seven_layer == 1) & (df.Pneumonia == -1)).sum()

print(f"\n=== AGREEMENT MATRIX ===")
print(f"  Both say POSITIVE:           {both_yes:>6d}  (agree)")
print(f"  Both say NEGATIVE:           {both_no:>6d}  (agree)")
print(f"  7-layer YES, 3-layer NO:     {seven_yes_three_no:>6d}  (DISAGREE)")
print(f"  7-layer YES, 3-layer UNC:    {seven_yes_three_unc:>6d}  (DISAGREE)")
print(f"  7-layer NO,  3-layer YES:    {seven_no_three_yes:>6d}  (DISAGREE)")

agreement = (both_yes + both_no) / len(df) * 100
print(f"\n  Overall agreement: {agreement:.1f}%")

# Analyze the big disagreement group
disagree = df[(df.seven_layer == 1) & (df.Pneumonia != 1)]
print(f"\n{'='*70}")
print(f"ANALYZING {len(disagree)} CASES: 7-layer says PNEUMONIA, 3-layer says NO/UNCERTAIN")
print(f"{'='*70}")

# What did the 3-layer classify these as instead?
print(f"\n  What 3-layer labeled these as:")
print(f"  Pneumonia column value:")
print(f"    Absent (0):    {(disagree['Pneumonia'] == 0).sum()}")
print(f"    Uncertain (-1): {(disagree['Pneumonia'] == -1).sum()}")

print(f"\n  Other diseases the 3-layer found in these cases:")
diseases = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Lung_Opacity", "No_Finding", "Pleural_Effusion", "Pneumothorax"]
for cls in diseases:
    p = (disagree[cls] == 1).sum()
    pct = p / len(disagree) * 100
    print(f"    {cls:28s} PRESENT in {p:5d} ({pct:5.1f}%)")

# Sample actual report text from disagreement cases
print(f"\n{'='*70}")
print(f"SAMPLE DISAGREEMENT REPORTS (7-layer=YES, 3-layer=NO)")
print(f"{'='*70}")

samples = disagree[disagree["Pneumonia"] == 0].sample(5, random_state=42)
for i, (_, row) in enumerate(samples.iterrows()):
    imp = str(row["impression_text"])[:300] if pd.notna(row["impression_text"]) else "N/A"
    find = str(row["findings_text"])[:300] if pd.notna(row["findings_text"]) else "N/A"
    ss = row["soft_score"]
    print(f"\n--- Case {i+1} (soft_score={ss}) ---")
    print(f"  IMPRESSION: {imp}")
    print(f"  FINDINGS:   {find}")
    pneu = row["Pneumonia"]
    cons = row["Consolidation"]
    opac = row["Lung_Opacity"]
    nofind = row["No_Finding"]
    print(f"  3-layer -> Pneumonia={pneu}, Consolidation={cons}, Lung_Opacity={opac}, No_Finding={nofind}")

# Also check: cases where 3-layer says YES but 7-layer says NO
reverse = df[(df.seven_layer == 0) & (df.three_layer == 1)]
print(f"\n{'='*70}")
print(f"REVERSE: 3-layer says YES, 7-layer says NO ({len(reverse)} cases)")
print(f"{'='*70}")
samples2 = reverse.sample(min(3, len(reverse)), random_state=42)
for i, (_, row) in enumerate(samples2.iterrows()):
    imp = str(row["impression_text"])[:300] if pd.notna(row["impression_text"]) else "N/A"
    ss = row["soft_score"]
    print(f"\n--- Reverse Case {i+1} (soft_score={ss}) ---")
    print(f"  IMPRESSION: {imp}")

print(f"\n{'='*70}")
print(f"VERDICT")
print(f"{'='*70}")
print(f"  7-layer found {(df.seven_layer==1).sum()} pneumonia cases")
print(f"  3-layer found {(df.three_layer==1).sum()} pneumonia cases")
print(f"  Difference: {(df.seven_layer==1).sum() - (df.three_layer==1).sum()} extra cases in 7-layer")

"""Final verification of the advanced labeling pipeline output."""
import pandas as pd
import numpy as np

print("=" * 70)
print("  VERIFICATION: advanced_final_labels.csv")
print("=" * 70)

df = pd.read_csv("data/output/advanced_final_labels.csv")
print(f"\n  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

n_pos = (df["label"] == 1).sum()
n_neg = (df["label"] == 0).sum()
n_other = ((df["label"] != 0) & (df["label"] != 1)).sum()
print(f"\n  Label distribution:")
print(f"    POSITIVE (1): {n_pos:,}")
print(f"    NEGATIVE (0): {n_neg:,}")
print(f"    Other:        {n_other}")
assert n_other == 0, "ERROR: Non-binary labels found!"

pos_scores = df[df["label"] == 1]["soft_score"]
neg_scores = df[df["label"] == 0]["soft_score"]
print(f"\n  Soft score stats:")
print(f"    POS: mean={pos_scores.mean():.4f}  std={pos_scores.std():.4f}  "
      f"min={pos_scores.min():.4f}  max={pos_scores.max():.4f}")
print(f"    NEG: mean={neg_scores.mean():.4f}  std={neg_scores.std():.4f}  "
      f"min={neg_scores.min():.4f}  max={neg_scores.max():.4f}")

dups = df["study_id"].duplicated().sum()
print(f"\n  Duplicate study_ids: {dups}")
assert dups == 0, "ERROR: Duplicate study_ids found!"

print(f"\n  Null checks:")
for col in ["label", "soft_score", "subject_id", "study_id"]:
    nulls = df[col].isna().sum()
    status = "OK" if nulls == 0 else "FAIL"
    print(f"    {col}: {nulls} nulls [{status}]")
    assert nulls == 0, f"ERROR: Nulls in {col}!"

print(f"\n  Quality tiers:")
for tier, cnt in df["confidence_tier"].value_counts().items():
    n_tp = (df.loc[df["confidence_tier"] == tier, "label"] == 1).sum()
    n_tn = (df.loc[df["confidence_tier"] == tier, "label"] == 0).sum()
    print(f"    {tier:>8s}: {cnt:>8,}  (POS: {n_tp:,}, NEG: {n_tn:,})")

has_imp = (df["impression_text"].fillna("").str.len() > 0).sum()
has_fin = (df["findings_text"].fillna("").str.len() > 0).sum()
print(f"\n  Report text coverage:")
print(f"    With impression: {has_imp:,}/{len(df):,} ({100*has_imp/len(df):.1f}%)")
print(f"    With findings:   {has_fin:,}/{len(df):,} ({100*has_fin/len(df):.1f}%)")

# Sample reports
print(f"\n  ── Sample POSITIVE report ──")
pos_with = df[(df["label"] == 1) & (df["impression_text"].fillna("").str.len() > 10)]
if len(pos_with) > 0:
    s = pos_with.iloc[0]
    print(f"    study_id:   {s['study_id']}")
    print(f"    soft_score: {s['soft_score']:.4f}")
    print(f"    tier:       {s['confidence_tier']}")
    print(f"    assertion:  {s['assertion_status']}")
    imp = str(s["impression_text"])[:250]
    print(f"    impression: {imp}")

print(f"\n  ── Sample NEGATIVE report ──")
neg_with = df[(df["label"] == 0) & (df["impression_text"].fillna("").str.len() > 10)]
if len(neg_with) > 0:
    s2 = neg_with.iloc[0]
    print(f"    study_id:   {s2['study_id']}")
    print(f"    soft_score: {s2['soft_score']:.4f}")
    print(f"    tier:       {s2['confidence_tier']}")
    imp2 = str(s2["impression_text"])[:250]
    print(f"    impression: {imp2}")

print()
print("=" * 70)
print("  VERIFICATION: training_ready_labels.csv")
print("=" * 70)

df2 = pd.read_csv("data/output/training_ready_labels.csv")
n2_pos = (df2["label"] == 1).sum()
n2_neg = (df2["label"] == 0).sum()
ratio = n2_neg / max(n2_pos, 1)
print(f"\n  Shape: {df2.shape}")
print(f"  POSITIVE: {n2_pos:,}")
print(f"  NEGATIVE: {n2_neg:,}")
print(f"  Ratio:    1:{ratio:.1f}")
assert abs(ratio - 4.0) < 0.1, f"ERROR: Ratio {ratio} != 4.0!"

is_subset = set(df2["study_id"]).issubset(set(df["study_id"]))
print(f"  Is subset of full: {is_subset}")
assert is_subset, "ERROR: Training set has IDs not in full set!"

dups2 = df2["study_id"].duplicated().sum()
print(f"  Duplicates: {dups2}")
assert dups2 == 0, "ERROR: Duplicates in training set!"

# Score separation
pos_min = df["label"] == 1
neg_max_score = neg_scores.max()
pos_min_score = pos_scores.min()
print(f"\n  Score boundary analysis:")
print(f"    Lowest POSITIVE soft_score:  {pos_min_score:.4f}")
print(f"    Highest NEGATIVE soft_score: {neg_max_score:.4f}")
gap = pos_min_score - neg_max_score
print(f"    Gap: {gap:.4f} {'(clean separation)' if gap > 0 else '(slight overlap at boundary - expected)'}")

print()
print("=" * 70)
print("  ALL VERIFICATIONS PASSED")
print("=" * 70)
print()
print("  Output files ready for PP1/PP2 model training:")
print(f"    1. advanced_final_labels.csv  → {len(df):,} labels ({n_pos:,} POS, {n_neg:,} NEG)")
print(f"    2. training_ready_labels.csv  → {len(df2):,} labels ({n2_pos:,} POS, {n2_neg:,} NEG) @ 1:{ratio:.0f}")
print(f"    3. Report text included       → impression + findings for PP2")
print()

"""
Clean all 3 output CSVs by removing:
  1. Ambiguous assertion statuses (POSSIBLE, HISTORICAL, CONDITIONAL)
  2. Reports in the soft_score dead zone (0.25 < soft_score < 0.75)

Then re-balance training_ready_labels.csv to 1:4 POS:NEG ratio.
"""
import pandas as pd
import numpy as np

AMBIGUOUS_ASSERTIONS = ['POSSIBLE', 'HISTORICAL', 'CONDITIONAL']

def clean(df, name):
    before = len(df)
    pos_before = (df['label'] == 1).sum()
    neg_before = (df['label'] == 0).sum()

    # Step 1: Remove ambiguous assertion statuses
    assertion_mask = ~df['assertion_status'].isin(AMBIGUOUS_ASSERTIONS)
    removed_assertions = (~assertion_mask).sum()

    # Step 2: Apply soft_score confidence thresholds
    score_mask = (
        ((df['label'] == 1) & (df['soft_score'] >= 0.75)) |
        ((df['label'] == 0) & (df['soft_score'] <= 0.25))
    )
    removed_scores = (~score_mask).sum()

    # Combined filter
    combined = assertion_mask & score_mask
    df_clean = df[combined].copy()

    after = len(df_clean)
    pos_after = (df_clean['label'] == 1).sum()
    neg_after = (df_clean['label'] == 0).sum()

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  BEFORE:  {before:,} total  |  {pos_before:,} POS  |  {neg_before:,} NEG")
    print(f"  Removed by assertion filter (POSSIBLE/HISTORICAL/CONDITIONAL): {removed_assertions:,}")
    print(f"  Removed by soft_score filter (dead zone 0.25-0.75):            {(~score_mask).sum():,}")
    print(f"  Removed total (union of both filters):                         {before - after:,}")
    print(f"  AFTER:   {after:,} total  |  {pos_after:,} POS  |  {neg_after:,} NEG")
    ratio = neg_after / pos_after if pos_after > 0 else float('inf')
    print(f"  Ratio:   1:{ratio:.2f}")
    print(f"  Remaining assertion statuses: {dict(df_clean['assertion_status'].value_counts())}")

    return df_clean


# ── 1. advanced_final_labels.csv (archive) ──
print("\n" + "█"*60)
print("  CLEANING ALL OUTPUT FILES")
print("█"*60)

df1 = pd.read_csv('data/output/advanced_final_labels.csv')
df1_clean = clean(df1, 'advanced_final_labels.csv')
df1_clean.to_csv('data/output/advanced_final_labels.csv', index=False)
print(f"  ✅ Saved: data/output/advanced_final_labels.csv")


# ── 2. training_ready_labels.csv (PP2 — re-balance to 1:4) ──
df2 = pd.read_csv('data/output/training_ready_labels.csv')
df2_clean = clean(df2, 'training_ready_labels.csv (before re-balance)')

# Re-balance to 1:4
pos_df = df2_clean[df2_clean['label'] == 1]
neg_df = df2_clean[df2_clean['label'] == 0]
target_neg = min(len(neg_df), len(pos_df) * 4)

# Sample negatives: 30% hard negatives (higher soft_score) + 70% easy negatives
neg_sorted = neg_df.sort_values('soft_score', ascending=False)
n_hard = int(target_neg * 0.30)
n_easy = target_neg - n_hard

hard_negs = neg_sorted.head(n_hard)
easy_pool = neg_sorted.iloc[n_hard:]
easy_negs = easy_pool.sample(n=min(n_easy, len(easy_pool)), random_state=42)

neg_sampled = pd.concat([hard_negs, easy_negs])
df2_balanced = pd.concat([pos_df, neg_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

pos_final = (df2_balanced['label'] == 1).sum()
neg_final = (df2_balanced['label'] == 0).sum()
print(f"\n  RE-BALANCED training_ready_labels.csv:")
print(f"  {len(df2_balanced):,} total  |  {pos_final:,} POS  |  {neg_final:,} NEG  |  Ratio 1:{neg_final/pos_final:.2f}")
df2_balanced.to_csv('data/output/training_ready_labels.csv', index=False)
print(f"  ✅ Saved: data/output/training_ready_labels.csv")


# ── 3. final_image_training_manifest.csv (PP1) ──
df3 = pd.read_csv('data/output/final_image_training_manifest.csv')
df3_clean = clean(df3, 'final_image_training_manifest.csv')
df3_clean.to_csv('data/output/final_image_training_manifest.csv', index=False)
print(f"  ✅ Saved: data/output/final_image_training_manifest.csv")


# ── Final summary ──
print("\n" + "█"*60)
print("  FINAL SUMMARY — ALL FILES CLEANED")
print("█"*60)
for fname in ['advanced_final_labels.csv', 'training_ready_labels.csv', 'final_image_training_manifest.csv']:
    df = pd.read_csv(f'data/output/{fname}')
    p = (df['label']==1).sum()
    n = (df['label']==0).sum()
    assertions = dict(df['assertion_status'].value_counts())
    print(f"\n  {fname}")
    print(f"    Total: {len(df):,}  |  POS: {p:,}  |  NEG: {n:,}  |  Ratio 1:{n/p:.2f}")
    print(f"    Assertions: {assertions}")
    print(f"    Soft score range: [{df['soft_score'].min():.6f}, {df['soft_score'].max():.6f}]")
    ambig = df[(df['soft_score'] > 0.25) & (df['soft_score'] < 0.75)]
    print(f"    Ambiguous zone reports: {len(ambig)}")

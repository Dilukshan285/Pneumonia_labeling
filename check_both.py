import pandas as pd

df = pd.read_csv('data/output/final_image_training_manifest.csv')

# Both impression AND findings
both = df[df['impression_text'].notna() & df['findings_text'].notna()]
both_pos = both[both['label'] == 1]
both_neg = both[both['label'] == 0]

print("Reports with BOTH impression + findings:")
print(f"  Total images: {len(both):,}")
print(f"  Unique studies: {both['study_id'].nunique():,}")
print(f"  POS studies: {both_pos['study_id'].nunique():,}")
print(f"  NEG studies: {both_neg['study_id'].nunique():,}")

print(f"\nEnough for 10K+10K? POS={'YES' if both_pos['study_id'].nunique()>=10000 else 'NO ('+str(both_pos['study_id'].nunique())+')'}, NEG={'YES' if both_neg['study_id'].nunique()>=10000 else 'NO ('+str(both_neg['study_id'].nunique())+')'}")

"""
Extract 100 POSITIVE and 100 NEGATIVE reports from pp1_train.csv
Save as individual text files in:
  data/output/reports/positive/
  data/output/reports/negative/
"""
import pandas as pd
import os

df = pd.read_csv('data/output/pp1_train.csv')

# Only reports with BOTH impression AND findings
df = df[df['impression_text'].notna() & df['findings_text'].notna()]

# Get unique studies (one report per study, not per image)
pos = df[df['label'] == 1].drop_duplicates(subset='study_id').head(100)
neg = df[df['label'] == 0].drop_duplicates(subset='study_id').head(100)

for label_name, subset in [('positive', pos), ('negative', neg)]:
    folder = f'data/output/reports/{label_name}'
    os.makedirs(folder, exist_ok=True)
    
    for _, row in subset.iterrows():
        sid = row['study_id']
        score = row['soft_score']
        impression = row['impression_text'] if pd.notna(row['impression_text']) else ''
        findings = row['findings_text'] if pd.notna(row['findings_text']) else ''
        
        content = f"Study ID: {sid}\n"
        content += f"Label: {row['label']}\n"
        content += f"Soft Score: {score:.4f}\n"
        content += f"View: {row['ViewPosition']}\n"
        content += f"\n{'='*50}\n"
        content += f"IMPRESSION:\n{impression}\n"
        content += f"\n{'='*50}\n"
        content += f"FINDINGS:\n{findings}\n"
        
        fname = f"s{sid}_score_{score:.4f}.txt"
        with open(os.path.join(folder, fname), 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"  ✅ {label_name}: {len(subset)} reports saved to {folder}/")

print("\n  Done!")

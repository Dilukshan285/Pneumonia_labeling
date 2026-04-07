import pandas as pd

chx = pd.read_csv(r'c:\Users\dviya\Desktop\Pneumonia_labeling\data\raw\mimic-cxr-2.0.0-chexpert.csv')

other_diseases = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'Pleural Effusion', 'Pleural Other', 'Pneumothorax', 'Support Devices'
]

# Studies where NO other disease is positive (1.0) or uncertain (-1.0)
has_no_other = pd.Series(True, index=chx.index)
for d in other_diseases:
    has_no_other = has_no_other & ((chx[d].isna()) | (chx[d] == 0.0))

# Clean negatives: no other disease AND no pneumonia
clean_neg = chx[has_no_other & (chx['Pneumonia'].isna() | (chx['Pneumonia'] == 0.0))]
print(f"Clean negatives (no other disease, no pneumonia): {len(clean_neg)}")
print(f"No Finding==1.0 among clean negs: {(clean_neg['No Finding'] == 1.0).sum()}")

# How many have No Finding=1.0 and Pneumonia not positive
nf_only = chx[(chx['No Finding'] == 1.0)]
print(f"\nNo Finding = 1.0 total: {len(nf_only)}")

# Studies where Pneumonia=1.0
pneu_pos = chx[chx['Pneumonia'] == 1.0]
print(f"Pneumonia = 1.0 total: {len(pneu_pos)}")

# Studies with ONLY pneumonia (no other disease positive or uncertain)
pneu_clean = pneu_pos.copy()
for d in other_diseases:
    pneu_clean = pneu_clean[(pneu_clean[d].isna()) | (pneu_clean[d] == 0.0)]
print(f"Pneumonia = 1.0 AND no other disease: {len(pneu_clean)}")

# Total unique study_ids
print(f"\nTotal unique study_ids: {chx['study_id'].nunique()}")
print(f"Total rows: {len(chx)}")

# Check overlap with parsed_reports
parsed = pd.read_csv(r'c:\Users\dviya\Desktop\Pneumonia_labeling\data\intermediate\parsed_reports.csv', usecols=['study_id'])
# Keep parsed study_ids as-is ('s' prefix), add 's' prefix to CheXpert
parsed['study_id'] = parsed['study_id'].astype(str).str.strip()
chx['study_id'] = 's' + chx['study_id'].astype(str).str.strip()
print(f"\nParsed reports total: {len(parsed)}")
print(f"Parsed study_id sample: {parsed['study_id'].iloc[:3].tolist()}")
print(f"CheXpert study_id sample: {chx['study_id'].iloc[:3].tolist()}")
merged = parsed.merge(chx[['study_id']], on='study_id', how='inner')
print(f"Parsed reports with CheXpert entry: {len(merged)}")

"""
Stage 6: Image Linking & Frontal View Extraction

This script links the text-level labels (study_id) to the physical image IDs (dicom_id),
filtering strictly for Frontal views (AP/PA) to prepare for PP1 DenseNet-121 training.
"""

import os
import pandas as pd
from collections import Counter

# Paths
PROJECT_DIR = r"c:\Users\dviya\Desktop\Pneumonia_labeling"
LABELS_CSV = os.path.join(PROJECT_DIR, "data", "output", "training_ready_labels.csv")
METADATA_CSV = os.path.join(PROJECT_DIR, "data", "raw", "mimic-cxr-2.0.0-metadata.csv")
OUTPUT_MANIFEST = os.path.join(PROJECT_DIR, "data", "output", "final_image_training_manifest.csv")

def main():
    print("=" * 70)
    print("  STAGE 6: IMAGE LINKING & FRONTAL EXTRACTION")
    print("=" * 70)
    
    # 1. Load Data
    print("\nLoading data...")
    df_labels = pd.read_csv(LABELS_CSV)
    print(f"  Loaded {len(df_labels):,} labels from training_ready_labels.csv")
    
    if not os.path.exists(METADATA_CSV):
        print(f"\nERROR: Could not find metadata CSV at {METADATA_CSV}")
        return
        
    df_meta = pd.read_csv(METADATA_CSV)
    print(f"  Loaded {len(df_meta):,} raw image metadata entries")
    
    # 2. Merge Data
    print("\nLinking images to labels...")
    # Align the formatting of study_id (metadata has '50414267', labels has 's50414267')
    df_meta['study_id'] = 's' + df_meta['study_id'].astype(str)
    
    # We use inner join so we only keep images that have a label in our 1:4 dataset
    df_merged = df_meta.merge(df_labels, on=['subject_id', 'study_id'], how='inner')
    print(f"  Matched {len(df_merged):,} total images to our labeled reports")
    
    # 3. View Position Filtering
    print("\nFiltering for Frontal Views (AP/PA) only...")
    view_counts = Counter(df_merged['ViewPosition'].fillna('UNKNOWN').tolist())
    for view, count in view_counts.items():
        print(f"    {view:<10}: {count:,}")
        
    mask_frontal = df_merged['ViewPosition'].isin(['AP', 'PA'])
    df_frontal = df_merged[mask_frontal].copy()
    
    n_dropped = set(df_merged['dicom_id']) - set(df_frontal['dicom_id'])
    print(f"\n  Dropped {len(n_dropped):,} non-frontal images (Lateral, etc.)")
    print(f"  Remaining Frontal images: {len(df_frontal):,}")
    
    # 4. Generate image paths
    print("\nGenerating absolute image paths...")
    # The standard MIMIC-CXR-JPG structure is: p10/p10000032/s50414267/174413ec-...jpg
    def generate_path(row):
        subj_str = str(row['subject_id'])
        p_folder = f"p{subj_str[:2]}"
        return f"{p_folder}/p{subj_str}/s{row['study_id']}/{row['dicom_id']}.jpg"
        
    df_frontal['image_rel_path'] = df_frontal.apply(generate_path, axis=1)
    
    # 5. Final Output Cleanup
    output_cols = [
        'dicom_id', 'subject_id', 'study_id', 'image_rel_path', 'ViewPosition',
        'label', 'soft_score', 'confidence_tier', 'label_source', 'assertion_status'
    ]
    
    # Add PP2 text columns if they exist
    if 'impression_text' in df_frontal.columns:
        output_cols.extend(['impression_text', 'findings_text'])
        
    df_final = df_frontal[output_cols].copy()
    
    # Print final verification
    n_pos = (df_final['label'] == 1).sum()
    n_neg = (df_final['label'] == 0).sum()
    
    print("\n" + "=" * 70)
    print("  FINAL MANIFEST VERIFICATION")
    print("=" * 70)
    print(f"  Total Training Images: {len(df_final):,}")
    print(f"  POSITIVE Class (1):    {n_pos:,}  ({100*n_pos/len(df_final):.1f}%)")
    print(f"  NEGATIVE Class (0):    {n_neg:,}  ({100*n_neg/len(df_final):.1f}%)")
    print(f"  Ratio:                 1:{n_neg/max(n_pos, 1):.1f}")
    
    print("\n  Saving final manifest...")
    df_final.to_csv(OUTPUT_MANIFEST, index=False)
    file_size_mb = os.path.getsize(OUTPUT_MANIFEST) / (1024 * 1024)
    print(f"  Saved to: {OUTPUT_MANIFEST}")
    print(f"  Size:     {file_size_mb:.1f} MB")
    print("=" * 70)

if __name__ == '__main__':
    main()

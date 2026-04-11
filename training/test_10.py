"""
Randomly sample 5 positive and 5 negative test cases,
generate AI report, and print side-by-side with human report.
"""

import os
import torch
import warnings
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms

from inference import CXRToTextModel, BART_MODEL_NAME, CV_MODEL_DIR, NLP_MODEL_DIR, DEVICE, IMAGE_SIZE
from transformers import BartTokenizer

warnings.filterwarnings("ignore")

def main():
    print("Loading AI Models...")
    tokenizer = BartTokenizer.from_pretrained(BART_MODEL_NAME)
    model = CXRToTextModel(
        CV_MODEL_DIR / "best_model.pth", 
        NLP_MODEL_DIR / "best_model.pth"
    )
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df = pd.pd = pd.read_csv("data/output/multi_label_dataset/ml_test.csv")
    
    # Grab 10 positive and 10 negative
    pos_df = df[df['Pneumonia'] == 1.0].sample(10, random_state=42)
    neg_df = df[df['Pneumonia'] == 0.0].sample(10, random_state=42)
    
    cases = []
    for _, row in pos_df.iterrows():
        cases.append(("POSITIVE (Pneumonia)", row))
    for _, row in neg_df.iterrows():
        cases.append(("NEGATIVE (Clean/Other)", row))
        
    for label, row in cases:
        img_id = row['dicom_id']
        
        # Build image path
        path_pos = Path(f"data/images/images/Test/positive/{img_id}.jpg")
        path_neg = Path(f"data/images/images/Test/negative/{img_id}.jpg")
        
        img_path = path_pos if path_pos.exists() else path_neg
        if not img_path.exists():
            continue
            
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.autocast("cuda"):
            pred = model.generate_report(img_t, tokenizer)[0]
            
        true_f = str(row['findings_text']) if pd.notna(row['findings_text']) else ""
        true_i = str(row['impression_text']) if pd.notna(row['impression_text']) else ""
        
        print(f"\n{'='*70}")
        print(f"CASE TYPE   : {label}")
        print(f"IMAGE ID    : {img_id}")
        print("-" * 70)
        print(f"[TRUE HUMAN]: FINDINGS: {true_f}  IMPRESSION: {true_i}")
        print("-" * 70)
        print(f"[AI CALLED] : {pred}")

if __name__ == "__main__":
    main()

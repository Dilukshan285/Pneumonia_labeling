"""
PP1 Task D: Test Set Evaluation (ROUGE & BLEU)
================================================================
Pass the final frozen model against the untouched 2,227 test images
to officially calculate Linguistic Accuracy (ROUGE-1, ROUGE-L).
"""

import os
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_ROOT = DATA_DIR / "images" / "images"
CV_MODEL_DIR = PROJECT_ROOT / "models" / "multilabel_convnext"
NLP_MODEL_DIR = PROJECT_ROOT / "models" / "biobart_report"
BART_MODEL_NAME = "GanjinZero/biobart-base"
TEST_CSV = DATA_DIR / "output" / "multi_label_dataset" / "ml_test.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 16  # Inference uses less VRAM, we can batch

# ============================================================================
# DATASET AND MODEL (reused architecture)
# ============================================================================
class EvaluationDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.transform = transform
        
        # Build path lookup 
        self.image_paths = {}
        # Test images are all in the 'Test' folder
        for cls_dir in ["positive", "negative"]:
            dir_path = self.image_root / "Test" / cls_dir
            if dir_path.exists():
                for fname in os.listdir(dir_path):
                    if fname.endswith(".jpg"):
                        self.image_paths[fname.replace(".jpg", "")] = str(dir_path / fname)

        # Filter
        valid_mask = self.df["dicom_id"].isin(self.image_paths)
        self.df = self.df[valid_mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Image
        img_path = self.image_paths[row["dicom_id"]]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        # Target Text
        findings = str(row["findings_text"]) if pd.notna(row["findings_text"]) else ""
        impression = str(row["impression_text"]) if pd.notna(row["impression_text"]) else ""

        if findings and impression:
            text = f"FINDINGS: {findings} IMPRESSION: {impression}"
        elif impression:
            text = f"IMPRESSION: {impression}"
        elif findings:
            text = f"FINDINGS: {findings}"
        else:
            text = ""
            
        return image, text

class CXRToTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Vision Encoder
        self.vision_encoder = models.convnext_base(weights=None)
        self.vision_encoder.avgpool = nn.Identity()
        self.vision_encoder.classifier = nn.Identity()
        
        # The Bridge
        self.bridge = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Text Decoder
        self.text_model = BartForConditionalGeneration.from_pretrained(BART_MODEL_NAME)
        
        # Load custom weights (The final full checkpoint from Phase B)
        # Note: We must unpack the full state dict for the pipeline
        full_ckpt = torch.load(NLP_MODEL_DIR / "best_model.pth", map_location=DEVICE, weights_only=False)
        self.load_state_dict(full_ckpt["model_state_dict"])
        self.eval()

    @torch.no_grad()
    def _encode_images(self, images):
        features = self.vision_encoder(images)
        B, C, H, W = features.shape
        features = features.view(B, C, H * W).permute(0, 2, 1)
        hidden_states = self.bridge(features)
        attention_mask = torch.ones(B, H * W, dtype=torch.long, device=images.device)
        return hidden_states, attention_mask

    @torch.no_grad()
    def generate_report(self, images, tokenizer, max_length=256):
        hidden_states, enc_attention_mask = self._encode_images(images)
        encoder_outputs = BaseModelOutput(last_hidden_state=hidden_states)

        generated_ids = self.text_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_attention_mask,
            max_length=max_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


# ============================================================================
# EVALUATION LOOP
# ============================================================================
def main():
    print("=" * 70)
    print("Evaluating BioBART Multi-Modal Model on TEST SET")
    print("=" * 70)

    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

    # 1. Load Data
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = EvaluationDataset(TEST_CSV, IMAGE_ROOT, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"[1/3] Test dataset loaded: {len(test_dataset)} images.")

    # 2. Load Model
    print("[2/3] Loading fully trained multi-modal architecture...")
    tokenizer = BartTokenizer.from_pretrained(BART_MODEL_NAME)
    model = CXRToTextModel()
    model.to(DEVICE)
    model.eval()

    # 3. Setup Metrics
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4
    
    all_rouge1 = []
    all_rougeL = []
    all_bleu = []

    print("[3/3] Generating reports and calculating scores...")
    pbar = tqdm(test_loader, desc="Testing", unit="batch")
    
    for images, true_texts in pbar:
        images = images.to(DEVICE)
        
        # Batch Generate
        with torch.autocast("cuda"):
            pred_texts = model.generate_report(images, tokenizer)
        
        # Calculate Scores
        for pred, true in zip(pred_texts, true_texts):
            pred = pred.strip()
            true = true.strip()
            
            if not true:
                continue
                
            # ROUGE
            scores = scorer.score(true, pred)
            all_rouge1.append(scores['rouge1'].fmeasure)
            all_rougeL.append(scores['rougeL'].fmeasure)
            
            # BLEU
            ref_tokens = true.split()
            pred_tokens = pred.split()
            b_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
            all_bleu.append(b_score)

    # 4. Final Output
    avg_r1 = sum(all_rouge1) / max(len(all_rouge1), 1)
    avg_rl = sum(all_rougeL) / max(len(all_rougeL), 1)
    avg_bl = sum(all_bleu) / max(len(all_bleu), 1)

    print("\n" + "=" * 70)
    print("FINAL TEST SET SCORES (2,227 Images)")
    print("=" * 70)
    print(f"  ROUGE-1 F1:   {avg_r1:.4f}  (Measures exact word overlap)")
    print(f"  ROUGE-L F1:   {avg_rl:.4f}  (Measures sentence structure/flow)")
    print(f"  BLEU Score:   {avg_bl:.4f}  (Measures linguistic precision)")
    print("-" * 70)
    print("* Note: In MIMIC-CXR studies, SOTA ROUGE-L is typically ~0.30 - 0.40")

if __name__ == "__main__":
    main()

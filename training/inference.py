"""
PP1 Task C: Single Image Inference
================================================================
Pass any unseen X-Ray image into the script to generate a clinical report.
Usage: python scripts/inference.py --image <path_to_jpg>
"""

import os
import argparse
import ast
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CV_MODEL_DIR = PROJECT_ROOT / "models" / "multilabel_convnext"
NLP_MODEL_DIR = PROJECT_ROOT / "models" / "biobart_report"
BART_MODEL_NAME = "GanjinZero/biobart-base"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

# Reconstruct the architectural bridge
class CXRToTextModel(nn.Module):
    def __init__(self, convnext_path, bart_path):
        super().__init__()
        # Vision Encoder
        self.vision_encoder = models.convnext_base(weights=None)
        self.vision_encoder.avgpool = nn.Identity()
        self.vision_encoder.classifier = nn.Identity()
        
        # Load ConvNeXt weights
        ckpt = torch.load(convnext_path, map_location=DEVICE, weights_only=False)
        filtered_dict = {
            k.replace("backbone.", ""): v 
            for k, v in ckpt["model_state_dict"].items() 
            if "backbone.features" in k
        }
        self.vision_encoder.load_state_dict(filtered_dict, strict=False)
        self.vision_encoder.eval()

        # The Bridge
        self.bridge = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Text Decoder
        self.text_model = BartForConditionalGeneration.from_pretrained(BART_MODEL_NAME)
        
        # We must load the full bridge+BART weights from our Phase B training
        full_ckpt = torch.load(bart_path, map_location=DEVICE, weights_only=False)
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

def main():
    parser = argparse.ArgumentParser(description="Generate CXR report from an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the JPG X-ray image")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[!] Error: Image not found at {image_path}")
        return

    print("=" * 60)
    print("MIMIC-CXR AI Report Generator")
    print("=" * 60)
    
    print("[1/3] Loading AI Models...")
    tokenizer = BartTokenizer.from_pretrained(BART_MODEL_NAME)
    model = CXRToTextModel(
        CV_MODEL_DIR / "best_model.pth", 
        NLP_MODEL_DIR / "best_model.pth"
    )
    model.to(DEVICE)

    print("[2/3] Processing Image...")
    # Standard transform matching the training pipeline
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    print("[3/3] Generating Report...\n")
    print("-" * 60)
    
    with torch.autocast("cuda"):
        report = model.generate_report(img_t, tokenizer)[0]
    
    # Format text cleanly (replace mass spaces)
    report = " ".join(report.split())
    
    print(report)
    print("-" * 60)

if __name__ == "__main__":
    main()

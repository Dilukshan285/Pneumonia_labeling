"""
Step P1 — Verify GPU Detection
Confirms that PyTorch can see the RTX 4060 GPU via CUDA.
This MUST pass before running Stage 2 LF5 (NLI zero-shot classification).
"""

import sys


def main():
    print("=" * 70)
    print("STEP P1 — VERIFY GPU DETECTION")
    print("=" * 70)
    print()

    try:
        import torch
        print(f"  PyTorch version:    {torch.__version__}")
        print(f"  CUDA available:     {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA version:       {torch.version.cuda}")
            print(f"  GPU device name:    {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory (total): {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print()
            print("  [OK] GPU is detected and ready for Stage 2 LF5.")
        else:
            print()
            print("  [FAIL] CUDA is NOT available.")
            print("         Stage 2 LF5 will fail or run extremely slowly on CPU.")
            print("         Check your CUDA installation and PyTorch build.")
            return 1
    except ImportError:
        print("  [FAIL] PyTorch is not installed.")
        print("         Run: pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121")
        return 1

    print()

    # Also verify spaCy + negspacy
    try:
        import spacy
        print(f"  spaCy version:      {spacy.__version__}")
        nlp = spacy.load("en_core_web_sm")
        print(f"  en_core_web_sm:     [OK] loaded successfully")
    except ImportError:
        print("  [FAIL] spaCy is not installed.")
        return 1
    except OSError:
        print("  [FAIL] en_core_web_sm model not found.")
        print("         Run: python -m spacy download en_core_web_sm")
        return 1

    try:
        import negspacy
        print(f"  negspaCy:           [OK] installed")
    except ImportError:
        print("  [FAIL] negspaCy is not installed.")
        return 1

    try:
        import snorkel
        print(f"  Snorkel version:    {snorkel.__version__}")
    except ImportError:
        print("  [FAIL] Snorkel is not installed.")
        return 1

    try:
        import transformers
        print(f"  Transformers:       {transformers.__version__}")
    except ImportError:
        print("  [FAIL] Transformers is not installed.")
        return 1

    print()
    print("=" * 70)
    print("STEP P1 COMPLETE — All dependencies verified.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

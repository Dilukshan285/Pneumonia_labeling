"""
Shared Configuration for Pneumonia Labeling Pipeline
All paths and constants used across all pipeline stages.
"""

import os

# ============================================================================
# BASE PATHS
# ============================================================================

# Project root directory
PROJECT_DIR = r"C:\Users\dviya\Desktop\Pneumonia_labeling"

# MIMIC-CXR reports directory (contains p10/, p11/, ... p19/ subfolders)
REPORTS_DIR = r"C:\Users\dviya\Downloads\mimic-cxr-reports\files"

# ============================================================================
# DATA DIRECTORIES
# ============================================================================

DATA_DIR = os.path.join(PROJECT_DIR, "data")
DATA_RAW = os.path.join(DATA_DIR, "raw")
DATA_INTERMEDIATE = os.path.join(DATA_DIR, "intermediate")
DATA_OUTPUT = os.path.join(DATA_DIR, "output")

# ============================================================================
# RAW INPUT FILES (from MIMIC-CXR download)
# ============================================================================

SPLIT_CSV = os.path.join(DATA_RAW, "mimic-cxr-2.0.0-split.csv")
CHEXPERT_CSV = os.path.join(DATA_RAW, "mimic-cxr-2.0.0-chexpert.csv")
METADATA_CSV = os.path.join(DATA_RAW, "mimic-cxr-2.0.0-metadata.csv")

# ============================================================================
# INTERMEDIATE OUTPUT FILES
# ============================================================================

# Step P3 output
MASTER_REPORTS_CSV = os.path.join(DATA_INTERMEDIATE, "master_reports.csv")

# Stage 1 output
PARSED_REPORTS_CSV = os.path.join(DATA_INTERMEDIATE, "parsed_reports.csv")

# Stage 2 outputs
PREFILTER_NEGATIVES_CSV = os.path.join(DATA_INTERMEDIATE, "prefilter_negatives.csv")
SNORKEL_SOFT_SCORES_CSV = os.path.join(DATA_INTERMEDIATE, "snorkel_soft_scores.csv")

# Stage 3 outputs
CONFIDENT_POOL_CSV = os.path.join(DATA_INTERMEDIATE, "confident_pool.csv")
UNCERTAIN_POOL_CSV = os.path.join(DATA_INTERMEDIATE, "uncertain_pool.csv")
THRESHOLD_SENSITIVITY_CSV = os.path.join(DATA_INTERMEDIATE, "threshold_sensitivity.csv")

# ============================================================================
# FINAL OUTPUT FILES
# ============================================================================

ACTIVE_LEARNING_QUEUE_CSV = os.path.join(DATA_OUTPUT, "active_learning_queue.csv")
FINAL_LABELS_CSV = os.path.join(DATA_OUTPUT, "final_pneumonia_labels.csv")
TRAINING_DATASET_CSV = os.path.join(DATA_OUTPUT, "training_dataset.csv")
PP2_SEQUENCES_JSON = os.path.join(DATA_OUTPUT, "pp2_sequences.json")
MANUAL_VALIDATION_CSV = os.path.join(DATA_OUTPUT, "manual_validation_labels.csv")
COHEN_KAPPA_CSV = os.path.join(DATA_OUTPUT, "cohen_kappa_validation.csv")

# ============================================================================
# LOGS DIRECTORY
# ============================================================================

LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
P4_INSPECTION_REPORT = os.path.join(LOGS_DIR, "p4_inspection_report.txt")

# ============================================================================
# PIPELINE CONSTANTS
# ============================================================================

RANDOM_SEED = 42

# Stage 2 — LF5 NLI model
NLI_MODEL_NAME = "facebook/bart-large-mnli"
NLI_BATCH_SIZE = 32         # fp16 halves VRAM: ~800MB model + ~5GB batches = safe on 8GB; OOM auto-reduces
NLI_MAX_TOKENS = 1024       # BART context limit
NLI_CONFIDENCE_THRESHOLD = 0.40  # Below this → ABSTAIN

# Stage 3 — Confidence thresholds
POSITIVE_THRESHOLD = 0.75   # soft_score >= this → POSITIVE
NEGATIVE_THRESHOLD = 0.25   # soft_score <= this → NEGATIVE

# Stage 4 — Active learning
ACTIVE_LEARNING_COUNT = 200  # Number of most uncertain reports to label

# Stage 5 — Validation
VALIDATION_SAMPLE_SIZE = 300  # Number of reports to sample for Kappa

# Snorkel LabelModel training parameters
SNORKEL_CARDINALITY = 3     # NEGATIVE=0, POSITIVE=1, UNCERTAIN=2
SNORKEL_EPOCHS = 500
SNORKEL_LR = 0.01
SNORKEL_OPTIMIZER = "adam"

# Label encoding
LABEL_NEGATIVE = 0
LABEL_POSITIVE = 1
LABEL_UNCERTAIN = 2
LABEL_ABSTAIN = -1

# Manual label soft_score overrides
MANUAL_POSITIVE_SCORE = 0.95
MANUAL_NEGATIVE_SCORE = 0.05
MANUAL_UNCERTAIN_SCORE = 0.50

# Pre-filter soft_score for reports with no pneumonia/lung terms
PREFILTER_NEGATIVE_SCORE = 0.02

"""
Multi-Label Classification Configuration
=========================================
Defines 14 CXR pathology classes with:
  - Clinical keyword lists (positive & negative triggers)
  - NLI hypothesis templates for BART-MNLI
  - Assertion detection patterns
  - Pipeline weights and thresholds
"""

import os

# ============================================================================
# PATHS
# ============================================================================

PROJECT_DIR = r"C:\Users\dviya\Desktop\Pneumonia_labeling"
DATA_OUTPUT = os.path.join(PROJECT_DIR, "data", "output")
ML_DATASET_DIR = os.path.join(DATA_OUTPUT, "multi_label_dataset")

# Source splits (from existing pipeline)
PP1_TRAIN_CSV = os.path.join(DATA_OUTPUT, "pp1_train.csv")
PP1_VAL_CSV   = os.path.join(DATA_OUTPUT, "pp1_val.csv")
PP1_TEST_CSV  = os.path.join(DATA_OUTPUT, "pp1_test.csv")

# Output multi-label CSVs
ML_TRAIN_CSV = os.path.join(ML_DATASET_DIR, "ml_train.csv")
ML_VAL_CSV   = os.path.join(ML_DATASET_DIR, "ml_val.csv")
ML_TEST_CSV  = os.path.join(ML_DATASET_DIR, "ml_test.csv")

# ============================================================================
# 14 PATHOLOGY CLASSES
# ============================================================================

PATHOLOGY_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged_Cardiomediastinum",
    "Fracture",
    "Lung_Lesion",
    "Lung_Opacity",
    "No_Finding",
    "Pleural_Effusion",
    "Pleural_Other",
    "Pneumonia",
    "Pneumothorax",
    "Support_Devices",
]

NUM_CLASSES = len(PATHOLOGY_CLASSES)

# ============================================================================
# LAYER 1: KEYWORD DEFINITIONS
# Each entry: list of positive keywords that indicate the condition
# ============================================================================

POSITIVE_KEYWORDS = {
    "Atelectasis": [
        "atelectasis", "atelectatic", "volume loss", "collapse",
        "collapsed lung", "subsegmental atelectasis", "bibasilar atelectasis",
        "basilar atelectasis", "discoid atelectasis", "plate-like atelectasis",
        "platelike atelectasis", "compressive atelectasis",
        "left lower lobe collapse", "right lower lobe collapse",
        "lobar collapse", "lung collapse",
    ],
    "Cardiomegaly": [
        "cardiomegaly", "cardiac enlargement", "enlarged heart",
        "enlarged cardiac", "heart is enlarged", "heart size is enlarged",
        "cardiac silhouette is enlarged", "heart is moderately enlarged",
        "heart is mildly enlarged", "moderately enlarged heart",
        "mildly enlarged heart", "severely enlarged heart",
        "mild cardiomegaly", "moderate cardiomegaly", "severe cardiomegaly",
        "heart size is prominent", "heart size is top normal",
    ],
    "Consolidation": [
        "consolidation", "consolidative", "airspace consolidation",
        "focal consolidation", "lobar consolidation",
        "airspace opacity", "airspace opacification", "airspace disease",
        "air bronchogram", "air bronchograms",
    ],
    "Edema": [
        "edema", "pulmonary edema", "interstitial edema",
        "alveolar edema", "vascular congestion", "pulmonary congestion",
        "cephalization", "vascular engorgement", "vascular redistribution",
        "upper zone redistribution", "pulmonary venous congestion",
        "pulmonary venous hypertension", "fluid overload",
        "congestive heart failure", "chf", "congestive failure",
    ],
    "Enlarged_Cardiomediastinum": [
        "enlarged cardiomediastinum", "widened mediastinum",
        "mediastinal widening", "mediastinal enlargement",
        "widened mediastinal", "mediastinal shift",
        "mediastinal venous engorgement", "mediastinal vascular engorgement",
    ],
    "Fracture": [
        "fracture", "fractured", "rib fracture", "rib fractures",
        "displaced fracture", "nondisplaced fracture",
        "compression fracture", "vertebral fracture",
        "sternal fracture", "healed fracture", "healing fracture",
        "old fracture", "acute fracture", "pathologic fracture",
    ],
    "Lung_Lesion": [
        "lung lesion", "lung mass", "pulmonary mass",
        "pulmonary nodule", "lung nodule", "nodular opacity",
        "mass lesion", "spiculated", "cavitary lesion",
        "cavitary mass", "cavitation", "pulmonary metastasis",
        "pulmonary metastases", "metastatic disease", "metastases",
    ],
    "Lung_Opacity": [
        "opacity", "opacification", "opacities",
        "hazy opacity", "hazy opacification", "haziness",
        "infiltrate", "infiltrates", "reticular opacity",
        "reticular opacities", "interstitial opacity",
        "interstitial opacities", "parenchymal opacity",
        "ground glass", "ground-glass",
        "airspace opacities", "focal opacity",
    ],
    "No_Finding": [
        "no acute cardiopulmonary process",
        "no acute cardiopulmonary abnormality",
        "no acute intrathoracic process",
        "no acute disease", "lungs are clear",
        "clear lungs", "unremarkable chest",
        "normal chest", "normal chest radiograph",
        "no acute findings", "no significant abnormality",
        "no evidence of acute cardiopulmonary disease",
        "no evidence of acute disease",
        "no acute process",
    ],
    "Pleural_Effusion": [
        "pleural effusion", "effusion", "effusions",
        "pleural fluid", "layering effusion",
        "bilateral effusions", "bilateral pleural effusions",
        "right effusion", "left effusion",
        "small effusion", "moderate effusion", "large effusion",
        "hydrothorax", "costophrenic angle blunting",
        "blunting of the costophrenic",
        "meniscus sign",
    ],
    "Pleural_Other": [
        "pleural thickening", "pleural calcification",
        "pleural plaque", "pleural plaques",
        "pleural scarring", "pleural abnormality",
        "fibrothorax", "empyema",
        "loculated effusion", "loculated fluid",
        "loculated pleural",
    ],
    "Pneumonia": [
        "pneumonia", "pneumonic", "infectious process",
        "infectious consolidation", "infection",
        "bronchopneumonia", "aspiration pneumonia",
        "community-acquired pneumonia", "hospital-acquired pneumonia",
        "lobar pneumonia", "multilobar pneumonia",
        "bilateral pneumonia", "right lower lobe pneumonia",
        "superimposed pneumonia", "developing pneumonia",
        "evolving pneumonia", "resolving pneumonia",
        "suspected pneumonia",
    ],
    "Pneumothorax": [
        "pneumothorax", "pneumothoraces",
        "tension pneumothorax", "hydropneumothorax",
        "small pneumothorax", "large pneumothorax",
        "right pneumothorax", "left pneumothorax",
        "apical pneumothorax",
    ],
    "Support_Devices": [
        "endotracheal tube", "et tube", "tracheostomy",
        "central line", "central venous catheter", "central venous line",
        "picc", "picc line", "port-a-cath", "port catheter",
        "swan-ganz", "pacemaker", "pacer", "defibrillator", "icd",
        "chest tube", "pleural drain", "pigtail catheter",
        "nasogastric tube", "ng tube", "orogastric tube",
        "dobbhoff", "dobhoff", "feeding tube", "enteric tube",
        "sternal wires", "sternotomy wires", "mediastinal clips",
        "surgical clips", "drain", "drainage catheter",
        "monitoring device", "support device",
    ],
}

# Keywords that indicate a condition is being NEGATED
# (supplements negspaCy detection for tricky patterns)
NEGATION_CUES = [
    "no ", "no evidence of", "without", "not ", "negative for",
    "absence of", "ruled out", "rules out", "excluding",
    "has resolved", "has cleared", "has improved",
    "resolved", "cleared", "improved",
    "removed", "was removed", "has been removed",
    "prior", "old", "previous", "history of",
    "cannot be excluded",  # treated as UNCERTAIN, not positive
    "cannot be ruled out",  # treated as UNCERTAIN, not positive
]

UNCERTAINTY_CUES = [
    "possible", "possibly", "probable", "probably",
    "suspected", "suggesting", "suggestive",
    "may represent", "might represent", "could represent",
    "may reflect", "might reflect", "could reflect",
    "may be", "might be", "could be",
    "cannot be excluded", "cannot be ruled out",
    "not excluded", "not be excluded",
    "is not excluded", "differential",
    "questionable", "equivocal", "indeterminate",
    "uncertain", "versus", " vs ",
    "consider", "would have to be considered",
    "clinical correlation",
]

# ============================================================================
# LAYER 2: NLI HYPOTHESES (for BART-MNLI zero-shot)
# ============================================================================

NLI_HYPOTHESES = {
    "Atelectasis":                "This chest X-ray shows atelectasis or lung collapse",
    "Cardiomegaly":               "This chest X-ray shows cardiomegaly or enlarged heart",
    "Consolidation":              "This chest X-ray shows consolidation or airspace disease",
    "Edema":                      "This chest X-ray shows pulmonary edema or fluid overload",
    "Enlarged_Cardiomediastinum":  "This chest X-ray shows enlarged cardiomediastinum or widened mediastinum",
    "Fracture":                   "This chest X-ray shows a fracture",
    "Lung_Lesion":                "This chest X-ray shows a lung lesion, mass, or nodule",
    "Lung_Opacity":               "This chest X-ray shows a lung opacity or infiltrate",
    "No_Finding":                 "This chest X-ray is normal with no acute findings",
    "Pleural_Effusion":           "This chest X-ray shows pleural effusion",
    "Pleural_Other":              "This chest X-ray shows pleural abnormality such as thickening or calcification",
    "Pneumonia":                  "This chest X-ray shows pneumonia or lung infection",
    "Pneumothorax":               "This chest X-ray shows pneumothorax",
    "Support_Devices":            "This chest X-ray shows support devices such as tubes, lines, or catheters",
}

NLI_CONTRADICTION_HYPOTHESES = {
    "Atelectasis":                "No atelectasis or lung collapse is present",
    "Cardiomegaly":               "Heart size is normal, no cardiomegaly",
    "Consolidation":              "No consolidation or airspace disease is seen",
    "Edema":                      "No pulmonary edema or fluid overload",
    "Enlarged_Cardiomediastinum":  "The mediastinum is normal in width",
    "Fracture":                   "No fracture is identified",
    "Lung_Lesion":                "No lung mass, lesion, or nodule is present",
    "Lung_Opacity":               "The lungs are clear without opacity",
    "No_Finding":                 "There are abnormal findings on this chest X-ray",
    "Pleural_Effusion":           "No pleural effusion is present",
    "Pleural_Other":              "No pleural abnormality is present",
    "Pneumonia":                  "No pneumonia or lung infection is present",
    "Pneumothorax":               "No pneumothorax is present",
    "Support_Devices":            "No support devices, tubes, or lines are present",
}

# ============================================================================
# LAYER 3: ASSERTION PATTERNS
# ============================================================================

# Patterns that indicate definite PRESENCE
ASSERTION_PRESENT_PATTERNS = [
    r"\b(?:is|are|has|have|shows?|demonstrates?|reveals?|indicates?)\b.*\b{kw}\b",
    r"\b{kw}\b.*\b(?:is|are)\b.*\b(?:seen|noted|identified|present|demonstrated|visualized)\b",
    r"\b(?:new|worsening|increasing|progressive|developing|evolving|interval)\b.*\b{kw}\b",
    r"\b{kw}\b.*\b(?:worse|worsened|increased|progressed)\b",
    r"\b(?:consistent with|compatible with|representing|suggestive of)\b.*\b{kw}\b",
]

# Patterns that indicate definite ABSENCE  
ASSERTION_ABSENT_PATTERNS = [
    r"\bno\b.*\b{kw}\b",
    r"\b(?:without|absence of)\b.*\b{kw}\b",
    r"\b{kw}\b.*\b(?:has|have)?\s*(?:resolved|cleared|improved|removed)\b",
    r"\b{kw}\b.*\bnot\b.*\b(?:seen|identified|present|demonstrated|visualized)\b",
    r"\b(?:no evidence of|negative for|rules? out)\b.*\b{kw}\b",
]

# Patterns that indicate UNCERTAINTY
ASSERTION_UNCERTAIN_PATTERNS = [
    r"\b(?:possible|probable|suspected|questionable|equivocal)\b.*\b{kw}\b",
    r"\b{kw}\b.*\b(?:cannot be excluded|cannot be ruled out|not excluded)\b",
    r"\b(?:may|might|could)\b.*\b(?:represent|reflect|be)\b.*\b{kw}\b",
    r"\b(?:versus|vs\.?|or)\b.*\b{kw}\b",
    r"\b{kw}\b.*\b(?:versus|vs\.?|or)\b",
]

# ============================================================================
# CONSENSUS WEIGHTS
# ============================================================================

LAYER_WEIGHTS = {
    "layer1_keywords":   1.0,    # Basic keyword matching
    "layer2_nli":        1.5,    # Semantic NLI understanding (highest)
    "layer3_assertions": 1.2,    # Sentence-level assertion detection
}

# ============================================================================
# PROCESSING SETTINGS
# ============================================================================

NLI_MODEL_NAME = "MoritzLaurer/deberta-v3-large-mnli-fever-anli-ling-wanli"
NLI_BATCH_SIZE = 16          # Conservative for 14 conditions x batch
NLI_MAX_TOKENS = 512         # Truncate long reports

# Final label encoding
LABEL_PRESENT   =  1
LABEL_ABSENT    =  0
LABEL_UNCERTAIN = -1

RANDOM_SEED = 42

"""
Steps 2.1, 2.2, 2.3 — Keyword Lists for Pneumonia Labeling (v3.1 CORRECTED)

Three keyword lists:
  POSITIVE — pneumonia confirmed OR strongly suspected (radiologist would treat)
  NEGATIVE — pneumonia explicitly absent (radiologist would NOT treat)
  EXCLUDE  — genuinely ambiguous (excluded from training entirely)

v3.1 CORRECTIONS from v3.0:
  - MOVED strong suspicion terms BACK to POSITIVE (they were wrongly in NEGATIVE)
  - MOVED genuinely ambiguous terms to new EXCLUDE list (not forced into NEGATIVE)
  - MOVED "superimposed/resolving/improving pneumonia" to POSITIVE (pneumonia IS present)
  - MOVED bare unnegated "consolidation to suggest pneumonia" etc. to POSITIVE
  - ADDED negated suspicion phrases to NEGATIVE ("not consistent with pneumonia" etc.)
  - ADDED plural gaps to NEGATIVE ("no parenchymal opacities" etc.)
  - REMOVED sentence-level negation check (caused false negatives)
"""

KEYWORD_LIST_VERSION = "v3.1"

# ============================================================================
# STEP 2.1 — POSITIVE KEYWORDS
# Pneumonia IS present or strongly suspected. Radiologist would treat.
# ============================================================================

POSITIVE_KEYWORDS = [
    # ---- Consolidation variants ----
    "lobar consolidation",
    "focal consolidation",
    "multilobar consolidation",
    "multifocal consolidation",
    "infectious consolidation",
    "bilateral consolidation",
    "patchy consolidation",
    "dense consolidation",
    "segmental consolidation",
    "airspace consolidation",

    # ---- Opacity variants ----
    "patchy opacity",
    "airspace opacity",
    "alveolar opacity",
    "pulmonary opacity",
    "pneumonic opacity",
    "parenchymal opacity",
    "focal opacity",
    "hazy opacity",
    "retrocardiac opacity",
    "basilar opacity",
    "bilateral opacities",
    "diffuse opacity",
    "perihilar opacity",
    "confluent opacity",

    # ---- Lobe-specific opacity ----
    "right lower lobe opacity",
    "right lower lobe opacities",
    "left lower lobe opacity",
    "left lower lobe opacities",
    "right upper lobe opacity",
    "right upper lobe opacities",
    "left upper lobe opacity",
    "left upper lobe opacities",
    "right middle lobe opacity",
    "right middle lobe opacities",
    "lower lobe opacity",
    "lower lobe opacities",
    "upper lobe opacity",
    "upper lobe opacities",
    "middle lobe opacity",
    "middle lobe opacities",
    "right basilar opacity",
    "right basilar opacities",
    "left basilar opacity",
    "left basilar opacities",
    "right base opacity",
    "right base opacities",
    "left base opacity",
    "left base opacities",
    "right lung opacity",
    "right lung opacities",
    "left lung opacity",
    "left lung opacities",
    "lingular opacity",
    "lingular opacities",
    "perihilar opacities",
    "new focal opacity",
    "new focal opacities",

    # ---- Infiltrate variants ----
    "perihilar infiltrate",
    "pulmonary infiltrate",
    "bilateral infiltrates",
    "patchy infiltrate",
    "interstitial infiltrate",

    # ---- Disease/process ----
    "airspace disease",
    "focal airspace disease",
    "air bronchogram",

    # ---- Lobe-specific pneumonia ----
    "right lower lobe pneumonia",
    "left lower lobe pneumonia",
    "right middle lobe pneumonia",
    "right upper lobe pneumonia",
    "left upper lobe pneumonia",
    "multifocal pneumonia",
    "bilateral pneumonia",

    # ---- Pneumonia subtypes (confirmed present) ----
    "aspiration pneumonia",
    "worsening pneumonia",
    "persistent pneumonia",
    "recurrent pneumonia",
    "developing pneumonia",
    "early pneumonia",
    "evolving pneumonia",
    "atypical pneumonia",
    "favor pneumonia",
    "favoring pneumonia",

    # ---- Pneumonia still present (improving but NOT gone) ----
    "resolving pneumonia",
    "improving pneumonia",
    "superimposed pneumonia",

    # ---- Strong radiologist suspicion (clinically actionable = POSITIVE) ----
    "consistent with pneumonia",
    "compatible with pneumonia",
    "suggestive of pneumonia",
    "suspicious for pneumonia",
    "worrisome for pneumonia",
    "concerning for pneumonia",
    "representing pneumonia",
    "likely pneumonia",
    "probable pneumonia",

    # ---- Finding + suspicion phrases (pneumonia suspected based on finding) ----
    "consolidation concerning for pneumonia",
    "consolidation worrisome for pneumonia",
    "consolidation to suggest pneumonia",
    "opacity concerning for pneumonia",
    "opacities concerning for pneumonia",
    "opacity to suggest pneumonia",
    "opacities to suggest pneumonia",

    # ---- Single-word / short terms (least specific last) ----
    "bronchopneumonia",
    "pneumonitis",
    "pneumonia",
    "consolidation",
    "infiltrate",
    "opacification",
]

# ============================================================================
# STEP 2.2 — NEGATIVE KEYWORDS
# Pneumonia is explicitly ABSENT. Only patterns with clear negation.
# ============================================================================

NEGATIVE_KEYWORDS = [
    # ---- Explicit pneumonia negation ----
    "no evidence of pneumonia",
    "no evidence of acute pneumonia",
    "no radiographic evidence of pneumonia",
    "no radiographic evidence for pneumonia",
    "no convincing evidence for pneumonia",
    "no convincing signs of pneumonia",
    "no signs of pneumonia",
    "no definite acute focal pneumonia",
    "no acute focal pneumonia",
    "no acute pneumonia",
    "no pneumonic process",
    "without evidence of pneumonia",
    "without pneumonia",
    "no pneumonia",

    # ---- Resolved = pneumonia WAS present, now GONE ----
    "resolved pneumonia",
    "resolved lingular pneumonia",
    "resolved right lower lobe pneumonia",
    "resolved left lower lobe pneumonia",
    "resolved right upper lobe pneumonia",
    "resolved left upper lobe pneumonia",
    "resolved right middle lobe pneumonia",
    "resolved bilateral pneumonia",
    "resolved multifocal pneumonia",
    "resolved aspiration pneumonia",
    "resolved bronchopneumonia",
    "previously noted pneumonia has resolved",
    "pneumonia has resolved",
    "pneumonia has cleared",

    # ---- Negated consolidation/infiltrate ----
    "no evidence of consolidation",
    "no focal consolidation",
    "no focal airspace consolidation",
    "no focal airspace disease",
    "no airspace consolidation",
    "no airspace disease",
    "no consolidation",
    "without consolidation",
    "no infiltrate",
    "no infiltrates",
    "no acute infiltrate",
    "no pulmonary infiltrate",
    "without infiltrate",
    "without infiltrates",

    # ---- Negated opacity ----
    "no opacity",
    "no opacities",
    "no focal opacity",
    "no focal opacities",
    "no focal airspace",
    "no parenchymal opacity",
    "no parenchymal opacities",
    "no parenchymal abnormality",
    "no new parenchymal opacities",
    "no new opacities",

    # ---- Bare negation of single-word positive terms ----
    "no opacification",
    "no bronchopneumonia",
    "no pneumonitis",
    "without opacification",
    "without bronchopneumonia",
    "without pneumonitis",

    # ---- "without [multi-word positive]" patterns ----
    "without focal consolidation",
    "without airspace consolidation",
    "without focal airspace consolidation",
    "without airspace disease",
    "without focal airspace disease",
    "without focal opacity",
    "without parenchymal opacity",
    "without parenchymal opacities",
    "without evidence of consolidation",
    "without focal airspace",

    # ---- "no definite [X]" patterns ----
    "no definite pneumonia",
    "no definite consolidation",
    "no definite focal consolidation",
    "no definite opacity",
    "no definite infiltrate",
    "no definite focal opacity",
    "no definite airspace consolidation",
    "no definite airspace disease",
    "without definite consolidation",
    "without definite focal consolidation",
    "without definite pneumonia",
    "without definite opacity",

    # ---- "no obvious [X]" patterns ----
    "no obvious pneumonia",
    "no obvious consolidation",
    "no obvious focal consolidation",
    "no obvious infiltrate",

    # ---- General negative findings ----
    "no acute cardiopulmonary process",
    "no acute cardiopulmonary abnormality",
    "no acute cardiopulmonary",
    "no acute pulmonary process",
    "no acute pulmonary abnormality",
    "no acute process",
    "no acute findings",
    "no acute abnormality",
    "no active disease",

    # ---- Clear lungs ----
    "lungs are clear",
    "lungs remain clear",
    "lungs are well expanded",
    "lungs are well aerated",
    "lung fields are clear",
    "clear lung fields",
    "clear lungs",
    "unremarkable lungs",
    "normal lung parenchyma",

    # ---- Negated "to suggest / concerning for / worrisome for" phrases ----
    "no consolidation to suggest pneumonia",
    "no focal consolidation to suggest pneumonia",
    "no evidence of consolidation to suggest pneumonia",
    "no airspace consolidation to suggest pneumonia",
    "no opacity to suggest pneumonia",
    "no focal opacity to suggest pneumonia",
    "no evidence of opacity to suggest pneumonia",
    "no opacities to suggest pneumonia",
    "no parenchymal opacities to suggest pneumonia",
    "no consolidation concerning for pneumonia",
    "no focal consolidation concerning for pneumonia",
    "no acute focal consolidation concerning for pneumonia",
    "no acute focal consolidations concerning for pneumonia",
    "no opacity concerning for pneumonia",
    "no focal opacity concerning for pneumonia",
    "no opacities concerning for pneumonia",
    "no new opacities concerning for pneumonia",
    "no opacification concerning for pneumonia",
    "no focal opacification concerning for pneumonia",
    "no consolidation worrisome for pneumonia",
    "no focal consolidation worrisome for pneumonia",
    "no opacity worrisome for pneumonia",

    # ---- Negated "suggestive/suspicious of/for" ----
    "no consolidation suggestive of pneumonia",
    "no opacity suggestive of pneumonia",
    "no opacities suggestive of pneumonia",
    "no parenchymal opacities suggestive of pneumonia",
    "no consolidation suspicious for pneumonia",
    "no opacity suspicious for pneumonia",
    "no opacities suspicious for pneumonia",
    "no parenchymal opacities suspicious for pneumonia",

    # ---- "without [X] to suggest/concerning for" ----
    "without consolidation to suggest pneumonia",
    "without focal consolidation to suggest pneumonia",
    "without evidence of consolidation to suggest pneumonia",
    "without consolidation concerning for pneumonia",
    "without focal consolidation concerning for pneumonia",
    "without opacity to suggest pneumonia",
    "without opacity concerning for pneumonia",

    # ---- Negated superimposed pneumonia ----
    "no superimposed pneumonia",
    "no evidence of superimposed pneumonia",
    "without superimposed pneumonia",
    "without evidence of superimposed pneumonia",
    "no superimposed infection or pneumonia",

    # ---- "no [X] or pneumonia" ----
    "no pneumothorax or pneumonia",
    "no chf or pneumonia",
    "no pulmonary edema or pneumonia",
    "no edema or pneumonia",
    "no evidence of congestive heart failure or pneumonia",
    "no evidence of decompensated congestive heart failure or pneumonia",
    "no convincing signs of aspiration or pneumonia",
    "no convincing signs of pulmonary edema or pneumonia",
    "no acute cardiopulmonary process or pneumonia",

    # ---- "no findings to suggest pneumonia" ----
    "no findings to suggest pneumonia",
    "no findings suggesting pneumonia",
    "no findings suggestive of pneumonia",
    "no radiographic findings to suggest pneumonia",
    "no radiographic findings suggesting pneumonia",
    "no radiographic findings suggestive of pneumonia",
    "no imaging findings to suggest pneumonia",
    "no imaging findings suggestive of pneumonia",
    "no findings to indicate pneumonia",
    "no findings indicative of pneumonia",
    "without findings to suggest pneumonia",
    "without findings suggestive of pneumonia",

    # ---- Negated suspicion phrases (catches "not consistent with pneumonia" etc.) ----
    "not consistent with pneumonia",
    "not compatible with pneumonia",
    "not suggestive of pneumonia",
    "not suspicious for pneumonia",
    "not worrisome for pneumonia",
    "not concerning for pneumonia",
    "not representing pneumonia",
    "not likely pneumonia",
    "unlikely pneumonia",
    "unlikely to represent pneumonia",
    "unlikely to be pneumonia",
    "does not suggest pneumonia",
    "does not represent pneumonia",
    "do not suggest pneumonia",
    "findings not consistent with pneumonia",
    "findings are not consistent with pneumonia",
    "not indicative of pneumonia",
]

# ============================================================================
# STEP 2.3 — EXCLUDE KEYWORDS
# Genuinely ambiguous — excluded from training entirely.
# NOT forced into POSITIVE or NEGATIVE.
# ============================================================================

EXCLUDE_KEYWORDS = [
    # ---- Genuine clinical uncertainty ----
    "cannot entirely exclude pneumonia",
    "cannot exclude pneumonia",
    "cannot rule out pneumonia",
    "pneumonia cannot be excluded",
    "pneumonia not excluded",
    "possible pneumonia",
    "possibly pneumonia",
    "questionable pneumonia",
    "suspected pneumonia",
    "may represent pneumonia",
    "could represent pneumonia",
    "might represent pneumonia",
    "could be pneumonia",
    "may be pneumonia",
    "concern for pneumonia",
    "rule out pneumonia",
    "underlying pneumonia",
    "versus pneumonia",

    # ---- Differential diagnosis (not confirmed either way) ----
    "atelectasis or pneumonia",
    "atelectasis and or pneumonia",
    "atelectasis and/or pneumonia",
    "aspiration or pneumonia",
    "edema or pneumonia",
    "pulmonary edema or pneumonia",
    "hemorrhage or pneumonia",
    "effusion or pneumonia",
    "infection or pneumonia",
    "collapse or pneumonia",
    "consolidation or pneumonia",

    # ---- Ambiguous radiological findings (could be atelectasis/edema/etc.) ----
    "bibasilar opacity",
    "bibasilar opacities",
    "new opacity",
    "new opacities",
    "increasing opacity",
    "increasing opacities",
    "worsening opacity",
    "worsening opacities",
    "new focal parenchymal opacities",
    "newly appeared parenchymal opacities",
    "developing consolidation",
    "evolving consolidation",
]

# ============================================================================
# COMBINED LIST — used by Step 2.0 pre-filter
# ============================================================================

ALL_KEYWORDS = POSITIVE_KEYWORDS + NEGATIVE_KEYWORDS + EXCLUDE_KEYWORDS

"""
================================================================================
LAYERS 5–7 v2 — WORLD-CLASS PROBABILISTIC META-ENSEMBLE CONSENSUS
================================================================================

Replaces the naive "ALL 4 must agree" gate with a probabilistic approach that:

1. SIGNAL FUSION: Uses continuous confidence scores from ALL systems as features
   instead of binary agree/disagree votes. Treats each system as a noisy sensor
   producing a continuous probability estimate.

2. META-LEARNER: Trains a logistic regression meta-classifier on ultra-strict
   seed labels (where ground truth is near-certain) to learn optimal system
   weights. This automatically discovers which systems are most reliable.

3. TEMPERATURE SCALING: Simple 2-parameter calibration (temp + bias) validated
   on held-out seeds. Does NOT overfit like isotonic regression on clean labels.

4. MULTI-TIER LABELS: Gold/Silver/Bronze quality tiers based on both ensemble
   probability AND cross-system agreement count, giving downstream training
   flexibility.

5. BALANCED OUTPUT: Intelligent stratified sampling to achieve clinically
   realistic 1:4 POS:NEG ratio while preserving the hardest negatives (those
   closest to the decision boundary) for robust model training.

6. REPORT TEXT: Includes impression_text and findings_text for each labeled
   study, enabling the multimodal PP2 pipeline.

Architecture:
   ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐
   │ Snorkel  │ │ DeBERTa  │ │ AdvBERT  │ │ GatorTron│ │ Assert  │
   │soft_score│ │model_b_sc│ │model_c_sc│ │l3_conf   │ │n_present│
   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬────┘
        │             │            │             │            │
        └─────┬───────┴────────────┴─────────────┴────┬───────┘
              │                                       │
              ▼                                       ▼
    ┌─────────────────────────────────────────────────────────┐
    │    LOGISTIC REGRESSION META-LEARNER                     │
    │    Trained on ultra-strict seed labels (N~46K)          │
    │    5-fold stratified cross-validation                   │
    │    Features: [snorkel, deberta, advbert, gator, assert] │
    └───────────────────┬─────────────────────────────────────┘
                        │
                        ▼  P(pneumonia | all signals)
    ┌───────────────────────────────────────────────────┐
    │    TEMPERATURE SCALING CALIBRATION                │
    │    P_cal = sigmoid((logit(P_raw) - bias) / T)    │
    │    Optimized on held-out seed validation set      │
    └───────────────────┬───────────────────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────────────────┐
    │    MULTI-TIER LABEL ASSIGNMENT                    │
    │    GOLD:   P ≥ 0.85 + 3/4 agree (or P ≥ 0.95)   │
    │    SILVER: P ≥ 0.70 + 2/4 agree                  │
    │    NEGATIVE analogous                             │
    │    EXCLUDED: everything else                      │
    └───────────────────┬───────────────────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────────────────┐
    │    ADVERSARIAL VALIDATION (CheXpert double-check) │
    │    → TIER upgrade if CheXpert agrees              │
    └───────────────────┬───────────────────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────────────────┐
    │    FINAL OUTPUT with report text                  │
    │    advanced_final_labels.csv  (all accepted)      │
    │    training_ready_labels.csv  (balanced 1:4)      │
    └───────────────────────────────────────────────────┘

Input:  All layer CSVs + parsed_reports.csv + CheXpert
Output: advanced_final_labels.csv (full) + training_ready_labels.csv (balanced)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from scipy.optimize import minimize
from scipy.special import expit, logit

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    accuracy_score, classification_report, log_loss,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_INTERMEDIATE, DATA_OUTPUT, PARSED_REPORTS_CSV,
    SNORKEL_SOFT_SCORES_CSV, CHEXPERT_CSV,
    LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN, LABEL_ABSTAIN,
    RANDOM_SEED,
)

# ============================================================================
# FILE PATHS
# ============================================================================
SEEDS_CSV     = os.path.join(DATA_INTERMEDIATE, "layer1_seeds.csv")
ENSEMBLE_CSV  = os.path.join(DATA_INTERMEDIATE, "layer2_nli_ensemble.csv")
PUBMEDBERT_CSV= os.path.join(DATA_INTERMEDIATE, "layer3_pubmedbert.csv")
ASSERTIONS_CSV= os.path.join(DATA_INTERMEDIATE, "layer4_assertions.csv")
PREFILTER_CSV = os.path.join(DATA_INTERMEDIATE, "prefilter_negatives.csv")

# Output
FINAL_CSV          = os.path.join(DATA_OUTPUT, "advanced_final_labels.csv")
TRAINING_READY_CSV = os.path.join(DATA_OUTPUT, "training_ready_labels.csv")

# ============================================================================
# CONSTANTS
# ============================================================================
L_EXCLUDED = 99
N_CV_FOLDS = 5
TARGET_POS_NEG_RATIO = 4   # 1:4 POS:NEG in balanced output

# Meta-learner feature columns
FEATURE_COLS = [
    'snorkel_score',        # Snorkel LabelModel soft probability
    'model_b_score',        # DeBERTa-v3 NLI confidence
    'model_c_score',        # Adversarial-prompt NLI confidence
    'l3_prob',              # GatorTron signed probability (neg=-conf, pos=+conf)
    'n_present_norm',       # Normalized PRESENT assertion count
    'n_absent_norm',        # Normalized ABSENT assertion count
    'n_possible_norm',      # Normalized POSSIBLE assertion count
    'assertion_polarity',   # +1 if PRESENT dominant, -1 if ABSENT, 0 otherwise
    'l2_pos_votes',         # Number of NLI models voting POSITIVE
    'l2_neg_votes',         # Number of NLI models voting NEGATIVE
    'snorkel_x_gator',      # Interaction: Snorkel * GatorTron agreement
    'nli_mean',             # Mean of DeBERTa + AdvBERT scores
]

np.random.seed(RANDOM_SEED)


# ============================================================================
# TEMPERATURE SCALING CALIBRATION
# ============================================================================

class TemperatureScaling:
    """
    2-parameter calibration: P_cal = sigmoid((logit(P_raw) - bias) / temperature)
    
    Unlike isotonic regression which memorizes, temperature scaling learns only
    2 parameters and generalizes well. The temperature controls the "sharpness"
    of predictions, and the bias corrects for base-rate mismatches.
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.bias = 0.0
    
    def fit(self, probs, labels):
        """Optimize temperature and bias to minimize NLL on validation set."""
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = logit(probs)
        
        def nll_loss(params):
            t, b = params
            calibrated_logits = (logits - b) / max(t, 0.01)
            calibrated_probs = expit(calibrated_logits)
            calibrated_probs = np.clip(calibrated_probs, 1e-7, 1 - 1e-7)
            return -np.mean(
                labels * np.log(calibrated_probs) +
                (1 - labels) * np.log(1 - calibrated_probs)
            )
        
        result = minimize(nll_loss, x0=[1.0, 0.0], method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-6})
        self.temperature = max(result.x[0], 0.01)
        self.bias = result.x[1]
        return self
    
    def predict(self, probs):
        """Apply learned calibration."""
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = logit(probs)
        calibrated_logits = (logits - self.bias) / self.temperature
        return expit(calibrated_logits)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    t_start = time.time()
    
    print("=" * 78)
    print("  LAYERS 5-7 v2 — WORLD-CLASS PROBABILISTIC META-ENSEMBLE CONSENSUS")
    print("=" * 78)
    print()
    print("  Architecture: Logistic Regression Meta-Learner + Temperature Scaling")
    print("  Calibration:  2-parameter (temperature + bias) — NO overfitting")
    print("  Output:       Multi-tier labels + balanced training set + report text")
    print()
    
    # ==================================================================
    # PHASE 1: LOAD ALL DATA SOURCES
    # ==================================================================
    print("─" * 78)
    print("  PHASE 1: LOADING ALL DATA SOURCES")
    print("─" * 78)
    print()
    
    # Snorkel soft scores
    df_snorkel = pd.read_csv(SNORKEL_SOFT_SCORES_CSV, low_memory=False)
    df_snorkel['study_id'] = df_snorkel['study_id'].astype(str)
    print(f"    Snorkel scores:     {len(df_snorkel):>10,}")
    
    # Layer 2: NLI ensemble
    df_l2 = pd.read_csv(ENSEMBLE_CSV, low_memory=False)
    df_l2['study_id'] = df_l2['study_id'].astype(str)
    print(f"    Layer 2 (NLI):      {len(df_l2):>10,}")
    
    # Layer 3: GatorTron
    df_l3 = pd.read_csv(PUBMEDBERT_CSV, low_memory=False)
    df_l3['study_id'] = df_l3['study_id'].astype(str)
    print(f"    Layer 3 (GatorTron): {len(df_l3):>10,}")
    
    # Layer 4: Assertions
    df_l4 = pd.read_csv(ASSERTIONS_CSV, low_memory=False)
    df_l4['study_id'] = df_l4['study_id'].astype(str)
    print(f"    Layer 4 (Assert):   {len(df_l4):>10,}")
    
    # Seeds
    df_seeds = pd.read_csv(SEEDS_CSV, low_memory=False)
    df_seeds['study_id'] = df_seeds['study_id'].astype(str)
    n_seed_pos = (df_seeds['seed_label'] == 1).sum()
    n_seed_neg = (df_seeds['seed_label'] == 0).sum()
    print(f"    Seeds (L1):         {len(df_seeds):>10,}  ({n_seed_pos:,} POS, {n_seed_neg:,} NEG)")
    
    # Parsed reports (for impression/findings text)
    df_reports = pd.read_csv(PARSED_REPORTS_CSV, low_memory=False,
                             usecols=['study_id', 'impression_text', 'findings_text'])
    df_reports['study_id'] = df_reports['study_id'].astype(str)
    print(f"    Parsed reports:     {len(df_reports):>10,}")
    
    # Pre-filter negatives
    if os.path.exists(PREFILTER_CSV):
        df_prefilter = pd.read_csv(PREFILTER_CSV, low_memory=False,
                                   usecols=['study_id', 'subject_id'])
        df_prefilter['study_id'] = df_prefilter['study_id'].astype(str)
        print(f"    Pre-filter NEG:     {len(df_prefilter):>10,}")
    else:
        df_prefilter = pd.DataFrame(columns=['study_id', 'subject_id'])
    
    prefilter_ids = set(df_prefilter['study_id'].tolist())
    
    # CheXpert
    try:
        df_cx = pd.read_csv(CHEXPERT_CSV, usecols=['study_id', 'Pneumonia'],
                            dtype={'study_id': int})
        df_cx['study_id'] = 's' + df_cx['study_id'].astype(str)
        df_cx = df_cx.drop_duplicates(subset='study_id', keep='first')
        cx_map = {1.0: LABEL_POSITIVE, 0.0: LABEL_NEGATIVE, -1.0: LABEL_UNCERTAIN}
        df_cx['cx_label'] = df_cx['Pneumonia'].map(cx_map).fillna(LABEL_ABSTAIN).astype(int)
        print(f"    CheXpert labels:    {len(df_cx):>10,}")
        has_chexpert = True
    except Exception as e:
        print(f"    CheXpert: UNAVAILABLE ({e})")
        has_chexpert = False
    
    print()
    
    # ==================================================================
    # PHASE 2: FEATURE ENGINEERING
    # ==================================================================
    print("─" * 78)
    print("  PHASE 2: FEATURE ENGINEERING — CONTINUOUS SIGNAL FUSION")
    print("─" * 78)
    print()
    
    # Start with Snorkel
    df = df_snorkel[['study_id', 'subject_id', 'soft_score']].copy()
    df.rename(columns={'soft_score': 'snorkel_score'}, inplace=True)
    
    # Merge L2
    l2_cols = ['study_id', 'l2_label', 'model_b_score', 'model_b_label']
    if 'model_c_score' in df_l2.columns:
        l2_cols.append('model_c_score')
    if 'model_a_label' in df_l2.columns:
        l2_cols.append('model_a_label')
    df = df.merge(df_l2[l2_cols], on='study_id', how='left')
    
    # Merge L3
    df = df.merge(df_l3[['study_id', 'l3_label', 'l3_confidence']],
                  on='study_id', how='left')
    
    # Merge L4
    l4_cols = ['study_id', 'l4_label', 'n_present', 'n_absent',
               'n_possible', 'n_conditional', 'n_historical', 'dominant_assertion']
    available_l4 = [c for c in l4_cols if c in df_l4.columns]
    df = df.merge(df_l4[available_l4], on='study_id', how='left')
    
    # Merge CheXpert
    if has_chexpert:
        df = df.merge(df_cx[['study_id', 'cx_label']], on='study_id', how='left')
        df['cx_label'] = df['cx_label'].fillna(LABEL_ABSTAIN).astype(int)
    else:
        df['cx_label'] = LABEL_ABSTAIN
    
    print(f"    Merged dataset: {len(df):,} reports")
    
    # ── Fill NaNs ──
    df['model_b_score'] = df['model_b_score'].fillna(0.5)
    df['model_c_score'] = df.get('model_c_score', pd.Series(0.5, index=df.index)).fillna(0.5)
    df['l3_confidence'] = df['l3_confidence'].fillna(0.5)
    df['l3_label'] = df['l3_label'].fillna(L_EXCLUDED).astype(int)
    df['l2_label'] = df['l2_label'].fillna(L_EXCLUDED).astype(int)
    df['l4_label'] = df['l4_label'].fillna(L_EXCLUDED).astype(int)
    df['n_present'] = df['n_present'].fillna(0).astype(int)
    df['n_absent'] = df['n_absent'].fillna(0).astype(int)
    for col in ['n_possible', 'n_conditional', 'n_historical']:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
        else:
            df[col] = 0
    df['dominant_assertion'] = df['dominant_assertion'].fillna('NONE')
    
    # Model A label
    if 'model_a_label' in df.columns:
        df['model_a_label'] = df['model_a_label'].fillna(-1).astype(int)
    else:
        df['model_a_label'] = -1
    if 'model_b_label' in df.columns:
        df['model_b_label'] = df['model_b_label'].fillna(-1).astype(int)
    else:
        df['model_b_label'] = -1
    
    # ── Engineered features ──
    print("    Engineering features...")
    
    # GatorTron signed probability: positive if L3=POS, negative if L3=NEG
    df['l3_prob'] = df.apply(
        lambda r: r['l3_confidence'] if r['l3_label'] == LABEL_POSITIVE
        else (1.0 - r['l3_confidence']) if r['l3_label'] == LABEL_NEGATIVE
        else 0.5, axis=1
    )
    
    # Normalized assertion counts (cap at 5 for stability)
    df['n_present_norm'] = df['n_present'].clip(upper=5) / 5.0
    df['n_absent_norm'] = df['n_absent'].clip(upper=5) / 5.0
    df['n_possible_norm'] = df['n_possible'].clip(upper=5) / 5.0
    
    # Assertion polarity: +1 if PRESENT dominant, -1 if ABSENT, 0 otherwise
    df['assertion_polarity'] = df['dominant_assertion'].map({
        'PRESENT': 1.0, 'ABSENT': -1.0, 'POSSIBLE': 0.3,
        'HISTORICAL': -0.5, 'CONDITIONAL': -0.3, 'NONE': 0.0
    }).fillna(0.0)
    
    # NLI vote counts
    df['l2_pos_votes'] = (
        (df['model_a_label'] == LABEL_POSITIVE).astype(int) +
        (df['model_b_label'] == LABEL_POSITIVE).astype(int)
    )
    df['l2_neg_votes'] = (
        (df['model_a_label'] == LABEL_NEGATIVE).astype(int) +
        (df['model_b_label'] == LABEL_NEGATIVE).astype(int)
    )
    
    # Interaction: Snorkel × GatorTron agreement signal
    df['snorkel_x_gator'] = df['snorkel_score'] * df['l3_prob']
    
    # NLI mean confidence
    df['nli_mean'] = (df['model_b_score'] + df['model_c_score']) / 2.0
    
    # Count binary system agreements for POSITIVE
    df['n_sys_pos'] = (
        (df['snorkel_score'] >= 0.75).astype(int) +
        (df['l2_label'] == LABEL_POSITIVE).astype(int) +
        (df['l3_label'] == LABEL_POSITIVE).astype(int) +
        (df['l4_label'] == LABEL_POSITIVE).astype(int)
    )
    # Count binary system agreements for NEGATIVE
    df['n_sys_neg'] = (
        (df['snorkel_score'] <= 0.25).astype(int) +
        (df['l2_label'] == LABEL_NEGATIVE).astype(int) +
        (df['l3_label'] == LABEL_NEGATIVE).astype(int) +
        (df['l4_label'] == LABEL_NEGATIVE).astype(int)
    )
    
    print(f"    Features: {len(FEATURE_COLS)} dimensions")
    print(f"    Feature names: {FEATURE_COLS}")
    print()
    
    # ==================================================================
    # PHASE 3: META-LEARNER TRAINING (on Seeds)
    # ==================================================================
    print("─" * 78)
    print("  PHASE 3: META-LEARNER — LOGISTIC REGRESSION ON SEED LABELS")
    print("─" * 78)
    print()
    
    # Prepare seed training data
    seed_ids = set(df_seeds['study_id'].tolist())
    seed_labels_map = dict(zip(df_seeds['study_id'], df_seeds['seed_label']))
    
    # Get seed rows from merged dataframe (exclude pre-filter)
    mask_seed = df['study_id'].isin(seed_ids) & ~df['study_id'].isin(prefilter_ids)
    df_train = df[mask_seed].copy()
    df_train['seed_label'] = df_train['study_id'].map(seed_labels_map)
    
    # Drop any with missing labels
    df_train = df_train.dropna(subset=['seed_label'])
    df_train['seed_label'] = df_train['seed_label'].astype(int)
    
    n_train = len(df_train)
    n_train_pos = (df_train['seed_label'] == 1).sum()
    n_train_neg = (df_train['seed_label'] == 0).sum()
    print(f"    Training set: {n_train:,} seeds ({n_train_pos:,} POS, {n_train_neg:,} NEG)")
    
    X_train = df_train[FEATURE_COLS].values.astype(np.float64)
    y_train = df_train['seed_label'].values
    
    # ── 5-Fold Cross-Validation ──
    print(f"    Running {N_CV_FOLDS}-fold stratified cross-validation...")
    
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_aucs = []
    cv_f1s = []
    cv_accuracies = []
    oof_probs = np.zeros(n_train)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        scaler_fold = StandardScaler()
        X_fold_train_s = scaler_fold.fit_transform(X_fold_train)
        X_fold_val_s = scaler_fold.transform(X_fold_val)
        
        lr_fold = LogisticRegression(
            C=1.0, max_iter=2000, random_state=RANDOM_SEED,
            class_weight='balanced',  # Handle imbalance in seeds
            solver='lbfgs'
        )
        lr_fold.fit(X_fold_train_s, y_fold_train)
        
        val_probs = lr_fold.predict_proba(X_fold_val_s)[:, 1]
        oof_probs[val_idx] = val_probs
        
        auc = roc_auc_score(y_fold_val, val_probs)
        preds = (val_probs >= 0.5).astype(int)
        f1 = f1_score(y_fold_val, preds)
        acc = accuracy_score(y_fold_val, preds)
        
        cv_aucs.append(auc)
        cv_f1s.append(f1)
        cv_accuracies.append(acc)
        print(f"      Fold {fold_idx+1}: AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")
    
    mean_auc = np.mean(cv_aucs)
    mean_f1 = np.mean(cv_f1s)
    mean_acc = np.mean(cv_accuracies)
    print()
    print(f"    ── Cross-Validation Summary ──")
    print(f"    Mean AUC:      {mean_auc:.4f} ± {np.std(cv_aucs):.4f}")
    print(f"    Mean F1:       {mean_f1:.4f} ± {np.std(cv_f1s):.4f}")
    print(f"    Mean Accuracy: {mean_acc:.4f} ± {np.std(cv_accuracies):.4f}")
    print()
    
    # ── Train Final Model on ALL Seeds ──
    print("    Training final meta-learner on all seeds...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    meta_lr = LogisticRegression(
        C=1.0, max_iter=2000, random_state=RANDOM_SEED,
        class_weight='balanced', solver='lbfgs'
    )
    meta_lr.fit(X_train_scaled, y_train)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'coefficient': meta_lr.coef_[0],
        'abs_coeff': np.abs(meta_lr.coef_[0])
    }).sort_values('abs_coeff', ascending=False)
    
    print(f"\n    ── Feature Importance (Learned Weights) ──")
    for _, row in importance.iterrows():
        sign = "+" if row['coefficient'] > 0 else "-"
        bar = "█" * int(row['abs_coeff'] * 10)
        print(f"      {sign}{row['abs_coeff']:.4f} {bar:20s} {row['feature']}")
    print()
    
    # ── Temperature Scaling Calibration ──
    print("    Fitting temperature scaling on OOF predictions...")
    
    ts = TemperatureScaling()
    ts.fit(oof_probs, y_train)
    print(f"    Temperature: {ts.temperature:.4f}")
    print(f"    Bias:        {ts.bias:.4f}")
    
    # Verify calibration on seeds
    calibrated_oof = ts.predict(oof_probs)
    pos_cal = calibrated_oof[y_train == 1]
    neg_cal = calibrated_oof[y_train == 0]
    print(f"    Calibrated OOF — POS: mean={pos_cal.mean():.4f} std={pos_cal.std():.4f}")
    print(f"    Calibrated OOF — NEG: mean={neg_cal.mean():.4f} std={neg_cal.std():.4f}")
    print()
    
    # ==================================================================
    # PHASE 4: PREDICT ON ALL REPORTS
    # ==================================================================
    print("─" * 78)
    print("  PHASE 4: PREDICTING ENSEMBLE PROBABILITY FOR ALL REPORTS")
    print("─" * 78)
    print()
    
    # Non-prefilter reports
    mask_non_pf = ~df['study_id'].isin(prefilter_ids)
    df_infer = df[mask_non_pf].copy()
    
    X_all = df_infer[FEATURE_COLS].values.astype(np.float64)
    X_all_scaled = scaler.transform(X_all)
    
    raw_probs = meta_lr.predict_proba(X_all_scaled)[:, 1]
    calibrated_probs = ts.predict(raw_probs)
    
    df_infer['ensemble_prob'] = calibrated_probs
    
    # Pre-filter reports get P = 0.01 (near-certain negative)
    df_pf = df[df['study_id'].isin(prefilter_ids)].copy()
    df_pf['ensemble_prob'] = 0.01
    
    # Combine
    df_all = pd.concat([df_infer, df_pf], ignore_index=True)
    
    print(f"    Total predicted: {len(df_all):,}")
    print(f"    Non-prefilter:   {len(df_infer):,}")
    print(f"    Pre-filter NEG:  {len(df_pf):,}")
    print()
    
    # Score distribution
    probs = df_all['ensemble_prob'].values
    print(f"    Score distribution (all reports):")
    print(f"      Min:  {probs.min():.4f}")
    print(f"      P10:  {np.percentile(probs, 10):.4f}")
    print(f"      P25:  {np.percentile(probs, 25):.4f}")
    print(f"      P50:  {np.percentile(probs, 50):.4f}")
    print(f"      P75:  {np.percentile(probs, 75):.4f}")
    print(f"      P90:  {np.percentile(probs, 90):.4f}")
    print(f"      Max:  {probs.max():.4f}")
    print()
    
    # ==================================================================
    # PHASE 5: MULTI-TIER LABEL ASSIGNMENT
    # ==================================================================
    print("─" * 78)
    print("  PHASE 5: MULTI-TIER LABEL ASSIGNMENT")
    print("─" * 78)
    print()
    
    # Optimal threshold from precision-recall on seeds
    # Use OOF predictions for unbiased threshold selection
    precisions, recalls, thresholds_pr = precision_recall_curve(y_train, oof_probs)
    f1_scores_pr = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_threshold_idx = np.argmax(f1_scores_pr)
    optimal_threshold = thresholds_pr[best_threshold_idx] if best_threshold_idx < len(thresholds_pr) else 0.5
    print(f"    Optimal threshold (max F1 on seeds): {optimal_threshold:.4f}")
    print(f"    F1 at optimal threshold: {f1_scores_pr[best_threshold_idx]:.4f}")
    print()
    
    def assign_label_and_tier(row):
        p = row['ensemble_prob']
        sid = row['study_id']
        n_pos = row['n_sys_pos']
        n_neg = row['n_sys_neg']
        
        # Pre-filter → NEGATIVE TIER-2
        if sid in prefilter_ids:
            return LABEL_NEGATIVE, 'TIER-2', 'pre_filter'
        
        # ── POSITIVE assignment ──
        # GOLD POSITIVE: Very high probability + strong system agreement
        if p >= 0.85 and n_pos >= 3:
            return LABEL_POSITIVE, 'GOLD', 'consensus'
        # Also GOLD if probability is overwhelming
        if p >= 0.95 and n_pos >= 2:
            return LABEL_POSITIVE, 'GOLD', 'consensus'
        # SILVER POSITIVE: High probability + moderate agreement
        if p >= 0.70 and n_pos >= 2:
            return LABEL_POSITIVE, 'SILVER', 'consensus'
        # BRONZE POSITIVE: Above optimal threshold + some agreement
        if p >= max(optimal_threshold, 0.55) and n_pos >= 2:
            return LABEL_POSITIVE, 'BRONZE', 'consensus'
        
        # ── NEGATIVE assignment ──
        # GOLD NEGATIVE: Very low probability + strong agreement
        if p <= 0.15 and n_neg >= 3:
            return LABEL_NEGATIVE, 'GOLD', 'consensus'
        if p <= 0.05 and n_neg >= 2:
            return LABEL_NEGATIVE, 'GOLD', 'consensus'
        # SILVER NEGATIVE: Low probability + moderate agreement
        if p <= 0.30 and n_neg >= 2:
            return LABEL_NEGATIVE, 'SILVER', 'consensus'
        # BRONZE NEGATIVE: Below anti-threshold
        if p <= (1 - max(optimal_threshold, 0.55)) and n_neg >= 2:
            return LABEL_NEGATIVE, 'BRONZE', 'consensus'
        # Wide-gate NEGATIVE: Clear negative signal
        if p <= 0.35 and n_neg >= 3:
            return LABEL_NEGATIVE, 'SILVER', 'consensus'
        
        # ── EXCLUDED ──
        return L_EXCLUDED, 'EXCLUDED', 'excluded'
    
    print("    Assigning labels and tiers...")
    results = df_all.apply(assign_label_and_tier, axis=1, result_type='expand')
    df_all['consensus_label'] = results[0]
    df_all['quality_tier'] = results[1]
    df_all['label_source'] = results[2]
    
    # Statistics
    label_counts = Counter(df_all['consensus_label'].tolist())
    n_pos = label_counts.get(LABEL_POSITIVE, 0)
    n_neg = label_counts.get(LABEL_NEGATIVE, 0)
    n_exc = label_counts.get(L_EXCLUDED, 0)
    n_total = len(df_all)
    
    print(f"\n    Label Assignment Results:")
    print(f"      POSITIVE:  {n_pos:>8,} ({100*n_pos/n_total:.1f}%)")
    print(f"      NEGATIVE:  {n_neg:>8,} ({100*n_neg/n_total:.1f}%)")
    print(f"      EXCLUDED:  {n_exc:>8,} ({100*n_exc/n_total:.1f}%)")
    print(f"      POS:NEG ratio: 1:{n_neg/max(n_pos,1):.1f}")
    print()
    
    tier_counts = Counter(df_all['quality_tier'].tolist())
    print(f"    Quality Tier Distribution:")
    for tier in ['GOLD', 'SILVER', 'BRONZE', 'TIER-2', 'EXCLUDED']:
        cnt = tier_counts.get(tier, 0)
        if cnt > 0:
            print(f"      {tier:>8s}: {cnt:>8,} ({100*cnt/n_total:.1f}%)")
    print()
    
    # ==================================================================
    # PHASE 6: ADVERSARIAL VALIDATION (CheXpert double-check)
    # ==================================================================
    print("─" * 78)
    print("  PHASE 6: ADVERSARIAL VALIDATION — CheXpert AGREEMENT")
    print("─" * 78)
    print()
    
    # Upgrade tier if CheXpert agrees
    def adversarial_tier_upgrade(row):
        if row['consensus_label'] == L_EXCLUDED:
            return row['quality_tier']
        
        cx = row['cx_label']
        consensus = row['consensus_label']
        current_tier = row['quality_tier']
        
        # CheXpert agrees → upgrade tier
        if cx == consensus:
            upgrades = {'BRONZE': 'SILVER', 'SILVER': 'GOLD'}
            return upgrades.get(current_tier, current_tier)
        
        # CheXpert disagrees (and is not ABSTAIN) → note but don't downgrade
        # CheXpert is just one noisy system, don't let it veto consensus
        return current_tier
    
    df_all['confidence_tier'] = df_all.apply(adversarial_tier_upgrade, axis=1)
    
    tier_final = Counter(df_all[df_all['consensus_label'] != L_EXCLUDED]['confidence_tier'].tolist())
    print(f"    Final Tier Distribution (accepted labels only):")
    for tier in ['GOLD', 'SILVER', 'BRONZE', 'TIER-2']:
        cnt = tier_final.get(tier, 0)
        if cnt > 0:
            print(f"      {tier:>8s}: {cnt:>8,}")
    
    # CheXpert agreement rate
    df_accepted = df_all[df_all['consensus_label'] != L_EXCLUDED].copy()
    cx_avail = df_accepted[df_accepted['cx_label'] != LABEL_ABSTAIN]
    if len(cx_avail) > 0:
        cx_agree = (cx_avail['cx_label'] == cx_avail['consensus_label']).sum()
        print(f"\n    CheXpert agreement (where available): {cx_agree:,}/{len(cx_avail):,} "
              f"({100*cx_agree/len(cx_avail):.1f}%)")
    print()
    
    # ==================================================================
    # PHASE 7: COUNT SYSTEM AGREEMENTS & BUILD OUTPUT
    # ==================================================================
    print("─" * 78)
    print("  PHASE 7: BUILDING FINAL OUTPUT WITH REPORT TEXT")
    print("─" * 78)
    print()
    
    # Count agreeing systems
    def count_agreeing(row):
        target = row['consensus_label']
        if target == LABEL_POSITIVE:
            return row['n_sys_pos']
        elif target == LABEL_NEGATIVE:
            return row['n_sys_neg']
        return 0
    
    df_accepted = df_all[df_all['consensus_label'] != L_EXCLUDED].copy()
    df_accepted['n_systems_agree'] = df_accepted.apply(count_agreeing, axis=1)
    
    # Merge report text
    print("    Merging report text (impression + findings)...")
    df_accepted = df_accepted.merge(
        df_reports[['study_id', 'impression_text', 'findings_text']],
        on='study_id', how='left'
    )
    df_accepted['impression_text'] = df_accepted['impression_text'].fillna('')
    df_accepted['findings_text'] = df_accepted['findings_text'].fillna('')
    
    # Assertion status
    df_accepted['assertion_status'] = df_accepted['dominant_assertion'].fillna('NONE')
    
    # Build final output
    output_cols = [
        'subject_id', 'study_id', 'consensus_label', 'ensemble_prob',
        'confidence_tier', 'label_source', 'n_systems_agree',
        'snorkel_score', 'model_b_score', 'model_c_score', 'l3_confidence',
        'assertion_status', 'impression_text', 'findings_text',
    ]
    
    df_final = df_accepted[output_cols].copy()
    df_final.rename(columns={
        'consensus_label': 'label',
        'ensemble_prob': 'soft_score',
    }, inplace=True)
    
    # Verify binary
    labels_present = set(df_final['label'].unique())
    assert labels_present <= {LABEL_POSITIVE, LABEL_NEGATIVE}, \
        f"Non-binary labels: {labels_present}"
    
    # Save full output
    os.makedirs(os.path.dirname(FINAL_CSV), exist_ok=True)
    df_final.to_csv(FINAL_CSV, index=False)
    final_size_mb = os.path.getsize(FINAL_CSV) / (1024 * 1024)
    
    n_final = len(df_final)
    n_final_pos = int((df_final['label'] == LABEL_POSITIVE).sum())
    n_final_neg = int((df_final['label'] == LABEL_NEGATIVE).sum())
    
    print(f"\n    ╔══════════════════════════════════════════════════════════╗")
    print(f"    ║  FULL LABEL SET: {FINAL_CSV}")
    print(f"    ╠══════════════════════════════════════════════════════════╣")
    print(f"    ║  Total:      {n_final:>10,}                              ║")
    print(f"    ║  POSITIVE:   {n_final_pos:>10,}  ({100*n_final_pos/n_final:.1f}%)                   ║")
    print(f"    ║  NEGATIVE:   {n_final_neg:>10,}  ({100*n_final_neg/n_final:.1f}%)                   ║")
    print(f"    ║  Ratio:         1:{n_final_neg/max(n_final_pos,1):.1f}                              ║")
    print(f"    ║  Size:          {final_size_mb:.1f} MB                             ║")
    print(f"    ╚══════════════════════════════════════════════════════════╝")
    print()
    
    # Quality tier distribution
    print(f"    Quality Tiers:")
    for tier in ['GOLD', 'SILVER', 'BRONZE', 'TIER-2']:
        mask = df_final['confidence_tier'] == tier
        cnt = mask.sum()
        if cnt > 0:
            n_tp = int((df_final.loc[mask, 'label'] == LABEL_POSITIVE).sum())
            n_tn = int((df_final.loc[mask, 'label'] == LABEL_NEGATIVE).sum())
            print(f"      {tier:>8s}: {cnt:>8,}  (POS: {n_tp:,}, NEG: {n_tn:,})")
    print()
    
    # Source distribution
    source_counts = Counter(df_final['label_source'].tolist())
    print(f"    Label Sources:")
    for src, cnt in source_counts.most_common():
        print(f"      {src:>15s}: {cnt:>8,} ({100*cnt/n_final:.1f}%)")
    print()
    
    # Agreement distribution
    agree_counts = Counter(df_final['n_systems_agree'].tolist())
    print(f"    System Agreement:")
    for n, cnt in sorted(agree_counts.items(), reverse=True):
        print(f"      {n}/4 agree: {cnt:>8,}")
    print()
    
    # Calibrated score stats
    pos_scores = df_final[df_final['label'] == LABEL_POSITIVE]['soft_score']
    neg_scores = df_final[df_final['label'] == LABEL_NEGATIVE]['soft_score']
    print(f"    Calibrated Soft Scores:")
    if len(pos_scores) > 0:
        print(f"      POSITIVE: mean={pos_scores.mean():.4f}  std={pos_scores.std():.4f}  "
              f"min={pos_scores.min():.4f}  max={pos_scores.max():.4f}")
    if len(neg_scores) > 0:
        print(f"      NEGATIVE: mean={neg_scores.mean():.4f}  std={neg_scores.std():.4f}  "
              f"min={neg_scores.min():.4f}  max={neg_scores.max():.4f}")
    print()
    
    # ==================================================================
    # PHASE 8: BALANCED TRAINING SET (1:4 Ratio)
    # ==================================================================
    print("─" * 78)
    print("  PHASE 8: BALANCED TRAINING SET (1:4 POS:NEG)")
    print("─" * 78)
    print()
    
    all_pos = df_final[df_final['label'] == LABEL_POSITIVE].copy()
    all_neg = df_final[df_final['label'] == LABEL_NEGATIVE].copy()
    
    target_neg_count = min(len(all_neg), n_final_pos * TARGET_POS_NEG_RATIO)
    
    if target_neg_count < len(all_neg):
        # Intelligent stratified sampling:
        # Keep hardest negatives (highest soft_score = closest to boundary)
        # + random sample from easy negatives
        # This ensures the model learns to distinguish borderline cases
        
        n_hard = min(int(target_neg_count * 0.30), len(all_neg))  # 30% hard
        n_easy = target_neg_count - n_hard
        
        # Sort by soft_score descending (hardest first)
        all_neg_sorted = all_neg.sort_values('soft_score', ascending=False)
        hard_neg = all_neg_sorted.head(n_hard)
        remaining_neg = all_neg_sorted.iloc[n_hard:]
        easy_neg = remaining_neg.sample(n=min(n_easy, len(remaining_neg)),
                                         random_state=RANDOM_SEED)
        sampled_neg = pd.concat([hard_neg, easy_neg], ignore_index=True)
    else:
        sampled_neg = all_neg
    
    df_balanced = pd.concat([all_pos, sampled_neg], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Save balanced set
    df_balanced.to_csv(TRAINING_READY_CSV, index=False)
    balanced_size_mb = os.path.getsize(TRAINING_READY_CSV) / (1024 * 1024)
    
    n_bal = len(df_balanced)
    n_bal_pos = int((df_balanced['label'] == LABEL_POSITIVE).sum())
    n_bal_neg = int((df_balanced['label'] == LABEL_NEGATIVE).sum())
    
    print(f"    ╔══════════════════════════════════════════════════════════╗")
    print(f"    ║  BALANCED TRAINING SET: training_ready_labels.csv       ║")
    print(f"    ╠══════════════════════════════════════════════════════════╣")
    print(f"    ║  Total:      {n_bal:>10,}                              ║")
    print(f"    ║  POSITIVE:   {n_bal_pos:>10,}  ({100*n_bal_pos/n_bal:.1f}%)                   ║")
    print(f"    ║  NEGATIVE:   {n_bal_neg:>10,}  ({100*n_bal_neg/n_bal:.1f}%)                   ║")
    print(f"    ║  Ratio:         1:{n_bal_neg/max(n_bal_pos,1):.1f}                              ║")
    print(f"    ║  Size:          {balanced_size_mb:.1f} MB                             ║")
    print(f"    ╚══════════════════════════════════════════════════════════╝")
    print()
    
    # Sampling strategy
    if target_neg_count < len(all_neg):
        print(f"    Sampling strategy: {n_hard:,} hard negatives (30%) + {n_easy:,} easy negatives (70%)")
        print(f"    Hard negatives = highest soft_score → closest to decision boundary")
        print(f"    This ensures PP1 learns robust boundary discrimination")
    else:
        print(f"    All negatives kept (already below target ratio)")
    print()
    
    # Report text coverage
    has_impression = (df_balanced['impression_text'].str.len() > 0).sum()
    has_findings = (df_balanced['findings_text'].str.len() > 0).sum()
    print(f"    Report text coverage:")
    print(f"      With impression: {has_impression:,}/{n_bal:,} ({100*has_impression/n_bal:.1f}%)")
    print(f"      With findings:   {has_findings:,}/{n_bal:,} ({100*has_findings/n_bal:.1f}%)")
    print()
    
    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    t_total = time.time() - t_start
    
    print("=" * 78)
    print("  PIPELINE COMPLETE — WORLD-CLASS PROBABILISTIC META-ENSEMBLE")
    print("=" * 78)
    print()
    print(f"  Meta-learner:    Logistic Regression (balanced, L2-regularized)")
    print(f"  CV AUC:          {mean_auc:.4f} ± {np.std(cv_aucs):.4f}")
    print(f"  CV F1:           {mean_f1:.4f} ± {np.std(cv_f1s):.4f}")
    print(f"  Calibration:     Temperature={ts.temperature:.4f}, Bias={ts.bias:.4f}")
    print(f"  Optimal thresh:  {optimal_threshold:.4f}")
    print()
    print(f"  Full labels:     {FINAL_CSV}")
    print(f"                   {n_final:,} reports ({n_final_pos:,} POS, {n_final_neg:,} NEG)")
    print()
    print(f"  Training set:    {TRAINING_READY_CSV}")
    print(f"                   {n_bal:,} reports ({n_bal_pos:,} POS, {n_bal_neg:,} NEG)")
    print(f"                   Ratio: 1:{n_bal_neg/max(n_bal_pos,1):.1f}")
    print()
    print(f"  Report text:     impression_text + findings_text included")
    print(f"                   Ready for PP2 multimodal training")
    print()
    print(f"  Excluded:        {n_exc:,} reports ({100*n_exc/n_total:.1f}%) — disagreement")
    print(f"  Runtime:         {t_total:.1f}s")
    print()
    print("  Systems used:")
    print("    [1] Snorkel LabelModel (6-LF weak supervision)")
    print("    [2] DeBERTa-v3 NLI zero-shot classification")
    print("    [3] Adversarial-prompt BART-MNLI classification")
    print("    [4] GatorTron-Base (345M params, 90B+ clinical words)")
    print("    [5] Sentence-level clinical assertion classification")
    print("    [6] CheXpert reference labels (adversarial validation)")
    print("    [+] Meta-ensemble with temperature-scaled calibration")
    print("=" * 78)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

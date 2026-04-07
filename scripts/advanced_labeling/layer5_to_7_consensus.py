"""
Layer 5 — Cross-System Consensus Gate
Layer 6 — Adversarial Validation against CheXpert
Layer 7 — Calibrated Confidence Scoring + Final Output

This script chains Layers 5, 6, and 7 because they are fast data operations
that depend on the outputs of Layers 1–4.

POSITIVE consensus (ALL must be true):
  1. Snorkel soft_score >= 0.75
  2. NLI Ensemble (Layer 2, 3-model majority) = POSITIVE
  3. GatorTron (Layer 3) = POSITIVE with confidence >= 0.70
  4. Assertion (Layer 4) has >=1 PRESENT sentence

NEGATIVE consensus (ALL must be true):
  1. Snorkel soft_score <= 0.25
  2. NLI Ensemble = NEGATIVE
  3. GatorTron = NEGATIVE with confidence >= 0.70
  4. Assertion has NO PRESENT sentence

Pre-filter negatives (64,227 reports) pass directly as NEGATIVE.

Layer 6: Reports where consensus agrees with CheXpert → TIER-1
Layer 7: Calibrated soft_score via logistic regression on system confidences

Input:  All layer outputs + snorkel_soft_scores.csv + chexpert labels
Output: advanced_final_labels.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_INTERMEDIATE, DATA_OUTPUT,
    SNORKEL_SOFT_SCORES_CSV, CHEXPERT_CSV,
    LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNCERTAIN, LABEL_ABSTAIN,
    RANDOM_SEED,
)

# Input files
SEEDS_CSV = os.path.join(DATA_INTERMEDIATE, "layer1_seeds.csv")
ENSEMBLE_CSV = os.path.join(DATA_INTERMEDIATE, "layer2_nli_ensemble.csv")
PUBMEDBERT_CSV = os.path.join(DATA_INTERMEDIATE, "layer3_pubmedbert.csv")
ASSERTIONS_CSV = os.path.join(DATA_INTERMEDIATE, "layer4_assertions.csv")
PREFILTER_CSV = os.path.join(DATA_INTERMEDIATE, "prefilter_negatives.csv")

# Output
FINAL_CSV = os.path.join(DATA_OUTPUT, "advanced_final_labels.csv")

# Consensus thresholds
SNORKEL_POS_THRESH = 0.75
SNORKEL_NEG_THRESH = 0.25

# Layer labels for excluded
L_EXCLUDED = 99


def main():
    t_start = time.time()

    print("=" * 70)
    print("LAYERS 5–7 — CONSENSUS GATE + ADVERSARIAL + CALIBRATION")
    print("=" * 70)
    print()

    # ==================================================================
    # LOAD ALL LAYER OUTPUTS
    # ==================================================================
    print("  Loading all layer outputs...")

    # Snorkel soft scores
    df_snorkel = pd.read_csv(SNORKEL_SOFT_SCORES_CSV, low_memory=False)
    df_snorkel['study_id'] = df_snorkel['study_id'].astype(str)
    print(f"    Snorkel scores: {len(df_snorkel):,}")

    # Layer 2: NLI ensemble (now 3-model)
    df_l2 = pd.read_csv(ENSEMBLE_CSV, low_memory=False)
    df_l2['study_id'] = df_l2['study_id'].astype(str)
    print(f"    Layer 2 (NLI 3-model): {len(df_l2):,}")

    # Layer 3: GatorTron
    df_l3 = pd.read_csv(PUBMEDBERT_CSV, low_memory=False)
    df_l3['study_id'] = df_l3['study_id'].astype(str)
    print(f"    Layer 3 (GatorTron):   {len(df_l3):,}")

    # Layer 4: Assertions
    df_l4 = pd.read_csv(ASSERTIONS_CSV, low_memory=False)
    df_l4['study_id'] = df_l4['study_id'].astype(str)
    print(f"    Layer 4 (Assert):      {len(df_l4):,}")

    # Pre-filter negatives
    if os.path.exists(PREFILTER_CSV):
        df_prefilter = pd.read_csv(PREFILTER_CSV, low_memory=False,
                                   usecols=['study_id', 'subject_id'])
        df_prefilter['study_id'] = df_prefilter['study_id'].astype(str)
        n_prefilter = len(df_prefilter)
        print(f"    Pre-filter NEG:        {n_prefilter:,}")
    else:
        df_prefilter = pd.DataFrame(columns=['study_id', 'subject_id'])
        n_prefilter = 0

    print()

    # ==================================================================
    # MERGE ALL LAYERS
    # ==================================================================
    print("  Merging all layers on study_id...")

    # Start with Snorkel scores
    df = df_snorkel[['study_id', 'subject_id', 'soft_score']].copy()
    df.rename(columns={'soft_score': 'snorkel_score'}, inplace=True)

    # Merge Layer 2 (now includes model_b_score AND model_c_score)
    l2_merge_cols = ['study_id', 'l2_label', 'model_b_score']
    if 'model_c_score' in df_l2.columns:
        l2_merge_cols.append('model_c_score')
    df = df.merge(df_l2[l2_merge_cols], on='study_id', how='left')
    df['l2_label'] = df['l2_label'].fillna(L_EXCLUDED).astype(int)

    # Merge Layer 3
    df = df.merge(
        df_l3[['study_id', 'l3_label', 'l3_confidence']],
        on='study_id', how='left'
    )
    df['l3_label'] = df['l3_label'].fillna(L_EXCLUDED).astype(int)

    # Merge Layer 4
    df = df.merge(
        df_l4[['study_id', 'l4_label', 'n_present', 'n_absent',
               'dominant_assertion']],
        on='study_id', how='left'
    )
    df['l4_label'] = df['l4_label'].fillna(L_EXCLUDED).astype(int)

    n_merged = len(df)
    print(f"    Merged: {n_merged:,} reports")
    print()

    # ==================================================================
    # LAYER 5 — CROSS-SYSTEM CONSENSUS GATE
    # ==================================================================
    print("=" * 70)
    print("LAYER 5 — CROSS-SYSTEM CONSENSUS GATE")
    print("=" * 70)
    print()

    # Mark pre-filter study_ids
    prefilter_ids = set(df_prefilter['study_id'].tolist())

    def consensus_vote(row):
        sid = row['study_id']

        # Pre-filter negatives pass directly
        if sid in prefilter_ids:
            return LABEL_NEGATIVE

        snorkel = row['snorkel_score']
        l2 = int(row['l2_label'])
        l3 = int(row['l3_label'])
        l4 = int(row['l4_label'])

        # POSITIVE consensus — ALL 4 systems must agree
        if (snorkel >= SNORKEL_POS_THRESH and
            l2 == LABEL_POSITIVE and
            l3 == LABEL_POSITIVE and
            l4 == LABEL_POSITIVE):
            return LABEL_POSITIVE

        # NEGATIVE consensus — ALL 4 systems must agree
        if (snorkel <= SNORKEL_NEG_THRESH and
            l2 == LABEL_NEGATIVE and
            l3 == LABEL_NEGATIVE and
            l4 == LABEL_NEGATIVE):
            return LABEL_NEGATIVE

        # Partial NEGATIVE: Snorkel says NEG + at least 2 others agree
        neg_votes = sum([
            l2 == LABEL_NEGATIVE,
            l3 == LABEL_NEGATIVE,
            l4 == LABEL_NEGATIVE,
        ])
        if snorkel <= SNORKEL_NEG_THRESH and neg_votes >= 2:
            return LABEL_NEGATIVE

        # Everything else: excluded
        return L_EXCLUDED

    print("  Applying consensus rules...")
    df['consensus_label'] = df.apply(consensus_vote, axis=1)

    cons_counts = Counter(df['consensus_label'].tolist())
    n_cons_pos = cons_counts.get(LABEL_POSITIVE, 0)
    n_cons_neg = cons_counts.get(LABEL_NEGATIVE, 0)
    n_cons_exc = cons_counts.get(L_EXCLUDED, 0)

    print(f"    POSITIVE (all 4 agree):    {n_cons_pos:>8,} ({100*n_cons_pos/n_merged:.1f}%)")
    print(f"    NEGATIVE (all/most agree): {n_cons_neg:>8,} ({100*n_cons_neg/n_merged:.1f}%)")
    print(f"    EXCLUDED (disagreement):   {n_cons_exc:>8,} ({100*n_cons_exc/n_merged:.1f}%)")
    print()

    # ==================================================================
    # LAYER 6 — ADVERSARIAL VALIDATION (CheXpert agreement check)
    # ==================================================================
    print("=" * 70)
    print("LAYER 6 — ADVERSARIAL VALIDATION (CheXpert agreement)")
    print("=" * 70)
    print()

    # Load CheXpert labels
    try:
        df_cx = pd.read_csv(CHEXPERT_CSV, usecols=['study_id', 'Pneumonia'],
                            dtype={'study_id': int})
        df_cx['study_id'] = 's' + df_cx['study_id'].astype(str)
        df_cx = df_cx.drop_duplicates(subset='study_id', keep='first')

        # Map CheXpert values
        cx_map = {1.0: LABEL_POSITIVE, 0.0: LABEL_NEGATIVE, -1.0: LABEL_UNCERTAIN}
        df_cx['cx_label'] = df_cx['Pneumonia'].map(cx_map).fillna(LABEL_ABSTAIN).astype(int)

        df = df.merge(df_cx[['study_id', 'cx_label']], on='study_id', how='left')
        df['cx_label'] = df['cx_label'].fillna(LABEL_ABSTAIN).astype(int)

        # TIER assignment
        def assign_tier(row):
            if row['consensus_label'] == L_EXCLUDED:
                return "EXCLUDED"
            if row['cx_label'] == LABEL_ABSTAIN:
                return "TIER-2"
            if row['consensus_label'] == row['cx_label']:
                return "TIER-1"
            return "TIER-2"

        df['confidence_tier'] = df.apply(assign_tier, axis=1)

        tier_counts = Counter(df['confidence_tier'].tolist())
        print(f"    TIER-1 (consensus + CheXpert agree): {tier_counts.get('TIER-1', 0):>8,}")
        print(f"    TIER-2 (consensus only):             {tier_counts.get('TIER-2', 0):>8,}")
        print(f"    EXCLUDED:                            {tier_counts.get('EXCLUDED', 0):>8,}")

    except Exception as e:
        print(f"    WARNING: Could not load CheXpert ({e})")
        print(f"    Skipping adversarial validation. All labels = TIER-2.")
        df['cx_label'] = LABEL_ABSTAIN
        df['confidence_tier'] = df['consensus_label'].apply(
            lambda x: "EXCLUDED" if x == L_EXCLUDED else "TIER-2"
        )

    print()

    # ==================================================================
    # LAYER 7 — CALIBRATED CONFIDENCE SCORING
    # ==================================================================
    print("=" * 70)
    print("LAYER 7 — CALIBRATED CONFIDENCE SCORING")
    print("=" * 70)
    print()

    # Build feature matrix for calibration
    df_accepted = df[df['consensus_label'] != L_EXCLUDED].copy()
    n_accepted = len(df_accepted)

    # Features for calibration — now includes model_c_score
    df_accepted['model_b_score'] = df_accepted['model_b_score'].fillna(0.0)
    if 'model_c_score' in df_accepted.columns:
        df_accepted['model_c_score'] = df_accepted['model_c_score'].fillna(0.0)
    else:
        df_accepted['model_c_score'] = 0.0
    df_accepted['l3_confidence'] = df_accepted['l3_confidence'].fillna(0.5)
    df_accepted['n_present'] = df_accepted['n_present'].fillna(0).astype(int)

    feature_cols = ['snorkel_score', 'model_b_score', 'model_c_score', 'l3_confidence']

    X = df_accepted[feature_cols].values
    y = (df_accepted['consensus_label'] == LABEL_POSITIVE).astype(int).values

    # Train isotonic regression for calibration
    if len(np.unique(y)) >= 2 and len(y) >= 100:
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Logistic regression to combine features
            lr = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
            lr.fit(X_scaled, y)
            raw_probs = lr.predict_proba(X_scaled)[:, 1]

            # Isotonic regression for calibration
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
            calibrated = iso.fit_transform(raw_probs, y)

            df_accepted['calibrated_score'] = calibrated

            print(f"    Calibration trained on {n_accepted:,} consensus labels")
            print(f"    Features: {feature_cols}")
            print(f"    Calibrated score stats:")
            print(f"      POSITIVE mean: {calibrated[y==1].mean():.4f}")
            print(f"      NEGATIVE mean: {calibrated[y==0].mean():.4f}")

        except Exception as e:
            print(f"    WARNING: Calibration failed ({e}). Using Snorkel scores.")
            df_accepted['calibrated_score'] = df_accepted['snorkel_score']
    else:
        print(f"    Not enough data for calibration. Using Snorkel scores.")
        df_accepted['calibrated_score'] = df_accepted['snorkel_score']

    print()

    # ==================================================================
    # FINAL OUTPUT
    # ==================================================================
    print("=" * 70)
    print("GENERATING FINAL OUTPUT")
    print("=" * 70)
    print()

    # Count systems agreeing
    def count_agreeing(row):
        target = row['consensus_label']
        count = 0
        # Snorkel
        if target == LABEL_POSITIVE and row['snorkel_score'] >= SNORKEL_POS_THRESH:
            count += 1
        elif target == LABEL_NEGATIVE and row['snorkel_score'] <= SNORKEL_NEG_THRESH:
            count += 1
        # L2 (NLI ensemble — already majority-voted)
        if row['l2_label'] == target:
            count += 1
        # L3 (GatorTron)
        if row['l3_label'] == target:
            count += 1
        # L4 (Assertion)
        if row['l4_label'] == target:
            count += 1
        return count

    df_accepted['n_systems_agree'] = df_accepted.apply(count_agreeing, axis=1)

    # Determine label source
    df_accepted['label_source'] = df_accepted['study_id'].apply(
        lambda sid: 'pre_filter' if sid in prefilter_ids else 'consensus'
    )

    # Get dominant assertion
    df_accepted['assertion_status'] = df_accepted['dominant_assertion'].fillna('NONE')

    # Build final output
    output_cols = [
        'subject_id', 'study_id', 'consensus_label', 'calibrated_score',
        'confidence_tier', 'label_source', 'n_systems_agree',
        'snorkel_score', 'model_b_score', 'model_c_score', 'l3_confidence',
        'assertion_status',
    ]

    # Rename consensus_label → label for downstream compatibility
    df_final = df_accepted[output_cols].copy()
    df_final.rename(columns={
        'consensus_label': 'label',
        'calibrated_score': 'soft_score',
    }, inplace=True)

    # Verify binary only
    assert set(df_final['label'].unique()) <= {LABEL_POSITIVE, LABEL_NEGATIVE}, \
        f"Non-binary labels found: {df_final['label'].unique()}"

    # Save
    os.makedirs(os.path.dirname(FINAL_CSV), exist_ok=True)
    df_final.to_csv(FINAL_CSV, index=False)
    file_size_mb = os.path.getsize(FINAL_CSV) / (1024 * 1024)

    # Final statistics
    n_final = len(df_final)
    n_pos = int((df_final['label'] == LABEL_POSITIVE).sum())
    n_neg = int((df_final['label'] == LABEL_NEGATIVE).sum())
    n_tier1 = int((df_final['confidence_tier'] == 'TIER-1').sum())
    n_tier2 = int((df_final['confidence_tier'] == 'TIER-2').sum())

    t_total = time.time() - t_start

    print(f"  FINAL LABEL SET:")
    print(f"    Total:     {n_final:>8,}")
    print(f"    POSITIVE:  {n_pos:>8,} ({100*n_pos/n_final:.1f}%)")
    print(f"    NEGATIVE:  {n_neg:>8,} ({100*n_neg/n_final:.1f}%)")
    print(f"    Ratio:     1:{n_neg/max(n_pos,1):.1f} POS:NEG")
    print()
    print(f"  QUALITY TIERS:")
    print(f"    TIER-1 (double-confirmed): {n_tier1:>8,} ({100*n_tier1/n_final:.1f}%)")
    print(f"    TIER-2 (consensus only):   {n_tier2:>8,} ({100*n_tier2/n_final:.1f}%)")
    print()

    source_counts = Counter(df_final['label_source'].tolist())
    print(f"  LABEL SOURCES:")
    for src, cnt in source_counts.most_common():
        print(f"    {src:>15s}: {cnt:>8,} ({100*cnt/n_final:.1f}%)")
    print()

    # System agreement distribution
    agree_counts = Counter(df_final['n_systems_agree'].tolist())
    print(f"  SYSTEM AGREEMENT:")
    for n, cnt in sorted(agree_counts.items(), reverse=True):
        print(f"    {n}/4 systems agree: {cnt:>8,}")
    print()

    # Calibrated score stats
    pos_scores = df_final[df_final['label'] == LABEL_POSITIVE]['soft_score']
    neg_scores = df_final[df_final['label'] == LABEL_NEGATIVE]['soft_score']
    print(f"  CALIBRATED SOFT SCORES:")
    if len(pos_scores) > 0:
        print(f"    POSITIVE: mean={pos_scores.mean():.4f} std={pos_scores.std():.4f}")
    if len(neg_scores) > 0:
        print(f"    NEGATIVE: mean={neg_scores.mean():.4f} std={neg_scores.std():.4f}")
    print()

    # Reports excluded
    n_excluded = n_merged - n_final
    print(f"  EXCLUDED from training: {n_excluded:,} ({100*n_excluded/n_merged:.1f}%)")
    print(f"    (These reports had disagreement between systems)")
    print()

    print(f"  File: {FINAL_CSV}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Runtime: {t_total:.1f}s")
    print()
    print("=" * 70)
    print("  ADVANCED CONSENSUS PIPELINE COMPLETE")
    print("  Models used:")
    print("    NLI: BART-MNLI + DeBERTa-v3-zeroshot-v2.0 + DeBERTa-v3-MNLI")
    print("    BERT: GatorTron-Base (345M, 90B+ clinical words)")
    print("  Output ready for PP1/PP2 model training.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

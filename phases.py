"""
phases.py
---------
The four pre-registered falsification phases.

All thresholds are locked per OSF pre-registration (March 11, 2026).
https://osf.io/9tbjm
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from utils import (
    jaccard,
    bootstrap_auc_ci,
    cross_val_auc,
    youden_threshold,
    gaussian_copula_sample,
    ks_statistics,
)

# ── Pre-registered thresholds (DO NOT MODIFY) ─────────────────────────────────
THRESHOLD_P1_JACCARD  = 0.50   # mean pairwise Jaccard < this → confirmed
THRESHOLD_P2_AUC_DROP = 0.15   # AUC drop > this → confirmed
THRESHOLD_P3_CARE_AUC = 0.70   # care-only AUC > this → confirmed
THRESHOLD_P4_DISC_AUC = 0.60   # discriminator AUC < this → confirmed

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    tree_method="hist",   # use "gpu_hist" if GPU available
)


def _prep(X: pd.DataFrame, y: pd.Series):
    """Impute and return numpy arrays."""
    imp = SimpleImputer(strategy="median")
    X_np = imp.fit_transform(X.values)
    y_np = y.fillna(False).astype(int).values
    return X_np, y_np


def phase1_ground_truth_stability(
    cohort: pd.DataFrame,
) -> dict:
    """
    Phase 1: Ground Truth Stability

    Compute pairwise Jaccard similarity across three sepsis definitions.
    Confirmed if mean pairwise Jaccard < 0.50.

    Parameters
    ----------
    cohort : DataFrame with columns sepsis2, sepsis3, cms_sepsis

    Returns
    -------
    dict with keys: confirmed, mean_jaccard, jaccard_2_3, jaccard_2_cms,
                    jaccard_3_cms, sepsis2_n, sepsis3_n, cms_n, threshold
    """
    print("Phase 1: Computing label concordance...")

    s2  = cohort["sepsis2"].fillna(False).values
    s3  = cohort["sepsis3"].fillna(False).values
    cms = cohort["cms_sepsis"].fillna(False).values

    j_23  = jaccard(s2, s3)
    j_2c  = jaccard(s2, cms)
    j_3c  = jaccard(s3, cms)
    mean_j = np.mean([j_23, j_2c, j_3c])

    confirmed = mean_j < THRESHOLD_P1_JACCARD

    result = {
        "phase": 1,
        "name": "Ground Truth Stability",
        "confirmed": confirmed,
        "metrics": {
            "mean_jaccard": round(float(mean_j), 4),
            "jaccard_sep2_sep3": round(float(j_23), 4),
            "jaccard_sep2_cms":  round(float(j_2c), 4),
            "jaccard_sep3_cms":  round(float(j_3c), 4),
        },
        "cohort": {
            "total": int(len(cohort)),
            "sepsis2_positive": int(s2.sum()),
            "sepsis3_positive": int(s3.sum()),
            "cms_positive":     int(cms.sum()),
        },
        "threshold": THRESHOLD_P1_JACCARD,
        "interpretation": (
            f"Mean pairwise Jaccard={mean_j:.4f}. "
            f"{'Below' if confirmed else 'Above'} threshold of "
            f"{THRESHOLD_P1_JACCARD}. Phase {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}."
        ),
    }

    print(
        f"  Jaccard 2v3={j_23:.4f}  2vCMS={j_2c:.4f}  3vCMS={j_3c:.4f}  "
        f"mean={mean_j:.4f}  → {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}"
    )
    return result


def phase2_feature_dependence(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    bio_features: list,
    all_features: list,
) -> dict:
    """
    Phase 2: Feature Dependence Test

    Compare XGBoost AUROC with full features vs biological features only.
    Confirmed if AUC drop > 0.15.

    Parameters
    ----------
    feature_matrix : full feature DataFrame
    labels         : binary sepsis labels (Sepsis-3)
    bio_features   : list of biological feature column names
    all_features   : list of all feature column names (bio + care-intensity)

    Returns
    -------
    dict with keys: confirmed, auc_drop, auc_full, auc_bio, threshold
    """
    print("Phase 2: Training XGBoost (full vs bio-only)...")

    clf_full = XGBClassifier(**XGB_PARAMS)
    clf_bio  = XGBClassifier(**XGB_PARAMS)

    X_full, y = _prep(feature_matrix[all_features], labels)
    X_bio,  _ = _prep(feature_matrix[bio_features], labels)

    # 5-fold CV — collect out-of-fold predictions for CI computation
    # This matches the published platform methodology exactly.
    print("  Full features 5-fold CV (collecting OOF predictions)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_full = np.zeros(len(y))
    oof_bio  = np.zeros(len(y))

    for train_idx, val_idx in skf.split(X_full, y):
        clf_full.fit(X_full[train_idx], y[train_idx])
        oof_full[val_idx] = clf_full.predict_proba(X_full[val_idx])[:, 1]

    print("  Bio-only features 5-fold CV (collecting OOF predictions)...")
    for train_idx, val_idx in skf.split(X_bio, y):
        clf_bio.fit(X_bio[train_idx], y[train_idx])
        oof_bio[val_idx] = clf_bio.predict_proba(X_bio[val_idx])[:, 1]

    auc_full = roc_auc_score(y, oof_full)
    auc_bio  = roc_auc_score(y, oof_bio)
    auc_drop = auc_full - auc_bio
    confirmed = auc_drop > THRESHOLD_P2_AUC_DROP

    # Bootstrap CI on OOF predictions (not in-sample — matches paper)
    print("  Bootstrap CIs (1,000 iterations on OOF predictions)...")
    _, ci_lo, ci_hi = bootstrap_auc_ci(y, oof_full)

    result = {
        "phase": 2,
        "name": "Feature Dependence Test",
        "confirmed": confirmed,
        "metrics": {
            "auc_full":     round(float(auc_full), 4),
            "auc_bio_only": round(float(auc_bio),  4),
            "auc_drop":     round(float(auc_drop),  4),
            "auc_full_ci_lower": round(float(ci_lo), 4),
            "auc_full_ci_upper": round(float(ci_hi), 4),
        },
        "threshold": THRESHOLD_P2_AUC_DROP,
        "interpretation": (
            f"AUC drop={auc_drop:.4f}. "
            f"{'Above' if confirmed else 'Below'} threshold of "
            f"{THRESHOLD_P2_AUC_DROP}. Phase {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}."
        ),
    }

    print(
        f"  AUC full={auc_full:.4f}  bio={auc_bio:.4f}  "
        f"drop={auc_drop:.4f}  → {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}"
    )
    return result


def phase3_care_intensity_universality(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    care_features: list,
) -> dict:
    """
    Phase 3: Care-Intensity Universality Test

    Train logistic regression (primary) and XGBoost (sensitivity) using
    only care-intensity features. Confirmed if care-only AUC > 0.70.

    Parameters
    ----------
    feature_matrix : full feature DataFrame
    labels         : binary sepsis labels (Sepsis-3)
    care_features  : list of care-intensity feature column names

    Returns
    -------
    dict with keys: confirmed, care_intensity_auc, threshold
    """
    print("Phase 3: Care-intensity universality test...")

    X_care, y = _prep(feature_matrix[care_features], labels)

    # Primary: logistic regression
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    auc_lr = cross_val_auc(X_care, y, lr_pipe)
    print(f"  Logistic regression AUC={auc_lr:.4f}")

    # Sensitivity: XGBoost
    clf = XGBClassifier(**XGB_PARAMS)
    auc_xgb = cross_val_auc(X_care, y, clf)
    print(f"  XGBoost sensitivity AUC={auc_xgb:.4f}")

    # Primary result is logistic regression
    care_auc  = auc_lr
    confirmed = care_auc > THRESHOLD_P3_CARE_AUC

    result = {
        "phase": 3,
        "name": "Care-Intensity Universality",
        "confirmed": confirmed,
        "metrics": {
            "care_intensity_auc":      round(float(care_auc),  4),
            "care_intensity_auc_xgb":  round(float(auc_xgb),  4),
        },
        "threshold": THRESHOLD_P3_CARE_AUC,
        "interpretation": (
            f"Care-only AUC={care_auc:.4f}. "
            f"{'Above' if confirmed else 'Below'} threshold of "
            f"{THRESHOLD_P3_CARE_AUC}. Phase {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}."
        ),
    }

    print(f"  Care AUC={care_auc:.4f}  → {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")
    return result


def phase4_synthetic_validation(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    care_features: list,
    n_synthetic: int = 50000,
) -> dict:
    """
    Phase 4: Synthetic Validation Test

    Generate synthetic records matching care-intensity distributions
    using a Gaussian copula. Train discriminator to distinguish real
    from synthetic. Confirmed if discriminator AUC < 0.60.

    Parameters
    ----------
    feature_matrix : full feature DataFrame
    labels         : binary sepsis labels (Sepsis-3)
    care_features  : list of care-intensity feature column names
    n_synthetic    : number of synthetic records to generate

    Returns
    -------
    dict with keys: confirmed, discriminator_auc, ks_statistics, threshold
    """
    print("Phase 4: Synthetic validation test...")

    real_df = feature_matrix.loc[
        labels.fillna(False).astype(bool), care_features
    ].dropna()

    print(f"  Real sepsis cases: {len(real_df):,}")
    print(f"  Generating {n_synthetic:,} synthetic records via Gaussian copula...")
    synth_df = gaussian_copula_sample(real_df, n_synthetic=n_synthetic)

    # Discriminator dataset: real=1, synthetic=0
    real_feat  = real_df.values
    synth_feat = synth_df.values
    X_disc = np.vstack([real_feat, synth_feat])
    y_disc = np.concatenate([
        np.ones(len(real_feat)),
        np.zeros(len(synth_feat))
    ])

    print("  Training discriminator XGBoost (5-fold CV)...")
    clf = XGBClassifier(**XGB_PARAMS)
    disc_auc = cross_val_auc(X_disc, y_disc, clf)

    confirmed = disc_auc < THRESHOLD_P4_DISC_AUC

    # KS statistics
    ks = ks_statistics(real_df, synth_df)

    result = {
        "phase": 4,
        "name": "Synthetic Validation",
        "confirmed": confirmed,
        "metrics": {
            "discriminator_auc": round(float(disc_auc), 4),
            "n_real":            int(len(real_df)),
            "n_synthetic":       int(n_synthetic),
            "ks_statistics":     {
                k: {
                    "statistic": round(v["statistic"], 4),
                    "pvalue":    round(v["pvalue"], 4),
                }
                for k, v in ks.items()
            },
        },
        "threshold": THRESHOLD_P4_DISC_AUC,
        "interpretation": (
            f"Discriminator AUC={disc_auc:.4f}. "
            f"{'Below' if confirmed else 'Above'} threshold of "
            f"{THRESHOLD_P4_DISC_AUC}. "
            f"Synthetic records are {'indistinguishable from' if confirmed else 'distinguishable from'} "
            f"real cases. Phase {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}."
        ),
    }

    print(f"  Discriminator AUC={disc_auc:.4f}  → {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")
    return result

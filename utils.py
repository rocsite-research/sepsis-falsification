"""
utils.py
--------
Shared statistical utilities: Jaccard, bootstrap CI, cross-validation,
Youden threshold, Gaussian copula sampling.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ks_2samp


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity between two binary arrays."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    intersection = (a & b).sum()
    union = (a | b).sum()
    return float(intersection / union) if union > 0 else 0.0


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple:
    """
    Bootstrap confidence interval for AUROC.

    Returns
    -------
    (auc, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    aucs = np.array(aucs)
    alpha = (1 - ci) / 2
    return (
        float(roc_auc_score(y_true, y_prob)),
        float(np.percentile(aucs, alpha * 100)),
        float(np.percentile(aucs, (1 - alpha) * 100)),
    )


def cross_val_auc(
    X: np.ndarray,
    y: np.ndarray,
    clf,
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """
    Stratified k-fold cross-validated AUROC.

    Parameters
    ----------
    X    : feature matrix
    y    : binary labels
    clf  : sklearn-compatible classifier with predict_proba
    n_splits : number of CV folds

    Returns
    -------
    mean AUROC across folds
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_val)[:, 1]
        if len(np.unique(y_val)) < 2:
            continue
        aucs.append(roc_auc_score(y_val, y_prob))
    return float(np.mean(aucs))


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Optimal classification threshold via Youden's J statistic
    (maximizes sensitivity + specificity - 1).
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def gaussian_copula_sample(
    df_real: pd.DataFrame,
    n_synthetic: int = 50000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic records matching care-intensity distributions
    using a Gaussian copula.

    Method:
    1. Transform each marginal to uniform via rank transform
    2. Apply probit transform to get Gaussian marginals
    3. Fit correlation matrix
    4. Sample from multivariate normal
    5. Back-transform through empirical quantiles

    Parameters
    ----------
    df_real      : DataFrame of real sepsis cases (care-intensity features)
    n_synthetic  : number of synthetic records to generate

    Returns
    -------
    DataFrame of synthetic records with same column names
    """
    rng = np.random.default_rng(seed)
    from scipy.stats import norm

    df = df_real.dropna().copy()
    n, p = df.shape

    # Step 1-2: rank transform → uniform → probit
    u = df.rank() / (n + 1)
    z = norm.ppf(u.values.clip(1e-6, 1 - 1e-6))

    # Step 3: correlation matrix
    corr = np.corrcoef(z.T)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)

    # Step 4: sample multivariate normal
    try:
        z_synth = rng.multivariate_normal(
            mean=np.zeros(p), cov=corr, size=n_synthetic
        )
    except np.linalg.LinAlgError:
        # Fallback: add small diagonal for numerical stability
        corr += np.eye(p) * 1e-6
        z_synth = rng.multivariate_normal(
            mean=np.zeros(p), cov=corr, size=n_synthetic
        )

    # Step 5: back-transform through empirical quantiles
    u_synth = norm.cdf(z_synth)
    synth = np.zeros_like(u_synth)
    for j in range(p):
        col_sorted = np.sort(df.iloc[:, j].values)
        quantiles = np.linspace(0, 1, len(col_sorted))
        synth[:, j] = np.interp(u_synth[:, j], quantiles, col_sorted)

    return pd.DataFrame(synth, columns=df.columns)


def ks_statistics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> dict:
    """
    Kolmogorov-Smirnov test for each feature between real and synthetic.

    Returns dict of {feature: {"statistic": float, "pvalue": float}}
    """
    results = {}
    for col in real_df.columns:
        if col not in synth_df.columns:
            continue
        r = real_df[col].dropna().values
        s = synth_df[col].dropna().values
        if len(r) < 10 or len(s) < 10:
            continue
        stat, pval = ks_2samp(r, s)
        results[col] = {"statistic": float(stat), "pvalue": float(pval)}
    return results

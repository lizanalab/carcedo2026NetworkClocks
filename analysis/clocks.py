"""
Clock fitting/prediction, factored so the SAME estimator can be trained on a
pooled matrix (leaky reproduction) or a single training fold (nested CV).

Two clocks, matching the manuscript:
  - module-PCA clock  : StandardScaler -> PCA -> RidgeCV on top-K PCs
                        (05_pca_clock_v2.ipynb, K=60)
  - one-CpG-per-module: ElasticNetCV on one representative CpG per module
                        (05_network_clock_v2.ipynb, PageRank rep)

Also a whole-array PCA clock (the baseline the manuscript compares against).
"""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error

RIDGE_ALPHAS = np.logspace(-3, 3, 25)


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return {
        "MAE":   float(mean_absolute_error(y_true, y_pred)),
        "RMSE":  float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "R2":    float(r2_score(y_true, y_pred)),
        "medAE": float(median_absolute_error(y_true, y_pred)),
        "n":     int(len(y_true)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PCA clock (module-level or whole-array — both are just "features")
# ─────────────────────────────────────────────────────────────────────────────
def fit_pca_clock(X_tr, y_tr, K=60, alphas=RIDGE_ALPHAS, cv=5, seed=42,
                  max_components=None):
    """Fit StandardScaler -> PCA -> RidgeCV(top-K PCs). X_tr: samples x features.

    PCA is fit with `max_components` (>= K) so the same fitted clock can be
    re-sliced at smaller K without refitting; predict() uses self['K'].
    """
    X_tr = np.asarray(X_tr, float)
    sc = StandardScaler().fit(X_tr)
    Xs = sc.transform(X_tr)
    n_comp = min(max_components or K, Xs.shape[1], Xs.shape[0] - 1)
    K_eff = min(K, n_comp)
    pca = PCA(n_components=n_comp, random_state=seed).fit(Xs)
    Z = pca.transform(Xs)[:, :K_eff]
    ridge = RidgeCV(alphas=alphas, cv=cv).fit(Z, y_tr)
    return {"kind": "pca", "scaler": sc, "pca": pca, "ridge": ridge,
            "K": K_eff, "alpha": float(ridge.alpha_)}


def predict_pca_clock(clock, X, K=None):
    X = np.asarray(X, float)
    K = K or clock["K"]
    Z = clock["pca"].transform(clock["scaler"].transform(X))[:, :K]
    return clock["ridge"].predict(Z)


def pca_k_sweep(X_tr, y_tr, K_list, alphas=RIDGE_ALPHAS, cv=5, seed=42):
    """Fit PCA once at max(K), return {K: fitted clock} for each K in K_list."""
    maxK = max(K_list)
    base = fit_pca_clock(X_tr, y_tr, K=maxK, alphas=alphas, cv=cv, seed=seed,
                         max_components=maxK)
    out = {}
    for K in K_list:
        Keff = min(K, base["pca"].n_components_)
        Z = base["pca"].transform(base["scaler"].transform(np.asarray(X_tr, float)))[:, :Keff]
        ridge = RidgeCV(alphas=alphas, cv=cv).fit(Z, y_tr)
        out[K] = {"kind": "pca", "scaler": base["scaler"], "pca": base["pca"],
                  "ridge": ridge, "K": Keff, "alpha": float(ridge.alpha_)}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# One-CpG-per-module ElasticNet clock
# ─────────────────────────────────────────────────────────────────────────────
def fit_enet_clock(X_tr, y_tr, l1_ratio=0.5, n_alphas=100, cv=5, seed=42):
    """ElasticNetCV on representative CpGs. X_tr: samples x n_reps."""
    X_tr = np.asarray(X_tr, float)
    sc = StandardScaler().fit(X_tr)
    # n_jobs=-1: parallelise the CV path across cores. Results-identical to the
    # single-core fit (same alphas/folds/seed); added for the real full run.
    m = ElasticNetCV(l1_ratio=l1_ratio, n_alphas=n_alphas, cv=cv,
                     random_state=seed, max_iter=10000,
                     n_jobs=-1).fit(sc.transform(X_tr), y_tr)
    return {"kind": "enet", "scaler": sc, "model": m,
            "n_nonzero": int(np.sum(m.coef_ != 0))}


def predict_enet_clock(clock, X):
    return clock["model"].predict(clock["scaler"].transform(np.asarray(X, float)))

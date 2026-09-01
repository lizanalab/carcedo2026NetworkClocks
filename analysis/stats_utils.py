"""
Statistics helpers for A2 (significance of "outperforms") and reference-clock
prediction (real via pyaging, mocked in synthetic mode).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from . import config
from .clocks import metrics


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CIs
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(y_true, y_pred, stat="MAE", n_boot=2000, seed=config.SEED, alpha=0.05):
    """Percentile bootstrap CI for a metric ('MAE','RMSE','R2','medAE') of a
    single clock's predictions. Returns (point, lo, hi)."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = metrics(y_true, y_pred)[stat]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        vals[b] = metrics(y_true[idx], y_pred[idx])[stat]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


# ─────────────────────────────────────────────────────────────────────────────
# Paired tests on per-sample absolute errors
# ─────────────────────────────────────────────────────────────────────────────
def paired_wilcoxon_ae(ae_proposed, ae_other):
    """Wilcoxon signed-rank on paired absolute errors. Returns (stat, p, median_diff).
    median_diff = median(ae_other - ae_proposed): >0 means proposed is better."""
    ae_p = np.asarray(ae_proposed, float); ae_o = np.asarray(ae_other, float)
    diff = ae_o - ae_p
    if np.allclose(diff, 0):
        return np.nan, 1.0, 0.0
    try:
        stat, p = wilcoxon(ae_p, ae_o, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        stat, p = np.nan, 1.0
    return float(stat), float(p), float(np.median(diff))


def permutation_test_ae(ae_proposed, ae_other, n_perm=10000, seed=config.SEED):
    """Sign-flip permutation test on paired AE differences (two-sided).
    Statistic = mean(ae_other - ae_proposed)."""
    rng = np.random.RandomState(seed)
    diff = np.asarray(ae_other, float) - np.asarray(ae_proposed, float)
    obs = diff.mean()
    n = len(diff)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], n)
        if abs((signs * diff).mean()) >= abs(obs):
            count += 1
    return float(obs), float((count + 1) / (n_perm + 1))


def correct_pvalues(pvals, method="holm"):
    """Multiple-testing correction. Returns array of adjusted p-values."""
    pvals = np.asarray(pvals, float)
    ok = ~np.isnan(pvals)
    adj = np.full_like(pvals, np.nan)
    if ok.sum():
        adj[ok] = multipletests(pvals[ok], method=method)[1]
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# Reference clocks: real via pyaging, mocked for synthetic review
# ─────────────────────────────────────────────────────────────────────────────
def predict_reference_clocks(df_meth_full_sxc, clocks=None, seed=config.SEED,
                             ages_for_mock=None):
    """Return {clock_key: predictions aligned to df_meth_full_sxc.index}.

    Real mode: run pyaging on the FULL (all-CpG) Sample x CpG matrix.
    Synthetic mode: pyaging can't score fake cg IDs, so return mock predictions
    = age + clock-specific bias/noise, so the A2 machinery has something to test.
    Mock predictions are clearly fictional and only exercise the code path.
    """
    clocks = clocks or config.REFERENCE_CLOCKS
    if config.USE_SYNTHETIC:
        if ages_for_mock is None:
            raise ValueError("synthetic reference-clock mock needs ages_for_mock")
        rng = np.random.RandomState(seed)
        ages = np.asarray(ages_for_mock, float)
        out = {}
        # give each clock a different (fictional) error profile
        profiles = {"horvath2013": (0.90, 6.0), "hannum": (0.95, 4.5),
                    "altumage": (0.97, 4.0), "skinandblood": (0.93, 5.0),
                    "han": (0.96, 4.2)}
        for c in clocks:
            slope, noise = profiles.get(c, (0.95, 5.0))
            out[c] = slope * ages + (1 - slope) * ages.mean() + rng.normal(0, noise, len(ages))
        return out
    # ── real pyaging path ──
    import pyaging as pya
    adata = pya.preprocess.df_to_adata(df_meth_full_sxc.copy(),
                                       imputer_strategy="mean", verbose=False)
    pya.pred.predict_age(adata, clocks, verbose=False)
    out = {}
    for c in clocks:
        col = c if c in adata.obs.columns else next(
            (x for x in adata.obs.columns if x.lower() == c.lower()), None)
        if col is not None:
            out[c] = adata.obs[col].reindex(df_meth_full_sxc.index).values
    return out

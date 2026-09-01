"""
I/O helpers and a self-contained ComBat.

`load_methylation` is copied verbatim (behaviour-wise) from the loader shared by
05_network_clock_v2.ipynb and 05_pca_clock_v2.ipynb, so real files load
identically. `combat` is a numpy re-implementation of the Johnson-Li-Rabinovic
(2007) empirical-Bayes batch correction with an optional covariate model matrix
(`mod`) — used by B4 to compare ComBat with vs. without age protected.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Data loader (shared with the clock notebooks)
# ─────────────────────────────────────────────────────────────────────────────
def load_methylation(beta_path, sheet_path):
    """Load a methylation matrix as Sample x CpG, aligned to metadata.

    Returns (df_meth indexed by Sample_ID, meta DataFrame with Age + Study_ID).
    Tolerates CpG-as-rows or Sample-as-rows layouts (auto-transpose).
    """
    beta_path, sheet_path = str(beta_path), str(sheet_path)
    # Peek orientation cheaply (first 5 rows) before loading the whole file.
    peek = pd.read_csv(beta_path, sep="\t", nrows=5)
    first_col = peek.columns[0]
    cpg_as_rows = peek[first_col].astype(str).str.startswith("cg").sum() >= 3

    if cpg_as_rows:
        # Large layout: CpGs in rows, samples in columns. Read in row-blocks
        # straight to float32 and assemble once, so the transient memory peak
        # stays near the final matrix size (~2.6 GB) instead of ballooning to
        # the full ~12 GB text buffer (which swaps a 16 GB machine).
        _hdr = pd.read_csv(beta_path, sep="\t", nrows=0).columns
        sample_ids = [c for c in _hdr[1:]]
        _dtypes = {c: np.float32 for c in sample_ids}
        _dtypes[first_col] = str
        blocks, cpg_names = [], []
        for chunk in pd.read_csv(beta_path, sep="\t", dtype=_dtypes,
                                 chunksize=20000, low_memory=False):
            cpg_names.append(chunk[first_col].astype(str).values)
            blocks.append(chunk[sample_ids].to_numpy(np.float32).T)  # samples x cpg_block
        arr = np.concatenate(blocks, axis=1)                          # samples x all_cpgs
        del blocks
        cpg_index = np.concatenate(cpg_names)
        df_meth = pd.DataFrame(arr, index=[str(s) for s in sample_ids],
                               columns=cpg_index)
        # keep only real cg* columns (drop any control/rs rows)
        keep_cols = df_meth.columns.astype(str).str.startswith("cg")
        df_meth = df_meth.loc[:, keep_cols]
    else:
        _hdr = pd.read_csv(beta_path, sep="\t", nrows=0).columns
        _dtypes = {c: np.float32 for c in _hdr}
        _dtypes[_hdr[0]] = str
        df = pd.read_csv(beta_path, sep="\t", dtype=_dtypes, low_memory=False)
        cpg_cols = [c for c in df.columns if str(c).startswith("cg")]
        idc = next((c for c in ["Sample_ID", "ID_REF", "ID", "Sample", "geo_accession"]
                    if c in df.columns), None)
        df_meth = df[cpg_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
        if idc:
            df_meth.index = df[idc].astype(str)

    meta = pd.read_csv(sheet_path)
    for c in meta.columns:
        if c.lower() == "age":
            meta = meta.rename(columns={c: "Age"})
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")

    mid = next((c for c in ["Sample_ID", "ID_REF", "ID", "Sample", "geo_accession"]
                if c in meta.columns), None)
    if mid:
        meta = meta.set_index(meta[mid].astype(str))
    common = df_meth.index.intersection(meta.index)
    # Fallback: some matrices use bare Sentrix IDs (e.g. 9482801045_R02C01) instead
    # of the prefixed Sample_ID. Re-key meta on Sentrix_ID_Position and retry.
    if len(common) == 0 and {"Sentrix_ID", "Sentrix_Position"}.issubset(meta.columns):
        sentrix_key = (meta["Sentrix_ID"].astype(str) + "_"
                       + meta["Sentrix_Position"].astype(str))
        meta2 = meta.copy()
        meta2.index = sentrix_key
        common2 = df_meth.index.intersection(meta2.index)
        if len(common2) > len(common):
            meta = meta2
            common = common2
    df_meth = df_meth.loc[common]
    meta = meta.loc[common]

    if "Study_ID" not in meta.columns:
        for alt in ["GSE", "Study", "Series", "Dataset"]:
            if alt in meta.columns:
                meta = meta.rename(columns={alt: "Study_ID"})
                break
        if "Study_ID" not in meta.columns:
            meta["Study_ID"] = "unknown"
    return df_meth, meta


# ─────────────────────────────────────────────────────────────────────────────
# ComBat (Johnson et al. 2007), numpy, with optional covariate model matrix
# ─────────────────────────────────────────────────────────────────────────────
def combat(data_fxs: np.ndarray, batch: np.ndarray, mod: np.ndarray | None = None,
           eps: float = 1e-8) -> np.ndarray:
    """Parametric empirical-Bayes ComBat.

    data_fxs : features x samples array (CpGs x samples).
    batch    : length-n_samples array of batch labels.
    mod      : optional n_samples x n_covariates design matrix of biological
               covariates to PROTECT (e.g. an age column). Do NOT include the
               intercept — it is added internally.
    Returns the batch-adjusted features x samples array.
    """
    data = np.asarray(data_fxs, float)
    g, n = data.shape
    batch = np.asarray(batch)
    levels = pd.unique(batch)
    batchmod = np.column_stack([(batch == b).astype(float) for b in levels])  # n x nb

    design = batchmod if mod is None else np.column_stack([batchmod, mod])
    # Standardise across genes using the full design
    B_hat, *_ = np.linalg.lstsq(design, data.T, rcond=None)     # coef x genes
    nb = batchmod.shape[1]
    grand_mean = (np.array([ (batch == b).mean() for b in levels ]) @ B_hat[:nb, :])
    stand_mean = np.outer(np.ones(n), grand_mean)               # n x genes
    if mod is not None:
        stand_mean += mod @ B_hat[nb:, :]
    resid = data - stand_mean.T
    var_pooled = (resid ** 2).mean(axis=1)
    var_pooled[var_pooled < eps] = eps
    s_data = (data - stand_mean.T) / np.sqrt(var_pooled)[:, None]

    # Batch effect EB estimates
    gamma_hat, delta_hat = [], []
    for b in levels:
        idx = np.where(batch == b)[0]
        gamma_hat.append(s_data[:, idx].mean(axis=1))
        delta_hat.append(s_data[:, idx].var(axis=1) + eps)
    gamma_hat = np.array(gamma_hat)                 # nb x genes
    delta_hat = np.array(delta_hat)

    gamma_bar = gamma_hat.mean(axis=1)
    t2 = gamma_hat.var(axis=1)
    def aprior(d):
        m, s2 = d.mean(), d.var()
        return (2 * s2 + m ** 2) / s2
    def bprior(d):
        m, s2 = d.mean(), d.var()
        return (m * s2 + m ** 3) / s2
    a_prior = np.array([aprior(d) for d in delta_hat])
    b_prior = np.array([bprior(d) for d in delta_hat])

    adj = np.empty_like(s_data)
    for i, b in enumerate(levels):
        idx = np.where(batch == b)[0]
        n_b = len(idx)
        g_star, d_star = _it_sol(s_data[:, idx], gamma_hat[i], delta_hat[i],
                                 gamma_bar[i], t2[i], a_prior[i], b_prior[i])
        adj[:, idx] = ((s_data[:, idx] - g_star[:, None]) / np.sqrt(d_star)[:, None])
    bayes = adj * np.sqrt(var_pooled)[:, None] + stand_mean.T
    return bayes


def _it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, tol=1e-4, maxit=200):
    n = (~np.isnan(sdat)).sum(axis=1)
    g_old, d_old = g_hat.copy(), d_hat.copy()
    for _ in range(maxit):
        g_new = (t2 * n * g_hat + d_old * g_bar) / (t2 * n + d_old)
        ssq = ((sdat - g_new[:, None]) ** 2).sum(axis=1)
        d_new = (0.5 * ssq + b) / (n / 2.0 + a - 1.0)
        if (np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-12)) < tol and
                np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-12)) < tol):
            g_old, d_old = g_new, d_new
            break
        g_old, d_old = g_new, d_new
    return g_old, d_old


# ─────────────────────────────────────────────────────────────────────────────
# small helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_fig(fig, path, dpi=300):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(p.with_suffix(".pdf")), bbox_inches="tight")
    return p


class ResultsLog:
    """Append-only plain-text results log written into review/outputs/."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lines = []

    def __call__(self, msg=""):
        print(msg)
        self.lines.append(str(msg))

    def flush(self):
        self.path.write_text("\n".join(self.lines) + "\n")
        return self.path

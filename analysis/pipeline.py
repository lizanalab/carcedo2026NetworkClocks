"""
Reusable co-methylation-network pipeline steps, factored out so they can be
re-run **inside a CV fold** (the whole point of task A1).

Each function takes an explicit Sample x CpG methylation frame (or the array
plus a sample mask) so the SAME code can run on the pooled data (to reproduce
the original "leaky" result) or on a single training fold (nested CV).

Steps mirror the published pipeline:
  1. age_filter_cpgs      : |Spearman rho(CpG, age)| >= RHO_THRESHOLD
  2. build_network        : Spearman CpG-CpG corr, keep |r| >= TAU_THRESHOLD
  3. infomap_modules      : Infomap two-level community detection
  4. module_summaries     : per-module median (PCA clock) / PageRank rep (Ridge clock)

The network + Infomap match outputs/02_network/* (Spearman, tau=0.70, two-level).
Infomap is preferred; if the `infomap` package is unavailable we fall back to
networkx greedy modularity and log a clear warning (results then differ from the
published Infomap modules — only used so the code runs where Infomap is missing).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import rankdata

from .config import RHO_THRESHOLD, TAU_THRESHOLD, INFOMAP_ARGS


# ─────────────────────────────────────────────────────────────────────────────
# 1. Age-associated CpG filtering
# ─────────────────────────────────────────────────────────────────────────────
def spearman_with_age(betas_sxc: pd.DataFrame, ages: np.ndarray) -> pd.Series:
    """Spearman rho between every CpG (columns of a Sample x CpG frame) and age.

    Vectorised: rank both sides, then Pearson on ranks. Handles NaNs by
    per-column mean imputation on the ranks (rare after normalisation).
    Returns a Series indexed by CpG (signed rho).
    """
    # Column-wise average-tie ranks, NaN -> mean of that column's real ranks
    # (== the original per-column _rank_nan). Vectorised with scipy.rankdata
    # (nan_policy='omit') in CpG blocks so a 337k-CpG matrix stays tractable:
    # ~80s/pass and bounded memory, instead of >19 min for a whole-frame rank.
    # Verified numerically identical (max abs diff 0.0) to the old loop.
    cols = betas_sxc.columns
    n_c = betas_sxc.shape[1]
    yr = rankdata(ages).astype(np.float64)
    yr = yr - yr.mean()
    yss = (yr ** 2).sum()
    rho = np.empty(n_c, dtype=np.float64)
    BLOCK = 20000
    for s in range(0, n_c, BLOCK):
        e = min(s + BLOCK, n_c)
        Xb = betas_sxc.iloc[:, s:e].to_numpy(np.float32)
        Xr = rankdata(Xb, axis=0, nan_policy="omit")          # real ranked 1..k
        cm = np.nanmean(Xr, axis=0)                            # per-col mean rank
        ii = np.where(np.isnan(Xr))
        Xr[ii] = np.take(cm, ii[1])                            # NaN -> mean rank
        Xr -= Xr.mean(axis=0, keepdims=True)
        denom = np.sqrt((Xr ** 2).sum(axis=0) * yss)
        denom[denom == 0] = np.nan
        rho[s:e] = (Xr * yr[:, None]).sum(axis=0) / denom
    return pd.Series(rho, index=cols)


def _rank_nan(col: np.ndarray) -> np.ndarray:
    m = ~np.isnan(col)
    r = np.full(col.shape, np.nan)
    r[m] = rankdata(col[m])
    if (~m).any():
        r[~m] = np.nanmean(r)
    return r


def age_filter_cpgs(betas_sxc: pd.DataFrame, ages: np.ndarray,
                    rho_threshold: float = RHO_THRESHOLD):
    """Return (selected_cpgs, rho_series) keeping |rho| >= threshold."""
    rho = spearman_with_age(betas_sxc, ages)
    keep = rho[rho.abs() >= rho_threshold].index.tolist()
    return keep, rho


# ─────────────────────────────────────────────────────────────────────────────
# 2. Co-methylation network (Spearman, prune at tau)
# ─────────────────────────────────────────────────────────────────────────────
def build_network(betas_sxc: pd.DataFrame, cpgs, tau: float = TAU_THRESHOLD,
                  method: str = "spearman") -> nx.Graph:
    """Spearman correlation network over `cpgs`; edge kept when |corr| >= tau.

    Edge weight = |corr| (matches how PageRank is later run with weight='weight').
    """
    sub = betas_sxc[cpgs]
    if method == "spearman":
        corr = sub.rank().corr().values          # Spearman via rank+Pearson
    else:
        corr = sub.corr().values
    np.fill_diagonal(corr, 0.0)
    cpgs = list(cpgs)
    G = nx.Graph()
    G.add_nodes_from(cpgs)
    iu = np.triu_indices(len(cpgs), k=1)
    w = corr[iu]
    keep = np.abs(w) >= tau
    ii, jj = iu[0][keep], iu[1][keep]
    ww = np.abs(w[keep])
    G.add_weighted_edges_from((cpgs[a], cpgs[b], float(wt))
                              for a, b, wt in zip(ii, jj, ww))
    return G


# ─────────────────────────────────────────────────────────────────────────────
# 3. Infomap module detection
# ─────────────────────────────────────────────────────────────────────────────
def infomap_modules(G: nx.Graph, args: str = INFOMAP_ARGS) -> dict:
    """Return {node -> module_id (int, 1-based)}.

    Prefers the `infomap` package (two-level, weighted). Isolated nodes each get
    their own singleton module. Falls back to greedy modularity if Infomap is
    unavailable (logged).
    """
    try:
        from infomap import Infomap
    except Exception as e:                       # pragma: no cover
        warnings.warn(f"[infomap] unavailable ({e}); using greedy-modularity "
                      f"fallback — modules will differ from the published Infomap run.")
        return _greedy_modules(G)

    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    im = Infomap(args)
    for u, v, d in G.edges(data=True):
        im.add_link(idx[u], idx[v], float(d.get("weight", 1.0)))
    # ensure isolated nodes exist as Infomap nodes
    for n in nodes:
        im.add_node(idx[n])
    im.run()
    modmap = {}
    for node in im.nodes:
        modmap[nodes[node.node_id]] = int(node.module_id)
    # nodes Infomap dropped (fully isolated) -> unique singleton modules
    nxt = (max(modmap.values()) + 1) if modmap else 1
    for n in nodes:
        if n not in modmap:
            modmap[n] = nxt
            nxt += 1
    return _renumber(modmap)


def _greedy_modules(G: nx.Graph) -> dict:
    from networkx.algorithms.community import greedy_modularity_communities
    modmap = {}
    if G.number_of_edges() > 0:
        comms = greedy_modularity_communities(G, weight="weight")
        for m, comm in enumerate(comms, start=1):
            for n in comm:
                modmap[n] = m
    nxt = (max(modmap.values()) + 1) if modmap else 1
    for n in G.nodes():
        if n not in modmap:
            modmap[n] = nxt
            nxt += 1
    return _renumber(modmap)


def _renumber(modmap: dict) -> dict:
    """Relabel module ids to contiguous 1..K ordered by descending size."""
    from collections import Counter
    sizes = Counter(modmap.values())
    order = {old: new for new, (old, _) in
             enumerate(sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])), start=1)}
    return {n: order[m] for n, m in modmap.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Module summarisation
# ─────────────────────────────────────────────────────────────────────────────
def module_cpg_lists(modmap: dict, available_cpgs=None) -> dict:
    """{module_id -> [cpgs]} optionally restricted to `available_cpgs`."""
    from collections import defaultdict
    avail = set(available_cpgs) if available_cpgs is not None else None
    out = defaultdict(list)
    for cpg, m in modmap.items():
        if avail is None or cpg in avail:
            out[m].append(cpg)
    return dict(out)


def module_median_matrix(betas_sxc: pd.DataFrame, mod_cpgs: dict) -> pd.DataFrame:
    """Sample x Module matrix: median methylation across each module's CpGs.

    Singletons -> the single CpG value. Column order = sorted module id.
    """
    cols = {}
    for m in sorted(mod_cpgs):
        cpgs = [c for c in mod_cpgs[m] if c in betas_sxc.columns]
        if not cpgs:
            continue
        cols[m] = (betas_sxc[cpgs[0]] if len(cpgs) == 1
                   else betas_sxc[cpgs].median(axis=1))
    X = pd.DataFrame(cols, index=betas_sxc.index)
    return X.fillna(X.mean())


def pagerank_representatives(G: nx.Graph, mod_cpgs: dict, ranked_modules) -> list:
    """Top-PageRank CpG from each module in `ranked_modules` order.

    Mirrors pagerank_reps() in 05_network_clock_v2.ipynb.
    """
    reps = []
    for m in ranked_modules:
        nodes = [c for c in mod_cpgs.get(m, []) if G.has_node(c)]
        if not nodes:
            continue
        sub = G.subgraph(nodes)
        try:
            pr = nx.pagerank(sub, weight="weight")
        except Exception:
            pr = {n: 1.0 for n in nodes}
        reps.append(max(pr, key=pr.get))
    return reps


def rank_modules_by_age(mod_cpgs: dict, rho: pd.Series) -> list:
    """Module ids ordered by descending mean |rho(CpG, age)| (module scoring
    used to pick 'top modules' — matches df_scores in the Ridge notebook)."""
    scored = []
    for m, cpgs in mod_cpgs.items():
        vals = [abs(rho[c]) for c in cpgs if c in rho.index and not np.isnan(rho[c])]
        if vals:
            scored.append((m, float(np.mean(vals))))
    scored.sort(key=lambda kv: -kv[1])
    return [m for m, _ in scored]

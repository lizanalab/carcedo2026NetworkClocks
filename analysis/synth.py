"""
Fictional-data generator for code review.

Produces small files in review/synthetic_data/ that match the REAL pipeline's
formats exactly, and are internally consistent with the pipeline logic:

    raw (with study batch effects)
      -> ComBat  -> Beta_ALL.tsv          (all CpGs, post-ComBat)
      -> |rho(age)|>=0.35 -> BetaMatrix_0.35.tsv
      -> |rho(age)|>=0.30 -> BetaMatrix_0.30.tsv
    network on the 0.35 set -> Infomap -> module_assignments_0.70.csv + network_0.70.gexf

Also writes: Samplesheet.csv, 450K/EPICv1/EPICv2 manifests, and a synthetic
manuscript text carrying the exact phrases G3/G4 look for.

The age signal and module co-methylation structure are real (latent-variable
construction), so filtering, the network, Infomap, PCA and the clocks all behave
like they would on real data — only the numbers are fictional.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

from . import config
from .io_utils import combat
from .pipeline import build_network, infomap_modules


def _cg_id(i: int) -> str:
    return f"cg{i:08d}"


def generate(n_samples=300, n_studies=6, n_modules=40,
             cpgs_per_module=(4, 40), n_noise_cpgs=2600, seed=config.SEED,
             out_dir: Path | None = None, verbose=True):
    rng = np.random.RandomState(seed)
    out = Path(out_dir or config.SYNTH_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # ── samples / studies / ages ────────────────────────────────────────────
    study_ids = np.array([f"GSE{100+ (i % n_studies)}" for i in range(n_samples)])
    rng.shuffle(study_ids)
    ages = rng.uniform(20, 85, n_samples)
    z_age = (ages - ages.mean()) / ages.std()
    sample_ids = np.array([f"{study_ids[i]}__S{i:04d}" for i in range(n_samples)])

    # ── build true modules of co-methylated, age-associated CpGs ────────────
    cpg_names, cpg_module, betas_clean = [], [], []
    ci = 0
    for m in range(1, n_modules + 1):
        size = rng.randint(cpgs_per_module[0], cpgs_per_module[1] + 1)
        # module latent shared across its CpGs: a MODERATE age component (so each
        # CpG's |rho(age)| lands ~0.4-0.6, above the 0.35 cutoff) plus a strong
        # module-specific latent (so within-module co-methylation dominates and
        # cross-module correlation stays below tau=0.70 -> distinct modules).
        age_coef = rng.uniform(0.45, 0.85) * (1 if rng.rand() < 0.5 else -1)
        u_m = rng.normal(0, 1, n_samples)
        latent = age_coef * z_age + 1.0 * u_m
        latent = (latent - latent.mean()) / latent.std()
        for _ in range(size):
            mean_j = rng.uniform(0.15, 0.85)
            load_j = rng.uniform(0.05, 0.14) * (1 if rng.rand() < 0.5 else -1)
            eps = rng.normal(0, 0.02, n_samples)
            beta = np.clip(mean_j + load_j * latent + eps, 0.01, 0.99)
            cpg_names.append(_cg_id(ci)); ci += 1
            cpg_module.append(m)
            betas_clean.append(beta)
    # ── solo age-CpGs: age-associated but NOT co-methylated -> singletons ───
    n_solo = 40
    for _ in range(n_solo):
        age_coef = rng.uniform(0.45, 0.85) * (1 if rng.rand() < 0.5 else -1)
        latent = age_coef * z_age + 1.0 * rng.normal(0, 1, n_samples)
        latent = (latent - latent.mean()) / latent.std()
        mean_j = rng.uniform(0.15, 0.85)
        load_j = rng.uniform(0.06, 0.14) * (1 if rng.rand() < 0.5 else -1)
        beta = np.clip(mean_j + load_j * latent + rng.normal(0, 0.02, n_samples), 0.01, 0.99)
        cpg_names.append(_cg_id(ci)); ci += 1
        cpg_module.append(-1)                        # -1 = solo/singleton ground truth
        betas_clean.append(beta)
    n_signal = len(cpg_names)

    # ── noise CpGs: no age association, weak private structure ──────────────
    for _ in range(n_noise_cpgs):
        mean_j = rng.uniform(0.1, 0.9)
        beta = np.clip(mean_j + rng.normal(0, 0.03, n_samples), 0.01, 0.99)
        cpg_names.append(_cg_id(ci)); ci += 1
        cpg_module.append(0)
        betas_clean.append(beta)

    clean = np.array(betas_clean)                    # features x samples, batch-free
    n_cpgs = clean.shape[0]

    # ── add study batch effects -> raw (pre-ComBat) ─────────────────────────
    raw = clean.copy()
    for s in pd.unique(study_ids):
        idx = np.where(study_ids == s)[0]
        add = rng.normal(0, 0.05, (n_cpgs, 1))       # per-CpG additive batch shift
        mul = rng.uniform(0.85, 1.15, (n_cpgs, 1))   # per-CpG scale shift
        raw[:, idx] = clean[:, idx] * mul + add
    raw = np.clip(raw, 0.001, 0.999)

    # ── ComBat (no covariate) -> full post-ComBat matrix ────────────────────
    full = combat(raw, study_ids, mod=None)
    full = np.clip(full, 0.001, 0.999)

    df_full = pd.DataFrame(full, index=cpg_names, columns=sample_ids)
    df_raw  = pd.DataFrame(raw,  index=cpg_names, columns=sample_ids)

    # ── age filtering on the post-ComBat matrix ─────────────────────────────
    Xsxc = df_full.T                                  # samples x cpgs
    from .pipeline import spearman_with_age
    rho = spearman_with_age(Xsxc, ages).abs()
    keep035 = rho[rho >= config.RHO_THRESHOLD].index.tolist()
    keep030 = rho[rho >= 0.30].index.tolist()
    if verbose:
        print(f"  synth: {n_cpgs} CpGs ({n_signal} signal), "
              f"|rho|>=0.35 -> {len(keep035)}, |rho|>=0.30 -> {len(keep030)}")

    # ── network + Infomap on the 0.35 set ───────────────────────────────────
    G = build_network(Xsxc, keep035, tau=config.TAU_THRESHOLD)
    modmap = infomap_modules(G)
    n_mods = len(set(modmap.values()))
    if verbose:
        print(f"  synth network: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges -> {n_mods} modules")

    # ── write files (real formats) ──────────────────────────────────────────
    df_full.loc[keep035].to_csv(out / "BetaMatrix_0.35.tsv", sep="\t")
    df_full.loc[keep030].to_csv(out / "BetaMatrix_0.30.tsv", sep="\t")
    df_full.to_csv(out / "Beta_ALL.tsv", sep="\t")
    df_raw.to_csv(out / "Beta_raw_preComBat.tsv", sep="\t")

    sheet = pd.DataFrame({
        "Study_ID": study_ids, "Sample_ID": sample_ids,
        "Sample_Name": sample_ids, "Sample_Group": "All",
        "Sentrix_ID": [f"{rng.randint(1e9,9e9)}" for _ in range(n_samples)],
        "Sentrix_Position": "R01C01",
        "Age": np.round(ages, 1),
        "Sex": rng.choice(["M", "F"], n_samples),
        "Cell_Type": "WB",
        "Disease": rng.choice(["control", "case"], n_samples, p=[0.8, 0.2]),
        "Smoking": rng.choice(["never", "former", "current"], n_samples),
    })
    sheet.to_csv(out / "Samplesheet.csv", index=False)

    pd.DataFrame({"CpG": list(modmap.keys()), "Module": list(modmap.values())}
                 ).sort_values("Module").to_csv(out / "module_assignments_0.70.csv", index=False)
    nx.write_gexf(G, str(out / "network_0.70.gexf"))

    # ── manifests: 450K has all; EPIC v1 drops ~8%, EPIC v2 drops ~15% ──────
    all_cg = cpg_names
    pd.DataFrame({"IlmnID": all_cg, "CHR": rng.randint(1, 23, n_cpgs),
                  "UCSC_RefGene_Name": _fake_genes(all_cg, rng)}
                 ).to_csv(out / "manifest_450k.csv", index=False)
    v1 = [c for c in all_cg if rng.rand() > 0.08]
    v2 = [c for c in all_cg if rng.rand() > 0.15]
    pd.DataFrame({"IlmnID": v1}).to_csv(out / "manifest_epicv1.csv", index=False)
    pd.DataFrame({"IlmnID": v2}).to_csv(out / "manifest_epicv2.csv", index=False)

    _write_go_enrichment(out / "go_enrichment_raw.csv", modmap, rng)
    _write_synth_manuscript(out / "manuscript_synthetic.txt")

    return dict(cpg_names=cpg_names, cpg_module=np.array(cpg_module),
                ages=ages, study_ids=study_ids, sample_ids=sample_ids,
                keep035=keep035, keep030=keep030, modmap=modmap, n_modules=n_mods)


def _fake_genes(cgs, rng):
    genes = ["ELOVL2", "FHL2", "PENK", "KLF14", "CCDC102B", "OTUD7A",
             "SCGN", "NKX2-1", "TRIM59", "MEIS1", "GLRA1", ""]
    return [genes[rng.randint(0, len(genes))] for _ in cgs]


def _write_go_enrichment(path: Path, modmap: dict, rng):
    """Fictional RAW per-module GO enrichment table (gProfiler-style) WITH the
    counts + p-values needed to compute fold enrichment (observed/expected).
    Real run: point config.REAL_GO_ENRICH at the true enrichment output."""
    from collections import Counter
    terms = [("GO:0048731", "system development"),
             ("GO:0007275", "multicellular organism development"),
             ("GO:0007399", "nervous system development"),
             ("GO:0009653", "anatomical structure morphogenesis"),
             ("GO:0048856", "anatomical structure development"),
             ("GO:0032502", "developmental process"),
             ("GO:0007268", "chemical synaptic transmission"),
             ("GO:0050877", "nervous system process")]
    background = 20000
    sizes = Counter(modmap.values())
    big = [m for m, s in sizes.items() if s >= 10]
    rows = []
    for m in big:
        query_size = int(sizes[m] * rng.uniform(0.4, 0.9))    # genes mapped from module CpGs
        n_terms = rng.randint(2, 5)
        for (go, term) in [terms[i] for i in rng.choice(len(terms), n_terms, replace=False)]:
            term_size = int(rng.uniform(200, 5000))
            expected = query_size * term_size / background
            observed = int(min(query_size, max(1, expected * rng.uniform(1.5, 6.0))))
            # nominal p via hypergeometric upper tail
            from scipy.stats import hypergeom
            p = hypergeom.sf(observed - 1, background, term_size, query_size)
            rows.append({"module": f"Module_{m}", "GO_id": go, "term": term,
                         "p_value": float(p), "intersection_size": observed,
                         "term_size": term_size, "query_size": query_size,
                         "effective_domain_size": background, "is_parent": True})
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_synth_manuscript(path: Path):
    """A stand-in manuscript that deliberately contains every G3/G4 issue, so
    the editorial grep finds real hits during review. Replace REAL_MANUSCRIPT_FILES
    with the true manuscript for the real run."""
    text = """Synthetic manuscript stand-in for editorial-grep validation.

The "PCA clock" outperforms the whole-array baseline, and the ‘PCA clock’ is
robust. Elsewhere we call it the PCA clock without quotes. We ran Infomap on the
network; earlier we ran **Infomap** in bold, and later plain Infomap.

The remaining variance is dominated by CpGs do not co-vary with age, which is a
mangled sentence. The clock transfers across between measurement platforms,
which is also mangled.

References
13. Smith J, et al. Epigenetic clocks. Nature Aging. 2021.
14. Smith J, et al. Epigenetic clocks. Nature Aging. 2021.
31. Doe A, Roe B. Network modules of methylation. Genome Biol. 2020.
37. Doe A, Roe B. Network modules of methylation. Genome Biol. 2020.
"""
    path.write_text(text)


def ensure_synthetic(force=False, **kw):
    """Generate synthetic files if missing (or force). Returns the ground-truth dict.
    Skips regeneration when files already exist and force=False (returns None)."""
    out = config.SYNTH_DIR
    marker = out / "module_assignments_0.70.csv"
    if marker.exists() and not force:
        return None
    return generate(out_dir=out, **kw)

# A network approach to DNA methylation clocks

Code, trained clocks, and analysis for **"A network approach to DNA methylation clocks"**
(A. Carcedo *et al.*, 2026).

This repository builds a **network-based epigenetic age clock**. CpG sites are first pre-filtered
by their Spearman correlation with age, organised into a CpG–CpG correlation network,
clustered into **modules** with Infomap, and used to train two complementary clock variants:

1. **Network Clock (Ridge)** — Ridge regression on a fixed number of CpGs sampled per module.
2. **Module-PCA Clock** — Ridge regression on the top-*K* principal components of the
   per-module summary matrix (this is the best-performing variant in the paper).

Both are benchmarked against five published reference clocks (Horvath, Hannum, AltumAge,
Skin & Blood, Han) on **three external EPIC validation cohorts**.

---

## Contents

- [Quick start](#quick-start) — apply a trained clock to your own β-matrix in ~10 lines
- [Installation](#installation)
- [Repository layout](#repository-layout)
- [The `analysis/` folder](#the-analysis-folder) — reusable training & module-discovery pipeline
- [Analysis pipeline](#analysis-pipeline) — notebook-by-notebook, in run order
- [Key parameters](#key-parameters)
- [Reproducing the figures](#reproducing-the-figures)
- [Data](#data) — training + validation accessions
- [Data preprocessing](#data-preprocessing)
- [Outputs](#outputs) — what each results file contains
- [Runtime & hardware](#runtime--hardware)
- [Citation](#citation) · [License](#license) · [Contact](#contact)

---

## Quick start

If you only want to **age-predict your own samples** with the trained clock (no retraining):

```python
import pandas as pd, numpy as np, pickle

# beta: a DataFrame of β-values, rows = samples, columns = CpG probe IDs (cg#######)
beta = pd.read_csv("my_betas.tsv", sep="\t", index_col=0)

# load the trained Module-PCA clock (a dict of fitted components, not an object)
with open("outputs/05_pca_clock/saved_clocks/pca_clock.pkl", "rb") as fh:
    clock = pickle.load(fh)

def predict_age(beta, clock):
    order, cdef = clock["cluster_order"], clock["cluster_definitions"]
    # one feature per module: median β across the module's CpGs (missing CpGs skipped)
    X = np.column_stack([
        beta.reindex(columns=[c for c in cdef[m]["cpgs"] if c in beta.columns]).median(axis=1).values
        for m in order])
    X  = np.where(np.isnan(X), np.nanmean(X, axis=0), X)         # mean-impute empty modules
    Xs = (X - clock["scaler_mean"]) / clock["scaler_scale"]
    Z  = (Xs - clock["pca_mean"]) @ clock["pca_components"].T    # project onto PCs
    K, mdl = clock["best_K"], clock["clock_by_K"][clock["best_K"]]
    return Z[:, :K] @ np.asarray(mdl["pc_coefs"]) + mdl["intercept"]

pred_age = predict_age(beta, clock)   # predicted epigenetic age, one value per sample
```

Notes:
- Input β-values must be on the **same normalisation footing** as the training data
  (noob + BMIQ; see [Data preprocessing](#data-preprocessing)). The clock handles the
  450K → EPIC probe mismatch internally by intersecting on shared CpGs.
- CpGs absent from your array are dropped from their module's median; a module with no
  measured CpG falls back to the cohort mean.

To **retrain from scratch** or reproduce every figure, follow the
[analysis pipeline](#analysis-pipeline) below.

---

## Installation

```bash
git clone https://github.com/lizanalab/carcedo2026NetworkClocks.git
cd carcedo2026NetworkClocks
conda env create -f environment.yml     # creates env "dnam-network-clock"
conda activate dnam-network-clock
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with **Python 3.11** on Ubuntu 24.04 and macOS 14. Core dependencies (pinned in
`environment.yml` / `requirements.txt`): `pyaging`, `networkx ≥ 3.2`, `infomap`,
`scikit-learn`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `coloraide`.

---

## Repository layout

```
.
├── notebook/                 # analysis notebooks, run in the order in the table below
│   ├── 01_filter_betas.ipynb
│   ├── 02_build_network.ipynb
│   ├── 03_network_clock_ridge.ipynb
│   ├── 03_pca_clock_v3.ipynb
│   ├── test_network_clocks.ipynb
│   ├── plot_figure_2_clock_problems.ipynb
│   ├── plot_communities_network.ipynb
│   ├── plot_figure_6_clocks_comparison.ipynb   # composite figure — consumes CSVs only
│   └── clock_random_vs_pagerank.ipynb          # supplementary
├── DataPreprocessing/        # raw IDAT → β-matrix (R + Python), see Data preprocessing
│   ├── Meta/                 # per-dataset metadata used to build sample sheets
│   ├── 00_SampleSheet.R
│   ├── 01_CellType_Noob_BMIQ.R
│   ├── 02_Beta2M.py
│   ├── 03_ComBat_All.R
│   └── 04_M2Beta.py
├── samples/                  # exact sample lists (see Data)
│   ├── SampleSheet_training.csv     #   1,917 training samples, grouped by dataset
│   └── SampleSheet_validation.csv   #   187 external-validation samples, grouped by dataset
├── outputs/                  # all computed intermediates + figures (see Outputs)
├── analysis/                 # reusable training & module-discovery pipeline (NEW — see below)
│   ├── config.py             #   parameters: rho=0.35, tau=0.70, K=60, seed=42
│   ├── pipeline.py           #   CpG age-filter -> co-methylation network -> Infomap modules
│   ├── clocks.py             #   train Module-PCA / ElasticNet clocks
│   ├── io_utils.py           #   data loading, ComBat, figure saving
│   ├── stats_utils.py        #   reference-clock prediction + significance tests
│   ├── synth.py              #   synthetic-data generator -> run the pipeline without real data
│   └── __init__.py
├── environment.yml           # conda environment
├── requirements.txt          # pip fallback
└── README.md
```

> **Note on file names.** Every notebook is referenced by the **exact** name above throughout
> this README. The composite figure notebook is `plot_figure_6_clocks_comparison.ipynb`; it
> only re-arranges subplots that the upstream notebooks have already written to `outputs/`.

---

## The `analysis/` folder

The `analysis/` folder packages the **training and module-discovery workflow** — CpG
age-filtering, co-methylation network construction, Infomap module detection, and clock
fitting — as small, importable Python functions. It complements the notebooks (which run these
steps on the real data and render the paper's figures) by exposing each step as a documented
function that can be called directly or re-run inside a cross-validation fold. The same steps
also run on a bundled **synthetic dataset**, so the full pipeline is reproducible without the
restricted methylation matrices.

| File | Provides | Pipeline step |
|------|----------|---------------|
| `config.py` | all thresholds in one place: rho = 0.35, tau = 0.70, K = 60, seed = 42 | — |
| `pipeline.py` | `age_filter_cpgs`, `build_network`, `infomap_modules`, `module_median_matrix`, `pagerank_representatives` | **filter -> network -> Infomap -> module summaries** |
| `clocks.py` | `fit_pca_clock`, `pca_k_sweep`, `fit_enet_clock`, `predict_*` | **clock training** |
| `io_utils.py` | `load_methylation`, `combat`, `save_fig`, `ResultsLog` | data loading + ComBat |
| `stats_utils.py` | `predict_reference_clocks`, bootstrap CIs, paired significance tests | benchmark vs reference clocks |
| `synth.py` | `generate`, `ensure_synthetic` — synthetic beta-matrix with a planted age signal | run the pipeline with no downloads |

### Run the whole workflow on synthetic data (no downloads)

```python
import os
os.environ["EPICLOCK_SYNTHETIC"] = "1"            # route the pipeline at synthetic data

from analysis import config, synth, io_utils
from analysis.pipeline import (age_filter_cpgs, build_network, infomap_modules,
                                module_cpg_lists, module_median_matrix)
from analysis.clocks import fit_pca_clock, predict_pca_clock, metrics

synth.ensure_synthetic()                          # writes fictional files into analysis/synthetic_data/
betas, meta = io_utils.load_methylation(config.SYNTH_DIR / "BetaMatrix_0.35.tsv",
                                        config.SYNTH_DIR / "Samplesheet.csv")
ages = meta["Age"].values

keep, rho = age_filter_cpgs(betas, ages, rho_threshold=config.RHO_THRESHOLD)  # step 1: filter CpGs
G         = build_network(betas, keep, tau=config.TAU_THRESHOLD)              # step 2: co-methylation net
modmap    = infomap_modules(G)                                                # step 3: Infomap modules
X         = module_median_matrix(betas, module_cpg_lists(modmap))             # per-module summary matrix
clock     = fit_pca_clock(X.values, ages, K=config.K_PCS_DEFAULT)             # step 4: Module-PCA clock
print(metrics(ages, predict_pca_clock(clock, X.values)))
# -> {'MAE': 3.57, 'RMSE': 4.34, 'R2': 0.95, ...}  on ~300 synthetic samples
```

To run on **real data**, skip the synthetic block and pass your own beta-matrix (CpGs x samples,
tab-separated) and sample sheet (with an `Age` column) to
`io_utils.load_methylation(beta_path, sheet_path)`; the four pipeline steps are identical.
All thresholds live in `analysis/config.py`. See the [Data](#data) section for the training and
validation accessions.

---

## Analysis pipeline

Notebooks are designed to run **in order**. Each one computes results **and** renders the
figures that depend on those results inline, so a notebook can be read end-to-end. Every
notebook reads from `outputs/` (or raw data) and writes back to `outputs/`, so intermediate
results are cached and later steps do not recompute upstream ones.

| # | Notebook | Reads | Computes | Writes / figures |
|---|----------|-------|----------|------------------|
| 1 | `01_filter_betas.ipynb` | raw β-matrix | Filter CpGs by \|Spearman ρ(age)\| ≥ 0.35 | `BetaMatrix_0.35.tsv`, `Correlations.txt`; filter diagnostics |
| 2 | `02_build_network.ipynb` | filtered β-matrix | CpG×CpG correlation network at τ = 0.70 + Infomap clustering | module assignments; correlation histogram |
| 3a | `03_network_clock_ridge.ipynb` | filtered β + modules | Ridge "Network Clock": random *N* CpGs per module, sweep over *N* and #modules *M* | R² vs *N* (Fig 6a), *N*=100 test scatter (Fig 6b) |
| 3b | `03_pca_clock_v3.ipynb` | filtered β + modules | Three PCA variants: **per-module**, whole-network, whole-dataset; sweep over *K* | per-clock *K*-sweep (Fig 6d), top-*K* loadings |
| 4 | `test_network_clocks.ipynb` | trained clocks + 3 EPIC cohorts | Hold-out evaluation on external cohorts vs 5 reference clocks | external scatters, *K*/*N* curves, error boxplots (Fig 6c/e/f) |
| 5 | `plot_figure_2_clock_problems.ipynb` | reference-clock metadata | Clock overlap, array coverage, ρ(age) distributions | Figure 2 (clock problems) |
| 6 | `plot_communities_network.ipynb` | network + modules | Network / module visualisations | community-coloured graph plots |
| 7 | `plot_figure_6_clocks_comparison.ipynb` | CSVs from #3–#4 | **Composite only** — assembles subplots into the main results figure | 2×3 composite (Figure 6) |
| S | `clock_random_vs_pagerank.ipynb` | filtered β + modules | Module selection by size, CpG selection random vs PageRank | Supplementary figure |

---

## Key parameters

All tunable thresholds live in one place (`config.py` / the top cell of `01_filter_betas.ipynb`).
Defaults reproduce the paper:

| Symbol | Value | Meaning |
|--------|-------|---------|
| ρ (rho) | **0.35** | Minimum \|Spearman correlation with age\| for a CpG to enter the network |
| τ (tau) | **0.70** | Minimum \|correlation\| between two CpGs to place an edge |
| *K* | **60** | Number of principal components in the Module-PCA clock |
| *N* | **100** | CpGs sampled per module in the Ridge Network Clock |
| seed | **42** | Global random seed (reproducible folds, PCA, Infomap sampling) |

---

## Reproducing the figures

This repository ships the **trained clocks** (`outputs/05_pca_clock/saved_clocks/`) and the
per-CpG age correlations (`outputs/Correlations.txt`). The large β-matrices and the per-figure
intermediate tables are not redistributed here; regenerate them as follows:

1. Download the training and validation data from the accessions in [Data](#data) (GEO / ArrayExpress).
2. Run `DataPreprocessing/` (noob + BMIQ, ComBat) to build the combined β-matrix.
3. Run the notebooks in order (`01` → `04`); each writes its intermediates to `outputs/`,
   recreating `outputs/05_clock/…` and `outputs/test_network_clock_3cohorts/…`.
4. Run the plotting notebooks — `plot_figure_6_clocks_comparison.ipynb` (main results, Figure 6)
   and `plot_figure_2_clock_problems.ipynb` (Figure 2) — which assemble the tables into the
   final figures.

To only **apply** the published clock to your own samples you need none of the above — see
[Quick start](#quick-start).

---

## Data

**Training set** — 1,917 samples across 12 studies, ages 18–94 (all 450K):

| Accession | Platform | n (healthy/controls) | Source |
|-----------|----------|----------------------|--------|
| GSE87571  | 450K | 664 | GEO |
| GSE51032  | 450K | 424 | GEO |
| GSE125105 | 450K | 210 | GEO |
| GSE42861  | 450K | 209 | GEO |
| GSE61496  | 450K | 150 | GEO |
| GSE59065  | 450K | 97  | GEO |
| GSE87648  | 450K | 73  | GEO |
| GSE81961  | 450K | 25  | GEO |
| E-MTAB-4931 | 450K | 24 | ArrayExpress |
| GSE99624  | 450K | 16  | GEO |
| GSE87640  | 450K | 13  | GEO |
| GSE107737 | 450K | 12  | GEO |

**External validation cohorts** (EPIC v1):

| Dataset | N | Disease groups | Tissue |
|---------|---|----------------|--------|
| GSE235717 | 35 | aging study (no cases/controls) | whole blood |
| GSE217633 | 88 | Control (44) / HIV (44) | whole blood |
| GSE200376 | 64 | Control (19) / psoriatic arthritis (25) / psoriasis vulgaris (20) | **PBMC** |

**Exact sample lists.** The `samples/` folder lists every sample used, grouped by dataset, with
columns `Dataset, Sample_ID, Age, Sex, Disease, CellType`:
`samples/SampleSheet_training.csv` (1,917 samples, 12 datasets) and
`samples/SampleSheet_validation.csv` (187 samples, 3 datasets). Sample IDs are GSM accessions
(E-MTAB-4931 uses its ArrayExpress IDAT basenames); both sheets are derived from the per-dataset
metadata in `DataPreprocessing/Meta/`.

---

## Data preprocessing

Raw IDAT → combined β-matrix. Scripts live in `DataPreprocessing/` and run in order:

| Script | What it does | Arguments |
|--------|--------------|-----------|
| `00_SampleSheet.R` | Read metadata in `DataPreprocessing/Meta/` + raw IDATs for a dataset; build a sample sheet | (1) accession e.g. `GSE87571`, (2) source e.g. `GEO`, (3) index e.g. `1` |
| `01_CellType_Noob_BMIQ.R` | Signal correction, probe filtering, **noob + BMIQ** normalisation; also estimates cell composition | (1) accession, (2) source, (3) index, (4) array e.g. `450K`/`EPIC` |
| `02_Beta2M.py` | Convert a dataset's β-value matrix to M-values | (1) accession, (2) index |
| `03_ComBat_All.R` | ComBat batch-effect removal across datasets | (1) biological covariate e.g. `Age`, `Sex`, or `Without` |
| `04_M2Beta.py` | Convert the combined M-value matrix back to β-values | (1) biological covariate |

The external EPIC cohorts are normalised with the **same** noob + BMIQ pipeline
(`01_CellType_Noob_BMIQ.R` with `array = EPIC`) so training and validation are comparable.

---

## Outputs

`outputs/` is organised by pipeline stage. The main files a reader will want:

| File | Produced by | In repo? | Contents |
|------|-------------|:--:|----------|
| `outputs/05_pca_clock/saved_clocks/pca_clock.pkl` | 03b | ✅ | Trained **Module-PCA clock** (best variant; used in Quick start) |
| `outputs/05_pca_clock/saved_clocks/pca_network_clock.pkl` | 03b | ✅ | Whole-network PCA clock (baseline) |
| `outputs/05_pca_clock/pca_clock_summary.csv` | 03b | ✅ | Module-PCA *K*-sweep metrics (R² / MAE per *K*) |
| `outputs/05_pca_clock/pca_network_clock_summary.csv` | 03b | ✅ | Whole-network PCA *K*-sweep metrics |
| `outputs/Correlations.txt` | 01 | ✅ | Per-CpG Spearman ρ with age (7,329 CpGs) |
| `outputs/01_filtered_betas/BetaMatrix_0.35.tsv` | 01 | ✗ | Age-filtered β-matrix — regenerate from GEO (too large to commit) |
| `outputs/02_network/module_assignments_0.70.csv` | 02 | ✗ | Infomap module label per CpG — regenerate by running notebook 02 |
| `outputs/05_clock/…`, `outputs/test_network_clock_3cohorts/…` | 03–04 | ✗ | Sweep results + external predictions consumed by Fig 6 — regenerate by running notebooks 03–04 |

> **What ships in this repo.** The trained clocks, their *K*-sweep summaries, and the per-CpG
> age-correlation file are committed under `outputs/`, so you can apply the clock immediately
> (see [Quick start](#quick-start)). The large β-matrices and the per-figure intermediate CSVs
> are **not** committed (size / GEO redistribution terms); reproduce them by downloading the
> data from the accessions in [Data](#data) and running the notebooks in order — see
> [Reproducing the figures](#reproducing-the-figures).

---

## Runtime & hardware

| Step | Approx. time | Peak RAM |
|------|--------------|----------|
| Load 12.6 GB β-matrix | ~1–2 min | ~5–6 GB |
| Filter CpGs (01) | ~1 min | ~6 GB |
| Build network + Infomap (02) | ~5–15 min | ~6 GB |
| Fit clocks (03) | ~1–5 min each | ~6 GB |
| External evaluation (04) | ~1 min | ~2 GB |

A workstation with **≥ 16 GB RAM** is sufficient; no GPU required.

---

## Citation

If you use this code or the trained clocks, please cite:

> A. Carcedo *et al.* (2026). *A network approach to DNA methylation clocks.* bioRxiv.
> https://doi.org/10.64898/2026.06.18.733218

---

## License

- **Code:** Apache 2.0 (see `LICENSE`).
- **Raw methylation data:** available from GEO / ArrayExpress under each dataset's original
  terms (accessions above and in the manuscript Data Availability Statement).

---

## Contact

Anton Carcedo — anton.carcedo@umu.se · ORCID [0009-0004-3506-6285](https://orcid.org/0009-0004-3506-6285)
Questions and bug reports: please open a GitHub issue.

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
import pandas as pd, pickle

# beta: a DataFrame of β-values, rows = samples, columns = CpG probe IDs (cg#######)
beta = pd.read_csv("my_betas.tsv", sep="\t", index_col=0)

# load the trained Module-PCA clock (downloaded from the Zenodo archive, see Data section)
with open("outputs/05_pca_clock/module_pca_clock.pkl", "rb") as fh:
    clock = pickle.load(fh)

pred_age = clock.predict(beta)     # predicted epigenetic age, one value per sample
```

Notes:
- Input β-values must be on the **same normalisation footing** as the training data
  (noob + BMIQ; see [Data preprocessing](#data-preprocessing)). The clock handles the
  450K → EPIC probe mismatch internally by intersecting on shared CpGs.
- CpGs absent from your array are mean-imputed against the training distribution.

To **retrain from scratch** or reproduce every figure, follow the
[analysis pipeline](#analysis-pipeline) below.

---

## Installation

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
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
├── notebooks/                # analysis notebooks, run in the order in the table below
│   ├── 01_filter_betas.ipynb
│   ├── 02_build_network.ipynb
│   ├── 03_network_clock_ridge.ipynb
│   ├── 03_pca_clock.ipynb
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
├── outputs/                  # all computed intermediates + figures (see Outputs)
├── environment.yml           # conda environment
├── requirements.txt          # pip fallback
└── README.md
```

> **Note on file names.** Every notebook is referenced by the **exact** name above throughout
> this README. The composite figure notebook is `plot_figure_6_clocks_comparison.ipynb`; it
> only re-arranges subplots that the upstream notebooks have already written to `outputs/`.

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
| 3b | `03_pca_clock.ipynb` | filtered β + modules | Three PCA variants: **per-module**, whole-network, whole-dataset; sweep over *K* | per-clock *K*-sweep (Fig 6d), top-*K* loadings |
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

**Fast path (figures in minutes).** Download the pre-computed intermediates from the
[Zenodo archive](https://doi.org/10.5281/zenodo.XXXXXXX) into `outputs/`, set
`RECOMPUTE = False` at the top of each plotting notebook, then run the composite/plotting
notebooks:

```
plot_figure_6_clocks_comparison.ipynb
plot_figure_2_clock_problems.ipynb
plot_communities_network.ipynb
```

**Full path (from raw IDATs).** Follow [Data preprocessing](#data-preprocessing) to build the
β-matrix, then run notebooks 1 → 7 in order. The 12.6 GB β-matrix passes (network building,
PCA fitting) take a few minutes each on a workstation with ≥ 16 GB RAM; intermediates are
cached to `outputs/` automatically.

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

Derived, ready-to-use artefacts are archived on Zenodo (DOI
[10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)):
the filtered β-matrix (`BetaMatrix_0.35.tsv`), per-CpG age correlations (`Correlations.txt`),
and the trained clock objects (Ridge + three PCA variants).

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

| File | Produced by | Contents |
|------|-------------|----------|
| `BetaMatrix_0.35.tsv` | 01 | Age-filtered β-matrix (CpGs × samples) |
| `Correlations.txt` | 01 | Per-CpG Spearman ρ with age |
| `module_assignments.csv` | 02 | Infomap module label per CpG |
| `module_pca_clock.pkl` | 03b | Trained Module-PCA clock (used in Quick start) |
| `network_clock_ridge.pkl` | 03a | Trained Ridge Network Clock |
| `external_predictions.csv` | 04 | Per-sample predicted vs chronological age, all clocks, 3 cohorts |
| `mae_by_clock.csv` | 04 | MAE / R² per clock per cohort (the Fig 6f numbers) |

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

Versioned repository snapshot: https://doi.org/10.5281/zenodo.XXXXXXX

---

## License

- **Code:** Apache 2.0 (see `LICENSE`).
- **Derived data & trained clock objects** (Zenodo): CC BY 4.0.
- **Raw methylation data:** available from GEO / ArrayExpress under each dataset's original
  terms (accessions above and in the manuscript Data Availability Statement).

---

## Contact

Anton Carcedo — anton.carcedo@umu.se · ORCID [0009-0004-3506-6285](https://orcid.org/0009-0004-3506-6285)
Questions and bug reports: please open a GitHub issue.

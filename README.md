# A network approach to DNA methylation clocks

Code and analysis for **"A network approach to DNA methylation clocks"**, Carcedo et al., 2026.

This repository implements a network-based epigenetic age clock built on DNA methylation data. CpG sites pre-filtered by their Spearman correlation with age are organised into a correlation network, Infomap-clustered into modules, and used to train two complementary clock variants (a Ridge regression on per-module-sampled CpGs and a PCA clock on whole modules), benchmarked against five published reference clocks (Horvath, Hannum, AltumAge, Skin & Blood, Han) on three external EPIC cohorts.

## Repository structure

All analysis notebooks live in `notebooks/` and are designed to run in pipeline order.
Each one **computes results and renders the figures that depend on those results inline**,
so you can read a notebook end-to-end and see both the numbers and the plots they produce.
The final `plot_clocks.ipynb` is purely a composite step: it consumes the CSVs the other
notebooks have written and arranges several of their subplots into the main multi-panel
figure of the paper.

| Notebook | What it computes | Figures rendered inline |
|---|---|---|
| `01_filter_betas.ipynb` | Filter CpGs by |Spearman ρ(age)| ≥ 0.35 | filter diagnostics |
| `02_build_network.ipynb` | CpG×CpG correlation network at τ=0.70 + Infomap clustering | correlation histogram |
| `03_network_clock_ridge.ipynb` | Ridge "Network Clock" — random N per module, M-sweep | R² vs N (panel a), N=100 test scatter (panel b) |
| `03_pca_clock_v2.ipynb` | Three PCA clock variants: modules / whole network / whole dataset | per-clock K-sweep, top-K loadings |
| `test_network_clocks.ipynb` | Hold-out evaluation on three external EPIC cohorts | external scatters, K/N curves, error boxplots |
| `plot_figure_2_clock_problems.ipynb` | Clock overlap, array coverage, ρ(age) distribution analysis | the full clock-problems figure |
| `plot_communities_network.ipynb` | Network/module visualisations | community-coloured graph plots |
| `plot_figure_6_clocks_comparison.ipynb` | **Composite only** — assembles subplots from the above into the main results figure | 2×3 composite figure |

## Reproducing the figures

The fastest path: download the pre-computed intermediates from the [Zenodo archive](https://doi.org/[10.5281/zenodo.XXXXXXX]) into `outputs/`, set `RECOMPUTE = False` where applicable, and run the plotting notebooks (`plot_clocks.ipynb`, `figure_clock_problems.ipynb`, `plot_communities_rho_compare.ipynb`). Figures render in minutes.

The full path (from raw IDATs): see the [Data](#data) section below for accessions, then run the notebooks in order. The 12.6 GB beta matrix passes (network building, PCA clock fitting) take a few minutes each on a workstation with ≥16 GB RAM; intermediate results are cached automatically.

## Installation

```bash
git clone https://github.com/[user]/[repo].git
cd [repo]
conda env create -f environment.yml
conda activate dnam-network-clock
```

Or with pip:

```bash
pip install -r requirements.txt
```

Tested with Python 3.11 on Ubuntu 24.04 and macOS 14. Key dependencies (versions pinned in `environment.yml` / `requirements.txt`): `pyaging`, `networkx ≥ 3.2`, `infomap`, `scikit-learn`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `coloraide`.

## Data

Raw methylation data for the training set (1,917 samples across 12 studies, ages 18–94) are publicly available:

| Accession | Platform | n (healthy/controls) | Source |
|---|---|---|---|
| GSE87571 | 450K | 664 | GEO |
| GSE51032 | 450K | 424 | GEO |
| GSE125105 | 450K | 210 | GEO |
| GSE42861 | 450K | 209 | GEO |
| GSE61496 | 450K | 150 | GEO |
| GSE59065 | 450K | 97 | GEO |
| GSE87648 | 450K | 73 | GEO |
| GSE81961 | 450K | 25 | GEO |
| E-MTAB-4931 | 450K | 24 | ArrayExpress |
| GSE99624 | 450K | 16 | GEO |
| GSE87640 | 450K | 13 | GEO |
| GSE107737 | 450K | 12 | GEO |

External validation cohorts (EPIC v1):

| Dataset | N | Disease groups | Tissue |
|---|---|---|---|
| GSE235717 | 35 | aging study (no cases/controls) | WB |
| GSE217633 | 88 | Control (44) / HIV (44) | WB |
| GSE200376 | 64 | Control (19) / psoriasis_arthritis (25) / psoriasis_vulgaris (20) | **PBMC** |

Preprocessing of the raw IDATs into a combined β-value matrix is described in the manuscript Methods. The derived **filtered beta matrix** (`BetaMatrix_0.35.tsv`), per-CpG **age correlations** (`Correlations.txt`), and **trained clock pickles** (Ridge + three PCA variants) are archived on Zenodo with DOI [10.5281/zenodo.XXXXXXX] for direct reproduction of the figures.

## Citation

If you use this code or the trained clocks, please cite:

> Carcedo, A. et al. (2026). A network approach to DNA methylation clocks. *PLOS Computational Biology*. https://doi.org/[paper DOI]

A versioned snapshot of this repository is archived at:

> https://doi.org/[10.5281/zenodo.XXXXXXX]

## License

Code in this repository is licensed under the **Apache 2.0** (see [LICENSE](LICENSE)).

Derived data and trained clock pickles archived on [Zenodo](https://doi.org/[10.5281/zenodo.XXXXXXX]) are licensed under **CC BY 4.0**.

Raw methylation data are available from GEO and ArrayExpress under each dataset's original terms (accessions in the Data section above and in the manuscript Data Availability Statement).

## Contact

Questions or issues: contact [anton.carcedo@umu.se/ 0009-0004-3506-6285].

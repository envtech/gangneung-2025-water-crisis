# Gangneung 2025 Water Crisis: Code and Data

Code and data for the manuscript:

> Lee, K.-H. *East Asian Arctic-Sector Circulation Anomalies during the Unprecedented 2025 Gangneung Water Crisis: Multi-Resolution Diagnostics and Explainable Machine Learning.* Weather and Climate Extremes (submitted). DOI: `TODO`

**Archived release:** Zenodo DOI `TODO` (created from release `v1.0.0`)

## Overview

The repository reproduces the analyses of the 2025 Gangneung drought and water-supply crisis:

1. Hydroclimatic characterization (station data, SPEI, Mann–Kendall / Sen's slope trends)
2. Reservoir and Landsat NDWI surface-water analysis
3. NCEP2 circulation diagnostics (East Asian Arctic-sector anomalies, Takaya–Nakamura wave-activity flux, Hovmöller)
4. Machine-learning prediction of the 30-day trailing mean precipitation anomaly (six models, expanding-window temporal CV) and SHAP interpretation (explanatory and non-causal)

## Repository structure

```
config/      analysis settings (baseline period, domains, seeds, hyperparameters)
data/        raw/ (redistributable only), processed/, README.md, download_instructions.md
src/         reusable functions
scripts/     numbered analysis scripts (run in order)
results/     figures/ and tables/ produced by the scripts
```

## Installation

```bash
conda env create -f environment.yml
conda activate gangneung2025
```

## Running the analysis

```bash
python scripts/01_met_trends.py
python scripts/02_spei.py
python scripts/03_ncep2_anomaly_sector.py
python scripts/04_waf_hovmoller.py
python scripts/05_ml_cv_shap.py
python scripts/06_landsat_ndwi.py
```

Data that cannot be redistributed must be downloaded first; see `data/download_instructions.md`.

## Figure and table map

| Output | Script | Notes |
|---|---|---|
| Fig. 1 | `TODO` | |
| Fig. 2 | (schematic) | Workflow diagram |
| Fig. 3 | `06_landsat_ndwi.py` | Landsat water extent, Obong Reservoir storage |
| Fig. 4 | `01_met_trends.py` | Monthly hydroclimate 2023–2025 vs. 1993–2022 climatology |
| Fig. 5 | `02_spei.py` | SPEI-1/3/6/12 |
| Fig. 6, Table 3 | `TODO` | Trends, Sen's slope, Mann–Kendall |
| Fig. 7, Table S1 | `03_ncep2_anomaly_sector.py` | August 2025 anomalies, sector sensitivity |
| Fig. 8–10 | `05_ml_cv_shap.py` | Global / local / event-comparison SHAP |
| Fig. 11, Fig. S2 | `04_waf_hovmoller.py` | Z500 anomalies, WAF, Hovmöller |
| Fig. 12 | `TODO` | |
| Fig. S1 | `TODO` | Daily AO index, August 2025 |
| Table 1, 2, 4 | `TODO` | |
| Table S2–S5 | `05_ml_cv_shap.py` | CV folds, hyperparameters, preprocessing, event windows |

## Key analysis settings

- Climatological baseline: 1993–2022 (harmonic regression, annual + semiannual terms for daily anomalies)
- East Asian Arctic sector: 65–85°N, 95–175°E
- Local NCEP2 domain: 37.75°N ± 2.5°, 128.90°E ± 2.5°
- ML target: 30-day trailing mean precipitation anomaly at Gangneung; valid samples 13 Mar 1993–5 Sep 2025 (n = 11,865)
- Cross-validation: five-fold expanding-window `TimeSeriesSplit`; fixed hyperparameters; random seed 42
- Full details: `config/settings.yaml` and Supporting Material (Text S1, Tables S1–S5)

## Data availability

See `data/README.md` for sources, licenses, and what is included.

## Citation

See `CITATION.cff`, or cite the paper and the Zenodo archive.

## License

Code: MIT (`LICENSE`). Data and figures produced by the authors: CC BY 4.0 (`LICENSE-data`). Third-party data keep their original terms.

## Contact

Kwon-Ho Lee — `TODO: affiliation / contact`

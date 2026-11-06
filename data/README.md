# Data

| Dataset | Variables | Source | Included here? | License / terms |
|---|---|---|---|---|
| Gangneung ASOS (Station 105) | daily precipitation, temperature, humidity, wind, pressure | Korea Meteorological Administration (KMA) | Yes: `raw/gangneung_asos_1993_2025_merged.csv` | `TODO`: confirm KMA terms of use and add the required attribution |
| Obong Reservoir | daily storage rate | Korea Rural Community Corporation (KRC) | `TODO` (check KRC terms) | `TODO` |
| NCEP/DOE Reanalysis 2 | T2m, precipitation rate, precipitable water, Z500, V500, winds | NOAA PSL | No (download) | NOAA open data |
| Arctic Oscillation index | daily AO | NOAA CPC | No (download) or processed series | NOAA open data |
| Landsat 8/9 | NDWI scenes (Apr–Nov 2025) | USGS | No (download) | USGS public domain |
| Derived products | monthly SPEI, anomalies, ML feature table, SHAP outputs | This study | Yes, `processed/` | CC BY 4.0 |

## `raw/gangneung_asos_1993_2025_merged.csv`

Daily ASOS observations at Gangneung (station 105), 1 Jan 1993 – 1 Jan 2026 (12,054 rows), downloaded from the KMA open-data portal as four period files and concatenated without modification. The file is UTF-8 with BOM and keeps the original Korean column headers (62 columns). Columns used in `scripts/01_met_trends.py`:

| CSV column | Meaning | Name in code |
|---|---|---|
| `지점명` | station name (강릉 = Gangneung) | `site_name` |
| `일시` | date | `date` |
| `평균기온(°C)` | daily mean air temperature | `tmean` |
| `일강수량(mm)` | daily precipitation (empty = no precipitation) | `precip` |
| `평균 풍속(m/s)` | daily mean wind speed | `wspd` |
| `평균 상대습도(%)` | daily mean relative humidity | `rh` |
| `평균 현지기압(hPa)` | daily mean station pressure | `pressure` |

## Processed files

| File | Description | Columns |
|---|---|---|
| `processed/TODO.csv` | | |

## Notes

- ML sample period: 13 Mar 1993–5 Sep 2025 (11,865 daily samples after removing missing rows).
- Anomalies are relative to the 1993–2022 baseline.
- Do not commit files over 100 MB; use Zenodo for large files.

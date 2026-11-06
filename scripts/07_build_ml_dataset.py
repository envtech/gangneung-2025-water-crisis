#!/usr/bin/env python3
"""
05_build_ml_dataset.py
======================
Build the machine-learning dataset for the precipitation-deficit analysis
(Text S1.1-S1.2, Table S4 of the Supporting Material).

Steps
-----
1. Read the base daily table (Gangneung ASOS variables, ET0 and reservoir storage).
2. Extract daily NCEP2 variables as area means over (i) a 5 x 5 degree box around
   Gangneung and (ii) the East Asian Arctic sector (65-85N, 95-175E), unweighted
   grid-cell means; precipitation rate is converted to mm/day.
3. Merge the daily NOAA CPC Arctic Oscillation (AO) index.
4. Fill gaps of at most 5 consecutive days by linear interpolation.
5. Convert every variable to a daily anomaly relative to a harmonic climatology
   (annual + semiannual harmonics) fitted to 1993-2022 and applied to the full
   period.
6. Add trailing 7-, 30- and 90-day means of the predictor anomalies (a value is
   kept when at least int(0.8 * window) observations are available) and the target
   ``precip_anom_ma30`` (30-day trailing mean of the Gangneung ASOS precipitation
   anomaly).
7. Select predictors: only anomaly variables; Gangneung ASOS precipitation and
   the local NCEP2 precipitation rate (and everything derived from them) are
   excluded to avoid target leakage.  The Arctic-sector precipitation rate is kept.
8. Remove multicollinearity by iterative variance-inflation-factor (VIF)
   screening: the predictor with the largest VIF is dropped until all VIF <= 10.

Input
-----
    data/processed/base_dataset_asos_reservoir.csv
        daily table with the columns ``date``, ``tmean``, ``precip``, ``ET0_mm`` and
        ``storage_rate`` (reservoir storage rate, %); other columns are ignored.
        ASOS/ET0 columns come from ``data/processed/gangneung_daily_et0.csv``
        (``02_compute_spei.py``); the reservoir data are merged in separately.
    data/raw/ncep2_daily/{pressure,gaussian_grid,surface}/<name>.<year>.nc
        NCEP2 daily files, one per year, 1993-2025 (see data/download_instructions.md)
    data/raw/norm.daily.ao.index.b500101.current.ascii
        NOAA CPC daily AO index (downloaded automatically if the file is missing;
        keep the copy used for the paper so that the analysis is reproducible)

Output
------
    data/processed/final_ml_dataset_vif_selected_precip_target.csv   input of 05_ml_cv_shap.py
    data/interim/final_ml_dataset.csv        all raw and anomaly variables (large, not tracked)
    results/tables/vif_report.csv            final VIF of the retained predictors
    results/tables/vif_removed_features.csv  predictors removed by the VIF screening
    results/tables/na_after_merge_report.csv, na_diagnostic_report.csv   missing-value reports
    results/tables/table_s4_preprocessing.csv                        Table S4

Usage
-----
    python scripts/05_build_ml_dataset.py
    python scripts/05_build_ml_dataset.py --base my_base.csv --ncep2-dir /path/to/ncep2_daily

Requires: xarray, netCDF4, numpy, pandas, statsmodels.
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = REPO_ROOT / "data" / "processed" / "base_dataset_asos_reservoir.csv"
DEFAULT_NCEP2_DIR = REPO_ROOT / "data" / "raw" / "ncep2_daily"
DEFAULT_AO_FILE = REPO_ROOT / "data" / "raw" / "norm.daily.ao.index.b500101.current.ascii"
DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"
DEFAULT_INTERIM = REPO_ROOT / "data" / "interim"
DEFAULT_TABLEDIR = REPO_ROOT / "results" / "tables"
AO_URL = "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.ao.index.b500101.current.ascii"

YEAR_START, YEAR_END = 1993, 2025   # analysis period (matches the base dataset)
CLIM_START, CLIM_END = 1993, 2022   # baseline period of the harmonic climatology

GN_LAT, GN_LON = 37.75, 128.90      # Gangneung
BOX_HALF_DEG = 2.5                  # half width of the Gangneung box (5 x 5 deg)
ARCTIC_LAT = (65, 85)               # East Asian Arctic sector
ARCTIC_EA_LON = (95, 175)

VIF_THRESHOLD = 10.0
MOVING_WINDOWS = [7, 30, 90]
MIN_VALID_FRACTION = 0.8            # minimum share of valid values in a moving window
MAX_INTERP_GAP_DAYS = 5

ML_TARGET_COL = "precip_anom_ma30"  # 30-day trailing mean of the ASOS precipitation anomaly

# Target-leakage control: the ASOS precipitation and the local NCEP2 precipitation
# rate (a direct proxy of local rainfall) are not used as predictors.
EXCLUDE_LOCAL_PRECIP_BASE = {"precip", "prate"}

# NCEP2 daily sources: alias -> (file under the NCEP2 directory, variable, pressure level)
# The file name contains the year as {year}.
NCEP2_SOURCES = {
    "z500":   ("pressure/hgt.{year}.nc",                 "hgt",    500),
    "u500":   ("pressure/uwnd.{year}.nc",                "uwnd",   500),
    "v500":   ("pressure/vwnd.{year}.nc",                "vwnd",   500),
    "t2m":    ("gaussian_grid/air.2m.gauss.{year}.nc",   "air",    None),
    "prate":  ("gaussian_grid/prate.sfc.gauss.{year}.nc", "prate", None),
    "skt":    ("gaussian_grid/skt.sfc.gauss.{year}.nc",  "skt",    None),
    "pr_wtr": ("surface/pr_wtr.eatm.{year}.nc",          "pr_wtr", None),
}

# Variables (before the "_anom" suffix) converted to anomalies and used as features
BASE_FEATURE_COLS = [
    "tmean", "precip", "ET0_mm",                                     # ASOS / ET0
    "z500", "u500", "v500", "t2m", "prate", "skt", "pr_wtr",          # NCEP2, Gangneung box
    "ao_index",                                                        # AO index
    "z500_arctic", "u500_arctic", "v500_arctic", "t2m_arctic",        # NCEP2, Arctic sector
    "prate_arctic", "skt_arctic", "pr_wtr_arctic",                    #   (65-85N, 95-175E)
]


# -----------------------------------------------------------------------------
# 1. NCEP2 daily area means
# -----------------------------------------------------------------------------
def load_daily_var_multiyear(ncep2_dir, template, varname, level=None,
                             lat_range=None, lon_range=None):
    """Read the yearly daily files of one variable and return its area-mean series.

    By default the area is the Gangneung box (GN_LAT/GN_LON +/- BOX_HALF_DEG);
    pass lat_range/lon_range for another region (e.g. the Arctic sector).
    The mean is an unweighted mean of the grid cells in the box.
    Returns a DataFrame with the columns ``date`` and ``varname``.
    """
    if lat_range is None:
        lat_range = (GN_LAT - BOX_HALF_DEG, GN_LAT + BOX_HALF_DEG)
    if lon_range is None:
        lon_range = (GN_LON - BOX_HALF_DEG, GN_LON + BOX_HALF_DEG)

    frames, missing_years = [], []
    for year in range(YEAR_START, YEAR_END + 1):
        fpath = Path(ncep2_dir) / template.format(year=year)
        if not fpath.exists():
            missing_years.append(year)
            continue
        with xr.open_dataset(fpath) as ds:
            if varname not in ds.data_vars:
                raise KeyError(f"'{varname}' not found in {fpath}. Variables: {list(ds.data_vars)}")
            da = ds[varname]
            lat_name = "lat" if "lat" in da.coords else "latitude"
            lon_name = "lon" if "lon" in da.coords else "longitude"
            da = da.sortby(lat_name)  # ascending latitude (needed for slice selection)

            if level is not None:
                lvl_name = "level" if "level" in da.coords else "lev"
                da = da.sel({lvl_name: level}, method="nearest")
            extra = [d for d in da.dims if d not in ("time", lat_name, lon_name) and da.sizes[d] == 1]
            if extra:
                da = da.squeeze(extra, drop=True)

            box = da.sel({lat_name: slice(*lat_range), lon_name: slice(*lon_range)})
            series = box.mean([lat_name, lon_name])
            frames.append(series.to_dataframe(name=varname)[[varname]])
    if not frames:
        raise FileNotFoundError(
            f"No file matching {Path(ncep2_dir) / template} found for {YEAR_START}-{YEAR_END}. "
            "See data/download_instructions.md.")
    if missing_years:
        print(f"  WARNING: {template}: no file for years {missing_years}")
    out = pd.concat(frames).reset_index()
    out.columns = ["date", varname]
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out


def build_ncep2_table(ncep2_dir):
    """Daily NCEP2 table: Gangneung-box variables and Arctic-sector variables."""
    print("Loading NCEP2 daily variables (many yearly files; this can take a while) ...")
    local = {}
    for alias, (template, varname, level) in NCEP2_SOURCES.items():
        df_var = load_daily_var_multiyear(ncep2_dir, template, varname, level=level)
        local[alias] = df_var.rename(columns={varname: alias})

    ncep2_df = None
    for df_var in local.values():
        ncep2_df = df_var if ncep2_df is None else pd.merge(ncep2_df, df_var, on="date", how="outer")
    ncep2_df["prate"] = ncep2_df["prate"] * 86400  # kg m-2 s-1 -> mm day-1
    print("NCEP2 columns (Gangneung box):", [c for c in ncep2_df.columns if c != "date"])

    print(f"\nLoading Arctic-sector ({ARCTIC_LAT[0]}-{ARCTIC_LAT[1]}N, "
          f"{ARCTIC_EA_LON[0]}-{ARCTIC_EA_LON[1]}E) NCEP2 variables ...")
    for alias, (template, varname, level) in NCEP2_SOURCES.items():
        arctic_var = load_daily_var_multiyear(ncep2_dir, template, varname, level=level,
                                              lat_range=ARCTIC_LAT, lon_range=ARCTIC_EA_LON)
        arctic_var = arctic_var.rename(columns={varname: f"{alias}_arctic"})
        ncep2_df = pd.merge(ncep2_df, arctic_var, on="date", how="left")
        print(f"  added: {alias}_arctic")
    ncep2_df["prate_arctic"] = ncep2_df["prate_arctic"] * 86400  # mm day-1

    print(f"NCEP2 daily table: {len(ncep2_df)} rows, "
          f"{ncep2_df['date'].min().date()} to {ncep2_df['date'].max().date()}")
    return ncep2_df


# -----------------------------------------------------------------------------
# 2. AO index
# -----------------------------------------------------------------------------
def load_ao(ao_file):
    """Daily NOAA CPC AO index as a DataFrame (date, ao_index); -99 = missing."""
    ao_file = Path(ao_file)
    if not ao_file.exists():
        print(f"Downloading the AO index to {ao_file} ...")
        ao_file.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(AO_URL, ao_file)
    ao_df = pd.read_fwf(ao_file, colspecs=[(0, 4), (4, 7), (7, 10), (10, 17)],
                        header=None, names=["year", "month", "day", "ao_index"])
    ao_df["ao_index"] = ao_df["ao_index"].replace(-99.000, np.nan)
    ao_df["date"] = pd.to_datetime(ao_df[["year", "month", "day"]])
    print(f"AO index loaded: {len(ao_df)} rows")
    return ao_df[["date", "ao_index"]]


# -----------------------------------------------------------------------------
# 3. Merge, anomalies, moving means
# -----------------------------------------------------------------------------
def merge_all(base_csv, ncep2_df, ao_df, tabledir):
    """Merge base table, NCEP2 and AO; report and fill short gaps."""
    base = pd.read_csv(base_csv, parse_dates=["date"]).sort_values("date")
    merged = base.merge(ncep2_df, on="date", how="left").merge(ao_df, on="date", how="left")
    print(f"\nMerged table: {len(merged)} rows")

    merge_na = merged.isna().sum()
    merge_na_df = merge_na[merge_na > 0].to_frame("n_missing")
    merge_na_df["pct_missing"] = (merge_na_df["n_missing"] / len(merged) * 100).round(1)
    merge_na_df.to_csv(tabledir / "na_after_merge_report.csv")
    print("Missing values after the merge (saved to na_after_merge_report.csv):")
    print(merge_na_df)

    # Short gaps: linear interpolation over at most MAX_INTERP_GAP_DAYS consecutive values
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].interpolate(method="linear", limit=MAX_INTERP_GAP_DAYS)
    return merged


def harmonic_design_matrix(doy_frac):
    """Design matrix of the annual + semiannual harmonic regression."""
    w = 2 * np.pi * doy_frac
    return np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])


def add_anomalies(merged):
    """Add ``<variable>_anom`` columns: deviation from a harmonic climatology fitted
    to CLIM_START-CLIM_END.  Anomalies are used (instead of raw values) because the
    question is why 2025 was unusual relative to the seasonal cycle."""
    base_feature_cols = [c for c in BASE_FEATURE_COLS if c in merged.columns]
    anomaly_targets = ["storage_rate"] + base_feature_cols

    merged["doy"] = merged["date"].dt.dayofyear
    merged["doy_frac"] = merged["doy"] / 365.25
    clim_mask = (merged["date"].dt.year >= CLIM_START) & (merged["date"].dt.year <= CLIM_END)

    anomaly_cols = []
    for col in anomaly_targets:
        if col not in merged.columns:
            continue
        valid = clim_mask & merged[col].notna()
        x_design = harmonic_design_matrix(merged.loc[valid, "doy_frac"].values)
        y_design = merged.loc[valid, col].values
        coefs, *_ = np.linalg.lstsq(x_design, y_design, rcond=None)
        climatology = harmonic_design_matrix(merged["doy_frac"].values) @ coefs
        merged[f"{col}_anom"] = merged[col] - climatology
        anomaly_cols.append(f"{col}_anom")

    print(f"\n{len(anomaly_cols)} anomaly variables created (suffix _anom)")
    return merged, base_feature_cols, anomaly_cols


def add_moving_means(merged, anomaly_cols):
    """Trailing means (MOVING_WINDOWS days) of the anomaly variables."""
    merged = merged.sort_values("date").reset_index(drop=True)
    for col in [c for c in anomaly_cols if c != "storage_rate_anom"]:
        for w in MOVING_WINDOWS:
            merged[f"{col}_ma{w}"] = merged[col].rolling(
                w, min_periods=int(w * MIN_VALID_FRACTION)).mean()
    print(f"Moving-mean variables added. Total columns: {len(merged.columns)}")
    return merged


# -----------------------------------------------------------------------------
# 4. Predictor selection and VIF screening
# -----------------------------------------------------------------------------
def select_predictors(merged, base_feature_cols):
    """Anomaly predictors (and their moving means) without target-leaking variables."""
    predictor_base = [c for c in base_feature_cols if c not in EXCLUDE_LOCAL_PRECIP_BASE]
    anom_whitelist = [f"{c}_anom" for c in predictor_base]
    ma_cols = [c for c in merged.columns
               if any(c.startswith(f"{b}_anom_ma") for b in predictor_base)]
    feature_cols = [c for c in anom_whitelist if c in merged.columns] + ma_cols

    def is_leakage_feature(c):
        return (c == ML_TARGET_COL or c.startswith("precip_anom") or c.startswith("prate_anom"))

    return [c for c in feature_cols if not is_leakage_feature(c)]


def iterative_vif_elimination(x, threshold=VIF_THRESHOLD):
    """Drop the predictor with the largest VIF until all VIF <= threshold.

    Returns the reduced DataFrame and a log of (removed predictor, VIF at removal).
    """
    x_cur = x.copy()
    removed_log = []
    while True:
        vif_vals = pd.Series(
            [variance_inflation_factor(x_cur.values, i) for i in range(x_cur.shape[1])],
            index=x_cur.columns)
        max_vif = vif_vals.max()
        if max_vif <= threshold or x_cur.shape[1] <= 1:
            break
        worst = vif_vals.idxmax()
        removed_log.append((worst, round(max_vif, 1)))
        print(f"  removed: {worst} (VIF={max_vif:.1f})")
        x_cur = x_cur.drop(columns=[worst])
    return x_cur, removed_log


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build the ML dataset (anomalies, VIF screening).")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="base daily table (default: %(default)s)")
    parser.add_argument("--ncep2-dir", type=Path, default=DEFAULT_NCEP2_DIR,
                        help="NCEP2 daily files (default: %(default)s)")
    parser.add_argument("--ao-file", type=Path, default=DEFAULT_AO_FILE, help="AO index file (default: %(default)s)")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED,
                        help="final dataset directory (default: %(default)s)")
    parser.add_argument("--interim-dir", type=Path, default=DEFAULT_INTERIM,
                        help="intermediate files (default: %(default)s)")
    parser.add_argument("--tabledir", type=Path, default=DEFAULT_TABLEDIR,
                        help="report tables (default: %(default)s)")
    args = parser.parse_args()

    if not args.base.exists():
        raise FileNotFoundError(
            f"Base dataset not found: {args.base}\n"
            "It must contain: date, tmean, precip, ET0_mm, storage_rate "
            "(see the docstring of this script).")
    for d in (args.processed_dir, args.interim_dir, args.tabledir):
        d.mkdir(parents=True, exist_ok=True)

    ncep2_df = build_ncep2_table(args.ncep2_dir)
    ao_df = load_ao(args.ao_file)
    merged = merge_all(args.base, ncep2_df, ao_df, args.tabledir)

    merged, base_feature_cols, anomaly_cols = add_anomalies(merged)
    merged = add_moving_means(merged, anomaly_cols)
    if ML_TARGET_COL not in merged.columns:
        raise KeyError(f"Target '{ML_TARGET_COL}' was not created; check the precipitation "
                       "anomaly and moving-mean steps.")
    merged.to_csv(args.interim_dir / "final_ml_dataset.csv", index=False)
    print(f"Saved: {args.interim_dir / 'final_ml_dataset.csv'} (raw values and anomalies)")

    # ---- predictor selection and diagnostics ----
    feature_cols = select_predictors(merged, base_feature_cols)
    print(f"\nPredictor candidates: {len(feature_cols)}")
    print(feature_cols)

    na_report = merged[feature_cols].isna().sum().sort_values(ascending=False)
    na_report_df = pd.DataFrame({"n_missing": na_report,
                                 "pct_missing": (na_report / len(merged) * 100).round(1)})
    na_report_df.to_csv(args.tabledir / "na_diagnostic_report.csv")
    print("\nMissing values per predictor (top 15; full list in na_diagnostic_report.csv):")
    print(na_report_df.head(15))

    n_after_dropna = merged[feature_cols].dropna().shape[0]
    print(f"\nRows before dropna(): {len(merged)}; after dropna(): {n_after_dropna}")
    if n_after_dropna == 0:
        raise ValueError(
            "No complete rows. Check na_diagnostic_report.csv: predictors with pct_missing near "
            "100 usually point to NCEP2 files whose names or paths differ from NCEP2_SOURCES.")
    x = merged[feature_cols].dropna()

    # ---- VIF screening ----
    print(f"\nIterative VIF elimination (threshold = {VIF_THRESHOLD}) ...")
    x_reduced, removed_log = iterative_vif_elimination(x, VIF_THRESHOLD)
    removed_df = pd.DataFrame(removed_log, columns=["feature", "VIF_at_removal"])
    removed_df.to_csv(args.tabledir / "vif_removed_features.csv", index=False)
    print(f"\nRemoved predictors ({len(removed_log)}):")
    print(removed_df.to_string(index=False))

    final_vif = pd.DataFrame({
        "feature": x_reduced.columns,
        "VIF": [variance_inflation_factor(x_reduced.values, i) for i in range(x_reduced.shape[1])],
    }).sort_values("VIF", ascending=False)
    final_vif.to_csv(args.tabledir / "vif_report.csv", index=False)
    print(f"\nRetained predictors ({x_reduced.shape[1]}; all VIF <= {VIF_THRESHOLD}):")
    print(final_vif.to_string(index=False))

    # ---- final dataset ----
    final_feature_cols = list(x_reduced.columns)
    output_cols = ["date", ML_TARGET_COL, "storage_rate", "storage_rate_anom"] + final_feature_cols
    output_cols = list(dict.fromkeys([c for c in output_cols if c in merged.columns]))
    final_file = args.processed_dir / "final_ml_dataset_vif_selected_precip_target.csv"
    merged[output_cols].to_csv(final_file, index=False)
    print(f"\nSaved: {final_file}\nTarget: {ML_TARGET_COL}")

    # ---- Table S4 (preprocessing settings) ----
    s4_rows = [
        ("AI analysis period", f"{YEAR_START}-{YEAR_END}"),
        ("Climatological baseline", f"{CLIM_START}-{CLIM_END}"),
        ("Primary target", ML_TARGET_COL),
        ("Local NCEP2 domain", f"{GN_LAT}N ± {BOX_HALF_DEG}°, {GN_LON}E ± {BOX_HALF_DEG}°"),
        ("East Asian Arctic sector",
         f"{ARCTIC_LAT[0]}-{ARCTIC_LAT[1]}N, {ARCTIC_EA_LON[0]}-{ARCTIC_EA_LON[1]}E"),
        ("Spatial averaging", "Unweighted grid-cell mean"),
        ("Daily anomaly", "Harmonic regression: annual + semiannual terms"),
        ("Moving-average windows", ", ".join(map(str, MOVING_WINDOWS)) + " days; trailing"),
        ("Minimum observations per moving window", "80%"),
        ("VIF threshold", str(VIF_THRESHOLD)),
        ("Missing-value interpolation", f"Linear; maximum {MAX_INTERP_GAP_DAYS} consecutive values"),
        ("Leakage control",
         "Exclude ASOS precip-derived predictors and local NCEP2 prate-derived predictors"),
    ]
    s4_file = args.tabledir / "table_s4_preprocessing.csv"
    pd.DataFrame(s4_rows, columns=["Component", "Setting"]).to_csv(s4_file, index=False)
    print(f"Saved: {s4_file}")


if __name__ == "__main__":
    main()

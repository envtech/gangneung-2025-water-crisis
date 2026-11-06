#!/usr/bin/env python3
"""
04_fig12_arctic_v500_longitude.py
=================================
Fig. 12: longitude profile of the August 2025 500-hPa meridional-wind (V500)
anomaly in the Arctic band (65-85N), with the East Asian Arctic sector
(95-175E) highlighted and the 1979-2025 historical rank of the sector mean.

Method
------
* Data: NCEP/DOE Reanalysis 2 monthly-mean V500, August of 1979-2025.
* Latitude mean: cosine-latitude weighted mean over 65-85N (as in Fig. 7), giving
  a profile as a function of longitude.
* Anomaly: August value minus the 1993-2022 August mean at each longitude.
* Envelope: empirical 5th-95th percentile of the 1993-2022 August anomalies at
  each longitude (descriptive, not a formal significance interval).
* Sector mean: mean of the profile over 95-175E (equal to the cosine-latitude
  weighted box mean of Fig. 7; this is asserted in the code).
* Rank: position of 2025 among the 1979-2025 sector means, most negative = 1.
* No Arctic-to-Korea causality is inferred from this profile alone.

Input (see data/download_instructions.md)
-----------------------------------------
    data/raw/ncep2/pressure/vwnd.mon.mean.nc

Output
------
    results/figures/fig12_arctic_v500_longitude.png            (400 dpi)
    results/tables/fig12_longitude_profile.csv                  profile, envelope, sector flag
    results/tables/fig12_sector_historical_rank.csv             sector mean for every year

Usage
-----
    python scripts/04_fig12_arctic_v500_longitude.py
    python scripts/04_fig12_arctic_v500_longitude.py --vwnd /path/to/vwnd.mon.mean.nc

Requires: numpy, pandas, xarray, netCDF4, matplotlib.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: the figure is written to a file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import MultipleLocator

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VWND = REPO_ROOT / "data" / "raw" / "ncep2" / "pressure" / "vwnd.mon.mean.nc"
DEFAULT_FIGDIR = REPO_ROOT / "results" / "figures"
DEFAULT_TABLEDIR = REPO_ROOT / "results" / "tables"
FIG_FILE = "fig12_arctic_v500_longitude.png"

TARGET_YEAR, TARGET_MONTH = 2025, 8
CLIM_START, CLIM_END = 1993, 2022   # baseline of the anomalies and of the envelope
RANK_START, RANK_END = 1979, 2025   # period of the historical ranking
LEVEL_HPA = 500
ARCTIC_LAT = (65.0, 85.0)
EAST_ASIA_LON = (95.0, 175.0)       # East Asian Arctic sector
HIST_PCT_LOW, HIST_PCT_HIGH = 5, 95
FIGSIZE = (9.2, 4.8)
OUTPUT_DPI = 400

# Values reported in the manuscript; a warning is printed when the data disagree
MANUSCRIPT_SECTOR_ANOMALY = -5.38   # m/s
MANUSCRIPT_RANK = 1
CHECK_TOLERANCE = 0.05              # m/s


# -----------------------------------------------------------------------------
# Computation
# -----------------------------------------------------------------------------
def load_august_v500(vwnd_file):
    """August V500 for RANK_START-RANK_END, 65-85N, longitudes in [0, 360)."""
    vwnd_file = Path(vwnd_file)
    if not vwnd_file.is_file():
        raise FileNotFoundError(f"NCEP2 monthly file not found: {vwnd_file}\nSee data/download_instructions.md.")

    with xr.open_dataset(vwnd_file) as ds:
        if "vwnd" not in ds.data_vars:
            raise KeyError(f"Variable 'vwnd' not found; available: {list(ds.data_vars)}")
        da = ds["vwnd"]
        lat_name = "lat" if "lat" in da.coords else "latitude"
        lon_name = "lon" if "lon" in da.coords else "longitude"
        level_name = "level" if "level" in da.coords else "lev"
        if lat_name not in da.coords or lon_name not in da.coords or level_name not in da.coords:
            raise ValueError("Expected latitude, longitude and pressure-level coordinates.")

        da = da.sel({level_name: LEVEL_HPA}, method="nearest")
        selected_level = float(da[level_name].values)
        if abs(selected_level - LEVEL_HPA) > 0.1:
            raise ValueError(f"Requested {LEVEL_HPA} hPa, but the closest available level is {selected_level} hPa")

        da = da.assign_coords({lon_name: (da[lon_name] % 360)}).sortby(lon_name)  # longitudes in [0, 360)
        da = da.sortby(lat_name).sel({lat_name: slice(*ARCTIC_LAT)})
        if da.sizes[lat_name] < 2:
            raise ValueError(f"Insufficient latitude grid points in {ARCTIC_LAT}")
        da = da.sel(time=da.time.dt.month == TARGET_MONTH)
        da = da.sel(time=(da.time.dt.year >= RANK_START) & (da.time.dt.year <= RANK_END))
        da = da.load()

    years = da.time.dt.year.values.astype(int)
    if len(np.unique(years)) != len(years):
        raise ValueError("Expected one August monthly record per year. Check the input dataset.")
    missing = sorted(set(range(RANK_START, RANK_END + 1)) - set(years))
    if missing:
        raise ValueError(f"The historical rank requires the full {RANK_START}-{RANK_END} record. Missing: {missing}")
    return da, lat_name, lon_name


def compute_profile(da, lat_name, lon_name):
    """Anomaly profile, envelope, sector statistics and rank table."""
    # Cosine-latitude weighted latitude mean (same weighting as Fig. 7)
    weights = np.cos(np.deg2rad(da[lat_name]))
    band = da.weighted(weights).mean(dim=lat_name)  # (time, lon)
    base = band.sel(time=(band.time.dt.year >= CLIM_START) & (band.time.dt.year <= CLIM_END))
    aug_anom = band - base.mean(dim="time", skipna=True)

    target = aug_anom.sel(time=aug_anom.time.dt.year == TARGET_YEAR).isel(time=0)
    hist = aug_anom.sel(time=(aug_anom.time.dt.year >= CLIM_START) & (aug_anom.time.dt.year <= CLIM_END))
    lower = hist.quantile(HIST_PCT_LOW / 100, dim="time").values.astype(float)
    upper = hist.quantile(HIST_PCT_HIGH / 100, dim="time").values.astype(float)
    lons = target[lon_name].values.astype(float)
    vals = target.values.astype(float)

    # Sector mean for every year and the 1979-2025 ranking (most negative = 1)
    sector_mask = (lons >= EAST_ASIA_LON[0]) & (lons <= EAST_ASIA_LON[1])
    if sector_mask.sum() < 2:
        raise ValueError("Insufficient longitude grid points in the East Asian Arctic sector")
    sector_series = aug_anom.isel({lon_name: np.flatnonzero(sector_mask)}).mean(lon_name)
    rank_table = pd.DataFrame({
        "year": sector_series.time.dt.year.values.astype(int),
        "sector_v500_anomaly_m_s": sector_series.values.astype(float),
    }).sort_values("year")
    target_sector = float(rank_table.loc[rank_table.year == TARGET_YEAR, "sector_v500_anomaly_m_s"].iloc[0])

    # Independent check: the Fig. 7 box mean (cosine-latitude weighted over lat and lon)
    target_full = da.sel(time=da.time.dt.year == TARGET_YEAR).isel(time=0)
    clim_full = da.sel(time=(da.time.dt.year >= CLIM_START) & (da.time.dt.year <= CLIM_END)).mean("time")
    sub = (target_full - clim_full).sel({lon_name: slice(*EAST_ASIA_LON)})
    direct_weighted = float(sub.weighted(np.cos(np.deg2rad(sub[lat_name]))).mean((lat_name, lon_name)))
    unweighted = float(sub.mean((lat_name, lon_name)))
    assert np.isclose(target_sector, direct_weighted, atol=1e-7), (target_sector, direct_weighted)

    rank = int((rank_table.sector_v500_anomaly_m_s < target_sector).sum() + 1)
    return {"lons": lons, "vals": vals, "lower": lower, "upper": upper, "sector_mask": sector_mask,
            "rank_table": rank_table, "target_sector": target_sector, "rank": rank,
            "n_years": len(rank_table), "direct_weighted": direct_weighted, "unweighted": unweighted}


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
def make_figure(res):
    lons, vals = res["lons"], res["vals"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(lons, res["lower"], res["upper"], color="0.76", alpha=0.55, lw=0, zorder=1,
                    label=f"Historical 5th–95th percentile ({CLIM_START}–{CLIM_END})")
    ax.axvspan(*EAST_ASIA_LON, color="#e2b468", alpha=0.27, zorder=0,
               label="East Asian Arctic sector (95–175°E)")
    ax.axhline(0, color="0.35", linewidth=0.9, zorder=2)
    ax.fill_between(lons, 0, vals, where=vals >= 0, color="#486fa6", alpha=0.42, zorder=3, interpolate=True)
    ax.fill_between(lons, 0, vals, where=vals < 0, color="#b95858", alpha=0.43, zorder=3, interpolate=True)
    ax.plot(lons, vals, color="black", lw=1.7, label="August 2025 anomaly", zorder=4)

    ax.set_xlim(0, 360)
    ax.xaxis.set_major_locator(MultipleLocator(45))
    ax.set_xticks(np.arange(0, 361, 45), [f"{d}°E" for d in range(0, 361, 45)])
    ax.set_ylabel("500-hPa meridional-wind anomaly (m s$^{-1}$)", fontsize=14)
    ax.set_xlabel("Longitude", fontsize=14)
    ax.text(0.52, 0.03,
            f"East Asian Arctic-sector mean: {res['target_sector']:+.2f} m s$^{{-1}}$  \n  "
            f"Rank: {res['rank']}/{res['n_years']} ({RANK_START}–{RANK_END})",
            transform=ax.transAxes, fontsize=12, va="bottom", ha="left",
            bbox=dict(facecolor="white", edgecolor="0.80", alpha=0.9, pad=3.5))
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(axis="y", color="0.86", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fig. 12: Arctic-band V500 anomaly by longitude, August 2025.")
    parser.add_argument("--vwnd", type=Path, default=DEFAULT_VWND, help="NCEP2 vwnd.mon.mean.nc (default: %(default)s)")
    parser.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR, help="figure output (default: %(default)s)")
    parser.add_argument("--tabledir", type=Path, default=DEFAULT_TABLEDIR, help="table output (default: %(default)s)")
    args = parser.parse_args()
    args.figdir.mkdir(parents=True, exist_ok=True)
    args.tabledir.mkdir(parents=True, exist_ok=True)

    da, lat_name, lon_name = load_august_v500(args.vwnd)
    res = compute_profile(da, lat_name, lon_name)

    print("\n=== Fig. 12 verification ===")
    print(f"Target: {TARGET_YEAR}/{TARGET_MONTH:02d}; climatology: August {CLIM_START}-{CLIM_END}")
    print(f"Latitude band {ARCTIC_LAT[0]:.0f}-{ARCTIC_LAT[1]:.0f}N (cosine-latitude weighted); "
          f"sector {EAST_ASIA_LON[0]:.0f}-{EAST_ASIA_LON[1]:.0f}E")
    print(f"2025 sector V500 anomaly (weighted):   {res['target_sector']:+.3f} m/s")
    print(f"Unweighted box mean (for comparison):  {res['unweighted']:+.3f} m/s")
    print(f"Fig. 7 direct weighted check:          {res['direct_weighted']:+.3f} m/s")
    print(f"Rank (most negative first): {res['rank']}/{res['n_years']} ({RANK_START}-{RANK_END})")
    if abs(res["target_sector"] - MANUSCRIPT_SECTOR_ANOMALY) > CHECK_TOLERANCE or res["rank"] != MANUSCRIPT_RANK:
        print(f"WARNING: result differs from the manuscript ({MANUSCRIPT_SECTOR_ANOMALY} m/s, "
              f"rank {MANUSCRIPT_RANK}/{res['n_years']}). Check the NCEP2 product, grid and averaging.")
    else:
        print("Matches the manuscript sector anomaly and rank within tolerance.")

    fig = make_figure(res)
    fig_file = args.figdir / FIG_FILE
    fig.savefig(fig_file, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    profile_file = args.tabledir / "fig12_longitude_profile.csv"
    rank_file = args.tabledir / "fig12_sector_historical_rank.csv"
    pd.DataFrame({
        "longitude_E": res["lons"],
        "v500_anomaly_2025_m_s": res["vals"],
        "historical_p05_m_s": res["lower"],
        "historical_p95_m_s": res["upper"],
        "is_East_Asian_Arctic_sector": res["sector_mask"].astype(int),
    }).to_csv(profile_file, index=False)
    res["rank_table"].to_csv(rank_file, index=False)

    print("\nSaved files:")
    for path in (fig_file, profile_file, rank_file):
        print(f"  {path}")
    print("Note: the percentile shading is descriptive, not a formal significance test.")


if __name__ == "__main__":
    main()

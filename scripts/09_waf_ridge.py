#!/usr/bin/env python3
"""
04_fig11_waf_ridge.py
=====================
Fig. 11: daily 500-hPa geopotential-height (Z500) anomalies and Takaya-Nakamura
(2001) wave-activity flux (WAF) during the ridge-formation (10-16 Aug 2025) and
ridge-persistence (17-23 Aug 2025) phases, as one 4 x 2 figure
(panels a-d: 10, 12, 14, 16 Aug; panels e-h: 17, 19, 21, 23 Aug).

Method
------
* Basic state: 1993-2022 August mean Z500, u500 and v500 (NCEP2 monthly means).
* Perturbation: daily Z500 minus the August basic-state Z500; the quasi-
  geostrophic streamfunction is psi' = g Z' / f (grid points equatorward of
  10 degrees are masked).
* WAF (horizontal components at 500 hPa, scaled by p/1000):
      W = p/(2|U|) * ( U (psi_x^2 - psi psi_xx) + V (psi_x psi_y - psi psi_xy),
                       U (psi_x psi_y - psi psi_xy) + V (psi_y^2 - psi psi_yy) )
  where (U, V) is the August basic-state wind; points with |U| < 3 m/s are masked.
* One common quiver scale (18 x the 95th percentile of |W| over all eight days)
  is used for all panels so that arrow lengths are comparable.
* The star marks Gangneung.

Input (NCEP2; see data/download_instructions.md)
------------------------------------------------
    data/raw/ncep2/pressure/hgt.mon.mean.nc, uwnd.mon.mean.nc, vwnd.mon.mean.nc
    data/raw/ncep2_daily/pressure/hgt.2025.nc, uwnd.2025.nc, vwnd.2025.nc

Output
------
    results/figures/fig11_z500_waf_ridge_evolution.png   (300 dpi)
    results/tables/fig11_gangneung_nearest_grid.csv      Z500 anomaly and WAF at the
                                                         grid point nearest to Gangneung

Usage
-----
    python scripts/04_fig11_waf_ridge.py
    python scripts/04_fig11_waf_ridge.py --monthly-dir /path/monthly/pressure --daily-dir /path/daily/pressure

Requires: numpy, pandas, xarray, netCDF4, matplotlib, cartopy
(cartopy downloads the Natural Earth coastline/border data on first use).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: the figure is written to a file
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MONTHLY_DIR = REPO_ROOT / "data" / "raw" / "ncep2" / "pressure"
DEFAULT_DAILY_DIR = REPO_ROOT / "data" / "raw" / "ncep2_daily" / "pressure"
DEFAULT_FIGDIR = REPO_ROOT / "results" / "figures"
DEFAULT_TABLEDIR = REPO_ROOT / "results" / "tables"
FIG_FILE = "fig11_z500_waf_ridge_evolution.png"

CLIM_START, CLIM_END = 1993, 2022   # basic-state baseline
TARGET_MONTH = 8                    # August
DAILY_YEAR = 2025
LEVEL = 500                         # hPa
P0 = 1000.0                         # hPa

FORMATION_DATES = ["2025-08-10", "2025-08-12", "2025-08-14", "2025-08-16"]
PERSISTENCE_DATES = ["2025-08-17", "2025-08-19", "2025-08-21", "2025-08-23"]
SNAPSHOT_DATES = FORMATION_DATES + PERSISTENCE_DATES

OMEGA = 7.292e-5     # Earth rotation rate (s-1)
A_EARTH = 6.371e6    # Earth radius (m)
G = 9.80665          # gravity (m s-2)

LON_MIN, LON_MAX = 90, 180
LAT_MIN, LAT_MAX = 15, 85
GN_LAT, GN_LON = 37.75, 128.90
MIN_LAT_FOR_F = 10   # |lat| below this: f treated as missing
MIN_WIND_SPEED = 3.0  # m/s; basic-state wind speed below this is masked

Z_MIN, Z_MAX = -80, 80   # colour range (m)
QUIVER_STEP = 2          # plot every 2nd grid point
QUIVER_SCALE_FACTOR = 18.0


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_level(path, varname, level):
    """Read one variable at `level`, latitude ascending. Returns (DataArray, lat_name, lon_name)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"NCEP2 file not found: {path}\nSee data/download_instructions.md.")
    with xr.open_dataset(path) as ds:
        da = ds[varname]
        lat_name = "lat" if "lat" in da.coords else "latitude"
        lon_name = "lon" if "lon" in da.coords else "longitude"
        lvl_name = "level" if "level" in da.coords else "lev"
        da = da.sortby(lat_name).sel({lvl_name: level}, method="nearest")
        extra = [d for d in da.dims if d not in ("time", lat_name, lon_name) and da.sizes[d] == 1]
        if extra:
            da = da.squeeze(extra, drop=True)
        return da.load(), lat_name, lon_name


def monthly_climatology(path, varname):
    """Mean over CLIM_START-CLIM_END of the TARGET_MONTH monthly means."""
    da, lat_name, lon_name = load_level(path, varname, LEVEL)
    month = da.sel(time=da["time"].dt.month == TARGET_MONTH)
    clim = month.sel(time=(month["time"].dt.year >= CLIM_START) &
                          (month["time"].dt.year <= CLIM_END)).mean("time")
    return clim, lat_name, lon_name


# -----------------------------------------------------------------------------
# Takaya-Nakamura wave-activity flux
# -----------------------------------------------------------------------------
class WafCalculator:
    """Daily Z500 anomaly and TN01 WAF relative to the August basic state."""

    def __init__(self, z_clim, u_clim, v_clim, lat_vals, lon_vals):
        self.z_clim = z_clim.values
        self.u = u_clim.values
        self.v = v_clim.values
        self.lat_rad = np.deg2rad(lat_vals)
        self.lon_rad = np.deg2rad(lon_vals)

        coslat = np.cos(self.lat_rad)
        self.coslat_safe = np.where(coslat < 0.1, np.nan, coslat)
        f = 2.0 * OMEGA * np.sin(self.lat_rad)
        self.f_safe = np.where(np.abs(lat_vals) < MIN_LAT_FOR_F, np.nan, f)

        umag = np.sqrt(self.u ** 2 + self.v ** 2)
        self.umag_safe = np.where(umag < MIN_WIND_SPEED, np.nan, umag)
        self.p_scale = LEVEL / P0

    def compute(self, z_day):
        """Return (Z anomaly, Wx, Wy) for one daily Z500 field."""
        z_anom = z_day - self.z_clim
        psi = G * z_anom / self.f_safe[:, np.newaxis]
        cos2d = self.coslat_safe[:, np.newaxis]

        psi_x = np.gradient(psi, self.lon_rad, axis=1) / (A_EARTH * cos2d)
        psi_y = np.gradient(psi, self.lat_rad, axis=0) / A_EARTH
        psi_xx = np.gradient(psi_x, self.lon_rad, axis=1) / (A_EARTH * cos2d)
        psi_yy = np.gradient(psi_y, self.lat_rad, axis=0) / A_EARTH
        psi_xy = np.gradient(psi_x, self.lat_rad, axis=0) / A_EARTH

        wx = self.p_scale / (2.0 * self.umag_safe) * (
            self.u * (psi_x ** 2 - psi * psi_xx) + self.v * (psi_x * psi_y - psi * psi_xy))
        wy = self.p_scale / (2.0 * self.umag_safe) * (
            self.u * (psi_x * psi_y - psi * psi_xy) + self.v * (psi_y ** 2 - psi * psi_yy))
        return z_anom, wx, wy


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
def common_quiver_scale(results):
    """One quiver scale for all panels (18 x the 95th percentile of |W|)."""
    all_mag = []
    for _, wx, wy in results.values():
        mag = np.sqrt(wx ** 2 + wy ** 2)
        vals = mag[np.isfinite(mag)]
        if vals.size:
            all_mag.append(vals)
    robust_mag = np.nanpercentile(np.concatenate(all_mag), 95) if all_mag else 1.0
    return max(robust_mag * QUIVER_SCALE_FACTOR, 1e-12)


def make_figure(results, lat_vals, lon_vals):
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(4, 2, figsize=(10, 15), subplot_kw={"projection": proj})
    cmap = plt.get_cmap("RdBu_r")
    norm = mpl.colors.Normalize(vmin=Z_MIN, vmax=Z_MAX)
    letters = list("abcdefgh")
    quiver_scale = common_quiver_scale(results)

    for i, date_str in enumerate(SNAPSHOT_DATES):
        row, col = divmod(i, 2)
        ax = axes[row, col]
        z_anom, wx, wy = results[date_str]

        ax.pcolormesh(lon_vals, lat_vals, z_anom, transform=proj, cmap=cmap, norm=norm, shading="auto")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.55)
        ax.add_feature(cfeature.BORDERS, linewidth=0.35)
        ax.quiver(lon_vals[::QUIVER_STEP], lat_vals[::QUIVER_STEP],
                  wx[::QUIVER_STEP, ::QUIVER_STEP], wy[::QUIVER_STEP, ::QUIVER_STEP],
                  transform=proj, scale=quiver_scale, width=0.0032, pivot="middle")
        ax.plot(GN_LON, GN_LAT, marker="*", markersize=14, markeredgecolor="black",
                markeredgewidth=0.7, transform=proj, zorder=6)

        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        if col != 0:
            gl.left_labels = False
        if row != 3:
            gl.bottom_labels = False
        ax.set_title(f"({letters[i]}) {date_str}", fontsize=14, fontweight="bold", pad=5)

    fig.text(0.018, 0.715, "Ridge formation (10–16 Aug)", rotation=90,
             va="center", ha="center", fontsize=14, fontweight="bold")
    fig.text(0.018, 0.315, "Ridge persistence (17–23 Aug)", rotation=90,
             va="center", ha="center", fontsize=14, fontweight="bold")

    cbar_ax = fig.add_axes([0.19, 0.055, 0.62, 0.018])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("500-hPa geopotential-height anomaly (m)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.095, wspace=0.05, hspace=0.10)
    return fig, quiver_scale


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fig. 11: Z500 anomaly and Takaya-Nakamura WAF.")
    parser.add_argument("--monthly-dir", type=Path, default=DEFAULT_MONTHLY_DIR,
                        help="NCEP2 monthly pressure-level files (default: %(default)s)")
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR,
                        help="NCEP2 daily pressure-level files (default: %(default)s)")
    parser.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR, help="figure output (default: %(default)s)")
    parser.add_argument("--tabledir", type=Path, default=DEFAULT_TABLEDIR, help="table output (default: %(default)s)")
    args = parser.parse_args()
    args.figdir.mkdir(parents=True, exist_ok=True)
    args.tabledir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {CLIM_START}-{CLIM_END} August climatological basic state ...")
    z_clim, lat_name, lon_name = monthly_climatology(args.monthly_dir / "hgt.mon.mean.nc", "hgt")
    u_clim, _, _ = monthly_climatology(args.monthly_dir / "uwnd.mon.mean.nc", "uwnd")
    v_clim, _, _ = monthly_climatology(args.monthly_dir / "vwnd.mon.mean.nc", "vwnd")

    print(f"Loading {DAILY_YEAR} daily fields ...")
    hgt_daily, _, _ = load_level(args.daily_dir / f"hgt.{DAILY_YEAR}.nc", "hgt", LEVEL)
    lat_vals = hgt_daily[lat_name].values
    lon_vals = hgt_daily[lon_name].values
    if z_clim.shape != hgt_daily.isel(time=0).shape:
        raise ValueError("Monthly and daily NCEP2 grids differ; both must be on the same 2.5-degree grid.")

    calc = WafCalculator(z_clim, u_clim, v_clim, lat_vals, lon_vals)
    results = {d: calc.compute(hgt_daily.sel(time=d, method="nearest").values) for d in SNAPSHOT_DATES}

    # Diagnostics at the grid point nearest to Gangneung
    gn_lat_idx = np.abs(lat_vals - GN_LAT).argmin()
    gn_lon_idx = np.abs(lon_vals - GN_LON).argmin()
    rows = []
    print("\n=== Gangneung-nearest-grid diagnostics ===")
    for d in SNAPSHOT_DATES:
        z_anom, wx, wy = results[d]
        row = {"date": d,
               "grid_lat": float(lat_vals[gn_lat_idx]), "grid_lon": float(lon_vals[gn_lon_idx]),
               "Z500_anomaly_m": float(z_anom[gn_lat_idx, gn_lon_idx]),
               "Wx": float(wx[gn_lat_idx, gn_lon_idx]), "Wy": float(wy[gn_lat_idx, gn_lon_idx])}
        rows.append(row)
        print(f"{d}: Z500 anomaly={row['Z500_anomaly_m']:+.1f} m, Wx={row['Wx']:+.3g}, Wy={row['Wy']:+.3g}")
    table_file = args.tabledir / "fig11_gangneung_nearest_grid.csv"
    pd.DataFrame(rows).to_csv(table_file, index=False)

    fig, quiver_scale = make_figure(results, lat_vals, lon_vals)
    fig_file = args.figdir / FIG_FILE
    fig.savefig(fig_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {fig_file}\nSaved: {table_file}")
    print(f"Common quiver scale used for all panels: {quiver_scale:.4g}")


if __name__ == "__main__":
    main()

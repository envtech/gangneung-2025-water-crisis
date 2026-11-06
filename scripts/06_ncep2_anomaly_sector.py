#!/usr/bin/env python3
"""
03_ncep2_anomaly_sector.py
==========================
Fig. 7 (and the sector sensitivity of Table S1): hydroclimatic and Arctic-to-East
Asia circulation anomalies in August 2025, from NCEP/DOE Reanalysis 2 (NCEP2)
monthly data.

Panels (all drawn on the same Arctic-to-East Asia map window, 90-180E, 20-85N)
-----------------------------------------------------------------------------
(a) 2-m air temperature anomaly        (b) precipitation-rate anomaly
(c) precipitable-water anomaly         (d) 500-hPa geopotential-height anomaly
(e) 500-hPa meridional-wind anomaly    (f) 500-hPa wind-speed anomaly with
                                           anomalous wind vectors

Markers: star = Gangneung; dashed box = Gangneung 5 x 5 degree region; yellow
dash-dotted box = East Asian Arctic sector.  Each panel reports the mean anomaly
of the Gangneung region and of the Arctic sector (area-weighted).

Definitions
-----------
* Anomaly = August 2025 minus the 1993-2022 mean August field.
* Wind-speed anomaly = |V(Aug 2025)| - |V(clim)|, where V(clim) is the
  1993-2022 mean August wind vector at 500 hPa (the speed of the mean wind, not the
  mean of daily speeds).
* Precipitation rate is converted from kg m-2 s-1 to mm day-1.
* Ranks (1 = most extreme) compare the August 2025 area-mean anomaly with all
  Augusts 1979-2025 (47 years); T2m and Z500: highest = 1, PR, PWC and V500:
  lowest = 1.
* Sectors (``--sector``):  primary 65-85N, 95-175E (default, used in the
  manuscript); alternative 65-80N, 100-170E (sensitivity test, Table S1).  The
  sector is a diagnostic domain fixed in advance, not a maximum selected from the
  results.
* NCEP2 is a 2.5-degree reanalysis: the figure shows synoptic- to planetary-scale
  circulation, not local terrain effects.  The figure gives circulation context
  and does not by itself establish a causal link between the Arctic anomalies and
  the Gangneung drought.

Input
-----
NCEP2 monthly-mean NetCDF files (NOAA PSL), not included in the repository; see
``data/download_instructions.md``.  Expected layout under ``--data-dir``
(default ``data/raw/ncep2``)::

    pressure/       hgt.mon.mean.nc  uwnd.mon.mean.nc  vwnd.mon.mean.nc
    gaussian_grid/  air.2m.mon.mean.nc  prate.sfc.mon.mean.nc
    surface/        pr_wtr.eatm.mon.mean.nc

Output
------
    results/figures/fig07_circulation_anomalies.png          (600 dpi)
    results/tables/fig07_anomaly_summary_primary.csv
(with ``--sector alternative``: ``..._alt_sector.png`` / ``..._alternative.csv``)

Usage
-----
    python scripts/03_ncep2_anomaly_sector.py
    python scripts/03_ncep2_anomaly_sector.py --sector alternative
    python scripts/03_ncep2_anomaly_sector.py --data-dir /path/to/ncep2
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: figures are saved, not shown
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Rectangle

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "ncep2"
DEFAULT_FIGDIR = REPO_ROOT / "results" / "figures"
DEFAULT_TABLEDIR = REPO_ROOT / "results" / "tables"

# key: (file relative to the data directory, variable name in the file)
FILES = {
    "T2m":  ("gaussian_grid/air.2m.mon.mean.nc",      "air"),
    "PWC":  ("surface/pr_wtr.eatm.mon.mean.nc",       "pr_wtr"),
    "PR":   ("gaussian_grid/prate.sfc.mon.mean.nc",   "prate"),
    "U500": ("pressure/uwnd.mon.mean.nc",             "uwnd"),
    "V500": ("pressure/vwnd.mon.mean.nc",             "vwnd"),
    "Z500": ("pressure/hgt.mon.mean.nc",              "hgt"),
}

TARGET_YEAR = 2025
TARGET_MONTH = 8
CLIM_START, CLIM_END = 1993, 2022   # climatology period
RANK_START = 1979                   # first year used for ranking

GN_LAT, GN_LON = 37.75, 128.90      # Gangneung
BOX_HALF_DEG = 2.5                  # half width of the Gangneung region (5 x 5 deg)

# East Asian Arctic sector: (lat1, lat2, lon1, lon2)
SECTORS = {
    "primary":     (65, 85, 95, 175),    # used in the manuscript
    "alternative": (65, 80, 100, 170),   # sensitivity test (Table S1)
}

# Map window of all panels: [lon1, lon2, lat1, lat2]
MAP_EXTENT = [90, 180, 20, 85]

# Colour-scale limits (symmetric about zero)
VLIM = {
    "T2m": 3.0,       # degC
    "PR": 5.0,        # mm day-1
    "PWC": 15.0,      # mm
    "Z500": 60.0,     # m
    "V500": 8.0,      # m s-1
    "WSPD500": 6.0,   # m s-1
}

# Variable metadata for the summary table: description, unit, direction of the
# "most extreme" value used for ranking
META = {
    "T2m": ("2-m air temperature", "°C", "positive"),
    "PR": ("Precipitation rate", "mm day-1", "negative"),
    "PWC": ("Precipitable water", "mm", "negative"),
    "Z500": ("500-hPa geopotential height", "m", "positive"),
    "V500": ("500-hPa meridional wind", "m s-1", "negative"),
}


# -----------------------------------------------------------------------------
# Reading and anomalies
# -----------------------------------------------------------------------------
def standardize_coords(da):
    """Use lat/lon names, sort latitude ascending, longitude in 0-360 ascending."""
    rename_dict = {}
    if "lat" not in da.coords and "latitude" in da.coords:
        rename_dict["latitude"] = "lat"
    if "lon" not in da.coords and "longitude" in da.coords:
        rename_dict["longitude"] = "lon"
    if rename_dict:
        da = da.rename(rename_dict)

    da = da.sortby("lat")
    if float(da.lon.min()) < 0:
        da = da.assign_coords(lon=(da.lon % 360)).sortby("lon")
    else:
        da = da.sortby("lon")
    return da


def load_var(path, varname, level=None):
    """Open one NetCDF variable; optionally select the nearest pressure level."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/download_instructions.md for the NCEP2 files.")
    ds = xr.open_dataset(path)
    da = standardize_coords(ds[varname])

    if level is not None:
        lname = "level" if "level" in da.coords else "lev"
        da = da.sel({lname: level}, method="nearest")

    # Drop leftover length-1 dimensions
    extra = [d for d in da.dims if d not in ("time", "lat", "lon") and da.sizes[d] == 1]
    if extra:
        da = da.squeeze(extra, drop=True)
    return da


def get_august(da):
    """Return all Augusts, the 1993-2022 August climatology, the August 2025 field
    and the 2025 anomaly."""
    aug = da.sel(time=da.time.dt.month == TARGET_MONTH)
    clim = aug.sel(time=(aug.time.dt.year >= CLIM_START)
                   & (aug.time.dt.year <= CLIM_END)).mean("time")
    target = aug.sel(time=aug.time.dt.year == TARGET_YEAR)
    if target.sizes.get("time", 0) == 0:
        raise ValueError(f"No August {TARGET_YEAR} data.")
    target = target.isel(time=0)
    return aug, clim, target, target - clim


def load_all(data_dir):
    """Load the six fields and compute August anomalies.

    Returns the per-variable dictionary and the 500-hPa wind-speed anomaly.
    """
    data = {}
    for key, (rel_path, varname) in FILES.items():
        level = 500 if key in ("U500", "V500", "Z500") else None
        da = load_var(Path(data_dir) / rel_path, varname, level)
        aug, clim, target, anom = get_august(da)
        if key == "PR":  # kg m-2 s-1 -> mm day-1
            aug, clim, target, anom = (x * 86400.0 for x in (aug, clim, target, anom))
        data[key] = dict(aug=aug, clim=clim, target=target, anom=anom)

    # Wind-speed anomaly: speed of the 2025 wind minus speed of the mean wind
    wspd_clim = np.hypot(data["U500"]["clim"], data["V500"]["clim"])
    wspd_tgt = np.hypot(data["U500"]["target"], data["V500"]["target"])
    return data, wspd_tgt - wspd_clim


# -----------------------------------------------------------------------------
# Area means and ranks
# -----------------------------------------------------------------------------
def weighted_mean_box(da, lat1, lat2, lon1, lon2):
    """cos(latitude)-weighted mean over a lat/lon box."""
    sub = da.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2))
    w = np.cos(np.deg2rad(sub.lat))
    return float(sub.weighted(w).mean(("lat", "lon")).values)


def gangneung_mean(da):
    return weighted_mean_box(da, GN_LAT - BOX_HALF_DEG, GN_LAT + BOX_HALF_DEG,
                             GN_LON - BOX_HALF_DEG, GN_LON + BOX_HALF_DEG)


def arctic_sector_mean(da, sector):
    return weighted_mean_box(da, *sector)


def anomaly_series(aug, clim, sector, region="gangneung"):
    """Area-mean August anomaly for every year RANK_START-TARGET_YEAR (pandas Series)."""
    use = aug.sel(time=(aug.time.dt.year >= RANK_START) & (aug.time.dt.year <= TARGET_YEAR))
    anom = use - clim
    if region == "gangneung":
        sub = anom.sel(lat=slice(GN_LAT - BOX_HALF_DEG, GN_LAT + BOX_HALF_DEG),
                       lon=slice(GN_LON - BOX_HALF_DEG, GN_LON + BOX_HALF_DEG))
    else:
        lat1, lat2, lon1, lon2 = sector
        sub = anom.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2))
    w = np.cos(np.deg2rad(sub.lat))
    s = sub.weighted(w).mean(("lat", "lon"))
    years = pd.to_datetime(s.time.values).year
    return pd.Series(np.asarray(s.values), index=years)


def rank_extreme(series, target_year, direction):
    """Rank of the target year (1 = most extreme) and the number of years."""
    s = series.dropna()
    x = float(s.loc[target_year])
    rank = int((s > x).sum() + 1) if direction == "positive" else int((s < x).sum() + 1)
    return rank, len(s)


def build_summary(data, wspd_anom, sector):
    """Summary table: Gangneung and Arctic-sector anomalies and their ranks."""
    rows = []
    for key, (desc, unit, direction) in META.items():
        anom = data[key]["anom"]
        gn_series = anomaly_series(data[key]["aug"], data[key]["clim"], sector, "gangneung")
        arc_series = anomaly_series(data[key]["aug"], data[key]["clim"], sector, "arctic")
        gn_rank, gn_n = rank_extreme(gn_series, TARGET_YEAR, direction)
        arc_rank, arc_n = rank_extreme(arc_series, TARGET_YEAR, direction)
        rows.append({
            "Variable": key, "Description": desc, "Unit": unit,
            "Gangneung_anomaly": gangneung_mean(anom),
            "Gangneung_rank_most_extreme_1": gn_rank, "Gangneung_N_Augusts": gn_n,
            "Arctic_sector_anomaly": arctic_sector_mean(anom, sector),
            "Arctic_rank_most_extreme_1": arc_rank, "Arctic_N_Augusts": arc_n,
            "Rank_period": f"{RANK_START}-{TARGET_YEAR}",
            "Climatology": f"{CLIM_START}-{CLIM_END}",
        })
    # Wind speed: no rank (not a signed extreme)
    rows.append({
        "Variable": "WSPD500", "Description": "500-hPa wind-speed anomaly", "Unit": "m s-1",
        "Gangneung_anomaly": gangneung_mean(wspd_anom),
        "Gangneung_rank_most_extreme_1": np.nan, "Gangneung_N_Augusts": np.nan,
        "Arctic_sector_anomaly": arctic_sector_mean(wspd_anom, sector),
        "Arctic_rank_most_extreme_1": np.nan, "Arctic_N_Augusts": np.nan,
        "Rank_period": f"{RANK_START}-{TARGET_YEAR}",
        "Climatology": f"{CLIM_START}-{CLIM_END}",
    })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def add_common(ax, sector, row=0, col=0):
    """Map decorations shared by all panels: coastlines, grid, Gangneung star and
    5 x 5 degree box, and the Arctic-sector box."""
    lat1, lat2, lon1, lon2 = sector
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.55, edgecolor="0.25")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="0.40")

    gl = ax.gridlines(draw_labels=True, linewidth=0.25, linestyle="--", alpha=0.40)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = (col == 0)
    gl.bottom_labels = (row == 1)
    gl.xlabel_style = {"size": 7.5}
    gl.ylabel_style = {"size": 7.5}

    # Gangneung and its 5 x 5 degree region
    ax.plot(GN_LON, GN_LAT, marker="*", markersize=8.5, markerfacecolor="black",
            markeredgecolor="white", markeredgewidth=0.5,
            transform=ccrs.PlateCarree(), zorder=12)
    ax.add_patch(Rectangle((GN_LON - BOX_HALF_DEG, GN_LAT - BOX_HALF_DEG),
                           2 * BOX_HALF_DEG, 2 * BOX_HALF_DEG,
                           fill=False, edgecolor="black", linewidth=0.75, linestyle="--",
                           transform=ccrs.PlateCarree(), zorder=11))

    # East Asian Arctic sector
    ax.add_patch(Rectangle((lon1, lat1), lon2 - lon1, lat2 - lat1,
                           fill=False, edgecolor="yellow", linewidth=2, linestyle="-.",
                           transform=ccrs.PlateCarree(), zorder=11))
    ax.text(lon1 + 2, lat2 - 2, "East Asian Arctic sector", fontsize=10, fontweight="bold",
            transform=ccrs.PlateCarree(), ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.2), zorder=13)


def annotate_box(ax, gn_value, arc_value, unit, gn_digits=2, arc_digits=2,
                 arc_rank=None, arc_n=None):
    """Text box (lower right): August 2025 anomaly of the Gangneung region and of
    the Arctic sector (with rank if given)."""
    gn_txt = f"Gangneung: {gn_value:+.{gn_digits}f} {unit}"
    arc_txt = f"Arctic sector: {arc_value:+.{arc_digits}f} {unit}"
    if arc_rank is not None and arc_n is not None:
        arc_txt += f"  (rank {arc_rank}/{arc_n})"
    ax.text(0.98, 0.03, gn_txt + "\n" + arc_txt, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor="0.55", alpha=0.88), zorder=20)


def add_colorbar(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.047, pad=0.055, aspect=28)
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=8, length=2)


def make_figure(data, wspd_anom, sector):
    """Build the six-panel figure; returns the matplotlib Figure."""
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "axes.titleweight": "bold", "savefig.dpi": 600})
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(14.0, 8.4))
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.985, bottom=0.07, top=0.91,
                          wspace=0.12, hspace=0.24)

    # Top row: (a) T2m, (b) precipitation rate, (c) precipitable water
    top = [
        ("T2m", "(a) 2-m air temperature anomaly", "RdBu_r", VLIM["T2m"], "°C"),
        ("PR",  "(b) Precipitation-rate anomaly",   "BrBG",   VLIM["PR"],  r"mm day$^{-1}$"),
        ("PWC", "(c) Precipitable-water anomaly",   "BrBG",   VLIM["PWC"], "mm"),
    ]
    for col, (key, title, cmap, lim, unit) in enumerate(top):
        ax = fig.add_subplot(gs[0, col], projection=proj)
        add_common(ax, sector, row=0, col=col)
        da = data[key]["anom"]
        im = ax.pcolormesh(da.lon, da.lat, da, transform=proj, cmap=cmap,
                           vmin=-lim, vmax=lim, shading="auto", rasterized=True)
        ax.set_title(title, loc="left", pad=5)
        add_colorbar(fig, ax, im, unit)
        annotate_box(ax, gangneung_mean(da), arctic_sector_mean(da, sector), unit)

    # (d) Z500 anomaly with contours
    ax = fig.add_subplot(gs[1, 0], projection=proj)
    add_common(ax, sector, row=1, col=0)
    da = data["Z500"]["anom"]
    im = ax.pcolormesh(da.lon, da.lat, da, transform=proj, cmap="RdBu_r",
                       vmin=-VLIM["Z500"], vmax=VLIM["Z500"], shading="auto", rasterized=True)
    cs = ax.contour(da.lon, da.lat, da, levels=np.arange(-60, 61, 20),
                    colors="0.25", linewidths=0.50, transform=proj)
    ax.clabel(cs, fmt="%d", fontsize=6)
    ax.contour(da.lon, da.lat, da, levels=[0], colors="black", linewidths=1.0, transform=proj)
    ax.set_title("(d) 500-hPa geopotential-height anomaly", loc="left", pad=5)
    add_colorbar(fig, ax, im, "m")
    annotate_box(ax, gangneung_mean(da), arctic_sector_mean(da, sector), "m",
                 gn_digits=1, arc_digits=1)

    # (e) V500 anomaly: explicit Arctic diagnostic with rank
    ax = fig.add_subplot(gs[1, 1], projection=proj)
    add_common(ax, sector, row=1, col=1)
    da = data["V500"]["anom"]
    im = ax.pcolormesh(da.lon, da.lat, da, transform=proj, cmap="RdBu_r",
                       vmin=-VLIM["V500"], vmax=VLIM["V500"], shading="auto", rasterized=True)
    ax.contour(da.lon, da.lat, da, levels=[0], colors="black", linewidths=0.7, transform=proj)
    ax.set_title("(e) 500-hPa meridional-wind anomaly", loc="left", pad=5)
    add_colorbar(fig, ax, im, r"m s$^{-1}$")
    arc_series = anomaly_series(data["V500"]["aug"], data["V500"]["clim"], sector, "arctic")
    arc_rank, arc_n = rank_extreme(arc_series, TARGET_YEAR, "negative")
    annotate_box(ax, gangneung_mean(da), arctic_sector_mean(da, sector), r"m s$^{-1}$",
                 arc_rank=arc_rank, arc_n=arc_n)

    # (f) wind-speed anomaly with anomalous wind vectors
    ax = fig.add_subplot(gs[1, 2], projection=proj)
    add_common(ax, sector, row=1, col=2)
    im = ax.pcolormesh(wspd_anom.lon, wspd_anom.lat, wspd_anom, transform=proj,
                       cmap="RdBu_r", vmin=-VLIM["WSPD500"], vmax=VLIM["WSPD500"],
                       shading="auto", rasterized=True)
    ua, va = data["U500"]["anom"], data["V500"]["anom"]
    step = 2  # plot every second grid point
    q = ax.quiver(ua.lon.values[::step], ua.lat.values[::step],
                  ua.values[::step, ::step], va.values[::step, ::step],
                  transform=proj, color="0.15", scale=95, width=0.003,
                  headwidth=3.6, headlength=4.2, zorder=8)
    ax.quiverkey(q, 0.86, 1.035, 5, r"5 m s$^{-1}$", labelpos="E", coordinates="axes",
                 fontproperties={"size": 7})
    ax.set_title("(f) 500-hPa wind-speed anomaly and vector anomaly", loc="left", pad=5)
    add_colorbar(fig, ax, im, r"Wind-speed anomaly (m s$^{-1}$)")
    annotate_box(ax, gangneung_mean(wspd_anom), arctic_sector_mean(wspd_anom, sector),
                 r"m s$^{-1}$")
    return fig


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fig. 7: NCEP2 August 2025 anomalies.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="NCEP2 monthly files (default: %(default)s)")
    parser.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR,
                        help="figure directory (default: %(default)s)")
    parser.add_argument("--tabledir", type=Path, default=DEFAULT_TABLEDIR,
                        help="table directory (default: %(default)s)")
    parser.add_argument("--sector", choices=list(SECTORS), default="primary",
                        help="East Asian Arctic sector definition (default: %(default)s)")
    args = parser.parse_args()

    sector = SECTORS[args.sector]
    print(f"Arctic sector ({args.sector}): {sector[0]}-{sector[1]}N, {sector[2]}-{sector[3]}E")

    data, wspd_anom = load_all(args.data_dir)
    summary = build_summary(data, wspd_anom, sector)

    suffix = "" if args.sector == "primary" else "_alt_sector"
    args.figdir.mkdir(parents=True, exist_ok=True)
    args.tabledir.mkdir(parents=True, exist_ok=True)
    table_file = args.tabledir / f"fig07_anomaly_summary_{args.sector}.csv"
    fig_file = args.figdir / f"fig07_circulation_anomalies{suffix}.png"

    summary.to_csv(table_file, index=False, encoding="utf-8-sig")
    fig = make_figure(data, wspd_anom, sector)
    fig.savefig(fig_file, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"\nSaved: {fig_file}\nSaved: {table_file}")


if __name__ == "__main__":
    main()

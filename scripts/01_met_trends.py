#!/usr/bin/env python3
"""
01_met_trends.py
================
Monthly hydroclimatic conditions at Gangneung (KMA ASOS station 105), used for
Fig. 4 of the manuscript.

For each variable (monthly precipitation total, monthly mean air temperature,
wind speed, relative humidity, and station pressure) the script produces two
figures:

  * ``comp_monthly_<var>.png`` -- seasonal-cycle comparison (Fig. 4 panels):
        the 1993-2022 monthly climatology (mean, +/-1 and +/-2 standard
        deviations, shown as shading) against the individual years 2023, 2024
        and 2025.
  * ``ts_monthly_<var>.png``   -- full monthly time series, with 1993-2022 in
        gray and 2023, 2024, 2025 highlighted.

Input
-----
A single daily ASOS CSV (KMA open-data portal format, UTF-8 with BOM, Korean
column headers), by default::

    data/raw/gangneung_asos_1993_2025_merged.csv

The Korean column names are mapped to English names internally
(see ``COLUMN_MAP``); the input file itself is not modified.

Output
------
PNG figures written to, by default::

    results/figures/fig04_met_trends/

Usage
-----
Run from anywhere; paths are resolved relative to the repository root::

    python scripts/01_met_trends.py
    python scripts/01_met_trends.py --input path/to/asos.csv --outdir path/to/out

Notes
-----
* KMA leaves the daily precipitation field empty on days without precipitation.
  Empty values therefore contribute zero to the monthly totals (pandas ``sum``
  skips NaN), which is the intended behavior here.
* Climatology statistics use the sample standard deviation (ddof = 1) of the
  30 monthly values (1993-2022) for each calendar month.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: figures are saved, not shown
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "gangneung_asos_1993_2025_merged.csv"
DEFAULT_OUTDIR = REPO_ROOT / "results" / "figures" / "fig04_met_trends"

# Stations to plot, in plotting order (the merged file contains Gangneung only).
SITE_ORDER = ["Gangneung"]
SITE_NAME_MAP = {  # Korean station name -> English
    "강릉": "Gangneung",
    "북강릉": "Bukgangneung",
    "대관령": "Daegwallyeong",
}

# Korean column names in the KMA daily (DD) ASOS file -> English names used here.
COLUMN_MAP = {
    "지점명": "site_name",
    "일시": "date",
    "평균기온(°C)": "tmean",          # daily mean air temperature (deg C)
    "일강수량(mm)": "precip",          # daily precipitation (mm)
    "평균 풍속(m/s)": "wspd",          # daily mean wind speed (m/s)
    "평균 상대습도(%)": "rh",          # daily mean relative humidity (%)
    "평균 현지기압(hPa)": "pressure",  # daily mean station pressure (hPa)
}

# Reference (climatology) period and highlighted years.
BASELINE_START, BASELINE_END = 1993, 2022
HIGHLIGHT_YEARS = {2023: "yellow", 2024: "orange", 2025: "red"}

# Variables to analyze.  agg: how daily values are aggregated to monthly values.
#   key      : column name after renaming
#   agg      : "sum" (precipitation) or "mean" (all other variables)
#   label    : y-axis label
#   unit     : unit string appended to the label
#   stem     : file-name stem of the output figures
#   ylim_ts  : y-limits of the time-series figure (None = automatic)
#   ylim_cmp : y-limits of the seasonal-comparison figure (None = automatic)
VARIABLES = [
    dict(key="precip", agg="sum", label="Monthly acc. precipitation", unit="(mm)",
         stem="acc_preci", ylim_ts=(0, 1200), ylim_cmp=(0, 800)),
    dict(key="tmean", agg="mean", label="Mean air temperature", unit="(°C)",
         stem="mean_t", ylim_ts=None, ylim_cmp=None),
    dict(key="wspd", agg="mean", label="Mean wind speed", unit="(m/s)",
         stem="mean_ws", ylim_ts=None, ylim_cmp=None),
    dict(key="rh", agg="mean", label="Mean RH", unit="(%)",
         stem="mean_rh", ylim_ts=None, ylim_cmp=None),
    dict(key="pressure", agg="mean", label="Mean pressure", unit="(hPa)",
         stem="mean_p", ylim_ts=None, ylim_cmp=None),
]

# Font sizes
TITLE_SIZE, LABEL_SIZE, TICK_SIZE = 20, 18, 16


# -----------------------------------------------------------------------------
# Data handling
# -----------------------------------------------------------------------------
def load_asos(csv_path):
    """Read the merged daily ASOS CSV and return a tidy DataFrame.

    Adds the columns ``site`` (English station name) and ``month`` (monthly
    pandas Period) and keeps only the columns listed in ``COLUMN_MAP``.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")
    df["site"] = df["site_name"].map(SITE_NAME_MAP)
    return df


def monthly_values(df, key, agg):
    """Aggregate daily values to monthly values for every station."""
    out = df.groupby(["site", "month"])[key].agg(agg).reset_index()
    out["month"] = out["month"].dt.to_timestamp()
    out["year"] = out["month"].dt.year
    out["month_num"] = out["month"].dt.month
    return out


def sites_to_plot(available):
    """Stations present in the data, in the order given by SITE_ORDER."""
    return [s for s in SITE_ORDER if s in set(available)]


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_monthly_trend(df, var, png_file):
    """Full monthly time series: 1993-2022 in gray, 2023-2025 highlighted."""
    key = var["key"]
    mon = monthly_values(df, key, var["agg"])
    sites = sites_to_plot(mon["site"].unique())
    if not sites:
        print(f"Warning: no data for the specified sites ({key}, time series).")
        return

    fig, axes = plt.subplots(len(sites), 1, figsize=(12, 5 * len(sites)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, site in zip(axes, sites):
        d = mon[mon["site"] == site]

        base = d[d["year"] <= BASELINE_END]
        ax.plot(base["month"], base[key], color="gray", linewidth=1.5,
                label=f"{BASELINE_START}-{BASELINE_END}")

        for year, color in HIGHLIGHT_YEARS.items():
            dy = d[d["year"] == year]
            if not dy.empty:
                ax.plot(dy["month"], dy[key], color=color, linewidth=2,
                        label=str(year))

        ax.set_title(site, fontsize=TITLE_SIZE)
        ax.set_ylabel(f"{var['label']} {var['unit']}", fontsize=LABEL_SIZE)
        if var["ylim_ts"]:
            ax.set_ylim(var["ylim_ts"])
        ax.grid(True)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        ax.legend(loc="best", fontsize=LABEL_SIZE)

    axes[-1].set_xlabel("Year", fontsize=LABEL_SIZE)
    fig.tight_layout()
    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_comparison(df, var, png_file):
    """Seasonal cycle: 1993-2022 climatology (mean, +/-1 SD, +/-2 SD) versus
    the individual years in HIGHLIGHT_YEARS."""
    key = var["key"]
    mon = monthly_values(df, key, var["agg"])
    sites = sites_to_plot(mon["site"].unique())
    if not sites:
        print(f"Warning: no data for the specified sites ({key}, comparison).")
        return

    fig, axes = plt.subplots(len(sites), 1, figsize=(12, 8 * len(sites)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, site in zip(axes, sites):
        d = mon[mon["site"] == site]

        # Climatology: statistics over the baseline years for each calendar month
        base = d[d["year"] <= BASELINE_END]
        if not base.empty:
            clim = base.groupby("month_num")[key].agg(["mean", "std"]).reset_index()
            x, m, s = clim["month_num"], clim["mean"], clim["std"]
            ax.plot(x, m, color="gray", linestyle="--", linewidth=3,
                    label=f"Mean ({BASELINE_START}-{BASELINE_END})")
            ax.fill_between(x, m - s, m + s, color="gray", alpha=0.2, label="1 STD")
            ax.fill_between(x, m - 2 * s, m + 2 * s, color="gray", alpha=0.1,
                            label="2 STD")

        # Individual years
        for year, color in HIGHLIGHT_YEARS.items():
            dy = d[d["year"] == year].sort_values("month_num")
            if not dy.empty:
                ax.plot(dy["month_num"], dy[key], color=color, linewidth=3,
                        label=str(year))

        ax.set_ylabel(f"{var['label']} {var['unit']}", fontsize=LABEL_SIZE)
        if var["ylim_cmp"]:
            ax.set_ylim(var["ylim_cmp"])
        ax.grid(True)
        ax.legend(title="", loc="best", fontsize=LABEL_SIZE)
        ax.set_xticks(range(1, 13))  # January ... December
        ax.tick_params(axis="both", labelsize=TICK_SIZE)

    axes[-1].set_xlabel("Month", fontsize=LABEL_SIZE)
    fig.tight_layout()
    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_august_summary(df, site="Gangneung", year=2025, month=8):
    """Print the August values of the event year against the 1993-2022 mean."""
    print(f"\n{site}: {year}-{month:02d} versus {BASELINE_START}-{BASELINE_END} "
          f"{month:02d}-month climatology")
    for var in VARIABLES:
        mon = monthly_values(df, var["key"], var["agg"])
        d = mon[(mon["site"] == site) & (mon["month_num"] == month)]
        clim = d[d["year"] <= BASELINE_END][var["key"]]
        val = d[d["year"] == year][var["key"]].iloc[0]
        z = (val - clim.mean()) / clim.std()
        print(f"  {var['label']:<28s} {val:8.1f}  clim {clim.mean():8.1f}  "
              f"anomaly {val - clim.mean():+7.1f}  ({z:+.2f} SD)  {var['unit']}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="merged daily ASOS CSV (default: %(default)s)")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help="output directory (default: %(default)s)")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_asos(args.input)
    print(f"Loaded {len(df)} daily records "
          f"({df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}).")

    for var in VARIABLES:
        print(f"Plotting {var['label']} ...")
        plot_monthly_trend(df, var, args.outdir / f"ts_monthly_{var['stem']}.png")
        plot_yearly_comparison(df, var, args.outdir / f"comp_monthly_{var['stem']}.png")

    print_august_summary(df)
    print(f"\nFigures written to {args.outdir}")


if __name__ == "__main__":
    main()

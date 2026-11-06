#!/usr/bin/env python3
"""
02_trends_fig6.py
=================
Long-term hydroclimatic trends at Gangneung: Fig. 6 and Table 3 of the manuscript.

Panels
------
(a) annual mean temperature      (b) summer (JJA) mean temperature
(c) annual total precipitation   (d) summer (JJA) total precipitation

Statistics
----------
* Trend test: two-sided Mann-Kendall test with tie correction and continuity
  correction (own implementation below); Kendall's tau = S / (n(n-1)/2).
* Trend magnitude: Theil-Sen slope with 95 % confidence interval
  (``scipy.stats.theilslopes``).
* Trend period: 1993-2024.  2025 is plotted for event context only and is NOT
  included in the trend test.
* Solid trend line: p < 0.05; dashed trend line: p >= 0.05.

Input
-----
Daily table written by ``02_compute_spei.py`` (run that script first)::

    data/processed/gangneung_daily_et0.csv

Required columns: ``date``, ``tmean`` (deg C), ``precip`` (mm).  Files with the
Korean ASOS headers ``일시``, ``평균기온(°C)``, ``일강수량(mm)`` are also accepted.
Empty daily precipitation has already been set to 0 mm and short temperature gaps
(<= 5 days) have been interpolated in that table.

Output
------
    results/figures/fig06_trends.png   (600 dpi)
    results/tables/table3_trend_statistics.csv

Usage
-----
    python scripts/02_trends_fig6.py
    python scripts/02_trends_fig6.py --input daily.csv --figdir figs/ --tabledir tables/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: figures are saved, not shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm, theilslopes

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "gangneung_daily_et0.csv"
DEFAULT_FIGDIR = REPO_ROOT / "results" / "figures"
DEFAULT_TABLEDIR = REPO_ROOT / "results" / "tables"
FIG_STEM = "fig06_trends"
TABLE_NAME = "table3_trend_statistics.csv"

TREND_START = 1993
TREND_END = 2024
EVENT_YEAR = 2025
ALPHA = 0.05  # significance level (and 1 - confidence level of the Sen-slope interval)

FIGSIZE = (13.6, 9.2)  # inches
DPI = 600

# Restrained scientific palette
TEMP_COLOR = "#C44E3B"
TEMP_LIGHT = "#E9A08F"
PRECIP_COLOR = "#3C78A8"
PRECIP_LIGHT = "#A9C7DE"
TREND_TEMP = "#8F1D14"
TREND_PRECIP = "#1F4E79"
EVENT_COLOR = "#B2182B"
EVENT_SHADE = "#F6E7C8"
GRID_COLOR = "#D8D8D8"
TEXT_COLOR = "#252525"


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
def mann_kendall_test(values):
    """Two-sided Mann-Kendall test with tie correction and continuity correction.

    Returns S, Var(S), the standardized statistic z, the p-value and Kendall's tau.
    """
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 3:
        return {"S": np.nan, "var_S": np.nan, "z": np.nan, "p": np.nan, "tau": np.nan}

    s = 0
    for i in range(n - 1):
        s += np.sign(y[i + 1:] - y[i]).sum()

    _, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    return {"S": float(s), "var_S": float(var_s), "z": float(z), "p": float(p), "tau": float(tau)}


def sen_fit(years, values):
    """Sen slope, intercept and confidence interval (scipy's Theil-Sen estimator)."""
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    slope, intercept, lo_slope, hi_slope = theilslopes(y, x, alpha=1 - ALPHA)
    return {"slope": float(slope), "intercept": float(intercept),
            "low_slope": float(lo_slope), "high_slope": float(hi_slope)}


def calc_stats(series):
    """Trend statistics of an annual series (index = year) over TREND_START-TREND_END."""
    s = series.loc[(series.index >= TREND_START) & (series.index <= TREND_END)].dropna()
    return {**mann_kendall_test(s.values), **sen_fit(s.index.values, s.values), "n": len(s)}


# -----------------------------------------------------------------------------
# Data handling
# -----------------------------------------------------------------------------
def choose_column(df, candidates):
    """Return the first column name in `candidates` that exists in `df`."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns were found: {candidates}")


def annual_series(csv_path):
    """Annual and JJA series of temperature (mean) and precipitation (sum).

    Returns four pandas Series indexed by year, covering TREND_START-EVENT_YEAR.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)

    date_col = choose_column(df, ["date", "Date", "DATE", "일시"])
    temp_col = choose_column(df, ["tmean", "평균기온(°C)"])
    precip_col = choose_column(df, ["precip", "일강수량(mm)"])

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
    df[precip_col] = pd.to_numeric(df[precip_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month

    # Keep the years needed for the trend plus the event year.
    df = df[(df["year"] >= TREND_START) & (df["year"] <= EVENT_YEAR)].copy()

    annual_temp = df.groupby("year")[temp_col].mean()
    annual_precip = df.groupby("year")[precip_col].sum(min_count=1)

    jja = df[df["month"].isin([6, 7, 8])]
    jja_temp = jja.groupby("year")[temp_col].mean()
    jja_precip = jja.groupby("year")[precip_col].sum(min_count=1)
    return annual_temp, jja_temp, annual_precip, jja_precip


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.4f}"


def panel_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.58)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))


def add_event_band(ax):
    """Pale vertical band marking the event year."""
    ax.axvspan(EVENT_YEAR - 0.45, EVENT_YEAR + 0.45,
               facecolor=EVENT_SHADE, edgecolor="none", alpha=0.75, zorder=0)


def add_stats_box(ax, slope, p, unit, significant):
    """Text box with Sen's slope, p-value and a one-line interpretation."""
    if significant and slope > 0:
        sig_txt = "Significant warming"
    elif significant and slope < 0:
        sig_txt = "Significant decrease"
    else:
        sig_txt = "No significant trend"
    ax.text(0.03, 0.95, f"Sen's slope = {slope:+.3f} {unit}\n{fmt_p(p)}\n{sig_txt}",
            transform=ax.transAxes, va="top", ha="left", fontsize=12, color=TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                      edgecolor="#BFBFBF", linewidth=0.8, alpha=0.94),
            zorder=10)


def add_event_marker(ax, year_value):
    """Star and label for the event year."""
    ax.scatter(EVENT_YEAR, year_value, s=125, marker="*", facecolor=EVENT_COLOR,
               edgecolor="white", linewidth=0.9, zorder=7)
    ax.annotate("2025", xy=(EVENT_YEAR, year_value), xytext=(-4, 14),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=12, fontweight="bold", color=EVENT_COLOR)


def add_sen_line(ax, stats, color, linewidth, label=None):
    """Sen trend line for TREND_START-TREND_END (solid if significant, else dashed)."""
    xtrend = np.arange(TREND_START, TREND_END + 1)
    ytrend = stats["intercept"] + stats["slope"] * xtrend
    ls = "-" if stats["p"] < ALPHA else "--"
    ax.plot(xtrend, ytrend, color=color, linewidth=linewidth, linestyle=ls,
            zorder=4, label=label)


def plot_temperature(ax, series, title, panel_label, stats):
    """Temperature panel: annual values (line + open circles) and Sen trend."""
    yrs = series.index.to_numpy(dtype=int)
    vals = series.to_numpy(dtype=float)
    trend_mask = (yrs >= TREND_START) & (yrs <= TREND_END)
    event_mask = yrs == EVENT_YEAR

    add_event_band(ax)
    ax.plot(yrs[trend_mask], vals[trend_mask], color=TEMP_LIGHT,
            linewidth=1.1, alpha=0.85, zorder=2)
    ax.scatter(yrs[trend_mask], vals[trend_mask], s=28, facecolor="white",
               edgecolor=TEMP_COLOR, linewidth=1.2, zorder=3)
    add_sen_line(ax, stats, TREND_TEMP, 2.6, label="Sen's slope")
    if event_mask.any():
        add_event_marker(ax, vals[event_mask][0])

    panel_style(ax)
    ax.set_title(f"{panel_label} {title}", loc="left", fontsize=14, fontweight="bold", pad=8)
    add_stats_box(ax, stats["slope"], stats["p"], "°C yr$^{-1}$", stats["p"] < ALPHA)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_xlim(TREND_START - 0.7, EVENT_YEAR + 0.7)


def plot_precipitation(ax, series, title, panel_label, stats):
    """Precipitation panel: annual totals (bars) and Sen trend."""
    yrs = series.index.to_numpy(dtype=int)
    vals = series.to_numpy(dtype=float)
    trend_mask = (yrs >= TREND_START) & (yrs <= TREND_END)
    event_mask = yrs == EVENT_YEAR

    add_event_band(ax)
    ax.bar(yrs[trend_mask], vals[trend_mask], width=0.72, color=PRECIP_LIGHT,
           edgecolor=PRECIP_COLOR, linewidth=0.65, alpha=0.92, zorder=2)
    add_sen_line(ax, stats, TREND_PRECIP, 2.4)
    if event_mask.any():
        add_event_marker(ax, vals[event_mask][0])

    panel_style(ax)
    ax.set_title(f"{panel_label} {title}", loc="left", fontsize=14, fontweight="bold", pad=8)
    add_stats_box(ax, stats["slope"], stats["p"], "mm yr$^{-1}$", stats["p"] < ALPHA)
    ax.set_ylabel("Precipitation (mm)", fontsize=12)
    ax.set_xlim(TREND_START - 0.7, EVENT_YEAR + 0.7)
    ax.set_ylim(bottom=0)


def make_figure(series_by_name, stats):
    """Build the four-panel figure; returns the matplotlib Figure."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelcolor": TEXT_COLOR, "axes.titlecolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR,
    })
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.095, top=0.90,
                        wspace=0.20, hspace=0.28)

    panels = [
        (axes[0, 0], plot_temperature, "Annual mean temperature", "(a)"),
        (axes[0, 1], plot_temperature, "Summer (JJA) mean temperature", "(b)"),
        (axes[1, 0], plot_precipitation, "Annual total precipitation", "(c)"),
        (axes[1, 1], plot_precipitation, "Summer (JJA) total precipitation", "(d)"),
    ]
    for ax, func, name, label in panels:
        func(ax, series_by_name[name], name, label, stats[name])

    for ax in axes[1, :]:
        ax.set_xlabel("Year", fontsize=11)
    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
    return fig


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def save_figure(fig, path):
    """Save the figure as a PNG file (DPI resolution)."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")


def stats_table(stats):
    """One row per variable: Sen's slope with CI, Kendall's tau, z, p-value (Table 3)."""
    rows = []
    for name, st in stats.items():
        rows.append({
            "Variable": name,
            "Period": f"{TREND_START}-{TREND_END}",
            "N": st["n"],
            "Sen_slope": st["slope"],
            "Sen_slope_low95": st["low_slope"],
            "Sen_slope_high95": st["high_slope"],
            "MK_tau": st["tau"],
            "MK_z": st["z"],
            "MK_p": st["p"],
            "Significant_p_lt_0.05": st["p"] < ALPHA,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fig. 6 and Table 3: long-term trends at Gangneung.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="daily CSV (default: %(default)s)")
    parser.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR,
                        help="figure directory (default: %(default)s)")
    parser.add_argument("--tabledir", type=Path, default=DEFAULT_TABLEDIR,
                        help="table directory (default: %(default)s)")
    args = parser.parse_args()

    annual_temp, jja_temp, annual_precip, jja_precip = annual_series(args.input)
    series_by_name = {
        "Annual mean temperature": annual_temp,
        "Summer (JJA) mean temperature": jja_temp,
        "Annual total precipitation": annual_precip,
        "Summer (JJA) total precipitation": jja_precip,
    }
    stats = {name: calc_stats(s) for name, s in series_by_name.items()}

    print(f"=== Trend statistics, {TREND_START}-{TREND_END} ===")
    for name, st in stats.items():
        print(f"{name:36s}  slope={st['slope']:+.6f}, p={st['p']:.6f}")

    args.figdir.mkdir(parents=True, exist_ok=True)
    args.tabledir.mkdir(parents=True, exist_ok=True)
    fig = make_figure(series_by_name, stats)
    fig_file = args.figdir / f"{FIG_STEM}.png"
    save_figure(fig, fig_file)
    plt.close(fig)

    table_file = args.tabledir / TABLE_NAME
    stats_table(stats).to_csv(table_file, index=False, encoding="utf-8-sig")

    print("\nSaved:")
    for path in (fig_file, table_file):
        print(f"  {path}")


if __name__ == "__main__":
    main()

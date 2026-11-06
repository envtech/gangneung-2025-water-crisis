#!/usr/bin/env python3
"""
02_spei_timeseries.py
=====================
SPEI time series at Gangneung and the August 2025 ranks (Fig. 5 and the ranking
columns of Table 2 of the manuscript).

The SPEI values are read from a precomputed monthly file; this script does NOT
recalculate SPEI.  It only (1) ranks the August 2025 value of each SPEI time
scale against the full record, and (2) draws the four-panel time-series figure.

Input
-----
Monthly SPEI table (CSV, UTF-8), by default::

    data/processed/gangneung_monthly_spei.csv

Written by ``02_compute_spei.py`` (run that script first).  Expected columns:
``date`` (month start; the first column is used whatever its name, so files with
the Korean header ``일시`` also work), ``SPEI_1``, ``SPEI_3``, ``SPEI_6``,
``SPEI_12``.  Other columns (P, PET, D = P - PET) are not used here.

Output
------
    results/figures/fig05_spei_timeseries.png
    results/tables/table2_spei_august2025_ranks.csv

Usage
-----
    python scripts/02_spei_timeseries.py
    python scripts/02_spei_timeseries.py --input my_spei.csv --fig out.png --table ranks.csv

Notes
-----
* Ranking uses the competition rank "number of strictly lower values + 1"
  (rank 1 = driest month); ties receive the same rank.
* The analysis period is cut to START_DATE-END_DATE *before* ranking, so the
  ranks refer to exactly the months shown.  Limiting the x-axis alone would not
  change the ranks.
* A SPEI time scale of k months is undefined for the first k-1 months, so the
  expected number of valid months is (total months) - (k - 1); the script warns
  if the file deviates from this.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: figures are saved, not shown
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "gangneung_monthly_spei.csv"
DEFAULT_FIG = REPO_ROOT / "results" / "figures" / "fig05_spei_timeseries.png"
DEFAULT_TABLE = REPO_ROOT / "results" / "tables" / "table2_spei_august2025_ranks.csv"

START_DATE = pd.Timestamp("1993-01-01")
END_DATE = pd.Timestamp("2025-12-01")        # last month used (month-start stamps)
TARGET_DATE = pd.Timestamp("2025-08-01")     # month to be ranked and annotated
EVENT_BAND_START = pd.Timestamp("2025-04-01")  # orange background band (2025 event)
EVENT_BAND_END = pd.Timestamp("2025-12-31")

SCALES = ["SPEI_1", "SPEI_3", "SPEI_6", "SPEI_12"]
LABELS = {s: s.replace("_", "-") for s in SCALES}
# Drought-severity thresholds drawn as red lines: value, line style
THRESHOLDS = [(-1.0, ":"), (-1.5, "--"), (-2.0, "-")]  # moderate, severe, extreme


# -----------------------------------------------------------------------------
# Data handling
# -----------------------------------------------------------------------------
def load_spei(csv_path):
    """Read the monthly SPEI table, validate it and cut it to the analysis period.

    Returns the DataFrame (sorted by month) and the name of its date column.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Date column: 'date' (or the Korean '일시' of older files); otherwise column 0.
    date_col = next((c for c in ("date", "일시") if c in df.columns), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors="raise").dt.to_period("M").dt.to_timestamp()

    for col in SCALES:
        if col not in df.columns:
            raise KeyError(f"Missing SPEI column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[date_col].duplicated().any():
        dup = df.loc[df[date_col].duplicated(), date_col].dt.strftime("%Y-%m").tolist()
        raise ValueError(f"Duplicate months in SPEI CSV: {dup[:12]}")

    # Apply the date limit BEFORE ranking and plotting.
    df = (df.loc[df[date_col].between(START_DATE, END_DATE)]
            .sort_values(date_col).reset_index(drop=True))
    expected = pd.date_range(START_DATE, END_DATE, freq="MS")
    missing = expected.difference(pd.DatetimeIndex(df[date_col]))
    if len(missing):
        raise ValueError(f"Missing months: {missing.strftime('%Y-%m').tolist()}")
    if TARGET_DATE not in set(df[date_col]):
        raise ValueError("The selected date range does not include August 2025.")
    return df, date_col


def rank_target_month(df, date_col):
    """Rank the target month (August 2025) for every SPEI time scale.

    Returns a DataFrame with the value, rank (1 = driest), number of valid
    months and percentile (rank / N * 100).
    """
    rows = []
    for scale in SCALES:
        valid = df[scale].dropna()
        value = df.loc[df[date_col].eq(TARGET_DATE), scale].iloc[0]
        if pd.isna(value):
            raise ValueError(f"{scale} for August 2025 is missing.")
        rank = int((valid < value).sum() + 1)  # competition rank
        n = len(valid)
        rows.append({
            "Index": LABELS[scale],
            "August 2025 value": value,
            "Rank": rank,
            "N valid months": n,
            "Rank label": f"{rank}/{n}",
            "Percentile (%)": rank / n * 100,
        })
        print(f"{LABELS[scale]}: value={value:.4f}; rank={rank}/{n}; "
              f"percentile={rank / n * 100:.2f}%")

    # Sanity check: a k-month SPEI is undefined for the first k-1 months.
    n_months = len(df)
    for row, scale in zip(rows, SCALES):
        k = int(scale.split("_")[1])
        expected_n = n_months - (k - 1)
        if row["N valid months"] != expected_n:
            print(f"WARNING: {row['Index']} has N={row['N valid months']} valid months, "
                  f"expected {expected_n}. Check the data and preprocessing.")
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_spei(df, date_col, rank_df, fig_file):
    """Four-panel SPEI time series (SPEI-1, -3, -6, -12) with August 2025 marked."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    for ax, scale, (_, info) in zip(axes, SCALES, rank_df.iterrows()):
        x = df[date_col]
        y = df[scale].to_numpy(dtype=float)

        # Background band for the 2025 event period
        ax.axvspan(EVENT_BAND_START, EVENT_BAND_END, color="orange", alpha=0.20, zorder=0)

        # Positive (wet, blue) and negative (dry, red) SPEI areas
        ax.fill_between(x, y, 0, where=(y >= 0) & np.isfinite(y), color="#7A9CC6",
                        alpha=0.85, interpolate=True, zorder=2)
        ax.fill_between(x, y, 0, where=(y < 0) & np.isfinite(y), color="#C67A7A",
                        alpha=0.85, interpolate=True, zorder=2)
        ax.plot(x, y, color="black", linewidth=0.4, zorder=3)

        # Drought-severity thresholds and zero line
        for threshold, linestyle in THRESHOLDS:
            ax.axhline(threshold, color="red", linestyle=linestyle, linewidth=1, zorder=1)
        ax.axhline(0, color="black", linewidth=0.6, zorder=1)

        # August 2025 marker and rank annotation
        val = info["August 2025 value"]
        ax.scatter([TARGET_DATE], [val], color="black", s=40, zorder=5)
        ax.annotate(f"Aug 2025\n(rank {info['Rank label']})", xy=(TARGET_DATE, val),
                    xytext=(-12, -29), textcoords="offset points", fontsize=11, ha="right",
                    arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 0.7})

        ax.set_ylim(-3, 3)
        ax.set_xlim(START_DATE, pd.Timestamp("2026-01-01"))
        ax.set_ylabel(LABELS[scale], fontsize=14, fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Year", fontsize=14)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].tick_params(axis="x", labelsize=12, length=6, width=1)
    fig.tight_layout()
    fig.savefig(fig_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fig. 5: SPEI time series and August 2025 ranks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="monthly SPEI CSV (default: %(default)s)")
    parser.add_argument("--fig", type=Path, default=DEFAULT_FIG,
                        help="output figure (default: %(default)s)")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE,
                        help="output rank table (default: %(default)s)")
    args = parser.parse_args()

    args.fig.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)

    df, date_col = load_spei(args.input)
    print(f"Ranking and display period: {START_DATE:%Y-%m} to {END_DATE:%Y-%m} "
          f"({len(df)} months)")

    rank_df = rank_target_month(df, date_col)
    rank_df.to_csv(args.table, index=False, encoding="utf-8-sig")

    plot_spei(df, date_col, rank_df, args.fig)
    print(f"Saved: {args.fig}\nSaved: {args.table}")


if __name__ == "__main__":
    main()

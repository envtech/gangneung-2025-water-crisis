#!/usr/bin/env python3
"""
02_table2_spei_summary.py
=========================
Table 2 of the manuscript: SPEI values for August 2025, their historical ranks,
and a sample-period audit.

For each SPEI time scale (1, 3, 6, 12 months) the script reports the August 2025
value, its rank among all months of the ranking period (rank 1 = driest), the
percentile, and the driest month of the record.  A second file documents the
sample (number of valid and missing months, first and last valid month, ties) so
that the denominators in the rank labels can be checked.

The script does not recompute SPEI.  Run ``02_compute_spei.py`` first.

Input
-----
    data/processed/gangneung_monthly_spei.csv
    Required columns: ``date`` (month start; the Korean header ``일시`` and, failing
    both, the first column are also accepted) and SPEI_1, SPEI_3, SPEI_6, SPEI_12.

Output
------
    results/tables/table2_spei_summary.csv      Table 2
    results/tables/table2_spei_data_audit.csv   sample-period audit

Usage
-----
    python scripts/02_table2_spei_summary.py
    python scripts/02_table2_spei_summary.py --input spei.csv --outdir out/

Notes
-----
* The ranking period is RANK_START-RANK_END (default: January 1993 - December
  2025, i.e. the full SPEI record).  Every month in this interval must be present
  (STRICT_MONTHLY_COMPLETENESS); a k-month SPEI is undefined for the first k - 1
  months, so the number of valid months is (months - k + 1).
* Competition rank: number of strictly lower values + 1; ties share the minimum
  rank.  Percentile = 100 * rank / number of valid months.
* The SPEI values depend on the period used to fit the climatological
  distributions in ``02_compute_spei.py`` (full record); this script does not
  change them.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "gangneung_monthly_spei.csv"
DEFAULT_OUTDIR = REPO_ROOT / "results" / "tables"

RANK_START = "1993-01"    # first month of the ranking period (inclusive)
RANK_END = "2025-12"      # last month of the ranking period (inclusive)
TARGET_MONTH = "2025-08"  # month to be ranked
SCALES = {1: "SPEI_1", 3: "SPEI_3", 6: "SPEI_6", 12: "SPEI_12"}
STRICT_MONTHLY_COMPLETENESS = True  # stop if a month is missing in the ranking period


# -----------------------------------------------------------------------------
# Data handling
# -----------------------------------------------------------------------------
def load_spei(csv_path):
    """Read the monthly SPEI table; add a monthly ``month`` Period column."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    date_col = next((c for c in ("date", "일시") if c in df.columns), df.columns[0])
    missing_cols = [c for c in SCALES.values() if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols}. Available: {df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise ValueError(f"Unparseable dates: {df[date_col].isna().sum()} rows")
    df["month"] = df[date_col].dt.to_period("M")
    if df["month"].duplicated().any():
        duplicated = df.loc[df["month"].duplicated(False), "month"].astype(str).tolist()
        raise ValueError(f"Duplicate months in SPEI data: {duplicated[:12]}")
    return df.sort_values("month").reset_index(drop=True)


def select_ranking_period(df):
    """Cut the table to RANK_START-RANK_END and check completeness.

    Returns the subset and the full list of expected calendar months.
    """
    start, end, target = (pd.Period(s, freq="M") for s in (RANK_START, RANK_END, TARGET_MONTH))
    if not (start <= target <= end):
        raise ValueError("TARGET_MONTH must fall inside RANK_START / RANK_END")

    period = df.loc[df["month"].between(start, end)].copy()
    expected = pd.period_range(start, end, freq="M")
    missing_months = expected.difference(pd.PeriodIndex(period["month"]))
    if len(missing_months):
        message = (f"Missing months in ranking interval ({len(missing_months)}): "
                   f"{list(map(str, missing_months[:20]))}")
        if STRICT_MONTHLY_COMPLETENESS:
            raise ValueError(message)
        print("WARNING:", message)
    if period.empty or not (period["month"] == target).any():
        raise ValueError(f"Target month {target} absent from ranking-period CSV")
    return period, expected


# -----------------------------------------------------------------------------
# Table 2
# -----------------------------------------------------------------------------
def summarize_scales(period, expected, source_first, source_last):
    """Return the Table 2 summary and the audit table (one row per SPEI scale)."""
    start, end, target = (pd.Period(s, freq="M") for s in (RANK_START, RANK_END, TARGET_MONTH))
    records, audits = [], []

    for scale, col in SCALES.items():
        vals = pd.to_numeric(period[col], errors="coerce")
        valid = vals.dropna()
        if valid.empty:
            raise ValueError(f"No valid values for {col} in ranking interval")

        idx = period.index[period["month"] == target][0]
        target_value = float(pd.to_numeric(period.loc[idx, col], errors="coerce"))
        if not np.isfinite(target_value):
            raise ValueError(f"Target SPEI not finite: {col} / {target}")

        # Rank 1 = driest. Competition rank: ties share the minimum rank.
        rank = int((valid < target_value).sum() + 1)
        ties = int((valid == target_value).sum())
        percentile = 100 * rank / len(valid)

        min_idx = valid.idxmin()
        driest = str(period.loc[min_idx, "month"])
        driest_value = float(valid.loc[min_idx])
        first_valid = str(period.loc[valid.index[0], "month"])
        last_valid = str(period.loc[valid.index[-1], "month"])

        records.append({
            "Index": f"SPEI-{scale}",
            "2025-08 value": round(target_value, 2),
            "Rank (driest=1)": f"{rank}/{len(valid)}",
            "Percentile": f"{percentile:.1f}%",
            "Historical driest month": driest,
            "Historical driest value": round(driest_value, 2),
        })
        audits.append({
            "scale": f"SPEI-{scale}",
            "original_csv_start": source_first, "original_csv_end": source_last,
            "ranking_start": str(start), "ranking_end": str(end),
            "available_calendar_months": len(expected),
            "observed_months": len(period), "valid_spei_count": len(valid),
            "missing_spei_count": int(vals.isna().sum()),
            "first_valid_spei_month": first_valid, "last_valid_spei_month": last_valid,
            "target_value_unrounded": target_value, "target_rank": rank,
            "target_ties": ties, "driest_month": driest,
            "driest_value_unrounded": driest_value,
        })
    return pd.DataFrame(records), pd.DataFrame(audits)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Table 2: August 2025 SPEI values and ranks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="monthly SPEI CSV (default: %(default)s)")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help="output directory (default: %(default)s)")
    args = parser.parse_args()

    df = load_spei(args.input)
    source_first, source_last = str(df["month"].iloc[0]), str(df["month"].iloc[-1])
    print(f"INPUT SOURCE: {args.input}")
    print(f"Source period: {source_first} to {source_last}; rows={len(df)}")

    period, expected = select_ranking_period(df)
    summary, audit = summarize_scales(period, expected, source_first, source_last)

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_file = args.outdir / "table2_spei_summary.csv"
    audit_file = args.outdir / "table2_spei_data_audit.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_file, index=False, encoding="utf-8-sig")

    print(f"Ranking period: {RANK_START}-{RANK_END} ({len(expected)} calendar months)")
    print("\nTable 2 summary:\n" + summary.to_string(index=False))
    print("\nAudit:\n" + audit[["scale", "valid_spei_count", "missing_spei_count",
                                "first_valid_spei_month", "last_valid_spei_month",
                                "target_ties"]].to_string(index=False))
    print(f"\nSaved: {summary_file}\nSaved: {audit_file}")


if __name__ == "__main__":
    main()

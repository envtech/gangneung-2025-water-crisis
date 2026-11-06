#!/usr/bin/env python3
"""
02_compute_spei.py
==================
Compute FAO-56 Penman-Monteith reference evapotranspiration (ET0) from daily
ASOS observations at Gangneung and, from it, the monthly climatic water balance
and the Standardized Precipitation Evapotranspiration Index (SPEI) at 1-, 3-, 6-
and 12-month time scales.

This script merges the former ``compute_pet_fao56.py`` and
``compute_spei_2025end.py``.  Run it BEFORE ``02_spei_timeseries.py``, which
plots the monthly SPEI table written here.

Input
-----
Merged daily KMA ASOS file (UTF-8 with BOM, Korean column headers), by default::

    data/raw/gangneung_asos_1993_2025_merged.csv

Output
------
    data/processed/gangneung_daily_et0.csv      daily inputs and ET0 (mm/day)
    data/processed/gangneung_monthly_spei.csv   monthly P, PET, D = P - PET, SPEI_1/3/6/12

Usage
-----
    python scripts/02_compute_spei.py
    python scripts/02_compute_spei.py --input asos.csv --daily-output et0.csv --monthly-output spei.csv

Method summary
--------------
1. Missing values.  Empty daily precipitation means no precipitation and is set
   to 0 mm.  The other variables are linearly interpolated over gaps of at most
   five consecutive days.
2. ET0.  FAO-56 Penman-Monteith equation (Allen et al., 1998), daily time step,
   soil heat flux G = 0.  Wind measured at 10 m is converted to 2 m; actual vapour
   pressure is derived from mean relative humidity and the mean saturation vapour
   pressure of Tmax and Tmin; net longwave radiation uses the measured daily
   solar radiation with the ratio Rs/Rso limited to 0.3-1.0.
3. Water balance.  Monthly P and ET0 totals, D = P - PET.
4. SPEI.  D is accumulated over k months (k = 1, 3, 6, 12).  For each calendar
   month separately, a generalized-logistic distribution (the three-parameter
   log-logistic family used for SPEI; L-moment fit via ``lmoments3``) is fitted
   to the accumulated series and the cumulative probabilities are transformed to
   standard-normal quantiles (probabilities clipped to [1e-6, 1 - 1e-6]).

Important
---------
* The fitting sample is the whole analysis period (START_DATE-END_DATE).
  Changing this period changes the fitted distributions and therefore the SPEI
  values and historical ranks.
* A k-month SPEI is undefined for the first k - 1 months of the record.
* The ASOS file contains one record for 1 Jan 2026; it is excluded from the
  monthly series (analysis period ends in December 2025).

Reference: Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998. Crop
evapotranspiration - Guidelines for computing crop water requirements. FAO
Irrigation and Drainage Paper 56.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lmoments3 import distr
from scipy import stats

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "gangneung_asos_1993_2025_merged.csv"
DEFAULT_DAILY = REPO_ROOT / "data" / "processed" / "gangneung_daily_et0.csv"
DEFAULT_MONTHLY = REPO_ROOT / "data" / "processed" / "gangneung_monthly_spei.csv"

# Station metadata: Gangneung ASOS (station no. 105)
LAT_DEG = 37.75147   # latitude (deg N)
ELEV_M = 27.12       # elevation (m a.s.l.)
WIND_HEIGHT_M = 10.0  # anemometer height (m)

# Analysis period (monthly series: Jan 1993 - Dec 2025)
START_DATE = pd.Timestamp("1993-01-01")
END_DATE_EXCLUSIVE = pd.Timestamp("2026-01-01")
SPEI_SCALES = (1, 3, 6, 12)
MAX_INTERP_GAP_DAYS = 5   # maximum gap (days) filled by linear interpolation
MIN_MONTHS_FOR_FIT = 10   # minimum number of values per calendar month for a fit

# Korean ASOS column names -> English names used in this script
PRECIP_COL = "일강수량(mm)"   # daily precipitation (mm); empty = no precipitation
DATE_COL = "일시"             # date
INTERP_COLUMNS = {
    "평균기온(°C)": "tmean",          # mean air temperature (deg C)
    "최저기온(°C)": "tmin",           # minimum air temperature (deg C)
    "최고기온(°C)": "tmax",           # maximum air temperature (deg C)
    "평균 상대습도(%)": "rhmean",     # mean relative humidity (%)
    "평균 풍속(m/s)": "wind10",       # mean wind speed at 10 m (m/s)
    "합계 일사량(MJ/m2)": "rs",       # daily total solar radiation (MJ m-2 day-1)
    "평균 현지기압(hPa)": "pres_hpa",  # mean station pressure (hPa)
}


# -----------------------------------------------------------------------------
# Step 1-2: daily data and ET0
# -----------------------------------------------------------------------------
def load_daily(csv_path):
    """Read the merged daily ASOS file and build the daily input table.

    Returns a DataFrame with columns: date, precip, tmean, tmin, tmax, rhmean,
    wind10, rs, pres_hpa.
    """
    raw = pd.read_csv(csv_path, encoding="utf-8-sig", parse_dates=[DATE_COL])
    raw = raw.sort_values(DATE_COL).reset_index(drop=True)

    daily = pd.DataFrame({"date": raw[DATE_COL]})
    # Empty precipitation = no precipitation
    daily["precip"] = raw[PRECIP_COL].fillna(0)
    # Other variables: short gaps only
    for src, new in INTERP_COLUMNS.items():
        daily[new] = raw[src].interpolate(method="linear", limit=MAX_INTERP_GAP_DAYS)

    remaining = daily[list(INTERP_COLUMNS.values())].isna().sum()
    print("Missing values remaining after interpolation:")
    print(remaining.to_string())
    return daily


def saturation_vapour_pressure(t_celsius):
    """Saturation vapour pressure (kPa) at air temperature t (deg C); FAO-56 Eq. 11."""
    return 0.6108 * np.exp(17.27 * t_celsius / (t_celsius + 237.3))


def compute_et0(daily):
    """Daily FAO-56 Penman-Monteith ET0 (mm/day) for the table from load_daily()."""
    lat = np.deg2rad(LAT_DEG)
    t = daily["tmean"].to_numpy()
    tmax = daily["tmax"].to_numpy()
    tmin = daily["tmin"].to_numpy()
    rh = daily["rhmean"].to_numpy()
    u_z = daily["wind10"].to_numpy()
    rs = daily["rs"].to_numpy()
    p_kpa = daily["pres_hpa"].to_numpy() / 10.0  # hPa -> kPa
    doy = daily["date"].dt.dayofyear.to_numpy()

    # Vapour pressures (kPa)
    es = (saturation_vapour_pressure(tmax) + saturation_vapour_pressure(tmin)) / 2.0
    ea = rh / 100.0 * es
    delta = 4098.0 * saturation_vapour_pressure(t) / (t + 237.3) ** 2  # slope (kPa/degC)
    gamma = 0.665e-3 * p_kpa                                          # psychrometric constant

    # Wind speed at 2 m from the measurement height (FAO-56 Eq. 47)
    u2 = u_z * 4.87 / np.log(67.8 * WIND_HEIGHT_M - 5.42)

    # Extraterrestrial radiation Ra (MJ m-2 day-1)
    gsc = 0.0820  # solar constant (MJ m-2 min-1)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
    delta_sol = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(delta_sol), -1, 1))
    ra = (24 * 60 / np.pi) * gsc * dr * (
        ws * np.sin(lat) * np.sin(delta_sol) + np.cos(lat) * np.cos(delta_sol) * np.sin(ws))

    # Net radiation (MJ m-2 day-1)
    rso = (0.75 + 2e-5 * ELEV_M) * ra          # clear-sky radiation
    rns = (1 - 0.23) * rs                       # net shortwave (albedo 0.23)
    sigma = 4.903e-9                            # Stefan-Boltzmann constant (MJ K-4 m-2 day-1)
    tmax_k, tmin_k = tmax + 273.16, tmin + 273.16
    rs_rso = np.clip(rs / np.where(rso == 0, np.nan, rso), 0.3, 1.0)  # limit relative radiation
    rnl = (sigma * (tmax_k ** 4 + tmin_k ** 4) / 2
           * (0.34 - 0.14 * np.sqrt(np.clip(ea, 0, None)))
           * (1.35 * rs_rso - 0.35))
    rn = rns - rnl

    # FAO-56 Penman-Monteith (G = 0)
    et0 = ((0.408 * delta * rn + gamma * (900 / (t + 273)) * u2 * (es - ea))
           / (delta + gamma * (1 + 0.34 * u2)))
    return np.maximum(et0, 0)  # no negative ET0


# -----------------------------------------------------------------------------
# Step 3-4: monthly water balance and SPEI
# -----------------------------------------------------------------------------
def monthly_water_balance(daily):
    """Monthly totals of precipitation (P) and ET0 (PET) and D = P - PET."""
    d = daily.loc[(daily["date"] >= START_DATE) & (daily["date"] < END_DATE_EXCLUSIVE)]
    if d.empty:
        raise ValueError("No daily input data in the selected analysis period.")
    monthly = d.set_index("date").resample("MS").agg(
        P=("precip", lambda x: x.sum(min_count=1)),
        PET=("ET0_mm", lambda x: x.sum(min_count=1)),
    )
    last_month = END_DATE_EXCLUSIVE - pd.offsets.MonthBegin(1)
    monthly = monthly.reindex(pd.date_range(START_DATE, last_month, freq="MS"))
    monthly.index.name = "date"
    monthly["D"] = monthly["P"] - monthly["PET"]
    return monthly


def compute_spei(d_series, scale):
    """SPEI for one accumulation period (months).

    D is accumulated over ``scale`` months; for every calendar month a
    generalized-logistic (log-logistic) distribution is fitted by L-moments and
    the cumulative probabilities are converted to standard-normal quantiles.
    """
    accum = d_series.rolling(scale, min_periods=scale).sum()
    spei = pd.Series(np.nan, index=accum.index, dtype=float)
    for month in range(1, 13):
        vals = accum[accum.index.month == month].dropna()
        if len(vals) < MIN_MONTHS_FOR_FIT:
            continue
        params = distr.glo.lmom_fit(vals.values)
        cdf = distr.glo.cdf(vals.values, **params)
        spei.loc[vals.index] = stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
    return spei


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute ET0 (FAO-56 PM) and monthly SPEI.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="merged daily ASOS CSV (default: %(default)s)")
    parser.add_argument("--daily-output", type=Path, default=DEFAULT_DAILY,
                        help="daily inputs and ET0 (default: %(default)s)")
    parser.add_argument("--monthly-output", type=Path, default=DEFAULT_MONTHLY,
                        help="monthly P, PET, D, SPEI (default: %(default)s)")
    args = parser.parse_args()

    # Daily ET0
    daily = load_daily(args.input)
    daily["ET0_mm"] = compute_et0(daily)
    print("\nDaily ET0 (mm/day), summary statistics:")
    print(daily["ET0_mm"].describe().to_string())
    print("\nMean ET0 by calendar month (seasonality check, mm/day):")
    print(daily.groupby(daily["date"].dt.month)["ET0_mm"].mean().round(2).to_string())
    args.daily_output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.daily_output, index=False)

    # Monthly water balance and SPEI
    monthly = monthly_water_balance(daily)
    for scale in SPEI_SCALES:
        monthly[f"SPEI_{scale}"] = compute_spei(monthly["D"], scale)
    cols = [f"SPEI_{s}" for s in SPEI_SCALES]
    print("\nNumber of valid SPEI months:")
    print(monthly[cols].count().to_string())
    target = pd.Timestamp("2025-08-01")
    if target in monthly.index:
        print("\nAugust 2025:")
        print(monthly.loc[target, cols].to_string())
    args.monthly_output.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(args.monthly_output)
    print(f"\nSaved: {args.daily_output}\nSaved: {args.monthly_output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
05_ml_cv_shap.py
================
Machine-learning and explainable-AI (SHAP) analysis of the 30-day precipitation
deficit (Text S1, Supporting Tables S2, S3, S5; SHAP figures).

Purpose
-------
Using the 30-day trailing mean of the Gangneung ASOS precipitation anomaly
(``precip_anom_ma30``) as the target, explore the relative information that local
thermodynamic / mid-tropospheric variables, East Asian Arctic-sector circulation
and the AO provide to fitted models.

Workflow
--------
1. Load the VIF-selected dataset produced by ``05_build_ml_dataset.py``.
2. Evaluate six models (Random Forest, XGBoost, LightGBM, HistGradientBoosting,
   Extra Trees, Ridge) with a 5-fold expanding-window ``TimeSeriesSplit``
   (RMSE, MAE, R2); the best model is the one with the lowest mean RMSE.
3. Refit the best model on all data and compute SHAP values (TreeExplainer for
   tree models, LinearExplainer for Ridge): global summary, crisis-period mean
   (13 Aug - 13 Sep 2025), and a local waterfall plot for 19 Aug 2025.
4. Compare mean SHAP values over equal 30-day windows for the 2015, 2022 and 2025
   events; each window ends on the day of the minimum ``precip_anom_ma30`` in
   Jul-Sep of that year.

Notes
-----
- ASOS-precipitation-derived and local NCEP2 precipitation-rate-derived variables
  are excluded from the predictors (target leakage).
- SHAP values describe the contribution structure inside the fitted model; they
  are NOT causal effects.

Input
-----
    data/processed/final_ml_dataset_vif_selected_precip_target.csv

Output
------
    results/tables/   supporting_table_S2_cv_structure.csv, supporting_table_S3_model_hyperparameters.csv,
                      supporting_table_S5_event_windows.csv, model_performance_precip_target.csv,
                      cv_folds_detail_<model>.csv, shap_crisis_period_mean_precip_target.csv,
                      shap_comparison_equal30d_precip_target.csv, predictor_list_precip_target.csv
    results/figures/  fig08_shap_summary_global.png,
                      fig09_shap_waterfall_peak.png,
                      fig10_shap_comparison_equal30d.png

Usage
-----
    python scripts/05_ml_cv_shap.py
    python scripts/05_ml_cv_shap.py --data my_dataset.csv --n-jobs 1

Requires: scikit-learn, xgboost, lightgbm, shap, pandas, numpy, matplotlib.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: figures are written to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "final_ml_dataset_vif_selected_precip_target.csv"
DEFAULT_TABLEDIR = REPO_ROOT / "results" / "tables"
DEFAULT_FIGDIR = REPO_ROOT / "results" / "figures"

TARGET_COL = "precip_anom_ma30"     # 30-day trailing mean ASOS precipitation anomaly
N_SPLITS = 5
RANDOM_STATE = 42

CRISIS_START, CRISIS_END = "2025-08-13", "2025-09-13"
PEAK_DATE = "2025-08-19"            # date of the local SHAP (waterfall) plot
COMPARISON_YEARS = [2015, 2022, 2025]
COMPARISON_WINDOW_DAYS = 30
COMPARISON_SEARCH_MONTHS = (7, 8, 9)  # same seasonal window for the historical comparison

NON_FEATURE_COLS = {"date", "storage_rate", "storage_rate_anom", TARGET_COL}
TREE_MODELS = {"RandomForest", "XGBoost", "LightGBM", "HistGradientBoosting", "ExtraTrees"}

# Hyperparameters as reported in Supporting Table S3
S3_ROWS = [
    ["Random Forest", "n_estimators=300; max_depth=6; min_samples_leaf=15; max_features=0.5; random_state=42"],
    ["XGBoost", "n_estimators=300; max_depth=3; learning_rate=0.02; subsample=0.7; colsample_bytree=0.6; "
                "reg_alpha=1.0; reg_lambda=2.0; min_child_weight=10; random_state=42"],
    ["LightGBM", "n_estimators=300; max_depth=3; learning_rate=0.02; subsample=0.7; colsample_bytree=0.6; "
                 "reg_alpha=1.0; reg_lambda=2.0; min_child_samples=20; random_state=42"],
    ["HistGradientBoosting", "max_iter=300; max_depth=3; learning_rate=0.02; min_samples_leaf=15; "
                             "l2_regularization=1.0; random_state=42"],
    ["Extra Trees", "n_estimators=300; max_depth=6; min_samples_leaf=15; max_features=0.5; random_state=42"],
    ["Ridge", "StandardScaler + Ridge(alpha=10.0)"],
]


# -----------------------------------------------------------------------------
# Data and models
# -----------------------------------------------------------------------------
def is_direct_precip_leakage(col):
    """Observed precipitation anomalies and local NCEP2 prate anomalies duplicate the target."""
    return col.startswith("precip_anom") or col.startswith("prate_anom")


def load_dataset(path):
    """Return the complete-case DataFrame, predictor names, X and y (sorted by date)."""
    df = pd.read_csv(path, parse_dates=["date"])
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target '{TARGET_COL}' not found in {path}. Run 05_build_ml_dataset.py first.")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and not is_direct_precip_leakage(c)]
    if not feature_cols:
        raise ValueError("No usable predictors.")
    print(f"Target: {TARGET_COL}")
    print(f"Predictors ({len(feature_cols)}): {feature_cols}")

    df_clean = df.dropna(subset=feature_cols + [TARGET_COL]).sort_values("date").reset_index(drop=True)
    print(f"Usable rows: {len(df_clean)} / total: {len(df)}")
    return df_clean, feature_cols, df_clean[feature_cols], df_clean[TARGET_COL]


def make_models(n_jobs=-1):
    """The six models (hyperparameters of Supporting Table S3)."""
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=15, max_features=0.5,
            random_state=RANDOM_STATE, n_jobs=n_jobs),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=2.0,
            min_child_weight=10, random_state=RANDOM_STATE, n_jobs=n_jobs),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=2.0,
            min_child_samples=20, verbose=-1, random_state=RANDOM_STATE, n_jobs=n_jobs),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, max_depth=3, learning_rate=0.02,
            min_samples_leaf=15, l2_regularization=1.0, random_state=RANDOM_STATE),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=15, max_features=0.5,
            random_state=RANDOM_STATE, n_jobs=n_jobs),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    }


# -----------------------------------------------------------------------------
# Cross-validation (Supporting Table S2)
# -----------------------------------------------------------------------------
def cross_validate(x, y, df_clean, tabledir, n_jobs):
    """5-fold expanding-window evaluation; returns the model-performance table."""
    model_names = list(make_models().keys())
    results = {name: [] for name in model_names}
    fold_structure = []

    for fold, (train_idx, test_idx) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS).split(x), start=1):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        fold_structure.append({
            "Fold": fold,
            "Training_start": df_clean.loc[train_idx[0], "date"].date(),
            "Training_end": df_clean.loc[train_idx[-1], "date"].date(),
            "Validation_start": df_clean.loc[test_idx[0], "date"].date(),
            "Validation_end": df_clean.loc[test_idx[-1], "date"].date(),
            "N_train": len(train_idx),
            "N_validation": len(test_idx),
        })
        for name, model in make_models(n_jobs).items():
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            pred_train = model.predict(x_train)
            results[name].append({
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "MAE": mean_absolute_error(y_test, pred),
                "R2_test": r2_score(y_test, pred),
                "R2_train": r2_score(y_train, pred_train),
            })

    pd.DataFrame(fold_structure).to_csv(tabledir / "supporting_table_S2_cv_structure.csv", index=False)

    perf_rows = []
    for name, folds in results.items():
        fdf = pd.DataFrame(folds)
        fdf.to_csv(tabledir / f"cv_folds_detail_{name}.csv", index=False)
        perf_rows.append({
            "model": name,
            "RMSE_mean": fdf["RMSE"].mean(), "RMSE_std": fdf["RMSE"].std(),
            "MAE_mean": fdf["MAE"].mean(), "MAE_std": fdf["MAE"].std(),
            "R2_test_mean": fdf["R2_test"].mean(), "R2_test_std": fdf["R2_test"].std(),
            "R2_train_mean": fdf["R2_train"].mean(),
        })
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(tabledir / "model_performance_precip_target.csv", index=False)
    print("\n=== Model performance (5-fold TimeSeriesSplit) ===")
    print(perf_df.to_string(index=False))
    return perf_df


# -----------------------------------------------------------------------------
# SHAP
# -----------------------------------------------------------------------------
def compute_shap(best_model_name, x, y, feature_cols, index, n_jobs):
    """Refit the best model on all data; return SHAP values (array), expected value, DataFrame."""
    final_model = make_models(n_jobs)[best_model_name]
    final_model.fit(x, y)

    if best_model_name in TREE_MODELS:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(x)
    else:  # Ridge pipeline: explain the linear model in standardized space
        scaler = final_model.named_steps["standardscaler"]
        ridge_model = final_model.named_steps["ridge"]
        x_scaled = pd.DataFrame(scaler.transform(x), columns=x.columns, index=x.index)
        explainer = shap.LinearExplainer(ridge_model, x_scaled)
        shap_values = explainer.shap_values(x_scaled)
    expected_value = float(np.array(explainer.expected_value).flatten()[0])
    shap_df = pd.DataFrame(shap_values, columns=feature_cols, index=index)
    return shap_values, expected_value, shap_df


def save_png(path):
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_global_summary(shap_values, x, feature_cols, figdir):
    plt.figure(figsize=(9, 7))
    # fixed rng: the vertical jitter of the beeswarm is otherwise random between runs
    shap.summary_plot(shap_values, x, feature_names=feature_cols, show=False,
                      rng=np.random.default_rng(RANDOM_STATE))
    plt.title("Global SHAP summary — 30-day precipitation anomaly target", fontsize=11)
    plt.tight_layout()
    save_png(figdir / "fig08_shap_summary_global.png")


def plot_waterfall(shap_values, expected_value, feature_cols, df_clean, figdir):
    """Local SHAP at PEAK_DATE (uses shap's legacy waterfall; API may change between shap versions)."""
    if not (df_clean["date"] == PEAK_DATE).any():
        print(f"WARNING: {PEAK_DATE} is not in the dataset; waterfall plot skipped.")
        return
    peak_idx = df_clean.index[df_clean["date"] == PEAK_DATE][0]
    plt.figure(figsize=(9, 6))
    shap.plots._waterfall.waterfall_legacy(
        expected_value, shap_values[peak_idx], feature_names=feature_cols,
        max_display=12, show=False)
    plt.title(f"Local SHAP — {PEAK_DATE} (30-day precipitation anomaly)", fontsize=11)
    plt.tight_layout()
    save_png(figdir / "fig09_shap_waterfall_peak.png")


def event_comparison(shap_df, df_clean, tabledir, figdir):
    """Mean SHAP over equal 30-day windows ending at the Jul-Sep minimum of the target (Table S5)."""
    comparison_rows, event_rows = {}, []
    for yr in COMPARISON_YEARS:
        season_mask = ((df_clean["date"].dt.year == yr) &
                       (df_clean["date"].dt.month >= COMPARISON_SEARCH_MONTHS[0]) &
                       (df_clean["date"].dt.month <= COMPARISON_SEARCH_MONTHS[2]))
        if season_mask.sum() == 0:
            continue
        yr_series = df_clean.loc[season_mask, ["date", TARGET_COL]].dropna()
        worst_row = yr_series.loc[yr_series[TARGET_COL].idxmin()]
        worst_end = pd.Timestamp(worst_row["date"])
        worst_start = worst_end - pd.Timedelta(days=COMPARISON_WINDOW_DAYS - 1)
        sub_mask = (df_clean["date"] >= worst_start) & (df_clean["date"] <= worst_end)

        label = f"{yr} ({worst_start.date()} to {worst_end.date()})"
        comparison_rows[label] = shap_df.loc[sub_mask].mean()
        event_rows.append({
            "Event_year": yr,
            "Selection_basis": f"Minimum {TARGET_COL} within Jul-Sep",
            "Window_start": worst_start.date(),
            "Window_end": worst_end.date(),
            "Window_days": COMPARISON_WINDOW_DAYS,
            "Minimum_target_value": float(worst_row[TARGET_COL]),
        })
    pd.DataFrame(event_rows).to_csv(tabledir / "supporting_table_S5_event_windows.csv", index=False)

    if not comparison_rows:
        return
    comparison_df = pd.DataFrame(comparison_rows)
    col_2025 = [c for c in comparison_df.columns if c.startswith("2025")]
    if col_2025:  # order predictors by |SHAP| in the 2025 window
        comparison_df = comparison_df.loc[comparison_df[col_2025[0]].abs().sort_values(ascending=False).index]
    comparison_df.to_csv(tabledir / "shap_comparison_equal30d_precip_target.csv")

    top_vars = comparison_df.head(8).index
    ax = comparison_df.loc[top_vars].plot(kind="barh", figsize=(10, 6))
    ax.set_xlabel("Mean SHAP value (contribution to 30-day precipitation anomaly)")
    ax.set_title("SHAP comparison: equal 30-day historical event windows")
    ax.axvline(0, color="black", linewidth=0.6)
    plt.tight_layout()
    save_png(figdir / "fig10_shap_comparison_equal30d.png")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Model comparison and SHAP analysis of the precipitation deficit.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="input CSV (default: %(default)s)")
    parser.add_argument("--tabledir", type=Path, default=DEFAULT_TABLEDIR, help="table output (default: %(default)s)")
    parser.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR, help="figure output (default: %(default)s)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel jobs for tree models (default: all cores)")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Input not found: {args.data}\nRun scripts/05_build_ml_dataset.py first.")
    args.tabledir.mkdir(parents=True, exist_ok=True)
    args.figdir.mkdir(parents=True, exist_ok=True)

    df_clean, feature_cols, x, y = load_dataset(args.data)
    pd.DataFrame(S3_ROWS, columns=["Model", "Hyperparameters"]).to_csv(
        args.tabledir / "supporting_table_S3_model_hyperparameters.csv", index=False)

    perf_df = cross_validate(x, y, df_clean, args.tabledir, args.n_jobs)
    best_model_name = perf_df.loc[perf_df["RMSE_mean"].idxmin(), "model"]
    print(f"\nBest model (lowest mean RMSE): {best_model_name}")

    shap_values, expected_value, shap_df = compute_shap(
        best_model_name, x, y, feature_cols, df_clean.index, args.n_jobs)

    plot_global_summary(shap_values, x, feature_cols, args.figdir)

    crisis_mask = (df_clean["date"] >= CRISIS_START) & (df_clean["date"] <= CRISIS_END)
    crisis_mean = shap_df.loc[crisis_mask].mean().sort_values(key=np.abs, ascending=False)
    crisis_mean.to_csv(args.tabledir / "shap_crisis_period_mean_precip_target.csv")

    plot_waterfall(shap_values, expected_value, feature_cols, df_clean, args.figdir)
    event_comparison(shap_df, df_clean, args.tabledir, args.figdir)

    pd.DataFrame({"Predictor": feature_cols}).to_csv(args.tabledir / "predictor_list_precip_target.csv", index=False)

    print("\nDone")
    print(f"Target: {TARGET_COL}\nBest model: {best_model_name}")
    print(f"Tables: {args.tabledir}\nFigures: {args.figdir}")


if __name__ == "__main__":
    main()

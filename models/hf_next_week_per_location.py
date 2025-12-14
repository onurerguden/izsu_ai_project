import warnings
warnings.filterwarnings("ignore")

# ✅ Add this BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error
)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# -----------------------------
# Config
# -----------------------------
CSV_PATH = "izsu_health_factor.csv"
DATE_COL = "Tarih"
LOC_COL = "NoktaAdi"
TARGET_COL = "HealthFactor"

OUT_DIR = Path("ai_outputs/per_location_next_week")
MODELS_DIR = OUT_DIR / "models"
REPORTS_DIR = OUT_DIR / "reports"
PLOTS_DIR = OUT_DIR / "plots"

for d in [OUT_DIR, MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Minimum split sizes (küçük seriler için)
MIN_TRAIN = 12
MIN_VAL = 3
MIN_TEST = 3
MIN_TOTAL = MIN_TRAIN + MIN_VAL + MIN_TEST  # 18


# -----------------------------
# Helpers
# -----------------------------
def safe_mape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) >= 2 and np.unique(y_true).size > 1:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = np.nan

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)

    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape, "MedAE": medae}


def get_models():
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=3,
            min_samples_leaf=8, subsample=0.7, random_state=42
        ),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            verbosity=0
        )
    return models


def make_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("model", model),
    ])


def sanitize_name(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")


def build_features(loc_df: pd.DataFrame):
    """
    Features at time t; target is next time (t+1) HF.
    """
    df = loc_df.sort_values(DATE_COL).copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # next-week target
    df["y_next"] = df[TARGET_COL].shift(-1)

    # calendar
    df["Month"] = df[DATE_COL].dt.month

    # HF lags/rolling (past only)
    df["HF_Lag1"] = df[TARGET_COL].shift(1)
    df["HF_Lag2"] = df[TARGET_COL].shift(2)
    df["HF_RollMean3"] = df[TARGET_COL].shift(1).rolling(3, min_periods=1).mean()
    df["HF_RollMean7"] = df[TARGET_COL].shift(1).rolling(7, min_periods=1).mean()
    df["HF_Change"] = df["HF_Lag1"] - df["HF_Lag2"]
    df["HF_DiffFromMean3"] = df["HF_Lag1"] - df["HF_RollMean3"]

    # score features (t anındaki ölçümler OK) + lag1
    score_cols = [c for c in df.columns if "_score" in c.lower()]
    for c in score_cols:
        df[f"{c}_Lag1"] = df[c].shift(1)

    base_feats = [
        "Month", "HF_Lag1", "HF_Lag2",
        "HF_RollMean3", "HF_RollMean7",
        "HF_Change", "HF_DiffFromMean3"
    ]
    same_time_scores = score_cols
    lag_score_feats = [f"{c}_Lag1" for c in score_cols]

    features = [f for f in (base_feats + same_time_scores + lag_score_feats) if f in df.columns]

    # drop rows where target missing (last row)
    df = df.dropna(subset=["y_next"]).copy()

    return df, features


def smart_time_split(df_feat: pd.DataFrame):
    n = len(df_feat)
    if n < MIN_TOTAL:
        return None

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_end = max(train_end, MIN_TRAIN)
    val_end = max(val_end, train_end + MIN_VAL)

    if n - val_end < MIN_TEST:
        val_end = n - MIN_TEST

    if train_end < MIN_TRAIN:
        return None
    if val_end - train_end < MIN_VAL:
        return None
    if n - val_end < MIN_TEST:
        return None

    train_df = df_feat.iloc[:train_end].copy()
    val_df = df_feat.iloc[train_end:val_end].copy()
    test_df = df_feat.iloc[val_end:].copy()

    return train_df, val_df, test_df


def save_location_plots(loc_name, model_name, pipe, feature_cols, train_df, val_df, test_df):
    """
    Saves plots into plots/<location_safe>/...
    """
    loc_safe = sanitize_name(loc_name)
    loc_plot_dir = PLOTS_DIR / loc_safe
    loc_plot_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    X_test = test_df[feature_cols]
    y_test = test_df["y_next"].values
    dates_test = pd.to_datetime(test_df[DATE_COL]).values

    y_pred = pipe.predict(X_test)
    residuals = y_test - y_pred

    # 01 Actual vs Pred (Test)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    mn = float(np.min([y_test.min(), y_pred.min()]))
    mx = float(np.max([y_test.max(), y_pred.max()]))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Actual (y_next)")
    plt.ylabel("Predicted")
    plt.title(f"{loc_name}\nTest: Actual vs Pred ({model_name})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loc_plot_dir / "01_actual_vs_pred_test.png", dpi=200)
    plt.close()

    # 02 Residual analysis (Test)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=20, alpha=0.8, edgecolor="black")
    plt.axvline(np.mean(residuals), linestyle="--")
    plt.title("Residuals Histogram (Test)")
    plt.xlabel("Residual (Actual - Pred)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.title("Residuals vs Predicted (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.grid(True, alpha=0.3)

    plt.suptitle(f"{loc_name} ({model_name})", y=1.02)
    plt.tight_layout()
    plt.savefig(loc_plot_dir / "02_residuals_test.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 03 Time series plot (Test) - dates on x-axis
    order = np.argsort(dates_test)
    dates_sorted = dates_test[order]
    y_test_sorted = y_test[order]
    y_pred_sorted = y_pred[order]

    plt.figure(figsize=(10, 5))
    plt.plot(dates_sorted, y_test_sorted, marker="o", label="Actual")
    plt.plot(dates_sorted, y_pred_sorted, marker="o", label="Predicted")
    plt.title(f"{loc_name}\nTest Timeline: Actual vs Pred ({model_name})")
    plt.xlabel("Date")
    plt.ylabel("HF (next week)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loc_plot_dir / "03_timeseries_test.png", dpi=200)
    plt.close()


def fit_best_model_for_location(loc_name: str, loc_df: pd.DataFrame):
    df_feat, feature_cols = build_features(loc_df)

    split = smart_time_split(df_feat)
    if split is None:
        return None

    train_df, val_df, test_df = split

    X_train, y_train = train_df[feature_cols], train_df["y_next"]
    X_val, y_val = val_df[feature_cols], val_df["y_next"]
    X_test, y_test = test_df[feature_cols], test_df["y_next"]

    models = get_models()
    rows = []

    best_name = None
    best_pipe = None
    best_score = -np.inf

    for name, model in models.items():
        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)

        pred_train = pipe.predict(X_train)
        pred_val = pipe.predict(X_val)
        pred_test = pipe.predict(X_test)

        m_train = evaluate(y_train, pred_train)
        m_val = evaluate(y_val, pred_val)
        m_test = evaluate(y_test, pred_test)

        row = {
            "Location": loc_name,
            "Model": name,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            **{f"Train_{k}": v for k, v in m_train.items()},
            **{f"Val_{k}": v for k, v in m_val.items()},
            **{f"Test_{k}": v for k, v in m_test.items()},
        }
        rows.append(row)

        test_r2 = m_test["R2"]
        score = test_r2 if not np.isnan(test_r2) else -m_test["MAE"]

        if score > best_score:
            best_score = score
            best_name = name
            best_pipe = pipe

    results_df = pd.DataFrame(rows)

    # Forecast next week from last known row (t)
    df_full = loc_df.sort_values(DATE_COL).copy()
    df_full[DATE_COL] = pd.to_datetime(df_full[DATE_COL])

    tmp = df_full.sort_values(DATE_COL).copy()
    tmp["Month"] = tmp[DATE_COL].dt.month
    tmp["HF_Lag1"] = tmp[TARGET_COL].shift(1)
    tmp["HF_Lag2"] = tmp[TARGET_COL].shift(2)
    tmp["HF_RollMean3"] = tmp[TARGET_COL].shift(1).rolling(3, min_periods=1).mean()
    tmp["HF_RollMean7"] = tmp[TARGET_COL].shift(1).rolling(7, min_periods=1).mean()
    tmp["HF_Change"] = tmp["HF_Lag1"] - tmp["HF_Lag2"]
    tmp["HF_DiffFromMean3"] = tmp["HF_Lag1"] - tmp["HF_RollMean3"]

    score_cols = [c for c in tmp.columns if "_score" in c.lower()]
    for c in score_cols:
        tmp[f"{c}_Lag1"] = tmp[c].shift(1)

    feature_cols_forecast = [c for c in feature_cols if c in tmp.columns]
    last_row = tmp.iloc[[-1]][feature_cols_forecast]

    next_week_pred = float(best_pipe.predict(last_row)[0])
    last_date = pd.to_datetime(df_full[DATE_COL].max())

    # --- PLOTS ---
    save_location_plots(
        loc_name=loc_name,
        model_name=best_name,
        pipe=best_pipe,
        feature_cols=feature_cols,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df
    )

    return {
        "feature_cols": feature_cols,
        "results_df": results_df,
        "best_name": best_name,
        "best_pipe": best_pipe,
        "next_week_pred": next_week_pred,
        "last_date": last_date,
        "n_feat_rows": len(df_feat),
    }


def save_global_summary_plots(summary_df: pd.DataFrame):
    # R2 histogram
    if "Test_R2" in summary_df.columns and summary_df["Test_R2"].notna().any():
        plt.figure(figsize=(8, 5))
        plt.hist(summary_df["Test_R2"].dropna().values, bins=15, edgecolor="black", alpha=0.85)
        plt.title("Global: Test R2 Distribution (Best model per location)")
        plt.xlabel("Test R2")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "_GLOBAL_test_r2_hist.png", dpi=200)
        plt.close()

    # MAE histogram
    if "Test_MAE" in summary_df.columns and summary_df["Test_MAE"].notna().any():
        plt.figure(figsize=(8, 5))
        plt.hist(summary_df["Test_MAE"].dropna().values, bins=15, edgecolor="black", alpha=0.85)
        plt.title("Global: Test MAE Distribution (Best model per location)")
        plt.xlabel("Test MAE")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "_GLOBAL_test_mae_hist.png", dpi=200)
        plt.close()


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH

    print(f"[LOAD] {csv_path}")
    df = pd.read_csv(csv_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    if any(c not in df.columns for c in [LOC_COL, DATE_COL, TARGET_COL]):
        raise ValueError(f"CSV must include columns: {LOC_COL}, {DATE_COL}, {TARGET_COL}")

    df = df.dropna(subset=[LOC_COL, DATE_COL, TARGET_COL]).copy()

    locations = sorted(df[LOC_COL].unique().tolist())
    print(f"[INFO] Locations: {len(locations)}")

    all_best_rows = []
    forecasts = []
    trained_count = 0

    for i, loc in enumerate(locations, 1):
        loc_df = df[df[LOC_COL] == loc].sort_values(DATE_COL).copy()

        usable = max(len(loc_df) - 1, 0)
        if usable < MIN_TOTAL:
            print(f"[SKIP] {loc} (rows={len(loc_df)} usable={usable}) < MIN_TOTAL({MIN_TOTAL})")
            continue

        out = fit_best_model_for_location(loc, loc_df)
        if out is None:
            print(f"[SKIP] {loc} (rows={len(loc_df)} usable={usable}) split failed")
            continue

        trained_count += 1
        loc_safe = sanitize_name(loc)

        metrics_path = REPORTS_DIR / f"{loc_safe}_metrics.csv"
        out["results_df"].to_csv(metrics_path, index=False)

        model_path = MODELS_DIR / f"{loc_safe}_best_{out['best_name']}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "location": loc,
                "best_model_name": out["best_name"],
                "pipeline": out["best_pipe"],
                "feature_cols": out["feature_cols"]
            }, f)

        best_row = out["results_df"][out["results_df"]["Model"] == out["best_name"]].iloc[0].to_dict()
        best_row["BestModelFile"] = str(model_path)
        best_row["FeatureRowsUsed"] = out["n_feat_rows"]
        all_best_rows.append(best_row)

        forecasts.append({
            "Location": loc,
            "LastDate": out["last_date"],
            "PredictedNextWeekHF": out["next_week_pred"],
            "BestModel": out["best_name"],
            "ModelFile": str(model_path)
        })

        print(f"[{i:03d}/{len(locations)}] TRAINED {loc} -> best={out['best_name']} | next_week_pred={out['next_week_pred']:.4f}")

    if trained_count == 0:
        print("[DONE] No locations trained. Series too short.")
        print(f"       Need usable >= {MIN_TOTAL} (after shifting y_next).")
        return

    summary_df = pd.DataFrame(all_best_rows)
    summary_path = OUT_DIR / "summary_best_models_all_locations.csv"
    summary_df.to_csv(summary_path, index=False)

    forecast_df = pd.DataFrame(forecasts).sort_values("PredictedNextWeekHF", ascending=False)
    forecast_path = OUT_DIR / "forecast_next_week_all_locations.csv"
    forecast_df.to_csv(forecast_path, index=False)

    save_global_summary_plots(summary_df)

    print("\n[SUCCESS] Completed!")
    print(f"[TRAINED] Locations trained: {trained_count}")
    print(f"[OUTPUT] Summary:  {summary_path}")
    print(f"[OUTPUT] Forecast: {forecast_path}")
    print(f"[OUTPUT] Models:   {MODELS_DIR}")
    print(f"[OUTPUT] Reports:  {REPORTS_DIR}")
    print(f"[OUTPUT] Plots:    {PLOTS_DIR}  (and per-location subfolders)")


if __name__ == "__main__":
    main()

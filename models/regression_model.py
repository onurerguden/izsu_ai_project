import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import TimeSeriesSplit

import warnings
warnings.filterwarnings("ignore")

# =====================================================
# CONFIG
# =====================================================
CSV_PATH = "izsu_health_factor.csv"
TARGET_COL = "HealthFactor"
DROP_THRESHOLD = -0.05     # drop definition
RISE_THRESHOLD = 0.05     # rise definition
OUTPUT_DIR = "ai_outputs"
N_SPLITS = 5
RANDOM_STATE = 42

# =====================================================
# DATA PREPARATION
# =====================================================
def load_and_prepare_data():
    df = pd.read_csv(CSV_PATH)
    df["Tarih"] = pd.to_datetime(df["Tarih"])
    df = df.sort_values(["NoktaAdi", "Tarih"]).reset_index(drop=True)

    # Delta calculation
    df["Delta"] = df.groupby("NoktaAdi")[TARGET_COL].diff()

    # Binary DropRisk target
    df["DropRisk"] = (df["Delta"] < DROP_THRESHOLD).astype(int)
    df["RiseRisk"] = (df["Delta"] > RISE_THRESHOLD).astype(int)

    grp = df.groupby("NoktaAdi")[TARGET_COL]
    df["Prev_Value"] = grp.shift(1)
    df["Prev_Delta"] = df.groupby("NoktaAdi")["Delta"].shift(1)

    # Safer rolling with min_periods=2 to avoid wiping small groups
    df["Mean_3Days"] = grp.shift(1).rolling(window=3, min_periods=2).mean()
    df["Std_3Days"] = grp.shift(1).rolling(window=3, min_periods=2).std()
    df["Month"] = df["Tarih"].dt.month

    # Drop locations with too few samples (best practice)
    counts = df.groupby("NoktaAdi").size()
    valid_locations = counts[counts >= 5].index
    df = df[df["NoktaAdi"].isin(valid_locations)]

    df = df.dropna(subset=[
        "Prev_Value",
        "Prev_Delta",
        "Mean_3Days",
        "Std_3Days",
        "Month",
        "DropRisk",
        "RiseRisk"
    ]).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            "Dataset empty after feature engineering. "
            "Likely causes: too-large rolling window or insufficient per-location samples."
        )
    return df

# =====================================================
# TRAIN / VALIDATE / TEST
# =====================================================
def train_validate_test(df, target_col):
    features = [
        "Prev_Value",
        "Prev_Delta",
        "Mean_3Days",
        "Std_3Days",
        "Month"
    ]

    X = df[features]
    y = df[target_col]

    if len(X) < 10:
        raise ValueError(f"Not enough samples for training: {len(X)} rows available.")

    print("\nClass Distribution")
    print(y.value_counts(normalize=True))

    max_splits = min(N_SPLITS, len(X) - 1)
    if max_splits < 2:
        raise ValueError("Not enough samples to perform TimeSeriesSplit.")
    tscv = TimeSeriesSplit(n_splits=max_splits)

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, probs)
        pr_df = pd.DataFrame({
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1]
        })
        # Recall-focused threshold (>= 0.80)
        best_row = pr_df[pr_df["recall"] >= 0.80].sort_values(
            "precision", ascending=False
        ).iloc[0]
        best_threshold = best_row["threshold"]
        preds = (probs >= best_threshold).astype(int)

        fold_metrics.append({
            "Fold": fold,
            "AUC": roc_auc_score(y_val, probs),
            "Recall": recall_score(y_val, preds),
            "Precision": precision_score(y_val, preds),
            "F1": f1_score(y_val, preds)
        })

    metrics_df = pd.DataFrame(fold_metrics)
    print("\nVALIDATION METRICS (TimeSeries CV)")
    print(metrics_df)
    print("\nMEAN METRICS")
    print(metrics_df.mean(numeric_only=True))

    # Final Train/Test split
    split = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_train, model.predict_proba(X_train)[:, 1])
    pr_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision[:-1],
        "recall": recall[:-1]
    })
    best_row = pr_df[pr_df["recall"] >= 0.80].sort_values(
        "precision", ascending=False
    ).iloc[0]
    best_threshold = best_row["threshold"]

    preds = (probs >= best_threshold).astype(int)

    df_test = df.iloc[split:].copy()
    df_test[f"{target_col}_Prob"] = probs

    test_metrics = {
        "AUC": roc_auc_score(y_test, probs),
        "Recall": recall_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "F1": f1_score(y_test, preds)
    }

    print("\nTEST SET METRICS")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nCLASSIFICATION REPORT (TEST)")
    print(classification_report(y_test, preds, target_names=["OK", target_col.upper()]))

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_confusion_matrix(
        y_test,
        preds,
        title=target_col.upper(),
        filename=f"{OUTPUT_DIR}/{target_col.lower()}_confusion_matrix.png"
    )

    prob_col = f"{target_col}_Prob"
    location_risk = (
        df_test
        .groupby("NoktaAdi")[prob_col]
        .agg(["mean", "max", "count"])
        .sort_values("mean", ascending=False)
    )

    print("\nPER-LOCATION RISK PROFILE (Top 10)")
    print(location_risk.head(10))

    df_test[f"Days_To_{target_col}_Est"] = np.where(
        df_test[prob_col] > 0.8, 1,
        np.where(df_test[prob_col] > 0.6, 3,
                 np.where(df_test[prob_col] > 0.4, 7, 14))
    )

    plot_feature_importance(
        model=model,
        features=features,
        title=f"{target_col.upper()} Feature Importance",
        filename=f"{OUTPUT_DIR}/{target_col.lower()}_feature_importance.png"
    )

    return model, features, metrics_df, test_metrics, df_test

# =====================================================
# VISUALIZATION
# =====================================================
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["OK", title],
        yticklabels=["OK", title]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_feature_importance(model, features, title, filename):
    importances = model.feature_importances_
    fi_df = (
        pd.DataFrame({
            "Feature": features,
            "Importance": importances
        })
        .sort_values("Importance", ascending=False)
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=fi_df,
        x="Importance",
        y="Feature"
    )
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    return fi_df


def plot_combined_confusion_matrix(y_true, y_pred, filename):
    labels = ["DROP", "OK", "RISE"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Combined Risk Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# =====================================================
# MAIN
# =====================================================

# This pipeline now includes:
# - Recall-optimized threshold tuning
# - Probabilistic risk scoring
# - Per-location risk profiling
# - Time-to-drop heuristic

def run_pipeline():
    print("DROP RISK CLASSIFICATION PIPELINE")

    try:
        df = load_and_prepare_data()

        print("\n--- DROP RISK MODEL ---")
        drop_model, features, _, _, df_drop_test = train_validate_test(df, target_col="DropRisk")

        print("\n--- RISE RISK MODEL ---")
        rise_model, _, _, _, df_rise_test = train_validate_test(df, target_col="RiseRisk")

        # Merge the two test DataFrames safely
        df_test = df_drop_test.copy()
        df_test["RiseRisk_Prob"] = df_rise_test["RiseRisk_Prob"].values

        df_test["True_Combined"] = np.where(
            df_test["DropRisk"] == 1, 0,
            np.where(df_test["RiseRisk"] == 1, 2, 1)
        )

        df_test["Pred_Combined"] = np.select(
            [
                df_test["DropRisk_Prob"] >= 0.5,
                df_test["RiseRisk_Prob"] >= 0.5
            ],
            [
                0,
                2
            ],
            default=1
        )

        plot_combined_confusion_matrix(
            y_true=df_test["True_Combined"],
            y_pred=df_test["Pred_Combined"],
            filename=f"{OUTPUT_DIR}/combined_confusion_matrix.png"
        )

    except ValueError as e:
        print(f"PIPELINE ERROR: {e}")
        return

    print("\nPIPELINE COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    run_pipeline()
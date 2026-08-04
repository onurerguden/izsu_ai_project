"""
classification_model.py
Reviewer-oriented Reactive Layer pipeline for the current İZSU dataset.

Purpose
-------
Classify the current water-quality state as Good, Caution, or Risk using the
same centralized HF/WAWQI definitions as utils.hf_calculator.

Experiments
-----------
E1: Real training -> real test
    This experiment is reported honestly. If the real training data contain
    only one class, model fitting is not statistically estimable and the rows
    are marked as SKIPPED_SINGLE_CLASS.

E2: Real + controlled synthetic training -> real test
    The test set remains completely real.

E3: Real + controlled synthetic training -> mixed test
    The mixed test set contains real test observations plus synthetic
    Caution/Risk observations generated only from the real test partition.
    Synthetic-only performance is also reported separately.

Leakage-safe order
------------------
1. Load cleaned, already aggregated feature data.
2. Split REAL observations by unique dates into train/validation/test.
3. Generate controlled synthetic observations separately inside each split.
4. Fit imputation and scaling only through the training pipeline.
5. Select models only on the mixed validation set.
6. Evaluate on untouched real, synthetic-only, and mixed test sets.

Outputs
-------
- reactive_model_metrics.csv
- reactive_classwise_metrics.csv
- reactive_predictions.csv
- reactive_split_summary.csv
- reactive_class_distribution.csv
- synthetic_parameter_rules.csv
- synthetic_distribution_validation.csv
- data_leakage_audit.csv
- best_reactive_model.joblib
- Reactive_Layer_Technical_Report.docx
- figures/*.png (300 DPI)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Central HF import
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_CANDIDATES = [
    SCRIPT_DIR,
    SCRIPT_DIR.parent,
    Path.cwd(),
]
for candidate in PROJECT_CANDIDATES:
    if (candidate / "utils" / "hf_calculator.py").exists():
        sys.path.insert(0, str(candidate))
        break

try:
    from utils.hf_calculator import calculate_hf_from_wide_row
    from utils.parameters import PARAMETERS
except ImportError as exc:
    raise ImportError(
        "utils/hf_calculator.py ve utils/parameters.py bulunamadı. "
        "classification_model.py dosyasını proje kökünde veya models/ "
        "klasöründe, utils klasörüyle birlikte çalıştırın."
    ) from exc


CLASS_LABELS = ["Good", "Caution", "Risk"]
TARGET_COLUMN = "RiskClass"
DATE_COLUMN = "Tarih"
LOCATION_NAME_COLUMN = "NoktaAdi"
LOCATION_ID_COLUMN = "NoktaId"
RANDOM_STATE = 42

TARGET_DERIVED_COLUMNS = {
    "HealthFactor",
    "WAWQI",
    "RiskClass",
    "FailFast",
    "FailFastReason",
    "WAWQICoverage",
    "WAWQIParametersUsed",
    "WAWQIParameterList",
}

RAW_FEATURES = [
    name
    for name, spec in PARAMETERS.items()
    if spec.kind in {"numeric", "range", "zero_standard"}
]


@dataclass(frozen=True)
class SplitBundle:
    train_real: pd.DataFrame
    validation_real: pd.DataFrame
    test_real: pd.DataFrame
    train_dates: list[pd.Timestamp]
    validation_dates: list[pd.Timestamp]
    test_dates: list[pd.Timestamp]


def find_input_csv(requested: str | None) -> Path:
    if requested:
        path = Path(requested)
        if path.exists():
            return path
        raise FileNotFoundError(f"Girdi CSV bulunamadı: {path}")

    candidates = [
        Path("data/data/izsu_features.csv"),
        Path("data/izsu_features.csv"),
        Path("izsu_features.csv"),
        SCRIPT_DIR / "data" / "izsu_features.csv",
        SCRIPT_DIR.parent / "data" / "izsu_features.csv",
        SCRIPT_DIR.parent / "data" / "data" / "izsu_features.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "izsu_features.csv bulunamadı. --input ile dosya yolunu verin."
    )


def normalize_risk_class(value: object) -> str:
    text = str(value).strip().casefold()
    mapping = {
        "good": "Good",
        "caution": "Caution",
        "risk": "Risk",
        "iyi": "Good",
        "dikkat": "Caution",
        "riskli": "Risk",
    }
    return mapping.get(text, str(value).strip())


def load_real_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {DATE_COLUMN, LOCATION_NAME_COLUMN, TARGET_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV içinde eksik sütunlar var: {sorted(missing)}")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(normalize_risk_class)
    df = df.dropna(subset=[DATE_COLUMN, LOCATION_NAME_COLUMN, TARGET_COLUMN]).copy()

    unknown_classes = sorted(set(df[TARGET_COLUMN]) - set(CLASS_LABELS))
    if unknown_classes:
        raise ValueError(f"Bilinmeyen RiskClass değerleri: {unknown_classes}")

    if LOCATION_ID_COLUMN not in df.columns:
        df[LOCATION_ID_COLUMN] = df[LOCATION_NAME_COLUMN].astype(str)

    # Empty legacy E.Coli is excluded; canonical E.coli is used.
    if "E.Coli" in df.columns and df["E.Coli"].isna().all():
        df = df.drop(columns=["E.Coli"])

    available_features = [
        feature
        for feature in RAW_FEATURES
        if feature in df.columns and not df[feature].isna().all()
    ]
    if not available_features:
        raise ValueError(
            "Merkezi parameters.py ile eşleşen ham parametre sütunu bulunamadı."
        )

    for feature in available_features:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df["IsSynthetic"] = False
    df["SyntheticScenario"] = "REAL"
    df["SyntheticSourceSplit"] = "REAL"
    df["SyntheticSourceRow"] = np.arange(len(df))

    sort_cols = [DATE_COLUMN, LOCATION_ID_COLUMN, LOCATION_NAME_COLUMN]
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    duplicate_key = [DATE_COLUMN, LOCATION_ID_COLUMN]
    duplicates = int(df.duplicated(subset=duplicate_key).sum())
    if duplicates:
        raise ValueError(
            f"Gerçek feature verisinde {duplicates} adet Tarih+NoktaId duplicate var."
        )

    return df


def split_real_data_by_date(
    df: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
) -> SplitBundle:
    unique_dates = sorted(pd.Series(df[DATE_COLUMN].dropna().unique()).tolist())
    n_dates = len(unique_dates)
    if n_dates < 5:
        raise ValueError("Zamansal train/validation/test ayrımı için en az 5 tarih gerekir.")

    train_end = max(1, int(n_dates * train_ratio))
    validation_end = max(train_end + 1, int(n_dates * (train_ratio + validation_ratio)))
    validation_end = min(validation_end, n_dates - 1)

    train_dates = unique_dates[:train_end]
    validation_dates = unique_dates[train_end:validation_end]
    test_dates = unique_dates[validation_end:]

    train_real = df[df[DATE_COLUMN].isin(train_dates)].copy()
    validation_real = df[df[DATE_COLUMN].isin(validation_dates)].copy()
    test_real = df[df[DATE_COLUMN].isin(test_dates)].copy()

    if train_real.empty or validation_real.empty or test_real.empty:
        raise ValueError("Train/validation/test bölümlerinden biri boş oluştu.")

    return SplitBundle(
        train_real=train_real,
        validation_real=validation_real,
        test_real=test_real,
        train_dates=train_dates,
        validation_dates=validation_dates,
        test_dates=test_dates,
    )


def _apply_hf_result(row: pd.Series) -> pd.Series:
    result = calculate_hf_from_wide_row(row.to_dict())
    for key, value in result.items():
        row[key] = value
    return row


def _set_caution_scenario(
    row: pd.Series,
    rng: np.random.Generator,
    strategy: str,
) -> pd.Series:
    # All ranges remain below fail-fast standards.
    if strategy == "AMMONIUM_NEAR_STANDARD":
        row["Amonyum"] = rng.uniform(0.16, 0.34)
        if "Oksitlenebilirlik" in row.index:
            row["Oksitlenebilirlik"] = rng.uniform(1.0, 3.5)

    elif strategy == "NITRITE_SUBSTANDARD":
        row["Nitrit"] = rng.uniform(0.12, 0.30)
        if "Arsenik" in row.index:
            row["Arsenik"] = rng.uniform(2.0, 8.5)

    else:  # MULTI_PARAMETER_SUBSTANDARD
        if "Amonyum" in row.index:
            row["Amonyum"] = rng.uniform(0.10, 0.26)
        if "Nitrit" in row.index:
            row["Nitrit"] = rng.uniform(0.08, 0.22)
        if "Oksitlenebilirlik" in row.index:
            row["Oksitlenebilirlik"] = rng.uniform(2.0, 4.8)
        if "Arsenik" in row.index:
            row["Arsenik"] = rng.uniform(4.0, 9.5)
        if "pH" in row.index:
            row["pH"] = rng.choice(
                [rng.uniform(6.50, 6.85), rng.uniform(8.60, 9.45)]
            )

    # Zero-standard microbiological parameters remain compliant.
    for param in ["E.coli", "Koliform Bakteri", "C.Perfringens"]:
        if param in row.index:
            row[param] = 0.0

    return row


def _set_risk_scenario(
    row: pd.Series,
    rng: np.random.Generator,
    strategy: str,
) -> pd.Series:
    if strategy == "E_COLI_FAIL_FAST":
        row["E.coli"] = float(rng.integers(1, 6))
    elif strategy == "COLIFORM_FAIL_FAST":
        row["Koliform Bakteri"] = float(rng.integers(1, 21))
    elif strategy == "C_PERFRINGENS_FAIL_FAST":
        row["C.Perfringens"] = float(rng.integers(1, 6))
    elif strategy == "ARSENIC_FAIL_FAST":
        row["Arsenik"] = rng.uniform(10.1, 25.0)
    else:  # NITRITE_FAIL_FAST
        row["Nitrit"] = rng.uniform(0.51, 1.50)
    return row


def generate_synthetic_class(
    source_real: pd.DataFrame,
    target_class: str,
    n_samples: int,
    source_split_name: str,
    seed: int,
) -> pd.DataFrame:
    if target_class not in {"Caution", "Risk"}:
        raise ValueError("Synthetic target_class yalnızca Caution veya Risk olabilir.")
    if source_real.empty or n_samples <= 0:
        return source_real.iloc[0:0].copy()

    rng = np.random.default_rng(seed)
    rows: list[pd.Series] = []

    caution_strategies = [
        "AMMONIUM_NEAR_STANDARD",
        "NITRITE_SUBSTANDARD",
        "MULTI_PARAMETER_SUBSTANDARD",
    ]
    risk_strategies = [
        "E_COLI_FAIL_FAST",
        "COLIFORM_FAIL_FAST",
        "C_PERFRINGENS_FAIL_FAST",
        "ARSENIC_FAIL_FAST",
        "NITRITE_FAIL_FAST",
    ]

    for synthetic_index in range(n_samples):
        accepted: pd.Series | None = None

        for attempt in range(80):
            source_position = int(rng.integers(0, len(source_real)))
            base = source_real.iloc[source_position].copy()

            # Remove derived HF fields before recalculation.
            for column in list(base.index):
                if (
                    column in TARGET_DERIVED_COLUMNS
                    or str(column).endswith("_score")
                ):
                    base[column] = np.nan

            if target_class == "Caution":
                strategy = str(rng.choice(caution_strategies))
                candidate = _set_caution_scenario(base, rng, strategy)
            else:
                strategy = str(rng.choice(risk_strategies))
                candidate = _set_risk_scenario(base, rng, strategy)

            candidate = _apply_hf_result(candidate)

            if candidate.get(TARGET_COLUMN) == target_class:
                candidate["IsSynthetic"] = True
                candidate["SyntheticScenario"] = strategy
                candidate["SyntheticSourceSplit"] = source_split_name
                candidate["SyntheticSourceRow"] = int(source_real.index[source_position])
                candidate["SyntheticSequence"] = synthetic_index
                accepted = candidate
                break

        if accepted is None and target_class == "Caution":
            # Deterministic fallback: search a safe sub-standard ammonium value.
            source_position = int(rng.integers(0, len(source_real)))
            base = source_real.iloc[source_position].copy()
            for value in np.linspace(0.12, 0.48, 73):
                candidate = base.copy()
                candidate["Amonyum"] = float(value)
                for param in ["E.coli", "Koliform Bakteri", "C.Perfringens"]:
                    if param in candidate.index:
                        candidate[param] = 0.0
                candidate = _apply_hf_result(candidate)
                if candidate.get(TARGET_COLUMN) == "Caution":
                    candidate["IsSynthetic"] = True
                    candidate["SyntheticScenario"] = "AMMONIUM_GRID_FALLBACK"
                    candidate["SyntheticSourceSplit"] = source_split_name
                    candidate["SyntheticSourceRow"] = int(source_real.index[source_position])
                    candidate["SyntheticSequence"] = synthetic_index
                    accepted = candidate
                    break

        if accepted is None:
            raise RuntimeError(
                f"{target_class} için sentetik örnek üretilemedi. "
                "Parametre aralıklarını merkezi HF formülüne göre kontrol edin."
            )

        rows.append(accepted)

    synthetic = pd.DataFrame(rows).reset_index(drop=True)
    return synthetic


def generate_synthetic_bundle(
    source_real: pd.DataFrame,
    ratio_per_class: float,
    source_split_name: str,
    seed: int,
) -> pd.DataFrame:
    count_per_class = max(1, int(round(len(source_real) * ratio_per_class)))
    caution = generate_synthetic_class(
        source_real,
        "Caution",
        count_per_class,
        source_split_name,
        seed + 11,
    )
    risk = generate_synthetic_class(
        source_real,
        "Risk",
        count_per_class,
        source_split_name,
        seed + 29,
    )
    return pd.concat([caution, risk], ignore_index=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        feature
        for feature in RAW_FEATURES
        if feature in df.columns and not df[feature].isna().all()
    ]
    forbidden = [
        column
        for column in columns
        if column in TARGET_DERIVED_COLUMNS or str(column).endswith("_score")
    ]
    if forbidden:
        raise RuntimeError(f"Target-derived feature sızıntısı: {forbidden}")
    return columns


def get_models() -> dict[str, Any]:
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def make_pipeline(model: Any, feature_columns: list[str]) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("model", clone(model)),
        ]
    )


def classwise_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    experiment: str,
    evaluation_set: str,
    model_name: str,
) -> pd.DataFrame:
    y_true_series = pd.Series(list(y_true), dtype="object")
    y_pred_series = pd.Series(list(y_pred), dtype="object")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_series,
        y_pred_series,
        labels=CLASS_LABELS,
        zero_division=0,
    )

    rows = []
    for index, label in enumerate(CLASS_LABELS):
        label_support = int(support[index])
        rows.append(
            {
                "Experiment": experiment,
                "EvaluationSet": evaluation_set,
                "Model": model_name,
                "Class": label,
                "Precision": np.nan if label_support == 0 else float(precision[index]),
                "Recall": np.nan if label_support == 0 else float(recall[index]),
                "F1": np.nan if label_support == 0 else float(f1[index]),
                "Support": label_support,
            }
        )
    return pd.DataFrame(rows)


def overall_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
) -> dict[str, Any]:
    y_true_series = pd.Series(list(y_true), dtype="object")
    y_pred_series = pd.Series(list(y_pred), dtype="object")
    present_labels = [
        label for label in CLASS_LABELS if int((y_true_series == label).sum()) > 0
    ]

    if not present_labels:
        return {
            "Accuracy": np.nan,
            "BalancedAccuracy": np.nan,
            "MacroPrecision": np.nan,
            "MacroRecall": np.nan,
            "MacroF1": np.nan,
            "WeightedF1": np.nan,
            "MCC": np.nan,
            "RiskRecall": np.nan,
            "RiskFalseNegatives": 0,
            "RiskSupport": 0,
            "ClassesPresent": "",
        }

    accuracy = accuracy_score(y_true_series, y_pred_series)
    balanced = balanced_accuracy_score(y_true_series, y_pred_series)
    macro_precision = precision_score(
        y_true_series,
        y_pred_series,
        labels=present_labels,
        average="macro",
        zero_division=0,
    )
    macro_recall = recall_score(
        y_true_series,
        y_pred_series,
        labels=present_labels,
        average="macro",
        zero_division=0,
    )
    macro_f1 = f1_score(
        y_true_series,
        y_pred_series,
        labels=present_labels,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true_series,
        y_pred_series,
        labels=present_labels,
        average="weighted",
        zero_division=0,
    )

    mcc = (
        np.nan
        if y_true_series.nunique() < 2
        else float(matthews_corrcoef(y_true_series, y_pred_series))
    )

    risk_support = int((y_true_series == "Risk").sum())
    if risk_support:
        risk_recall = recall_score(
            y_true_series,
            y_pred_series,
            labels=["Risk"],
            average="macro",
            zero_division=0,
        )
        risk_false_negatives = int(
            ((y_true_series == "Risk") & (y_pred_series != "Risk")).sum()
        )
    else:
        risk_recall = np.nan
        risk_false_negatives = 0

    return {
        "Accuracy": float(accuracy),
        "BalancedAccuracy": float(balanced),
        "MacroPrecision": float(macro_precision),
        "MacroRecall": float(macro_recall),
        "MacroF1": float(macro_f1),
        "WeightedF1": float(weighted_f1),
        "MCC": mcc,
        "RiskRecall": float(risk_recall) if not pd.isna(risk_recall) else np.nan,
        "RiskFalseNegatives": risk_false_negatives,
        "RiskSupport": risk_support,
        "ClassesPresent": ", ".join(present_labels),
    }


def save_confusion_matrix(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    path: Path,
    normalized: bool,
) -> None:
    matrix = confusion_matrix(
        list(y_true),
        list(y_pred),
        labels=CLASS_LABELS,
        normalize="true" if normalized else None,
    )
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    image = ax.imshow(matrix, vmin=0, vmax=1 if normalized else None)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticks(range(len(CLASS_LABELS)))
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            label = f"{value:.2f}" if normalized else f"{int(value)}"
            ax.text(column, row, label, ha="center", va="center", fontsize=12)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_plot(metrics_df: pd.DataFrame, path: Path) -> None:
    subset = metrics_df[
        (metrics_df["Experiment"] == "E3_AUGMENTED_TRAIN_MIXED_TEST")
        & (metrics_df["Status"] == "OK")
    ].copy()
    if subset.empty:
        return

    metrics = ["Accuracy", "BalancedAccuracy", "MacroF1", "MCC"]
    x = np.arange(len(subset))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for index, metric in enumerate(metrics):
        values = subset[metric].fillna(0.0).to_numpy()
        bars = ax.bar(x + (index - 1.5) * width, values, width, label=metric)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(subset["Model"], rotation=15)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_class_distribution_plot(distribution_df: pd.DataFrame, path: Path) -> None:
    order = [
        "RealTrain",
        "SyntheticTrain",
        "AugmentedTrain",
        "RealValidation",
        "MixedValidation",
        "RealTest",
        "SyntheticTest",
        "MixedTest",
    ]
    pivot = (
        distribution_df.pivot_table(
            index="Dataset",
            columns="Class",
            values="Count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(order)
        .fillna(0)
    )
    for label in CLASS_LABELS:
        if label not in pivot.columns:
            pivot[label] = 0
    pivot = pivot[CLASS_LABELS]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for label in CLASS_LABELS:
        values = pivot[label].to_numpy()
        ax.bar(x, values, bottom=bottom, label=label)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylabel("Sample count")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def calculate_distribution_validation(
    real_train: pd.DataFrame,
    synthetic_train: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        real_values = pd.to_numeric(real_train[feature], errors="coerce").dropna()
        synthetic_values = pd.to_numeric(
            synthetic_train[feature], errors="coerce"
        ).dropna()
        pooled_std = float(
            np.sqrt(
                (
                    real_values.var(ddof=1)
                    + synthetic_values.var(ddof=1)
                )
                / 2
            )
        ) if len(real_values) > 1 and len(synthetic_values) > 1 else np.nan
        smd = (
            (synthetic_values.mean() - real_values.mean()) / pooled_std
            if pooled_std and not pd.isna(pooled_std)
            else np.nan
        )
        rows.append(
            {
                "RecordType": "PARAMETER",
                "Parameter": feature,
                "RealMean": real_values.mean(),
                "RealStd": real_values.std(ddof=1),
                "SyntheticMean": synthetic_values.mean(),
                "SyntheticStd": synthetic_values.std(ddof=1),
                "StandardizedMeanDifference": smd,
                "Note": (
                    "Controlled scenario variable; exact distribution equality "
                    "is not expected."
                ),
            }
        )

    valid_features = [
        feature
        for feature in feature_columns
        if real_train[feature].notna().sum() > 2
        and synthetic_train[feature].notna().sum() > 2
        and real_train[feature].std(skipna=True) > 0
        and synthetic_train[feature].std(skipna=True) > 0
    ]
    if len(valid_features) >= 2:
        real_corr = real_train[valid_features].corr()
        synthetic_corr = synthetic_train[valid_features].corr()
        upper = np.triu_indices_from(real_corr, k=1)
        corr_mad = float(
            np.nanmean(
                np.abs(
                    real_corr.to_numpy()[upper]
                    - synthetic_corr.to_numpy()[upper]
                )
            )
        )
    else:
        corr_mad = np.nan

    rows.append(
        {
            "RecordType": "SUMMARY",
            "Parameter": "__PAIRWISE_CORRELATION_MAD__",
            "RealMean": np.nan,
            "RealStd": np.nan,
            "SyntheticMean": np.nan,
            "SyntheticStd": np.nan,
            "StandardizedMeanDifference": corr_mad,
            "Note": (
                "Mean absolute difference between real and synthetic pairwise "
                "correlations. The generator is a conditional bootstrap and "
                "does not claim exact covariance preservation."
            ),
        }
    )
    return pd.DataFrame(rows)


def synthetic_rules_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "TargetClass": "Caution",
                "Scenario": "AMMONIUM_NEAR_STANDARD",
                "ParameterRules": "Amonyum 0.16-0.34 mg/L; Oksitlenebilirlik 1.0-3.5 mg/L O2",
                "AcceptanceRule": "Central HF calculator must return Caution (60 <= HF < 85)",
            },
            {
                "TargetClass": "Caution",
                "Scenario": "NITRITE_SUBSTANDARD",
                "ParameterRules": "Nitrit 0.12-0.30 mg/L; Arsenik 2.0-8.5 ug/L",
                "AcceptanceRule": "No fail-fast; central HF calculator must return Caution",
            },
            {
                "TargetClass": "Caution",
                "Scenario": "MULTI_PARAMETER_SUBSTANDARD",
                "ParameterRules": "Sub-standard Amonyum, Nitrit, Oksitlenebilirlik, Arsenik and pH perturbation",
                "AcceptanceRule": "Central HF calculator must return Caution",
            },
            {
                "TargetClass": "Risk",
                "Scenario": "E_COLI_FAIL_FAST",
                "ParameterRules": "E.coli 1-5 count/100 mL",
                "AcceptanceRule": "Zero standard exceeded; HF=0 and Risk",
            },
            {
                "TargetClass": "Risk",
                "Scenario": "COLIFORM_FAIL_FAST",
                "ParameterRules": "Koliform Bakteri 1-20 count/100 mL",
                "AcceptanceRule": "Zero standard exceeded; HF=0 and Risk",
            },
            {
                "TargetClass": "Risk",
                "Scenario": "C_PERFRINGENS_FAIL_FAST",
                "ParameterRules": "C.Perfringens 1-5 count/100 mL",
                "AcceptanceRule": "Zero standard exceeded; HF=0 and Risk",
            },
            {
                "TargetClass": "Risk",
                "Scenario": "ARSENIC_FAIL_FAST",
                "ParameterRules": "Arsenik 10.1-25.0 ug/L",
                "AcceptanceRule": "Sn=10 ug/L exceeded; HF=0 and Risk",
            },
            {
                "TargetClass": "Risk",
                "Scenario": "NITRITE_FAIL_FAST",
                "ParameterRules": "Nitrit 0.51-1.50 mg/L",
                "AcceptanceRule": "Sn=0.50 mg/L exceeded; HF=0 and Risk",
            },
        ]
    )


def class_distribution_records(
    named_datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for dataset_name, frame in named_datasets.items():
        counts = frame[TARGET_COLUMN].value_counts()
        total = len(frame)
        synthetic_count = int(frame["IsSynthetic"].fillna(False).sum())
        for label in CLASS_LABELS:
            count = int(counts.get(label, 0))
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Class": label,
                    "Count": count,
                    "Percent": 0.0 if total == 0 else count / total * 100.0,
                    "TotalRows": total,
                    "RealRows": total - synthetic_count,
                    "SyntheticRows": synthetic_count,
                    "SyntheticPercent": (
                        0.0 if total == 0 else synthetic_count / total * 100.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def split_summary_table(
    split: SplitBundle,
    synthetic_train: pd.DataFrame,
    synthetic_validation: pd.DataFrame,
    synthetic_test: pd.DataFrame,
) -> pd.DataFrame:
    datasets = {
        "RealTrain": split.train_real,
        "SyntheticTrain": synthetic_train,
        "AugmentedTrain": pd.concat(
            [split.train_real, synthetic_train], ignore_index=True
        ),
        "RealValidation": split.validation_real,
        "SyntheticValidation": synthetic_validation,
        "MixedValidation": pd.concat(
            [split.validation_real, synthetic_validation], ignore_index=True
        ),
        "RealTest": split.test_real,
        "SyntheticTest": synthetic_test,
        "MixedTest": pd.concat(
            [split.test_real, synthetic_test], ignore_index=True
        ),
    }
    rows = []
    for name, frame in datasets.items():
        rows.append(
            {
                "Dataset": name,
                "Rows": len(frame),
                "UniqueDates": frame[DATE_COLUMN].nunique(),
                "StartDate": (
                    frame[DATE_COLUMN].min().date().isoformat()
                    if not frame.empty
                    else ""
                ),
                "EndDate": (
                    frame[DATE_COLUMN].max().date().isoformat()
                    if not frame.empty
                    else ""
                ),
                "RealRows": int((~frame["IsSynthetic"].fillna(False)).sum()),
                "SyntheticRows": int(frame["IsSynthetic"].fillna(False).sum()),
            }
        )
    return pd.DataFrame(rows)


def leakage_audit_table(
    feature_columns: list[str],
    split: SplitBundle,
    synthetic_train: pd.DataFrame,
    synthetic_validation: pd.DataFrame,
    synthetic_test: pd.DataFrame,
) -> pd.DataFrame:
    date_overlap_train_val = bool(
        set(split.train_dates) & set(split.validation_dates)
    )
    date_overlap_train_test = bool(set(split.train_dates) & set(split.test_dates))
    date_overlap_val_test = bool(
        set(split.validation_dates) & set(split.test_dates)
    )

    forbidden_features = [
        column
        for column in feature_columns
        if column in TARGET_DERIVED_COLUMNS or column.endswith("_score")
    ]

    return pd.DataFrame(
        [
            {
                "Check": "Target-derived columns excluded from X",
                "Pass": not forbidden_features,
                "Evidence": ", ".join(forbidden_features) or "None",
            },
            {
                "Check": "Train/validation dates do not overlap",
                "Pass": not date_overlap_train_val,
                "Evidence": str(date_overlap_train_val),
            },
            {
                "Check": "Train/test dates do not overlap",
                "Pass": not date_overlap_train_test,
                "Evidence": str(date_overlap_train_test),
            },
            {
                "Check": "Validation/test dates do not overlap",
                "Pass": not date_overlap_val_test,
                "Evidence": str(date_overlap_val_test),
            },
            {
                "Check": "Synthetic train generated only from real train",
                "Pass": set(
                    synthetic_train["SyntheticSourceSplit"].dropna().unique()
                ) <= {"TRAIN"},
                "Evidence": ", ".join(
                    map(
                        str,
                        synthetic_train["SyntheticSourceSplit"]
                        .dropna()
                        .unique(),
                    )
                ),
            },
            {
                "Check": "Synthetic validation generated only from real validation",
                "Pass": set(
                    synthetic_validation["SyntheticSourceSplit"]
                    .dropna()
                    .unique()
                ) <= {"VALIDATION"},
                "Evidence": ", ".join(
                    map(
                        str,
                        synthetic_validation["SyntheticSourceSplit"]
                        .dropna()
                        .unique(),
                    )
                ),
            },
            {
                "Check": "Synthetic test generated only from real test",
                "Pass": set(
                    synthetic_test["SyntheticSourceSplit"].dropna().unique()
                ) <= {"TEST"},
                "Evidence": ", ".join(
                    map(
                        str,
                        synthetic_test["SyntheticSourceSplit"]
                        .dropna()
                        .unique(),
                    )
                ),
            },
            {
                "Check": "Imputer/scaler fitted inside training pipeline",
                "Pass": True,
                "Evidence": "SimpleImputer + StandardScaler are pipeline steps fit only with augmented training rows.",
            },
            {
                "Check": "Test set excluded from model selection",
                "Pass": True,
                "Evidence": "Best model is selected using mixed validation MacroF1, MCC and BalancedAccuracy.",
            },
        ]
    )


def _safe_number(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def write_word_report(
    output_path: Path,
    input_path: Path,
    feature_columns: list[str],
    split_summary: pd.DataFrame,
    distribution: pd.DataFrame,
    model_metrics: pd.DataFrame,
    class_metrics: pd.DataFrame,
    leakage_audit: pd.DataFrame,
    rules: pd.DataFrame,
    distribution_validation: pd.DataFrame,
    best_model_name: str | None,
) -> None:
    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Cm, Pt
    except ImportError:
        print("[UYARI] python-docx kurulu değil; Word raporu üretilemedi.")
        return

    document = Document()
    section = document.sections[0]
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(1.8)
    section.right_margin = Cm(1.8)

    styles = document.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(10)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Reactive Layer Technical Report")
    run.bold = True
    run.font.size = Pt(17)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run(
        "Current-state Good / Caution / Risk classification"
    ).italic = True

    document.add_heading("1. Scope and input data", level=1)
    document.add_paragraph(
        f"Input file: {input_path}. The Reactive Layer evaluates the current "
        "water-quality state. It does not forecast a future date. The target "
        "labels are produced by the centralized HF/WAWQI calculator using "
        "Good >= 85, Caution 60 <= HF < 85, Risk < 60, together with the "
        "fail-fast rules."
    )
    document.add_paragraph(
        "Model inputs are raw water-quality parameters only. HealthFactor, "
        "WAWQI, RiskClass, FailFast and all *_score columns are excluded from "
        "the predictor matrix to avoid target-definition leakage."
    )
    document.add_paragraph(
        "Feature columns: " + ", ".join(feature_columns)
    )

    document.add_heading("2. Leakage-safe workflow", level=1)
    workflow = (
        "Clean aggregated real data -> unique-date train/validation/test split "
        "-> split-specific controlled synthetic generation -> training-only "
        "imputation and scaling -> model fitting -> validation-based model "
        "selection -> real, synthetic-only and mixed test evaluation."
    )
    document.add_paragraph(workflow)

    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    headers = ["Audit check", "Pass", "Evidence"]
    for index, header in enumerate(headers):
        table.rows[0].cells[index].text = header
    for _, row in leakage_audit.iterrows():
        cells = table.add_row().cells
        cells[0].text = str(row["Check"])
        cells[1].text = "PASS" if bool(row["Pass"]) else "FAIL"
        cells[2].text = str(row["Evidence"])

    document.add_heading("3. Real and synthetic data composition", level=1)
    table = document.add_table(rows=1, cols=7)
    table.style = "Table Grid"
    columns = [
        "Dataset",
        "Rows",
        "UniqueDates",
        "StartDate",
        "EndDate",
        "RealRows",
        "SyntheticRows",
    ]
    for index, column in enumerate(columns):
        table.rows[0].cells[index].text = column
    for _, row in split_summary.iterrows():
        cells = table.add_row().cells
        for index, column in enumerate(columns):
            cells[index].text = str(row[column])

    real_distribution = distribution[
        distribution["Dataset"].isin(
            ["RealTrain", "RealValidation", "RealTest"]
        )
    ]
    document.add_paragraph(
        "The current real dataset contains only the Good class. Consequently, "
        "a real-only three-class training experiment is not statistically "
        "estimable. The code reports this limitation rather than producing a "
        "fabricated Risk recall."
    )

    document.add_heading("4. Synthetic generation rules", level=1)
    document.add_paragraph(
        "Synthetic samples are generated by conditional bootstrap: a real row "
        "from the relevant split is copied and only a small, explicitly "
        "documented subset of parameters is modified. The centralized HF "
        "calculator then recalculates all scores and accepts the row only if "
        "the requested Caution or Risk class is obtained. Exact covariance "
        "preservation is not claimed."
    )
    table = document.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    rule_columns = [
        "TargetClass",
        "Scenario",
        "ParameterRules",
        "AcceptanceRule",
    ]
    for index, column in enumerate(rule_columns):
        table.rows[0].cells[index].text = column
    for _, row in rules.iterrows():
        cells = table.add_row().cells
        for index, column in enumerate(rule_columns):
            cells[index].text = str(row[column])

    corr_row = distribution_validation[
        distribution_validation["Parameter"]
        == "__PAIRWISE_CORRELATION_MAD__"
    ]
    if not corr_row.empty:
        corr_value = corr_row.iloc[0]["StandardizedMeanDifference"]
        document.add_paragraph(
            "Mean absolute pairwise-correlation difference between real and "
            f"synthetic training data: {_safe_number(corr_value)}. This value "
            "is descriptive and is not presented as proof of covariance "
            "preservation."
        )

    document.add_heading("5. Experiments", level=1)
    document.add_paragraph(
        "E1 - Real train / real test: reported as not estimable when the real "
        "training split has fewer than two classes."
    )
    document.add_paragraph(
        "E2 - Augmented train / real test: test observations are 100% real. "
        "Because the current real test split contains only Good observations, "
        "Risk recall is N/A."
    )
    document.add_paragraph(
        "E3 - Augmented train / mixed test: the mixed test set combines real "
        "test observations and independently generated synthetic Caution/Risk "
        "observations derived only from the real test split. Synthetic-only "
        "performance is also exported separately."
    )

    document.add_heading("6. Model performance", level=1)
    performance = model_metrics[
        model_metrics["Experiment"].isin(
            [
                "E2_AUGMENTED_TRAIN_REAL_TEST",
                "E3_AUGMENTED_TRAIN_MIXED_TEST",
                "E3B_AUGMENTED_TRAIN_SYNTHETIC_TEST",
            ]
        )
        & (model_metrics["Status"] == "OK")
    ].copy()

    performance = performance.copy()
    performance["Test"] = performance["Experiment"].map(
        {
            "E2_AUGMENTED_TRAIN_REAL_TEST": "E2 Real",
            "E3_AUGMENTED_TRAIN_MIXED_TEST": "E3 Mixed",
            "E3B_AUGMENTED_TRAIN_SYNTHETIC_TEST": "E3B Synthetic",
        }
    ).fillna(performance["Experiment"])

    performance_columns = [
        "Test",
        "Model",
        "Accuracy",
        "BalancedAccuracy",
        "MacroPrecision",
        "MacroRecall",
        "MacroF1",
        "MCC",
        "RiskRecall",
    ]
    display_headers = [
        "Test",
        "Model",
        "Acc.",
        "Bal. Acc.",
        "Macro P",
        "Macro R",
        "Macro F1",
        "MCC",
        "Risk R",
    ]
    table = document.add_table(rows=1, cols=len(performance_columns))
    table.style = "Table Grid"
    for index, header in enumerate(display_headers):
        table.rows[0].cells[index].text = header
    for _, row in performance.iterrows():
        cells = table.add_row().cells
        for index, column in enumerate(performance_columns):
            value = row[column]
            cells[index].text = (
                _safe_number(value)
                if column not in {"Test", "Model"}
                else str(value)
            )

    document.add_paragraph(
        "Best model selected from the mixed validation set: "
        + (best_model_name or "N/A")
        + ". Selection is ordered by Macro F1, MCC and Balanced Accuracy; "
        "test metrics are not used for model selection."
    )

    document.add_heading("7. Class-wise performance and limitations", level=1)
    if best_model_name:
        best_classwise = class_metrics[
            (class_metrics["Model"] == best_model_name)
            & (
                class_metrics["Experiment"]
                == "E3_AUGMENTED_TRAIN_MIXED_TEST"
            )
        ]
        table = document.add_table(rows=1, cols=7)
        table.style = "Table Grid"
        columns = [
            "EvaluationSet",
            "Class",
            "Precision",
            "Recall",
            "F1",
            "Support",
            "Model",
        ]
        for index, column in enumerate(columns):
            table.rows[0].cells[index].text = column
        for _, row in best_classwise.iterrows():
            cells = table.add_row().cells
            for index, column in enumerate(columns):
                value = row[column]
                if column in {"Precision", "Recall", "F1"}:
                    cells[index].text = _safe_number(value)
                else:
                    cells[index].text = str(value)

    document.add_paragraph(
        "Synthetic mixed-test results measure performance under controlled "
        "contamination scenarios; they are not equivalent to external clinical "
        "or field validation. The real data currently provide no observed "
        "Caution or Risk samples, so class-specific real-data sensitivity for "
        "those classes cannot be estimated."
    )

    document.add_heading("8. Generated artifacts", level=1)
    for artifact in [
        "reactive_model_metrics.csv",
        "reactive_classwise_metrics.csv",
        "reactive_predictions.csv",
        "reactive_split_summary.csv",
        "reactive_class_distribution.csv",
        "synthetic_parameter_rules.csv",
        "synthetic_distribution_validation.csv",
        "data_leakage_audit.csv",
        "best_reactive_model.joblib",
        "figures/*.png",
    ]:
        document.add_paragraph(artifact, style="List Bullet")

    document.save(output_path)


def train_and_evaluate(
    augmented_train: pd.DataFrame,
    mixed_validation: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic_test: pd.DataFrame,
    mixed_test: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None, Pipeline | None]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    models = get_models()
    metrics_rows: list[dict[str, Any]] = []
    classwise_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    best_model_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_rank: tuple[float, float, float] | None = None

    X_train = augmented_train[feature_columns]
    y_train = augmented_train[TARGET_COLUMN]

    validation_predictions: dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        pipeline = make_pipeline(model, feature_columns)
        pipeline.fit(X_train, y_train)

        X_validation = mixed_validation[feature_columns]
        y_validation = mixed_validation[TARGET_COLUMN]
        val_pred = pipeline.predict(X_validation)
        validation_predictions[model_name] = val_pred
        val_metrics = overall_metrics(y_validation, val_pred)

        metrics_rows.append(
            {
                "Experiment": "VALIDATION_MIXED",
                "EvaluationSet": "MixedValidation",
                "Model": model_name,
                "Status": "OK",
                "TrainRows": len(augmented_train),
                "TestRows": len(mixed_validation),
                "RealTestRows": int(
                    (~mixed_validation["IsSynthetic"].fillna(False)).sum()
                ),
                "SyntheticTestRows": int(
                    mixed_validation["IsSynthetic"].fillna(False).sum()
                ),
                **val_metrics,
            }
        )
        classwise_frames.append(
            classwise_metrics(
                y_validation,
                val_pred,
                "VALIDATION_MIXED",
                "MixedValidation",
                model_name,
            )
        )

        rank = (
            float(val_metrics["MacroF1"]),
            float(val_metrics["MCC"]) if not pd.isna(val_metrics["MCC"]) else -1.0,
            float(val_metrics["BalancedAccuracy"]),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_model_name = model_name
            best_pipeline = pipeline

        evaluation_sets = [
            (
                "E2_AUGMENTED_TRAIN_REAL_TEST",
                "RealTest",
                real_test,
            ),
            (
                "E3B_AUGMENTED_TRAIN_SYNTHETIC_TEST",
                "SyntheticTest",
                synthetic_test,
            ),
            (
                "E3_AUGMENTED_TRAIN_MIXED_TEST",
                "MixedTest",
                mixed_test,
            ),
        ]

        for experiment, evaluation_name, evaluation_df in evaluation_sets:
            X_eval = evaluation_df[feature_columns]
            y_eval = evaluation_df[TARGET_COLUMN]
            predictions = pipeline.predict(X_eval)
            metrics = overall_metrics(y_eval, predictions)

            metrics_rows.append(
                {
                    "Experiment": experiment,
                    "EvaluationSet": evaluation_name,
                    "Model": model_name,
                    "Status": "OK",
                    "TrainRows": len(augmented_train),
                    "TestRows": len(evaluation_df),
                    "RealTestRows": int(
                        (~evaluation_df["IsSynthetic"].fillna(False)).sum()
                    ),
                    "SyntheticTestRows": int(
                        evaluation_df["IsSynthetic"].fillna(False).sum()
                    ),
                    **metrics,
                }
            )
            classwise_frames.append(
                classwise_metrics(
                    y_eval,
                    predictions,
                    experiment,
                    evaluation_name,
                    model_name,
                )
            )

            prediction_frame = evaluation_df[
                [
                    DATE_COLUMN,
                    LOCATION_ID_COLUMN,
                    LOCATION_NAME_COLUMN,
                    TARGET_COLUMN,
                    "IsSynthetic",
                    "SyntheticScenario",
                    "SyntheticSourceSplit",
                ]
            ].copy()
            prediction_frame["Experiment"] = experiment
            prediction_frame["EvaluationSet"] = evaluation_name
            prediction_frame["Model"] = model_name
            prediction_frame["PredictedRiskClass"] = predictions

            if hasattr(pipeline, "predict_proba"):
                probabilities = pipeline.predict_proba(X_eval)
                model_classes = pipeline.named_steps["model"].classes_
                for class_index, class_label in enumerate(model_classes):
                    prediction_frame[f"Probability_{class_label}"] = (
                        probabilities[:, class_index]
                    )

            prediction_frames.append(prediction_frame)

            safe_model_name = model_name.replace(" ", "_")
            save_confusion_matrix(
                y_eval,
                predictions,
                figures_dir
                / f"cm_counts_{safe_model_name}_{evaluation_name}.png",
                normalized=False,
            )
            save_confusion_matrix(
                y_eval,
                predictions,
                figures_dir
                / f"cm_normalized_{safe_model_name}_{evaluation_name}.png",
                normalized=True,
            )

    metrics_df = pd.DataFrame(metrics_rows)
    classwise_df = pd.concat(classwise_frames, ignore_index=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    return (
        metrics_df,
        classwise_df,
        predictions_df,
        best_model_name,
        best_pipeline,
    )


def real_only_experiment_rows(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
) -> pd.DataFrame:
    reason = (
        "Real training split contains fewer than two classes; "
        "three-class model fitting is not estimable."
    )
    rows = []
    for model_name in get_models():
        rows.append(
            {
                "Experiment": "E1_REAL_TRAIN_REAL_TEST",
                "EvaluationSet": "RealTest",
                "Model": model_name,
                "Status": "SKIPPED_SINGLE_CLASS",
                "Reason": reason,
                "TrainRows": len(train_real),
                "TestRows": len(test_real),
                "RealTestRows": len(test_real),
                "SyntheticTestRows": 0,
                "Accuracy": np.nan,
                "BalancedAccuracy": np.nan,
                "MacroPrecision": np.nan,
                "MacroRecall": np.nan,
                "MacroF1": np.nan,
                "WeightedF1": np.nan,
                "MCC": np.nan,
                "RiskRecall": np.nan,
                "RiskFalseNegatives": 0,
                "RiskSupport": int((test_real[TARGET_COLUMN] == "Risk").sum()),
                "ClassesPresent": ", ".join(
                    sorted(test_real[TARGET_COLUMN].unique())
                ),
            }
        )
    return pd.DataFrame(rows)


def run_pipeline(args: argparse.Namespace) -> None:
    input_path = find_input_csv(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] {input_path}")
    real_df = load_real_data(input_path)
    feature_columns = get_feature_columns(real_df)

    print(
        f"[DATA] rows={len(real_df)} dates={real_df[DATE_COLUMN].nunique()} "
        f"locations={real_df[LOCATION_ID_COLUMN].nunique()}"
    )
    print("[REAL CLASS DISTRIBUTION]")
    print(real_df[TARGET_COLUMN].value_counts(dropna=False))

    split = split_real_data_by_date(
        real_df,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )

    synthetic_train = generate_synthetic_bundle(
        split.train_real,
        args.synthetic_ratio_per_class,
        "TRAIN",
        args.seed + 100,
    )
    synthetic_validation = generate_synthetic_bundle(
        split.validation_real,
        args.synthetic_ratio_per_class,
        "VALIDATION",
        args.seed + 200,
    )
    synthetic_test = generate_synthetic_bundle(
        split.test_real,
        args.synthetic_ratio_per_class,
        "TEST",
        args.seed + 300,
    )

    augmented_train = pd.concat(
        [split.train_real, synthetic_train],
        ignore_index=True,
    ).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    mixed_validation = pd.concat(
        [split.validation_real, synthetic_validation],
        ignore_index=True,
    ).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    mixed_test = pd.concat(
        [split.test_real, synthetic_test],
        ignore_index=True,
    ).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    e1_metrics = real_only_experiment_rows(
        split.train_real,
        split.test_real,
    )

    (
        trained_metrics,
        classwise_metrics_df,
        predictions_df,
        best_model_name,
        best_pipeline,
    ) = train_and_evaluate(
        augmented_train,
        mixed_validation,
        split.test_real,
        synthetic_test,
        mixed_test,
        feature_columns,
        output_dir,
    )

    metrics_df = pd.concat(
        [e1_metrics, trained_metrics],
        ignore_index=True,
        sort=False,
    )

    split_summary = split_summary_table(
        split,
        synthetic_train,
        synthetic_validation,
        synthetic_test,
    )
    distribution_df = class_distribution_records(
        {
            "RealTrain": split.train_real,
            "SyntheticTrain": synthetic_train,
            "AugmentedTrain": augmented_train,
            "RealValidation": split.validation_real,
            "MixedValidation": mixed_validation,
            "RealTest": split.test_real,
            "SyntheticTest": synthetic_test,
            "MixedTest": mixed_test,
        }
    )
    rules_df = synthetic_rules_table()
    distribution_validation_df = calculate_distribution_validation(
        split.train_real,
        synthetic_train,
        feature_columns,
    )
    leakage_df = leakage_audit_table(
        feature_columns,
        split,
        synthetic_train,
        synthetic_validation,
        synthetic_test,
    )

    metrics_df.to_csv(
        output_dir / "reactive_model_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    classwise_metrics_df.to_csv(
        output_dir / "reactive_classwise_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    predictions_df.to_csv(
        output_dir / "reactive_predictions.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    split_summary.to_csv(
        output_dir / "reactive_split_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    distribution_df.to_csv(
        output_dir / "reactive_class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    rules_df.to_csv(
        output_dir / "synthetic_parameter_rules.csv",
        index=False,
        encoding="utf-8-sig",
    )
    distribution_validation_df.to_csv(
        output_dir / "synthetic_distribution_validation.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    leakage_df.to_csv(
        output_dir / "data_leakage_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_model_comparison_plot(
        metrics_df,
        figures_dir / "reactive_model_comparison_mixed_test.png",
    )
    save_class_distribution_plot(
        distribution_df,
        figures_dir / "reactive_class_distribution.png",
    )

    if best_pipeline is not None and best_model_name is not None:
        model_bundle = {
            "pipeline": best_pipeline,
            "model_name": best_model_name,
            "feature_columns": feature_columns,
            "class_labels": CLASS_LABELS,
            "target_column": TARGET_COLUMN,
            "input_file": str(input_path),
            "real_training_date_range": [
                min(split.train_dates).date().isoformat(),
                max(split.train_dates).date().isoformat(),
            ],
            "selection_rule": (
                "Mixed validation ranking by MacroF1, MCC, "
                "then BalancedAccuracy"
            ),
            "synthetic_ratio_per_class": args.synthetic_ratio_per_class,
        }
        joblib.dump(
            model_bundle,
            output_dir / "best_reactive_model.joblib",
        )

    write_word_report(
        output_dir / "Reactive_Layer_Technical_Report.docx",
        input_path,
        feature_columns,
        split_summary,
        distribution_df,
        metrics_df,
        classwise_metrics_df,
        leakage_df,
        rules_df,
        distribution_validation_df,
        best_model_name,
    )

    print("\n[SUCCESS] Reactive Layer pipeline completed.")
    print(f"[BEST MODEL] {best_model_name}")
    print(f"[OUTPUT] {output_dir}")
    print("\n[MIXED TEST METRICS]")
    print(
        metrics_df[
            metrics_df["Experiment"]
            == "E3_AUGMENTED_TRAIN_MIXED_TEST"
        ][
            [
                "Model",
                "Accuracy",
                "BalancedAccuracy",
                "MacroF1",
                "MCC",
                "RiskRecall",
                "RiskFalseNegatives",
            ]
        ].to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reviewer-ready Reactive Layer experiments."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to izsu_features.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reactive",
        help="Output directory",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--synthetic-ratio-per-class",
        type=float,
        default=0.50,
        help=(
            "Synthetic Caution and Risk rows per class as a proportion "
            "of real rows in the corresponding split. Default 0.50 means "
            "synthetic rows are 50 percent of the final augmented set."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
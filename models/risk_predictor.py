"""
risk_predictor.py

Inference utility for the new Reactive Layer model bundle.

The saved bundle contains one fitted sklearn Pipeline. Missing-value imputation
and scaling therefore use the transformations learned from training data; the
prediction file's own median is never fitted during inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DATE_COL = "Tarih"
LOCATION_ID_COL = "NoktaId"
LOCATION_NAME_COL = "NoktaAdi"


def resolve_existing_path(
    path_text: str,
    project_root: Path,
    models_dir: Path,
    resource_name: str,
) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"{resource_name} bulunamadı: {path}")

    candidates = [
        project_root / path,
        models_dir / path,
        project_root / "models" / path,
    ]

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)

        if candidate.exists():
            return candidate

    searched = "\n".join(f" - {candidate}" for candidate in checked)
    raise FileNotFoundError(
        f"{resource_name} bulunamadı. Kontrol edilen yollar:\n{searched}"
    )


def resolve_output_path(
    path_text: str,
    project_root: Path,
    models_dir: Path,
) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        return path

    root_outputs = project_root / "outputs"
    models_outputs = models_dir / "outputs"

    if models_outputs.exists() and not root_outputs.exists():
        return models_dir / path

    return project_root / path


def predict(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    include_probabilities: bool,
) -> pd.DataFrame:
    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Girdi CSV bulunamadı: {input_path}")

    bundle = joblib.load(model_path)
    if "pipeline" not in bundle:
        raise ValueError(
            "Model paketi yeni Reactive Layer formatında değil: "
            "'pipeline' anahtarı bulunamadı."
        )

    pipeline = bundle["pipeline"]
    feature_columns = list(bundle["feature_columns"])
    model_name = bundle.get("model_name", "Unknown")
    class_labels = bundle.get("class_labels", [])

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(
            df[DATE_COL],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

    for column in feature_columns:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    predictions = pipeline.predict(df[feature_columns])
    df["PredictedRiskClass"] = predictions
    df["PredictionModel"] = model_name

    if include_probabilities and hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(df[feature_columns])
        learned_classes = pipeline.named_steps["model"].classes_
        for class_index, class_label in enumerate(learned_classes):
            df[f"Probability_{class_label}"] = probabilities[:, class_index]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    show_columns = [
        column
        for column in [
            DATE_COL,
            LOCATION_ID_COL,
            LOCATION_NAME_COL,
            "PredictedRiskClass",
            "PredictionModel",
        ]
        if column in df.columns
    ]

    print(f"[MODEL] {model_name}")
    print(f"[ROWS] {len(df)}")
    print(f"[OUTPUT] {output_path}")
    print(df[show_columns].head().to_string(index=False))

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict current Good/Caution/Risk classes."
    )
    parser.add_argument(
        "--model",
        default="outputs/reactive/best_reactive_model.joblib",
    )
    parser.add_argument(
        "--input",
        default="data/data/izsu_features.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/reactive/reactive_inference_predictions.csv",
    )
    parser.add_argument(
        "--no-probabilities",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = Path(__file__).resolve().parent
    project_root = (
        models_dir.parent if models_dir.name == "models" else Path.cwd()
    )

    model_path = resolve_existing_path(
        args.model,
        project_root,
        models_dir,
        "Model dosyası",
    )
    input_path = resolve_existing_path(
        args.input,
        project_root,
        models_dir,
        "Girdi CSV",
    )
    output_path = resolve_output_path(
        args.output,
        project_root,
        models_dir,
    )

    predict(
        model_path,
        input_path,
        output_path,
        include_probabilities=not args.no_probabilities,
    )


if __name__ == "__main__":
    main()
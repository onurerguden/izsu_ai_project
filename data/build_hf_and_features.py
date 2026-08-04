"""
build_hf_and_features.py

Builds the two model-ready İzSU datasets from izsu_data_cleaned.csv:

- izsu_health_factor.csv
- izsu_features.csv

All Health Factor, WAWQI, parameter-score, fail-fast and class calculations are
performed by utils.hf_calculator. This file does not contain a second HF formula.

It also writes reviewer-facing validation outputs after each successful run:

- hf_parameter_configuration.csv
- hf_formula_checks.csv
- hf_validation_summary.csv
- HF_Technical_Validation_Report.docx
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


STATE_NAME = "last_hf_success_date.txt"
HF_OUTPUT_NAME = "izsu_health_factor.csv"
FEATURE_OUTPUT_NAME = "izsu_features.csv"


# ---------------------------------------------------------------------------
# Import the single, central HF implementation from the project utils folder.
# This remains robust when this script is stored in project_root/data/.
# ---------------------------------------------------------------------------
def _add_project_root_to_path(script_path: Path) -> Path:
    candidates = [
        script_path.parent,
        script_path.parent.parent,
        Path.cwd(),
    ]

    for candidate in candidates:
        if (candidate / "utils" / "hf_calculator.py").exists():
            resolved = candidate.resolve()
            if str(resolved) not in sys.path:
                sys.path.insert(0, str(resolved))
            return resolved

    searched = "\n".join(str(path / "utils" / "hf_calculator.py") for path in candidates)
    raise FileNotFoundError(
        "utils/hf_calculator.py bulunamadı. Kontrol edilen yollar:\n" + searched
    )


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = _add_project_root_to_path(SCRIPT_PATH)

from utils.hf_calculator import (  # noqa: E402
    calculate_hf_for_group,
    parameter_configuration_table,
    validate_formula_invariants,
    validation_summary_table,
    write_docx_report,
)
from utils.parameters import PARAMETERS, canonical_parameter_name  # noqa: E402


# ---------------------------------------------------------------------------
# File and date helpers
# ---------------------------------------------------------------------------
def find_clean_csv(script_path: Path) -> tuple[Path, Path]:
    """Find izsu_data_cleaned.csv and return its directory and path."""

    script_dir = script_path.parent
    candidates = [
        script_dir / "izsu_data_cleaned.csv",
        script_dir / "data" / "izsu_data_cleaned.csv",
        script_dir.parent / "data" / "izsu_data_cleaned.csv",
        script_dir.parent / "data" / "data" / "izsu_data_cleaned.csv",
        Path.cwd() / "izsu_data_cleaned.csv",
        Path.cwd() / "data" / "izsu_data_cleaned.csv",
        Path.cwd() / "data" / "data" / "izsu_data_cleaned.csv",
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved.parent, resolved

    return script_dir, script_dir / "izsu_data_cleaned.csv"


def read_state_date(path: Path):
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8").strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    raise ValueError(
        f"State tarihi okunamadı: {path}. Beklenen biçim YYYY-MM-DD veya DD.MM.YYYY."
    )


def normalize_iso_dates(df: pd.DataFrame, column: str = "Tarih") -> pd.DataFrame:
    """Normalize date values so incremental duplicate removal is type-safe."""

    result = df.copy()
    parsed = pd.to_datetime(result[column], errors="coerce")
    result[column] = parsed.dt.strftime("%Y-%m-%d")
    return result


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically so state is never advanced before durable output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def first_nonempty(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return pd.NA

    if values.dtype == object:
        text_values = values.astype(str).str.strip()
        text_values = text_values[text_values != ""]
        if text_values.empty:
            return pd.NA
        return text_values.iloc[0]

    return values.iloc[0]


def build_work_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Select and normalize the columns required by HF and feature building."""

    column_choices = {
        "Tarih": ("Tarih_Clean", "Tarih"),
        "NoktaAdi": ("NoktaAdi_Clean", "NoktaAdi"),
        "ParametreAdi": ("ParametreAdi_Clean", "ParametreAdi"),
        "Birim": ("Birim_Clean", "Birim"),
        "Deger": ("Deger_Num", "Deger"),
        "DegerRaw": ("DegerRaw", "Deger"),
    }

    selected: dict[str, pd.Series] = {}
    for output_name, candidates in column_choices.items():
        source = next((name for name in candidates if name in raw.columns), None)
        if source is None:
            raise ValueError(
                f"Girdi CSV içinde '{output_name}' için uygun sütun bulunamadı: {candidates}"
            )
        selected[output_name] = raw[source]

    work = pd.DataFrame(selected)

    # Preserve point metadata when available. These fields are useful for the
    # reviewer-facing data summary and prevent two points with the same name
    # from being treated as one physical sampling point.
    metadata_columns = [
        "NoktaId",
        "NoktaTanimi",
        "Ilce",
        "IlceKodu",
        "NoktaKodu",
        "Enlem",
        "Boylam",
    ]
    for column in metadata_columns:
        work[column] = raw[column] if column in raw.columns else pd.NA

    work["Tarih"] = pd.to_datetime(work["Tarih"], errors="coerce").dt.date
    work["NoktaAdi"] = work["NoktaAdi"].astype("string").str.strip()
    work["ParametreAdi"] = work["ParametreAdi"].map(canonical_parameter_name)
    work["Deger"] = pd.to_numeric(work["Deger"], errors="coerce")

    work = work.dropna(subset=["Tarih", "NoktaAdi", "ParametreAdi"])
    work = work[work["NoktaAdi"] != ""]
    work = work[work["ParametreAdi"] != ""]

    return work.reset_index(drop=True)


# ---------------------------------------------------------------------------
# HF and feature construction
# ---------------------------------------------------------------------------
def group_identity_columns(work: pd.DataFrame) -> list[str]:
    """Use physical point ID when available; otherwise fall back to point name."""

    if "NoktaId" in work.columns and work["NoktaId"].notna().any():
        return ["Tarih", "NoktaId"]
    return ["Tarih", "NoktaAdi"]


def metadata_for_group(group: pd.DataFrame) -> dict:
    fields = [
        "NoktaId",
        "NoktaAdi",
        "NoktaTanimi",
        "Ilce",
        "IlceKodu",
        "NoktaKodu",
        "Enlem",
        "Boylam",
    ]
    return {
        field: first_nonempty(group[field])
        for field in fields
        if field in group.columns
    }


def calculate_hf_rows(work: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    identity_columns = group_identity_columns(work)

    for identity, group in work.groupby(identity_columns, dropna=False, sort=True):
        if not isinstance(identity, tuple):
            identity = (identity,)

        date_value = identity[0]
        result = calculate_hf_for_group(
            group,
            parameter_col="ParametreAdi",
            value_col="Deger",
            unit_col="Birim",
            raw_value_col="DegerRaw",
        )

        rows.append(
            {
                "Tarih": date_value,
                **metadata_for_group(group),
                **result,
            }
        )

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    result_df = normalize_iso_dates(result_df)

    ordered_prefix = [
        "Tarih",
        "NoktaId",
        "NoktaAdi",
        "NoktaTanimi",
        "Ilce",
        "IlceKodu",
        "NoktaKodu",
        "Enlem",
        "Boylam",
        "HealthFactor",
        "WAWQI",
        "FailFast",
        "FailFastReason",
        "RiskClass",
        "WAWQICoverage",
        "WAWQIParametersUsed",
        "WAWQIParameterList",
    ]
    score_columns = [f"{name}_score" for name in PARAMETERS]
    ordered = [column for column in ordered_prefix + score_columns if column in result_df.columns]
    remaining = [column for column in result_df.columns if column not in ordered]

    return (
        result_df[ordered + remaining]
        .sort_values(["Tarih", "NoktaId", "NoktaAdi"], na_position="last")
        .reset_index(drop=True)
    )


def build_wide_parameter_values(work: pd.DataFrame) -> pd.DataFrame:
    identity_columns = group_identity_columns(work)

    metadata_fields = [
        column
        for column in [
            "NoktaAdi",
            "NoktaTanimi",
            "Ilce",
            "IlceKodu",
            "NoktaKodu",
            "Enlem",
            "Boylam",
        ]
        if column not in identity_columns
    ]

    metadata = (
        work.groupby(identity_columns, dropna=False, sort=True)[metadata_fields]
        .agg(first_nonempty)
        .reset_index()
    )

    wide_values = (
        work.pivot_table(
            index=identity_columns,
            columns="ParametreAdi",
            values="Deger",
            aggfunc="mean",
        )
        .reset_index()
    )
    wide_values.columns.name = None

    wide_values = pd.merge(
        metadata,
        wide_values,
        on=identity_columns,
        how="outer",
        validate="one_to_one",
    )
    return normalize_iso_dates(wide_values)


def merge_features(work: pd.DataFrame, hf_new: pd.DataFrame) -> pd.DataFrame:
    wide_values = build_wide_parameter_values(work)

    has_point_ids = "NoktaId" in hf_new.columns and hf_new["NoktaId"].notna().any()
    merge_keys = ["Tarih", "NoktaId"] if has_point_ids else ["Tarih", "NoktaAdi"]

    # Metadata is retained from the HF table; remove duplicate metadata from
    # the wide table before merging.
    duplicate_metadata = [
        column
        for column in [
            "NoktaAdi",
            "NoktaTanimi",
            "Ilce",
            "IlceKodu",
            "NoktaKodu",
            "Enlem",
            "Boylam",
        ]
        if column in wide_values.columns and column not in merge_keys
    ]
    parameter_values = wide_values.drop(columns=duplicate_metadata)

    features = pd.merge(
        hf_new,
        parameter_values,
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )

    parameter_order = [name for name in PARAMETERS if name in features.columns]
    prefix = [column for column in hf_new.columns if column in features.columns]
    remaining = [
        column
        for column in features.columns
        if column not in prefix and column not in parameter_order
    ]

    return features[prefix + parameter_order + remaining]


def deduplication_keys(df: pd.DataFrame) -> list[str]:
    if "NoktaId" in df.columns and df["NoktaId"].notna().any():
        return ["Tarih", "NoktaId"]
    return ["Tarih", "NoktaAdi"]


def append_and_replace_existing(path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    """Append incremental rows and let recalculated rows replace older versions."""

    new_normalized = normalize_iso_dates(new_df)
    base = None

    if not path.exists():
        combined = new_normalized
    else:
        base = pd.read_csv(path, encoding="utf-8-sig")
        base = normalize_iso_dates(base)
        combined = pd.concat([base, new_normalized], ignore_index=True, sort=False)

    # During migration from an older output without NoktaId, use the compatible
    # date-name key once. Fresh outputs use the stable physical point ID.
    new_has_ids = "NoktaId" in new_normalized.columns and new_normalized["NoktaId"].notna().any()
    base_has_ids = (
        base is None
        or ("NoktaId" in base.columns and base["NoktaId"].notna().any())
    )
    keys = ["Tarih", "NoktaId"] if new_has_ids and base_has_ids else ["Tarih", "NoktaAdi"]
    combined = combined.drop_duplicates(subset=keys, keep="last")

    sort_columns = [column for column in ["Tarih", "NoktaId", "NoktaAdi"] if column in combined.columns]
    return combined.sort_values(sort_columns, na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reviewer-facing validation/report outputs
# ---------------------------------------------------------------------------
def write_validation_outputs(
    *,
    cleaned_raw: pd.DataFrame,
    hf_all: pd.DataFrame,
    data_dir: Path,
    input_path: Path,
) -> None:
    parameter_table = parameter_configuration_table()
    formula_checks = validate_formula_invariants()
    summary_table = validation_summary_table(cleaned_raw, hf_all, formula_checks)

    atomic_write_csv(parameter_table, data_dir / "hf_parameter_configuration.csv")
    atomic_write_csv(formula_checks, data_dir / "hf_formula_checks.csv")
    atomic_write_csv(summary_table, data_dir / "hf_validation_summary.csv")

    report_path = data_dir / "HF_Technical_Validation_Report.docx"
    try:
        write_docx_report(
            output_path=report_path,
            input_path=input_path,
            parameter_table=parameter_table,
            summary_table=summary_table,
            formula_checks=formula_checks,
        )
        print(f"[✓] Teknik HF raporu: {report_path}")
    except Exception as exc:
        print(f"[UYARI] Word raporu üretilemedi: {exc}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    data_dir, input_path = find_clean_csv(SCRIPT_PATH)

    if not input_path.exists():
        print(f"[HATA] Girdi bulunamadı: {input_path}")
        return

    hf_output_path = data_dir / HF_OUTPUT_NAME
    feature_output_path = data_dir / FEATURE_OUTPUT_NAME
    state_path = data_dir / STATE_NAME

    last_date = read_state_date(state_path)

    print(f"[i] Proje kökü : {PROJECT_ROOT}")
    print(f"[i] Girdi       : {input_path}")
    if last_date is None:
        print("[i] State yok: temiz verinin tamamı hesaplanacak.")
    else:
        print(f"[i] Yalnızca state kullanılıyor; son işlenen tarih: {last_date}")

    if last_date and (not hf_output_path.exists() or not feature_output_path.exists()):
        print(
            "[UYARI] State dosyası var ancak HF/features çıktılarından biri yok. "
            "Bu çalıştırma yalnızca state tarihinden sonraki kayıtları üretecektir. "
            "Tam yeniden üretim için last_hf_success_date.txt dosyasını silin."
        )

    cleaned_raw = pd.read_csv(input_path, encoding="utf-8-sig")
    work_all = build_work_dataframe(cleaned_raw)

    if last_date is not None:
        work_new = work_all[work_all["Tarih"] > last_date].copy()
    else:
        work_new = work_all.copy()

    if work_new.empty:
        print("[i] Hesaplanacak yeni tarih-nokta kaydı yok.")
        if hf_output_path.exists():
            hf_all = pd.read_csv(hf_output_path, encoding="utf-8-sig")
            hf_all = normalize_iso_dates(hf_all)
            write_validation_outputs(
                cleaned_raw=cleaned_raw,
                hf_all=hf_all,
                data_dir=data_dir,
                input_path=input_path,
            )
        return

    hf_new = calculate_hf_rows(work_new)
    features_new = merge_features(work_new, hf_new)

    hf_all = append_and_replace_existing(hf_output_path, hf_new)
    features_all = append_and_replace_existing(feature_output_path, features_new)

    # First persist both datasets. Advance state only after both writes succeed.
    atomic_write_csv(hf_all, hf_output_path)
    atomic_write_csv(features_all, feature_output_path)

    max_date = pd.to_datetime(hf_all["Tarih"], errors="coerce").dt.date.max()
    if pd.notna(max_date):
        state_path.write_text(max_date.strftime("%Y-%m-%d"), encoding="utf-8")

    write_validation_outputs(
        cleaned_raw=cleaned_raw,
        hf_all=hf_all,
        data_dir=data_dir,
        input_path=input_path,
    )

    class_counts = hf_all["RiskClass"].value_counts(dropna=False).to_dict()

    print("--------------------------------------------------")
    print("HF & Features tamamlandı — merkezi utils.hf_calculator kullanıldı")
    print(f"Yeni HF satırı       : {len(hf_new)}")
    print(f"Toplam HF satırı     : {len(hf_all)}")
    print(f"Yeni feature satırı  : {len(features_new)}")
    print(f"Toplam feature satırı: {len(features_all)}")
    print(f"Son tarih             : {max_date}")
    print(
        "Sınıf dağılımı       : "
        f"Good={class_counts.get('Good', 0)}, "
        f"Caution={class_counts.get('Caution', 0)}, "
        f"Risk={class_counts.get('Risk', 0)}, "
        f"Unknown={class_counts.get('Unknown', 0)}"
    )
    print(f"HF CSV                : {hf_output_path}")
    print(f"Features CSV          : {feature_output_path}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
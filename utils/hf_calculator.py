"""
hf_calculator.py
Calculates Health Factor (HF) scores for İzmir water quality
based on parameters, weights, and nonlinear risk adjustment.
"""

import math
import pandas as pd
from utils.parameters import PARAMETERS, ALPHA, HF_SCALE


def calculate_single_parameter_score(value: float, limit, alpha: float = ALPHA) -> float:
    """
    Calculates a normalized score (Si) for one parameter.
    Si = 1 - (value / limit)^alpha
    Handles both upper and range-based limits (e.g., pH).
    """
    if limit is None or pd.isna(value):
        return 1.0  # ignore parameters without defined limits
    if isinstance(limit, tuple):  # e.g., (6.5, 9.5)
        low, high = limit
        if low <= value <= high:
            return 1.0
        else:
            # Penalize deviation from normal pH range
            deviation = min(abs(value - low), abs(value - high))
            return max(0.0, 1 - (deviation / (high - low)) ** alpha)
    if limit == 0:  # fail-fast param handled elsewhere
        return 1.0 if value == 0 else 0.0
    ratio = min(value / limit, 1.5)  # clip excessive ratio
    return max(0.0, 1 - ratio ** alpha)


def calculate_health_factor(df_row: pd.Series) -> dict:
    """
    Computes the HF score and classification for a given sample (DataFrame row).
    Returns a dict with HF score and classification label.
    """
    total_score = 0.0
    fail_fast_triggered = False

    for param, info in PARAMETERS.items():
        value = df_row.get(param)
        if value is None or pd.isna(value):
            continue

        # Fail-fast: exceed limit on critical parameters
        if info.get("failfast") and info.get("limit") is not None:
            if value > info["limit"]:
                fail_fast_triggered = True
                break

        si = calculate_single_parameter_score(value, info.get("limit"))
        total_score += info["weight"] * si

    hf = HF_SCALE * total_score

    if fail_fast_triggered:
        classification = "RISK"
        hf = 0.0
    elif hf >= 85:
        classification = "GOOD"
    elif hf >= 60:
        classification = "CAUTION"
    else:
        classification = "RISK"

    return {"HF": round(hf, 2), "Classification": classification}


def apply_hf_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies HF calculation to all rows of the given DataFrame.
    Adds 'HF' and 'Classification' columns.
    """
    results = df.apply(calculate_health_factor, axis=1, result_type="expand")
    df = pd.concat([df, results], axis=1)
    return df
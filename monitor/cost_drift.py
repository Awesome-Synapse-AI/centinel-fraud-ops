from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy


DAILY_DOLLAR_THRESHOLD = 0.5


def kl_per_feature(baseline: pd.DataFrame, prod: pd.DataFrame, bins: int = 50) -> dict[str, float]:
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer.")

    scores: dict[str, float] = {}
    shared_columns = baseline.columns.intersection(prod.columns)
    for col in shared_columns:
        baseline_series = baseline[col]
        production_series = prod[col]
        if not pd.api.types.is_numeric_dtype(baseline_series) or not pd.api.types.is_numeric_dtype(
            production_series
        ):
            continue
        if pd.api.types.is_complex_dtype(baseline_series) or pd.api.types.is_complex_dtype(production_series):
            continue

        # Convert nullable and mixed numeric extension dtypes to a consistent
        # representation, then exclude NaN and infinities from both windows.
        baseline_values = baseline_series.to_numpy(dtype=np.float64, na_value=np.nan)
        production_values = production_series.to_numpy(dtype=np.float64, na_value=np.nan)
        baseline_values = baseline_values[np.isfinite(baseline_values)]
        production_values = production_values[np.isfinite(production_values)]
        if baseline_values.size == 0 or production_values.size == 0:
            scores[col] = 0.0
            continue

        lower = min(float(baseline_values.min()), float(production_values.min()))
        upper = max(float(baseline_values.max()), float(production_values.max()))
        if lower == upper:
            scores[col] = 0.0
            continue

        edges = np.linspace(lower, upper, bins + 1)
        p, _ = np.histogram(baseline_values, bins=edges)
        q, _ = np.histogram(production_values, bins=edges)
        epsilon = 1e-12
        p = (p.astype(np.float64) + epsilon) / (p.sum() + epsilon * bins)
        q = (q.astype(np.float64) + epsilon) / (q.sum() + epsilon * bins)
        scores[col] = float(entropy(p, q))
    return scores


def cost_weighted_drift(
    kl: dict[str, float],
    importance: dict[str, float],
    p_fraud: float,
    c_fn: float = 420.0,
    c_fp: float = 4.0,
) -> pd.DataFrame:
    """Rank features by KL drift weighted by model importance and expected loss."""
    if (
        isinstance(p_fraud, bool)
        or not isinstance(p_fraud, (int, float, np.number))
        or not np.isfinite(p_fraud)
        or not 0.0 <= p_fraud <= 1.0
    ):
        raise ValueError("p_fraud must be between 0 and 1.")
    if (
        isinstance(c_fn, bool)
        or isinstance(c_fp, bool)
        or not isinstance(c_fn, (int, float, np.number))
        or not isinstance(c_fp, (int, float, np.number))
        or not np.isfinite(c_fn)
        or not np.isfinite(c_fp)
        or c_fn < 0
        or c_fp < 0
    ):
        raise ValueError("Cost inputs must be non-negative.")
    expected_loss = p_fraud * c_fn + (1 - p_fraud) * c_fp
    rows = []
    for feature, raw_kl in kl.items():
        if isinstance(raw_kl, bool) or not isinstance(raw_kl, (int, float, np.number)):
            raise ValueError(f"KL divergence for {feature!r} must be a finite non-negative number.")
        kl_value = float(raw_kl)
        if not np.isfinite(kl_value) or kl_value < 0:
            raise ValueError(f"KL divergence for {feature!r} must be a finite non-negative number.")

        raw_weight = importance.get(feature, 0.0)
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float, np.number)):
            raise ValueError(f"Importance for {feature!r} must be a finite non-negative number.")
        weight = float(raw_weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"Importance for {feature!r} must be a finite non-negative number.")
        rows.append(
            {
                "feature": feature,
                "kl": kl_value,
                "importance": weight,
                "cost_drift_usd": kl_value * weight * expected_loss,
            }
        )

    columns = ["feature", "kl", "importance", "cost_drift_usd"]
    df = pd.DataFrame(rows, columns=columns).sort_values(
        "cost_drift_usd", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    df["cum_cost_drift"] = df["cost_drift_usd"].cumsum()
    return df

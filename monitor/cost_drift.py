from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy


def kl_per_feature(baseline: pd.DataFrame, prod: pd.DataFrame, bins: int = 50) -> dict[str, float]:
    scores: dict[str, float] = {}
    for col in baseline.columns:
        if not pd.api.types.is_numeric_dtype(baseline[col]):
            continue
        lower = min(float(baseline[col].min()), float(prod[col].min()))
        upper = max(float(baseline[col].max()), float(prod[col].max()))
        if lower == upper:
            scores[col] = 0.0
            continue
        edges = np.linspace(lower, upper, bins + 1)
        p, _ = np.histogram(baseline[col], bins=edges, density=True)
        q, _ = np.histogram(prod[col], bins=edges, density=True)
        p = p / p.sum() + 1e-12
        q = q / q.sum() + 1e-12
        scores[col] = float(entropy(p, q))
    return scores


def cost_weighted_drift(
    kl: dict[str, float],
    importance: dict[str, float],
    p_fraud: float,
    c_fn: float = 420.0,
    c_fp: float = 4.0,
) -> pd.DataFrame:
    expected_loss = p_fraud * c_fn + (1 - p_fraud) * c_fp
    rows = [
        {
            "feature": feature,
            "kl": kl_value,
            "importance": float(importance.get(feature, 0.0)),
            "cost_drift_usd": kl_value * float(importance.get(feature, 0.0)) * expected_loss,
        }
        for feature, kl_value in kl.items()
    ]
    df = pd.DataFrame(rows).sort_values("cost_drift_usd", ascending=False).reset_index(drop=True)
    df["cum_cost_drift"] = df["cost_drift_usd"].cumsum()
    return df

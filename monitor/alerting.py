"""Persist and forward high-cost drift alerts using managed GCP services."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests


def send_alert(
    top_features: pd.DataFrame,
    total_expected_loss: float,
    *,
    alert_function_url: str | None = None,
    db: Any | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Write an alert to Firestore, then notify the HTTP Cloud Function.

    The Firestore client is created lazily so importing drift math remains
    usable locally without Application Default Credentials.
    """
    if top_features.empty:
        raise ValueError("top_features must contain at least one ranked feature.")
    required = {"feature", "cost_drift_usd"}
    if missing := required.difference(top_features.columns):
        raise ValueError(f"top_features is missing columns: {sorted(missing)}")

    top = top_features.iloc[0]
    payload = {
        "total_expected_loss_usd": float(total_expected_loss),
        "top_feature": str(top["feature"]),
        "top_feature_cost_usd": float(top["cost_drift_usd"]),
    }
    if db is None:
        from google.cloud import firestore

        db = firestore.Client()
        timestamp = firestore.SERVER_TIMESTAMP
    else:
        # Tests and alternate Firestore-compatible clients do not need to
        # expose Google's sentinel; Firestore receives it in production.
        timestamp = None
    record = {"timestamp": timestamp, **payload}
    db.collection("alerts").add(record)

    url = alert_function_url or os.getenv("ALERT_FUNCTION_URL")
    if url:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    return payload

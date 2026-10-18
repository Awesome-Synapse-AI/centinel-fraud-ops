"""Persist and forward high-cost drift alerts using managed GCP services."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from numbers import Real
from typing import Any

import pandas as pd
import requests


def _identity_token(audience: str) -> str:
    """Get a Google OIDC token for invoking an IAM-protected endpoint."""
    try:
        # Prefer the active local gcloud identity. This lets a developer use
        # user credentials for invocation while ADC may point at another
        # service account for Firestore access.
        result = subprocess.run(
            [shutil.which("gcloud.cmd") or shutil.which("gcloud") or "gcloud.cmd", "auth", "print-identity-token"],
            check=True,
            capture_output=True,
            text=True,
        )
        token = result.stdout.strip()
        if token:
            return token
    except (OSError, subprocess.CalledProcessError):
        pass

    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import id_token

        return id_token.fetch_id_token(Request(), audience)
    except Exception as adc_error:
        raise RuntimeError(
            "Could not obtain an identity token for ALERT_FUNCTION_URL. "
            "Use an attached Google service account or run 'gcloud auth login'."
        ) from adc_error


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
    if not isinstance(top_features, pd.DataFrame) or top_features.empty:
        raise ValueError("top_features must contain at least one ranked feature.")
    required = {"feature", "cost_drift_usd"}
    if missing := required.difference(top_features.columns):
        raise ValueError(f"top_features is missing columns: {sorted(missing)}")

    if isinstance(total_expected_loss, bool) or not isinstance(total_expected_loss, Real):
        raise ValueError("total_expected_loss must be a finite non-negative number.")
    total_expected_loss = float(total_expected_loss)
    if not math.isfinite(total_expected_loss) or total_expected_loss < 0:
        raise ValueError("total_expected_loss must be a finite non-negative number.")

    top = top_features.iloc[0]
    raw_feature = top["feature"]
    if not isinstance(raw_feature, str) or not raw_feature.strip():
        raise ValueError("The top ranked feature must have a non-empty feature name.")
    feature = raw_feature.strip()
    top_feature_cost = top["cost_drift_usd"]
    if isinstance(top_feature_cost, bool) or not isinstance(top_feature_cost, Real):
        raise ValueError("The top feature cost must be a finite non-negative number.")
    top_feature_cost = float(top_feature_cost)
    if not math.isfinite(top_feature_cost) or top_feature_cost < 0:
        raise ValueError("The top feature cost must be a finite non-negative number.")
    if not isinstance(timeout, Real) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number.")

    payload = {
        "total_expected_loss_usd": total_expected_loss,
        "top_feature": feature,
        "top_feature_cost_usd": top_feature_cost,
    }
    if db is None:
        from google.cloud import firestore

        database = os.getenv("ALERT_FIRESTORE_DATABASE", "fraud-alerts")
        db = firestore.Client(database=database)
        timestamp = firestore.SERVER_TIMESTAMP
    else:
        # Tests and alternate Firestore-compatible clients do not need to
        # expose Google's sentinel; Firestore receives it in production.
        timestamp = None
    record = {"timestamp": timestamp, **payload}
    db.collection("alerts").add(record)

    url = os.getenv("ALERT_FUNCTION_URL") if alert_function_url is None else alert_function_url
    if url:
        token = _identity_token(url)
        response = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
        response.raise_for_status()
    return payload

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.model_loader import load_model_bundle
from config.category_encoders import encode_category_value
from config.model_schema import CATEGORICAL_RAW_FEATURE_NAMES


class PredictRequest(BaseModel):
    # These shortcuts make the demo convenient.  For production callers,
    # ``features`` accepts the exact feature names persisted with the model
    # bundle (for example ``ProductCD__encoded``).
    TransactionAmt: float | None = None
    card1: int | None = None
    ProductCD: str | None = None
    card2: float | None = None
    card3: float | None = None
    card5: float | None = None
    addr1: float | None = None
    dist1: float | None = None
    merch_fraud_rate_30d: float = 0.0
    merch_volume_30d: float = 0.0
    txn_count_24h: float = 1.0
    amt_sum_24h: float = 0.0
    distinct_productcd_24h: float = 1.0
    has_identity: int = 0
    features: dict[str, Any] = Field(default_factory=dict)

app = FastAPI(title="Fraud Scorer")
bundle: dict[str, Any] | None = None


def get_model_bundle() -> dict[str, Any]:
    global bundle
    if bundle is None:
        bundle = load_model_bundle()
    return bundle


@app.get("/health")
def health() -> dict[str, Any]:
    """Readiness endpoint: only report healthy after the champion is loadable."""
    active_bundle = get_model_bundle()
    return {
        "status": "ok",
        "model_feature_count": len(active_bundle["feature_names"]),
        "threshold": float(active_bundle.get("threshold", 0.5)),
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    active_bundle = get_model_bundle()
    model = active_bundle["model"]
    feature_names = active_bundle["feature_names"]
    threshold = float(active_bundle.get("threshold", 0.5))
    # Keep public field names aligned with the persisted training-time schema.
    # This avoids silently replacing TransactionAmt with zero (the behaviour
    # of the previous ``amount`` shortcut implementation).
    values = payload.model_dump(exclude_none=True)
    values.update(payload.features)
    categorical_mappings = active_bundle.get("categorical_mappings", {})
    for raw_name in CATEGORICAL_RAW_FEATURE_NAMES:
        encoded_name = f"{raw_name}__encoded"
        if encoded_name in values:
            continue
        mapping = categorical_mappings.get(raw_name)
        if mapping is not None:
            raw_value = values.get(raw_name, "__MISSING__")
            values[encoded_name] = encode_category_value(raw_value, mapping)
    row = {feature: values.get(feature, 0) for feature in feature_names}
    frame = pd.DataFrame([row]).fillna(0)
    fraud_prob = float(model.predict_proba(frame)[0, 1])
    return {"fraud_prob": fraud_prob, "flag": fraud_prob >= threshold, "threshold": threshold}

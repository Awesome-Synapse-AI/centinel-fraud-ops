from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.model_loader import load_model_bundle


class PredictRequest(BaseModel):
    amount: float = Field(alias="TransactionAmt")
    card1: int | None = None
    merchant: str | None = Field(default=None, alias="ProductCD")
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

    model_config = {"populate_by_name": True}


app = FastAPI(title="Fraud Scorer")
bundle: dict[str, Any] | None = None


def get_model_bundle() -> dict[str, Any]:
    global bundle
    if bundle is None:
        bundle = load_model_bundle()
    return bundle


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    active_bundle = get_model_bundle()
    model = active_bundle["model"]
    feature_names = active_bundle["feature_names"]
    values = payload.model_dump(by_alias=False)
    row = {feature: values.get(feature, 0) for feature in feature_names}
    frame = pd.DataFrame([row]).fillna(0)
    fraud_prob = float(model.predict_proba(frame)[0, 1])
    return {"fraud_prob": fraud_prob, "flag": fraud_prob >= 0.5}

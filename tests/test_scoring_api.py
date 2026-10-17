import numpy as np
from fastapi.testclient import TestClient

import app.main as scoring_app


class _CapturingModel:
    def __init__(self):
        self.frame = None

    def predict_proba(self, frame):
        self.frame = frame.copy()
        return np.array([[0.8, 0.2]])


def test_predict_uses_public_transaction_amount_alias_and_feature_overrides(monkeypatch):
    model = _CapturingModel()
    monkeypatch.setattr(
        scoring_app,
        "get_model_bundle",
        lambda: {
            "model": model,
            "feature_names": ["TransactionAmt", "card1", "ProductCD__encoded"],
            "threshold": 0.19,
        },
    )

    response = TestClient(scoring_app.app).post(
        "/predict",
        json={
            "TransactionAmt": 120.0,
            "card1": 13926,
            "features": {"ProductCD__encoded": 3.0},
        },
    )

    assert response.status_code == 200
    assert response.json()["flag"] is True
    assert model.frame.iloc[0].to_dict() == {
        "TransactionAmt": 120.0,
        "card1": 13926.0,
        "ProductCD__encoded": 3.0,
    }


def test_health_only_succeeds_when_a_model_bundle_is_loadable(monkeypatch):
    monkeypatch.setattr(
        scoring_app,
        "get_model_bundle",
        lambda: {"feature_names": ["a", "b"], "threshold": 0.25},
    )

    response = TestClient(scoring_app.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_feature_count": 2, "threshold": 0.25}

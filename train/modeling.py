from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, roc_auc_score


def train_xgb_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> xgb.XGBClassifier:
    neg = max(int((y_train == 0).sum()), 1)
    pos = max(int((y_train == 1).sum()), 1)
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        tree_method="hist",
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model


def calibrate_classifier(
    model: xgb.XGBClassifier,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(x_val, y_val)
    return calibrated


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    fraud_prob = model.predict_proba(x_test)[:, 1]
    flag = (fraud_prob >= 0.5).astype(int)
    return {
        "aucpr_test": float(average_precision_score(y_test, fraud_prob)),
        "roc_auc_test": float(roc_auc_score(y_test, fraud_prob)),
        "positive_rate_pred": float(np.mean(flag)),
    }


def feature_importance(model: xgb.XGBClassifier, feature_names: list[str]) -> dict[str, float]:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    return {name: float(gain.get(name, 0.0)) for name in feature_names}


def save_bundle(
    output_path: Path,
    calibrated_model,
    feature_names: list[str],
    metrics: dict[str, float],
    importance: dict[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": calibrated_model,
        "feature_names": feature_names,
        "metrics": metrics,
        "importance": importance,
    }
    joblib.dump(bundle, output_path)

    metrics_path = output_path.with_suffix(".metrics.json")
    importance_path = output_path.with_suffix(".importance.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    importance_path.write_text(json.dumps(importance, indent=2), encoding="utf-8")

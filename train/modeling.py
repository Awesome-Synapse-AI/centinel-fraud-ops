from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None


class TqdmXGBoostCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_rounds: int):
        self.total_rounds = total_rounds
        self._bar = None

    def before_training(self, model):
        if tqdm is not None:
            self._bar = tqdm(total=self.total_rounds, desc="xgboost", unit="round")
        return model

    def after_iteration(self, model, epoch: int, evals_log):
        if self._bar is not None:
            self._bar.update(1)
            metrics = []
            for data_name, values in evals_log.items():
                for metric_name, metric_values in values.items():
                    metrics.append(f"{data_name}-{metric_name}={metric_values[-1]:.5f}")
            if metrics:
                self._bar.set_postfix_str(", ".join(metrics))
        return False

    def after_training(self, model):
        if self._bar is not None:
            self._bar.close()
        return model


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
    callbacks = [TqdmXGBoostCallback(model.get_params().get("n_estimators", 0))]
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False, callbacks=callbacks)
    return model


def calibrate_classifier(
    model: xgb.XGBClassifier,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(x_val, y_val)
    return calibrated


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | dict]:
    fraud_prob = model.predict_proba(x_test)[:, 1]
    flag = (fraud_prob >= 0.5).astype(int)
    report = classification_report(y_test, flag, output_dict=True, zero_division=0)
    return {
        "aucpr_test": float(average_precision_score(y_test, fraud_prob)),
        "roc_auc_test": float(roc_auc_score(y_test, fraud_prob)),
        "positive_rate_pred": float(np.mean(flag)),
        "classification_report": report,
    }


def feature_importance(model: xgb.XGBClassifier, feature_names: list[str]) -> dict[str, float]:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    return {name: float(gain.get(name, 0.0)) for name in feature_names}


def save_bundle(
    output_path: Path,
    calibrated_model,
    feature_names: list[str],
    metrics: dict[str, float | dict],
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
    report_path = output_path.with_suffix(".report.txt")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    importance_path.write_text(json.dumps(importance, indent=2), encoding="utf-8")
    report = metrics.get("classification_report")
    if isinstance(report, dict):
        lines = ["classification_report"]
        for label, row in report.items():
            if isinstance(row, dict):
                lines.append(
                    f"{label}: precision={row.get('precision', 0.0):.6f} "
                    f"recall={row.get('recall', 0.0):.6f} "
                    f"f1={row.get('f1-score', 0.0):.6f} "
                    f"support={int(row.get('support', 0))}"
                )
            else:
                lines.append(f"{label}: {row}")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

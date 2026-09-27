from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)


class SpinnerProgress:
    def __init__(self, message: str):
        self.message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        def _spin() -> None:
            frames = "|/-\\"
            idx = 0
            while not self._stop.is_set():
                sys.stdout.write(f"\r[train] {self.message} {frames[idx % len(frames)]}")
                sys.stdout.flush()
                idx += 1
                time.sleep(0.1)
            sys.stdout.write(f"\r[train] {self.message} done{' ' * 20}\n")
            sys.stdout.flush()

        self._thread = threading.Thread(target=_spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return False


def train_xgb_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
) -> xgb.XGBClassifier:
    neg = max(int((y_train == 0).sum()), 1)
    pos = max(int((y_train == 1).sum()), 1)
    base_params = {
        "n_estimators": 600,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 1.0,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "scale_pos_weight": max(1.0, neg / pos),
    }
    if params:
        base_params.update(params)
    model = xgb.XGBClassifier(
        **base_params,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        tree_method="hist",
        random_state=42,
    )
    with SpinnerProgress("Training XGBoost classifier"):
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model


def calibrate_classifier(
    model: xgb.XGBClassifier,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(x_val, y_val)
    return calibrated


def find_best_threshold(
    y_true: pd.Series,
    prob_pred: np.ndarray,
    beta: float = 2.0,
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob_pred)
    if len(thresholds) == 0:
        return 0.5, 0.0

    beta_sq = beta**2
    denom = beta_sq * precision + recall
    f_scores = np.divide(
        (1 + beta_sq) * precision * recall,
        denom,
        out=np.zeros_like(precision, dtype=float),
        where=denom > 0,
    )
    best_idx = int(np.nanargmax(f_scores[:-1]))
    return float(thresholds[best_idx]), float(f_scores[best_idx])


def evaluate_model(
    model,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float | None = None,
    beta: float = 2.0,
) -> dict[str, float | dict]:
    val_prob = model.predict_proba(x_val)[:, 1]
    tuned_threshold, val_fbeta = find_best_threshold(y_val, val_prob, beta=beta)
    decision_threshold = tuned_threshold if threshold is None else threshold

    fraud_prob = model.predict_proba(x_test)[:, 1]
    flag = (fraud_prob >= decision_threshold).astype(int)
    report_text = classification_report(y_test, flag, zero_division=0)
    report = classification_report(y_test, flag, output_dict=True, zero_division=0)
    return {
        "threshold": float(decision_threshold),
        "val_threshold": float(tuned_threshold),
        "val_fbeta": float(val_fbeta),
        "aucpr_test": float(average_precision_score(y_test, fraud_prob)),
        "roc_auc_test": float(roc_auc_score(y_test, fraud_prob)),
        "positive_rate_pred": float(np.mean(flag)),
        "classification_report_text": report_text,
        "classification_report": report,
    }


def summarize_split_distributions(split_stats: list[dict[str, float]]) -> str:
    lines = ["split diagnostics"]
    for row in split_stats:
        lines.append(
            f"{row['split']}: rows={int(row['rows'])} fraud_rate={row['fraud_rate']:.6f} "
            f"min_TransactionDT={row['min_TransactionDT']:.0f} max_TransactionDT={row['max_TransactionDT']:.0f}"
        )
    return "\n".join(lines)


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
        "threshold": float(metrics.get("threshold", 0.5)),
        "val_threshold": float(metrics.get("val_threshold", 0.5)),
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

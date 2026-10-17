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
from sklearn.model_selection import ParameterGrid
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


def _average_precision_scorer(estimator, x: pd.DataFrame, y: pd.Series) -> float:
    """Score XGBoost probabilities without sklearn estimator-tag inference."""
    return float(average_precision_score(y, estimator.predict_proba(x)[:, 1]))


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
        "n_estimators": 500,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
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


def grid_search_xgb_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict[str, list] | None = None,
    validation_fraction: float = 0.2,
    results_dir: Path | None = None,
) -> dict:
    """Find parameters with one chronological holdout from the train split.

    This deliberately does not use k-fold cross-validation or shuffle rows.
    The last part of ``x_train`` is used as a future-like search holdout. The
    caller should still refit the selected parameters on the complete train
    split before using the project's external validation and test splits.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    split_at = int(len(x_train) * (1.0 - validation_fraction))
    if split_at < 1 or split_at >= len(x_train):
        raise ValueError("x_train must contain enough rows for a search holdout.")

    x_search_train = x_train.iloc[:split_at]
    y_search_train = y_train.iloc[:split_at]
    x_search_val = x_train.iloc[split_at:]
    y_search_val = y_train.iloc[split_at:]
    if y_search_train.nunique() < 2 or y_search_val.nunique() < 2:
        raise ValueError("Both chronological search partitions must contain both classes.")

    neg = max(int((y_train == 0).sum()), 1)
    pos = max(int((y_train == 1).sum()), 1)
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "scale_pos_weight": max(1.0, neg / pos),
        "random_state": 42,
        "n_jobs": 1,
    }
    search_grid = param_grid or {
        "n_estimators": [300, 500],
        "max_depth": [5, 7, 10],
        "learning_rate": [0.03, 0.1],
        "min_child_weight": [2, 3, 5],
        "subsample": [0.85],
        "colsample_bytree": [0.8],
    }
    results: list[dict] = []
    candidates = list(ParameterGrid(search_grid))
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
    with SpinnerProgress("Searching XGBoost hyperparameters"):
        for candidate_number, candidate in enumerate(candidates, start=1):
            run_dir = None
            if results_dir is not None:
                run_dir = results_dir / f"config-{candidate_number:03d}"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "config.json").write_text(
                    json.dumps(
                        {
                            "candidate_number": candidate_number,
                            "parameters": candidate,
                            "base_parameters": base_params,
                            "search_train_rows": len(x_search_train),
                            "search_validation_rows": len(x_search_val),
                        },
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
            print(
                f"[grid-search] run {candidate_number}/{len(candidates)} "
                f"config={json.dumps(candidate, sort_keys=True, default=str)}",
                flush=True,
            )
            estimator = xgb.XGBClassifier(**base_params, **candidate)
            started = time.perf_counter()
            estimator.fit(
                x_search_train,
                y_search_train,
                eval_set=[(x_search_val, y_search_val)],
                verbose=False,
            )
            score = _average_precision_scorer(estimator, x_search_val, y_search_val)
            probabilities = estimator.predict_proba(x_search_val)[:, 1]
            elapsed = time.perf_counter() - started
            result = {
                "candidate_number": candidate_number,
                "parameters": candidate,
                "average_precision": float(score),
                "roc_auc": float(roc_auc_score(y_search_val, probabilities)),
                "positive_rate_pred": float(np.mean(probabilities >= 0.5)),
                "fit_time_seconds": float(elapsed),
                "best_iteration": int(getattr(estimator, "best_iteration", -1)),
            }
            results.append(
                {
                    "params": candidate,
                    "mean_test_score": result["average_precision"],
                    "roc_auc": result["roc_auc"],
                    "fit_time": result["fit_time_seconds"],
                    "best_iteration": result["best_iteration"],
                }
            )
            if run_dir is not None:
                (run_dir / "result.json").write_text(
                    json.dumps(result, indent=2, default=str),
                    encoding="utf-8",
                )
            print(
                f"[grid-search] run {candidate_number}/{len(candidates)} done "
                f"time={elapsed:.2f}s average_precision={score:.6f} "
                f"roc_auc={result['roc_auc']:.6f}",
                flush=True,
            )

    if not results:
        raise ValueError("param_grid must contain at least one parameter combination.")
    best = max(results, key=lambda row: row["mean_test_score"])
    return {
        "best_params": dict(best["params"]),
        "best_score": float(best["mean_test_score"]),
        "results": results,
        "search_train_rows": len(x_search_train),
        "search_validation_rows": len(x_search_val),
    }


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
) -> dict[str, float | None | dict]:
    val_prob = model.predict_proba(x_val)[:, 1]
    tuned_threshold, val_fbeta = find_best_threshold(y_val, val_prob, beta=beta)
    decision_threshold = tuned_threshold if threshold is None else threshold

    fraud_prob = model.predict_proba(x_test)[:, 1]
    flag = (fraud_prob >= decision_threshold).astype(int)
    report_text = classification_report(y_test, flag, zero_division=0)
    report = classification_report(y_test, flag, output_dict=True, zero_division=0)
    # ROC AUC is undefined when the time-based test slice contains only one
    # class.  Do not let that valid (if undesirable) split shape prevent the
    # trained model from being persisted after evaluation.
    roc_auc_test = (
        float(roc_auc_score(y_test, fraud_prob))
        if y_test.nunique() >= 2
        else None
    )
    return {
        "threshold": float(decision_threshold),
        "val_threshold": float(tuned_threshold),
        "val_fbeta": float(val_fbeta),
        "aucpr_test": float(average_precision_score(y_test, fraud_prob)),
        "roc_auc_test": roc_auc_test,
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
    metrics: dict[str, float | None | dict],
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

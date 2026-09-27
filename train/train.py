import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from feast import FeatureStore
from google.cloud import bigquery

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.env import load_repo_env
from train.dataset import describe_split, prepare_xy, time_based_split
from train.modeling import (
    calibrate_classifier,
    evaluate_model,
    feature_importance,
    save_bundle,
    summarize_split_distributions,
    train_xgb_classifier,
)


def log_step(message: str) -> None:
    print(f"[train] {message}", flush=True)


def timed_step(message: str):
    class _Timer:
        def __enter__(self):
            self.started = time.perf_counter()
            log_step(message)
            return self

        def __exit__(self, exc_type, exc, tb):
            elapsed = time.perf_counter() - self.started
            log_step(f"{message} done in {elapsed:.2f}s")
            return False

    return _Timer()


def parse_args() -> argparse.Namespace:
    default_feature_repo = Path(__file__).resolve().parent.parent / "feature_repo"
    parser = argparse.ArgumentParser(description="Train the fraud model locally or in Cloud Run.")
    parser.add_argument(
        "--feature-repo",
        type=Path,
        default=default_feature_repo,
        help="Path to the Feast feature repository.",
    )
    parser.add_argument(
        "--feature-service",
        type=str,
        default="fraud_v1",
        help="Feast feature service name for documentation only.",
    )
    parser.add_argument(
        "--threshold-beta",
        type=float,
        default=1.0,
        help="Beta used to tune the decision threshold on validation data.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _feature_refs_from_service(service) -> list[str]:
    feature_refs: list[str] = []
    # Feast 0.6x resolves a FeatureService into projections. Older Feast
    # versions exposed the FeatureViews directly as ``service.features``.
    projections = getattr(service, "feature_view_projections", None)
    if projections is None:
        projections = getattr(service, "features", [])

    for item in projections:
        view_name = getattr(item, "name", None)
        fields = getattr(item, "features", None)
        if fields is None:
            fields = getattr(item, "schema", None)
        if not view_name or fields is None:
            continue
        for field in fields:
            field_name = getattr(field, "name", None)
            if field_name:
                feature_refs.append(f"{view_name}:{field_name}")
    if not feature_refs:
        raise ValueError("No features found in the requested Feast feature service.")
    return feature_refs


def load_entity_spine(table_ref: str) -> pd.DataFrame:
    client = bigquery.Client()
    query = f"""
        SELECT card1, TIMESTAMP(ts) AS ts, isFraud, TransactionDT
        FROM `{table_ref}`
        WHERE card1 IS NOT NULL
          AND ts IS NOT NULL
          AND isFraud IS NOT NULL
          AND TransactionDT IS NOT NULL
    """
    job = client.query(query)
    entity_df = job.to_dataframe()
    if entity_df.empty:
        raise ValueError(f"No training rows found in BigQuery table {table_ref}.")
    entity_df["ts"] = pd.to_datetime(entity_df["ts"], utc=True)
    return entity_df


def main() -> None:
    with timed_step("Loading repository environment"):
        load_repo_env()
    args = parse_args()
    table_ref = os.environ["FEAST_BIGQUERY_TABLE"]
    with timed_step("Initializing Feast feature store"):
        store = FeatureStore(repo_path=str(args.feature_repo.resolve()))
    with timed_step(f"Loading Feast feature service {args.feature_service}"):
        service = store.get_feature_service(args.feature_service)
        feature_refs = _feature_refs_from_service(service)
    log_step(f"Using Feast features: {feature_refs}")
    with timed_step(f"Loading entity spine from BigQuery table {table_ref}"):
        entity_df = load_entity_spine(table_ref)
    with timed_step("Fetching historical features from Feast"):
        df = (
            store.get_historical_features(
                entity_df=entity_df,
                features=feature_refs,
            )
            .to_df()
        )
    log_step(f"Fetched {len(df):,} rows with features")
    with timed_step("Creating time-based train/validation/test split"):
        split = time_based_split(df)
    split_stats = [
        describe_split(split.train, "train"),
        describe_split(split.val, "val"),
        describe_split(split.test, "test"),
    ]
    log_step(summarize_split_distributions(split_stats))

    with timed_step("Preparing training data"):
        x_train, y_train = prepare_xy(split.train)
    with timed_step("Preparing validation data"):
        x_val, y_val = prepare_xy(split.val)
    with timed_step("Preparing test data"):
        x_test, y_test = prepare_xy(split.test)

    with timed_step("Training XGBoost classifier"):
        model = train_xgb_classifier(x_train, y_train, x_val, y_val)
    with timed_step("Calibrating classifier"):
        calibrated = calibrate_classifier(model, x_val, y_val)
    with timed_step("Evaluating model"):
        metrics = evaluate_model(calibrated, x_val, y_val, x_test, y_test, beta=args.threshold_beta)
    report = metrics.get("classification_report", {})
    with timed_step("Computing feature importance"):
        importance = feature_importance(model, list(x_train.columns))
    with timed_step(f"Saving model bundle to {args.output}"):
        save_bundle(args.output, calibrated, list(x_train.columns), metrics, importance)

    log_step(f"Saved model bundle to {args.output}")
    for key, value in metrics.items():
        if key in {"classification_report", "classification_report_text"}:
            continue
        print(f"{key}={value:.6f}")
    report_text = metrics.get("classification_report_text")
    if isinstance(report_text, str):
        print("[eval] Classification report:", flush=True)
        print(report_text.rstrip(), flush=True)
    threshold = metrics.get("threshold")
    val_threshold = metrics.get("val_threshold")
    if isinstance(threshold, (float, int)) and isinstance(val_threshold, (float, int)):
        print(f"threshold={float(threshold):.6f}")
        print(f"val_threshold={float(val_threshold):.6f}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        print(f"MLflow tracking URI configured: {tracking_uri}")


if __name__ == "__main__":
    main()

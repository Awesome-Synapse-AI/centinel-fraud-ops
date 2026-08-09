import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from feast import FeatureStore
from google.cloud import bigquery

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.env import load_repo_env
from train.dataset import prepare_xy, time_based_split
from train.modeling import (
    calibrate_classifier,
    evaluate_model,
    feature_importance,
    save_bundle,
    train_xgb_classifier,
)


def log_step(message: str) -> None:
    print(f"[train] {message}", flush=True)


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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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
    entity_df = client.query(query).to_dataframe()
    if entity_df.empty:
        raise ValueError(f"No training rows found in BigQuery table {table_ref}.")
    entity_df["ts"] = pd.to_datetime(entity_df["ts"], utc=True)
    return entity_df


def main() -> None:
    log_step("Loading repository environment")
    load_repo_env()
    args = parse_args()
    table_ref = os.environ["FEAST_BIGQUERY_TABLE"]
    log_step("Initializing Feast feature store")
    store = FeatureStore(repo_path=str(args.feature_repo.resolve()))
    log_step(f"Loading entity spine from BigQuery table {table_ref}")
    entity_df = load_entity_spine(table_ref)
    log_step("Fetching historical features from Feast")
    df = (
        store.get_historical_features(
            entity_df=entity_df,
            features=[
                "tx_velocity:txn_count_24h",
                "tx_velocity:amt_sum_24h",
                "tx_velocity:distinct_productcd_24h",
                "merchant_risk:merch_fraud_rate_30d",
                "merchant_risk:merch_volume_30d",
            ],
        )
        .to_df()
    )
    log_step(f"Fetched {len(df):,} rows with features")
    log_step("Creating time-based train/validation/test split")
    split = time_based_split(df)

    log_step("Preparing training data")
    x_train, y_train = prepare_xy(split.train)
    log_step("Preparing validation data")
    x_val, y_val = prepare_xy(split.val)
    log_step("Preparing test data")
    x_test, y_test = prepare_xy(split.test)

    log_step("Training XGBoost classifier")
    model = train_xgb_classifier(x_train, y_train, x_val, y_val)
    log_step("Calibrating classifier")
    calibrated = calibrate_classifier(model, x_val, y_val)
    log_step("Evaluating model")
    metrics = evaluate_model(calibrated, x_test, y_test)
    log_step("Computing feature importance")
    importance = feature_importance(model, list(x_train.columns))
    log_step(f"Saving model bundle to {args.output}")
    save_bundle(args.output, calibrated, list(x_train.columns), metrics, importance)

    log_step(f"Saved model bundle to {args.output}")
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        print(f"MLflow tracking URI configured: {tracking_uri}")


if __name__ == "__main__":
    main()

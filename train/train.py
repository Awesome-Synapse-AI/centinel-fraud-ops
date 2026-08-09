import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from feast import FeatureStore

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fraud model locally or in Cloud Run.")
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Legacy parquet path fallback. Prefer Feast historical retrieval.",
    )
    parser.add_argument(
        "--feature-repo",
        type=Path,
        default=Path("feature_repo"),
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


def main() -> None:
    load_repo_env()
    args = parse_args()
    if args.features is not None:
        df = pd.read_parquet(args.features)
    else:
        store = FeatureStore(repo_path=str(args.feature_repo))
        source_df = pd.read_parquet(Path(os.environ.get("FEAST_ENTITY_SPINE", "outputs/staged/v1/features.parquet")))
        entity_df = source_df[["card1", "ts", "isFraud", "TransactionDT"]].copy()
        training_df = (
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
        df = training_df
    split = time_based_split(df)

    x_train, y_train = prepare_xy(split.train)
    x_val, y_val = prepare_xy(split.val)
    x_test, y_test = prepare_xy(split.test)

    model = train_xgb_classifier(x_train, y_train, x_val, y_val)
    calibrated = calibrate_classifier(model, x_val, y_val)
    metrics = evaluate_model(calibrated, x_test, y_test)
    importance = feature_importance(model, list(x_train.columns))
    save_bundle(args.output, calibrated, list(x_train.columns), metrics, importance)

    print(f"Saved model bundle to {args.output}")
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        print(f"MLflow tracking URI configured: {tracking_uri}")


if __name__ == "__main__":
    main()

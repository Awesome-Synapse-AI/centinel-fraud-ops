import argparse
import os
from pathlib import Path

from train.dataset import load_feature_frame, prepare_xy, time_based_split
from train.modeling import (
    calibrate_classifier,
    evaluate_model,
    feature_importance,
    save_bundle,
    train_xgb_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fraud model locally or in Cloud Run.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_feature_frame(args.features)
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

"""Grid-search XGBoost hyperparameters using the project's training flow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from feast import FeatureStore

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.env import load_repo_env
from train.dataset import describe_split, prepare_xy, time_based_split
from train.modeling import (
    calibrate_classifier,
    evaluate_model,
    feature_importance,
    grid_search_xgb_classifier,
    save_bundle,
    summarize_split_distributions,
    train_xgb_classifier,
)
from train.train import _feature_refs_from_service, load_entity_spine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search and train the fraud XGBoost model.")
    parser.add_argument(
        "--feature-repo",
        type=Path,
        default=REPO_ROOT / "feature_repo",
        help="Path to the Feast feature repository.",
    )
    parser.add_argument("--feature-service", default="fraud_v1")
    parser.add_argument(
        "--search-validation-fraction",
        type=float,
        default=0.2,
        help="Chronological fraction of the training split used as the search holdout.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output model bundle path.")
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="CSV path for all grid-search results (default: <output>.grid-search.csv).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for one config.json/result.json folder per search run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_repo_env()
    table_ref = os.environ["FEAST_BIGQUERY_TABLE"]

    store = FeatureStore(repo_path=str(args.feature_repo.resolve()))
    service = store.get_feature_service(args.feature_service)
    feature_refs = _feature_refs_from_service(service)
    entity_df = load_entity_spine(table_ref)
    df = store.get_historical_features(entity_df=entity_df, features=feature_refs).to_df()

    split = time_based_split(df)
    print(summarize_split_distributions([
        describe_split(split.train, "train"),
        describe_split(split.val, "val"),
        describe_split(split.test, "test"),
    ]))
    x_train, y_train = prepare_xy(split.train)
    x_val, y_val = prepare_xy(split.val)
    x_test, y_test = prepare_xy(split.test)

    search = grid_search_xgb_classifier(
        x_train,
        y_train,
        validation_fraction=args.search_validation_fraction,
        results_dir=args.results_dir or args.output.with_suffix(".grid-search-runs"),
    )
    best_params = search["best_params"]
    print(f"best_search_average_precision={search['best_score']:.6f}")
    print(f"best_params={json.dumps(best_params, sort_keys=True)}")

    model = train_xgb_classifier(x_train, y_train, x_val, y_val, params=best_params)
    calibrated = calibrate_classifier(model, x_val, y_val)
    metrics = evaluate_model(calibrated, x_val, y_val, x_test, y_test)
    metrics["best_search_average_precision"] = float(search["best_score"])
    metrics["best_params"] = best_params
    metrics["search_train_rows"] = search["search_train_rows"]
    metrics["search_validation_rows"] = search["search_validation_rows"]
    results_path = args.results or args.output.with_suffix(".grid-search.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search["results"]).to_csv(results_path, index=False)
    save_bundle(args.output, calibrated, list(x_train.columns), metrics, feature_importance(model, list(x_train.columns)))
    print(f"saved_model={args.output}")
    print(f"saved_grid_search_results={results_path}")
    print(f"saved_grid_search_run_artifacts={args.results_dir or args.output.with_suffix('.grid-search-runs')}")


if __name__ == "__main__":
    main()

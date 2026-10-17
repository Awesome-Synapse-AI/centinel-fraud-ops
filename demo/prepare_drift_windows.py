"""Prepare schema-aligned Parquet windows for the drift console."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import joblib
import pandas as pd
import pyarrow.dataset as arrow_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _feature_schema(bundle: object) -> tuple[list[str], dict[str, float]]:
    if not isinstance(bundle, dict):
        raise ValueError("Model bundle must be a mapping containing feature_names and importance.")
    names = bundle.get("feature_names")
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Model bundle must contain a non-empty list of feature_names.")
    if len(set(names)) != len(names):
        raise ValueError("Model bundle feature_names contains duplicates.")

    raw_importance = bundle.get("importance")
    if not isinstance(raw_importance, dict):
        raise ValueError("Model bundle must contain a feature importance mapping.")
    weights: dict[str, float] = {}
    for name in names:
        value = raw_importance.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Importance for {name!r} must be a finite non-negative number.")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Importance for {name!r} must be a finite non-negative number.")
        weights[name] = weight
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Model feature importances must have a positive total.")
    return names, {name: weight / total for name, weight in weights.items()}


def _load_window(path: Path, feature_names: list[str], window_name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"{window_name} window is empty: {path}")
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(f"{window_name} input is missing model features: {preview}{suffix}")
    # Selecting the persisted model schema also drops labels, IDs, and extras.
    return frame.loc[:, feature_names].copy()


def prepare_drift_windows(
    baseline_path: Path,
    production_path: Path,
    model_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    bundle = joblib.load(model_path)
    feature_names, importance = _feature_schema(bundle)
    baseline = _load_window(baseline_path, feature_names, "Baseline")
    production = _load_window(production_path, feature_names, "Production")

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_out = output_dir / "baseline_window.parquet"
    production_out = output_dir / "production_window.parquet"
    importance_out = output_dir / "shap_importance.json"
    baseline.to_parquet(baseline_out, index=False)
    production.to_parquet(production_out, index=False)
    importance_out.write_text(json.dumps(importance, indent=2) + "\n", encoding="utf-8")
    return baseline_out, production_out, importance_out


def prepare_drift_windows_from_source(
    source_path: Path,
    model_path: Path,
    output_dir: Path,
    production_fraction: float = 0.2,
) -> tuple[Path, Path, Path]:
    """Create chronological historical windows from one feature dataset."""
    if not math.isfinite(production_fraction) or not 0.0 < production_fraction < 1.0:
        raise ValueError("production_fraction must be greater than 0 and less than 1.")

    bundle = joblib.load(model_path)
    feature_names, importance = _feature_schema(bundle)
    # TransactionDT is only used to order and split the historical rows; it is
    # deliberately excluded from the model feature windows below.
    source_columns = set(arrow_dataset.dataset(source_path, format="parquet").schema.names)
    if "TransactionDT" not in source_columns:
        raise ValueError(f"Source dataset must contain TransactionDT for a chronological split: {source_path}")
    missing = [name for name in feature_names if name not in source_columns]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(f"Source dataset is missing model features: {preview}{suffix}")
    frame = pd.read_parquet(source_path, columns=["TransactionDT", *feature_names])
    if frame["TransactionDT"].isna().any():
        raise ValueError("Source dataset contains missing TransactionDT values; cannot create time windows.")
    if len(frame) < 2:
        raise ValueError("Source dataset must contain at least two rows to create two windows.")

    frame = frame.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    production_rows = min(len(frame) - 1, max(1, math.ceil(len(frame) * production_fraction)))
    split_index = len(frame) - production_rows
    baseline = frame.iloc[:split_index].loc[:, feature_names].copy()
    production = frame.iloc[split_index:].loc[:, feature_names].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_out = output_dir / "baseline_window.parquet"
    production_out = output_dir / "production_window.parquet"
    importance_out = output_dir / "shap_importance.json"
    baseline.to_parquet(baseline_out, index=False)
    production.to_parquet(production_out, index=False)
    importance_out.write_text(json.dumps(importance, indent=2) + "\n", encoding="utf-8")
    return baseline_out, production_out, importance_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align baseline and production Parquet windows to a trained model schema."
    )
    parser.add_argument("--source", type=Path, help="One historical Parquet dataset to split by TransactionDT.")
    parser.add_argument("--baseline", type=Path, help="Known-good or representative baseline Parquet.")
    parser.add_argument("--production", type=Path, help="Recent scored production Parquet.")
    parser.add_argument(
        "--production-fraction",
        type=float,
        default=0.2,
        help="For --source, fraction of latest rows assigned to the production window (default: 0.2).",
    )
    parser.add_argument("--model", type=Path, default=Path("outputs/best-params.joblib"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source is not None:
        if args.baseline is not None or args.production is not None:
            raise ValueError("Use --source by itself, or provide both --baseline and --production.")
        paths = prepare_drift_windows_from_source(
            args.source, args.model, args.output_dir, args.production_fraction
        )
    else:
        if args.baseline is None or args.production is None:
            raise ValueError("Provide --source, or provide both --baseline and --production.")
        paths = prepare_drift_windows(args.baseline, args.production, args.model, args.output_dir)
    print("Prepared drift inputs:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

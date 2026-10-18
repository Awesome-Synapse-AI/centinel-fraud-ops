"""Run the local cost-weighted drift analysis from prepared Parquet windows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monitor.alerting import send_alert
from monitor.cost_drift import DAILY_DOLLAR_THRESHOLD, cost_weighted_drift, kl_per_feature
from config.env import load_repo_env


def run_drift_analysis(
    baseline_path: Path,
    production_path: Path,
    importance_path: Path,
    *,
    bins: int = 50,
    p_fraud: float = 0.013,
    c_fn: float = 420.0,
    c_fp: float = 4.0,
) -> pd.DataFrame:
    baseline = pd.read_parquet(baseline_path)
    production = pd.read_parquet(production_path)
    if baseline.empty:
        raise ValueError(f"Baseline window is empty: {baseline_path}")
    if production.empty:
        raise ValueError(f"Production window is empty: {production_path}")

    importance = json.loads(importance_path.read_text(encoding="utf-8"))
    if not isinstance(importance, dict):
        raise ValueError(f"Importance JSON must contain a feature-to-weight object: {importance_path}")

    kl = kl_per_feature(baseline, production, bins=bins)
    if not kl:
        raise ValueError("The windows have no shared numeric features to compare.")
    return cost_weighted_drift(kl, importance, p_fraud=p_fraud, c_fn=c_fn, c_fp=c_fp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank local baseline-to-production feature drift by expected cost.")
    parser.add_argument("--baseline", type=Path, default=Path("outputs/demo/baseline_window.parquet"))
    parser.add_argument("--production", type=Path, default=Path("outputs/demo/production_window.parquet"))
    parser.add_argument("--importance", type=Path, default=Path("outputs/demo/shap_importance.json"))
    parser.add_argument("--bins", type=int, default=50, help="Shared histogram bins per feature (default: 50).")
    parser.add_argument("--p-fraud", type=float, default=0.013, help="Estimated production fraud rate.")
    parser.add_argument("--c-fn", type=float, default=420.0, help="False-negative cost in dollars.")
    parser.add_argument("--c-fp", type=float, default=4.0, help="False-positive cost in dollars.")
    parser.add_argument("--top", type=int, default=10, help="Number of top features to print; use 0 for all.")
    parser.add_argument("--output-csv", type=Path, help="Optional path for saving the full ranked table.")
    parser.add_argument(
        "--send-alert",
        action="store_true",
        help="Persist and forward an alert when cumulative ranked cost reaches the configured threshold.",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=DAILY_DOLLAR_THRESHOLD,
        help=f"Cumulative cost threshold in dollars (default: {DAILY_DOLLAR_THRESHOLD:.0f}).",
    )
    return parser.parse_args()


def main() -> None:
    load_repo_env()
    args = parse_args()
    if args.top < 0:
        raise ValueError("--top must be zero or a positive integer.")
    if not math.isfinite(args.alert_threshold) or args.alert_threshold <= 0:
        raise ValueError("--alert-threshold must be a positive finite number.")
    ranked = run_drift_analysis(
        args.baseline,
        args.production,
        args.importance,
        bins=args.bins,
        p_fraud=args.p_fraud,
        c_fn=args.c_fn,
        c_fp=args.c_fp,
    )
    displayed = ranked if args.top == 0 else ranked.head(args.top)
    print(displayed.to_string(index=False))
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(args.output_csv, index=False)
        print(f"Full ranking saved to {args.output_csv}")

    if args.send_alert:
        crossed = ranked.index[ranked["cum_cost_drift"] >= args.alert_threshold]
        if len(crossed) == 0:
            print(f"No alert sent: cumulative ranked cost did not reach ${args.alert_threshold:,.2f}.")
            return

        # Include the feature that crosses the threshold in the persisted summary.
        selected_features = ranked.iloc[: int(crossed[0]) + 1]
        production = pd.read_parquet(args.production)
        expected_loss_per_transaction = args.p_fraud * args.c_fn + (1 - args.p_fraud) * args.c_fp
        total_expected_loss = len(production) * expected_loss_per_transaction
        payload = send_alert(selected_features, total_expected_loss)
        print(f"Alert persisted and forwarded when configured: {json.dumps(payload, sort_keys=True)}")


if __name__ == "__main__":
    main()

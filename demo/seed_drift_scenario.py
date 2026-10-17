import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def inject_drift(
    df: pd.DataFrame,
    feature: str,
    shift_std: float = 2.5,
    affected_frac: float = 0.15,
    seed: int = 7,
) -> pd.DataFrame:
    if feature not in df.columns:
        raise ValueError(f"Feature {feature!r} is not present in the baseline window.")
    if not pd.api.types.is_numeric_dtype(df[feature]) or pd.api.types.is_bool_dtype(df[feature]):
        raise ValueError(f"Feature {feature!r} must be numeric.")
    if not math.isfinite(shift_std):
        raise ValueError("shift_std must be finite.")
    if not math.isfinite(affected_frac) or not 0.0 <= affected_frac <= 1.0:
        raise ValueError("affected_frac must be between 0 and 1.")
    if df.empty:
        raise ValueError("Baseline window is empty.")

    rng = np.random.default_rng(seed)
    prod = df.copy()
    n_affected = int(len(prod) * affected_frac)
    if n_affected:
        idx = rng.choice(prod.index, size=n_affected, replace=False)
        prod.loc[idx, feature] += shift_std * prod[feature].std()
    return prod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed synthetic production drift.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature", default="merch_fraud_rate_30d")
    parser.add_argument("--shift-std", type=float, default=2.5, help="Shift size in standard deviations.")
    parser.add_argument("--affected-frac", type=float, default=0.15, help="Fraction of rows to shift.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatable row selection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = pd.read_parquet(args.baseline)
    prod = inject_drift(
        baseline,
        feature=args.feature,
        shift_std=args.shift_std,
        affected_frac=args.affected_frac,
        seed=args.seed,
    )
    prod.to_parquet(args.output, index=False)
    print(f"Seeded synthetic drift on {args.feature} and wrote {args.output}")


if __name__ == "__main__":
    main()

import argparse
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
    rng = np.random.default_rng(seed)
    prod = df.copy()
    n_affected = int(len(prod) * affected_frac)
    idx = rng.choice(prod.index, size=n_affected, replace=False)
    prod.loc[idx, feature] += shift_std * prod[feature].std()
    return prod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed synthetic production drift.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature", default="merch_fraud_rate_30d")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = pd.read_parquet(args.baseline)
    prod = inject_drift(baseline, feature=args.feature)
    prod.to_parquet(args.output, index=False)
    print(f"Seeded synthetic drift on {args.feature}")


if __name__ == "__main__":
    main()

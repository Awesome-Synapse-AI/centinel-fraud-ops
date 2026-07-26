import argparse
from pathlib import Path

import pandas as pd

from scripts.fraud_data_common import (
    DEFAULT_OUTPUT_ROOT,
    add_common_args,
    build_base_dataset,
    build_time_window_features,
    ensure_output_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local engineered features from IEEE-CIS training data."
    )
    add_common_args(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "staged" / "v1",
        help="Directory where engineered parquet outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    df = build_base_dataset(
        transaction_csv=args.transaction_csv,
        identity_csv=args.identity_csv,
        join_how=args.join_how,
        sample_rows=args.sample_rows,
    )
    featured_df = build_time_window_features(df)

    output_path = args.output_dir / "features.parquet"
    featured_df.to_parquet(output_path, index=False)

    profile = pd.DataFrame(
        {
            "rows": [len(featured_df)],
            "columns": [featured_df.shape[1]],
            "fraud_rate": [featured_df["isFraud"].mean()],
            "with_identity_rate": [featured_df["has_identity"].mean()],
        }
    )
    profile.to_csv(args.output_dir / "dataset_profile.csv", index=False)

    print(f"Wrote features to {output_path}")
    print(f"Rows: {len(featured_df):,}")
    print(f"Columns: {featured_df.shape[1]:,}")


if __name__ == "__main__":
    main()

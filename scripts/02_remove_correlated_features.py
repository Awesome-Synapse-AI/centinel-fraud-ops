import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_data_common import (
    add_common_args,
    build_base_dataset,
    ensure_output_paths,
    numeric_feature_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join the training CSVs, find highly correlated numeric feature pairs, "
            "remove one feature from each pair, and save the processed data."
        )
    )
    add_common_args(parser)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/02_no_high_corr.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports_feature_corr"))
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.95,
        help="Absolute Pearson correlation threshold for removing one feature from a pair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_paths(args.output_csv, args.reports_dir)

    df = build_base_dataset(args.transaction_csv, args.identity_csv, args.join_how, args.sample_rows)
    numeric_cols = numeric_feature_columns(df)

    corr_matrix = df[numeric_cols].corr(method="pearson").abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "correlation"})
    )
    high_pairs = high_pairs[high_pairs["correlation"] >= args.corr_threshold]
    high_pairs = high_pairs.sort_values("correlation", ascending=False).reset_index(drop=True)
    high_pairs.to_csv(args.reports_dir / "highly_correlated_feature_pairs.csv", index=False)

    label_corr = df[numeric_cols + ["isFraud"]].corr(method="pearson")["isFraud"].abs()
    label_corr = label_corr.drop("isFraud", errors="ignore").fillna(0)

    to_drop: set[str] = set()
    decisions = []
    for _, row in high_pairs.iterrows():
        feature_a = row["feature_a"]
        feature_b = row["feature_b"]
        if feature_a in to_drop or feature_b in to_drop:
            continue

        a_label_corr = label_corr.get(feature_a, 0)
        b_label_corr = label_corr.get(feature_b, 0)
        if a_label_corr >= b_label_corr:
            kept_feature, removed_feature = feature_a, feature_b
        else:
            kept_feature, removed_feature = feature_b, feature_a

        to_drop.add(removed_feature)
        decisions.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "correlation": row["correlation"],
                "kept_feature": kept_feature,
                "removed_feature": removed_feature,
                "kept_abs_corr_with_isFraud": label_corr.get(kept_feature, 0),
                "removed_abs_corr_with_isFraud": label_corr.get(removed_feature, 0),
            }
        )

    pd.DataFrame(decisions).to_csv(
        args.reports_dir / "correlated_feature_removal_decisions.csv", index=False
    )

    processed_df = df.drop(columns=sorted(to_drop))
    processed_df.to_csv(args.output_csv, index=False)

    print(f"Joined rows: {len(df):,}")
    print(f"High-correlation pairs: {len(high_pairs):,}")
    print(f"Removed features: {len(to_drop):,}")
    print(f"Kept columns: {processed_df.shape[1]:,}")
    print(f"Saved processed data: {args.output_csv}")
    print(f"Saved reports: {args.reports_dir}")


if __name__ == "__main__":
    main()

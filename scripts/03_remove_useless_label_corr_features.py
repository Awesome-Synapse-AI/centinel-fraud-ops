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
            "Join the training CSVs, check numeric feature correlation with isFraud, "
            "remove weakly correlated numeric features, and save the processed data."
        )
    )
    add_common_args(parser)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/03_label_corr_filtered.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports_label_corr"))
    parser.add_argument(
        "--label-corr-threshold",
        type=float,
        default=0.01,
        help=(
            "Minimum absolute Pearson correlation with isFraud required for numeric "
            "features. Numeric features below this value are removed."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_paths(args.output_csv, args.reports_dir)

    df = build_base_dataset(args.transaction_csv, args.identity_csv, args.join_how, args.sample_rows)
    numeric_cols = numeric_feature_columns(df)

    label_corr = df[numeric_cols + ["isFraud"]].corr(method="pearson")["isFraud"]
    label_corr = label_corr.drop("isFraud", errors="ignore")
    label_report = (
        pd.DataFrame(
            {
                "feature": label_corr.index,
                "correlation_with_isFraud": label_corr.values,
                "abs_correlation": label_corr.abs().fillna(0).values,
            }
        )
        .sort_values("abs_correlation", ascending=False)
        .reset_index(drop=True)
    )
    label_report["decision"] = np.where(
        label_report["abs_correlation"] < args.label_corr_threshold, "remove", "keep"
    )
    label_report.to_csv(args.reports_dir / "feature_label_correlations.csv", index=False)

    to_drop = label_report.loc[label_report["decision"] == "remove", "feature"].tolist()
    processed_df = df.drop(columns=to_drop)
    processed_df.to_csv(args.output_csv, index=False)

    print(f"Joined rows: {len(df):,}")
    print(f"Numeric features checked: {len(numeric_cols):,}")
    print(f"Removed weak-label-correlation features: {len(to_drop):,}")
    print(f"Kept columns: {processed_df.shape[1]:,}")
    print(f"Saved processed data: {args.output_csv}")
    print(f"Saved reports: {args.reports_dir}")


if __name__ == "__main__":
    main()

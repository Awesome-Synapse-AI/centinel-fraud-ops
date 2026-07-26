import argparse
from pathlib import Path

import pandas as pd

from fraud_data_common import add_common_args, build_base_dataset, ensure_output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join the training CSVs with the project defaults and report missing values."
    )
    add_common_args(parser)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/01_joined_train.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports_missing"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_paths(args.output_csv, args.reports_dir)

    df = build_base_dataset(args.transaction_csv, args.identity_csv, args.join_how, args.sample_rows)

    missing_count = df.isna().sum()
    missing_report = (
        pd.DataFrame(
            {
                "column": missing_count.index,
                "missing_rows": missing_count.values,
                "total_rows": len(df),
                "missing_percent": missing_count.values / len(df) * 100,
            }
        )
        .query("missing_rows > 0")
        .sort_values(["missing_rows", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )
    missing_report.to_csv(args.reports_dir / "missing_values_by_column.csv", index=False)

    by_label_frames = []
    for label_value, label_df in df.groupby("isFraud", dropna=False):
        label_missing = label_df.isna().sum()
        label_report = pd.DataFrame(
            {
                "isFraud": label_value,
                "column": label_missing.index,
                "missing_rows": label_missing.values,
                "group_rows": len(label_df),
                "missing_percent_in_group": label_missing.values / len(label_df) * 100,
            }
        )
        by_label_frames.append(label_report.query("missing_rows > 0"))

    missing_by_label = (
        pd.concat(by_label_frames, ignore_index=True)
        .sort_values(["column", "isFraud"])
        .reset_index(drop=True)
    )
    missing_by_label.to_csv(args.reports_dir / "missing_values_by_isFraud.csv", index=False)

    df.to_csv(args.output_csv, index=False)

    print(f"Joined rows: {len(df):,}")
    print(f"Joined columns: {df.shape[1]:,}")
    print(f"Columns with missing values: {len(missing_report):,}")
    print(f"Total missing cells: {int(missing_report['missing_rows'].sum()):,}")
    print(f"Saved joined data: {args.output_csv}")
    print(f"Saved reports: {args.reports_dir}")


if __name__ == "__main__":
    main()

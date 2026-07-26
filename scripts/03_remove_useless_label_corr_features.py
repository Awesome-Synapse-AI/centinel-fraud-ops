import argparse
from pathlib import Path

from pyspark.sql import types as T

from fraud_data_common import (
    add_common_args,
    build_base_dataset_spark,
    create_spark_session,
    dataframe_from_records,
    ensure_output_paths,
    label_correlations,
    numeric_feature_columns_spark,
    write_single_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join the training CSVs with Spark, check numeric feature correlation with isFraud, "
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

    spark = create_spark_session(
        app_name="fraud_remove_weak_label_corr",
        spark_master=args.spark_master,
        shuffle_partitions=args.spark_shuffle_partitions,
        driver_memory=args.spark_driver_memory,
    )

    try:
        df = build_base_dataset_spark(
            spark,
            args.transaction_csv,
            args.identity_csv,
            args.join_how,
            args.sample_rows,
        )
        numeric_cols = numeric_feature_columns_spark(df)
        correlations = label_correlations(df, numeric_cols)

        label_rows = [
            {
                "feature": feature,
                "correlation_with_isFraud": correlations[feature],
                "abs_correlation": correlations[feature],
                "decision": "remove"
                if correlations[feature] < args.label_corr_threshold
                else "keep",
            }
            for feature in numeric_cols
        ]
        label_rows.sort(key=lambda row: row["abs_correlation"], reverse=True)

        report_schema = T.StructType(
            [
                T.StructField("feature", T.StringType(), False),
                T.StructField("correlation_with_isFraud", T.DoubleType(), False),
                T.StructField("abs_correlation", T.DoubleType(), False),
                T.StructField("decision", T.StringType(), False),
            ]
        )
        report_df = dataframe_from_records(spark, label_rows, report_schema)
        write_single_csv(report_df, args.reports_dir / "feature_label_correlations.csv")

        to_drop = [row["feature"] for row in label_rows if row["decision"] == "remove"]
        processed_df = df.drop(*to_drop)
        write_single_csv(processed_df, args.output_csv)

        print(f"Joined rows: {df.count():,}")
        print(f"Numeric features checked: {len(numeric_cols):,}")
        print(f"Removed weak-label-correlation features: {len(to_drop):,}")
        print(f"Kept columns: {len(processed_df.columns):,}")
        print(f"Saved processed data: {args.output_csv}")
        print(f"Saved reports: {args.reports_dir}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

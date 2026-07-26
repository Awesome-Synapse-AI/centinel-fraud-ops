import argparse
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql import types as T

from fraud_data_common import (
    add_common_args,
    build_base_dataset_spark,
    correlation_matrix,
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
            "Join the training CSVs with Spark, find highly correlated numeric feature pairs, "
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

    spark = create_spark_session(
        app_name="fraud_remove_correlated_features",
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
        matrix = correlation_matrix(df, numeric_cols)
        label_corr = label_correlations(df, numeric_cols)

        high_pairs = []
        for i, feature_a in enumerate(numeric_cols):
            for j in range(i + 1, len(numeric_cols)):
                feature_b = numeric_cols[j]
                corr_value = abs(matrix[i][j])
                if corr_value >= args.corr_threshold:
                    high_pairs.append(
                        {
                            "feature_a": feature_a,
                            "feature_b": feature_b,
                            "correlation": corr_value,
                        }
                    )
        high_pairs.sort(key=lambda row: row["correlation"], reverse=True)

        high_pair_schema = T.StructType(
            [
                T.StructField("feature_a", T.StringType(), False),
                T.StructField("feature_b", T.StringType(), False),
                T.StructField("correlation", T.DoubleType(), False),
            ]
        )
        high_pairs_df = dataframe_from_records(spark, high_pairs, high_pair_schema)
        write_single_csv(high_pairs_df, args.reports_dir / "highly_correlated_feature_pairs.csv")

        to_drop: set[str] = set()
        decisions = []
        for row in high_pairs:
            feature_a = row["feature_a"]
            feature_b = row["feature_b"]
            if feature_a in to_drop or feature_b in to_drop:
                continue

            a_label_corr = label_corr.get(feature_a, 0.0)
            b_label_corr = label_corr.get(feature_b, 0.0)
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
                    "kept_abs_corr_with_isFraud": float(label_corr.get(kept_feature, 0.0)),
                    "removed_abs_corr_with_isFraud": float(label_corr.get(removed_feature, 0.0)),
                }
            )

        decision_schema = T.StructType(
            [
                T.StructField("feature_a", T.StringType(), False),
                T.StructField("feature_b", T.StringType(), False),
                T.StructField("correlation", T.DoubleType(), False),
                T.StructField("kept_feature", T.StringType(), False),
                T.StructField("removed_feature", T.StringType(), False),
                T.StructField("kept_abs_corr_with_isFraud", T.DoubleType(), False),
                T.StructField("removed_abs_corr_with_isFraud", T.DoubleType(), False),
            ]
        )
        decisions_df = dataframe_from_records(spark, decisions, decision_schema)
        write_single_csv(decisions_df, args.reports_dir / "correlated_feature_removal_decisions.csv")

        processed_df = df.drop(*sorted(to_drop))
        write_single_csv(processed_df, args.output_csv)

        print(f"Joined rows: {df.count():,}")
        print(f"High-correlation pairs: {len(high_pairs):,}")
        print(f"Removed features: {len(to_drop):,}")
        print(f"Kept columns: {len(processed_df.columns):,}")
        print(f"Saved processed data: {args.output_csv}")
        print(f"Saved reports: {args.reports_dir}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

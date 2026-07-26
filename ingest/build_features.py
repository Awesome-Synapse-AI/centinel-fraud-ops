import argparse
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql import types as T

from scripts.fraud_data_common import (
    DEFAULT_OUTPUT_ROOT,
    TRANSACTION_START_SQL,
    add_common_args,
    build_base_dataset_spark,
    build_time_window_features_spark,
    create_spark_session,
    dataframe_from_records,
    ensure_output_dir,
    write_single_csv,
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
    parser.add_argument(
        "--preprocessed-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV produced by scripts/04_run_preprocessing_pipeline.py. "
            "When provided, ingestion uses this audited dataset instead of re-reading raw CSVs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    spark = create_spark_session(
        app_name="fraud_build_features",
        spark_master=args.spark_master,
        shuffle_partitions=args.spark_shuffle_partitions,
        driver_memory=args.spark_driver_memory,
    )

    try:
        if args.preprocessed_csv is not None:
            df = spark.read.option("header", True).option("inferSchema", True).csv(str(args.preprocessed_csv))
            if "ts" not in df.columns and "TransactionDT" in df.columns:
                df = df.withColumn(
                    "ts",
                    F.expr(
                        f"timestamp('{TRANSACTION_START_SQL}') + make_interval(0, 0, 0, 0, 0, 0, TransactionDT)"
                    ),
                )
            elif "ts" in df.columns:
                df = df.withColumn("ts", F.to_timestamp("ts"))
            if "transaction_date" not in df.columns and "ts" in df.columns:
                df = df.withColumn("transaction_date", F.to_date("ts").cast("string"))
            if "has_identity" not in df.columns:
                has_identity = (
                    F.when(F.col("DeviceType").isNotNull(), F.lit(1)).otherwise(F.lit(0))
                    if "DeviceType" in df.columns
                    else F.lit(0)
                )
                df = df.withColumn("has_identity", has_identity)
        else:
            df = build_base_dataset_spark(
                spark,
                transaction_csv=args.transaction_csv,
                identity_csv=args.identity_csv,
                join_how=args.join_how,
                sample_rows=args.sample_rows,
            )

        featured_df = build_time_window_features_spark(df)
        output_path = args.output_dir / "features.parquet"
        featured_df.write.mode("overwrite").parquet(str(output_path))

        profile_row = (
            featured_df.agg(
                F.count(F.lit(1)).alias("rows"),
                F.lit(len(featured_df.columns)).alias("columns"),
                F.avg(F.col("isFraud").cast("double")).alias("fraud_rate"),
                F.avg(F.col("has_identity").cast("double")).alias("with_identity_rate"),
            )
            .collect()[0]
            .asDict()
        )
        profile_schema = T.StructType(
            [
                T.StructField("rows", T.LongType(), False),
                T.StructField("columns", T.IntegerType(), False),
                T.StructField("fraud_rate", T.DoubleType(), False),
                T.StructField("with_identity_rate", T.DoubleType(), False),
            ]
        )
        profile_df = dataframe_from_records(spark, [profile_row], profile_schema)
        write_single_csv(profile_df, args.output_dir / "dataset_profile.csv")

        print(f"Wrote features to {output_path}")
        print(f"Rows: {profile_row['rows']:,}")
        print(f"Columns: {profile_row['columns']:,}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

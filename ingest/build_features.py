import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.env import load_repo_env

os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["hadoop.home.dir"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark import StorageLevel
from pyspark.sql import functions as F
from pyspark.sql import types as T
from google.cloud import bigquery

from scripts.fraud_data_common import (
    DEFAULT_OUTPUT_ROOT,
    TRANSACTION_START_SQL,
    add_common_args,
    build_base_dataset_spark,
    build_time_window_features_spark,
    cache_df,
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
    parser.add_argument(
        "--bigquery-table",
        type=str,
        default=os.environ.get("FEAST_BIGQUERY_TABLE"),
        help=(
            "Optional fully qualified BigQuery table name to load the parquet into, "
            "for example your-project.fraud_features.features_v1."
        ),
    )
    return parser.parse_args()


def resolve_preprocessed_csv_path(path: Path) -> Path:
    candidates = [path]
    if path.suffix.lower() != ".csv":
        candidates.append(path.with_suffix(".csv"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Preprocessed CSV not found. Checked: {searched}"
    )


def load_parquet_to_bigquery(parquet_path: Path, table_ref: str) -> None:
    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    pdf = pd.read_parquet(parquet_path)
    job = client.load_table_from_dataframe(pdf, table_ref, job_config=job_config)
    job.result()


def main() -> None:
    load_repo_env()
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
            preprocessed_csv = resolve_preprocessed_csv_path(args.preprocessed_csv)
            df = (
                spark.read.option("header", True)
                .option("inferSchema", True)
                .csv(str(preprocessed_csv))
            )
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

        # build_time_window_features_spark() reads df twice internally (once
        # to build the narrow window-feature projection, once for the final
        # join back to the full dataset). Without caching, both raw-CSV
        # reads/joins (or the preprocessed-CSV read) happen twice.
        df = cache_df(df)

        featured_df = build_time_window_features_spark(df).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        output_path = args.output_dir / "features.parquet"
        featured_df.write.mode("overwrite").parquet(str(output_path))
        if args.bigquery_table:
            load_parquet_to_bigquery(output_path, args.bigquery_table)

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
        if args.bigquery_table:
            print(f"Loaded features into BigQuery table {args.bigquery_table}")
        print(f"Rows: {profile_row['rows']:,}")
        print(f"Columns: {profile_row['columns']:,}")
        featured_df.unpersist()
        df.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

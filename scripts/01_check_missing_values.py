import argparse
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql import types as T

from fraud_data_common import (
    add_common_args,
    build_base_dataset_spark,
    create_spark_session,
    dataframe_from_records,
    ensure_output_paths,
    write_single_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join the training CSVs with Spark and report missing values."
    )
    add_common_args(parser)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/01_joined_train.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports_missing"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_paths(args.output_csv, args.reports_dir)

    spark = create_spark_session(
        app_name="fraud_missing_values",
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
        total_rows = df.count()

        missing_exprs = [
            F.sum(
                F.when(F.col(column).isNull() | F.isnan(F.col(column).cast("double")), 1).otherwise(0)
            ).alias(column)
            for column in df.columns
        ]
        missing_row = df.agg(*missing_exprs).collect()[0].asDict()
        missing_records = [
            {
                "column": column,
                "missing_rows": int(missing_rows),
                "total_rows": int(total_rows),
                "missing_percent": float(missing_rows) / total_rows * 100 if total_rows else 0.0,
            }
            for column, missing_rows in missing_row.items()
            if int(missing_rows) > 0
        ]
        missing_records.sort(key=lambda row: (-row["missing_rows"], row["column"]))

        missing_schema = T.StructType(
            [
                T.StructField("column", T.StringType(), False),
                T.StructField("missing_rows", T.LongType(), False),
                T.StructField("total_rows", T.LongType(), False),
                T.StructField("missing_percent", T.DoubleType(), False),
            ]
        )
        missing_report_df = dataframe_from_records(spark, missing_records, missing_schema)
        write_single_csv(missing_report_df, args.reports_dir / "missing_values_by_column.csv")

        label_rows = []
        for label_value in [0, 1]:
            label_df = df.filter(F.col("isFraud") == label_value)
            group_rows = label_df.count()
            if group_rows == 0:
                continue
            label_missing = label_df.agg(*missing_exprs).collect()[0].asDict()
            for column, missing_rows in label_missing.items():
                if int(missing_rows) <= 0:
                    continue
                label_rows.append(
                    {
                        "isFraud": int(label_value),
                        "column": column,
                        "missing_rows": int(missing_rows),
                        "group_rows": int(group_rows),
                        "missing_percent_in_group": float(missing_rows) / group_rows * 100,
                    }
                )

        label_rows.sort(key=lambda row: (row["column"], row["isFraud"]))
        label_schema = T.StructType(
            [
                T.StructField("isFraud", T.IntegerType(), False),
                T.StructField("column", T.StringType(), False),
                T.StructField("missing_rows", T.LongType(), False),
                T.StructField("group_rows", T.LongType(), False),
                T.StructField("missing_percent_in_group", T.DoubleType(), False),
            ]
        )
        missing_by_label_df = dataframe_from_records(spark, label_rows, label_schema)
        write_single_csv(missing_by_label_df, args.reports_dir / "missing_values_by_isFraud.csv")

        write_single_csv(df, args.output_csv)

        print(f"Joined rows: {total_rows:,}")
        print(f"Joined columns: {len(df.columns):,}")
        print(f"Columns with missing values: {len(missing_records):,}")
        print(f"Total missing cells: {sum(row['missing_rows'] for row in missing_records):,}")
        print(f"Saved joined data: {args.output_csv}")
        print(f"Saved reports: {args.reports_dir}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

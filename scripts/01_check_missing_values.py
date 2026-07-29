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
    is_missing_expr,
    write_single_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join the training CSVs with Spark, report missing values, and clean them."
    )
    add_common_args(parser)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/01_joined_train.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports_missing"))
    parser.add_argument(
        "--drop-column-missing-threshold",
        type=float,
        default=95.0,
        help="Drop columns whose missing-value percentage is above this threshold.",
    )
    parser.add_argument(
        "--drop-row-missing-threshold",
        type=float,
        default=5.0,
        help="Drop rows with any missing values when the row-level missing rate is at or below this threshold.",
    )
    parser.add_argument(
        "--numeric-fill-floor",
        type=float,
        default=-999.0,
        help="Fallback numeric fill value; the script uses min(this value, feature_min - 1).",
    )
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
            F.sum(F.when(is_missing_expr(df, column), 1).otherwise(0)).alias(column)
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

        columns_to_drop = [
            row["column"]
            for row in missing_records
            if row["missing_percent"] > args.drop_column_missing_threshold
        ]
        working_df = df.drop(*columns_to_drop) if columns_to_drop else df

        label_rows = []
        for label_value in [0, 1]:
            label_df = working_df.filter(F.col("isFraud") == label_value)
            group_rows = label_df.count()
            if group_rows == 0:
                continue
            label_missing_exprs = [
                F.sum(F.when(is_missing_expr(working_df, column), 1).otherwise(0)).alias(column)
                for column in working_df.columns
            ]
            label_missing = label_df.agg(*label_missing_exprs).collect()[0].asDict()
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

        row_has_missing = None
        for column in working_df.columns:
            condition = is_missing_expr(working_df, column)
            row_has_missing = condition if row_has_missing is None else (row_has_missing | condition)

        rows_with_missing = working_df.filter(row_has_missing).count() if row_has_missing is not None else 0
        row_missing_percent = (rows_with_missing / total_rows * 100) if total_rows else 0.0

        if row_missing_percent <= args.drop_row_missing_threshold:
            cleaned_df = working_df.filter(~row_has_missing) if row_has_missing is not None else working_df
            cleaning_strategy = "drop_rows"
        else:
            numeric_types = (
                T.ByteType,
                T.ShortType,
                T.IntegerType,
                T.LongType,
                T.FloatType,
                T.DoubleType,
                T.DecimalType,
            )
            fill_map = {}
            for field in working_df.schema.fields:
                if isinstance(field.dataType, numeric_types):
                    min_value = working_df.select(F.min(F.col(field.name).cast("double"))).first()[0]
                    if min_value is None:
                        fill_map[field.name] = float(args.numeric_fill_floor)
                    else:
                        fill_map[field.name] = float(min(args.numeric_fill_floor, min_value - 1.0))
                else:
                    fill_map[field.name] = "__MISSING__"
            cleaned_df = working_df.fillna(fill_map)
            cleaning_strategy = "fillna_sentinel"

        decision_rows = [
            {
                "step": "drop_sparse_columns",
                "threshold_percent": float(args.drop_column_missing_threshold),
                "affected_count": len(columns_to_drop),
                "detail": ",".join(columns_to_drop) if columns_to_drop else "",
            },
            {
                "step": "row_missing_policy",
                "threshold_percent": float(args.drop_row_missing_threshold),
                "affected_count": int(rows_with_missing),
                "detail": cleaning_strategy,
            },
        ]
        decision_schema = T.StructType(
            [
                T.StructField("step", T.StringType(), False),
                T.StructField("threshold_percent", T.DoubleType(), False),
                T.StructField("affected_count", T.LongType(), False),
                T.StructField("detail", T.StringType(), False),
            ]
        )
        decision_df = dataframe_from_records(spark, decision_rows, decision_schema)
        write_single_csv(decision_df, args.reports_dir / "missing_value_cleaning_decisions.csv")

        write_single_csv(cleaned_df, args.output_csv)

        print(f"Joined rows: {total_rows:,}")
        print(f"Joined columns: {len(df.columns):,}")
        print(f"Columns with missing values: {len(missing_records):,}")
        print(f"Total missing cells: {sum(row['missing_rows'] for row in missing_records):,}")
        print(f"Dropped sparse columns: {len(columns_to_drop):,}")
        print(f"Rows with missing values after column drop: {rows_with_missing:,} ({row_missing_percent:.2f}%)")
        print(f"Missing-value strategy: {cleaning_strategy}")
        print(f"Saved joined data: {args.output_csv}")
        print(f"Saved reports: {args.reports_dir}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

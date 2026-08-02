import argparse
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql import types as T

from fraud_data_common import (
    add_common_args,
    build_base_dataset_spark,
    cache_df,
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
        # df feeds many actions below (aggs, counts, the final write); cache it
        # once so each action doesn't re-read and re-join the raw CSVs.
        df = cache_df(df)

        missing_exprs = [F.count(F.lit(1)).alias("__total_rows__")] + [
            F.sum(F.when(is_missing_expr(df, column), 1).otherwise(0)).alias(column)
            for column in df.columns
        ]
        missing_row = df.agg(*missing_exprs).collect()[0].asDict()
        total_rows = int(missing_row.pop("__total_rows__"))
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
        # working_df feeds the per-label report, the row-missing filter, the
        # cleaning step, and the final write below — cache it once too.
        working_df = cache_df(working_df)

        non_label_columns = [c for c in working_df.columns if c != "isFraud"]
        label_agg_exprs = [F.count(F.lit(1)).alias("__group_rows__")] + [
            F.sum(F.when(is_missing_expr(working_df, column), 1).otherwise(0)).alias(column)
            for column in non_label_columns
        ]
        # One groupBy pass computes missing counts for every label value at
        # once, instead of filtering + aggregating separately per label.
        # Restrict to isFraud in (0, 1) to match the original script's
        # explicit "for label_value in [0, 1]" behavior.
        label_grouped_rows = (
            working_df.filter(F.col("isFraud").isin(0, 1))
            .groupBy("isFraud")
            .agg(*label_agg_exprs)
            .collect()
        )

        label_rows = []
        for grouped_row in label_grouped_rows:
            row_dict = grouped_row.asDict()
            label_value = row_dict.pop("isFraud")
            group_rows = int(row_dict.pop("__group_rows__"))
            if group_rows == 0:
                continue
            for column, missing_rows in row_dict.items():
                if missing_rows is None or int(missing_rows) <= 0:
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
            numeric_field_names = [
                field.name for field in working_df.schema.fields
                if isinstance(field.dataType, numeric_types)
            ]
            non_numeric_field_names = [
                field.name for field in working_df.schema.fields
                if field.name not in numeric_field_names
            ]
            # Single agg() computes the min for every numeric column in one
            # pass, instead of one .select(F.min(...)).first() job per column.
            min_row = (
                working_df.agg(
                    *[F.min(F.col(name).cast("double")).alias(name) for name in numeric_field_names]
                ).collect()[0].asDict()
                if numeric_field_names
                else {}
            )
            fill_map = {}
            for name in numeric_field_names:
                min_value = min_row.get(name)
                if min_value is None:
                    fill_map[name] = float(args.numeric_fill_floor)
                else:
                    fill_map[name] = float(min(args.numeric_fill_floor, min_value - 1.0))
            for name in non_numeric_field_names:
                fill_map[name] = "__MISSING__"
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
        working_df.unpersist()
        df.unpersist()

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
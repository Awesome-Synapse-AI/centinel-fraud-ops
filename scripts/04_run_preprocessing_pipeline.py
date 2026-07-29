import argparse
import os
import sys
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, functions as F
from pyspark.sql import types as T

from fraud_data_common import (
    DEFAULT_OUTPUT_ROOT,
    add_common_args,
    build_base_dataset_spark,
    correlation_matrix,
    create_spark_session,
    dataframe_from_records,
    ensure_output_dir,
    is_missing_expr,
    label_correlations,
    numeric_feature_columns_spark,
    write_single_csv,
)

os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["hadoop.home.dir"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local preprocessing pipeline with cached intermediate outputs: "
            "missing-value cleaning, high-correlation pruning, weak label-correlation pruning, "
            "and final numeric train/test dataset preparation."
        )
    )
    add_common_args(parser)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "preprocessing",
        help="Root directory for stage outputs and reports.",
    )
    parser.add_argument(
        "--test-transaction-csv",
        type=Path,
        default=Path("../dataset/test_transaction.csv"),
        help="Test transaction CSV used for final schema-aligned test-set preparation.",
    )
    parser.add_argument(
        "--test-identity-csv",
        type=Path,
        default=Path("../dataset/test_identity.csv"),
        help="Test identity CSV used for final schema-aligned test-set preparation.",
    )
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
        help="Fallback numeric fill value; step 1 uses min(this value, feature_min - 1).",
    )
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--label-corr-threshold", type=float, default=0.01)
    return parser.parse_args()


def _read_csv(spark, path: Path) -> DataFrame:
    return spark.read.option("header", True).option("inferSchema", True).csv(str(path))


def _write_stage_if_missing(path: Path, builder) -> None:
    if path.exists():
        print(f"Skipping existing stage output: {path}")
        return
    builder()


def _sanitize_feature_suffix(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_") or "value"


def run_step1(args: argparse.Namespace, spark, output_csv: Path, reports_dir: Path) -> None:
    ensure_output_dir(reports_dir)
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
    write_single_csv(
        dataframe_from_records(spark, missing_records, missing_schema),
        reports_dir / "missing_values_by_column.csv",
    )

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
    write_single_csv(
        dataframe_from_records(spark, label_rows, label_schema),
        reports_dir / "missing_values_by_isFraud.csv",
    )

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
                fill_map[field.name] = (
                    float(args.numeric_fill_floor)
                    if min_value is None
                    else float(min(args.numeric_fill_floor, min_value - 1.0))
                )
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
    write_single_csv(
        dataframe_from_records(spark, decision_rows, decision_schema),
        reports_dir / "missing_value_cleaning_decisions.csv",
    )
    write_single_csv(cleaned_df, output_csv)


def run_step2(args: argparse.Namespace, spark, input_csv: Path, output_csv: Path, reports_dir: Path) -> None:
    ensure_output_dir(reports_dir)
    df = _read_csv(spark, input_csv)
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

    pair_schema = T.StructType(
        [
            T.StructField("feature_a", T.StringType(), False),
            T.StructField("feature_b", T.StringType(), False),
            T.StructField("correlation", T.DoubleType(), False),
        ]
    )
    write_single_csv(
        dataframe_from_records(spark, high_pairs, pair_schema),
        reports_dir / "highly_correlated_feature_pairs.csv",
    )

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
    write_single_csv(
        dataframe_from_records(spark, decisions, decision_schema),
        reports_dir / "correlated_feature_removal_decisions.csv",
    )
    write_single_csv(df.drop(*sorted(to_drop)), output_csv)


def run_step3(args: argparse.Namespace, spark, input_csv: Path, output_csv: Path, reports_dir: Path) -> None:
    ensure_output_dir(reports_dir)
    df = _read_csv(spark, input_csv)
    numeric_cols = numeric_feature_columns_spark(df)
    correlations = label_correlations(df, numeric_cols)

    label_rows = [
        {
            "feature": feature,
            "correlation_with_isFraud": correlations[feature],
            "abs_correlation": correlations[feature],
            "decision": "remove" if correlations[feature] < args.label_corr_threshold else "keep",
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
    write_single_csv(
        dataframe_from_records(spark, label_rows, report_schema),
        reports_dir / "feature_label_correlations.csv",
    )

    to_drop = [row["feature"] for row in label_rows if row["decision"] == "remove"]
    write_single_csv(df.drop(*to_drop), output_csv)


def _build_test_base_dataset(
    spark,
    transaction_csv: Path,
    identity_csv: Path,
    join_how: str,
    sample_rows: int | None,
) -> DataFrame:
    transaction_df = _read_csv(spark, transaction_csv)
    identity_df = _read_csv(spark, identity_csv)
    if sample_rows is not None:
        transaction_df = transaction_df.limit(sample_rows)
        identity_df = identity_df.limit(sample_rows)

    joined_df = transaction_df.join(identity_df, on="TransactionID", how=join_how)
    has_identity_expr = (
        F.when(F.col("DeviceType").isNotNull(), F.lit(1)).otherwise(F.lit(0))
        if "DeviceType" in joined_df.columns
        else F.lit(0)
    )
    return (
        joined_df.withColumn("has_identity", has_identity_expr)
        .withColumn("TransactionDT", F.col("TransactionDT").cast("double"))
        .withColumn(
            "ts",
            F.from_unixtime(
                F.unix_timestamp(F.lit("2017-12-01 00:00:00")) + F.col("TransactionDT").cast("long")
            ).cast("timestamp"),
        )
        .withColumn("transaction_date", F.to_date("ts").cast("string"))
    )


def prepare_final_datasets(
    args: argparse.Namespace,
    spark,
    stage3_csv: Path,
    output_root: Path,
) -> None:
    final_dir = output_root / "prepared"
    ensure_output_dir(final_dir)

    train_stage3 = _read_csv(spark, stage3_csv)
    numeric_types = (
        T.ByteType,
        T.ShortType,
        T.IntegerType,
        T.LongType,
        T.FloatType,
        T.DoubleType,
        T.DecimalType,
    )

    numeric_feature_columns = [
        field.name
        for field in train_stage3.schema.fields
        if isinstance(field.dataType, numeric_types) and field.name != "isFraud"
    ]
    categorical_feature_columns = [
        field.name
        for field in train_stage3.schema.fields
        if isinstance(field.dataType, T.StringType) and field.name not in {"ts", "transaction_date"}
    ]

    train_model_input = train_stage3
    test_base = _build_test_base_dataset(
        spark,
        args.test_transaction_csv,
        args.test_identity_csv,
        args.join_how,
        args.sample_rows,
    )
    test_model_input = test_base

    if categorical_feature_columns:
        train_model_input = train_model_input.fillna("__MISSING__", subset=categorical_feature_columns)
        test_fill_columns = [column for column in categorical_feature_columns if column in test_model_input.columns]
        if test_fill_columns:
            test_model_input = test_model_input.fillna("__MISSING__", subset=test_fill_columns)

        index_columns = [f"{column}__idx" for column in categorical_feature_columns]
        encoded_columns = [f"{column}__ohe" for column in categorical_feature_columns]
        pipeline = Pipeline(
            stages=[
                StringIndexer(
                    inputCol=column,
                    outputCol=index_column,
                    handleInvalid="keep",
                )
                for column, index_column in zip(categorical_feature_columns, index_columns)
            ] + [
                OneHotEncoder(
                    inputCols=index_columns,
                    outputCols=encoded_columns,
                    handleInvalid="keep",
                )
            ]
        )
        pipeline_model = pipeline.fit(train_model_input)
        encoded_train = pipeline_model.transform(train_model_input)
        encoded_test = pipeline_model.transform(test_model_input)

        for column, encoded_column in zip(categorical_feature_columns, encoded_columns):
            category_values = (
                encoded_train.select(column)
                .where(F.col(column).isNotNull())
                .distinct()
                .orderBy(column)
                .rdd.flatMap(lambda row: row)
                .collect()
            )
            # StringIndexer keeps an extra bucket for unseen/null values, which we do not name explicitly.
            used_output_names: set[str] = set()
            for idx, category in enumerate(category_values):
                safe_value = _sanitize_feature_suffix(str(category))
                output_name = f"{column}__{safe_value}"
                dedupe_idx = 2
                while output_name in used_output_names:
                    output_name = f"{column}__{safe_value}_{dedupe_idx}"
                    dedupe_idx += 1
                used_output_names.add(output_name)
                encoded_train = encoded_train.withColumn(
                    output_name,
                    vector_to_array(F.col(encoded_column))[idx],
                )
                encoded_test = encoded_test.withColumn(
                    output_name,
                    vector_to_array(F.col(encoded_column))[idx],
                )
            if len(category_values) == 1:
                output_name = next(iter(used_output_names))
                encoded_train = encoded_train.withColumn(output_name, F.lit(0.0))
                encoded_test = encoded_test.withColumn(output_name, F.lit(0.0))
        train_stage3 = encoded_train
        test_base = encoded_test

    encoded_categorical_feature_columns = sorted(
        column
        for column in train_stage3.columns
        if "__" in column and any(column.startswith(f"{source}__") for source in categorical_feature_columns)
    )
    feature_columns = [*numeric_feature_columns, *encoded_categorical_feature_columns]
    ordered_train_columns = ["isFraud", *feature_columns]

    prepared_train = train_stage3.select(
        *[F.col("isFraud").cast("int").alias("isFraud")],
        *[F.col(column).cast("double").alias(column) for column in feature_columns],
    ).fillna(0.0, subset=feature_columns)
    write_single_csv(prepared_train, final_dir / "train_prepared.csv")

    available_test_columns = [column for column in feature_columns if column in test_base.columns]
    missing_test_columns = [column for column in feature_columns if column not in test_base.columns]
    prepared_test = test_base.select(
        *[F.col(column).cast("double").alias(column) for column in available_test_columns]
    )
    for column in missing_test_columns:
        prepared_test = prepared_test.withColumn(column, F.lit(0.0))
    prepared_test = prepared_test.select(*ordered_train_columns[1:]).fillna(0.0)
    write_single_csv(prepared_test, final_dir / "test_prepared.csv")

    summary_rows = [
        {
            "dataset": "train",
            "rows": prepared_train.count(),
            "columns": len(ordered_train_columns),
            "dropped_non_numeric_or_high_cardinality": len(train_stage3.columns) - len(ordered_train_columns),
        },
        {
            "dataset": "test",
            "rows": prepared_test.count(),
            "columns": len(feature_columns),
            "dropped_non_numeric_or_high_cardinality": len(test_base.columns) - len(available_test_columns),
        },
    ]
    summary_schema = T.StructType(
        [
            T.StructField("dataset", T.StringType(), False),
            T.StructField("rows", T.LongType(), False),
            T.StructField("columns", T.IntegerType(), False),
            T.StructField("dropped_non_numeric_or_high_cardinality", T.IntegerType(), False),
        ]
    )
    write_single_csv(
        dataframe_from_records(spark, summary_rows, summary_schema),
        final_dir / "prepared_dataset_summary.csv",
    )


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_root)

    step1_output = args.output_root / "01_joined_train.csv"
    step2_output = args.output_root / "02_no_high_corr.csv"
    step3_output = args.output_root / "03_label_corr_filtered.csv"

    spark = create_spark_session(
        app_name="fraud_preprocessing_pipeline",
        spark_master=args.spark_master,
        shuffle_partitions=args.spark_shuffle_partitions,
        driver_memory=args.spark_driver_memory,
    )

    try:
        _write_stage_if_missing(
            step1_output,
            lambda: run_step1(args, spark, step1_output, args.output_root / "reports_missing"),
        )
        _write_stage_if_missing(
            step2_output,
            lambda: run_step2(args, spark, step1_output, step2_output, args.output_root / "reports_feature_corr"),
        )
        _write_stage_if_missing(
            step3_output,
            lambda: run_step3(args, spark, step2_output, step3_output, args.output_root / "reports_label_corr"),
        )
        prepare_final_datasets(args, spark, step3_output, args.output_root)
    finally:
        spark.stop()

    print(f"Completed preprocessing pipeline under {args.output_root}")


if __name__ == "__main__":
    main()

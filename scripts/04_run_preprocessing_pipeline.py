import argparse
import json
import os
import sys
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame, functions as F
from pyspark.sql import types as T

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.model_schema import CATEGORICAL_RAW_FEATURE_NAMES, PREPROCESS_REQUIRED_COLUMNS

from fraud_data_common import (
    DEFAULT_OUTPUT_ROOT,
    add_common_args,
    build_base_dataset_spark,
    cache_df,
    correlation_matrix_with_label,
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
        default=REPO_ROOT / "dataset" / "test_transaction.csv",
        help="Test transaction CSV used for final schema-aligned test-set preparation.",
    )
    parser.add_argument(
        "--test-identity-csv",
        type=Path,
        default=REPO_ROOT / "dataset" / "test_identity.csv",
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


def _normalize_column_names(df: DataFrame) -> DataFrame:
    normalized_df = df
    for column in df.columns:
        normalized_column = column.replace("-", "_")
        if normalized_column != column:
            normalized_df = normalized_df.withColumnRenamed(column, normalized_column)
    return normalized_df


def run_step1(args: argparse.Namespace, spark, output_csv: Path, reports_dir: Path) -> None:
    ensure_output_dir(reports_dir)
    df = build_base_dataset_spark(
        spark,
        args.transaction_csv,
        args.identity_csv,
        args.join_how,
        args.sample_rows,
    )
    # df feeds many actions below; cache it once instead of re-reading and
    # re-joining the raw CSVs for each count()/agg()/write().
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
    write_single_csv(
        dataframe_from_records(spark, missing_records, missing_schema),
        reports_dir / "missing_values_by_column.csv",
    )

    columns_to_drop = [
        row["column"]
        for row in missing_records
        if row["missing_percent"] > args.drop_column_missing_threshold
        and row["column"] not in PREPROCESS_REQUIRED_COLUMNS
    ]
    working_df = df.drop(*columns_to_drop) if columns_to_drop else df
    # working_df feeds the per-label report, the row-missing filter, the
    # cleaning step, and the final write below.
    working_df = cache_df(working_df)

    non_label_columns = [c for c in working_df.columns if c != "isFraud"]
    label_agg_exprs = [F.count(F.lit(1)).alias("__group_rows__")] + [
        F.sum(F.when(is_missing_expr(working_df, column), 1).otherwise(0)).alias(column)
        for column in non_label_columns
    ]
    # One groupBy pass instead of filtering + aggregating separately per label.
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
    write_single_csv(
        dataframe_from_records(spark, label_rows, label_schema),
        reports_dir / "missing_values_by_isFraud.csv",
    )

    missing_checks = [is_missing_expr(working_df, column) for column in working_df.columns]
    row_has_missing = F.array_contains(F.array(*missing_checks), F.lit(True)) if missing_checks else None
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
        # Single agg() computes every numeric column's min in one pass,
        # instead of one .select(F.min(...)).first() job per column.
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
            fill_map[name] = (
                float(args.numeric_fill_floor)
                if min_value is None
                else float(min(args.numeric_fill_floor, min_value - 1.0))
            )
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
    write_single_csv(
        dataframe_from_records(spark, decision_rows, decision_schema),
        reports_dir / "missing_value_cleaning_decisions.csv",
    )
    write_single_csv(cleaned_df, output_csv)
    working_df.unpersist()
    df.unpersist()


def run_step2(args: argparse.Namespace, spark, input_csv: Path, output_csv: Path, reports_dir: Path) -> None:
    ensure_output_dir(reports_dir)
    df = cache_df(_read_csv(spark, input_csv))
    numeric_cols = numeric_feature_columns_spark(df)
    matrix, label_corr = correlation_matrix_with_label(df, numeric_cols)

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
        if feature_a in PREPROCESS_REQUIRED_COLUMNS and feature_b in PREPROCESS_REQUIRED_COLUMNS:
            continue
        a_label_corr = label_corr.get(feature_a, 0.0)
        b_label_corr = label_corr.get(feature_b, 0.0)
        if feature_a in PREPROCESS_REQUIRED_COLUMNS:
            kept_feature, removed_feature = feature_a, feature_b
        elif feature_b in PREPROCESS_REQUIRED_COLUMNS:
            kept_feature, removed_feature = feature_b, feature_a
        elif a_label_corr >= b_label_corr:
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
    df.unpersist()


def run_step3(args: argparse.Namespace, spark, input_csv: Path, output_csv: Path, reports_dir: Path) -> None:
    ensure_output_dir(reports_dir)
    df = cache_df(_read_csv(spark, input_csv))
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

    to_drop = [
        row["feature"]
        for row in label_rows
        if row["decision"] == "remove" and row["feature"] not in PREPROCESS_REQUIRED_COLUMNS
    ]
    write_single_csv(df.drop(*to_drop), output_csv)
    df.unpersist()


def _build_test_base_dataset(
    spark,
    transaction_csv: Path,
    identity_csv: Path,
    join_how: str,
    sample_rows: int | None,
) -> DataFrame:
    transaction_df = _normalize_column_names(_read_csv(spark, transaction_csv))
    identity_df = _normalize_column_names(_read_csv(spark, identity_csv))
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
    categorical_feature_columns = list(CATEGORICAL_RAW_FEATURE_NAMES)
    missing_categorical = set(categorical_feature_columns).difference(train_stage3.columns)
    if missing_categorical:
        raise ValueError(
            "Preprocessing output is missing categorical columns required by the model contract: "
            f"{sorted(missing_categorical)}"
        )

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
        for column in categorical_feature_columns:
            if column not in test_model_input.columns:
                test_model_input = test_model_input.withColumn(column, F.lit("__MISSING__"))
        test_fill_columns = [column for column in categorical_feature_columns if column in test_model_input.columns]
        if test_fill_columns:
            test_model_input = test_model_input.fillna("__MISSING__", subset=test_fill_columns)

        encoded_categorical_feature_columns = [f"{column}__encoded" for column in categorical_feature_columns]
        pipeline = Pipeline(
            stages=[
                StringIndexer(
                    inputCol=column,
                    outputCol=encoded_column,
                    handleInvalid="keep",
                )
                for column, encoded_column in zip(categorical_feature_columns, encoded_categorical_feature_columns)
            ]
        )
        pipeline_model = pipeline.fit(train_model_input)
        categorical_mappings = {
            column: {
                "values": [str(value) for value in stage.labels],
                "unknown_value": len(stage.labels),
            }
            for column, stage in zip(categorical_feature_columns, pipeline_model.stages)
        }
        (final_dir / "categorical_mappings.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "unknown_category": "__UNKNOWN__",
                    "mappings": categorical_mappings,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        train_stage3 = pipeline_model.transform(train_model_input)
        test_base = pipeline_model.transform(test_model_input)
    else:
        encoded_categorical_feature_columns = []

    feature_columns = [*numeric_feature_columns, *encoded_categorical_feature_columns]
    ordered_train_columns = ["isFraud", *feature_columns]

    prepared_train = train_stage3.select(
        *[F.col("isFraud").cast("int").alias("isFraud")],
        *[F.col(column).cast("double").alias(column) for column in feature_columns],
    ).fillna(0.0, subset=feature_columns)
    # Both write_single_csv() and the .count() below are actions on the same
    # chain (read -> join -> StringIndexer transform -> select); cache once
    # so the second action reuses the materialized result instead of
    # recomputing the whole chain from the raw CSVs again.
    prepared_train = cache_df(prepared_train)
    write_single_csv(prepared_train, final_dir / "train_prepared.csv")

    available_test_columns = [column for column in feature_columns if column in test_base.columns]
    missing_test_columns = [column for column in feature_columns if column not in test_base.columns]
    prepared_test = test_base.select(
        *[F.col(column).cast("double").alias(column) for column in available_test_columns]
    )
    for column in missing_test_columns:
        prepared_test = prepared_test.withColumn(column, F.lit(0.0))
    prepared_test = prepared_test.select(*ordered_train_columns[1:]).fillna(0.0)
    prepared_test = cache_df(prepared_test)
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
    prepared_train.unpersist()
    prepared_test.unpersist()


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

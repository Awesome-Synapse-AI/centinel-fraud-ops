import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.env import load_repo_env
from config.category_encoders import category_mapping_sha256, load_category_mappings
from config.model_schema import (
    CATEGORICAL_RAW_FEATURE_NAMES,
    INTEGER_MODEL_FEATURES,
    MODEL_FEATURE_NAMES,
)
from train.preflight import (
    validate_bigquery_feature_table,
    validate_feature_columns,
    verify_historical_feature_sample,
)

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
        "--feature-repo",
        type=Path,
        default=REPO_ROOT / "feature_repo",
        help="Feast repository used by --feast-sample-check.",
    )
    parser.add_argument(
        "--bigquery-table",
        type=str,
        default=None,
        help=(
            "Optional fully qualified BigQuery table name to load the parquet into, "
            "for example your-project.fraud_features.features_v1."
        ),
    )
    parser.add_argument(
        "--categorical-mappings",
        type=Path,
        default=None,
        help="JSON mappings emitted by preprocessing (default: sibling prepared/categorical_mappings.json).",
    )
    parser.add_argument(
        "--replace-bigquery-table",
        action="store_true",
        help="Explicitly allow replacing an existing BigQuery table. Without this, loads require an empty/new table.",
    )
    parser.add_argument(
        "--feast-sample-check",
        action="store_true",
        help="After upload, retrieve a small historical-feature sample through Feast before training.",
    )
    parser.add_argument("--feast-sample-rows", type=int, default=32)
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


def load_parquet_to_bigquery(parquet_path: Path, table_ref: str, replace_existing: bool = False) -> None:
    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(
        write_disposition=(
            bigquery.WriteDisposition.WRITE_TRUNCATE
            if replace_existing
            else bigquery.WriteDisposition.WRITE_EMPTY
        ),
        schema=[
            bigquery.SchemaField("TransactionID", "INT64"),
            bigquery.SchemaField("card1", "INT64"),
            bigquery.SchemaField("ts", "TIMESTAMP"),
            bigquery.SchemaField("TransactionDT", "FLOAT64"),
            bigquery.SchemaField("isFraud", "INT64"),
            bigquery.SchemaField("category_mapping_sha256", "STRING"),
            *[
                bigquery.SchemaField(name, "INT64" if name in INTEGER_MODEL_FEATURES else "FLOAT64")
                for name in MODEL_FEATURE_NAMES
            ],
        ],
    )
    pdf = pd.read_parquet(parquet_path)
    validate_feature_columns(
        {name: str(dtype) for name, dtype in pdf.dtypes.items()},
        f"Parquet file {parquet_path}",
    )
    if "ts" in pdf.columns:
        pdf["ts"] = pd.to_datetime(pdf["ts"], utc=True)
    job = client.load_table_from_dataframe(pdf, table_ref, job_config=job_config)
    job.result()


def apply_categorical_mappings(df, mappings: dict):
    missing = sorted(set(CATEGORICAL_RAW_FEATURE_NAMES).difference(df.columns))
    if missing:
        raise ValueError(f"Input data is missing categorical source columns: {missing}")
    encoded = df
    for column in CATEGORICAL_RAW_FEATURE_NAMES:
        item = mappings[column]
        values = item["values"]
        unknown_value = int(item["unknown_value"])
        pairs = []
        for index, value in enumerate(values):
            pairs.extend((F.lit(str(value)), F.lit(index)))
        category_map = F.create_map(*pairs) if pairs else F.create_map(F.lit("__NO_KNOWN_VALUE__"), F.lit(unknown_value))
        category = F.coalesce(F.col(column).cast("string"), F.lit("__MISSING__"))
        encoded = encoded.withColumn(
            f"{column}__encoded",
            F.coalesce(F.element_at(category_map, category), F.lit(unknown_value)).cast("double"),
        )
    return encoded


def model_input_projection(df, category_hash: str):
    missing = sorted(set(MODEL_FEATURE_NAMES).difference(df.columns))
    if missing:
        raise ValueError(
            "Generated feature table does not match the Feast model-input contract. "
            f"Missing features: {missing}"
        )
    projection = [
        F.col("TransactionID").cast("long").alias("TransactionID"),
        F.col("card1").cast("long").alias("card1"),
        F.col("ts").cast("timestamp").alias("ts"),
        F.col("TransactionDT").cast("double").alias("TransactionDT"),
        F.col("isFraud").cast("long").alias("isFraud"),
    ]
    projection.extend(
        F.col(name).cast("long" if name in INTEGER_MODEL_FEATURES else "double").alias(name)
        for name in MODEL_FEATURE_NAMES
    )
    result = df.select(*projection).fillna(0).withColumn(
        "category_mapping_sha256", F.lit(category_hash)
    )
    schema_types = {field.name: type(field.dataType).__name__ for field in result.schema.fields}
    validate_feature_columns(schema_types, "generated Spark feature table")
    return result


def run_feast_sample_check(table_ref: str, feature_repo: Path, parquet_path: Path, sample_rows: int) -> int:
    from feast import FeatureStore
    old_table = os.environ.get("FEAST_BIGQUERY_TABLE")
    os.environ["FEAST_BIGQUERY_TABLE"] = table_ref
    try:
        with tempfile.TemporaryDirectory(prefix="feast_handoff_") as temporary:
            temporary_repo = Path(temporary) / "feature_repo"
            temporary_repo.mkdir(parents=True)
            config_text = (feature_repo / "feature_store.yaml").read_text(encoding="utf-8")
            config_text = config_text.replace(
                "gs://fraud-cost-pipeline-artifacts/registry/registry.db",
                "data/registry/registry.db",
            )
            (temporary_repo / "feature_store.yaml").write_text(config_text, encoding="utf-8")
            (temporary_repo / "data" / "registry").mkdir(parents=True, exist_ok=True)
            feature_code = (feature_repo / "features.py").read_text(encoding="utf-8")
            feature_code = feature_code.replace("online=True", "online=False")
            (temporary_repo / "features.py").write_text(feature_code, encoding="utf-8")

            module_name = f"handoff_features_{uuid.uuid4().hex}"
            spec = importlib.util.spec_from_file_location(module_name, temporary_repo / "features.py")
            if spec is None or spec.loader is None:
                raise RuntimeError("Could not load temporary Feast feature definitions.")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
                store = FeatureStore(repo_path=str(temporary_repo))
                store.apply(
                    [module.card, module.tx_fv, module.merch_fv, module.model_input_fv, module.fraud_v1]
                )
                service = store.get_feature_service("fraud_v1")
                feature_refs = [
                    f"{projection.name}:{field.name}"
                    for projection in service.feature_view_projections
                    for field in projection.features
                ]
                metadata = pd.read_parquet(parquet_path, columns=["category_mapping_sha256"])
                hashes = metadata["category_mapping_sha256"].dropna().unique().tolist()
                if len(hashes) != 1:
                    raise ValueError("Parquet must contain exactly one categorical mapping fingerprint.")
                validate_bigquery_feature_table(
                    table_ref, feature_refs, expected_mapping_sha256=hashes[0]
                )
                entities = pd.read_parquet(
                    parquet_path,
                    columns=["card1", "ts", "isFraud", "TransactionDT"],
                )
                entities = entities.dropna().head(sample_rows).copy()
                if entities.empty:
                    raise ValueError("Generated Parquet has no valid entity rows for the Feast sample check.")
                entities["ts"] = pd.to_datetime(entities["ts"], utc=True)
                fetched = store.get_historical_features(entity_df=entities, features=feature_refs).to_df()
                validate_feature_columns(
                    {name: str(dtype) for name, dtype in fetched.dtypes.items()},
                    f"Feast historical sample from {table_ref}",
                )
                if fetched.empty:
                    raise ValueError(f"Feast returned no historical rows from {table_ref}.")
                return len(fetched)
            finally:
                sys.modules.pop(module_name, None)
    finally:
        if old_table is None:
            os.environ.pop("FEAST_BIGQUERY_TABLE", None)
        else:
            os.environ["FEAST_BIGQUERY_TABLE"] = old_table


def main() -> None:
    load_repo_env()
    args = parse_args()
    ensure_output_dir(args.output_dir)

    if args.feast_sample_check and not args.bigquery_table:
        raise ValueError("--feast-sample-check requires --bigquery-table.")
    if args.feast_sample_rows < 1:
        raise ValueError("--feast-sample-rows must be at least one.")

    if args.categorical_mappings is not None:
        mappings_path = args.categorical_mappings
    elif args.preprocessed_csv is not None:
        mappings_path = args.preprocessed_csv.parent / "prepared" / "categorical_mappings.json"
    else:
        mappings_path = REPO_ROOT / "outputs" / "preprocessing" / "prepared" / "categorical_mappings.json"
    categorical_mappings = load_category_mappings(mappings_path)
    category_hash = category_mapping_sha256(categorical_mappings)

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
            if args.sample_rows is not None:
                df = df.limit(args.sample_rows)
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
        df = apply_categorical_mappings(df, categorical_mappings)
        df = cache_df(df)

        featured_df = model_input_projection(
            build_time_window_features_spark(df), category_hash
        ).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        output_path = args.output_dir / "features.parquet"
        featured_df.write.mode("overwrite").parquet(str(output_path))
        if args.bigquery_table:
            load_parquet_to_bigquery(
                output_path,
                args.bigquery_table,
                replace_existing=args.replace_bigquery_table,
            )
            print(f"Loaded features into BigQuery table {args.bigquery_table}")
            if args.feast_sample_check:
                fetched_rows = run_feast_sample_check(
                    args.bigquery_table,
                    args.feature_repo,
                    output_path,
                    args.feast_sample_rows,
                )
                print(f"Feast historical handoff sample succeeded: {fetched_rows} rows")

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
        featured_df.unpersist()
        df.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

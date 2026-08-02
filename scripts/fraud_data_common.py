import argparse
import math
import shutil, sys
import tempfile
from pathlib import Path

from pyspark import StorageLevel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


DEFAULT_TRANSACTION_PATH = Path("../dataset/train_transaction.csv")
DEFAULT_IDENTITY_PATH = Path("../dataset/train_identity.csv")
DEFAULT_OUTPUT_ROOT = Path("../outputs")
TRANSACTION_START_SQL = "2017-12-01 00:00:00"
# IEEE-CIS uses TransactionDT as seconds since an arbitrary reference point,
# not as a real wall-clock timestamp. We derive a synthetic event timestamp
# from a fixed anchor only so windowed feature logic has a stable ordering axis.
CATEGORICAL_COLUMNS = {
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    *{f"M{i}" for i in range(1, 10)},
    *{f"id_{i}" for i in range(12, 39)},
}
TIMEDELTA_COLUMNS = {"TransactionDT", *{f"D{i}" for i in range(1, 16)}}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--transaction-csv", type=Path, default=DEFAULT_TRANSACTION_PATH)
    parser.add_argument("--identity-csv", type=Path, default=DEFAULT_IDENTITY_PATH)
    parser.add_argument(
        "--join-how",
        choices=["left", "inner", "outer"],
        default="left",
        help="How to join transaction rows to identity rows on TransactionID.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional row limit for faster test runs. Omit to process the full dataset.",
    )
    parser.add_argument(
        "--spark-master",
        default="local[*]",
        help="Spark master URL. Defaults to local Spark using all cores.",
    )
    parser.add_argument(
        "--spark-shuffle-partitions",
        type=int,
        default=8,
        help="Spark shuffle partitions for local preprocessing jobs.",
    )
    parser.add_argument(
        "--spark-driver-memory",
        default="6g",
        help="Driver memory to request for local Spark jobs.",
    )


def cache_df(df: DataFrame) -> DataFrame:
    """Persist a DataFrame that will be touched by multiple actions.

    Without this, every .count()/.collect()/.agg()/.write() on a DataFrame
    built from build_base_dataset_spark() re-triggers the full CSV read,
    join, and column derivation from scratch, since Spark DataFrames are
    lazy and nothing is cached by default. Scripts that call multiple
    actions on the same df (almost all of them) should persist it once
    right after it's built, and call df.unpersist() when done with it.
    """
    return df.persist(StorageLevel.MEMORY_AND_DISK)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_output_paths(output_csv: Path, reports_dir: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)


def create_spark_session(
    app_name: str,
    spark_master: str = "local[*]",
    shuffle_partitions: int = 8,
    driver_memory: str = "6g",
) -> SparkSession:
    return (
        SparkSession.builder
            .appName(app_name)
            .master(spark_master)
            .config(
                "spark.sql.shuffle.partitions",
                str(shuffle_partitions),
            )
            .config(
                "spark.driver.memory",
                driver_memory,
            )
            .config(
                "spark.pyspark.python",
                sys.executable,
            )
            .config(
                "spark.pyspark.driver.python",
                sys.executable,
            )
            .getOrCreate()
    )


def read_and_join_spark(
    spark: SparkSession,
    transaction_csv: Path,
    identity_csv: Path,
    join_how: str,
    sample_rows: int | None,
) -> DataFrame:
    transaction_df = spark.read.option("header", True).option("inferSchema", True).csv(
        str(transaction_csv)
    )
    identity_df = spark.read.option("header", True).option("inferSchema", True).csv(str(identity_csv))
    if sample_rows is not None:
        transaction_df = transaction_df.limit(sample_rows)
        identity_df = identity_df.limit(sample_rows)

    joined_df = transaction_df.join(identity_df, on="TransactionID", how=join_how)
    if "isFraud" not in joined_df.columns:
        raise ValueError("Joined data must contain the label column 'isFraud'.")
    return joined_df


def build_base_dataset_spark(
    spark: SparkSession,
    transaction_csv: Path,
    identity_csv: Path,
    join_how: str = "left",
    sample_rows: int | None = None,
) -> DataFrame:
    df = read_and_join_spark(spark, transaction_csv, identity_csv, join_how, sample_rows)
    has_identity_expr = (
        F.when(F.col("DeviceType").isNotNull(), F.lit(1)).otherwise(F.lit(0))
        if "DeviceType" in df.columns
        else F.lit(0)
    )
    if "TransactionDT" not in df.columns:
        raise ValueError("Expected TransactionDT column in transaction data.")
    return (
        df.withColumn("has_identity", has_identity_expr)
        .withColumn("TransactionDT", F.col("TransactionDT").cast("double"))
        .withColumn(
            "ts",
            F.from_unixtime(
                F.unix_timestamp(F.lit(TRANSACTION_START_SQL))
                + F.col("TransactionDT").cast("long")
            ).cast("timestamp"),
        )
        .withColumn("transaction_date", F.to_date("ts").cast("string"))
    )


def numeric_feature_columns_spark(df: DataFrame) -> list[str]:
    numeric_types = (
        T.ByteType,
        T.ShortType,
        T.IntegerType,
        T.LongType,
        T.FloatType,
        T.DoubleType,
        T.DecimalType,
    )
    excluded = {"TransactionID", "isFraud", "has_identity"}
    return [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, numeric_types)
        and field.name not in excluded
        and field.name not in CATEGORICAL_COLUMNS
    ]


def is_missing_expr(df: DataFrame, column: str):
    field = df.schema[column]
    numeric_types = (
        T.ByteType,
        T.ShortType,
        T.IntegerType,
        T.LongType,
        T.FloatType,
        T.DoubleType,
        T.DecimalType,
    )
    base = F.col(column).isNull()
    if isinstance(field.dataType, numeric_types):
        return base | F.isnan(F.col(column).cast("double"))
    return base


def build_time_window_features_spark(df: DataFrame) -> DataFrame:
    from pyspark.sql import Window

    required = {"card1", "TransactionAmt", "TransactionDT", "ts"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Expected columns missing for time-window features: {sorted(missing)}")

    if "TransactionID" not in df.columns:
        raise ValueError("Expected TransactionID column for feature joins.")

    product_column = None
    if "ProductCD" in df.columns:
        product_column = "ProductCD"
    elif "ProductCD__encoded" in df.columns:
        product_column = "ProductCD__encoded"

    ts_seconds = F.col("ts").cast("long")
    by_card_24h = Window.partitionBy("card1").orderBy(ts_seconds).rangeBetween(-86400, 0)

    # Compute window features on a narrow projection first to reduce shuffle width,
    # then join the derived columns back to the full dataset.
    window_select_cols = ["TransactionID", "card1", "TransactionAmt", "isFraud", "ts"]
    if product_column is not None:
        window_select_cols.insert(2, product_column)

    window_input = df.select(*window_select_cols)
    derived_features = window_input.withColumn("txn_count_24h", F.count(F.lit(1)).over(by_card_24h)).withColumn(
        "amt_sum_24h", F.sum("TransactionAmt").over(by_card_24h)
    )

    if product_column is not None:
        by_product_30d = Window.partitionBy(product_column).orderBy(ts_seconds).rangeBetween(-30 * 86400, -1)
        derived_features = (
            derived_features
            .withColumn("distinct_productcd_24h", F.approx_count_distinct(product_column).over(by_card_24h))
            .withColumn("merch_fraud_rate_30d", F.avg("isFraud").over(by_product_30d))
            .withColumn("merch_volume_30d", F.count(F.lit(1)).over(by_product_30d))
        )
    else:
        derived_features = (
            derived_features
            .withColumn("distinct_productcd_24h", F.lit(0))
            .withColumn("merch_fraud_rate_30d", F.lit(0.0))
            .withColumn("merch_volume_30d", F.lit(0))
        )

    derived_features = derived_features.select(
        "TransactionID",
        "txn_count_24h",
        "amt_sum_24h",
        "distinct_productcd_24h",
        "merch_fraud_rate_30d",
        "merch_volume_30d",
    )
    featured = df.join(derived_features, on="TransactionID", how="left")

    return featured.fillna(
        {
            "txn_count_24h": 0,
            "amt_sum_24h": 0.0,
            "distinct_productcd_24h": 0,
            "merch_fraud_rate_30d": 0.0,
            "merch_volume_30d": 0,
        }
    )


def write_single_csv(df: DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="spark_csv_", dir=str(output_csv.parent)))
    try:
        df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(temp_dir))
        part_files = list(temp_dir.glob("part-*.csv"))
        if not part_files:
            raise FileNotFoundError(f"No Spark CSV part file found in {temp_dir}")
        if output_csv.exists():
            output_csv.unlink()
        shutil.move(str(part_files[0]), str(output_csv))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def dataframe_from_records(spark: SparkSession, rows: list[dict], schema: T.StructType) -> DataFrame:
    return spark.createDataFrame(rows, schema=schema)


def safe_abs_correlation(value: float | None) -> float:
    if value is None or math.isnan(value):
        return 0.0
    return abs(float(value))


def _pearson_matrix(df: DataFrame, columns: list[str]):
    """Single Spark job computing the full Pearson correlation matrix for `columns`."""
    vector_df = (
        df.select(*[F.col(column).cast("double").alias(column) for column in columns])
        .na.fill(0.0)
    )
    assembler = VectorAssembler(inputCols=columns, outputCol="features", handleInvalid="keep")
    matrix_row = Correlation.corr(assembler.transform(vector_df).select("features"), "features", "pearson").head()
    return matrix_row[0]


def label_correlations(df: DataFrame, numeric_cols: list[str]) -> dict[str, float]:
    """Absolute Pearson correlation of each numeric column with isFraud.

    Computed as one Spark job over a single assembled correlation matrix
    (numeric_cols + isFraud), instead of calling df.stat.corr() once per
    column, which previously triggered one full-table scan per feature.
    """
    if not numeric_cols:
        return {}
    matrix = _pearson_matrix(df, [*numeric_cols, "isFraud"])
    label_idx = len(numeric_cols)
    return {
        column: safe_abs_correlation(float(matrix[i, label_idx]))
        for i, column in enumerate(numeric_cols)
    }


def correlation_matrix(df: DataFrame, numeric_cols: list[str]) -> list[list[float]]:
    if not numeric_cols:
        return []
    matrix = _pearson_matrix(df, numeric_cols)
    size = len(numeric_cols)
    return [[float(matrix[i, j]) for j in range(size)] for i in range(size)]


def correlation_matrix_with_label(
    df: DataFrame, numeric_cols: list[str]
) -> tuple[list[list[float]], dict[str, float]]:
    """Pairwise correlation matrix AND correlation-with-label in a single pass.

    Callers that need both correlation_matrix(...) and label_correlations(...)
    on the same numeric_cols (e.g. the high-correlation pruning step) should
    use this instead of calling both separately, since that would otherwise
    assemble the feature vector and run Correlation.corr twice.
    """
    if not numeric_cols:
        return [], {}
    matrix = _pearson_matrix(df, [*numeric_cols, "isFraud"])
    size = len(numeric_cols)
    pairwise = [[float(matrix[i, j]) for j in range(size)] for i in range(size)]
    label_corr = {
        column: safe_abs_correlation(float(matrix[i, size]))
        for i, column in enumerate(numeric_cols)
    }
    return pairwise, label_corr
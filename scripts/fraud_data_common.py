import argparse
import math
import shutil
import tempfile
from pathlib import Path

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


DEFAULT_TRANSACTION_PATH = Path("dataset/train_transaction.csv")
DEFAULT_IDENTITY_PATH = Path("dataset/train_identity.csv")
DEFAULT_OUTPUT_ROOT = Path("outputs")
TRANSACTION_START_SQL = "2017-12-01 00:00:00"


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
        SparkSession.builder.appName(app_name)
        .master(spark_master)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.memory", driver_memory)
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
    return (
        df.withColumn("has_identity", has_identity_expr)
        .withColumn(
            "ts",
            F.expr(f"timestamp('{TRANSACTION_START_SQL}') + make_interval(0, 0, 0, 0, 0, 0, TransactionDT)"),
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
    excluded = {"TransactionID", "isFraud"}
    return [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, numeric_types) and field.name not in excluded
    ]


def build_time_window_features_spark(df: DataFrame) -> DataFrame:
    from pyspark.sql import Window

    required = {"card1", "ProductCD", "TransactionAmt", "TransactionDT", "ts"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Expected columns missing for time-window features: {sorted(missing)}")

    ts_seconds = F.col("ts").cast("long")
    by_card_24h = Window.partitionBy("card1").orderBy(ts_seconds).rangeBetween(-86400, 0)
    by_product_30d = Window.partitionBy("ProductCD").orderBy(ts_seconds).rangeBetween(-30 * 86400, -1)

    featured = (
        df.withColumn("txn_count_24h", F.count(F.lit(1)).over(by_card_24h))
        .withColumn("amt_sum_24h", F.sum("TransactionAmt").over(by_card_24h))
        .withColumn("distinct_productcd_24h", F.approx_count_distinct("ProductCD").over(by_card_24h))
        .withColumn("merch_fraud_rate_30d", F.avg("isFraud").over(by_product_30d))
        .withColumn("merch_volume_30d", F.count(F.lit(1)).over(by_product_30d))
    )

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


def label_correlations(df: DataFrame, numeric_cols: list[str]) -> dict[str, float]:
    correlations: dict[str, float] = {}
    cleaned = df.select(
        *[F.col(column).cast("double").alias(column) for column in numeric_cols],
        F.col("isFraud").cast("double").alias("isFraud"),
    ).na.drop(subset=["isFraud"])
    for column in numeric_cols:
        corr_value = cleaned.stat.corr(column, "isFraud")
        correlations[column] = safe_abs_correlation(corr_value)
    return correlations


def correlation_matrix(df: DataFrame, numeric_cols: list[str]) -> list[list[float]]:
    if not numeric_cols:
        return []
    vector_df = (
        df.select(*[F.col(column).cast("double").alias(column) for column in numeric_cols])
        .na.fill(0.0)
    )
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features", handleInvalid="keep")
    matrix_row = Correlation.corr(assembler.transform(vector_df).select("features"), "features", "pearson").head()
    matrix = matrix_row[0]
    size = len(numeric_cols)
    return [[float(matrix[i, j]) for j in range(size)] for i in range(size)]

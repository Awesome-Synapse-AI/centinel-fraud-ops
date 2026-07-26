import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TRANSACTION_PATH = Path("dataset/train_transaction.csv")
DEFAULT_IDENTITY_PATH = Path("dataset/train_identity.csv")
DEFAULT_OUTPUT_ROOT = Path("outputs")
TRANSACTION_START = pd.Timestamp("2017-12-01")


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


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_and_join(
    transaction_csv: Path,
    identity_csv: Path,
    join_how: str,
    sample_rows: int | None,
) -> pd.DataFrame:
    transaction_df = pd.read_csv(transaction_csv, nrows=sample_rows, low_memory=False)
    identity_df = pd.read_csv(identity_csv, nrows=sample_rows, low_memory=False)
    joined_df = transaction_df.merge(identity_df, on="TransactionID", how=join_how)
    if "isFraud" not in joined_df.columns:
        raise ValueError("Joined data must contain the label column 'isFraud'.")
    return joined_df


def build_base_dataset(
    transaction_csv: Path,
    identity_csv: Path,
    join_how: str = "left",
    sample_rows: int | None = None,
) -> pd.DataFrame:
    df = read_and_join(transaction_csv, identity_csv, join_how, sample_rows)
    df["has_identity"] = df["DeviceType"].notna().astype(int) if "DeviceType" in df.columns else 0
    if "TransactionDT" in df.columns:
        df["ts"] = TRANSACTION_START + pd.to_timedelta(df["TransactionDT"], unit="s")
        df["transaction_date"] = df["ts"].dt.date.astype(str)
    return df


def build_time_window_features(df: pd.DataFrame) -> pd.DataFrame:
    if "ts" not in df.columns:
        raise ValueError("Expected timestamp column 'ts'. Run build_base_dataset first.")
    if "card1" not in df.columns or "ProductCD" not in df.columns or "TransactionAmt" not in df.columns:
        raise ValueError(
            "Expected card1, ProductCD, and TransactionAmt columns for time-window features."
        )

    featured = df.sort_values(["card1", "ts", "TransactionID"]).copy()
    featured = featured.set_index("ts")
    featured["txn_count_24h"] = (
        featured.groupby("card1")["TransactionAmt"].rolling("1D").count().reset_index(level=0, drop=True)
    )
    featured["amt_sum_24h"] = (
        featured.groupby("card1")["TransactionAmt"].rolling("1D").sum().reset_index(level=0, drop=True)
    )
    featured["distinct_productcd_24h"] = (
        featured.groupby("card1")["ProductCD"]
        .rolling("1D")
        .apply(lambda s: s.nunique(), raw=False)
        .reset_index(level=0, drop=True)
    )

    def trailing_product_stats(group: pd.DataFrame) -> pd.DataFrame:
        shifted_label = group["isFraud"].shift(1)
        group["merch_fraud_rate_30d"] = shifted_label.rolling("30D").mean()
        group["merch_volume_30d"] = shifted_label.rolling("30D").count()
        return group

    featured = featured.groupby("ProductCD", group_keys=False).apply(trailing_product_stats)
    featured = featured.reset_index()
    feature_cols = [
        "txn_count_24h",
        "amt_sum_24h",
        "distinct_productcd_24h",
        "merch_fraud_rate_30d",
        "merch_volume_30d",
    ]
    featured[feature_cols] = featured[feature_cols].fillna(0)
    return featured


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in {"TransactionID", "isFraud"}
    ]


def ensure_output_paths(output_csv: Path, reports_dir: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_feature_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> DatasetSplit:
    if "TransactionDT" not in df.columns:
        raise ValueError("Expected TransactionDT column for time-based splitting.")

    ordered = df.sort_values("TransactionDT").reset_index(drop=True)
    train_end = int(len(ordered) * train_frac)
    val_end = train_end + int(len(ordered) * val_frac)
    return DatasetSplit(
        train=ordered.iloc[:train_end].copy(),
        val=ordered.iloc[train_end:val_end].copy(),
        test=ordered.iloc[val_end:].copy(),
    )


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target = df["isFraud"].astype(int)
    features = df.drop(columns=["isFraud", "TransactionDT", "TransactionID"], errors="ignore")
    features = features.select_dtypes(include=["number", "bool"]).fillna(0)
    return features, target

"""Build test-split rows using the selected deployment bundle's 105-feature schema.

Inputs are the train_prepared.csv and test_prepared.csv outputs from
scripts/04_run_preprocessing_pipeline.py. The five rolling features are
recomputed over the combined chronological train/test stream. Fraud-rate
history uses labeled training rows only; test labels are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import Counter, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


WINDOW_FEATURES = [
    "txn_count_24h",
    "amt_sum_24h",
    "distinct_productcd_24h",
    "merch_fraud_rate_30d",
    "merch_volume_30d",
]
CARD_WINDOW_SECONDS = 24 * 60 * 60
PRODUCT_WINDOW_SECONDS = 30 * 24 * 60 * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-prepared", type=Path, default=Path("outputs/preprocessing/prepared/train_prepared.csv"))
    parser.add_argument("--test-prepared", type=Path, default=Path("outputs/preprocessing/prepared/test_prepared.csv"))
    parser.add_argument("--model", type=Path, default=Path("outputs/best-params.joblib"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/deployment-test"))
    parser.add_argument("--request-sample-size", type=int, default=20)
    return parser.parse_args()


def read_events(path: Path, is_test: bool) -> pd.DataFrame:
    columns = ["TransactionDT", "TransactionAmt", "card1", "ProductCD__encoded"]
    if not is_test:
        columns.append("isFraud")
    frame = pd.read_csv(path, usecols=columns)
    frame["row_index"] = np.arange(len(frame), dtype=np.int64) if is_test else -1
    frame["is_test"] = is_test
    if is_test:
        frame["isFraud"] = np.nan
    return frame


def rolling_features(train_path: Path, test_path: Path) -> np.ndarray:
    train = read_events(train_path, is_test=False)
    test = read_events(test_path, is_test=True)
    test_count = len(test)
    events = pd.concat([train, test], ignore_index=True)
    del train, test
    events.sort_values("TransactionDT", kind="mergesort", inplace=True, ignore_index=True)

    ts_values = events["TransactionDT"].to_numpy(dtype=np.int64)
    amount_values = events["TransactionAmt"].to_numpy(dtype=np.float64)
    card_values = events["card1"].to_numpy(dtype=np.int64)
    product_values = events["ProductCD__encoded"].to_numpy(dtype=np.int64)
    test_flags = events["is_test"].to_numpy(dtype=bool)
    row_indices = events["row_index"].to_numpy(dtype=np.int64)
    labels = events["isFraud"].to_numpy(dtype=np.float64)
    del events

    # Card state: events and product counts in the trailing 24 hours.
    card_state: dict[int, list] = {}
    # Product state: all prior volume and labeled-train fraud counts in 30 days.
    product_state: dict[int, list] = {}
    output = np.zeros((test_count, len(WINDOW_FEATURES)), dtype=np.float64)

    start = 0
    event_count = len(ts_values)
    while start < event_count:
        timestamp = int(ts_values[start])
        end = start + 1
        while end < event_count and ts_values[end] == timestamp:
            end += 1

        group: list[tuple[int, int, float, int, bool, int, float]] = []
        additions_by_card: dict[int, list[tuple[float, int]]] = {}
        additions_by_product: dict[int, list[float]] = {}
        for i in range(start, end):
            card = int(card_values[i])
            product = int(product_values[i])
            amount = float(amount_values[i])
            is_test = bool(test_flags[i])
            row_index = int(row_indices[i])
            label = float(labels[i])
            group.append((card, product, amount, row_index, is_test, i, label))
            additions_by_card.setdefault(card, []).append((amount, product))
            additions_by_product.setdefault(product, []).append(label)

        for card, additions in additions_by_card.items():
            state = card_state.setdefault(card, [deque(), Counter(), 0.0])
            history, product_counts, amount_sum = state
            cutoff = timestamp - CARD_WINDOW_SECONDS
            while history and history[0][0] < cutoff:
                old_ts, old_amount, old_product = history.popleft()
                product_counts[old_product] -= 1
                if product_counts[old_product] == 0:
                    del product_counts[old_product]
                amount_sum -= old_amount
            state[2] = amount_sum

        for product in additions_by_product:
            state = product_state.setdefault(product, [deque(), 0.0, 0])
            history, fraud_sum, fraud_count = state
            cutoff = timestamp - PRODUCT_WINDOW_SECONDS
            while history and history[0][0] < cutoff:
                _, old_label = history.popleft()
                if old_label >= 0:
                    fraud_sum -= old_label
                    fraud_count -= 1
            state[1], state[2] = fraud_sum, fraud_count

        for card, product, amount, row_index, is_test, event_index, label in group:
            if not is_test:
                continue
            card_history, card_products, card_amount = card_state[card]
            same_time = additions_by_card[card]
            distinct_products = set(card_products)
            distinct_products.update(item[1] for item in same_time)
            product_history, fraud_sum, fraud_count = product_state[product]
            output[row_index] = (
                len(card_history) + len(same_time),
                card_amount + sum(item[0] for item in same_time),
                len(distinct_products),
                fraud_sum / fraud_count if fraud_count else 0.0,
                len(product_history),
            )

        # Add the complete timestamp batch after calculating the exclusive
        # 30-day merchant history, while 24-hour velocity includes this batch.
        for card, product, amount, _, _, event_index, label in group:
            history, product_counts, amount_sum = card_state[card]
            history.append((timestamp, amount, product))
            product_counts[product] += 1
            card_state[card][2] = amount_sum + amount

            merchant_history, fraud_sum, fraud_count = product_state[product]
            merchant_history.append((timestamp, label if not np.isnan(label) else -1.0))
            if not np.isnan(label):
                product_state[product][1] = fraud_sum + label
                product_state[product][2] = fraud_count + 1

        start = end

    return output


def main() -> None:
    args = parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(args.model)
    feature_names = list(bundle["feature_names"])
    if len(feature_names) != 105 or len(set(feature_names)) != len(feature_names):
        raise ValueError(f"Expected a 105-feature model bundle; found {len(feature_names)} features.")
    if not set(WINDOW_FEATURES).issubset(feature_names):
        raise ValueError("Model feature schema does not contain the expected five window features.")

    with args.test_prepared.open(newline="", encoding="utf-8") as source:
        test_header = next(csv.reader(source))
    required_base = set(feature_names) - set(WINDOW_FEATURES)
    missing = sorted(required_base - set(test_header))
    if missing:
        raise ValueError(f"Prepared test CSV is missing model features: {missing}")
    if "TransactionDT" not in test_header:
        raise ValueError("Prepared test CSV must include TransactionDT to compute rolling features.")

    rolling = rolling_features(args.train_prepared, args.test_prepared)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "test_features.csv"
    sample_path = args.output_dir / "predict_requests.jsonl"
    sample_size = max(0, args.request_sample_size)

    with args.test_prepared.open(newline="", encoding="utf-8") as source, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=feature_names)
        writer.writeheader()
        with sample_path.open("w", encoding="utf-8", newline="\n") as requests:
            for row_index, row in enumerate(reader):
                feature_row = {name: row[name] for name in feature_names if name not in WINDOW_FEATURES}
                feature_row.update(dict(zip(WINDOW_FEATURES, rolling[row_index].tolist())))
                ordered_row = {name: feature_row[name] for name in feature_names}
                writer.writerow(ordered_row)
                if row_index < sample_size:
                    request = {name: float(value) for name, value in ordered_row.items()}
                    requests.write(json.dumps({"features": request}, separators=(",", ":")) + "\n")

    print(f"Wrote {output_csv} with 105 features and {len(rolling):,} rows")
    print(f"Wrote {sample_path} with {min(sample_size, len(rolling)):,} request bodies")


if __name__ == "__main__":
    main()

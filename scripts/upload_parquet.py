"""Upload a contract-validated feature Parquet to BigQuery."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.env import load_repo_env
from ingest.build_features import load_parquet_to_bigquery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=REPO_ROOT / "outputs" / "staged" / "v1" / "features.parquet",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="Fully qualified BigQuery table. Defaults to FEAST_BIGQUERY_TABLE from .env.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Explicitly replace an existing destination table. Without this, the load uses WRITE_EMPTY.",
    )
    return parser.parse_args()


def main() -> None:
    load_repo_env()
    args = parse_args()
    table_ref = args.table or os.environ.get("FEAST_BIGQUERY_TABLE")
    if not table_ref:
        raise ValueError("Pass --table or configure FEAST_BIGQUERY_TABLE in .env.")
    if not args.parquet.is_file() and not args.parquet.is_dir():
        raise FileNotFoundError(f"Parquet input not found: {args.parquet}")
    load_parquet_to_bigquery(args.parquet, table_ref, replace_existing=args.replace_existing)
    print(f"Loaded validated Parquet {args.parquet} -> {table_ref}")


if __name__ == "__main__":
    main()

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from google.cloud import bigquery

from config.env import load_repo_env


load_repo_env()

parquet_path = Path(__file__).resolve().parent.parent / "outputs/staged/v1/features.parquet"
table_ref = os.environ["FEAST_BIGQUERY_TABLE"]

client = bigquery.Client()
df = pd.read_parquet(parquet_path)
if "ts" in df.columns:
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
job = client.load_table_from_dataframe(
    df,
    table_ref,
    job_config=bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[bigquery.SchemaField("ts", "TIMESTAMP")],
    ),
)
job.result()
print(f"Loaded {parquet_path} -> {table_ref}")

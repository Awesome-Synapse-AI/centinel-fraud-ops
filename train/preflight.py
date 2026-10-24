"""Fail-fast validation of the BigQuery/Feast model-input handoff."""

from __future__ import annotations

import pandas as pd

from config.model_schema import (
    INTEGER_MODEL_FEATURES,
    MODEL_FEATURE_NAMES,
)


BIGQUERY_NUMERIC_TYPES = {"INTEGER", "INT64", "FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"}
BIGQUERY_INTEGER_TYPES = {"INTEGER", "INT64"}


def validate_feature_columns(columns: dict[str, str], source: str) -> None:
    """Validate required names and numeric types from Spark/Parquet or BigQuery."""
    missing = [name for name in MODEL_FEATURE_NAMES if name not in columns]
    incompatible = []
    for name in MODEL_FEATURE_NAMES:
        if name not in columns:
            continue
        dtype = columns[name].upper()
        if name in INTEGER_MODEL_FEATURES:
            if dtype not in BIGQUERY_INTEGER_TYPES and dtype not in {"LONGTYPE", "INTEGERTYPE", "SHORTTYPE"}:
                incompatible.append(f"{name}={columns[name]} (expected integer)")
        elif dtype not in BIGQUERY_NUMERIC_TYPES and dtype not in {"DOUBLETYPE", "FLOATTYPE", "DECIMALTYPE"}:
            incompatible.append(f"{name}={columns[name]} (expected numeric)")

    problems = []
    if missing:
        problems.append(f"missing model inputs: {', '.join(missing)}")
    if incompatible:
        problems.append(f"incompatible model-input types: {', '.join(incompatible)}")
    if problems:
        raise ValueError(f"Feature schema preflight failed for {source}: " + "; ".join(problems))


def validate_bigquery_feature_table(
    table_ref: str,
    feature_refs: list[str],
    expected_mapping_sha256: str | None = None,
) -> None:
    from google.cloud import bigquery

    client = bigquery.Client()
    table = client.get_table(table_ref)
    columns = {field.name: field.field_type.upper() for field in table.schema}
    validate_feature_columns(columns, f"BigQuery table {table_ref}")

    metadata_expected = {
        "card1": BIGQUERY_INTEGER_TYPES,
        "ts": {"TIMESTAMP"},
        "TransactionID": BIGQUERY_INTEGER_TYPES,
        "TransactionDT": BIGQUERY_NUMERIC_TYPES,
        "isFraud": BIGQUERY_INTEGER_TYPES,
        "category_mapping_sha256": {"STRING"},
    }
    metadata_errors = [
        f"{name}={columns.get(name, '<missing>')} (expected {sorted(allowed)})"
        for name, allowed in metadata_expected.items()
        if columns.get(name) not in allowed
    ]
    if metadata_errors:
        raise ValueError(
            f"Feature schema preflight failed for BigQuery table {table_ref}: "
            "invalid entity/timestamp columns: " + ", ".join(metadata_errors)
        )

    if expected_mapping_sha256 is not None:
        fingerprints = client.query(
            f"SELECT DISTINCT category_mapping_sha256 FROM `{table_ref}` LIMIT 2"
        ).to_dataframe()["category_mapping_sha256"].dropna().tolist()
        if fingerprints != [expected_mapping_sha256]:
            raise ValueError(
                f"Feature schema preflight failed for BigQuery table {table_ref}: "
                "categorical mapping fingerprint is missing, mixed, or does not match "
                "the supplied preprocessing artifact. Rebuild the feature table with "
                "the matching categorical_mappings.json."
            )

    service_names = {ref.split(":", 1)[-1] for ref in feature_refs}
    expected_names = set(MODEL_FEATURE_NAMES)
    if service_names != expected_names:
        raise ValueError(
            "Feast feature service does not match the shared model-input contract. "
            f"Missing from service: {sorted(expected_names - service_names)}; "
            f"extra in service: {sorted(service_names - expected_names)}"
        )


def verify_historical_feature_sample(
    store,
    table_ref: str,
    feature_refs: list[str],
    sample_rows: int = 32,
) -> int:
    """Fetch a small labeled entity sample through Feast before model fitting."""
    from google.cloud import bigquery

    if sample_rows < 1:
        raise ValueError("preflight sample row count must be at least one")
    client = bigquery.Client()
    query = f"""
        SELECT card1, TIMESTAMP(ts) AS ts, isFraud, TransactionDT
        FROM `{table_ref}`
        WHERE card1 IS NOT NULL AND ts IS NOT NULL
          AND isFraud IS NOT NULL AND TransactionDT IS NOT NULL
        ORDER BY ts
        LIMIT {int(sample_rows)}
    """
    entity_df = client.query(query).to_dataframe()
    if entity_df.empty:
        raise ValueError(f"No entity rows available for Feast preflight in {table_ref}.")
    entity_df["ts"] = pd.to_datetime(entity_df["ts"], utc=True)
    historical = store.get_historical_features(entity_df=entity_df, features=feature_refs).to_df()
    validate_feature_columns(
        {name: str(dtype) for name, dtype in historical.dtypes.items()},
        f"Feast historical sample from {table_ref}",
    )
    if historical.empty:
        raise ValueError(f"Feast returned no historical rows from {table_ref}.")
    return len(historical)

"""Serialization helpers for the categorical mappings shared by training and serving."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

from config.model_schema import CATEGORICAL_RAW_FEATURE_NAMES


def load_category_mappings(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Categorical mappings not found at {path}. Run scripts/04_run_preprocessing_pipeline.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("mappings"), dict):
        raise ValueError(f"Unsupported categorical mapping artifact: {path}")
    mappings = payload["mappings"]
    missing = sorted(set(CATEGORICAL_RAW_FEATURE_NAMES).difference(mappings))
    if missing:
        raise ValueError(f"Categorical mapping artifact is missing required columns: {missing}")
    for column in CATEGORICAL_RAW_FEATURE_NAMES:
        item = mappings[column]
        if not isinstance(item.get("values"), list) or int(item.get("unknown_value", -1)) < 0:
            raise ValueError(f"Invalid category mapping for {column} in {path}")
    return mappings


def encode_category_value(value, mapping: dict) -> int:
    normalized = "__MISSING__" if value is None else str(value)
    try:
        return mapping["values"].index(normalized)
    except ValueError:
        return int(mapping["unknown_value"])


def category_mapping_sha256(mappings: dict[str, dict]) -> str:
    canonical = json.dumps(mappings, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

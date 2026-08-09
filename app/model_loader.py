from __future__ import annotations

import os
from pathlib import Path

import joblib
from google.cloud import storage

from config.env import load_repo_env


load_repo_env()


def _download_from_gcs(gcs_uri: str, local_path: Path) -> Path:
    bucket_name, blob_name = gcs_uri.removeprefix("gs://").split("/", 1)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(local_path)
    return local_path


def resolve_model_path() -> Path:
    local_model_path = os.getenv("MODEL_PATH", "outputs/model.joblib")
    gcs_model_uri = os.getenv("MODEL_GCS_URI")

    if gcs_model_uri:
        cache_path = Path(os.getenv("MODEL_CACHE_PATH", "/tmp/champion_model.joblib"))
        return _download_from_gcs(gcs_model_uri, cache_path)
    return Path(local_model_path)


def load_model_bundle() -> dict:
    model_path = resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found at {model_path}")
    return joblib.load(model_path)

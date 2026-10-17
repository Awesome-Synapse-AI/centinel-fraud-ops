"""Publish the locally selected bundle to the GCS champion path used by the VM scorer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

# Make direct execution (`python train/publish_champion.py`) resolve the repo's
# top-level config package as well as module execution does.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.env import load_repo_env


load_repo_env()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a validated model bundle as champion.")
    parser.add_argument("--model", type=Path, default=Path("outputs/best-params.joblib"))
    parser.add_argument("--project-bucket", required=True)
    parser.add_argument("--champion-blob", default="models/champion/model.joblib")
    return parser.parse_args()


def publish(model_path: Path, bucket_name: str, champion_blob: str) -> str:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")
    bundle = joblib.load(model_path)
    required = {"model", "feature_names", "threshold"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(f"{model_path} is not a supported fraud model bundle.")
    if not bundle["feature_names"]:
        raise ValueError("Model bundle has no feature names.")

    from google.cloud import storage

    blob = storage.Client().bucket(bucket_name).blob(champion_blob)
    blob.upload_from_filename(str(model_path), content_type="application/octet-stream")
    return f"gs://{bucket_name}/{champion_blob}"


def main() -> None:
    args = parse_args()
    destination = publish(args.model, args.project_bucket, args.champion_blob)
    print(f"Published {args.model} to {destination}")


if __name__ == "__main__":
    main()

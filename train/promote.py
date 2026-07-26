import argparse
from pathlib import Path

from google.cloud import storage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a trained model artifact to champion.")
    parser.add_argument("--project-bucket", required=True)
    parser.add_argument("--source-blob", required=True)
    parser.add_argument(
        "--champion-blob",
        default="models/champion/model.joblib",
        help="Destination blob path for the promoted champion model.",
    )
    return parser.parse_args()


def promote(bucket_name: str, source_blob: str, champion_blob: str) -> None:
    bucket = storage.Client().bucket(bucket_name)
    source = bucket.blob(source_blob)
    if not source.exists():
        raise FileNotFoundError(f"GCS object not found: gs://{bucket_name}/{source_blob}")
    bucket.copy_blob(source, bucket, champion_blob)


def main() -> None:
    args = parse_args()
    promote(args.project_bucket, args.source_blob, args.champion_blob)
    print(
        f"Promoted gs://{args.project_bucket}/{args.source_blob} "
        f"to gs://{args.project_bucket}/{args.champion_blob}"
    )


if __name__ == "__main__":
    main()

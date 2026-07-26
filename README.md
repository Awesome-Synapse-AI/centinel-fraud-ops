# Cost-Weighted Fraud Detection Pipeline


## What is included

- Local ingestion and feature engineering for the IEEE-CIS fraud dataset
- Feast configuration for GCP and local development
- XGBoost training, calibration, evaluation, and promotion helpers
- FastAPI scoring service with GCS-backed champion model loading
- Cost-weighted drift monitoring utilities
- Streamlit demo app and synthetic drift seeding script
- Cloud Build, Cloud Run, Cloud Functions, and GKE deployment assets
- Updated `scripts/` utilities aligned with the same dataset conventions

## Quick start

1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put Kaggle IEEE-CIS training files at:

```text
dataset/train_transaction.csv
dataset/train_identity.csv
```

4. Run the local ingestion step:

```bash
python ingest/build_features.py --output-dir data/staged/v1
```

5. Train a model locally:

```bash
python train/train.py --features data/staged/v1/features.parquet --output outputs/model.joblib
```

6. Start the scorer:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```
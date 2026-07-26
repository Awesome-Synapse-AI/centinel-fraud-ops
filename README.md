# Cost-Weighted Fraud Detection Pipeline


## What is included

- A first-class `scripts/` preprocessing stage
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

4. Run the preprocessing audit stage from `scripts/`:

```bash
python scripts/04_run_preprocessing_pipeline.py --output-root outputs/preprocessing
```

5. Run the local ingestion step:

```bash
python ingest/build_features.py --preprocessed-csv outputs/preprocessing/03_label_corr_filtered.csv --output-dir data/staged/v1
```

6. Train a model locally:

```bash
python train/train.py --features data/staged/v1/features.parquet --output outputs/model.joblib
```

7. Start the scorer:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## How `scripts/` fits the plan

`scripts/` is the local preprocessing gate before feature-store ingestion:

- `01_check_missing_values.py`: profiles missingness after the required left join
- `02_remove_correlated_features.py`: reports and removes highly correlated numeric features
- `03_remove_useless_label_corr_features.py`: drops weak numeric features by label correlation
- `04_run_preprocessing_pipeline.py`: orchestrates the three scripts into one reproducible stage

The intended flow is:

```text
raw Kaggle CSVs
-> scripts/04_run_preprocessing_pipeline.py
-> outputs/preprocessing/03_label_corr_filtered.csv
-> ingest/build_features.py
-> data/staged/v1/features.parquet
-> train/train.py
```

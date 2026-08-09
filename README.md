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

6. Register and materialize Feast features:

```bash
feast -c feature_repo apply
feast -c feature_repo materialize-incremental 2026-07-26T00:00:00
```

7. Upload the staged parquet to BigQuery:

```bash
python scripts/upload_parquet.py
```

8. Train a model from the uploaded BigQuery table:

```bash
python train/train.py --output outputs/model.joblib
```

9. Start the scorer:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Feast point-in-time layer

The repo now includes [feature_repo/features.py](/D:/Awesome-Synapse-AI/centinel-fraud-ops/feature_repo/features.py) so Feast is part of the implemented pipeline, not just the plan.

Apply and materialize the feature repo:

```bash
feast -c feature_repo apply
feast -c feature_repo materialize-incremental 2026-07-26T00:00:00
```

Build a point-in-time correct training set from a label spine:

```python
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")
train_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "tx_velocity:txn_count_24h",
        "tx_velocity:amt_sum_24h",
        "merchant_risk:merch_fraud_rate_30d",
    ],
).to_df()
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
-> scripts/upload_parquet.py
-> BigQuery table in FEAST_BIGQUERY_TABLE
-> train/train.py
```

# Cost-Weighted Fraud Detection Pipeline

A fraud-detection pipeline built around reproducible data preparation, XGBoost model training and calibration, and a FastAPI inference service. The production inference service is deployed as a **Docker container on a single Google Compute Engine VM**. The trained model artifact remains in Google Cloud Storage (GCS), separate from the container image.

## What is included

- Reproducible preprocessing for the IEEE-CIS fraud dataset
- Local ingestion and feature engineering
- Feast configuration for GCP and local development
- XGBoost training, calibration, evaluation, and model promotion helpers
- FastAPI scoring service with GCS-backed champion-model loading
- Cost-weighted drift-monitoring utilities
- Streamlit demo app and synthetic drift-seeding script
- Docker image build and single-VM deployment tooling for GCP

## Architecture and deployment scope

The deployment described here uses Docker on a regular Compute Engine VM. It does not require a container orchestrator.

![Centinel Fraud Detection – System Architecture: end-to-end flow from data preparation and model training to deployment, monitoring, and demoing](docs/images/architecture.png)

1. Train and calibrate the model.
2. Publish the selected champion artifact to Cloud Storage.
3. Build the FastAPI scorer Docker image.
4. Push the image to Artifact Registry.
5. Create or reuse a Compute Engine VM and install Docker.
6. Pull and run one scorer container on the VM.
7. The container downloads the champion model from GCS using the VM's attached service account.

### GCP components

| Component | Purpose |
|---|---|
| Compute Engine (`e2-micro` starting point) | Runs the Docker container |
| Docker | Packages and runs the FastAPI scorer |
| Artifact Registry | Stores the scorer image |
| Cloud Storage | Stores the trained champion model artifact |
| VM service account | Grants the VM permission to pull the image and read the model |
| Cloud Logging / VM logs | Used to inspect service output and diagnose startup issues |

An `e2-micro` is a low-cost starting point, not a guaranteed fit for every model and workload. The model must fit in available memory alongside Python, FastAPI, and the web server. If the process is terminated for running out of memory or the VM becomes unstable, move to a larger machine type.

### How the pieces fit together

The diagram above groups the system into eight numbered stages. Below is how each stage maps to the actual code and GCP resources in this repository. A few boxes in the diagram use illustrative filenames for the monitoring/demo layer; the right-hand column gives the real path so the picture and the code stay reconcilable.

| # | Stage | What happens | Implemented by |
|---|---|---|---|
| 1 | Data preparation & model training | Join the raw IEEE-CIS CSVs, clean/encode/split them, engineer rolling velocity and merchant-risk features, optionally stage them to BigQuery, then train and calibrate the XGBoost model and pick a decision threshold. Runs locally or on a notebook/VM — nothing here depends on GCP being reachable except the optional BigQuery upload. | `scripts/04_run_preprocessing_pipeline.py` (preprocessing), `ingest/build_features.py` (feature engineering), `scripts/upload_parquet.py` (BigQuery upload), `train/train.py` / `train/modeling.py` / `train/grid_search.py` (training, calibration, threshold selection) |
| 2 | Data & feature store | The engineered features land in a BigQuery table that acts as Feast's offline store — the single source of truth both training and online feature retrieval read from. | BigQuery table named by `FEAST_BIGQUERY_TABLE`, written by `scripts/upload_parquet.py` |
| 3 | Feature retrieval | Feast defines the entities and feature views over that BigQuery table and exposes point-in-time-correct historical retrieval for training, plus a materialized online store for low-latency lookups. | `feature_repo/features.py`, `feature_repo/feature_store.yaml` (`feast apply` / `feast materialize-incremental`) |
| 4 | Model artifacts | The winning calibrated model bundle (model, feature list, threshold) is published to a Cloud Storage bucket as the "champion" artifact, independent of the serving container. Separately, `deploy/Dockerfile` builds the FastAPI scorer image, which is pushed to Artifact Registry. | `train/publish_champion.py` → `gs://<PROJECT_ID>-artifacts/models/champion/model.joblib`; `deploy/Dockerfile` + `deploy/vm/cloudbuild.yaml` → Artifact Registry image |
| 5 | Model serving (production) | A Compute Engine VM runs one Docker container with the FastAPI app. On startup the container downloads the champion model from GCS using the VM's attached service account, then serves `GET /health` (readiness/threshold check) and `POST /predict` (score a transaction and return the fraud probability, threshold, and flag). | `app/main.py` (routes), `app/model_loader.py` (GCS download + load), deployed by `deploy/vm/deploy-vm.ps1` |
| 6 | Demo & exploration (optional) | A local Streamlit console lets you score a sampled transaction against the running FastAPI service and inspect the cost-weighted drift leaderboard without touching production traffic. (This repository ships that console instead of a separate notebook/CLI demo app.) | `demo/streamlit_app.py` (`make demo`), fed by `demo/prepare_drift_windows.py` and `demo/seed_drift_scenario.py` — see [Demo console](#demo-console-scoring--cost-weighted-drift-review) below |
| 7 | Users / clients | Any HTTP client — the Streamlit demo, a web app, a batch job — calls the deployed scorer's `/predict` endpoint over HTTPS on port `8080` with JSON transaction features and gets back a probability, threshold, and flag. | `POST /predict` on the Compute Engine VM, consumed directly or via `demo/streamlit_app.py`'s "Scorer URL" setting |
| 8 | Monitoring & drift detection | Baseline vs. recent production feature windows are compared feature-by-feature (KL divergence), weighted by SHAP-style model importance and the fraud/false-positive cost assumptions, to rank drift by expected dollar impact. When cumulative cost crosses a threshold, an alert record is persisted and forwarded to a notification endpoint. | `monitor/cost_drift.py` (drift math) driven by `demo/run_drift_analysis.py --send-alert` or the Streamlit drift pane; `monitor/alerting.py` persists the alert to Firestore and calls the HTTP endpoint in `gcp/alert_function/main.py` (deployed via `deploy/deploy-alert-function.ps1`) |

In short: stages 1–4 are an offline pipeline that ends with two artifacts in GCP (a model in Cloud Storage, an image in Artifact Registry); stage 5 is the only stage that serves live traffic; stages 6–8 are optional local/operational tooling that read from the same feature and model artifacts to let you demo scoring and keep drift honest in dollar terms rather than raw statistics.

## Local development quick start

### 1. Create a Python 3.11 environment

```bash
python -m venv .venv
```

Activate the environment, then install the project dependencies:

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Place the Kaggle IEEE-CIS training files here:

```text
dataset/train_transaction.csv
dataset/train_identity.csv
```

Run the preprocessing stage:

```bash
python scripts/04_run_preprocessing_pipeline.py --output-root outputs/preprocessing-contract-v1
```

The preprocessing run preserves every required contract column through the
correlation filters and writes the fitted categorical mappings to
`outputs/preprocessing-contract-v1/prepared/categorical_mappings.json`.

Build the engineered features:

```bash
python ingest/build_features.py \
  --preprocessed-csv outputs/preprocessing-contract-v1/03_label_corr_filtered.csv \
  --categorical-mappings outputs/preprocessing-contract-v1/prepared/categorical_mappings.json \
  --output-dir data/staged/v1
```

This command applies those saved mappings, computes the rolling features, and
writes a contract-checked Parquet containing exactly the Feast model inputs
plus `card1`, `ts`, `TransactionDT`, `TransactionID`, and `isFraud`. It does not
upload to BigQuery unless `--bigquery-table` is supplied. Loads refuse to
replace an existing table unless `--replace-bigquery-table` is explicit.
The table also stores a SHA-256 fingerprint of the categorical mapping. Both
training commands compare that fingerprint with the supplied mapping artifact
and stop if the feature codes came from a different preprocessing run.

To verify the complete Parquet → BigQuery → Feast handoff safely, upload a
small sample to a new scratch table (choose a unique table ID):

```powershell
python ingest/build_features.py `
  --preprocessed-csv outputs/preprocessing-contract-v1/03_label_corr_filtered.csv `
  --categorical-mappings outputs/preprocessing-contract-v1/prepared/categorical_mappings.json `
  --sample-rows 5000 `
  --output-dir outputs/contract-handoff `
  --bigquery-table fraud-cost-pipeline.fraud_features.features_contract_check_20261018 `
  --feast-sample-check
```

The check uses a temporary local Feast registry, so it reads that scratch table
without changing the production registry or online store. It leaves the scratch
table in BigQuery for inspection.

### 3. Prepare features and train

Apply and materialize the Feast feature repository using the configured project and data sources:

```bash
feast -c feature_repo apply
feast -c feature_repo materialize-incremental 2026-07-26T00:00:00
```

Upload the staged parquet data to BigQuery:

```bash
python scripts/upload_parquet.py
```

Train a model:

```bash
python train/train.py --output outputs/model.joblib
```

To search XGBoost hyperparameters using a chronological holdout from the training split, then retrain the best configuration and evaluate it on the time-based validation/test splits:

```bash
python train/grid_search.py --output outputs/model-grid-search.joblib
```

Each parameter combination is logged as a child MLflow run, and the selected
model, metrics, and search CSV are logged on the parent run. By default,
tracking metadata is stored in `outputs/mlflow.db`, and artifacts are stored
in `mlruns/`; open the experiment with:

```bash
mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db --host 127.0.0.1 --port 5000
```

Set `MLFLOW_TRACKING_URI` (and optionally `MLFLOW_EXPERIMENT_NAME`) in `.env`
to send runs to a shared MLflow tracking server. MLflow tracks experiments;
the scorer still loads the champion bundle from GCS after publishing it.

The search writes `outputs/model-grid-search.grid-search.csv` and per-run `config.json` and `result.json` files under `outputs/model-grid-search.grid-search-runs/`. Use `--search-validation-fraction` to change the chronological holdout size.

### 4. Run the API locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

The production container listens on port `8080`. Configure the model source through the environment rather than relying on a developer-specific absolute file path.

## Demo console: scoring + cost-weighted drift review

`demo/` contains a self-contained Streamlit console plus the CLI helpers used to prepare its inputs. It is meant for local walkthroughs of the two things this pipeline cares about: scoring a transaction and watching for feature drift that is expensive in fraud-cost terms, not just statistically significant.

| File | Purpose |
|---|---|
| `demo/streamlit_app.py` | Two-pane console: score a sampled row from the prepared test CSV against the live FastAPI scorer, and view a cost-weighted drift leaderboard comparing a baseline window to a production window. |
| `demo/prepare_drift_windows.py` | Builds the `baseline_window.parquet` / `production_window.parquet` / `shap_importance.json` inputs the console reads, aligned to a trained model's feature schema. Either splits one historical Parquet file chronologically by `TransactionDT` (`--source`), or accepts two already-separated windows (`--baseline` / `--production`). |
| `demo/seed_drift_scenario.py` | Synthetic drift generator: shifts one numeric feature by a configurable number of standard deviations for a fraction of rows, so you can demo the drift leaderboard and alerting without waiting on real production drift. |
| `demo/run_drift_analysis.py` | CLI equivalent of the drift leaderboard pane. Ranks features by estimated dollar cost of drift (using the cost-weighted drift utilities in `monitor/cost_drift.py`) and, with `--send-alert`, persists/forwards an alert through `monitor/alerting.py` once cumulative cost crosses a threshold. |

### Running the console

```bash
make demo
# equivalent to:
streamlit run demo/streamlit_app.py
```

The sidebar exposes the settings below (each also configurable via environment variable, loaded from `.env` through `config/env.py`):

| Setting | Env var | Default | Notes |
|---|---|---|---|
| Scorer URL | `SCORER_URL` | `http://localhost:8080` | Base URL for the FastAPI service; the console calls `/health` and `/predict`. Start it with `make score`. |
| Prepared test CSV | `TEST_DATA_CSV` | `outputs/preprocessing/prepared/test_prepared.csv` | Source of sample rows for the scoring pane; produced by the preprocessing pipeline in step 2 above. |
| Drift data directory | `DEMO_DATA_DIR` | `outputs/demo` | Must contain `baseline_window.parquet`, `production_window.parquet`, and `shap_importance.json`, produced by `demo/prepare_drift_windows.py`. |

The scoring pane sends the selected row as named model features to `/predict` and shows the returned fraud probability, decision threshold, and flag status. The drift pane computes a per-feature KL divergence between the two windows, weighs it by SHAP-derived feature importance and the configured fraud-cost assumptions (`p_fraud`, false-negative cost, false-positive cost), and charts the top offending features by estimated dollar impact.

### End-to-end example

```bash
# 1. Start the scorer in a separate terminal
make score

# 2. Build baseline/production windows and importance weights from a trained
#    model and a historical feature dataset (chronological split)
python demo/prepare_drift_windows.py \
  --source data/staged/v1/features.parquet \
  --model outputs/model.joblib \
  --output-dir outputs/demo

# 3. Optional: inject synthetic drift into one feature for the demo
python demo/seed_drift_scenario.py \
  --baseline outputs/demo/baseline_window.parquet \
  --output outputs/demo/production_window.parquet \
  --feature merch_fraud_rate_30d

# 4. Launch the console
make demo
```

To rank drift from the command line instead of (or in addition to) the Streamlit pane, including sending an alert when the cumulative cost-weighted drift crosses the configured threshold:

```bash
python demo/run_drift_analysis.py \
  --baseline outputs/demo/baseline_window.parquet \
  --production outputs/demo/production_window.parquet \
  --importance outputs/demo/shap_importance.json \
  --top 10 \
  --send-alert
```

## Deploy the trained model to GCP with Docker

The deployment uses the model artifact in GCS and the scorer image in Artifact Registry. Do not bake the trained model into the Docker image or copy a local Windows path into the container.

### Prerequisites

- A GCP project with billing enabled
- Google Cloud CLI (`gcloud`) installed and authenticated
- Docker available locally if building the image locally; the repository deployment script may use Cloud Build to build and push the image
- The deployment files in `deploy/`, including `deploy/Dockerfile` and `deploy/vm/deploy-vm.ps1`
- A model artifact produced by the training workflow
- A VM service account with permission to read the model object and pull the container image

Set the project and deployment location in PowerShell. **Use your human Google account for interactive deployment and IAM changes**, rather than leaving `gcloud` authenticated as the pipeline service account:

```powershell
$PROJECT_ID = "fraud-cost-pipeline"
$REGION = "us-central1"
$ZONE = "us-central1-a"
$ACCOUNT = "windowsandandroid.app@gmail.com"

gcloud auth login $ACCOUNT
gcloud config set account $ACCOUNT
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION
gcloud config set compute/zone $ZONE

gcloud auth list
```

In `gcloud auth list`, confirm that `windowsandandroid.app@gmail.com` is the active account (marked with `*`). The active `gcloud` account controls CLI operations such as inspecting Compute Engine resources and changing IAM policy; it does **not** change the credentials Python uses to publish the model. Python model publishing continues to use the service-account credentials configured in `.env`. Keep those two authentication paths distinct.

### If you cannot inspect the VM or need IAM access

The pipeline service account `fraud-pipeline-sa@fraud-cost-pipeline.iam.gserviceaccount.com` cannot modify project IAM policies. Switch `gcloud` to the human account above before retrying IAM commands. For example, grant the human account Compute Engine Viewer access:

```powershell
$PROJECT = "fraud-cost-pipeline"
$ACCOUNT = "windowsandandroid.app@gmail.com"

gcloud projects add-iam-policy-binding $PROJECT `
  --member="user:$ACCOUNT" `
  --role="roles/compute.viewer"
```

This command itself requires permission to change the project's IAM policy. If it returns `PERMISSION_DENIED`, ask a project Owner or IAM Admin to grant the human account the required roles, including `roles/compute.viewer` and `roles/storage.objectAdmin` where needed. Do not try to work around missing IAM permissions by embedding credentials in the Docker image.

### Retrieve the deployed VM's external IP

After the VM exists, retrieve its external NAT IP with the VM's actual name and zone:

```powershell
$PROJECT = "fraud-cost-pipeline"
$ZONE = "us-central1-a"
$VM = "fraud-scorer-vm"

$IP = (gcloud compute instances describe $VM `
  --zone=$ZONE `
  --project=$PROJECT `
  --format="value(networkInterfaces[0].accessConfigs[0].natIP)").Trim()

$IP
```

If `$IP` is empty, verify the VM name and zone and check whether the VM has an external IP configured. `roles/compute.viewer` allows inspection but does not grant permission to create or modify VM resources.

Enable the APIs required for the image registry, build, VM, and model storage:

```powershell
gcloud services enable `
  compute.googleapis.com `
  artifactregistry.googleapis.com `
  cloudbuild.googleapis.com `
  storage.googleapis.com `
  --project=$PROJECT_ID
```

### Step 1 — Publish the champion model to Cloud Storage

Publish the trained/calibrated champion bundle to the project's artifact bucket:

```powershell
python train/publish_champion.py --project-bucket "$PROJECT_ID-artifacts"
```

The expected object location is:

```text
gs://<PROJECT_ID>-artifacts/models/champion/model.joblib
```

For example, with the project ID `fraud-cost-pipeline`, the model URI is:

```text
gs://fraud-cost-pipeline-artifacts/models/champion/model.joblib
```

Use the actual project ID and confirm that the object exists before deploying:

```powershell
gcloud storage ls "gs://$PROJECT_ID-artifacts/models/champion/model.joblib"
```

### Step 2 — Build and deploy the Docker scorer to a Compute Engine VM

From the repository root, run the deployment script:

```powershell
.\deploy\vm\deploy-vm.ps1 -ProjectId $PROJECT_ID
```

The script is intended to build and push the scorer image to Artifact Registry, create or reuse the VM, install Docker, pull the image, and start one container with a restart policy. Review the script's parameter defaults before running it if you need a different VM name, zone, repository, or machine type.

The container receives `MODEL_GCS_URI` and downloads the artifact from GCS at startup. The local training path (for example, `D:\Awesome-Synapse-AI\centinel-fraud-ops\outputs\best-params.joblib`) is only a source path on the training computer; it is not a valid path inside the GCP VM container.

### Step 3 — IAM permissions

Use the VM's attached service account for runtime access. Do not place a service-account JSON key inside the repository or Docker image.

Grant the VM service account:

- `roles/storage.objectViewer` on the model artifact bucket (or a narrower resource-level grant where practical)
- `roles/artifactregistry.reader` on the Artifact Registry repository

The identity performing deployment also needs sufficient permissions to build/push the image and create or update the Compute Engine VM. Keep deploy-time permissions separate from the VM's runtime permissions when possible.

### Step 4 — Allow API traffic and verify the service

The API listens on container port `8080`. The VM deployment must map the required host port to container port `8080` and allow inbound traffic only from the clients that should access the API. Avoid exposing an unauthenticated scoring endpoint to the public internet.

After deployment, connect to the VM and inspect the running container and its logs:

```bash
docker ps
docker logs --tail 200 fraud-scorer
```

Verify the API's health endpoint and a representative scoring request using the actual routes defined by `app/main.py`. If startup fails, check the container logs first, then verify `MODEL_GCS_URI`, the GCS object path, service-account permissions, dependency installation, and available VM memory.

### Deployment configuration contract

| Setting | Expected value / behavior |
|---|---|
| Container port | `8080` |
| Model source | `MODEL_GCS_URI` environment variable |
| Champion artifact | `gs://<PROJECT_ID>-artifacts/models/champion/model.joblib` |
| Runtime identity | Compute Engine VM's attached service account |
| Container count | One scorer container |
| Restart behavior | Docker restart policy configured by deployment tooling |
| Model loading | Download from GCS, then load the serialized bundle in the API process |

## How `scripts/` fits the data pipeline

- `scripts/01_check_missing_values.py`: profiles missingness after the required join
- `scripts/02_remove_correlated_features.py`: reports and removes highly correlated numeric features
- `scripts/03_remove_useless_label_corr_features.py`: drops weak numeric features by label correlation
- `scripts/04_run_preprocessing_pipeline.py`: orchestrates the preprocessing steps

The intended training flow is:

```text
raw Kaggle CSVs
  -> scripts/04_run_preprocessing_pipeline.py
  -> outputs/preprocessing/03_label_corr_filtered.csv
  -> ingest/build_features.py
  -> data/staged/v1/features.parquet
  -> scripts/upload_parquet.py
  -> BigQuery table configured by FEAST_BIGQUERY_TABLE
  -> train/train.py
  -> publish champion artifact to GCS
  -> build/push scorer Docker image
  -> Compute Engine VM runs the container
  -> FastAPI loads the champion artifact from GCS
```

## Operational notes

- Keep raw datasets, generated training outputs, and service-account keys out of the Docker build context. Check `.dockerignore` and `.gcloudignore` before publishing images.
- Deploy a newly trained model by publishing the new champion artifact to the agreed GCS location, then restart/redeploy the scorer if the application loads the model only during startup.
- Use a versioned artifact URI when reproducibility or rollback is important; retain the champion path as the deployment pointer only if that is how the training/promotion workflow is implemented.
- An `e2-micro` has tight memory constraints. Start with one API worker and monitor memory use; use a larger VM if the model or runtime is killed or becomes unstable.
- The VM needs a restart policy and operational checks because Docker on a VM does not automatically provide managed application scaling or deployment rollbacks.
- Set a billing budget alert and review charges for VM uptime, persistent disk, image storage/builds, network egress, and Cloud Storage.

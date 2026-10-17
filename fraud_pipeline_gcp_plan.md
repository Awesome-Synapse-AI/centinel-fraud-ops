# Cost-Weighted Fraud/Credit-Risk Detection Pipeline — GCP Build Plan

This document describes the complete fraud-detection pipeline and its GCP
deployment. Data preparation, feature engineering, and interactive training
remain on the laptop. GCP stores shared artifacts and runs the operational
services. The real-time FastAPI scorer runs as one Docker container on a
regular Compute Engine `e2-micro` VM.

The central modeling idea is to rank drift alerts by expected dollar loss:
`KL divergence × feature importance × expected fraud loss`, rather than by
p-value alone.

## 0. Design decisions and corrections

- The scorer is deployed directly to one Compute Engine `e2-micro` VM. The VM
  is not treated as a managed cluster; Docker runs one container directly on
  the host.
- The champion model is a versioned GCS object. The current pointer is
  `gs://PROJECT-artifacts/models/champion/model.joblib`.
- The image contains the API and runtime dependencies, but not the local
  `outputs/` directory or a Windows path. The VM service account reads the
  champion object at startup.
- The `e2-micro` is a cost-conscious baseline, not a capacity guarantee. Keep
  one Gunicorn worker and upgrade the VM if the process is killed or latency is
  unstable.
- The IEEE-CIS identity join is a left join because many transactions have no
  identity row. Preserve a `has_identity` feature.
- Spark distinct counts use `approx_count_distinct` where an exact windowed
  distinct count is not supported.
- Merchant-risk features use a trailing window that excludes the current row,
  preventing future-label leakage.
- The GCP Feast profile uses compatible BigQuery and Firestore stores; the local
  profile remains available for offline development.
- The serialized model must be loaded with the same scikit-learn version used
  to create it. This repository's current bundle uses scikit-learn `1.7.1`.

## 1. Architecture at a glance

| Component | Location | Service or implementation |
|---|---|---|
| Raw Kaggle data | Laptop | Local files under `dataset/` |
| Preprocessing and ingestion | Laptop | `scripts/`, `ingest/`, local Spark/Python |
| Processed feature Parquet | GCP and laptop | Cloud Storage plus local staging |
| Feast offline store | GCP | BigQuery |
| Feast online store | GCP | Firestore, when online retrieval is needed |
| Feast registry | GCP | Cloud Storage |
| Interactive training | Laptop | XGBoost, calibration, evaluation |
| Experiment tracking | Optional GCP | MLflow with GCS-backed artifacts |
| Champion model | GCP | Cloud Storage fixed champion path |
| Container image | GCP | Artifact Registry |
| Real-time scorer | GCP | One Compute Engine `e2-micro` VM with Docker |
| Drift monitoring | Laptop or GCP | `monitor/`, Firestore, alert function |
| Analyst demo | Optional GCP | Streamlit service |
| Alert boundary | Optional GCP | Cloud Function HTTP endpoint |

### Serving flow

`outputs/best-params.joblib` is validated and uploaded by
`train/publish_champion.py`. The VM runs the image built from
`deploy/Dockerfile`, sets `MODEL_GCS_URI`, and loads
`gs://PROJECT-artifacts/models/champion/model.joblib` through its attached
service account. The API exposes `/health` and `/predict` on port `8080`.

### Repository mapping

- `scripts/`: preprocessing audits and reproducible filtering
- `ingest/`: local CSV ingestion and feature engineering
- `feature_repo/`: GCP and local Feast profiles
- `train/`: training, calibration, evaluation, promotion, and publishing
- `app/`: FastAPI scorer and model loader
- `monitor/`: cost-weighted drift ranking and alerting
- `demo/`: Streamlit console and synthetic drift data
- `gcp/alert_function/`: notification boundary
- `deploy/Dockerfile`: scorer image
- `deploy/vm/`: VM build and deployment automation

## 2. GCP services, quotas, and responsibilities

| Service | Role in this plan | Main constraint |
|---|---|---|
| Compute Engine | Always-on scorer VM | Shared CPU and approximately 1 GB RAM on `e2-micro` |
| Artifact Registry | Docker image storage | Storage and image retention cost |
| Cloud Build | Build and push the scorer image | Build time and build quota |
| Cloud Storage | Models, Parquet, Feast registry, MLflow artifacts | Object access and retention policy |
| BigQuery | Feast offline store | Query and storage quotas |
| Firestore | Feast online store and alert history | Daily free quota and write volume |
| Cloud Run | Optional MLflow, demo, or retraining services | Shared request/compute quota |
| Cloud Functions | Optional alert notification endpoint | Invocation and runtime quota |
| Secret Manager | Optional API and webhook secrets | Secret access IAM |
| Cloud Logging/Monitoring | Optional VM/service visibility | Log volume and retention |

The production scorer is intentionally separate from optional retraining,
tracking, demo, and notification services. A problem in an optional service
must not prevent the VM from loading the champion model.

## 3. Prerequisites

- GCP project with billing enabled and a small budget alert.
- Google Cloud CLI authenticated as a human account for deployment and IAM.
- Python environment with the project dependencies.
- `config/service-account.json` available locally if using the repository `.env`
  for model publishing; never commit this key.
- The repository's current deployment files:
  `deploy/Dockerfile`, `deploy/vm/cloudbuild.yaml`, and
  `deploy/vm/deploy-vm.ps1`.
- For model publishing, install `scikit-learn==1.7.1` and
  `google-cloud-storage==3.3.0` in the active environment.

Use a human account for commands that inspect VMs or change IAM. Python model
publishing uses `GOOGLE_APPLICATION_CREDENTIALS` from `.env`, which is a
separate credential path.

## 4. Project bootstrap

Set `PROJECT=fraud-cost-pipeline`, `REGION=us-central1`, and
`ZONE=us-central1-a`. Run `gcloud config set project $PROJECT`,
`gcloud config set compute/region $REGION`, and `gcloud config set
compute/zone $ZONE`.

Enable the APIs needed by the full pipeline: Compute Engine, Artifact Registry,
Cloud Build, Cloud Storage, BigQuery, Firestore, Secret Manager, Cloud Run,
Cloud Functions, and the required monitoring APIs. The minimal VM deployment
needs Compute Engine, Artifact Registry, Cloud Build, and Cloud Storage.

Create or reuse:

- Artifact Registry repository `fraud-images` in `us-central1`.
- Cloud Storage bucket `gs://fraud-cost-pipeline-artifacts`.
- BigQuery dataset `fraud_features`.
- Firestore database in the selected region, if using the online store or
  alert history.

Create a runtime service account named
`fraud-pipeline-sa@fraud-cost-pipeline.iam.gserviceaccount.com` when needed.
Grant it `roles/artifactregistry.reader` for image pulls and
`roles/storage.objectViewer` for the champion model. Add BigQuery, Firestore,
Secret Manager, or Cloud Run roles only to the workloads that need them.

The identity performing deployment separately needs permission to submit Cloud
Build jobs, create/inspect Compute Engine instances, configure firewall rules,
and attach the runtime service account.

## 5. Module 1 — Data ingestion and Feast

### 5.1 Prepare the labeled data locally

Keep `train_transaction.csv` and `train_identity.csv` under `dataset/`. The
unlabeled test files are not needed for supervised training or serving.

Run the preprocessing gate:

`python scripts/04_run_preprocessing_pipeline.py --output-root
outputs/preprocessing`

The stage checks missingness, removes highly correlated features, and removes
weak numeric features by label correlation. Its output is consumed by
`ingest/build_features.py`.

### 5.2 Build time-aware features

Run `python ingest/build_features.py --preprocessed-csv
outputs/preprocessing/03_label_corr_filtered.csv --output-dir data/staged/v1`.
The feature process should preserve event time, use left joins for identity,
and exclude the current transaction from trailing fraud-rate windows.

Upload Parquet to the configured GCS path and load the BigQuery table using the
project's feature-ingestion workflow. Keep the local Parquet output for
reproducible training and rollback.

### 5.3 Apply Feast

The GCP profile in `feature_repo/feature_store.yaml` uses BigQuery for offline
features, Firestore for online features, and a GCS registry. The local profile
uses local files and SQLite for development.

Run `feast -c feature_repo apply`, then materialize the required historical
range. For a large initial backfill, split feature views or date ranges so
Firestore writes remain within the intended free quota. After the initial
backfill, run `feast materialize-incremental` for new activity.

## 6. Module 2 — Model development and tracking

### 6.1 Train locally

Train XGBoost using chronological train/validation/test splits. Use class
imbalance handling, early stopping, and a metric appropriate for fraud, such as
AUCPR. Save the calibrated bundle to
`outputs/best-params.joblib`.

The bundle must contain the fitted model, the persisted `feature_names`, and
the decision `threshold`. The scorer reconstructs rows using those exact
feature names and returns a fraud probability plus threshold decision.

Run `python train/train.py --output outputs/best-params.joblib` for the standard
training path, or use `train/grid_search.py` for chronological hyperparameter
search. Run the test suite with `python -m pytest tests`.

### 6.2 Calibration and serialization

Use the scikit-learn calibration API compatible with the installed version. The
serving environment and publishing environment must use the same scikit-learn
minor version as the model artifact. If the version changes, retrain or
re-export the model rather than ignoring `InconsistentVersionWarning`.

### 6.3 Optional MLflow tracking

MLflow can track local or optional Cloud Run training runs. Store artifacts in
GCS and keep the model-serving pointer independent of MLflow aliases. The
fixed champion path is simple to consume from the VM and supports a controlled
promotion step:

1. Train a candidate model.
2. Evaluate it against the cost-weighted gate.
3. Retain the artifact under a versioned path.
4. Copy the approved artifact to `models/champion/model.joblib`.
5. Restart or refresh the VM scorer and verify `/health`.

The existing `train/publish_champion.py` performs local validation and uploads
the selected bundle. `train/promote.py` can implement a separate approved-copy
workflow when versioned MLflow artifacts are used.

## 7. Module 3 — Build and deploy the scorer VM

### 7.1 Image contents

`deploy/Dockerfile` builds a Python 3.11 image with the FastAPI application,
model-loading code, and deployment dependencies. It runs Gunicorn with one
Uvicorn worker on `0.0.0.0:8080`. The image does not include raw datasets,
generated outputs, or service-account keys.

### 7.2 Build and push

`deploy/vm/cloudbuild.yaml` builds with `deploy/Dockerfile` and pushes the image
to `us-central1-docker.pkg.dev/$PROJECT/fraud-images/fraud-scorer:latest`.
`deploy/vm/deploy-vm.ps1` resolves the repository root, so it can be launched
from any working directory.

### 7.3 Create or reuse the VM

The deployment script creates or reuses `fraud-scorer-vm` with:

- Zone `us-central1-a`
- Machine type `e2-micro`
- Debian 12
- A 30 GB standard persistent boot disk
- The runtime service account
- The `fraud-scorer` network tag

It installs Docker, authenticates image pulls using the VM identity, pulls the
image, removes the previous `fraud-scorer` container, and starts one replacement
with `--restart unless-stopped` and host port `8080` mapped to container port
`8080`.

### 7.4 Model loading

The container receives `MODEL_GCS_URI=gs://$PROJECT-artifacts/models/champion/
model.joblib`. `app/model_loader.py` downloads the object to a local cache and
loads it with joblib. If `MODEL_GCS_URI` is absent during local development,
the loader falls back to `MODEL_PATH` and then
`outputs/best-params.joblib`.

### 7.5 Deployment commands

Publish the model with `python train\\publish_champion.py --model
outputs\\best-params.joblib --project-bucket fraud-cost-pipeline-artifacts`.
Then run `.\\deploy\\vm\\deploy-vm.ps1 -ProjectId
fraud-cost-pipeline`.

The deployment script also creates or reuses the `allow-fraud-scorer` firewall
rule. Start with a restricted `-SourceRange YOUR_PUBLIC_IP/32` whenever
possible; `0.0.0.0/0` is suitable only for temporary testing.

## 8. Module 4 — Verification and operations

Retrieve the VM IP with `gcloud compute instances describe fraud-scorer-vm
--zone=us-central1-a --format="value(networkInterfaces[0].accessConfigs[0].natIP)"`.
Call `http://IP:8080/health`. It must report `status: ok`, a model feature
count, and the calibrated threshold. A failed health check means the model
cannot be read or loaded and should block traffic.

Call `POST http://IP:8080/predict` with a JSON transaction payload. Confirm the
response contains `fraud_prob`, `flag`, and `threshold`.

Inspect the container with `gcloud compute ssh fraud-scorer-vm
--zone=us-central1-a --command="sudo docker ps; sudo docker logs --tail 100
fraud-scorer"`. Monitor memory and restart behavior. The `e2-micro` has limited
RAM and shared CPU; keep one worker and move to a larger machine type if the
model or runtime is OOM-killed.

## 9. Module 5 — Cost-weighted drift, alerting, and demo

This module closes the model operations loop: compare a reference feature
window with a recent production window, rank changed features by estimated
business cost, persist the most important alert, and let an analyst inspect
the score and drift leaderboard in one console. The math can run locally or in
a scheduled/monitoring process; only the scoring API runs on the `e2-micro`
VM. The demo and alert function are optional services and do not need to share
the VM's constrained memory.

### 9.1 Drift input windows

Provide two Parquet dataframes with aligned feature columns:

- **Baseline window**: a representative training or known-good period.
- **Production window**: a recent period of scored transactions.

Both windows should use the same feature transformations and names as the
persisted model schema. Exclude labels and identifiers that should not be
monitored as model inputs. The demo currently reads:

```text
outputs/demo/baseline_window.parquet
outputs/demo/production_window.parquet
outputs/demo/shap_importance.json
```

`shap_importance.json` maps feature names to non-negative importance weights.
Generate it from the trained model's SHAP or feature-importance analysis and
keep its feature names aligned with the Parquet columns. Importance should be
normalized consistently (for example, to sum to one) so cost scores can be
compared across runs.

If you already have separate known-good and recent production windows, pass
them with `--baseline` and `--production`. For a local historical demonstration,
split the staged feature dataset chronologically instead:

```powershell
python demo/prepare_drift_windows.py `
  --source outputs/staged/v1/features.parquet `
  --production-fraction 0.2 `
  --model outputs/best-params.joblib `
  --output-dir outputs/demo
```

With `--source`, the helper uses `TransactionDT` to put the latest 20% of rows
in the production window and earlier rows in the baseline window. This is a
historical comparison for local analysis; for actual monitoring, pass real
known-good and recent production feature windows separately. In either mode,
the helper selects `feature_names` from the model bundle in their saved order,
omitting labels, identifiers, and unrelated columns. It normalizes the bundle's
non-negative `importance` weights and writes the two window files and
`shap_importance.json` expected by the demo. The helper aligns columns but does
not re-engineer or impute feature values, so both windows must use the same
feature transformation/version.

### 9.2 Per-feature KL divergence

`monitor/cost_drift.py::kl_per_feature(baseline, prod, bins=50)` evaluates
shared numeric columns. For each feature it:

1. Replaces infinite values with missing values and drops missing entries for
   the distribution estimate.
2. Creates shared histogram edges spanning the baseline and production values.
3. Calculates the baseline distribution `p` and production distribution `q`.
4. Adds a small epsilon to histogram probabilities and calculates
   `KL(p || q)`.

Features with no usable values or no range are assigned zero drift. Non-numeric
columns and columns present in only one window are skipped. The `bins` argument
controls histogram resolution; use the same value when comparing runs.

KL is directional: this implementation measures baseline-to-production drift.
It is a ranking signal, not a statistical significance test or proof that a
feature caused a change in fraud. Small production windows and sparse features
can make histogram estimates noisy, so compare scores over meaningful windows
and inspect the underlying distributions before acting.

### 9.3 Cost-weighted ranking

`cost_weighted_drift(kl, importance, p_fraud, c_fn=420, c_fp=4)` calculates an
expected per-transaction loss:

```text
expected_loss = p_fraud × c_fn + (1 - p_fraud) × c_fp
cost_drift_usd(feature) = KL(feature) × importance(feature) × expected_loss
```

It rejects `p_fraud` values outside `[0, 1]` and negative cost inputs. An
importance value missing for a feature defaults to zero. Results are sorted by
`cost_drift_usd` descending and include `kl`, `importance`, each feature's
`cost_drift_usd`, and `cum_cost_drift`. The cumulative column supports choosing
an alert cutoff by aggregate ranked cost.

The default false-negative and false-positive costs (`$420` and `$4`) and
fraud-rate estimate are assumptions, not universal constants. Calibrate them
with the business owner and actual loss data. In the demo, `p_fraud=0.013` is a
sample value. `DAILY_DOLLAR_THRESHOLD=5000` is a policy starting point; connect
it to an explicit escalation rule rather than interpreting it as a model
metric.

### 9.4 Run drift analysis locally

Create a baseline Parquet window, a production window, and the importance JSON.
If the windows are not prepared yet, use section 9.1's
`demo/prepare_drift_windows.py` command. Run the same cost-weighted analysis
from the command line with the documented defaults (`bins=50`, `p_fraud=0.013`,
`c_fn=420`, `c_fp=4`):

```powershell
python demo/run_drift_analysis.py
```

The command reads the three files from `outputs/demo`, prints the top ten
features, and optionally writes the complete ranking to CSV:

```powershell
python demo/run_drift_analysis.py `
  --bins 50 `
  --p-fraud 0.013 `
  --c-fn 420 `
  --c-fp 4 `
  --top 10 `
  --output-csv outputs/demo/drift_ranking.csv
```

Use `--top 0` to print every feature. The command rejects empty windows or
importance JSON that is not a feature-to-weight object, and reports an error
if the windows have no shared numeric features. The underlying calculation
can also be run directly in Python:

```python
import json
import pandas as pd

from monitor.cost_drift import kl_per_feature, cost_weighted_drift

baseline = pd.read_parquet("outputs/demo/baseline_window.parquet")
production = pd.read_parquet("outputs/demo/production_window.parquet")
importance = json.loads(
    open("outputs/demo/shap_importance.json", encoding="utf-8").read()
)

kl = kl_per_feature(baseline, production, bins=50)
ranked = cost_weighted_drift(
    kl,
    importance,
    p_fraud=0.013,
    c_fn=420.0,
    c_fp=4.0,
)
print(ranked.head(10).to_string(index=False))
```

The command-line helper `demo/seed_drift_scenario.py` creates a synthetic
production window by shifting a configurable fraction of one numeric feature
by a multiple of its standard deviation:

```powershell
python demo/seed_drift_scenario.py `
  --baseline outputs/demo/baseline_window.parquet `
  --output outputs/demo/production_window.parquet `
  --feature merch_fraud_rate_30d
```

The default affects 15% of rows, shifts by 2.5 standard deviations, and uses a
fixed seed (`7`) for repeatable demo results. Override these with
`--affected-frac`, `--shift-std`, and `--seed`. Choose a feature that exists in
the baseline parquet and is numeric. Point `--output` at the production window
path from above, then run `demo/run_drift_analysis.py` to see the resulting
ranking.

### 9.5 Persist and forward alerts

`monitor/alerting.py::send_alert(top_features, total_expected_loss, ...)`
expects a non-empty ranked dataframe with `feature` and `cost_drift_usd`
columns. It takes the top-ranked feature and constructs this compact payload:

```json
{
  "total_expected_loss_usd": 123.45,
  "top_feature": "merch_fraud_rate_30d",
  "top_feature_cost_usd": 67.89
}
```

With no injected database client, it lazily creates a Firestore client and
adds a timestamped record to the `alerts` collection. Firestore credentials
come from Application Default Credentials (ADC) in the deployed environment;
local execution can inject a compatible client for testing. The caller should
compute and pass `total_expected_loss` from its selected alert window and
policy rather than assuming that it is identical to a single row's KL score.

When `ALERT_FUNCTION_URL` is set, or `alert_function_url` is passed explicitly,
the same payload is POSTed to the HTTP endpoint. The request has a 10-second
default timeout; non-success HTTP responses raise an exception. Firestore is
written before the HTTP call, so a notification failure leaves the alert
recorded for later review.

The HTTP function in `gcp/alert_function/main.py` currently validates the JSON
object and numeric `total_expected_loss_usd`, then returns a small success
response. It is a notification boundary stub: implement the approved Slack,
email, or ticketing destination there, and protect the endpoint with
appropriate authentication before production use. Deploy it only if the
monitoring process is configured with its URL. The monitoring process needs
Firestore write access and outbound access to the endpoint; the scorer VM does
not need those permissions just to serve predictions.

### 9.6 Streamlit analyst console

`demo/streamlit_app.py` presents two panels:

- **Score a transaction**: sends amount, `card1`, and `ProductCD` to the
  scorer's `POST /predict` endpoint and displays fraud probability and the
  model's threshold decision.
- **Drift leaderboard**: reads baseline and production Parquet plus the
  importance JSON, computes KL and cost-weighted rankings, charts the highest
  costs, and displays the top ten rows.

Configure these environment variables before launching:

```text
SCORER_URL=http://localhost:8080
DEMO_DATA_DIR=outputs/demo
```

For local development, start the scorer in one terminal and Streamlit in
another:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8080
streamlit run demo/streamlit_app.py
```

For a remote VM, set `SCORER_URL` to `http://VM_EXTERNAL_IP:8080` while testing.
The API currently uses plain HTTP and no authentication, so restrict the VM
firewall source range to the demo host/client. For a public or shared demo,
place the scorer behind authenticated HTTPS and configure `SCORER_URL` to that
URL. The demo's Parquet and JSON files must be available on the machine running
Streamlit; they are not fetched from GCS by this app.

If the files are missing, the app displays an informational prompt instead of
rendering the leaderboard. `DEMO_DATA_DIR` can point at a local directory or a
mounted/shared directory provided by an optional hosting service.

### 9.7 Suggested operating sequence

1. Establish a baseline window from a known-good period and preserve its
   feature schema.
2. Produce a recent production window using the same transformations.
3. Calculate KL divergence and cost-weighted rankings.
4. Review the top features and cumulative expected cost against the agreed
   escalation threshold.
5. Persist the alert in Firestore and send the optional HTTP notification.
6. Use the Streamlit console to inspect feature rankings and query the VM
   scorer with representative transactions.
7. Investigate data quality, upstream changes, and model performance before
   promoting a replacement model.

The drift calculation and alert transport are monitoring operations; they do
not automatically retrain or replace the champion model. Keep model promotion
as an explicit, evaluated step.

## 10. Security

The scorer currently exposes plain HTTP and does not implement authentication.
Do not leave port `8080` open to the world for a real deployment. Restrict the
firewall source range to known clients, or add an authenticated HTTPS proxy.
Use Secret Manager for API keys and grant secret access only to the workload
that needs each key.

Keep the local service-account JSON out of Git, `.dockerignore`, and Cloud Build
source archives. Use least-privilege IAM: the VM needs image read and model
read, while the deployment identity needs build and VM administration.

## 11. Cost and capacity

The always-on VM and persistent disk are the primary serving costs. Artifact
Registry storage, Cloud Storage, BigQuery, Firestore, logging, and optional
serverless services should be monitored against quotas and a small billing
budget alert.

The serving capacity is deliberately explicit: one VM, one container, one
worker. There is no application autoscaling or managed rollback. Scale up the
machine type when measured memory, throughput, or latency requires it. Keep
versioned model objects so a previous champion can be restored.

## 12. Rollout checklist

- [ ] Authenticate the CLI as a human deployment account.
- [ ] Enable the required APIs and configure the project, region, and zone.
- [ ] Create/reuse the Artifact Registry repository and GCS artifact bucket.
- [ ] Grant the VM service account image-reader and model-reader access.
- [ ] Run preprocessing, feature engineering, and Feast materialization.
- [ ] Train and validate the calibrated model locally.
- [ ] Publish `outputs/best-params.joblib` to the champion GCS path.
- [ ] Build and deploy with `deploy/vm/deploy-vm.ps1`.
- [ ] Verify `/health` and `/predict`.
- [ ] Restrict the firewall source range.
- [ ] Verify Docker logs and memory behavior.
- [ ] Retain a versioned model artifact for rollback.

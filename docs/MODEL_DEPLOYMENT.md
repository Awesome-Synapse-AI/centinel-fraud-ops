# Model Deployment in GCP

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

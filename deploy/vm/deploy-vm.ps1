param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,
    [string]$Region = "us-central1",
    [string]$Zone = "us-central1-a",
    [string]$VmName = "fraud-scorer-vm",
    [string]$Repository = "fraud-images",
    [string]$ImageName = "fraud-scorer",
    [string]$ModelGcsUri = "",
    [string]$ServiceAccount = "",
    [string]$SourceRange = "0.0.0.0/0"
)

# gcloud on Windows can write informational YAML fields (for example,
# "Encryption: Google-managed key") through the native stderr stream. Do not
# let PowerShell convert that normal output into a terminating exception; the
# wrapper below checks gcloud's actual exit code.
$ErrorActionPreference = "Continue"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$buildConfig = Join-Path $repoRoot "deploy\vm\cloudbuild.yaml"

function Invoke-Gcloud {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & gcloud @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud failed with exit code $LASTEXITCODE"
    }
}

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud CLI is required. Install Google Cloud CLI and run gcloud auth login first."
}

Invoke-Gcloud config set project $ProjectId
Invoke-Gcloud services enable compute.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

$repoExists = $true
& gcloud artifacts repositories describe $Repository --location=$Region --project=$ProjectId *> $null
if ($LASTEXITCODE -ne 0) { $repoExists = $false }
if (-not $repoExists) {
    Invoke-Gcloud artifacts repositories create $Repository --repository-format=docker --location=$Region --project=$ProjectId
}

$image = "$Region-docker.pkg.dev/$ProjectId/$Repository/${ImageName}:latest"
Write-Host "Building and pushing $image"
Invoke-Gcloud builds submit --config=$buildConfig `
    --substitutions="_IMAGE=$image" `
    --project=$ProjectId `
    $repoRoot

if ([string]::IsNullOrWhiteSpace($ModelGcsUri)) {
    $ModelGcsUri = "gs://$ProjectId-artifacts/models/champion/model.joblib"
}

if ([string]::IsNullOrWhiteSpace($ServiceAccount)) {
    $projectNumber = (& gcloud projects describe $ProjectId --format="value(projectNumber)").Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($projectNumber)) {
        throw "Could not determine the project number. Pass -ServiceAccount explicitly."
    }
    $ServiceAccount = "$projectNumber-compute@developer.gserviceaccount.com"
}

Write-Host "Granting the VM service account access to the image and model bucket"
Invoke-Gcloud projects add-iam-policy-binding $ProjectId `
    --member="serviceAccount:$ServiceAccount" `
    --role="roles/artifactregistry.reader" `
    --quiet
Invoke-Gcloud projects add-iam-policy-binding $ProjectId `
    --member="serviceAccount:$ServiceAccount" `
    --role="roles/storage.objectViewer" `
    --quiet

$vmExists = $true
& gcloud compute instances describe $VmName --zone=$Zone --project=$ProjectId *> $null
if ($LASTEXITCODE -ne 0) { $vmExists = $false }
if (-not $vmExists) {
    Write-Host "Creating $VmName as an e2-micro"
    Invoke-Gcloud compute instances create $VmName `
        --zone=$Zone `
        --machine-type=e2-micro `
        --image-family=debian-12 `
        --image-project=debian-cloud `
        --boot-disk-size=30GB `
        --boot-disk-type=pd-standard `
        --service-account=$ServiceAccount `
        --scopes=cloud-platform `
        --tags=fraud-scorer `
        --project=$ProjectId
}

$firewallExists = $true
& gcloud compute firewall-rules describe allow-fraud-scorer --project=$ProjectId *> $null
if ($LASTEXITCODE -ne 0) { $firewallExists = $false }
if (-not $firewallExists) {
    Invoke-Gcloud compute firewall-rules create allow-fraud-scorer `
        --allow=tcp:8080 `
        --target-tags=fraud-scorer `
        --source-ranges=$SourceRange `
        --project=$ProjectId
}

$remoteCommand = @"
set -e
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo gcloud auth configure-docker $Region-docker.pkg.dev --quiet
sudo docker pull $image
sudo docker rm -f fraud-scorer 2>/dev/null || true
sudo docker run -d --name fraud-scorer --restart unless-stopped -p 8080:8080 -e MODEL_GCS_URI=$ModelGcsUri $image
sudo docker ps --filter name=fraud-scorer
"@

Write-Host "Installing Docker and starting the scorer on $VmName"
Invoke-Gcloud compute ssh $VmName --zone=$Zone --project=$ProjectId --command=$remoteCommand

$externalIp = (& gcloud compute instances describe $VmName --zone=$Zone --project=$ProjectId --format="value(networkInterfaces[0].accessConfigs[0].natIP)").Trim()
Write-Host "Scorer deployed. Health check: http://$externalIp`:8080/health"

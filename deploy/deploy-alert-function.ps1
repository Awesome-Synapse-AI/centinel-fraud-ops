param(
    [string]$ProjectId = "fraud-cost-pipeline",
    [string]$Region = "us-central1",
    [string]$FunctionName = "fraud-drift-alert"
)

$ErrorActionPreference = "Continue"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$source = Join-Path $repoRoot "gcp\alert_function"

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
Invoke-Gcloud services enable cloudfunctions.googleapis.com run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

Write-Host "Deploying IAM-protected HTTP function $FunctionName to $ProjectId/$Region"
Invoke-Gcloud functions deploy $FunctionName `
    --gen2 `
    --runtime=python311 `
    --region=$Region `
    --source=$source `
    --entry-point=notify `
    --trigger-http `
    --no-allow-unauthenticated `
    --project=$ProjectId `
    --quiet

$endpoint = (& gcloud functions describe $FunctionName `
    --gen2 `
    --region=$Region `
    --project=$ProjectId `
    --format="value(serviceConfig.uri)").Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($endpoint)) {
    throw "Function deployed, but gcloud could not read its HTTPS endpoint."
}

Write-Host "Alert function deployed with IAM authentication."
Write-Host "Endpoint: $endpoint"
Write-Host "Set for this PowerShell session with: `$env:ALERT_FUNCTION_URL = `"$endpoint`""

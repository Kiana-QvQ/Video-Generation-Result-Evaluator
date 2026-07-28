param(
    [switch]$Public,
    [string]$BindHost,
    [int]$Port = 50051
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Project environment is missing. Run .\setup.ps1 first."
}

$hostAddress = if ($BindHost) {
    $BindHost
} elseif ($Public) {
    "0.0.0.0"
} else {
    "127.0.0.1"
}

$env:PYTHONNOUSERSITE = "1"
$env:EVALUATOR_FACE_DEVICE = "auto"
$env:EVALUATOR_IQA_DEVICE = "auto"
$env:EVALUATOR_SEMANTIC_DEVICE = "auto"
$env:EVALUATOR_GRPC_HOST = $hostAddress
$env:EVALUATOR_GRPC_PORT = [string]$Port

Set-Location $root
Write-Host "gRPC listening on ${hostAddress}:${Port}"
& $python (Join-Path $root "start.py") --transport grpc
exit $LASTEXITCODE

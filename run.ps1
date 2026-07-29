param(
    [switch]$Public,
    [switch]$WithGrpc,
    [switch]$WithVlm,
    [ValidateSet("2b", "2.5-3b")]
    [string]$VlmModel = "2b",
    [string]$BindHost,
    [int]$Port = 7860,
    [int]$GrpcPort = 50051
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
$env:EVALUATOR_HOST = $hostAddress
$env:EVALUATOR_PORT = [string]$Port

Set-Location $root
$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listeners) {
    $pids = ($listeners | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
    throw "Port $Port is already in use by PID $pids. Stop the existing service before starting another instance."
}
$startArguments = @()
if ($WithGrpc) {
    $startArguments += "--with-grpc"
    $startArguments += "--grpc-port"
    $startArguments += [string]$GrpcPort
    Write-Host "Starting HTTP on ${hostAddress}:${Port} and gRPC on ${hostAddress}:${GrpcPort}"
} else {
    Write-Host "Starting HTTP on http://${hostAddress}:${Port}"
}
if ($WithVlm) {
    $startArguments += "--with-vlm"
    $startArguments += "--vlm-model"
    $startArguments += $VlmModel
}
& $python (Join-Path $root "start.py") @startArguments
exit $LASTEXITCODE

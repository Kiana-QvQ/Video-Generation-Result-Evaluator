param(
    [switch]$Public,
    [switch]$WithVlm,
    [ValidateSet("2b", "2.5-3b")]
    [string]$VlmModel = "2b",
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

Set-Location $root
$startArguments = @("--host", $hostAddress, "--port", [string]$Port)
if ($WithVlm) {
    $startArguments += "--with-vlm"
    $startArguments += "--vlm-model"
    $startArguments += $VlmModel
}
& $python (Join-Path $root "start_grpc.py") @startArguments
exit $LASTEXITCODE

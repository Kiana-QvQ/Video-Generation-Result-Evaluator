$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

$env:PYTHONNOUSERSITE = "1"
$env:EVALUATOR_FACE_DEVICE = "cpu"
$env:EVALUATOR_IQA_DEVICE = "cpu"
$env:EVALUATOR_SEMANTIC_DEVICE = "auto"

if (-not (Test-Path $python)) {
    throw "Project environment is missing. Run .\setup.ps1 first."
}

Set-Location $root
& $python -m uvicorn web_app:app --host 127.0.0.1 --port 7860

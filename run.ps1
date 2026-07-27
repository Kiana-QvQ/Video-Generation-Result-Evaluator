$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

$env:PYTHONNOUSERSITE = "1"
$env:EVALUATOR_FACE_DEVICE = "auto"
$env:EVALUATOR_IQA_DEVICE = "auto"
$env:EVALUATOR_SEMANTIC_DEVICE = "auto"

if (-not (Test-Path $python)) {
    throw "Project environment is missing. Run .\setup.ps1 first."
}

Set-Location $root
try {
    $providers = (& $python -c "import onnxruntime as ort; print(','.join(ort.get_available_providers()))").Trim()
    Write-Host "ONNX Runtime providers: $providers"
    if ($env:EVALUATOR_FACE_DEVICE -eq "auto" -and $providers -notlike "*CUDAExecutionProvider*") {
        Write-Warning "ArcFace will run on CPU: CUDAExecutionProvider is not installed. Run .\setup.ps1 -Optional."
    }
} catch {
    Write-Warning "ONNX Runtime is not installed. ArcFace will use its fallback until .\setup.ps1 -Optional is run."
}
& $python -m uvicorn web_app:app --host 127.0.0.1 --port 7860

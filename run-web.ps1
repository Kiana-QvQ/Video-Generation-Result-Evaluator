$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$cache = Join-Path $root "model_cache"

$env:PYTHONNOUSERSITE = "1"
$env:TORCH_HOME = $cache
$env:HF_HOME = Join-Path $cache "huggingface"
$env:HF_HUB_CACHE = Join-Path $cache "huggingface\hub"
$env:HF_DATASETS_CACHE = Join-Path $cache "huggingface\datasets"
$env:TRANSFORMERS_CACHE = Join-Path $cache "huggingface\transformers"
$env:TORCH_EXTENSIONS_DIR = Join-Path $cache "torch_extensions"
$env:MPLCONFIGDIR = Join-Path $cache "matplotlib"
$env:GRADIO_TEMP_DIR = Join-Path $root "outputs\gradio_temp"
$env:EVALUATOR_FACE_DEVICE = "cpu"
$env:EVALUATOR_IQA_DEVICE = "cpu"
$env:EVALUATOR_SEMANTIC_DEVICE = "cpu"

if (-not (Test-Path $python)) {
    throw "Project environment is missing. Run .\setup.ps1 first."
}

Set-Location $root
& $python -m uvicorn web_app:app --host 127.0.0.1 --port 7860

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$cache = Join-Path $root "model_cache"

$env:PYTHONNOUSERSITE = "1"
$env:TORCH_HOME = $cache
$env:HF_HOME = Join-Path $cache "huggingface"
$env:HF_HUB_CACHE = Join-Path $cache "huggingface\hub"
$env:TRANSFORMERS_CACHE = Join-Path $cache "huggingface\transformers"
$env:VBENCH_CACHE_DIR = Join-Path $cache "vbench"
$env:TORCH_EXTENSIONS_DIR = Join-Path $cache "torch_extensions"
$env:MPLCONFIGDIR = Join-Path $cache "matplotlib"
$env:PIP_CACHE_DIR = Join-Path $cache "pip"

if (-not (Test-Path $python)) {
    throw "Project environment is missing. Run .\setup.ps1 first."
}

New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR | Out-Null
& $python -m pip install -r (Join-Path $root "requirements-vbench.txt")
Write-Host "VBench package dependencies were installed into .venv."
Write-Host "VBench model weights will be stored under model_cache\vbench."

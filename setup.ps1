param(
    [switch]$Cuda,
    [switch]$Optional,
    [switch]$VBench
)

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
$env:VBENCH_CACHE_DIR = Join-Path $cache "vbench"
$env:TORCH_EXTENSIONS_DIR = Join-Path $cache "torch_extensions"
$env:MPLCONFIGDIR = Join-Path $cache "matplotlib"
$env:PIP_CACHE_DIR = Join-Path $cache "pip"

if (-not (Test-Path $python)) {
    Write-Host "Creating project-local Python environment..."
    python -m venv (Join-Path $root ".venv")
}

Write-Host "Installing base dependencies into .venv..."
New-Item -ItemType Directory -Force -Path $cache | Out-Null
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR | Out-Null
& $python -m pip install --upgrade pip
if ($Cuda) {
    Write-Host "Installing the CUDA-enabled PyTorch build..."
    & $python -m pip install `
        --index-url https://download.pytorch.org/whl/cu118 `
        torch==2.5.1 torchvision==0.20.1
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA PyTorch installation failed. The CPU environment was not changed."
    }
}
& $python -m pip install -r (Join-Path $root "requirements.txt")
if ($Optional) {
    Write-Host "Installing optional exact evaluator backends..."
    & $python -m pip install -r (Join-Path $root "requirements\optional.txt")
}
if ($VBench) {
    Write-Host "Installing the optional VBench backend..."
    & $python -m pip install -r (Join-Path $root "requirements\vbench.txt")
}

Write-Host ""
Write-Host "Project environment is ready."
Write-Host "Start with: .\run.ps1"
if ($Optional) {
    Write-Host "Optional evaluator packages are ready."
    Write-Host "Download weights with: .\scripts\download-optional-assets.ps1 -SkipPythonPackages"
}
if ($VBench) {
    Write-Host "VBench package is ready. Download its assets with: .\scripts\download-vbench-models.ps1"
}
if ($Cuda) {
    Write-Host "CUDA mode requested. Verify with: .\.venv\Scripts\python.exe -c `"import torch; print(torch.version.cuda, torch.cuda.is_available())`""
}

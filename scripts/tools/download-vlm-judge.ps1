param(
    [ValidateSet("2b", "2.5-3b")]
    [string]$JudgeModel = "2b"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$modelSpec = @{
    "2b" = @{
        Repo = "Qwen/Qwen2-VL-2B-Instruct-AWQ"
        Name = "Qwen2-VL-2B-Instruct-AWQ"
    }
    "2.5-3b" = @{
        Repo = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
        Name = "Qwen2.5-VL-3B-Instruct-AWQ"
    }
}[$JudgeModel]
$target = Join-Path $root "model_cache\vlm_judge\$($modelSpec.Name)"

if (-not (Test-Path $python)) {
    throw "Project Python environment is missing. Run .\setup.ps1 first."
}

$env:HF_HOME = Join-Path $root "model_cache\huggingface"
$env:HF_HUB_CACHE = Join-Path $root "model_cache\huggingface\hub"
$env:TRANSFORMERS_CACHE = Join-Path $root "model_cache\huggingface\transformers"

& $python (Join-Path $root "tools\download_hf_snapshot.py") `
    $modelSpec.Repo `
    $target

Write-Host "VLM Judge ($JudgeModel) is stored under: $target"

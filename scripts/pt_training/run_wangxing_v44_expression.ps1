$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Pipeline = Join-Path $Root "scripts\pt_training\run_wangxing_v44_pipeline.py"
Push-Location $Root
try {
    Write-Host "[PT v4.4] Explicit 85% expression + 15% face-crop candidate." -ForegroundColor Cyan
    & $Python $Pipeline --device cuda --epochs 80 --batch-size 16 --learning-rate 3e-4 --seed 42
    exit $LASTEXITCODE
}
finally { Pop-Location }

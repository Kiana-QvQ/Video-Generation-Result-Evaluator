$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pipeline = Join-Path $ProjectRoot "scripts\pt_training\run_wangxing_v42_pipeline.py"

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "outputs\vedio_pred\models\wangxing_v42_expression_res1k.pt"))) {
    throw "v4.2 model was not found."
}

Push-Location $ProjectRoot
try {
    Write-Host "[PT v4.2] Evaluate-only mode; no retraining." -ForegroundColor Cyan
    & $Python $Pipeline `
      --evaluate-only `
      --model-path outputs/vedio_pred/models/wangxing_v42_expression_res1k.pt `
      --official-metrics outputs/forensics/wangxing_v42_expression_official_holdout_metrics.json `
      --test-manifest-root outputs/vedio_pred/wangxing_v42_expression_test_manifests `
      --report outputs/vedio_pred/wangxing_v42_expression_evaluation_report.json `
      --test-set "25+25" data/test/single_video `
      --test-set "32+32" data/test/wangxing_32x32 `
      --device cuda
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

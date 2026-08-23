$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pipeline = Join-Path $ProjectRoot "scripts\pt_training\run_wangxing_v43_pipeline.py"

Push-Location $ProjectRoot
try {
    Write-Host "[PT v4.3] Expression + face-crop candidate; old models unchanged." -ForegroundColor Cyan
    & $Python $Pipeline `
      --base-manifest outputs/vedio_pred/wangxing_v3_generalization_manifest_res1k.json `
      --v43-manifest outputs/vedio_pred/wangxing_v43_expression_generalization_manifest_res1k.json `
      --augmentation-root data/_aug/wangxing_v43_expression_photometric `
      --cache-dir outputs/vedio_pred/cache_wangxing_v43_expression_res1k `
      --model-path outputs/vedio_pred/models/wangxing_v43_expression_res1k.pt `
      --train-metrics outputs/vedio_pred/wangxing_v43_expression_metrics_res1k.json `
      --official-metrics outputs/forensics/wangxing_v43_expression_official_holdout_metrics.json `
      --test-manifest-root outputs/vedio_pred/wangxing_v43_expression_test_manifests `
      --report outputs/vedio_pred/wangxing_v43_expression_pipeline_report.json `
      --test-set "25+25" data/test/single_video `
      --test-set "32+32" data/test/wangxing_32x32 `
      --device cuda `
      --epochs 80 `
      --batch-size 16 `
      --learning-rate 3e-4 `
      --seed 42
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pipeline = Join-Path $ProjectRoot "scripts\pt_training\run_wangxing_v41_pipeline.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

$Arguments = @(
    $Pipeline,
    "--base-manifest", "outputs/vedio_pred/wangxing_v3_generalization_manifest_res1k.json",
    "--v41-manifest", "outputs/vedio_pred/wangxing_v41_expression_generalization_manifest_res1k.json",
    "--augmentation-root", "data/_aug/wangxing_v41_expression_photometric",
    "--cache-dir", "outputs/vedio_pred/cache_wangxing_v41_expression_res1k",
    "--model-path", "outputs/vedio_pred/models/wangxing_v41_expression_res1k.pt",
    "--train-metrics", "outputs/vedio_pred/wangxing_v41_expression_metrics_res1k.json",
    "--official-metrics", "outputs/forensics/wangxing_v41_expression_official_holdout_metrics.json",
    "--test-manifest-root", "outputs/vedio_pred/wangxing_v41_expression_test_manifests",
    "--report", "outputs/vedio_pred/wangxing_v41_expression_pipeline_report.json",
    "--test-set", "25+25", "data/test/single_video",
    "--test-set", "32+32", "data/test/wangxing_32x32",
    "--device", "cuda",
    "--epochs", "80",
    "--batch-size", "16",
    "--learning-rate", "3e-4",
    "--seed", "42"
)

Push-Location $ProjectRoot
try {
    Write-Host "[PT v4.1] Expression-only candidate; old models stay unchanged." -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[PT v4.1] Pipeline failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

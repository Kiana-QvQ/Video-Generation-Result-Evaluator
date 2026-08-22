$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pipeline = Join-Path $ProjectRoot "scripts\pt_training\run_wangxing_v4_pipeline.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "outputs\vedio_pred\models\wangxing_v4_expression_res1k.pt"))) {
    throw "v4 expression model was not found."
}

$Arguments = @(
    $Pipeline,
    "--evaluate-only",
    "--model-path", "outputs/vedio_pred/models/wangxing_v4_expression_res1k.pt",
    "--official-metrics", "outputs/forensics/wangxing_v4_expression_official_holdout_metrics.json",
    "--test-manifest-root", "outputs/vedio_pred/wangxing_v4_expression_test_manifests",
    "--report", "outputs/vedio_pred/wangxing_v4_expression_evaluation_report.json",
    "--source-profile", "outputs/forensics/wangxing_source_profile_web_v3_test_excluded.json",
    "--forensics-profile", "outputs/forensics/forensics_profiles_web_v3_test_excluded.json",
    "--test-set", "25+25", "data/test/single_video",
    "--test-set", "32+32", "data/test/wangxing_32x32",
    "--device", "cuda"
)

Push-Location $ProjectRoot
try {
    Write-Host "[PT v4] Evaluating existing model only..." -ForegroundColor Cyan
    Write-Host "[PT v4] No retraining will be performed." -ForegroundColor DarkCyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[PT v4] Evaluation failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Host "[PT v4] Evaluation completed." -ForegroundColor Green
}
finally {
    Pop-Location
}

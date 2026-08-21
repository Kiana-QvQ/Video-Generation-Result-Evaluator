$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pipeline = Join-Path $ProjectRoot "scripts\pt_training\run_wangxing_v4_pipeline.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

$Arguments = @(
    $Pipeline,
    "--base-manifest", "outputs/vedio_pred/wangxing_v3_generalization_manifest_res1k.json",
    "--v4-manifest", "outputs/vedio_pred/wangxing_v4_expression_generalization_manifest_res1k.json",
    "--augmentation-root", "data/_aug/wangxing_v4_expression_photometric",
    "--cache-dir", "outputs/vedio_pred/cache_wangxing_v4_expression_res1k",
    "--model-path", "outputs/vedio_pred/models/wangxing_v4_expression_res1k.pt",
    "--train-metrics", "outputs/vedio_pred/wangxing_v4_expression_metrics_res1k.json",
    "--official-metrics", "outputs/forensics/wangxing_v4_expression_official_holdout_metrics.json",
    "--test-manifest-root", "outputs/vedio_pred/wangxing_v4_expression_test_manifests",
    "--report", "outputs/vedio_pred/wangxing_v4_expression_pipeline_report.json",
    "--source-profile", "outputs/forensics/wangxing_source_profile_web_v3_test_excluded.json",
    "--forensics-profile", "outputs/forensics/forensics_profiles_web_v3_test_excluded.json",
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
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

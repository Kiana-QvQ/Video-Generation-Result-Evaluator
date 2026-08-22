$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pipeline = Join-Path $ProjectRoot "scripts\web_forensics\run_web_forensics_v2.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

$Arguments = @(
    $Pipeline,
    "all",
    "--dataset-root", "data/test/web_forensics_v41",
    "--split-manifest", "outputs/vedio_pred/wangxing_dual_pt_split_res1k.json",
    "--profile-exclusion", "data/forensics/web_forensics_v41_profile_exclusion.json",
    "--base-profile-exclusion", "data/forensics/web_forensics_v2_profile_exclusion.json",
    "--forensics-profile", "outputs/forensics/forensics_profiles_web_v41_test_excluded.json",
    "--source-profile", "outputs/forensics/wangxing_source_profile_web_v41_test_excluded.json",
    "--calibrator", "outputs/forensics/forensics_authenticity_calibrator_web_v41.json",
    "--fusion-head", "outputs/forensics/web_forensics_fusion_v41_expression.json",
    "--feature-cache", "outputs/forensics/web_forensics_v41_expression_feature_cache.npz",
    "--weighted-policy", "outputs/forensics/wangxing_authenticity_weighted_policy_v41.json",
    "--ranking-root", "C:\Users\zhanghaotian\Desktop\ppt_video",
    "--ranking-cache-root", "outputs/forensics/ppt_video_wangxing_policy_v41_cache",
    "--ranking-output-root", "outputs/forensics/ppt_video_wangxing_policy_v41",
    "--test-set", "25+25", "data/test/web_forensics_v41/single_video/manifest_25x25.json",
    "--test-set", "32+32", "data/test/wangxing_32x32/single_video/manifest.json",
    "--expression-only",
    "--transition-features",
    "--device", "cuda",
    "--wangxing-device", "cuda",
    "--profile-max-videos", "120",
    "--output-root", "outputs/forensics/web_forensics_v41_results"
)

Push-Location $ProjectRoot
try {
    Write-Host "[Web v4.1] Expression-only candidate; current web is unchanged." -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[Web v4.1] Pipeline failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

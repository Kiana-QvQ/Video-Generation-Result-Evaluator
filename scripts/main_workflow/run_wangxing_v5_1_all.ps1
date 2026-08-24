$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Calibrate = Join-Path $ProjectRoot "scripts\pt_training\calibrate_wangxing_v5_realness.py"
$Evaluate = Join-Path $ProjectRoot "scripts\pt_training\evaluate_wangxing_v5_realness.py"
$Web = Join-Path $ProjectRoot "scripts\web_forensics\run_wangxing_web_v51.py"

$RequiredArtifacts = @(
    "outputs\vedio_pred\models\wangxing_v3_res1k.pt",
    "outputs\vedio_pred\models\wangxing_v5_drive.json",
    "outputs\forensics\forensics_profiles_web_v3_test_excluded.json",
    "outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json"
)

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

Push-Location $ProjectRoot
try {
    Write-Host "[V5.1 stage 0/4] Checking V5.0 artifacts..." -ForegroundColor Cyan
    foreach ($artifact in $RequiredArtifacts) {
        if (-not (Test-Path -LiteralPath $artifact)) {
            throw "Missing required artifact: $artifact"
        }
    }
    Write-Host "[V5.1 stage 0/4] V5.0 artifacts present." -ForegroundColor Green

    Write-Host "[V5.1 stage 1/4] Calibrate on ppt test1 only..." -ForegroundColor Cyan
    & $Python $Calibrate `
        --ranking-root "C:\Users\zhanghaotian\Desktop\ppt_video" `
        --fit-group test1 `
        --holdout-group test2 `
        --drive-model outputs\vedio_pred\models\wangxing_v5_drive.json `
        --v3-model outputs\vedio_pred\models\wangxing_v3_res1k.pt `
        --forensics-profile outputs\forensics\forensics_profiles_web_v3_test_excluded.json `
        --source-profile outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json `
        --cache-dir outputs\forensics\cache_wangxing_v5_1_ppt `
        --au-output-root outputs\forensics\cache_wangxing_v5_1_ppt\au `
        --device cuda `
        --wangxing-device cuda `
        --output outputs\forensics\wangxing_v5_realness_calibrator.json
    if ($LASTEXITCODE -ne 0) { throw "V5.1 calibration failed: $LASTEXITCODE" }

    Write-Host "[V5.1 stage 2/4] PT holdout and binary regression..." -ForegroundColor Cyan
    & $Python $Evaluate `
        --calibrator outputs\forensics\wangxing_v5_realness_calibrator.json `
        --holdout-group test2 `
        --ranking-root "C:\Users\zhanghaotian\Desktop\ppt_video" `
        --v3-model outputs\vedio_pred\models\wangxing_v3_res1k.pt `
        --drive-model outputs\vedio_pred\models\wangxing_v5_drive.json `
        --forensics-profile outputs\forensics\forensics_profiles_web_v3_test_excluded.json `
        --source-profile outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json `
        --cache-dir outputs\forensics\cache_wangxing_v5_1_ppt `
        --au-output-root outputs\forensics\cache_wangxing_v5_1_ppt\au `
        --output-root outputs\vedio_pred\wangxing_v5_1_results `
        --min-pairwise 0.8333333333 `
        --enforce-gates `
        --device cuda `
        --wangxing-device cuda `
        --test-set "25+25" data\test\single_video\manifest.json `
        --test-set "32+32" data\test\wangxing_32x32\single_video\manifest.json
    if ($LASTEXITCODE -ne 0) { throw "V5.1 PT evaluation failed: $LASTEXITCODE" }

    Write-Host "[V5.1 stage 3/4] Offline Web evaluation..." -ForegroundColor Cyan
    & $Python $Web `
        --calibrator outputs\forensics\wangxing_v5_realness_calibrator.json `
        --holdout-group test2 `
        --ranking-root "C:\Users\zhanghaotian\Desktop\ppt_video" `
        --v3-model outputs\vedio_pred\models\wangxing_v3_res1k.pt `
        --drive-model outputs\vedio_pred\models\wangxing_v5_drive.json `
        --forensics-profile outputs\forensics\forensics_profiles_web_v3_test_excluded.json `
        --source-profile outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json `
        --cache-dir outputs\forensics\cache_wangxing_v5_1_web `
        --au-output-root outputs\forensics\cache_wangxing_v5_1_web\au `
        --output-root outputs\forensics\wangxing_v5_1_web_results `
        --min-pairwise 0.8333333333 `
        --enforce-gates `
        --device cuda `
        --wangxing-device cuda `
        --test-set "25+25" data\test\single_video\manifest.json `
        --test-set "32+32" data\test\wangxing_32x32\single_video\manifest.json
    if ($LASTEXITCODE -ne 0) { throw "V5.1 Web evaluation failed: $LASTEXITCODE" }

    Write-Host "[V5.1 stage 4/4] Offline pipeline completed; online Web remains V3." -ForegroundColor Green
}
finally {
    Pop-Location
}

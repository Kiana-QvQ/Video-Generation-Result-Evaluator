$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Build = Join-Path $ProjectRoot "scripts\pt_training\build_wangxing_v5_2_ranking_manifest.py"
$Train = Join-Path $ProjectRoot "scripts\pt_training\train_wangxing_v5_rank.py"
$Evaluate = Join-Path $ProjectRoot "scripts\pt_training\evaluate_wangxing_v5_rank.py"
$Web = Join-Path $ProjectRoot "scripts\web_forensics\run_wangxing_web_v52.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

Push-Location $ProjectRoot
try {
    Write-Host "[V5.2 stage 0/5] Checking V5.1 artifacts..." -ForegroundColor Cyan
    $required = @(
        "outputs\vedio_pred\models\wangxing_v3_res1k.pt",
        "outputs\vedio_pred\models\wangxing_v5_drive.json",
        "outputs\forensics\wangxing_v5_realness_calibrator.json",
        "outputs\forensics\forensics_profiles_web_v3_test_excluded.json",
        "outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json"
    )
    foreach ($relative in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $relative))) {
            throw "Missing V5.1 prerequisite: $relative"
        }
    }

    Write-Host "[V5.2 stage 1/5] Building grouped ranking manifest..." -ForegroundColor Cyan
    & $Python $Build `
        --ppt-root "C:\Users\zhanghaotian\Desktop\ppt_video" `
        --ltx-root "C:\Users\zhanghaotian\Desktop\LTX" `
        --holdout-group ppt_test2 `
        --output data\ranking\wangxing_v5_2\manifest.json
    if ($LASTEXITCODE -ne 0) { throw "Manifest stage failed: $LASTEXITCODE" }

    Write-Host "[V5.2 stage 2/5] Fitting linear pairwise RankHead..." -ForegroundColor Cyan
    & $Python $Train `
        --manifest data\ranking\wangxing_v5_2\manifest.json `
        --calibrator outputs\forensics\wangxing_v5_realness_calibrator.json `
        --v3-model outputs\vedio_pred\models\wangxing_v3_res1k.pt `
        --drive-model outputs\vedio_pred\models\wangxing_v5_drive.json `
        --forensics-profile outputs\forensics\forensics_profiles_web_v3_test_excluded.json `
        --source-profile outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json `
        --cache-dir outputs\forensics\cache_wangxing_v5_2 `
        --au-output-root outputs\forensics\cache_wangxing_v5_2\au `
        --device cuda `
        --wangxing-device cuda `
        --C 0.5 `
        --seed 42 `
        --output outputs\forensics\wangxing_v5_2_rank_policy.json
    if ($LASTEXITCODE -ne 0) { throw "Rank training stage failed: $LASTEXITCODE" }

    Write-Host "[V5.2 stage 3/5] Evaluating PT holdout and binary regressions..." -ForegroundColor Cyan
    & $Python $Evaluate `
        --manifest data\ranking\wangxing_v5_2\manifest.json `
        --rank-policy outputs\forensics\wangxing_v5_2_rank_policy.json `
        --calibrator outputs\forensics\wangxing_v5_realness_calibrator.json `
        --v3-model outputs\vedio_pred\models\wangxing_v3_res1k.pt `
        --drive-model outputs\vedio_pred\models\wangxing_v5_drive.json `
        --forensics-profile outputs\forensics\forensics_profiles_web_v3_test_excluded.json `
        --source-profile outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json `
        --cache-dir outputs\forensics\cache_wangxing_v5_2 `
        --au-output-root outputs\forensics\cache_wangxing_v5_2\au `
        --output-root outputs\vedio_pred\wangxing_v5_2_results `
        --min-pairwise 0.8333333333 `
        --test-set "25+25" data\test\single_video\manifest.json `
        --test-set "32+32" data\test\wangxing_32x32\single_video\manifest.json `
        --enforce-gates
    if ($LASTEXITCODE -ne 0) { throw "PT evaluation gate failed: $LASTEXITCODE" }

    Write-Host "[V5.2 stage 4/5] Running offline Web-equivalent evaluation..." -ForegroundColor Cyan
    & $Python $Web `
        --manifest data\ranking\wangxing_v5_2\manifest.json `
        --rank-policy outputs\vedio_pred\wangxing_v5_2_results\rank_policy_validated.json `
        --calibrator outputs\forensics\wangxing_v5_realness_calibrator.json `
        --v3-model outputs\vedio_pred\models\wangxing_v3_res1k.pt `
        --drive-model outputs\vedio_pred\models\wangxing_v5_drive.json `
        --forensics-profile outputs\forensics\forensics_profiles_web_v3_test_excluded.json `
        --source-profile outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json `
        --cache-dir outputs\forensics\cache_wangxing_v5_2_web `
        --au-output-root outputs\forensics\cache_wangxing_v5_2_web\au `
        --output-root outputs\forensics\wangxing_v5_2_web_results `
        --min-pairwise 0.8333333333 `
        --test-set "25+25" data\test\single_video\manifest.json `
        --test-set "32+32" data\test\wangxing_32x32\single_video\manifest.json `
        --enforce-gates
    if ($LASTEXITCODE -ne 0) { throw "Web evaluation gate failed: $LASTEXITCODE" }

    Write-Host "[V5.2 stage 5/5] Completed. Online Web remains V3." -ForegroundColor Green
}
finally {
    Pop-Location
}

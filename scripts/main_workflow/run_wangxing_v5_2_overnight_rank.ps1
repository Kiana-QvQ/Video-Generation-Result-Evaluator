$ErrorActionPreference = "Stop"

# Overnight rank-focused run:
# - rebuild manifest + refit RankHead (cache-friendly)
# - evaluate holdout / same-prompt ppt groups with role-anchored real display
# - SKIP 25+25 / 32+32 / Web (already passed earlier tonight)
# Morning check: outputs\vedio_pred\wangxing_v5_2_results_overnight\leadership_brief.json

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Build = Join-Path $ProjectRoot "scripts\pt_training\build_wangxing_v5_2_ranking_manifest.py"
$Train = Join-Path $ProjectRoot "scripts\pt_training\train_wangxing_v5_rank.py"
$Evaluate = Join-Path $ProjectRoot "scripts\pt_training\evaluate_wangxing_v5_rank.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

Push-Location $ProjectRoot
try {
    Write-Host "[V5.2 overnight 0/3] Checking prerequisites..." -ForegroundColor Cyan
    $required = @(
        "outputs\vedio_pred\models\wangxing_v3_res1k.pt",
        "outputs\vedio_pred\models\wangxing_v5_drive.json",
        "outputs\forensics\wangxing_v5_realness_calibrator.json",
        "outputs\forensics\forensics_profiles_web_v3_test_excluded.json",
        "outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json"
    )
    foreach ($relative in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $relative))) {
            throw "Missing prerequisite: $relative"
        }
    }

    Write-Host "[V5.2 overnight 1/3] Building ranking manifest..." -ForegroundColor Cyan
    & $Python $Build `
        --ppt-root "C:\Users\zhanghaotian\Desktop\ppt_video" `
        --ltx-root "data\LTX" `
        --holdout-group ppt_test2 `
        --complete-from-pools `
        --min-complete-train 5 `
        --real-pool "data\MD_CL" `
        --real-pool "data\video" `
        --seedance-pool "data\WangXing_Seedance" `
        --output data\ranking\wangxing_v5_2\manifest.json `
        --completion-report data\ranking\wangxing_v5_2\completion_report.json
    if ($LASTEXITCODE -ne 0) { throw "Manifest stage failed: $LASTEXITCODE" }

    Write-Host "[V5.2 overnight 2/3] Fitting RankHead (reuse feature cache)..." -ForegroundColor Cyan
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
        --output outputs\forensics\wangxing_v5_2_rank_policy_overnight.json
    if ($LASTEXITCODE -ne 0) { throw "Rank training stage failed: $LASTEXITCODE" }

    Write-Host "[V5.2 overnight 3/3] Evaluating ranking (skip binary/web)..." -ForegroundColor Cyan
    & $Python $Evaluate `
        --manifest data\ranking\wangxing_v5_2\manifest.json `
        --rank-policy outputs\forensics\wangxing_v5_2_rank_policy_overnight.json `
        --calibrator outputs\forensics\wangxing_v5_realness_calibrator.json `
        --v3-model outputs\vedio_pred\models\wangxing_v3_res1k.pt `
        --drive-model outputs\vedio_pred\models\wangxing_v5_drive.json `
        --forensics-profile outputs\forensics\forensics_profiles_web_v3_test_excluded.json `
        --source-profile outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json `
        --cache-dir outputs\forensics\cache_wangxing_v5_2 `
        --au-output-root outputs\forensics\cache_wangxing_v5_2\au `
        --output-root outputs\vedio_pred\wangxing_v5_2_results_overnight `
        --min-pairwise 0.8333333333 `
        --skip-binary-tests `
        --prior-binary-brief outputs\vedio_pred\wangxing_v5_2_results\leadership_brief.json
    if ($LASTEXITCODE -ne 0) { throw "Overnight evaluate failed: $LASTEXITCODE" }

    Write-Host "[V5.2 overnight] Done." -ForegroundColor Green
    Write-Host "Morning brief: outputs\vedio_pred\wangxing_v5_2_results_overnight\leadership_brief.json" -ForegroundColor Yellow
    Write-Host "Check holdout.display_order_satisfied and class_mean_score_display." -ForegroundColor Yellow
}
finally {
    Pop-Location
}

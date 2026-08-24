$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RankScript = Join-Path $ProjectRoot "scripts\web_forensics\train_wangxing_v5_rank.py"
$WebScript = Join-Path $ProjectRoot "scripts\web_forensics\run_wangxing_web_v5.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

Push-Location $ProjectRoot
try {
    Write-Host "[Web v5] Building guarded rank policy..." -ForegroundColor Cyan
    & $Python $RankScript `
        --ranking-root "C:\Users\zhanghaotian\Desktop\ppt_video" `
        --output outputs\forensics\wangxing_authenticity_policy_v5.json `
        --min-queries 30 `
        --forensics-profile outputs\forensics\forensics_profiles_web_v3_test_excluded.json `
        --source-profile outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json `
        --expression-only
    if ($LASTEXITCODE -ne 0) {
        throw "[Web v5] rank policy stage failed with exit code $LASTEXITCODE"
    }

    Write-Host "[Web v5] Evaluating 25+25 and 32+32..." -ForegroundColor Cyan
    & $Python $WebScript `
        --v3-model outputs\vedio_pred\models\wangxing_v3_res1k.pt `
        --drive-model outputs\vedio_pred\models\wangxing_v5_drive.json `
        --drive-cache outputs\vedio_pred\cache_wangxing_v5_drive_web `
        --forensics-profile outputs\forensics\forensics_profiles_web_v3_test_excluded.json `
        --source-profile outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json `
        --rank-policy outputs\forensics\wangxing_authenticity_policy_v5.json `
        --output-root outputs\forensics\wangxing_v5_web_results `
        --device cuda `
        --wangxing-device cuda `
        --test-set "25+25" data\test\single_video\manifest.json `
        --test-set "32+32" data\test\wangxing_32x32\single_video\manifest.json
    if ($LASTEXITCODE -ne 0) {
        throw "[Web v5] evaluation failed with exit code $LASTEXITCODE"
    }
    Write-Host "[Web v5] completed." -ForegroundColor Green
}
finally {
    Pop-Location
}

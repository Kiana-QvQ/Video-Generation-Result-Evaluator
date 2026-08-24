$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pipeline = Join-Path $ProjectRoot "scripts\pt_training\run_wangxing_v5_pipeline.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

Push-Location $ProjectRoot
try {
    Write-Host "[PT v5] Frozen V3 + DriveHead; final tests are read-only." -ForegroundColor Cyan
    & $Python $Pipeline `
        --train-manifest outputs\vedio_pred\wangxing_v3_generalization_manifest_res1k.json `
        --drive-cache outputs\vedio_pred\cache_wangxing_v5_drive_res1k `
        --drive-model outputs\vedio_pred\models\wangxing_v5_drive.json `
        --drive-metrics outputs\vedio_pred\wangxing_v5_drive_metrics_res1k.json `
        --source-profile outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json `
        --forensics-profile outputs\forensics\forensics_profiles_web_v3_test_excluded.json `
        --output-root outputs\vedio_pred\wangxing_v5_cascade_results `
        --test-set "25+25" data\test\single_video\manifest.json `
        --test-set "32+32" data\test\wangxing_32x32\single_video\manifest.json `
        --seed 42
    if ($LASTEXITCODE -ne 0) {
        throw "[PT v5] pipeline failed with exit code $LASTEXITCODE"
    }
    Write-Host "[PT v5] completed." -ForegroundColor Green
}
finally {
    Pop-Location
}

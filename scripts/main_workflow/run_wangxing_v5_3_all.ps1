param(
    [string]$PptRoot = "C:\Users\zhanghaotian\Desktop\ppt_video",
    [string]$RankingManifestV52 = "data/ranking/wangxing_v5_2/manifest.json",
    [string]$RuntimeManifest = "data/ranking/wangxing_v5_3/manifest_v5_3_runtime.json",
    [string]$SamePromptManifest = "data/ranking/wangxing_v5_3/manifest_v5_3_same_prompt.json",
    [string]$RankPolicyValidated = "outputs/vedio_pred/wangxing_v5_2_results/rank_policy_validated.json",
    [string]$RankPolicyFallback = "outputs/forensics/wangxing_v5_2_rank_policy.json",
    [string]$GatePolicy = "outputs/forensics/wangxing_v5_3_display_gate.json",
    [string]$OutputRoot = "outputs/forensics/wangxing_v5_3_runtime_results",
    [string]$Device = "cuda",
    [string]$WangxingDevice = "cuda",
    [switch]$SkipUnitTests,
    [switch]$RebuildV52Ranking
)

$ErrorActionPreference = "Stop"

# V5.3 one-click (does NOT retrain V3 / route A):
#   0) check V5.2 assets
#   1) optional rebuild V5.2 ranking (if -RebuildV52Ranking)
#   2) build V5.3 runtime + same-prompt manifests
#   3) calibrate content gate from ranking train only
#   4) validate manifests
#   5) evaluate internal manifest D (leadership)
#   6) unit tests
#
# Public web still needs V5_DISPLAY_CASCADE / V5_3_CONTENT_GATE flags.

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

function Assert-ExitCode([string]$Stage) {
    if ($LASTEXITCODE -ne 0) {
        throw "[V5.3] $Stage failed with exit code $LASTEXITCODE"
    }
}

function Resolve-RankPolicy {
    param([string]$Validated, [string]$Fallback)
    if (Test-Path -LiteralPath (Join-Path $ProjectRoot $Validated)) {
        return $Validated
    }
    if (Test-Path -LiteralPath (Join-Path $ProjectRoot $Fallback)) {
        Write-Host "[V5.3] Using fallback rank policy: $Fallback" -ForegroundColor Yellow
        return $Fallback
    }
    throw "Missing rank policy. Run V5.2 first or pass a valid policy path."
}

Push-Location $ProjectRoot
try {
    Write-Host "[V5.3 stage 0/6] Checking V5.2 prerequisites..." -ForegroundColor Cyan
    $required = @(
        "outputs\vedio_pred\models\wangxing_v3_res1k.pt",
        "outputs\vedio_pred\models\wangxing_v5_drive.json",
        "outputs\forensics\wangxing_v5_realness_calibrator.json",
        "outputs\forensics\forensics_profiles_web_v3_test_excluded.json",
        "outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json"
    )
    foreach ($relative in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $relative))) {
            throw "Missing prerequisite: $relative (run V5.1/V5.2 first)"
        }
    }
    if (-not (Test-Path -LiteralPath $PptRoot)) {
        throw "ppt_video root not found: $PptRoot"
    }

    if ($RebuildV52Ranking -or -not (Test-Path -LiteralPath (Join-Path $ProjectRoot $RankingManifestV52))) {
        Write-Host "[V5.3 stage 1a/6] Rebuilding V5.2 ranking manifest + RankHead..." -ForegroundColor Cyan
        & (Join-Path $ProjectRoot "scripts\main_workflow\run_wangxing_v5_2_all.ps1")
        Assert-ExitCode "V5.2 rebuild"
    }
    else {
        Write-Host "[V5.3 stage 1a/6] Reusing existing V5.2 ranking assets." -ForegroundColor Cyan
    }

    $RankPolicy = Resolve-RankPolicy -Validated $RankPolicyValidated -Fallback $RankPolicyFallback

    Write-Host "[V5.3 stage 1b/6] Building V5.3 runtime manifests..." -ForegroundColor Cyan
    & $Python "scripts\web_forensics\build_wangxing_v5_3_runtime_manifest.py" `
        --input $RankingManifestV52 `
        --output $RuntimeManifest `
        --runtime-mode web_regression `
        --full-only
    Assert-ExitCode "build full runtime manifest"

    & $Python "scripts\web_forensics\build_wangxing_v5_3_runtime_manifest.py" `
        --input $RankingManifestV52 `
        --output $SamePromptManifest `
        --runtime-mode web_regression `
        --full-only `
        --same-prompt-only
    Assert-ExitCode "build same-prompt manifest"

    Write-Host "[V5.3 stage 2/6] Calibrating content gate from ranking train..." -ForegroundColor Cyan
    & $Python "scripts\web_forensics\calibrate_wangxing_v5_3_gate.py" `
        --ranking-manifest $RankingManifestV52 `
        --output $GatePolicy `
        --rows-output "outputs\forensics\wangxing_v5_3_gate_train_rows.json" `
        --margin 0.03 `
        --device $Device `
        --wangxing-device $WangxingDevice
    Assert-ExitCode "gate calibration"

    Write-Host "[V5.3 stage 3/6] Validating manifests..." -ForegroundColor Cyan
    & $Python "scripts\web_forensics\validate_wangxing_v5_3_manifest.py" `
        $RuntimeManifest `
        --output "outputs\forensics\wangxing_v5_3_runtime_results\manifest_validation.json"
    Assert-ExitCode "validate full manifest"

    & $Python "scripts\web_forensics\validate_wangxing_v5_3_manifest.py" `
        $SamePromptManifest `
        --output "outputs\forensics\wangxing_v5_3_runtime_results\same_prompt_manifest_validation.json"
    Assert-ExitCode "validate same-prompt manifest"

    Write-Host "[V5.3 stage 4/6] Evaluating internal manifest D (all full groups)..." -ForegroundColor Cyan
    & $Python "scripts\pt_training\evaluate_wangxing_v5_3_runtime.py" `
        --manifest $RuntimeManifest `
        --rank-policy $RankPolicy `
        --calibrator "outputs\forensics\wangxing_v5_realness_calibrator.json" `
        --forensics-profile "outputs\forensics\forensics_profiles_web_v3_test_excluded.json" `
        --source-profile "outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json" `
        --output-root $OutputRoot `
        --device $Device `
        --wangxing-device $WangxingDevice
    Assert-ExitCode "runtime evaluate (full)"

    Write-Host "[V5.3 stage 5/6] Evaluating same-prompt leadership subset..." -ForegroundColor Cyan
    & $Python "scripts\pt_training\evaluate_wangxing_v5_3_runtime.py" `
        --manifest $SamePromptManifest `
        --rank-policy $RankPolicy `
        --calibrator "outputs\forensics\wangxing_v5_realness_calibrator.json" `
        --forensics-profile "outputs\forensics\forensics_profiles_web_v3_test_excluded.json" `
        --source-profile "outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json" `
        --output-root "outputs\forensics\wangxing_v5_3_same_prompt_results" `
        --device $Device `
        --wangxing-device $WangxingDevice
    Assert-ExitCode "runtime evaluate (same-prompt)"

    if (-not $SkipUnitTests) {
        Write-Host "[V5.3 stage 6/6] Unit tests..." -ForegroundColor Cyan
        & $Python -m pytest `
            tests\test_wangxing_v5_3_runtime.py `
            tests\test_web_forensics_display.py `
            tests\test_wangxing_v5_cascade.py `
            -q
        Assert-ExitCode "unit tests"
    }
    else {
        Write-Host "[V5.3 stage 6/6] Skipped unit tests." -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "[V5.3] Completed (route A V3 retrain NOT included)." -ForegroundColor Green
    Write-Host "Runtime manifest : $RuntimeManifest" -ForegroundColor Yellow
    Write-Host "Same-prompt man. : $SamePromptManifest" -ForegroundColor Yellow
    Write-Host "Gate policy      : $GatePolicy" -ForegroundColor Yellow
    Write-Host "Full results     : $OutputRoot\leadership_brief.json" -ForegroundColor Yellow
    Write-Host "Same-prompt brief: outputs\forensics\wangxing_v5_3_same_prompt_results\leadership_brief.json" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Public web (optional):" -ForegroundColor Cyan
    Write-Host "  set V5_DISPLAY_CASCADE=1" -ForegroundColor Cyan
    Write-Host "  set V5_3_CONTENT_GATE=1   # after reviewing gate FP on holdout" -ForegroundColor Cyan
    Write-Host "  python start.py" -ForegroundColor Cyan
}
finally {
    Pop-Location
}

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
    [switch]$RebuildV52Ranking,
    [switch]$FailOnOrdering,
    [double]$MinPairwise = 0.8333333333
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_invoke_native.ps1")

# V5.3 one-click (does NOT retrain V3 / route A):
#   0) check V5.2 assets
#   1) optional rebuild V5.2 ranking (if -RebuildV52Ranking)
#   2) build V5.3 runtime + same-prompt manifests
#   3) calibrate content gate from ranking train only
#   4) validate manifests
#   5) evaluate internal manifest D (leadership + group/pairwise gates)
#   6) unit tests
#
# Overnight rank + V5.3: run_wangxing_v5_3_overnight.ps1

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

function Invoke-V53Python([string]$Stage, [string[]]$ArgumentList) {
    Invoke-PythonChecked -Python $Python -Stage "[V5.3] $Stage" -ArgumentList $ArgumentList
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
        Invoke-ScriptChecked `
            -ScriptPath (Join-Path $ProjectRoot "scripts\main_workflow\run_wangxing_v5_2_all.ps1") `
            -Stage "[V5.3] V5.2 rebuild"
    }
    else {
        Write-Host "[V5.3 stage 1a/6] Reusing existing V5.2 ranking assets." -ForegroundColor Cyan
    }

    $RankPolicy = Resolve-RankPolicy -Validated $RankPolicyValidated -Fallback $RankPolicyFallback

    Write-Host "[V5.3 stage 1b/6] Building V5.3 runtime manifests..." -ForegroundColor Cyan
    Invoke-V53Python "build full runtime manifest" @(
        "scripts\web_forensics\build_wangxing_v5_3_runtime_manifest.py",
        "--input", $RankingManifestV52,
        "--output", $RuntimeManifest,
        "--runtime-mode", "web_regression",
        "--full-only"
    )

    Invoke-V53Python "build same-prompt manifest" @(
        "scripts\web_forensics\build_wangxing_v5_3_runtime_manifest.py",
        "--input", $RankingManifestV52,
        "--output", $SamePromptManifest,
        "--runtime-mode", "web_regression",
        "--full-only",
        "--same-prompt-only"
    )

    Write-Host "[V5.3 stage 2/6] Calibrating content gate from ranking train..." -ForegroundColor Cyan
    Invoke-V53Python "gate calibration" @(
        "scripts\web_forensics\calibrate_wangxing_v5_3_gate.py",
        "--ranking-manifest", $RankingManifestV52,
        "--output", $GatePolicy,
        "--rows-output", "outputs\forensics\wangxing_v5_3_gate_train_rows.json",
        "--margin", "0.03",
        "--device", $Device,
        "--wangxing-device", $WangxingDevice
    )

    Write-Host "[V5.3 stage 3/6] Validating manifests..." -ForegroundColor Cyan
    Invoke-V53Python "validate full manifest" @(
        "scripts\web_forensics\validate_wangxing_v5_3_manifest.py",
        $RuntimeManifest,
        "--output", "outputs\forensics\wangxing_v5_3_runtime_results\manifest_validation.json"
    )

    Invoke-V53Python "validate same-prompt manifest" @(
        "scripts\web_forensics\validate_wangxing_v5_3_manifest.py",
        $SamePromptManifest,
        "--strict-unique-sha256",
        "--output", "outputs\forensics\wangxing_v5_3_runtime_results\same_prompt_manifest_validation.json"
    )

    Write-Host "[V5.3 stage 4/6] Evaluating internal manifest D (all full groups)..." -ForegroundColor Cyan
    $evaluateArgs = @(
        "--manifest", $RuntimeManifest,
        "--rank-policy", $RankPolicy,
        "--calibrator", "outputs\forensics\wangxing_v5_realness_calibrator.json",
        "--forensics-profile", "outputs\forensics\forensics_profiles_web_v3_test_excluded.json",
        "--source-profile", "outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json",
        "--output-root", $OutputRoot,
        "--device", $Device,
        "--wangxing-device", $WangxingDevice,
        "--min-pairwise", "$MinPairwise"
    )
    if ($FailOnOrdering) {
        $evaluateArgs += "--fail-on-ordering"
    }
    $fullEvaluateArgs = @(
        "scripts\pt_training\evaluate_wangxing_v5_3_runtime.py"
    ) + $evaluateArgs
    Invoke-V53Python "runtime evaluate (full)" $fullEvaluateArgs

    Write-Host "[V5.3 stage 5/6] Evaluating same-prompt leadership subset..." -ForegroundColor Cyan
    $samePromptOutput = if ($OutputRoot -like "*_overnight") {
        "outputs\forensics\wangxing_v5_3_same_prompt_results_overnight"
    } else {
        "outputs\forensics\wangxing_v5_3_same_prompt_results"
    }
    $samePromptArgs = @(
        "--manifest", $SamePromptManifest,
        "--rank-policy", $RankPolicy,
        "--calibrator", "outputs\forensics\wangxing_v5_realness_calibrator.json",
        "--forensics-profile", "outputs\forensics\forensics_profiles_web_v3_test_excluded.json",
        "--source-profile", "outputs\forensics\wangxing_source_profile_web_v3_test_excluded.json",
        "--output-root", $samePromptOutput,
        "--device", $Device,
        "--wangxing-device", $WangxingDevice,
        "--min-pairwise", "$MinPairwise"
    )
    if ($FailOnOrdering) {
        $samePromptArgs += "--fail-on-ordering"
    }
    $fullSamePromptArgs = @(
        "scripts\pt_training\evaluate_wangxing_v5_3_runtime.py"
    ) + $samePromptArgs
    Invoke-V53Python "runtime evaluate (same-prompt)" $fullSamePromptArgs

    if (-not $SkipUnitTests) {
        Write-Host "[V5.3 stage 6/6] Unit tests..." -ForegroundColor Cyan
        Invoke-V53Python "unit tests" @(
            "-m", "pytest",
            "tests\test_wangxing_v5_3_runtime.py",
            "tests\test_web_forensics_display.py",
            "tests\test_wangxing_v5_cascade.py",
            "-q"
        )
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
    Write-Host "Same-prompt brief: $samePromptOutput\leadership_brief.json" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Check holdout.group_ordering and holdout.pairwise_ordering in leadership_brief." -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Public web (optional):" -ForegroundColor Cyan
    Write-Host "  python start.py --v5-display" -ForegroundColor Cyan
    Write-Host "  python start.py --v5-display --v5-3-content-gate   # after holdout FP review" -ForegroundColor Cyan
}
finally {
    Pop-Location
}

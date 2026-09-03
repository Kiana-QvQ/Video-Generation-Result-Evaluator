param(
    [switch]$SkipEvaluation
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

function Invoke-PythonStage([string]$Stage, [string[]]$CommandArgs) {
    Write-Host "[XiaoYue manifold v2] $Stage" -ForegroundColor Cyan
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "[XiaoYue manifold v2] Stage failed: $Stage (exit $LASTEXITCODE)"
    }
}

Push-Location $ProjectRoot
try {
    $Manifest = "data\xiaoyue\experiment_7x7\manifests\face_manifold_manifest.json"
    $OutputRoot = "outputs\xiaoyue\experiment_7x7_face_v2"
    $FeatureCache = "outputs\xiaoyue\experiment_7x7_face\all_real_face_cache.npz"
    Invoke-PythonStage "1/4 build real-manifold manifest" @(
        "scripts\data_build\build_xiaoyue_face_manifold_manifest.py",
        "--output", $Manifest
    )

    Invoke-PythonStage "2/4 fit face-only web manifold profile" @(
        "scripts\web_forensics\evaluate_xiaoyue_face_manifold_v2.py",
        "fit",
        "--manifest", $Manifest,
        "--cache", $FeatureCache,
        "--profile", "$OutputRoot\xiaoyue_face_manifold_profile.json"
    )

    Invoke-PythonStage "3/4 fit face-only PT checkpoint and test" @(
        "scripts\pt_training\train_xiaoyue_face_manifold_v2.py",
        "--manifest", $Manifest,
        "--train-cache", $FeatureCache,
        "--test-cache", "$OutputRoot\test_features.npz",
        "--profile", "$OutputRoot\xiaoyue_face_manifold_profile.json",
        "--model", "$OutputRoot\models\xiaoyue_face_manifold_v2.pt",
        "--metrics", "$OutputRoot\xiaoyue_face_manifold_v2_metrics.json"
    )

    if (-not $SkipEvaluation) {
        Invoke-PythonStage "4/4 run face-only web test" @(
            "scripts\web_forensics\evaluate_xiaoyue_face_manifold_v2.py",
            "evaluate",
            "--manifest", $Manifest,
            "--profile", "$OutputRoot\xiaoyue_face_manifold_profile.json",
            "--cache", "$OutputRoot\test_features.npz",
            "--output-root", "$OutputRoot\web_test"
        )
    }
    else {
        Write-Host "[XiaoYue manifold v2] 4/4 web evaluation skipped." -ForegroundColor Yellow
    }

    $summary = [ordered]@{
        schema_version = "xiaoyue_face_manifold_v2_pipeline_run_v1"
        subject = "xiaoyue"
        real_manifold_bank = 104
        ai_train = 6
        test_real = 1
        test_ai = 1
        manifest = $Manifest
        profile = "$OutputRoot\xiaoyue_face_manifold_profile.json"
        pt_model = "$OutputRoot\models\xiaoyue_face_manifold_v2.pt"
        pt_metrics = "$OutputRoot\xiaoyue_face_manifold_v2_metrics.json"
        web_results = "$OutputRoot\web_test"
        full_frame_features_used = $false
        background_used = $false
        mouth_priority = $true
        test_training_allowed = $false
        note = "Real-manifold calibration uses accepted real videos only; the 1+1 test pair remains evaluation-only."
    }
    $summaryPath = Join-Path $ProjectRoot "$OutputRoot\xiaoyue_face_manifold_v2_pipeline_run.json"
    New-Item -ItemType Directory -Force -Path (Split-Path $summaryPath) | Out-Null
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
    Write-Host "[XiaoYue manifold v2] Completed. Summary: $summaryPath" -ForegroundColor Green
}
finally {
    Pop-Location
}

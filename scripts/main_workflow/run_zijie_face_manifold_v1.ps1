param(
    [string]$Device = "cuda",
    [switch]$SkipAuExtraction,
    [switch]$SkipOfflineEvaluation
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

function Invoke-PythonStage([string]$Stage, [string[]]$CommandArgs) {
    Write-Host "[ZiJie manifold v1] $Stage" -ForegroundColor Cyan
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "[ZiJie manifold v1] Stage failed: $Stage (exit $LASTEXITCODE)"
    }
}

Push-Location $ProjectRoot
try {
    $SourceRoot = "D:\Pycharm2023.21\project\MiniMax_H3\data\zijie_hk_neutralgray_matte_1k_uncropped_v1"
    $AuRoot = "data\au\zijie\source"
    $Manifest = "data\zijie\manifests\zijie_face_manifold_v1.json"
    $OutputRoot = "outputs\zijie\face_manifold_v1"

    Invoke-PythonStage "1/6 build group-isolated manifest" @(
        "scripts\data_build\build_zijie_face_manifold_manifest.py",
        "--source-root", $SourceRoot,
        "--au-root", $AuRoot,
        "--output", $Manifest
    )

    if (-not $SkipAuExtraction) {
        Invoke-PythonStage "2/6 extract AU for neutral-gray real videos" @(
            "scripts\au\extract_libreface_au.py",
            "--input-root", $SourceRoot,
            "--exclude-dir", "AI_Make",
            "--output-root", $AuRoot,
            "--device", $Device,
            "--batch-size", "32",
            "--num-workers", "0",
            "--face-fallback", "insightface",
            "--normalize-input-first",
            "--continue-on-error",
            "--failure-log", "$OutputRoot\real_au_failures.json"
        )
    Invoke-PythonStage "3/6 extract AU for H3 generated videos" @(
            "scripts\au\extract_libreface_au.py",
            "--input-root", "$SourceRoot\AI_Make",
            "--output-root", "$AuRoot\AI_Make",
            "--device", $Device,
            "--batch-size", "32",
            "--num-workers", "0",
            "--face-fallback", "insightface",
            "--continue-on-error",
            "--failure-log", "$OutputRoot\ai_au_failures.json"
        )
    }
    else {
        Write-Host "[ZiJie manifold v1] 2/6 and 3/6 AU extraction skipped." -ForegroundColor Yellow
    }

    Invoke-PythonStage "4/6 verify split and AU coverage" @(
        "scripts\data_build\build_zijie_face_manifold_manifest.py",
        "--source-root", $SourceRoot,
        "--au-root", $AuRoot,
        "--output", $Manifest,
        "--require-au"
    )

    Invoke-PythonStage "5/6 fit PT checkpoint and evaluate holdout" @(
        "scripts\pt_training\train_zijie_face_manifold_v1.py",
        "--manifest", $Manifest,
        "--train-cache", "$OutputRoot\cache\train_features.npz",
        "--test-cache", "$OutputRoot\cache\pt_holdout_features.npz",
        "--profile", "$OutputRoot\zijie_face_manifold_profile.json",
        "--model", "$OutputRoot\models\zijie_face_manifold_v1.pt",
        "--metrics", "$OutputRoot\zijie_face_manifold_v1_metrics.json"
    )

    if (-not $SkipOfflineEvaluation) {
        Invoke-PythonStage "6/6 produce offline profile holdout report" @(
            "scripts\pt_training\evaluate_zijie_face_manifold_v1.py",
            "--manifest", $Manifest,
            "--profile", "$OutputRoot\zijie_face_manifold_profile.json",
            "--cache", "$OutputRoot\cache\offline_holdout_features.npz",
            "--output-root", "$OutputRoot\offline_profile_test"
        )
    }
    else {
        Write-Host "[ZiJie manifold v1] 6/6 offline profile test skipped." -ForegroundColor Yellow
    }

    $summary = [ordered]@{
        schema_version = "zijie_face_manifold_v1_pipeline_run"
        subject = "zijie"
        source_root = $SourceRoot
        manifest = $Manifest
        pt_model = "$OutputRoot\models\zijie_face_manifold_v1.pt"
        pt_metrics = "$OutputRoot\zijie_face_manifold_v1_metrics.json"
        offline_profile_results = "$OutputRoot\offline_profile_test"
        production_web_changed = $false
        full_frame_features_used = $false
        background_used = $false
        absolute_brightness_used = $false
        mouth_priority = $true
        test_training_allowed = $false
        note = "Capture-ID and content-deduplicated H3 holdout is offline only. No webpage route or UI setting was changed."
    }
    $summaryPath = Join-Path $ProjectRoot "$OutputRoot\zijie_face_manifold_v1_pipeline_run.json"
    New-Item -ItemType Directory -Force -Path (Split-Path $summaryPath) | Out-Null
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
    Write-Host "[ZiJie manifold v1] Completed. Summary: $summaryPath" -ForegroundColor Green
}
finally {
    Pop-Location
}

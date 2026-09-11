param(
    [switch]$SkipManifestBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SourceRoot = "D:\Pycharm2023.21\project\MiniMax_H3\data\zijie_hk_neutralgray_matte_1k_uncropped_v1"
$Manifest = "data\zijie\manifests\zijie_face_manifold_v1.json"
$OutputRoot = "outputs\zijie\face_temporal_v4"

function Invoke-PythonStage([string]$Name, [string[]]$Arguments) {
    Write-Host "[ZiJie V4] $Name" -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "[ZiJie V4] $Name failed with exit code $LASTEXITCODE"
    }
}

Push-Location $ProjectRoot
try {
    if (-not $SkipManifestBuild) {
        Invoke-PythonStage "1/4 verify manifest and AU coverage" @(
            "scripts\data_build\build_zijie_face_manifold_manifest.py",
            "--source-root", $SourceRoot,
            "--au-root", "data\au\zijie\source",
            "--output", $Manifest,
            "--require-au"
        )
    }
    Invoke-PythonStage "2/4 train PT and evaluate fixed holdout" @(
        "scripts\pt_training\train_zijie_face_temporal_v4.py",
        "--manifest", $Manifest,
        "--train-cache", "$OutputRoot\cache\train_features.npz",
        "--test-cache", "$OutputRoot\cache\holdout_window_features.npz",
        "--profile", "$OutputRoot\zijie_face_temporal_v4_profile.json",
        "--model", "$OutputRoot\models\zijie_face_temporal_v4.pt",
        "--metrics", "$OutputRoot\zijie_face_temporal_v4_metrics.json"
    )
    Invoke-PythonStage "3/4 evaluate offline webpage-equivalent branch" @(
        "scripts\pt_training\evaluate_zijie_face_temporal_v4.py",
        "--manifest", $Manifest,
        "--profile", "$OutputRoot\zijie_face_temporal_v4_profile.json",
        "--cache", "$OutputRoot\cache\holdout_window_features.npz",
        "--output-root", "$OutputRoot\offline_profile_test"
    )
    Invoke-PythonStage "4/4 validate accuracy, recall and score gap" @(
        "scripts\pt_training\validate_zijie_face_temporal_v4.py",
        "--pt-metrics", "$OutputRoot\zijie_face_temporal_v4_metrics.json",
        "--offline-results", "$OutputRoot\offline_profile_test\all_results.json",
        "--output", "$OutputRoot\acceptance_gate.json",
        "--minimum-gap", "0.03"
    )
    Write-Host "[ZiJie V4] Completed. Existing V3 and production web are unchanged." -ForegroundColor Green
}
finally {
    Pop-Location
}

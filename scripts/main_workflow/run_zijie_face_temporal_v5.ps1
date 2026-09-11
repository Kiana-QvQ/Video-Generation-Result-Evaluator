param(
    [switch]$SkipAuExtraction
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LtxRoot = "D:\Pycharm2023.21\project\MiniMax_H3\artifacts\zijie_ltx25_v4_1500_compare\ai_generated_videos_for_training_20260910"
$Metrics = "D:\Pycharm2023.21\project\MiniMax_H3\artifacts\zijie_ltx25_v4_1500_compare\final_checkpoint_review\gt_similarity_metrics.tsv"
$AuRoot = "data\au\zijie\ltx25_v4_1500_20260910"
$GtVideo = "D:\Pycharm2023.21\project\MiniMax_H3\work\zijie_hk_gate_v3_review\gt_real.mp4"
$GtAu = "data\au\zijie\ltx25_v4_1500_gt\gt_real.csv"
$Manifest = "data\zijie\manifests\zijie_face_temporal_v5.json"

function Invoke-PythonStage([string]$Name, [string[]]$Arguments) {
    Write-Host "[ZiJie V5] $Name" -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "[ZiJie V5] $Name failed with exit code $LASTEXITCODE"
    }
}

Push-Location $ProjectRoot
try {
    if (-not $SkipAuExtraction) {
        Invoke-PythonStage "1/3 extract/verify AU for 16 LTX videos" @(
            "scripts\au\extract_libreface_au.py",
            "--input-root", $LtxRoot,
            "--output-root", $AuRoot,
            "--device", "cuda",
            "--batch-size", "64",
            "--num-workers", "2",
            "--face-fallback", "insightface",
            "--continue-on-error",
            "--failure-log", "outputs\zijie\face_temporal_v5\au_failures.json"
        )
    }
    Invoke-PythonStage "2/3 build isolated binary and ranking manifests" @(
        "scripts\data_build\build_zijie_face_temporal_v5_manifest.py",
        "--ltx-root", $LtxRoot,
        "--ltx-au-root", $AuRoot,
        "--quality-metrics", $Metrics,
        "--gt-video", $GtVideo,
        "--gt-au", $GtAu,
        "--output", $Manifest,
        "--require-au"
    )
    Invoke-PythonStage "3/3 train and validate PT/web-equivalent V5" @(
        "scripts\pt_training\run_zijie_face_temporal_v5.py",
        "--manifest", $Manifest,
        "--output-root", "outputs\zijie\face_temporal_v5",
        "--base-profile", "outputs\zijie\face_temporal_v4\zijie_face_temporal_v4_profile.json"
    )
    Write-Host "[ZiJie V5] Done. Production web remains unchanged." -ForegroundColor Green
}
finally {
    Pop-Location
}

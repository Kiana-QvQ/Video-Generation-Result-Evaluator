param(
    [string]$Device = "cuda",
    [int]$Epochs = 80,
    [int]$BatchSize = 4,
    [double]$LearningRate = 0.0003,
    [int]$Seed = 42,
    [switch]$SkipTraining,
    [switch]$SkipWebEvaluation
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

function Invoke-PythonStage([string]$Stage, [string[]]$CommandArgs) {
    Write-Host "[XiaoYue face 7x7] $Stage" -ForegroundColor Cyan
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "[XiaoYue face 7x7] Stage failed: $Stage (exit $LASTEXITCODE)"
    }
}

Push-Location $ProjectRoot
try {
    $Root = "data\xiaoyue\experiment_7x7"
    $ManifestRoot = "$Root\manifests"
    $ProfileRoot = "$Root\face_profiles"
    $OutputRoot = "outputs\xiaoyue\experiment_7x7_face"
    $CacheRoot = "$OutputRoot\cache"

    Invoke-PythonStage "1/6 build isolated 6+1 AI / 7 real dataset" @(
        "scripts\data_build\build_xiaoyue_7x7_dataset.py",
        "--output-root", $Root
    )

    Invoke-PythonStage "2/6 extract missing AI AU CSVs" @(
        "scripts\au\extract_libreface_au.py",
        "--input-root", "$Root\train\ai",
        "--output-root", "$Root\au\train\ai",
        "--device", $Device,
        "--batch-size", "32",
        "--num-workers", "0",
        "--face-fallback", "insightface",
        "--normalize-input-first",
        "--continue-on-error",
        "--failure-log", "$OutputRoot\ai_train_au_failures.json"
    )

    Invoke-PythonStage "3/6 verify all train/test AU pairs" @(
        "scripts\data_build\build_xiaoyue_7x7_dataset.py",
        "--output-root", $Root,
        "--require-train-au"
    )

    Invoke-PythonStage "4/6 fit face-only web profile" @(
        "scripts\web_forensics\evaluate_xiaoyue_face_v1.py",
        "fit",
        "--manifest", "$ManifestRoot\pt_manifest.json",
        "--cache", "$CacheRoot\web_train_features.npz",
        "--profile", "$ProfileRoot\xiaoyue_face_web_profile.json"
    )

    if (-not $SkipTraining) {
        Invoke-PythonStage "5/6 train face-only PT model" @(
            "scripts\pt_training\train_xiaoyue_face_v1.py",
            "--manifest", "$ManifestRoot\pt_manifest.json",
            "--cache", "$CacheRoot\pt_features.npz",
            "--model", "$OutputRoot\models\xiaoyue_face_mouth_v1.pt",
            "--metrics", "$OutputRoot\xiaoyue_face_mouth_v1_metrics.json",
            "--epochs", "$Epochs",
            "--batch-size", "$BatchSize",
            "--learning-rate", "$LearningRate",
            "--seed", "$Seed",
            "--device", $Device
        )
    }
    else {
        Write-Host "[XiaoYue face 7x7] 5/6 PT training skipped." -ForegroundColor Yellow
    }

    if (-not $SkipWebEvaluation) {
        Invoke-PythonStage "6/6 evaluate face-only web test pair" @(
            "scripts\web_forensics\evaluate_xiaoyue_face_v1.py",
            "evaluate",
            "--manifest", "$ManifestRoot\web_test_manifest.json",
            "--profile", "$ProfileRoot\xiaoyue_face_web_profile.json",
            "--cache", "$CacheRoot\web_test_features.npz",
            "--output-root", "$OutputRoot\web_test"
        )
    }
    else {
        Write-Host "[XiaoYue face 7x7] 6/6 webpage evaluation skipped." -ForegroundColor Yellow
    }

    $summary = [ordered]@{
        schema_version = "xiaoyue_face_7x7_pipeline_run_v1"
        subject = "xiaoyue"
        train_real = 6
        train_ai = 6
        test_real = 1
        test_ai = 1
        dataset_root = $Root
        manifest_root = $ManifestRoot
        face_profile = "$ProfileRoot\xiaoyue_face_web_profile.json"
        pt_model = "$OutputRoot\models\xiaoyue_face_mouth_v1.pt"
        pt_metrics = "$OutputRoot\xiaoyue_face_mouth_v1_metrics.json"
        web_results = "$OutputRoot\web_test"
        production_web_changed = $false
        old_xiaoyue_assets_changed = $false
        wangxing_assets_changed = $false
        full_frame_features_used = $false
        mouth_priority = $true
        test_training_allowed = $false
        note = "Small diagnostic experiment; one-pair test accuracy must not be treated as a generalization benchmark."
    }
    $summaryPath = Join-Path $ProjectRoot "$OutputRoot\xiaoyue_face_7x7_pipeline_run.json"
    New-Item -ItemType Directory -Force -Path (Split-Path $summaryPath) | Out-Null
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
    Write-Host "[XiaoYue face 7x7] Completed. Summary: $summaryPath" -ForegroundColor Green
}
finally {
    Pop-Location
}

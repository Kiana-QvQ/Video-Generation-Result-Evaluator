param(
    [string]$TestRoot = "data/xiaoyue/processed/test_reference",
    [string]$Device = "cuda",
    [string]$WangxingDevice = "cpu",
    [int]$Epochs = 80,
    [int]$BatchSize = 8,
    [double]$LearningRate = 0.0003,
    [int]$Seed = 42,
    [switch]$RebuildProfiles,
    [switch]$SkipWebEvaluation,
    [switch]$SkipPtTraining
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

function Invoke-PythonStage([string]$Stage, [string[]]$Args) {
    Write-Host "[XiaoYue] $Stage" -ForegroundColor Cyan
    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "[XiaoYue] Stage failed: $Stage (exit $LASTEXITCODE)"
    }
}

Push-Location $ProjectRoot
try {
    Write-Host "[XiaoYue 1/7] Extracting AU for isolated test/reference videos..." -ForegroundColor Cyan
    Invoke-PythonStage "test AU extraction" @(
        "scripts\au\extract_libreface_au.py",
        "--input-root", $TestRoot,
        "--output-root", "data\au\xiaoyue\test",
        "--device", $Device,
        "--batch-size", "32",
        "--num-workers", "0",
        "--face-fallback", "insightface",
        "--normalize-input-first",
        "--continue-on-error",
        "--failure-log", "outputs\xiaoyue\test_au_failures.json"
    )

    Write-Host "[XiaoYue 2/7] Building isolated PT train/test manifests..." -ForegroundColor Cyan
    Invoke-PythonStage "PT manifest" @(
        "scripts\data_build\build_xiaoyue_pt_manifest.py",
        "--source-manifest", "data\xiaoyue\processed\specialization_manifest.json",
        "--real-au-root", "data\au\xiaoyue\real",
        "--generated-au-root", "data\au\xiaoyue\generated",
        "--test-root", $TestRoot,
        "--test-au-root", "data\au\xiaoyue\test",
        "--output", "data\xiaoyue\processed\pt_manifest.json",
        "--test-output", "data\xiaoyue\processed\pt_test_manifest.json"
    )

    if ($RebuildProfiles) {
        Write-Host "[XiaoYue 3/7] Rebuilding XiaoYue identity/source/expression profiles..." -ForegroundColor Cyan
        Invoke-PythonStage "specialization profiles" @(
            "scripts\data_build\build_xiaoyue_specialization_profiles.py",
            "--manifest", "data\xiaoyue\processed\specialization_manifest.json",
            "--real-video-root", "data\xiaoyue\processed\real_candidates",
            "--generated-video-root", "data\xiaoyue\processed\ai_candidates",
            "--negative-root", "data\negative\ravdess\videos",
            "--real-au-root", "data\au\xiaoyue\real",
            "--generated-au-root", "data\au\xiaoyue\generated",
            "--profile-root", "data\xiaoyue\profiles",
            "--identity-frames", "8",
            "--device", "cpu"
        )
        Write-Host "[XiaoYue 4/7] Rebuilding sampled web forensics profile..." -ForegroundColor Cyan
        Invoke-PythonStage "web forensics profile" @(
            "scripts\data_build\build_forensics_profiles.py",
            "--real-au-root", "data\au\xiaoyue\real",
            "--seedance-au-root", "data\au\xiaoyue\generated",
            "--real-video-root", "data\xiaoyue\processed\real_candidates",
            "--seedance-video-root", "data\xiaoyue\processed\ai_candidates",
            "--output", "data\xiaoyue\profiles\xiaoyue_forensics_profiles.json",
            "--max-motion-videos", "0",
            "--max-videos", "8",
            "--max-frames", "16",
            "--sample-fps", "4",
            "--min-landmark-ratio", "0.35",
            "--min-pose-ratio", "0.35"
        )
    }
    else {
        Write-Host "[XiaoYue 3/7] Reusing existing XiaoYue profiles..." -ForegroundColor Yellow
    }

    $required = @(
        "data\xiaoyue\profiles\xiaoyue_identity_profile.json",
        "data\xiaoyue\profiles\xiaoyue_expression_profile.json",
        "data\xiaoyue\profiles\xiaoyue_source_profile.json",
        "data\xiaoyue\profiles\xiaoyue_forensics_profiles.json"
    )
    foreach ($file in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $file))) {
            throw "[XiaoYue] Missing profile: $file"
        }
    }

    if (-not $SkipPtTraining) {
        Write-Host "[XiaoYue 5/7] Training and evaluating isolated PT model..." -ForegroundColor Cyan
        Invoke-PythonStage "PT training/evaluation" @(
            "scripts\pt_training\train_xiaoyue_v3.py",
            "--manifest", "data\xiaoyue\processed\pt_manifest.json",
            "--source-profile", "data\xiaoyue\profiles\xiaoyue_source_profile.json",
            "--forensics-profile", "data\xiaoyue\profiles\xiaoyue_forensics_profiles.json",
            "--cache-dir", "outputs\xiaoyue\pt_v3_cache",
            "--model-path", "outputs\xiaoyue\models\xiaoyue_temporal_v3.pt",
            "--metrics-output", "outputs\xiaoyue\xiaoyue_temporal_v3_metrics.json",
            "--epochs", "$Epochs",
            "--batch-size", "$BatchSize",
            "--learning-rate", "$LearningRate",
            "--seed", "$Seed",
            "--device", $Device
        )
    }
    else {
        Write-Host "[XiaoYue 5/7] Skipped PT training by request." -ForegroundColor Yellow
    }

    if (-not $SkipWebEvaluation) {
        Write-Host "[XiaoYue 6/7] Running webpage-equivalent evaluation..." -ForegroundColor Cyan
        Invoke-PythonStage "web evaluation" @(
            "scripts\web_forensics\evaluate_xiaoyue_dataset.py",
            "--manifest", "data\xiaoyue\processed\pt_test_manifest.json",
            "--forensics-profile", "data\xiaoyue\profiles\xiaoyue_forensics_profiles.json",
            "--identity-profile", "data\xiaoyue\profiles\xiaoyue_identity_profile.json",
            "--expression-profile", "data\xiaoyue\profiles\xiaoyue_expression_profile.json",
            "--source-profile", "data\xiaoyue\profiles\xiaoyue_source_profile.json",
            "--output-root", "outputs\xiaoyue\web_test",
            "--device", $Device,
            "--wangxing-device", $WangxingDevice,
            "--max-frames", "32",
            "--sample-fps", "8"
        )
    }
    else {
        Write-Host "[XiaoYue 6/7] Skipped web evaluation by request." -ForegroundColor Yellow
    }

    Write-Host "[XiaoYue 7/7] Writing run summary..." -ForegroundColor Cyan
    $summary = [ordered]@{
        schema_version = "xiaoyue_pipeline_run_v1"
        subject = "xiaoyue"
        training_manifest = "data/xiaoyue/processed/pt_manifest.json"
        test_manifest = "data/xiaoyue/processed/pt_test_manifest.json"
        pt_model = "outputs/xiaoyue/models/xiaoyue_temporal_v3.pt"
        pt_metrics = "outputs/xiaoyue/xiaoyue_temporal_v3_metrics.json"
        web_results = "outputs/xiaoyue/web_test"
        production_web_changed = $false
        wangxing_assets_changed = $false
        test_training_allowed = $false
    }
    $summaryPath = Join-Path $ProjectRoot "outputs\xiaoyue\xiaoyue_pipeline_run.json"
    New-Item -ItemType Directory -Force -Path (Split-Path $summaryPath) | Out-Null
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
    Write-Host "[XiaoYue] Completed. Summary: $summaryPath" -ForegroundColor Green
}
finally {
    Pop-Location
}

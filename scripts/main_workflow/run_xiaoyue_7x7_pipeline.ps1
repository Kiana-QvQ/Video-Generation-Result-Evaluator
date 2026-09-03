param(
    [string]$Device = "cuda",
    [string]$WangxingDevice = "cpu",
    [int]$Epochs = 80,
    [int]$BatchSize = 4,
    [double]$LearningRate = 0.0003,
    [int]$Seed = 42,
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
    Write-Host "[XiaoYue 7x7] $Stage" -ForegroundColor Cyan
    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "[XiaoYue 7x7] Stage failed: $Stage (exit $LASTEXITCODE)"
    }
}

Push-Location $ProjectRoot
try {
    $Root = "data\xiaoyue\experiment_7x7"
    $ManifestRoot = "$Root\manifests"
    $ProfileRoot = "$Root\profiles"
    $OutputRoot = "outputs\xiaoyue\experiment_7x7"

    Invoke-PythonStage "1/7 build isolated 7+7 manifests" @(
        "scripts\data_build\build_xiaoyue_7x7_dataset.py",
        "--output-root", $Root
    )

    Invoke-PythonStage "2/7 extract AU for six AI training clips" @(
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

    Invoke-PythonStage "3/7 verify isolated manifests and AU files" @(
        "scripts\data_build\build_xiaoyue_7x7_dataset.py",
        "--output-root", $Root,
        "--require-train-au"
    )

    Invoke-PythonStage "4/7 rebuild isolated XiaoYue profiles" @(
        "scripts\data_build\build_xiaoyue_specialization_profiles.py",
        "--manifest", "$ManifestRoot\specialization_manifest.json",
        "--real-video-root", "$Root\train\real",
        "--generated-video-root", "$Root\train\ai",
        "--negative-root", "data\negative\ravdess\videos",
        "--real-au-root", "$Root\au\train\real",
        "--generated-au-root", "$Root\au\train\ai",
        "--profile-root", $ProfileRoot,
        "--identity-frames", "8",
        "--device", "cpu"
    )

    Invoke-PythonStage "5/7 rebuild isolated webpage forensics profile" @(
        "scripts\data_build\build_forensics_profiles.py",
        "--real-au-root", "$Root\au\train\real",
        "--seedance-au-root", "$Root\au\train\ai",
        "--real-video-root", "$Root\train\real",
        "--seedance-video-root", "$Root\train\ai",
        "--output", "$ProfileRoot\xiaoyue_forensics_profiles.json",
        "--max-motion-videos", "0",
        "--max-videos", "6",
        "--max-frames", "16",
        "--sample-fps", "4",
        "--min-landmark-ratio", "0.35",
        "--min-pose-ratio", "0.35"
    )

    if (-not $SkipPtTraining) {
        Invoke-PythonStage "6/7 train and evaluate isolated PT model" @(
            "scripts\pt_training\train_xiaoyue_v3.py",
            "--manifest", "$ManifestRoot\pt_manifest.json",
            "--source-profile", "$ProfileRoot\xiaoyue_source_profile.json",
            "--forensics-profile", "$ProfileRoot\xiaoyue_forensics_profiles.json",
            "--cache-dir", "$OutputRoot\pt_v3_cache",
            "--model-path", "$OutputRoot\models\xiaoyue_temporal_v3_7x7.pt",
            "--metrics-output", "$OutputRoot\xiaoyue_temporal_v3_7x7_metrics.json",
            "--epochs", "$Epochs",
            "--batch-size", "$BatchSize",
            "--learning-rate", "$LearningRate",
            "--seed", "$Seed",
            "--device", $Device
        )
    }
    else {
        Write-Host "[XiaoYue 7x7] 6/7 PT training skipped." -ForegroundColor Yellow
    }

    if (-not $SkipWebEvaluation) {
        Invoke-PythonStage "7/7 webpage-equivalent test" @(
            "scripts\web_forensics\evaluate_xiaoyue_dataset.py",
            "--manifest", "$ManifestRoot\web_test_manifest.json",
            "--forensics-profile", "$ProfileRoot\xiaoyue_forensics_profiles.json",
            "--identity-profile", "$ProfileRoot\xiaoyue_identity_profile.json",
            "--expression-profile", "$ProfileRoot\xiaoyue_expression_profile.json",
            "--source-profile", "$ProfileRoot\xiaoyue_source_profile.json",
            "--output-root", "$OutputRoot\web_test",
            "--device", $Device,
            "--wangxing-device", $WangxingDevice,
            "--max-frames", "32",
            "--sample-fps", "8"
        )
    }
    else {
        Write-Host "[XiaoYue 7x7] 7/7 webpage test skipped." -ForegroundColor Yellow
    }

    $summary = [ordered]@{
        schema_version = "xiaoyue_7x7_pipeline_run_v1"
        subject = "xiaoyue"
        train_real = 6
        train_ai = 6
        test_real = 1
        test_ai = 1
        dataset_root = "$Root"
        manifest_root = "$ManifestRoot"
        profile_root = "$ProfileRoot"
        output_root = "$OutputRoot"
        pt_model = "$OutputRoot/models/xiaoyue_temporal_v3_7x7.pt"
        pt_metrics = "$OutputRoot/xiaoyue_temporal_v3_7x7_metrics.json"
        web_results = "$OutputRoot/web_test"
        production_web_changed = $false
        old_xiaoyue_assets_changed = $false
        wangxing_assets_changed = $false
        test_training_allowed = $false
        note = "This is a small paired experiment; its 1+1 test result is diagnostic, not a generalization benchmark."
    }
    $summaryPath = Join-Path $ProjectRoot "$OutputRoot\xiaoyue_7x7_pipeline_run.json"
    New-Item -ItemType Directory -Force -Path (Split-Path $summaryPath) | Out-Null
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
    Write-Host "[XiaoYue 7x7] Completed. Summary: $summaryPath" -ForegroundColor Green
}
finally {
    Pop-Location
}

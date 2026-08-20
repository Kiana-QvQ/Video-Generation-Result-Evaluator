param(
    [ValidateSet("RAVDESS", "MetaHuman")]
    [string]$NegativeDataset = "RAVDESS",

    [string]$MetaHumanArchive = "",
    [string]$MetaHumanUrl = "",
    [string]$RavdessActors = "1,2",
    [ValidateSet("ZENODO", "HUGGINGFACE")]
    [string]$RavdessSource = "ZENODO",
    [string]$RavdessEmotions = "1,2,3,4,5,6,7,8",
    [string]$RavdessCacheRoot = "data\cache\ravdess",
    [int]$MaxNegativeVideos = 48,
    [ValidateSet("cpu", "cuda", "auto")]
    [string]$Device = "cuda",
    [int]$BatchSize = 64,
    [int]$NumWorkers = 2,
    [string]$OriginalAuRoot = "data\au\MD_CL",
    [string]$EmotionProfileOutput = "data\au\original_emotion_au_profile.json",
    [int]$EmotionMinSamplesPerClass = 3,
    [switch]$SkipNegativePreparation,
    [switch]$ForceAuExtraction
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python was not found: $python"
}

Set-Location -LiteralPath $projectRoot
$runnerArgs = @(
    "scripts\au\run_au_training_pipeline.py",
    "--negative-dataset", $NegativeDataset,
    "--ravdess-actors", $RavdessActors,
    "--ravdess-source", $RavdessSource,
    "--ravdess-emotions", $RavdessEmotions,
    "--ravdess-cache-root", $RavdessCacheRoot,
    "--max-negative-videos", "$MaxNegativeVideos",
    "--device", $Device,
    "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers",
    "--original-au-root", $OriginalAuRoot,
    "--emotion-profile-output", $EmotionProfileOutput,
    "--emotion-min-samples-per-class", "$EmotionMinSamplesPerClass"
)

if (-not [string]::IsNullOrWhiteSpace($MetaHumanArchive)) {
    $runnerArgs += @("--metahuman-archive", $MetaHumanArchive)
}
if (-not [string]::IsNullOrWhiteSpace($MetaHumanUrl)) {
    $runnerArgs += @("--metahuman-url", $MetaHumanUrl)
}
if ($SkipNegativePreparation) {
    $runnerArgs += "--skip-negative-preparation"
}
if ($ForceAuExtraction) {
    $runnerArgs += "--force-au-extraction"
}

& $python @runnerArgs
if ($LASTEXITCODE -ne 0) {
    throw "AU training pipeline failed with exit code $LASTEXITCODE."
}

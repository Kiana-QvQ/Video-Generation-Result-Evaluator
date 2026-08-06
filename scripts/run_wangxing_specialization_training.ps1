param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cpu",
    [int]$IdentityLimit = 100,
    [int]$IdentityFrames = 1,
    [switch]$SkipIdentity
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = (Get-Command python).Source
}

$logPath = Join-Path $projectRoot "tmp\wangxing_specialization_training.log"
New-Item -ItemType Directory -Force -Path (Split-Path $logPath) | Out-Null

$arguments = @(
    "scripts\train_wangxing_specialization.py",
    "--device", $Device,
    "--identity-limit", $IdentityLimit,
    "--identity-frames", $IdentityFrames,
    "--identity-output", "data\au\wangxing_identity_profile.json",
    "--expression-output", "data\au\wangxing_expression_profile.json",
    "--source-profile-output", "data\au\wangxing_source_profile.json"
)
$manifest = Join-Path $projectRoot "data\au\WangXing_Seedance\pseudo_expression_manifest.json"
if (Test-Path -LiteralPath $manifest) {
    $arguments += @("--seedance-label-manifest", $manifest)
} else {
    Write-Warning "Seedance pseudo-label manifest is missing; run run_seedance_expression_labeling.cmd first."
}
if ($SkipIdentity) {
    $arguments += "--skip-identity"
}

Write-Host "Training Wang Xing specialization. Long-running identity work is foregrounded."
Write-Host "Log: $logPath"
& $python @arguments 2>&1 | Tee-Object -FilePath $logPath
if ($LASTEXITCODE -ne 0) {
    throw "Wang Xing specialization training failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "Profiles:"
Get-Item `
    (Join-Path $projectRoot "data\au\wangxing_expression_profile.json"),
    (Join-Path $projectRoot "data\au\wangxing_identity_profile.json"),
    (Join-Path $projectRoot "data\au\wangxing_source_profile.json") `
    -ErrorAction SilentlyContinue |
    Select-Object FullName, Length, LastWriteTime

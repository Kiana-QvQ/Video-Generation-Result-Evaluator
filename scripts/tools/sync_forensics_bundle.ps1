param(
    [string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string[]]$Targets = @()
)

$sourceRoot = Join-Path $RepositoryRoot "evaluator"
if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
    throw "Source evaluator package not found: $sourceRoot"
}

if (-not $Targets -or $Targets.Count -eq 0) {
    $Targets = @(
        Get-ChildItem -LiteralPath $RepositoryRoot -Directory |
            Where-Object {
                $_.Name -ne "evaluator" -and
                (Test-Path -LiteralPath (Join-Path $_.FullName "modules/forensics"))
            } |
            ForEach-Object { $_.Name }
    )
}

$relativeFiles = @(
    "detail_expression_metrics.py",
    "modules/core/au_from_video.py",
    "modules/core/detail_expression_runtime.py",
    "modules/core/face_landmarker.py",
    "modules/assets/ASSET_USAGE.md",
    "modules/assets/models/au_ssl_tcae.pt",
    "modules/assets/models/au_ssl_tcae.json",
    "modules/forensics/README.md",
    "modules/forensics/__init__.py",
    "modules/forensics/au_ssl.py",
    "modules/forensics/au_ssl_backbone.py",
    "modules/forensics/facial_motion.py",
    "modules/forensics/frequency_forensics.py",
    "modules/forensics/nr_vqa.py",
    "modules/forensics/perturbation.py",
    "modules/forensics/pseudo_label_calibration.py",
    "modules/forensics/physiological_rhythm.py",
    "modules/forensics/report.py",
    "modules/forensics/texture_detail.py"
)

$toolFiles = @(
    "scripts/calibrate_pseudo_labels.py",
    "scripts/run_perturbation_robustness.py",
    "scripts/train_au_ssl_backbone.py"
)

foreach ($targetName in $Targets) {
    $targetRoot = Join-Path $RepositoryRoot $targetName
    if (-not (Test-Path -LiteralPath $targetRoot -PathType Container)) {
        throw "Target directory not found: $targetRoot"
    }

    foreach ($relative in $relativeFiles) {
        $source = Join-Path $sourceRoot $relative
        $destination = Join-Path $targetRoot $relative
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Source file not found: $source"
        }
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) |
            Out-Null
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }

    foreach ($tool in $toolFiles) {
        $source = Join-Path $RepositoryRoot $tool
        $destination = Join-Path $targetRoot $tool
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) |
            Out-Null
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }

    Write-Output "Synced evaluator bundle to $targetName"
}

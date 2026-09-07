param(
    [string]$OutputPath = "experiments/20260907_authenticity_contract/baseline_snapshot.json"
)

$ErrorActionPreference = "Stop"

function Get-FileRecord {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return [ordered]@{
            path = $Path
            exists = $false
        }
    }

    $item = Get-Item -LiteralPath $Path
    $hash = Get-FileHash -LiteralPath $Path -Algorithm SHA256
    return [ordered]@{
        path = $Path
        exists = $true
        bytes = [int64]$item.Length
        last_write_time_utc = $item.LastWriteTimeUtc.ToString("o")
        sha256 = $hash.Hash.ToLowerInvariant()
    }
}

$targets = @(
    "outputs/vedio_pred/models/wangxing_v3_res1k.pt",
    "outputs/vedio_pred/models/wangxing_v5_drive.json",
    "outputs/forensics/wangxing_v5_realness_calibrator.json",
    "outputs/forensics/wangxing_v5_3_display_gate.json",
    "outputs/forensics/wangxing_v5_web_results/25x25/summary.json",
    "outputs/forensics/wangxing_v5_web_results/32x32/summary.json",
    "outputs/vedio_pred/wangxing_v5_hard_dev/hard-dev_metrics.json",
    "outputs/forensics/wangxing_v5_3_runtime_results_overnight/summary.json",
    "outputs/xiaoyue/experiment_7x7_face_v2/xiaoyue_face_manifold_profile.json",
    "outputs/xiaoyue/experiment_7x7_face_v2/xiaoyue_face_manifold_v2_metrics.json",
    "data/test/single_video/manifest.json",
    "data/test/wangxing_32x32/single_video/manifest.json",
    "data/dev/wangxing_hard_cases/single_video/manifest.json"
)

$gitHead = (git rev-parse HEAD).Trim()
$gitStatus = @(git status --short)
$snapshot = [ordered]@{
    schema_version = "authenticity_contract_baseline_snapshot_v1"
    created_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    git_head = $gitHead
    git_status_short = $gitStatus
    baseline_runtime = [ordered]@{
        binary_decision = "existing production path; do not modify during experiments"
        candidate_rule = "candidate artifacts remain isolated until frozen acceptance passes"
    }
    files = @(
        $targets | ForEach-Object { Get-FileRecord -Path $_ }
    )
}

$directory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $directory | Out-Null
$snapshot | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Output "Wrote $OutputPath"

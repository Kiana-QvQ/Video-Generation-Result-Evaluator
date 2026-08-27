param(
    [string]$RankPolicyOvernight = "outputs/forensics/wangxing_v5_2_rank_policy_overnight.json",
    [string]$OutputRoot = "outputs/forensics/wangxing_v5_3_runtime_results_overnight",
    [string]$LogFile = "outputs/forensics/wangxing_v5_3_overnight.log",
    [switch]$SkipRankRetrain
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_invoke_native.ps1")

# Overnight V5.3 optimization (does NOT retrain V3 / route A):
#   1) optional: refit RankHead only (reuse V5.2 overnight rank script)
#   2) run V5.3 manifest + gate + evaluate with the overnight rank policy
#
# Morning check:
#   outputs\forensics\wangxing_v5_3_runtime_results_overnight\leadership_brief.json
#   -> holdout.group_ordering / holdout.pairwise_ordering

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LogPath = Join-Path $ProjectRoot $LogFile
$LogDir = Split-Path -Parent $LogPath
if (-not (Test-Path -LiteralPath $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Write-Log([string]$Message, [string]$Color = "White") {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
    Write-Host $Message -ForegroundColor $Color
}

Push-Location $ProjectRoot
try {
    Write-Log "[V5.3 overnight 0/2] Log file: $LogFile" Cyan

    if (-not $SkipRankRetrain) {
        Write-Log "[V5.3 overnight 1/2] Refitting RankHead (skip 25+25/32+32)..." Cyan
        Invoke-ScriptChecked `
            -ScriptPath (Join-Path $ProjectRoot "scripts\main_workflow\run_wangxing_v5_2_overnight_rank.ps1") `
            -Stage "V5.2 overnight rank" `
            -LogPath $LogPath
    }
    else {
        Write-Log "[V5.3 overnight 1/2] Skipped RankHead refit (-SkipRankRetrain)." Yellow
    }

    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $RankPolicyOvernight))) {
        throw "Missing overnight rank policy: $RankPolicyOvernight"
    }

    Write-Log "[V5.3 overnight 2/2] Running V5.3 with overnight rank policy..." Cyan
    Invoke-ScriptChecked `
        -ScriptPath (Join-Path $ProjectRoot "scripts\main_workflow\run_wangxing_v5_3_all.ps1") `
        -Stage "V5.3 overnight evaluate" `
        -BoundParameters @{
            RankPolicyValidated = $RankPolicyOvernight
            RankPolicyFallback  = $RankPolicyOvernight
            OutputRoot          = $OutputRoot
            SkipUnitTests       = $true
            FailOnOrdering      = $true
        } `
        -LogPath $LogPath

    Write-Log "[V5.3 overnight 2b/2] Publishing web rank policy..." Cyan
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    Invoke-PythonChecked -Python $Python -Stage "publish web rank policy" -ArgumentList @(
        "scripts\web_forensics\publish_wangxing_v53_web_rank_policy.py",
        "--source", $RankPolicyOvernight
    ) -OnLine {
        param($Line)
        Add-Content -LiteralPath $LogPath -Value $Line -Encoding UTF8
        Write-Host $Line
    }

    Write-Log "[V5.3 overnight] Done." Green
    Write-Log "Brief: $OutputRoot\leadership_brief.json" Yellow
    Write-Log "Same-prompt: outputs\forensics\wangxing_v5_3_same_prompt_results_overnight\leadership_brief.json" Yellow
    Write-Log "Web rank policy: outputs\forensics\wangxing_v5_3_web_rank_policy.json" Yellow
    Write-Log "Log: $LogFile" Yellow
}
catch {
    Write-Log "[V5.3 overnight] FAILED: $($_.Exception.Message)" Red
    throw
}
finally {
    Pop-Location
}

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PtScript = Join-Path $ProjectRoot "scripts\pt_training\run_wangxing_v5.ps1"
$WebScript = Join-Path $ProjectRoot "scripts\web_forensics\run_wangxing_web_v5.ps1"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

Push-Location $ProjectRoot
try {
    Write-Host "===== Wang Xing V5: PT =====" -ForegroundColor Cyan
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $PtScript
    if ($LASTEXITCODE -ne 0) {
        throw "PT v5 failed with exit code $LASTEXITCODE"
    }

    Write-Host "===== Wang Xing V5: Web =====" -ForegroundColor Cyan
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $WebScript
    if ($LASTEXITCODE -ne 0) {
        throw "Web v5 failed with exit code $LASTEXITCODE"
    }
    Write-Host "===== Wang Xing V5: all completed =====" -ForegroundColor Green
}
finally {
    Pop-Location
}

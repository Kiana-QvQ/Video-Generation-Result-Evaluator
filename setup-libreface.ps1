param(
    [switch]$Cuda
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv-libreface\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Creating isolated LibreFace environment..."
    python -m venv (Join-Path $root ".venv-libreface")
}

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "LibreFace pip upgrade failed."
}

& $python -m pip install -r (Join-Path $root "requirements\libreface.txt")
if ($LASTEXITCODE -ne 0) {
    throw "LibreFace dependency installation failed."
}

& $python -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "LibreFace environment failed pip check."
}

Write-Host "LibreFace environment is ready: $python"
Write-Host "Use --libreface-python `"$python`" with scripts\extract_libreface_au.py."

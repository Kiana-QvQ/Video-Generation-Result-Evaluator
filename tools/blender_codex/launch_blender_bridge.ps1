param(
    [int]$Port = 9876,
    [string]$BlendFile = ""
)

$ErrorActionPreference = "Stop"
$blender = "D:\Steam\steamapps\common\Blender\blender.exe"
$bridge = Join-Path $PSScriptRoot "blender_bridge_addon.py"

if (-not (Test-Path -LiteralPath $blender)) {
    throw "Blender was not found at $blender"
}
if (-not (Test-Path -LiteralPath $bridge)) {
    throw "Bridge script was not found at $bridge"
}

$env:BLENDER_CODEX_HOST = "127.0.0.1"
$env:BLENDER_CODEX_PORT = "$Port"

$arguments = @()
if ($BlendFile) {
    $arguments += $BlendFile
}
$arguments += @("--python", $bridge)

$process = Start-Process `
    -FilePath $blender `
    -ArgumentList $arguments `
    -WorkingDirectory (Split-Path -Parent $bridge) `
    -PassThru

Write-Output "Started Blender bridge (PID $($process.Id)) on 127.0.0.1:$Port"

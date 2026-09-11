param(
    [Parameter(Mandatory = $true)]
    [int]$PipelineProcessId
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$OutputRoot = Join-Path $ProjectRoot "outputs\zijie\face_manifold_v1"
$Log = Join-Path $OutputRoot "acceptance_gate.log"

Wait-Process -Id $PipelineProcessId -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath (Join-Path $OutputRoot "zijie_face_manifold_v1_metrics.json"))) {
    Add-Content -LiteralPath $Log -Value "[ZiJie gate] Pipeline exited without PT metrics."
    exit 2
}
if (-not (Test-Path -LiteralPath (Join-Path $OutputRoot "offline_profile_test\all_results.json"))) {
    Add-Content -LiteralPath $Log -Value "[ZiJie gate] Pipeline exited without offline profile results."
    exit 2
}

& $Python "scripts\pt_training\validate_zijie_face_manifold_v1.py" `
    "--pt-metrics" "outputs\zijie\face_manifold_v1\zijie_face_manifold_v1_metrics.json" `
    "--offline-results" "outputs\zijie\face_manifold_v1\offline_profile_test\all_results.json" `
    "--output" "outputs\zijie\face_manifold_v1\acceptance_gate.json" *>&1 |
    Tee-Object -FilePath $Log
exit $LASTEXITCODE

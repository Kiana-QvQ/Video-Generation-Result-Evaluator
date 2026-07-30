param(
    [ValidateSet("2b", "2.5-3b")]
    [string]$JudgeModel = "2b",
    [string]$SglangImage = ""
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..")).Path
$env:DOCKER_CONFIG = Join-Path $root ".docker"
New-Item -ItemType Directory -Force -Path $env:DOCKER_CONFIG | Out-Null
if ([string]::IsNullOrWhiteSpace($SglangImage)) {
    $SglangImage = $env:FRAME_AUDIT_SGLANG_IMAGE
}
if ([string]::IsNullOrWhiteSpace($SglangImage)) {
    $SglangImage = "lmsysorg/sglang:latest"
}

$modelSpec = @{
    "2b" = @{
        Name = "Qwen2-VL-2B-Instruct-AWQ"
        ServedName = "qwen2-vl-2b-awq"
    }
    "2.5-3b" = @{
        Name = "Qwen2.5-VL-3B-Instruct-AWQ"
        ServedName = "qwen2.5-vl-3b-awq"
    }
}[$JudgeModel]
$modelPath = Join-Path $root "model_cache\vlm_judge\$($modelSpec.Name)"
$weightFiles = Get-ChildItem -LiteralPath $modelPath -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -eq ".safetensors" -or $_.Name -eq "pytorch_model.bin" }
if (-not (Test-Path $modelPath -PathType Container) -or -not $weightFiles) {
    throw "$($modelSpec.Name) is missing. Run .\scripts\download-vlm-judge.ps1 -JudgeModel $JudgeModel first."
}

& docker image inspect $SglangImage *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "SGLang image is not cached locally; pulling $SglangImage..."
    & docker pull $SglangImage
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to pull $SglangImage. Docker Hub may be rate-limiting anonymous requests. Run 'docker login' or set FRAME_AUDIT_SGLANG_IMAGE to an accessible mirror."
    }
}

docker run --rm --gpus all `
    --pull never `
    --shm-size 32g `
    --ipc=host `
    -p 30000:30000 `
    --name "frame-audit-qwen-$JudgeModel" `
    -v "${modelPath}:/models/judge:ro" `
    $SglangImage `
    python3 -m sglang.launch_server `
    --model-path /models/judge `
    --host 0.0.0.0 `
    --port 30000 `
    --served-model-name $modelSpec.ServedName

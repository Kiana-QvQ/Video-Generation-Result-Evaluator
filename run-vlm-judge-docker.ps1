param(
    [ValidateSet("2b", "2.5-3b")]
    [string]$JudgeModel = "2b"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:DOCKER_CONFIG = Join-Path $root ".docker"
New-Item -ItemType Directory -Force -Path $env:DOCKER_CONFIG | Out-Null

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
if (-not (Test-Path (Join-Path $modelPath "model.safetensors"))) {
    throw "$($modelSpec.Name) is missing. Run .\download-compact-models.ps1 -JudgeModel $JudgeModel first."
}

docker run --rm --gpus all `
    --shm-size 32g `
    --ipc=host `
    -p 30000:30000 `
    -v "${modelPath}:/models/judge:ro" `
    lmsysorg/sglang:latest `
    python3 -m sglang.launch_server `
    --model-path /models/judge `
    --host 0.0.0.0 `
    --port 30000 `
    --served-model-name $modelSpec.ServedName

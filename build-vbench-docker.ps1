param(
    [string]$CudaBaseImage = "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:DOCKER_CONFIG = Join-Path $root ".docker"
New-Item -ItemType Directory -Force -Path $env:DOCKER_CONFIG | Out-Null

Set-Location $root
$env:CUDA_BASE_IMAGE = $CudaBaseImage
docker compose -f docker-compose.vbench.yml build

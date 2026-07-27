param(
    [switch]$SkipDinoRepository
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$downloadHelper = Join-Path $root "tools\download_url.py"
$cache = Join-Path $root "model_cache"
$vbench = Join-Path $cache "vbench"
$viclip = Join-Path $cache "viclip"
$dinoRoot = Join-Path $vbench "dino_model\facebookresearch_dino_main"
$dinoCompat = Join-Path $root "tools\dino_compat"
$env:DOCKER_CONFIG = Join-Path $root ".docker"
New-Item -ItemType Directory -Force -Path $env:DOCKER_CONFIG | Out-Null

function Download-File([string]$Url, [string]$Destination) {
    if ((Test-Path $Destination) -and ((Get-Item $Destination).Length -gt 0)) {
        Write-Host "Already present: $Destination"
        return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    Write-Host "Downloading: $Url"
    & curl.exe -L --fail --retry 6 --retry-delay 5 --retry-all-errors --http1.1 `
        --output $Destination $Url
    if ($LASTEXITCODE -ne 0) {
        if (-not (Test-Path $python)) {
            throw "Download failed with exit code ${LASTEXITCODE}: $Url"
        }
        Write-Host "curl failed; retrying with project Python requests..."
        & $python $downloadHelper $Url $Destination
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed with both curl and Python requests: $Url"
        }
    }
    if ((Get-Item $Destination).Length -le 0) {
        throw "Downloaded file is empty: $Destination"
    }
}

function Install-DinoCompatibility {
    if (-not (Test-Path (Join-Path $dinoCompat "hubconf.py"))) {
        throw "Tracked DINO compatibility files are missing: $dinoCompat"
    }
    New-Item -ItemType Directory -Force -Path $dinoRoot | Out-Null
    Copy-Item -Path (Join-Path $dinoCompat "*") -Destination $dinoRoot -Force
    Write-Host "Installed offline DINO compatibility source: $dinoRoot"
}

if (-not $SkipDinoRepository) {
    if (-not (Test-Path (Join-Path $dinoRoot ".git"))) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dinoRoot) | Out-Null
        try {
            git clone --depth 1 https://github.com/facebookresearch/dino.git $dinoRoot
            if ($LASTEXITCODE -ne 0) {
                throw "git clone returned exit code $LASTEXITCODE"
            }
        } catch {
            Write-Host "git clone failed; downloading the DINO source archive..."
            try {
                $dinoZip = Join-Path $vbench "dino_model\dino-main.zip"
                Download-File `
                    "https://github.com/facebookresearch/dino/archive/refs/heads/main.zip" `
                    $dinoZip
                $dinoExtract = Join-Path $vbench "dino_model\_extract"
                if (Test-Path $dinoExtract) {
                    Remove-Item -LiteralPath $dinoExtract -Recurse -Force
                }
                Expand-Archive -LiteralPath $dinoZip -DestinationPath $dinoExtract -Force
                $extractedRoot = Get-ChildItem -LiteralPath $dinoExtract -Directory | Select-Object -First 1
                if (Test-Path $dinoRoot) {
                    Remove-Item -LiteralPath $dinoRoot -Recurse -Force
                }
                Move-Item -LiteralPath $extractedRoot.FullName -Destination $dinoRoot
                Remove-Item -LiteralPath $dinoExtract -Recurse -Force
            } catch {
                Write-Host "DINO archive is unavailable; using the local timm compatibility source."
            }
        }
    }
}

if (-not (Test-Path (Join-Path $dinoRoot "hubconf.py"))) {
    Install-DinoCompatibility
}

Download-File `
    "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth" `
    (Join-Path $vbench "dino_model\dino_vitbase16_pretrain.pth")

Download-File `
    "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth?download=true" `
    (Join-Path $vbench "amt_model\amt-s.pth")

Download-File `
    "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth" `
    (Join-Path $vbench "aesthetic_model\emb_reader\sa_0_4_vit_l_14_linear.pth")

Download-File `
    "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt" `
    (Join-Path $vbench "clip_model\ViT-L-14.pt")

Download-File `
    "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth" `
    (Join-Path $vbench "pyiqa_model\musiq_spaq_ckpt-358bb6af.pth")

Download-File `
    "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth?download=true" `
    (Join-Path $viclip "ViClip-InternVid-10M-FLT.pth")

$clipB32 = Join-Path $cache "clip\ViT-B-32.pt"
$vbenchClipB32 = Join-Path $vbench "clip_model\ViT-B-32.pt"
if ((Test-Path $clipB32) -and -not (Test-Path $vbenchClipB32)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $vbenchClipB32) | Out-Null
    Copy-Item -LiteralPath $clipB32 -Destination $vbenchClipB32
}

$raftArchive = Join-Path $vbench "raft_model\models.zip"
$raftRoot = Join-Path $vbench "raft_model"
try {
    Download-File `
        "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip" `
        $raftArchive
} catch {
    Write-Host "Official Dropbox RAFT archive unavailable; downloading raft-things.pth from Hugging Face mirror."
    $raftModelPath = Join-Path $raftRoot "models\raft-things.pth"
    Download-File `
        "https://huggingface.co/ddrfan/RAFT/resolve/main/raft-things.pth?download=true" `
        $raftModelPath
}
if (-not (Test-Path (Join-Path $raftRoot "models\raft-things.pth"))) {
    Expand-Archive -LiteralPath $raftArchive -DestinationPath $raftRoot -Force
}

Write-Host ""
Write-Host "VBench model assets are under: $vbench"
Get-ChildItem -LiteralPath $vbench -Recurse -File |
    Select-Object FullName, Length

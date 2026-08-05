param(
    [switch]$SkipPythonPackages
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$cache = Join-Path $root "model_cache"
$insightfaceRoot = Join-Path $cache "insightface"
$insightfaceModels = Join-Path $insightfaceRoot "models"
$iqaRoot = Join-Path $cache "hub\pyiqa"
$iqaWeights = $iqaRoot
$clipRoot = Join-Path $cache "clip"
$clipWeight = Join-Path $clipRoot "ViT-B-32.pt"
$manifestPath = Join-Path $cache "OPTIONAL_ASSETS.json"

function Download-File {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$ExpectedSha256 = "",
        [long]$ExpectedBytes = 0
    )
    if (Test-Path $Destination) {
        $item = Get-Item $Destination
        $valid = $false
        if ($ExpectedBytes -gt 0) { $valid = $item.Length -eq $ExpectedBytes }
        if ($ExpectedSha256) {
            $actualSha256 = (Get-FileHash $Destination -Algorithm SHA256).Hash
            $valid = $item.Length -gt 0 -and $actualSha256 -eq $ExpectedSha256
        }
        if ($valid) { Write-Host "Verified existing asset: $Destination"; return }
        Remove-Item -LiteralPath $Destination -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    $partial = "$Destination.part"
    Write-Host "Downloading: $Url"
    & curl.exe -L --fail --retry 6 --retry-delay 5 --retry-all-errors --http1.1 `
        --output $partial $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed with exit code ${LASTEXITCODE}: $Url"
    }
    if (-not (Test-Path $partial) -or (Get-Item $partial).Length -le 0) {
        Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
        throw "Downloaded file is empty: $Url"
    }
    if ($ExpectedBytes -gt 0 -and (Get-Item $partial).Length -ne $ExpectedBytes) {
        Remove-Item -LiteralPath $partial -Force
        throw "Downloaded file has an unexpected size: $Url"
    }
    if ($ExpectedSha256 -and ((Get-FileHash $partial -Algorithm SHA256).Hash -ne $ExpectedSha256)) {
        Remove-Item -LiteralPath $partial -Force
        throw "Downloaded file has an unexpected SHA256: $Url"
    }
    Move-Item -LiteralPath $partial -Destination $Destination -Force
}

if (-not $SkipPythonPackages) {
    if (-not (Test-Path $python)) {
        throw "Project Python environment is missing. Run .\setup.ps1 first."
    }
    & $python -m pip install -r (Join-Path $root "requirements\optional.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "Optional dependency installation failed."
    }
}

New-Item -ItemType Directory -Force -Path $insightfaceModels, $iqaWeights, $clipRoot | Out-Null

$buffaloArchive = Join-Path $insightfaceRoot "buffalo_l.zip"
Download-File `
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" `
    $buffaloArchive

$buffaloExtract = Join-Path $insightfaceRoot "_extract_buffalo_l"
if (-not (Test-Path (Join-Path $insightfaceModels "buffalo_l"))) {
    if (Test-Path $buffaloExtract) {
        Remove-Item -LiteralPath $buffaloExtract -Recurse -Force
    }
    Expand-Archive -LiteralPath $buffaloArchive -DestinationPath $buffaloExtract -Force
    $buffaloTarget = Join-Path $insightfaceModels "buffalo_l"
    New-Item -ItemType Directory -Force -Path $buffaloTarget | Out-Null
    $files = Get-ChildItem -LiteralPath $buffaloExtract -File
    if (-not $files) {
        throw "No files were found after extracting buffalo_l."
    }
    $files | Move-Item -Destination $buffaloTarget -Force
    Remove-Item -LiteralPath $buffaloExtract -Recurse -Force
}

Download-File `
    "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/MANIQA_PIPAL-ae6d356b.pth?download=true" `
    (Join-Path $iqaWeights "MANIQA_PIPAL-ae6d356b.pth") `
    "" `
    543335435
Download-File `
    "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/musiq_koniq_ckpt-e95806b9.pth?download=true" `
    (Join-Path $iqaWeights "musiq_koniq_ckpt-e95806b9.pth") `
    "" `
    108610983
Download-File `
    "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt" `
    $clipWeight `
    "40D365715913C9DA98579312B702A82C18BE219CC2A73407C4526F58EBA950AF"

$manifest = [ordered]@{
    generated_at = (Get-Date).ToString("o")
    project = $root
    note = "Downloaded only. The evaluator has not been started by this script."
    cpu_safe_defaults = @{
        EVALUATOR_FACE_DEVICE = "cpu"
        EVALUATOR_SEMANTIC_DEVICE = "cpu"
        EVALUATOR_IQA_DEVICE = "cpu"
    }
    assets = @(
        @{
            name = "insightface_buffalo_l"
            path = (Join-Path $insightfaceModels "buffalo_l")
            archive = $buffaloArchive
            bytes = (Get-Item $buffaloArchive).Length
            sha256 = (Get-FileHash $buffaloArchive -Algorithm SHA256).Hash
            source = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
        },
        @{
            name = "maniqa_pipal"
            path = (Join-Path $iqaWeights "MANIQA_PIPAL-ae6d356b.pth")
            bytes = (Get-Item (Join-Path $iqaWeights "MANIQA_PIPAL-ae6d356b.pth")).Length
            sha256 = (Get-FileHash (Join-Path $iqaWeights "MANIQA_PIPAL-ae6d356b.pth") -Algorithm SHA256).Hash
            source = "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/MANIQA_PIPAL-ae6d356b.pth"
        },
        @{
            name = "musiq_koniq"
            path = (Join-Path $iqaWeights "musiq_koniq_ckpt-e95806b9.pth")
            bytes = (Get-Item (Join-Path $iqaWeights "musiq_koniq_ckpt-e95806b9.pth")).Length
            sha256 = (Get-FileHash (Join-Path $iqaWeights "musiq_koniq_ckpt-e95806b9.pth") -Algorithm SHA256).Hash
            source = "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/musiq_koniq_ckpt-e95806b9.pth"
        },
        @{
            name = "openai_clip_vit_b32"
            path = $clipWeight
            bytes = (Get-Item $clipWeight).Length
            sha256 = (Get-FileHash $clipWeight -Algorithm SHA256).Hash
            source = "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
        }
    )
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $manifestPath

Write-Host ""
Write-Host "Optional assets are downloaded under model_cache."
Write-Host "No evaluator or model inference was started."

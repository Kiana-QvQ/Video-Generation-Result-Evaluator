# Model Selection

The project uses a hardware-first policy instead of downloading an unverified
VideoScore2 quantization package.

| GPU memory | Default judge | Approx. weight size | Role |
| --- | --- | ---: | --- |
| 8 GB | Qwen2-VL-2B-Instruct-AWQ | 2.74 GB | Default ETVA judge |
| 12 GB or more | Qwen2.5-VL-3B-Instruct-AWQ | 3.42 GB | Optional quality upgrade |
| 24 GB or more | TIGER-Lab/VideoScore2 BF16 | 16.6 GB | Optional large-model upgrade |

Only one VLM judge should be resident at a time. The 8GB default is already
cached under `model_cache/vlm_judge/Qwen2-VL-2B-Instruct-AWQ`; do not download
VideoScore2 just to obtain a theoretical 4.2GB quantized size.

## Runtime Tiers

The evaluator detects CUDA and total VRAM at runtime. You can override the
detected value when testing a remote machine:

```powershell
$env:EVALUATOR_GPU_MEMORY_GB = "8"
```

The scheduler uses these tiers:

- `cpu`: no GPU model acceleration.
- `compact_8gb`: Qwen2-VL-2B AWQ, eight ViCLIP frames, four ETVA frames.
- `balanced_12gb`: Qwen2.5-VL-3B AWQ, serial VLM/ViCLIP execution.
- `full_24gb`: VideoScore2 is opt-in only and requires a verified backend.

ViCLIP is released before the Qwen Judge request. LPIPS also releases its
temporary GPU model after a full-reference pass.

## VBench Offline Path

The DINO checkpoint remains under `model_cache/vbench/dino_model`. When the
GitHub DINO repository cannot be cloned, the downloader and runner install the
tracked `tools/dino_compat` hub entrypoint. It uses the same ViT-B/16 parameter
layout through `timm`, so the existing checkpoint can be loaded offline.

## Docker Build

The VBench image no longer clones VBench from GitHub during the build. It
installs the published package and accepts a CUDA base image override:

```powershell
.\build-vbench-docker.ps1 -CudaBaseImage "nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04"
```

The default remains `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`.

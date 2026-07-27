# Compact Profile

The project now follows the compact serial profile:

- VBench assets: `model_cache/vbench`
- ViCLIP checkpoint: `model_cache/viclip`
- Qwen2-VL-2B AWQ ETVA judge: `model_cache/vlm_judge`
- ArcFace, LPIPS, MANIQA, and MUSIQ: `model_cache`

Current cache size is approximately 8.6 GB.

## Download

```powershell
.\download-vbench-models.ps1 -SkipDinoRepository
.\download-compact-models.ps1
```

The DINO checkpoint is present. If the original GitHub checkout is unavailable,
the runner installs a small local timm-compatible hub source from
`tools/dino_compat`, so the subject-consistency dimension does not depend on
GitHub.

## VLM Judge

The compact VLM is Qwen2-VL-2B AWQ. It is the default ETVA judge for the 8GB
GPU profile and should be loaded serially:

```powershell
.\run-vlm-judge-docker.ps1
```

This starts an OpenAI-compatible service on port `30000`. The SGLang Docker
image must be available locally or pullable by Docker Desktop.

The downloaded weights alone do not start inference. Keep the judge process
running while evaluating a prompt:

```powershell
$env:ETVA_JUDGE_ENABLED = "1"
.\run-vlm-judge-docker.ps1
```

ViCLIP is enabled automatically for CUDA evaluations when its checkpoint is
present. To force it on CPU for a smoke test:

```powershell
$env:EVALUATOR_VICLIP_ENABLED = "1"
```

## VBench Docker

Build the GPU backend after the Docker registry can pull the CUDA base image:

```powershell
.\build-vbench-docker.ps1
```

The VBench runner then prefers the local Docker image and mounts
`model_cache/vbench` into the container.

## Hardware

The current machine has an RTX 2080 SUPER with 8 GB VRAM. Use CPU offload,
short frame samples, and serial model loading. Qwen2.5-VL-3B AWQ is an optional
12GB upgrade. VideoScore2 BF16 is reserved for 24GB-class GPUs.

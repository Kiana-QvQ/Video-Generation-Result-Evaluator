# 模型与资源

项目采用硬件优先、串行加载策略。评估器不会为了获得理论上的更小模型而
自动下载未经验证的量化包，也不会同时常驻多个重量级模型。

## 硬件档位

| 显存 | 默认模型 | 用途 |
| --- | --- | --- |
| CPU | 轻量代理后端 | 基础评估和离线检查 |
| 8 GB | Qwen2-VL-2B-Instruct-AWQ | 默认 ETVA Judge，串行推理 |
| 12 GB 以上 | Qwen2.5-VL-3B-Instruct-AWQ | 可选质量升级 |
| 24 GB 以上 | VideoScore2 BF16 | 可选大型质量升级，需单独验证后端 |

运行时会自动检测 CUDA 和显存，也可以在测试远程机器时覆盖显存：

```powershell
$env:EVALUATOR_GPU_MEMORY_GB = "8"
```

实际选择规则和缓存配置保存在
`config/model_profile.json`，由 `evaluator/model_profile.py` 读取。

## 缓存目录

所有模型和框架缓存都放在 `model_cache/`，包括：

- `model_cache/vlm_judge/`：Qwen VLM Judge
- `model_cache/viclip/`：ViCLIP 权重
- `model_cache/vbench/`：VBench 资源和 DINO/RAFT 等依赖
- `model_cache/insightface/`：ArcFace 权重
- `model_cache/hub/pyiqa/`：下载脚本保存的 MANIQA 和 MUSIQ 原始权重
- `model_cache/hub/checkpoints/`：pyiqa/torch.hub 实际读取的 MANIQA 和 MUSIQ 权重
- `model_cache/clip/`：OpenAI CLIP ViT-B/32 权重

这些目录已被 `.gitignore` 忽略，不应提交到仓库。

## 可选模型安装

先安装可选 Python 后端，再下载权重：

```powershell
.\setup.ps1 -Optional
.\scripts\download-optional-assets.ps1 -SkipPythonPackages
```

下载脚本会生成 `model_cache/OPTIONAL_ASSETS.json`，记录资源大小、来源和
SHA256。基础评估不依赖这些大模型；缺失时网页端会显示 `OPTIONAL` 或
`OFFLINE`，不会伪造精确模型结果。评估器首次使用时会把完整 IQA 权重原子桥接到
`hub/checkpoints`，清理同名 `.partial` 残片，并阻止 pyiqa 自动重新下载大文件。

## VLM Judge

默认 8GB 档位使用 Qwen2-VL-2B AWQ。下载并启动本地 OpenAI 兼容服务：

```powershell
.\scripts\download-vlm-judge.ps1
$env:ETVA_JUDGE_ENABLED = "1"
.\scripts\run-vlm-judge-docker.ps1
```

12GB 升级模型：

```powershell
.\scripts\download-vlm-judge.ps1 -JudgeModel 2.5-3b
.\scripts\run-vlm-judge-docker.ps1 -JudgeModel 2.5-3b
```

服务默认监听 `127.0.0.1:30000` 映射的 Docker 端口。下载权重不会自动
启动服务，评估期间需要保持 Judge 容器运行。

## VBench

VBench 是独立的可选后端，建议使用 Docker 运行：

```powershell
.\setup.ps1 -VBench
.\scripts\download-vbench-models.ps1 -SkipDinoRepository
.\docker\build-vbench.ps1
```

如果需要完整的 DINO 源码，可省略 `-SkipDinoRepository`。下载失败时，
脚本会回退到仓库内的 `tools/dino_compat`，保证 DINO 入口可离线加载。
VBench 结果写入 `outputs/vbench/`，不参与主流程的五类加权总分。

自定义 CUDA 基础镜像：

```powershell
.\docker\build-vbench.ps1 `
    -CudaBaseImage "nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04"
```

## 评估准则

原始需求截图已整理到
`docs/evaluation-criteria.png`。主流程的五类权重为：

| 类别 | 权重 | 主要输入 |
| --- | ---: | --- |
| 角色一致性 | 35% | 参考图、参考视频或结果自身 |
| 质感和细节 | 15% | GT 视频优先，否则使用无 GT 质量后端 |
| 表情准确 | 15% | 参考动作视频、Prompt 或人工评分 |
| 时间稳定性 | 25% | 参考视频、landmark、光流和 warping |
| 美学质量 | 10% | 人工评分与技术代理 |

只有逐帧对应的 GT 视频才会计算 PSNR、SSIM、LPIPS。普通参考图和参考动作
视频不能替代 GT。

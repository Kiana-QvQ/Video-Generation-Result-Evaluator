# 视频生成模型结果评估器

一个本地运行的视频生成结果评估器。上传生成结果视频后，可按需补充
GT 视频、参考图、参考动作视频和文本 Prompt，系统会选择适用的评估路径，
并为每个指标记录状态、实际后端和警告。

## 评估范围

| 类别 | 权重 | 主要逻辑 |
| --- | ---: | --- |
| 角色一致性 | 35% | ArcFace 或人脸特征代理 |
| 质感和细节 | 15% | 有 GT 时使用 PSNR、SSIM、LPIPS，否则使用无 GT 质量后端 |
| 表情准确 | 15% | 参考动作、Prompt 对齐或人工评分 |
| 时间稳定性 | 25% | landmark jitter、人脸框抖动、光流和 warping |
| 美学质量 | 10% | 人工评分与曝光、清晰度、色彩等技术代理 |

只有与结果逐帧对应的 GT 视频才会计算 PSNR、SSIM、LPIPS。参考图和参考
动作视频不能替代 GT。

## 快速开始

```powershell
.\setup.ps1
.\run.ps1
```

打开 `http://127.0.0.1:7860`。当前唯一的实际服务入口是
`web_app.py`；`run-web.ps1` 和 `run-web.cmd` 仅作为旧命令和双击启动的
兼容包装器。

如果 PowerShell 的执行策略阻止脚本运行，可以使用：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run.ps1
```

## 可选后端

基础环境不依赖重模型，也不会把代理结果标记成精确模型结果。安装可选后端：

```powershell
.\setup.ps1 -Optional
.\download-optional-assets.ps1 -SkipPythonPackages
```

显存和模型选择、VLM Judge、ViCLIP 与 VBench 的完整说明见
[`docs/MODEL_AND_ASSETS.md`](docs/MODEL_AND_ASSETS.md)。

常用的独立后端命令：

```powershell
# VBench
.\setup.ps1 -VBench
.\download-vbench-models.ps1 -SkipDinoRepository
.\build-vbench-docker.ps1

# 8GB 默认 VLM Judge
.\download-compact-models.ps1
.\run-vlm-judge-docker.ps1
```

VBench 和 VLM Judge 都是可选能力，不会自动启动，也不影响基础五类评估。
Docker 构建需要 Docker Desktop、NVIDIA 容器支持和可用的 CUDA 基础镜像。

## 输出与测试

网页端会在 `outputs/web_runs/` 保存任务状态、原始 JSON、汇总 CSV 和逐帧
CSV；所有本地缓存和生成结果都已加入 `.gitignore`。

运行测试：

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

原始评估准则截图整理在
[`docs/evaluation-criteria.png`](docs/evaluation-criteria.png)。

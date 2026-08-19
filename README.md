# Video Generation Evaluator

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

## 两条入口（不要混）

| 入口 | 用途 | 怎么启动 |
| --- | --- | --- |
| **`evaluator/`（对方核心）** | 视频评估 Flask Web | `cd evaluator` 后运行 `python server.py` |
| **仓库根目录 Frame Audit / gRPC** | `web_app.py` + gRPC | `python start.py` / `.\run-grpc.ps1` |

本仓库里：先在**项目侧**（`scripts/`、`outputs/`、`human_review/` 等）验证算法；确认有效后再合并进 `evaluator/modules/`，不要直接改散对方宿主入口。

```powershell
# 对方评估器 Web
cd evaluator
..\.venv\Scripts\python.exe server.py

# Frame Audit Web
.\.venv\Scripts\python.exe start.py

# gRPC（根目录脚本仅保留这一套）
.\run-grpc.ps1
```

打开评估器：`http://127.0.0.1:5000`（`evaluator/server.py` 默认端口）。

## 仓库里有什么 / 没有什么

GitHub **不包含**数据集、原始视频、全量 AU、本地 `.pt`、完整 `outputs/`、大模型缓存，以及内部领导汇报口径文稿。  
详见 [`docs/仓库内容与数据策略.md`](docs/仓库内容与数据策略.md)。

## 多用户队列

网页任务按客户端 IP 隔离。每个 IP 内部按提交顺序 FIFO 排队；不同 IP
之间使用 HRRN（最高响应比优先）和等待老化调度，较轻任务可以先完成，
长任务会随等待时间提升优先级，不会无限等待。评估本身仍使用单 worker，
避免多个用户同时占用同一组 GPU 模型。任务状态会返回估计耗时和调度器名称。

如果服务部署在可信反向代理后，并且代理会正确设置
`X-Forwarded-For`，可启用：

```powershell
$env:FRAME_AUDIT_TRUST_PROXY_HEADERS = "true"
.\.venv\Scripts\python.exe start.py
```

不要在没有可信代理的情况下开启该选项，否则客户端可以伪造来源 IP。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `evaluator/` | 对方核心：`server.py` / `app.py` / `main.py` / `modules/` |
| `scripts/` / `outputs/` / `human_review/` / `docs/` | 本项目验证、训练评估与人审（验证后再进 `evaluator/modules`） |
| `web_app.py` / `start.py` | Frame Audit Web（也被 gRPC 复用） |
| `grpc_server.py` / `start_grpc.py` / `run-grpc.*` / `grpc_api/` | gRPC（根目录仅保留这组 `.ps1`/`.cmd`） |
| `requirements.txt` / `requirements/` | 依赖 |
| `web/` / `config/` / `docker/` / `tools/` | Frame Audit / 可选后端资源 |
| `data/` | 数据集与 AU |
| `data/test/with_reference/` | 原始带参考生成实验的可读导出，不是 `performance_v8` 投票题库 |

带参考实验的唯一源数据位于
`human_review/data/raw_archive/experiments_20260811`。导出副本由
`scripts/export_human_review_reference_set.py` 生成：每组实验包含
`prompt.txt`、`reference_inputs/`、`generated_videos/` 和 `experiment.json`；
当前源归档共 18 组，其中 14 组有提示词并导出，4 组无提示词而跳过。

## 可选后端

基础环境不依赖重模型，也不会把代理结果标记成精确模型结果。可选资源：

```powershell
.\scripts\download-optional-assets.ps1 -SkipPythonPackages
```

显存和模型选择、VLM Judge、ViCLIP 与 VBench 的完整说明见
[`docs/MODEL_AND_ASSETS.md`](docs/MODEL_AND_ASSETS.md)。

常用的独立后端命令：

```powershell
# VBench（隔离 Docker，不要装进 .venv）
.\scripts\download-vbench-models.ps1 -SkipDinoRepository
.\docker\build-vbench.ps1

# 本地 VLM Judge
.\scripts\download-vlm-judge.ps1
.\.venv\Scripts\python.exe start.py --with-vlm
# 或
.\run-grpc.ps1 -WithVlm
```

VBench 是可选能力；Qwen VLM Judge 默认不会随基础启动自动加载。
本地 Qwen Judge 读取 `model_cache/vlm_judge/` 中的权重。

## 输出与测试

网页端会在 `outputs/web_runs/` 保存任务状态、原始 JSON、汇总 CSV 和逐帧
CSV；所有本地缓存和生成结果都已加入 `.gitignore`。

运行测试：

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

原始评估准则截图整理在
[`docs/evaluation-criteria.png`](docs/evaluation-criteria.png)。

公网或非回环地址部署前请阅读 [`docs/SECURITY.md`](docs/SECURITY.md)；
必须配置 API key、TLS、资源限制和任务保留策略。VBench、Local VLM 与
LibreFace 使用隔离依赖环境，不要把它们安装到同一个 `.venv`。

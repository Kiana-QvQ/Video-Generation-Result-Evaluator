# 视频生成模型结果评估器

这是一个本地运行的首版视频生成结果评估器，采用“上传生成结果视频 + 上传参考素材”
的方式完成五类评估。系统会根据输入素材自动选择全参考、参考视频、Prompt 对齐、
结果自身对齐或人工评分路径，并把每个指标的状态和后端写入结果。

| 类别 | 权重 | 主要逻辑 |
| --- | ---: | --- |
| 角色一致性 | 35% | ArcFace（可选）或人脸特征代理；平均相似度、尾部 10% 相似度、方差 |
| 质感和细节 | 15% | **有 GT：PSNR、SSIM、LPIPS；无 GT：不计算这三项，改用 MANIQA、MUSIQ（可选）和高频能比** |
| 表情准确 | 15% | 有参考表情视频：VideoCLIP/ViCLIP 插槽 + 运动轨迹代理；否则人工 1~5 分 |
| 时间稳定性 | 25% | 身份变化、MediaPipe landmark jitter（可选）或人脸框抖动、warping error、参考光流误差 |
| 美学质量 | 10% | 人工 1~5 分为主，曝光/清晰度/色彩技术代理为辅助 |

## 首版输入范围

| 输入 | 是否必填 | 用途 |
| --- | --- | --- |
| 生成结果视频 | 必填 | 被评估的视频 |
| GT 参考视频 | 可选 | 必须与结果内容和时间逐帧对应；仅用于第 2 类 PSNR、SSIM、LPIPS，并可作为其他类别的同步参考 |
| 参考图 | 可选 | 角色外观、身份一致性基准；不是 GT |
| 参考动作视频 | 可选 | 表情、动作和时间稳定性基准；不是 GT |
| 文本 Prompt | 可选 | 文本-视频语义对齐 |

**GT 和普通参考素材不是同一个概念。** 只有上传逐帧对应的 GT 视频，
第 2 类才会计算 PSNR、SSIM、LPIPS。参考图或参考动作视频只能用于身份、
运动、语义和稳定性等相关评估，不能替代 GT。

首版推荐输入组合：

```text
结果视频 + Prompt + 参考图
结果视频 + Prompt + 参考动作视频
结果视频 + GT 视频
结果视频 + GT 视频 + 参考图/参考动作视频
```

## 运行

```powershell
.\setup.ps1
.\run.ps1
```

打开 `http://127.0.0.1:7860`。

## 网页版首入口

首版网页不需要注册或登录，直接运行：

```powershell
.\run-web.ps1
```

然后打开 `http://127.0.0.1:7860`。网页端和 Gradio 端共享同一个
`evaluate_all` 评估后端；网页端额外提供拖拽上传、模型缓存状态、评分卡、
警告列表以及 `summary.csv`、逐帧 CSV、原始 JSON 下载。

如果 Windows PowerShell 的执行策略禁止运行 `.ps1`，可以直接双击或运行：

```powershell
.\run-web.cmd
```

也可以只对当前 PowerShell 进程临时放行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run-web.ps1
```

网页端的模型状态会显示为：

- `READY`：当前环境已安装依赖并找到本地权重。
- `OPTIONAL`：可选后端或未配置模型，不会伪造评分。
- `OFFLINE`：当前指标无法使用，结果会显示 `partial` 或 `unavailable`。

当前首版已接入或可识别的模型后端包括 ArcFace、CLIP ViT-B/32、LPIPS、
MANIQA、MUSIQ、MediaPipe Face Mesh 和 VBench。VideoScore、ViCLIP、
Video-Bench 和在线 VLM Judge 暂不伪造为已完成模型，后续应通过独立适配器接入。

### 可选精确后端

基础版本不依赖重模型也可以运行，但会明确标注代理后端。若需要 ArcFace、
MediaPipe Face Mesh、MANIQA/MUSIQ，可在项目虚拟环境中安装：

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

可选模型首次运行可能需要下载权重。当前环境如果未安装 VideoCLIP/ViCLIP，
表情模块会使用人脸/画面运动轨迹代理，不会把代理结果标记成 VideoCLIP。
当前项目增加了本地 OpenAI CLIP ViT-B/32 逐帧文本-视频对齐基线；它用于
Prompt 条件检查，但仍不是 VideoCLIP/ViCLIP。

下载 ArcFace、MANIQA 和 MUSIQ 资源到项目缓存：

```powershell
.\download-optional-assets.ps1 -SkipPythonPackages
```

资源会放在 `model_cache/`，清单和 SHA256 在
`model_cache/OPTIONAL_ASSETS.json`。项目启动脚本默认把 ArcFace 和 IQA
推理固定到 CPU，避免 8GB 显存机器在首次加载时直接占满显存；需要时可手动
将 `EVALUATOR_FACE_DEVICE` 或 `EVALUATOR_IQA_DEVICE` 改为 `cuda`。

## 评估逻辑

- 结果视频：必填。
- 文本 Prompt：可选；填写后会计算文本-视频语义对齐，并参与第 3 类综合分数。
- GT 视频：只有 GT 存在时，第 2 类使用 PSNR、SSIM、LPIPS；同时可作为身份、表情、光流参考的后备素材。
- 参考图：优先作为身份基准，也可用于无 GT 高频细节参考，但不计算 PSNR、SSIM、LPIPS。
- 参考视频：优先作为表情、动作和时间稳定性参考，但不等同于 GT。
- 没有参考表情视频时，表情使用人工 1~5 分。
- 美学始终保留人工 1~5 分入口，符合截图中“暂无成熟自动方法”的说明。

不同类别使用不同的输入优先级：身份为参考图 > 参考视频 > GT，表情和时间稳定性为
参考视频 > GT > 结果自身对齐，第 2 类严格要求 GT。结果中会保留
每个类别的 `status`、`backend`、`note` 和 `warnings`，因此精确模型缺失时不会静默
伪造指标。

## 主流方法对应关系

首版采用可落地的分层方案，而不是用一个模型替代所有评价：

- 有 GT 的全参考质量：PSNR、SSIM、LPIPS，适用于有逐帧对应真实视频的场景。
- 无 GT 的感知质量：MANIQA、MUSIQ（可选）和高频细节代理。
- 文本-视频对齐：首版使用本地 CLIP 逐帧基线；后续可接入 VideoCLIP、ViCLIP、VLM 评分器。
- 多维度标准基准：VBench / VBench++ / VBench-2.0 作为独立可选后端。
- 组合关系和细粒度语义：后续可接入 T2V-CompBench、ETVA 等专用评估。
- 整体人类偏好：后续可接入 VideoScore 或 Video-Bench 类视频评分模型。

因此，首版输出的加权总分用于内部横向比较，不应被解释为跨数据集、跨分辨率、
跨编码格式的绝对质量标准。

## 输出

- 五类汇总：类别、权重、状态、核心结果、实际后端。
- 评估模式：`full_reference`、`reference_material`、`prompt_only` 或 `result_only`。
- 加权分数：显示标准化分数和参与权重；含代理后端时状态为 `partial`。
- 逐帧明细：有 GT 时显示 PSNR、SSIM、LPIPS；同时显示身份相似度、表情运动相似度、warping error 等。
- CSV：保存到 `outputs/holistic_metrics_*.csv`。
- 原始 JSON：保存完整输入来源、指标、后端、警告和评估计划。

## 测试

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖有 GT 和无 GT 两条主路径，并校验第 2 类指标的 GT 使用边界。
VBench 仍保留为独立可选页，不替代五类主流程。

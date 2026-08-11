# 取证初始分支

本目录实现两路相互独立的证据分支：

1. `facial_motion.py`
   - 读取 AU 与 Face Mesh CSV。
   - 提取 AU 动态、关键点分组运动、共激活、相位滞后、加速度与 jerk。
   - 新画像在可用时使用 `frame_time_in_ms`，导数按真实秒计算，而不是按 CSV 行号。
   - 旧版 LibreFace 导出若列名是毫秒、数值却是秒，会自动识别（与 `au_compliance` 相同启发式），保证网页 AU 缓存与 `data/au/MD_CL` 使用同一物理时间尺度。
   - 训练无关先验（AU 共激活 / 动态节律）会轻度增强分支证据，无需重建画像。
   - Face Mesh 覆盖不足时，打分回退为仅 AU 特征，不会把缺失关键点当成 0。
   - 可构建真拍域，或真拍 vs Seedance 双域画像。

2. `texture_detail.py`
   - 读取 RGB 帧或视频路径。
   - 度量局部高频细节、Laplacian、梯度统计、DCT 高频能量与时序 warp 残差。
   - 训练无关光流残差线索（均匀性 / 二阶微时序自然度）会轻度增强纹理证据，无需重建画像。
   - 可构建真拍域，或真拍 vs Seedance 双域画像。

`report.py` 保持两路证据分离，仅在校准后的「真拍概率」层面融合。

## 最小用法

```python
from evaluator.modules.forensics import (
    analyze_forensics,
    build_two_domain_facial_motion_profile,
    extract_facial_motion_features,
    extract_texture_detail_features,
)

motion_profile = build_two_domain_facial_motion_profile(
    real_csv_paths=["real_01.csv", "real_02.csv"],
    seedance_csv_paths=["seedance_01.csv", "seedance_02.csv"],
)
motion_result = analyze_forensics(
    facial_motion="candidate_au.csv",
    facial_motion_profile=motion_profile,
)

texture_features = extract_texture_detail_features("candidate.mp4")
report = analyze_forensics(
    facial_motion="candidate_au.csv",
    facial_motion_profile=motion_profile,
    texture_detail=texture_features,
)
```

当前实现是特征与画像基线，**不是**通用 Seedance 检测器。可靠真伪判定需要匹配的 hold-out 真拍与 Seedance 视频，并记录生成版本、分辨率、编码与生成模式。

## 构建画像

画像构建可与另一次 AU 提取任务并行：

```powershell
.\.venv\Scripts\python.exe scripts\build_forensics_profiles.py `
  --real-au-root data\au\MD_CL `
  --seedance-au-root outputs\au_cache\wangxing_seedance_expression_v1 `
  --output outputs\forensics\forensics_profiles.json
```

若两边视频根目录都可用，可再加：

```powershell
  --real-video-root data\MD_CL `
  --seedance-video-root data\WangXing_Seedance `
  --max-videos 120
```

纹理画像是可选的，成本高于 CSV 画像。构建器会记录无法读取的文件，而不是静默丢弃。

## 评估候选样本

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_forensics.py `
  --profile outputs\forensics\forensics_profiles.json `
  --au-csv data\au\WangXing_Seedance\candidate.csv `
  --video data\WangXing_Seedance\candidate.mp4 `
  --output outputs\forensics\candidate_report.json
```

报告中 `facial_motion` 与 `texture_detail` 证据保持分离。双域画像只产生 `raw_real_domain_evidence_0_1`，这**不是**概率。在提供「按源视频 / 生成批次 hold-out」的概率校准器之前，`real_capture_likelihood_0_1` 保持为空；没有校准器时，来源判定始终为 `uncertain`。

窗口级结果保留原始 `window_evidence` 列表，并增加 `window_summaries`（均值、最差窗口、均值+最差），便于定位可疑片段。

画像训练完成后构建校准器：

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_forensics.py `
  --profile outputs\forensics\forensics_profiles.json `
  --holdout-manifest data\forensics\holdout_split.json `
  --output outputs\forensics\forensics_authenticity_calibrator.json `
  --update-profile outputs\forensics\forensics_profiles.json
```

holdout 清单保证源视频不进入画像训练，并在校准时使用同一批 AU/视频配对样本。当前数据集专用校准器只依赖真拍/生成域标签，不要求 Seedance 版本、生成模式、输入类型或编码元数据。hold-out 样本数低于配置下限时写入 `provisional`；运行时会忽略 provisional 校准器。

校准报告包含 ROC AUC、Brier 分数与期望校准误差。这些指标描述的是声明的 holdout 划分，不是跨引擎泛化能力。

打分字段：

```text
facial_expression_muscle_score_0_1
texture_detail_score_0_1
raw_real_domain_evidence_0_1
real_capture_likelihood_0_1
```

前三项是画像证据。`real_capture_likelihood_0_1` 是校准后的概率，在 hold-out 校准器就绪前保持为空。这些字段**不表示**情绪分类准确率、提示词正确性、MANIQA/MUSIQ 画质，或普通五项里的表情/纹理分。

`evaluate_all` 也可挂载同一结果，且不改变普通五项总分：

```python
result = evaluate_all(
    result_path="candidate.mp4",
    ground_truth=None,
    reference_image=None,
    reference_video=None,
    forensics_profile_path="outputs/forensics/forensics_profiles.json",
    forensics_au_path="data/au/WangXing_Seedance/candidate.csv",
)
```

协作包内也可直接使用已同步的画像：

```python
from evaluator.modules.core.paths import profile_path
from evaluator.modules.forensics import analyze_forensics

report = analyze_forensics(
    facial_motion="candidate_au.csv",
    facial_motion_profile=profile_path("forensics_profiles", required=True),
)
```

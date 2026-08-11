# 协作方可用训练资产清单

对方工程扁平覆盖 `detail_expression_metrics.py` + `modules/` 后，下列画像位于：

`modules/assets/profiles/`

| 资产 | 样本/作用 | 评估时是否使用 |
|------|-----------|----------------|
| `wangxing_expression_profile.json` | **648** 条表情 AU 画像 | **是**（表情专项主路径） |
| `forensics_profiles.json` | 真拍/生成纹理与肌肉 forensics + 内嵌校准器 | **是**（质感/表情肌肉证据） |
| `forensics_authenticity_calibrator.json` | 校准器副本 | 运行时用 forensics_profiles 内嵌块；文件一并交付 |
| `wangxing_source_profile.json` | 真拍 vs 生成域 | **是**（写入表情 details.`wangxing_source`） |
| `wangxing_identity_profile.json` | 王兴身份 | **是**（有视频路径且 InsightFace 可用时写入 details.`wangxing_identity`；失败不阻断） |
| `wangxing_au_profile.json` | AU 合规离线画像 | 交付；主 Web 评估未强制打分 |
| `original_emotion_au_profile.json` | 情绪 AU 参考 | 交付；离线/专项用 |
| `holdout_split.json` | hold-out 划分 | 交付；校准脚本用 |
| `model_profile.json` | 硬件/模型策略 | 交付；holistic 路径用 |

## AU 输入优先级

1. 旁路 `视频.csv` / `视频_au.csv`（最高质量）
2. 从生成视频 **自动合成 AU**（MediaPipe FaceLandmarker / Face Mesh）→ 仍对照 **648** 样本 profile
3. 仅当合成失败：回退 `Expression/` 约 116 张动作原型图

## 模型文件（需随对方工程或首次下载）

| 文件 | 说明 |
|------|------|
| `modules/assets/models/face_landmarker.task` | AU 合成用；缺失时首次联网下载到该路径或系统临时目录 |
| `vedio_pred/models/*.pt` | 真伪检测；不在 modules zip 内，需保留对方工程原有目录 |
| `checkpoints/Qwen*` / CLIP | 文本/QA；大模型未发则该项跳过，不影响表情 profile |

## 交付命令

```powershell
python scripts/pack_evaluator_bundle.py --flat-host "对方工程"
```

只覆盖入口与 `modules/`，不覆盖对方 `main.py` / `Expression` / `checkpoints` / `input`。

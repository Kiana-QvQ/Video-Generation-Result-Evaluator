# Evaluator package（协作交付）

面向合作方的可独立使用目录。网页 `web/` / `web_app.py` **不是**本包依赖。

## 目录

```text
evaluator/
├── __init__.py
├── detail_expression_metrics.py   # 黄框两项：以合作方文件为准
├── README.md
├── paths.py
├── assets/
│   ├── MANIFEST.json
│   └── profiles/                  # 已打包的画像 / 校准器
├── forensics/
├── wangxing/
├── core/
└── backends/
```

## 黄框两项

以 `detail_expression_metrics.py` 为唯一公开入口：

```python
from evaluator.detail_expression_metrics import (
    compute_detail_metric,
    compute_face_expression_metric,
)
```

## Profile 解析

路径工具会按候选依次解析：已存在的绝对路径 → 已存在的仓库相对路径（如 `data/...`）→ `evaluator/assets/profiles/<文件名>` → 其它回退：

```python
from evaluator.core.paths import profile_path, resolve_profile, verify_bundled_profiles

expression = profile_path("wangxing_expression_profile", required=True)
print(verify_bundled_profiles())
```

已同步进 `assets/profiles/` 的文件：

| 文件 | 用途 |
|------|------|
| `wangxing_expression_profile.json` | 表情画像 |
| `wangxing_identity_profile.json` | 身份画像 |
| `wangxing_source_profile.json` | 来源画像 |
| `wangxing_au_profile.json` | AU 画像 |
| `forensics_profiles.json` | 真伪双域画像 + 内嵌运行时校准器 |
| `forensics_authenticity_calibrator.json` | 运行时校准器 payload（与上者内嵌字段一致；不是完整校准报告） |
| `holdout_split.json` | hold-out 清单 |
| `model_profile.json` | 模型推荐配置 |

说明：推理时读 `forensics_profiles.json` 里的 `authenticity_calibrator` 即可；独立校准器文件只放 Platt 参数，便于核对，不含 holdout 样本明细。

刷新包内副本：

```powershell
python scripts/pack_evaluator_bundle.py --sync-only
```

打完整协作包（含 evaluator + assets）：

```powershell
python scripts/pack_evaluator_bundle.py --output outputs/evaluator_bundle.zip
```

## 边界

| 模块 | 职责 |
|------|------|
| `detail_expression_metrics` | 黄框内容质量（合作方） |
| `wangxing/` | 王兴身份/表情 AU 专项 |
| `forensics/` | 真拍 vs AI |
| `core/` | 五项评估与视频 IO |

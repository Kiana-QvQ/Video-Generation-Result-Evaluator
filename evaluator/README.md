# Evaluator 协作包

发给对方时，**整份 `evaluator/` 文件夹**即可。解压后顶层应是：

```text
evaluator/
├── __init__.py
├── detail_expression_metrics.py   # 公开入口（黄框两项）
├── README.md
└── modules/                       # 合并实现目录
    ├── assets/                    # 画像与校准器
    ├── core/
    ├── forensics/
    └── wangxing/
```

不要打散文件；不要只拷 `detail_expression_metrics.py`。  
`__pycache__` 可不发送。仓库根的 `backends/`、`web/` **不属于**本包。

## 使用（相对路径）

把 **`evaluator` 的上一级目录** 加入 `PYTHONPATH`：

```python
from evaluator.detail_expression_metrics import (
    prepare_generated_video,
    compute_detail_metric,
    compute_face_expression_metric,
)

video = prepare_generated_video(
    r"candidate.mp4",
    sample_fps=8,
    max_frames=24,
    au_csv=r"candidate.csv",  # 表情肌肉完整链路建议提供
)
detail = compute_detail_metric(video, None, None, 24)
expression = compute_face_expression_metric(video, None, None, 24)
print(detail.name, detail.score, detail.status)
print(expression.name, expression.score, expression.status)
```

画像从包内 `modules/assets/profiles/` 用相对路径解析
（`evaluator.modules.core.paths.profile_path`），不依赖本机盘符绝对路径。

## 打 zip

```powershell
python scripts/pack_evaluator_bundle.py --output outputs/evaluator.zip
```

解压得到名为 `evaluator` 的文件夹，内容即上表结构。

刷新画像副本：

```powershell
python scripts/pack_evaluator_bundle.py --sync-only
```

## 模块职责

| 路径 | 职责 |
|------|------|
| `detail_expression_metrics.py` | 黄框公开入口 |
| `modules/wangxing/` | 王兴身份与表情 AU 专项 |
| `modules/forensics/` | 真拍 vs AI 证据分支 |
| `modules/core/` | 路径、视频采样与专项 runtime |
| `modules/assets/profiles/` | 打包画像与校准器 |

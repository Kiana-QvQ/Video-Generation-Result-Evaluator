# Evaluator 协作包

发给对方时有两种形态：

### A. 独立包（推荐新接入）

**整份 `evaluator/` 文件夹**。解压后顶层应是：

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

### B. 对方现有宿主工程（扁平覆盖）

对方工程已是「根目录 `detail_expression_metrics.py` + `modules/`」时，执行：

```powershell
python scripts/data_tools/pack_evaluator_bundle.py --flat-host "对方工程"
```

或覆盖到 `Evaluator/`（若该目录即对方完整工程根）：

```powershell
python scripts/data_tools/pack_evaluator_bundle.py --flat-host "Evaluator"
```

只覆盖 `detail_expression_metrics.py` 与 `modules/`，**不改**对方的
`main.py` / `app.py` / `Expression` / `BiaoQing` / `checkpoints` / `input`。

> Windows 注意：`Evaluator` 与 `evaluator` 在同一盘符下是**同一个目录**（大小写不敏感）。
> 对方完整工程请放在其它名字下（例如本仓库的 `对方工程/`），不要与协作包
> `evaluator/` 重名，否则扁平覆盖会误删包内 `modules/`。

无 AU CSV 时，表情专项会自动用同级 `Expression/` 动作原型匹配，避免 UI
假 `0.0%` 与五项 `N/A`。

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
python scripts/data_tools/pack_evaluator_bundle.py --output outputs/evaluator.zip
```

刷新画像副本：

```powershell
python scripts/data_tools/pack_evaluator_bundle.py --sync-only
```

## 模块职责

| 路径 | 职责 |
|------|------|
| `detail_expression_metrics.py` | 黄框公开入口 |
| `modules/wangxing/` | 王兴身份与表情 AU 专项 |
| `modules/forensics/` | 真拍 vs AI 证据分支 |
| `modules/core/` | 路径、视频采样、Face Landmarker、Expression 回退 |
| `modules/assets/profiles/` | 打包画像与校准器 |

## ѵ���ʲ��Ƿ�ᱻ�Է�����

��� `modules/assets/ASSET_USAGE.md`��������·���� 648 ���� profile������· AU ���Զ��ϳ� AU�����/��Դ����д�� details��


# Evaluator 项目说明

当前目录是完整宿主工程，直接运行本目录的 `main.py`、`app.py` 或
`server.py`，不需要额外发送或加载 `evaluator111`。

Evaluator 已接入：

- `detail_expression_metrics.py`
- `modules/`
- `Expression/` 动作原型回退
- `modules/assets/profiles/` 画像与校准文件

无 AU CSV 时，表情专项会自动使用同级 `Expression/` 动作原型匹配，
避免 UI 显示假 `0.0%` 与五项 `N/A`。完整王兴 AU 专项仍建议在视频旁
提供同名 `.csv` 或 `_au.csv`。

`BiaoQing/` 是原始高清素材；当前运行主要使用压缩后的 `Expression/`。
`input/`、`uploads/` 和 `results/` 属于可替换的运行时目录。

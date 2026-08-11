# 扁平接入说明（对方宿主工程）

已覆盖：`detail_expression_metrics.py` + `modules/`
未改：`main.py` / `app.py` / `Expression` / `checkpoints` / `input`

无 AU CSV 时自动用同级 Expression/ 动作原型匹配，避免假 0.0% 与五项 N/A。
完整王兴 AU 专项仍建议视频旁放 同名.csv / 同名_au.csv。

Windows：不要把宿主工程命名为 Evaluator，若与协作包 evaluator 同盘会冲突。

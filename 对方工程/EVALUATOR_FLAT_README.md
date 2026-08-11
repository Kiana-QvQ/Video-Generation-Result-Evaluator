# 扁平接入说明（对方宿主工程）

已覆盖到对方项目根目录：

- `detail_expression_metrics.py`
- `modules/`

不要改对方的 `main.py` / `app.py` / `Expression` / `checkpoints`。

无 AU CSV 时：优先从视频自动合成 AU，对照王兴表情 profile（约 648 样本）；
仅合成失败才回退 Expression/ 动作原型图。

详见 modules/assets/ASSET_USAGE.md。

# 王兴专项（项目侧）

在**不改对方 `evaluator` 宿主代码**的前提下，扩展王兴真伪专项：

- AU 多技术学习头（source + 运动域 + SSL + 生理 + 质量门）
- 双尺度视频 `.pt` 支路
- 加权融合硬判（默认 AU 0.65 / `.pt` 0.35）

## 命令

```powershell
# 模型槽位（预留，不下载）
.\.venv\Scripts\python.exe scripts\run_wangxing_specialization_fused.py slots

# 单条（可 --no-pt 只跑 AU）
.\.venv\Scripts\python.exe scripts\run_wangxing_specialization_fused.py score-one `
  --au data\au\MD_CL\...\xxx.csv `
  --video data\MD_CL\...\xxx.mp4

# 留出评估（.pt 有缓存会复用 outputs/vedio_pred/cache）
.\.venv\Scripts\python.exe scripts\run_wangxing_specialization_fused.py evaluate
```

`.pt` 路径：`outputs/vedio_pred/models/wangxing_dual_scale_classifier.pt`

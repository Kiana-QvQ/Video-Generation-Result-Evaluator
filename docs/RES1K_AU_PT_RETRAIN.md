# AU+.pt 中期融合重训（test/AI 五条只测不训，长边 >= 1024）

网页 forensics 默认文件不覆盖。不要跑 `prepare_change_seedance_training.py`
（那套会把 Change 写进训练集）。

**主路径**：把 AU 25 维证据与双尺度视频特征 **拼接后进同一个 MLP**（early concat），
不再用 0.65/0.35 事后加权。产物路径全部带 `joint_au_pt`，不覆盖旧 dual / noleak。

在仓库根目录、`.venv`：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_res1k_au_pt_training.py
```

若 1k 降采样文件已存在，prepare 会跳过 ffmpeg，只补写 `pairs`（video↔AU）。

## 1) 联合 AU+视频 .pt（新路径）

```powershell
.\.venv\Scripts\python.exe scripts\train_wangxing_joint_au_pt.py train `
  --manifest outputs\vedio_pred\wangxing_dual_pt_split_res1k.json `
  --cache-dir outputs\vedio_pred\cache_joint_au_pt_res1k `
  --model-path outputs\vedio_pred\models\wangxing_joint_au_pt_res1k.pt `
  --metrics-output outputs\vedio_pred\wangxing_joint_au_pt_holdout_metrics_res1k.json `
  --source-profile outputs\forensics\wangxing_source_profile_holdout_excluded.json `
  --forensics-profile outputs\forensics\forensics_profiles.json `
  --epochs 80 --batch-size 16 --learning-rate 3e-4 --seed 42 `
  --device cuda
```

特征抽取（24f@1024 + 8f@2048 + AU 25 维）在 CPU；只有 MLP 训练上 GPU。
默认 `--device cuda`，若本机没有 CUDA 再改 `--device cpu`。

## 2) 评估：官方 holdout（防回退）

```powershell
.\.venv\Scripts\python.exe scripts\train_wangxing_joint_au_pt.py evaluate `
  --holdout-manifest data\forensics\holdout_split.json `
  --model-path outputs\vedio_pred\models\wangxing_joint_au_pt_res1k.pt `
  --source-profile outputs\forensics\wangxing_source_profile_holdout_excluded.json `
  --forensics-profile outputs\forensics\forensics_profiles.json `
  --output outputs\forensics\wangxing_joint_au_pt_official_holdout_metrics.json
```

## 3) 评估：test/AI 五条（只测）

```powershell
.\.venv\Scripts\python.exe scripts\train_wangxing_joint_au_pt.py evaluate `
  --holdout-manifest data\forensics\holdout_test_AI.json `
  --model-path outputs\vedio_pred\models\wangxing_joint_au_pt_res1k.pt `
  --source-profile outputs\forensics\wangxing_source_profile_holdout_excluded.json `
  --forensics-profile outputs\forensics\forensics_profiles.json `
  --output outputs\forensics\wangxing_joint_au_pt_test_AI_metrics.json
```

## （可选）仅视频 dual .pt，不含 AU

若还想单独训纯视频 res1k dual（对比基线）：

```powershell
.\.venv\Scripts\python.exe scripts\train_wangxing_video_pt.py train `
  --manifest outputs\vedio_pred\wangxing_dual_pt_split_res1k.json `
  --cache-dir outputs\vedio_pred\cache_res1k `
  --model-path outputs\vedio_pred\models\wangxing_dual_scale_classifier_res1k.pt `
  --metrics-output outputs\vedio_pred\wangxing_dual_pt_holdout_metrics_res1k.json `
  --epochs 80 --batch-size 16 --learning-rate 3e-4 --seed 42
```

说明：
- 训练假/真的 1k 副本在 `data/_aug/seedance_le1024/` 与 `data/_aug/mdcl_le1024/`
- `*_le1024.mp4` 复用原片 AU（去掉 `_le1024` 后按 stem 匹配）
- 评估 Change 用原生 720p，不要升采样后再测
- ImissU 无 MediaPipe landmark，AU 支路可能偏弱
- 未改默认 dual `.pt` / noleak 头 / `forensics_profiles.json`；网页暂不接线

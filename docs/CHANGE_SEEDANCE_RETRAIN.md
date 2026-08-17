# Change Seedance 增强重训命令（默认产物不覆盖，长边 >= 1024）

在仓库根目录、使用 `.venv`。先跑准备（默认把 Change 与高分辨率 Seedance 重采样到 1k）：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_change_seedance_training.py
# 仅跳过“高分辨率 Seedance→1k”批量增强时（Change→1k 仍会做）：
# .\.venv\Scripts\python.exe scripts\prepare_change_seedance_training.py --skip-downsample
```

## 1) Source 画像（排除官方 holdout + Change eval）

```powershell
.\.venv\Scripts\python.exe scripts\train_wangxing_specialization.py --skip-identity `
  --seedance-label-manifest data\au\WangXing_Seedance\pseudo_expression_manifest_change_aug.json `
  --holdout-manifest data\forensics\holdout_split_plus_change_eval.json `
  --source-profile-output outputs\forensics\wangxing_source_profile_change_aug_noleak.json `
  --expression-output outputs\forensics\wangxing_expression_profile_change_aug_tmp.json
```

## 2) AU 学习头（注入 Change train AU，输出新 JSON）

```powershell
.\.venv\Scripts\python.exe scripts\train_learned_fusion_head.py train `
  --forensics-profile outputs\forensics\forensics_profiles_quality_filtered.json `
  --source-profile outputs\forensics\wangxing_source_profile_change_aug_noleak.json `
  --holdout-manifest data\forensics\holdout_split_plus_change_eval.json `
  --extra-generated-au-manifest data\forensics\change_seedance_protocol.json `
  --model-type logistic `
  --seed 42 `
  --output outputs\forensics\learned_fusion_head_logistic_change_aug.json
```

## 3) 双尺度 `.pt`（新模型路径，不覆盖旧 dual）

```powershell
.\.venv\Scripts\python.exe scripts\train_wangxing_video_pt.py train `
  --manifest outputs\vedio_pred\wangxing_dual_pt_split_change_aug.json `
  --cache-dir outputs\vedio_pred\cache_change_aug `
  --model-path outputs\vedio_pred\models\wangxing_dual_scale_classifier_change_aug.pt `
  --metrics-output outputs\vedio_pred\wangxing_dual_pt_holdout_metrics_change_aug.json `
  --epochs 80 --batch-size 16 --learning-rate 3e-4 --seed 42
```

## 4) 评估：官方 holdout（防回退）

```powershell
.\.venv\Scripts\python.exe scripts\train_learned_fusion_head.py evaluate `
  --forensics-profile outputs\forensics\forensics_profiles_quality_filtered.json `
  --source-profile outputs\forensics\wangxing_source_profile_change_aug_noleak.json `
  --holdout-manifest data\forensics\holdout_split.json `
  --head outputs\forensics\learned_fusion_head_logistic_change_aug.json `
  --output outputs\forensics\learned_fusion_holdout_metrics_change_aug.json

.\.venv\Scripts\python.exe scripts\run_wangxing_specialization_fused.py evaluate `
  --holdout-manifest data\forensics\holdout_split.json `
  --source-profile outputs\forensics\wangxing_source_profile_change_aug_noleak.json `
  --learned-head outputs\forensics\learned_fusion_head_logistic_change_aug.json `
  --pt-model outputs\vedio_pred\models\wangxing_dual_scale_classifier_change_aug.pt `
  --pt-cache-dir outputs\vedio_pred\cache_change_aug `
  --output outputs\forensics\wangxing_specialization_fused_holdout_metrics_change_aug.json
```

## 5) 评估：Change OOD

```powershell
.\.venv\Scripts\python.exe scripts\train_learned_fusion_head.py evaluate `
  --forensics-profile outputs\forensics\forensics_profiles_quality_filtered.json `
  --source-profile outputs\forensics\wangxing_source_profile_change_aug_noleak.json `
  --holdout-manifest data\forensics\holdout_change_eval.json `
  --head outputs\forensics\learned_fusion_head_logistic_change_aug.json `
  --output outputs\forensics\learned_fusion_change_ood_metrics.json

.\.venv\Scripts\python.exe scripts\run_wangxing_specialization_fused.py evaluate `
  --holdout-manifest data\forensics\holdout_change_eval.json `
  --source-profile outputs\forensics\wangxing_source_profile_change_aug_noleak.json `
  --learned-head outputs\forensics\learned_fusion_head_logistic_change_aug.json `
  --pt-model outputs\vedio_pred\models\wangxing_dual_scale_classifier_change_aug.pt `
  --pt-cache-dir outputs\vedio_pred\cache_change_aug `
  --quality-gate `
  --output outputs\forensics\wangxing_specialization_fused_change_ood_metrics.json
```

说明：
- 未换输出路径时，原 MD_CL→Seedance 默认逻辑与旧模型文件不变。
- Change 原生 720p 仅作 AU；`.pt` 训练用 `data/_aug/change_le1024/` 的 1k 版。
- ImissU 无 MediaPipe landmark，Change OOD 上可能仍偏弱。

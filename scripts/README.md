# scripts

脚本按用途分类。常用命令仍写在各类文档里；这里只作索引。

可直接运行的 Python 脚本留在本目录，因为测试和脚本之间用 `from scripts.xxx import`。
一次性工具在 `tools/`，过期排查脚本在 `archive/`。

## 测试集

| 脚本 | 作用 |
|---|---|
| `export_human_review_reference_set.py` | 导出带参考实验到 `data/test/with_reference` |
| `build_independent_25x25_test_sets.py` | 构建 `single_video` 25+25 |
| `build_web_forensics_v2_dataset.py` | 构建 web 鉴伪测试集 |
| `evaluate_single_video_forensics_dataset.py` | 跑单视频鉴伪评估 |
| `run_web_forensics_v2.py` | 跑 web 鉴伪 v2 |
| `create_forensics_holdout_split.py` | 划分鉴伪 holdout |

## AU

| 脚本 | 作用 |
|---|---|
| `extract_libreface_au.py` | 抽 AU |
| `libreface_worker.py` | AU 抽取子进程 |
| `build_au_profile.py` | 建 AU profile |
| `build_original_emotion_au_profile.py` | 建原始情绪 AU profile |
| `evaluate_au_compliance.py` | AU 符合度评估 |
| `validate_au_models.py` | 校验 AU 模型 |
| `fit_au_leakage_classifier.py` | AU 泄漏分类器 |
| `retrain_au_from_csv.py` | 从 CSV 重训 AU |
| `run_au_training_pipeline.py` | AU 训练流水线 |
| `run_au_training_pipeline.ps1` | 上面的 PowerShell 入口 |
| `train_au_ssl_backbone.py` | AU SSL 骨干 |
| `download_ravdess_negative.py` | 下载 RAVDESS 负例 |

## 鉴伪

| 脚本 | 作用 |
|---|---|
| `build_forensics_manifest.py` | 鉴伪清单 |
| `build_forensics_profiles.py` | 鉴伪 profile |
| `build_joint_forensics_manifest.py` | 联合鉴伪清单 |
| `calibrate_forensics.py` | 校准 |
| `calibrate_pseudo_labels.py` | 伪标签校准 |
| `evaluate_forensics.py` | 鉴伪评估 |
| `evaluate_joint_forensics.py` | 联合鉴伪评估 |
| `evaluate_holdout_detection_metrics.py` | holdout 指标 |
| `train_joint_forensics.py` | 训练联合鉴伪 |
| `validate_forensics_profiles.py` | 校验 profile |
| `validate_texture_detail_profile.py` | 纹理细节 profile |
| `run_perturbation_robustness.py` | 扰动鲁棒性 |
| `run_hard_detection_pipeline.py` | 硬判流水线 |
| `train_learned_fusion_head.py` | 学习融合头 |

## 望星

| 脚本 | 作用 |
|---|---|
| `prepare_wangxing_v3_generalization.py` | v3 泛化数据 |
| `train_wangxing_v3_generalization.py` | 训练 v3 |
| `run_wangxing_v3_pipeline.py` | v3 流水线 |
| `run_wangxing_v3_pipeline.cmd` | v3 入口 |
| `train_wangxing_joint_au_pt.py` | AU+PT v1 |
| `train_wangxing_joint_au_pt_v2.py` | AU+PT v2 |
| `prepare_res1k_au_pt_training.py` | Res1k AU+PT 数据 |
| `train_wangxing_video_pt.py` | 视频 PT |
| `train_wangxing_multi_scale_pt.py` | 多尺度 PT |
| `train_wangxing_specialization.py` | 专项训练 |
| `evaluate_wangxing_specialization.py` | 专项评估 |
| `validate_wangxing_specialization.py` | 专项校验 |
| `run_wangxing_specialization_fused.py` | 融合评估 |
| `run_wangxing_specialization_training.ps1` | 专项训练入口 |
| `run_wangxing_specialization_training.cmd` | 专项训练入口 |
| `run_wangxing_frame_ablation.py` | 帧消融 |
| `evaluate_wangxing_source_detection_metrics.py` | 源检测指标 |
| `label_seedance_expressions.py` | Seedance 表情标注 |
| `run_seedance_expression_labeling.cmd` | 标注入口 |
| `build_expression_reference_manifest.py` | 表情参考清单 |
| `build_wangxing_video_manifest.ps1` | 望星视频清单 |

## 生成质量

| 脚本 | 作用 |
|---|---|
| `evaluate_generated_video.py` | 单条生成视频评估 |
| `run_five_prompt_tests.py` | 五条 prompt 测试 |
| `run_reference_pair_queue_tests.py` | 带参考成对队列测试 |

## 打包

| 脚本 | 作用 |
|---|---|
| `pack_evaluator_bundle.py` | 打包评估器 |

## tools/

下载和本地服务，不参与训练主流程。

| 脚本 | 作用 |
|---|---|
| `tools/download-optional-assets.ps1` | 下载可选模型 |
| `tools/download-vbench-models.ps1` | 下载 VBench |
| `tools/download-vlm-judge.ps1` | 下载 VLM Judge |
| `tools/run-vlm-judge-docker.ps1` | Docker 启动 Judge |
| `tools/run-vlm-judge-local.py` | 本地启动 Judge |
| `tools/sync_forensics_bundle.ps1` | 同步鉴伪资源包 |

## archive/

一次性排查或已有替代流程的脚本，默认不要再当主入口。

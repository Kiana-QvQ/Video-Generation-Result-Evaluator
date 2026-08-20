# 脚本目录说明

当前脚本按英文功能子目录组织。项目内部已同步更新模块导入和固定路径，
避免 OpenCV、FFmpeg、LibreFace 对中文路径的兼容性问题。

## 当前常用入口

### web_forensics

- `web_forensics/evaluate_single_video_forensics_dataset.py`：网页卡片式单视频评估；
- `web_forensics/run_web_forensics_v2.py`：网页 v2 融合流程；
- `web_forensics/web_authenticity_policy.py`：困难开发集真实性策略；
- `data_build/build_web_forensics_v2_dataset.py`：构建网页测试集；
- `data_build/build_independent_25x25_test_sets.py`：构建 25+25 测试集；
- `data_build/build_wangxing_32x32_final_test.py`：构建新的 32+32 最终测试集；
- `data_tools/export_human_review_reference_set.py`：导出真实带参考人审任务。

### PT 训练

- `pt_training/run_wangxing_v3_pipeline.py`：v3 数据、训练和测试一键流程；
- `pt_training/run_wangxing_v4_pipeline.py`：v4 数据、训练和多测试集一键流程；
- `pt_training/run_wangxing_v3_pipeline.cmd`：v3 Windows 入口；
- `pt_training/prepare_wangxing_v3_generalization.py`：准备 v3 泛化数据；
- `pt_training/train_wangxing_v3_generalization.py`：训练和评估 v3；
- `pt_training/prepare_wangxing_v4_photometric.py`：准备光照/压缩增强；
- `pt_training/train_wangxing_v4_face.py`：训练和评估人脸几何分支 v4；
- `pt_training/train_wangxing_joint_au_pt.py`：AU+PT v1；
- `pt_training/train_wangxing_joint_au_pt_v2.py`：AU+PT v2；
- `pt_training/prepare_res1k_au_pt_training.py`：准备 Res1k AU+PT 数据。

### AU 和画像

- `au/extract_libreface_au.py`：提取 AU CSV；
- `au/libreface_worker.py`：LibreFace 子进程；
- `data_build/build_au_profile.py`：构建 AU profile；
- `data_build/build_original_emotion_au_profile.py`：构建原始情绪 profile；
- `au/evaluate_au_compliance.py`：AU 符合度评估；
- `au/validate_au_models.py`：校验 AU 模型；

### wangxing

- `wangxing/train_wangxing_specialization.py`：专项训练；
- `wangxing/evaluate_wangxing_specialization.py`：专项评估；
- `wangxing/validate_wangxing_specialization.py`：专项校验；
- `wangxing/run_wangxing_specialization_fused.py`：专项融合评估；
- `wangxing/label_seedance_expressions.py`：Seedance 表情标注。

## 分类目录

- `main_workflow`：当前命令入口说明；
- `helper_tools`：下载、打包和外部工具说明；
- `data_build`：数据集、profile 和 holdout 构建；
- `calibration_validation`：校准器和伪标签校准；
- `history`：旧实验和不再作为主入口的脚本；
- `archive`、`tools`：保留的历史兼容目录。

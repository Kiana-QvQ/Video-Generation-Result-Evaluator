# 真实/生成视频检测器

这个目录实现监督式真实/生成视频检测器，同时保留旧的真实视频 One-Class 模型作为后备。

## 数据目录

默认训练数据目录是：

```text
video_pred/
└── MD_CL/
    ├── CL_jingya01/
    ├── CL_kaixin01/
    └── ...
```

脚本会递归读取 `mp4`、`avi`、`mov`、`mkv`、`webm` 和 `flv` 文件。

## 王兴双尺度 .pt（推荐新流程）

协议：

- 特征：`24 帧 @ 1024` + `8 帧 @ 2048`，拼接后用**训练集**均值/方差归一化
- 训练真拍：从 `data/MD_CL` 按子目录均衡抽 **120**（排除 holdout）
- 训练生成：全部非 holdout Seedance
- 测试：`data/forensics/holdout_split.json`（与 AU 硬判同一 holdout）
- 产出：`outputs/vedio_pred/models/wangxing_dual_scale_classifier.pt`

在仓库根目录、建议 `.venv`：

```powershell
# 1) 只建划分清单（快）
.\.venv\Scripts\python.exe scripts\train_wangxing_video_pt.py build-split

# 2) 抽帧+归一化+训练+holdout 测试（慢：1k/2k 解码）
.\.venv\Scripts\python.exe scripts\train_wangxing_video_pt.py train

# 3) 单条预测
.\.venv\Scripts\python.exe scripts\train_wangxing_video_pt.py predict --video data\WangXing_Seedance\xxx.mp4
```

看结果：

- `outputs/vedio_pred/wangxing_dual_pt_split.json` — 训练/测试路径
- `outputs/vedio_pred/wangxing_dual_pt_holdout_metrics.json` — 生成召回 / 整体准确率
- `outputs/vedio_pred/models/wangxing_dual_scale_classifier.pt` — 模型

## 监督式训练（旧单尺度入口）

在 `E:\python\Evaluator` 目录执行：

```powershell
python video_pred\train_real_fake_detector.py
```

模型、特征缓存和训练产生的文件都保存在 `video` 目录：

```text
video_pred/
├── cache/
│   ├── real_features_f8_s48.npz
│   └── fake_features_f8_s48.npz
└── models/
    └── real_fake_video_classifier.pt
```

真实视频默认读取 `MD_CL`，假视频默认读取 `WangXing_Seedance`。训练时按真实视频目录和假视频文件名分组划分验证集，并自动平衡两类样本。

## 单独预测

```powershell
python video_pred\predict_real_video.py --video E:\python\Evaluator\input\generated.mp4
```

输出包括：

- `真实概率`
- `生成概率`
- 监督分类器 logit 和温度
- 验证集准确率、F1 和 ROC-AUC

概率采用平滑映射，固定保留 2% 到 98% 的不确定区间，不会输出绝对的 0% 或 100%。

## 重要说明

当前 `WangXing_Seedance` 已作为假视频监督训练集。新模型使用空间帧编码器、双向 GRU 时序编码和温度校准，输出概率比旧 One-Class 模型更接近真实/生成分类概率。`real_video_autoencoder.pt` 仍可用于没有假视频标签时的后备检测。

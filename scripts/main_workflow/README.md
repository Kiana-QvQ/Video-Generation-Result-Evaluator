# Current Main Workflow

当前主入口按英文目录组织，直接从项目根目录运行。

## web_forensics

```powershell
.\.venv\Scripts\python.exe scripts\web_forensics\evaluate_single_video_forensics_dataset.py
.\.venv\Scripts\python.exe scripts\web_forensics\run_web_forensics_v2.py all
```

## PT v3

```powershell
.\.venv\Scripts\python.exe scripts\pt_training\run_wangxing_v3_pipeline.py --device cuda
```

## AU

```powershell
.\.venv\Scripts\python.exe scripts\au\extract_libreface_au.py
```

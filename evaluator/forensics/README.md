# Forensics Initial Branches

This folder contains the first implementation of two independent evidence
branches:

1. `facial_motion.py`
   - Reads AU and Face Mesh CSV files.
   - Extracts AU dynamics, landmark-group motion, co-activation, phase lag,
     acceleration, and jerk.
   - Builds real-domain or real-versus-Seedance profiles.

2. `texture_detail.py`
   - Reads RGB video frames or a video path.
   - Measures local high-frequency detail, Laplacian detail, gradient
     statistics, DCT high-frequency energy, and temporal warp residuals.
   - Builds real-domain or real-versus-Seedance profiles.

`report.py` keeps both branches separate and only fuses calibrated
real-capture likelihoods.

## Minimal Usage

```python
from evaluator.forensics import (
    analyze_forensics,
    build_two_domain_facial_motion_profile,
    extract_facial_motion_features,
    extract_texture_detail_features,
)

motion_profile = build_two_domain_facial_motion_profile(
    real_csv_paths=["real_01.csv", "real_02.csv"],
    seedance_csv_paths=["seedance_01.csv", "seedance_02.csv"],
)
motion_result = analyze_forensics(
    facial_motion="candidate_au.csv",
    facial_motion_profile=motion_profile,
)

texture_features = extract_texture_detail_features("candidate.mp4")
report = analyze_forensics(
    facial_motion="candidate_au.csv",
    facial_motion_profile=motion_profile,
    texture_detail=texture_features,
)
```

The current implementation is a feature and profile baseline. It is not a
universal Seedance detector. A reliable authenticity decision requires
matched, held-out real and Seedance videos with the generation version,
resolution, codec, and generation mode recorded.

## Build Profiles

The profile builder can run while a separate AU extraction job is active:

```powershell
.\.venv\Scripts\python.exe scripts\build_forensics_profiles.py `
  --real-au-root data\au\MD_CL `
  --seedance-au-root outputs\au_cache\wangxing_seedance_expression_v1 `
  --output outputs\forensics\forensics_profiles.json
```

When both video roots are available, add:

```powershell
  --real-video-root data\MD_CL `
  --seedance-video-root data\WangXing_Seedance `
  --max-videos 120
```

The texture profile is optional because it is more expensive than the CSV
profile. The builder records unreadable files instead of silently dropping
them.

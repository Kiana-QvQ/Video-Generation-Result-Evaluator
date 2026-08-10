# Forensics Initial Branches

This folder contains the first implementation of two independent evidence
branches:

1. `facial_motion.py`
   - Reads AU and Face Mesh CSV files.
   - Extracts AU dynamics, landmark-group motion, co-activation, phase lag,
     acceleration, and jerk.
   - New profiles use `frame_time_in_ms` when available, so derivatives are
     measured in real seconds rather than CSV row units.
   - If Face Mesh coverage is insufficient, scoring falls back to AU-only
     features instead of treating missing landmarks as zero measurements.
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

## Evaluate a Candidate

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_forensics.py `
  --profile outputs\forensics\forensics_profiles.json `
  --au-csv data\au\WangXing_Seedance\candidate.csv `
  --video data\WangXing_Seedance\candidate.mp4 `
  --output outputs\forensics\candidate_report.json
```

The report keeps `facial_motion` and `texture_detail` evidence separate. A
two-domain profile produces only `raw_real_domain_evidence_0_1`; this is not a
probability. `real_capture_likelihood_0_1` remains null until a
source-video/generation-batch-held-out probability calibrator is supplied.
Without that calibrator, the source decision is always `uncertain`.
Window-level results retain the original `window_evidence` list and add
`window_summaries` with mean, worst-window, and mean-plus-worst evidence for
review localization.

Build a calibrator after the profile has been trained:

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_forensics.py `
  --profile outputs\forensics\forensics_profiles.json `
  --holdout-manifest data\forensics\holdout_split.json `
  --output outputs\forensics\forensics_authenticity_calibrator.json `
  --update-profile outputs\forensics\forensics_profiles.json
```

The holdout manifest keeps source videos out of profile training and uses the
same paired AU/video samples for calibration. The current dataset-specific
calibrator uses only real/generated domain labels and does not require
Seedance version, generation mode, input type, or codec metadata. It writes
`provisional` when fewer than the configured minimum number of held-out
samples exists; provisional calibrators are ignored at runtime.

The calibration report includes ROC AUC, Brier score, and expected calibration
error. These metrics describe the declared holdout split, not universal
cross-engine performance.

The scoring fields are:

```text
facial_expression_muscle_score_0_1
texture_detail_score_0_1
raw_real_domain_evidence_0_1
real_capture_likelihood_0_1
```

The first three fields are profile evidence. `real_capture_likelihood_0_1`
is a calibrated probability and remains null until the held-out calibrator is
ready. These fields do not mean emotion classification accuracy, prompt
correctness, MANIQA/MUSIQ image quality, or the ordinary five-category
expression/texture scores.

`evaluate_all` can also attach the same result without changing the ordinary
five-category total:

```python
result = evaluate_all(
    result_path="candidate.mp4",
    ground_truth=None,
    reference_image=None,
    reference_video=None,
    forensics_profile_path="outputs/forensics/forensics_profiles.json",
    forensics_au_path="data/au/WangXing_Seedance/candidate.csv",
)
```

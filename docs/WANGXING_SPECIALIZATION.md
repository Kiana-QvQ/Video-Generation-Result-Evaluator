# Wang Xing Specialization

The Wang Xing option is a separate two-stage evaluator. It does not change
the ordinary five category scores:

1. Open-set identity: video-level ArcFace prototypes compare real Wang Xing
   videos, Wang Xing Seedance videos, and non-Wang Xing videos. Each frame is
   quality-weighted, video consistency is measured, and a calibrated
   positive-vs-negative probability is used with an uncertain band.
2. Expression support domain: real `data/au/MD_CL` CSVs are summarized with
   AU intensity/presence, Face Mesh geometry, pose, velocity, acceleration, and
   facial event statistics.

Identity must pass before the expression conclusion is shown. The result can
be `wangxing`, `not_wangxing`, or `uncertain`; low-quality videos are not
forced into a binary decision.

## Seedance Content Labels

Seedance videos are used for both identity/source training and expression
training. Their filenames are not treated as labels. Run the automatic
content-labeling pass first:

```powershell
.\scripts\run_seedance_expression_labeling.cmd `
  --device cpu
```

The manifest is written to
`data/au/WangXing_Seedance/pseudo_expression_manifest.json`. Each record
contains the top two real Wang Xing expression domains, compatibility score,
margin, quality, and one of `high_confidence`, `ambiguous`, `low_confidence`,
or `unknown`. Only `high_confidence` records enter expression training.

## Long Training

The identity build can take longer than ten minutes on CPU. Run it manually
in the project root so progress stays visible:

```powershell
.\scripts\run_wangxing_specialization_training.ps1 `
  -Device cpu `
  -IdentityLimit 240 `
  -IdentityFrames 1
```

If PowerShell execution policy blocks `.ps1`, use the one-process bypass
command without changing the machine policy:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\run_wangxing_specialization_training.ps1 `
  -Device cpu -IdentityLimit 0 -IdentityFrames 1
```

Or use the CMD launcher, which does not depend on PowerShell script policy:

```powershell
.\scripts\run_wangxing_specialization_training.cmd `
  --device cpu --identity-limit 0 --identity-frames 1
```

When the Seedance manifest exists, the training launcher automatically adds
it. The resulting source profile distinguishes `real_wangxing` from
`generated_wangxing` using the same multimodal facial sequence features.

`-IdentityLimit 0` uses every video below each source root. Increasing
`-IdentityFrames` improves the training prototype but increases runtime.
The script writes `tmp/wangxing_specialization_training.log`.
The checked-in local profile was built from a balanced 240-video-per-root
identity calibration subset with one training frame per video; runtime
evaluation still samples multiple frames and applies the consistency gate.

The expression profile is built from all recognized real Wang Xing emotion
CSV files. The default mapping is:

```text
kaixin -> smile
fennu, shengqi -> anger
jingya -> surprise
kongju -> fear
beishang -> sadness
yanwu -> disgust
```

## Negative Data

The repository contains a bounded RAVDESS subset at
`data/negative/ravdess`: six actors and 240 manifest-selected videos
(plus previously extracted files that remain available to the profile).
The manifest records the selected actor/emotion balance. CREMA-D and VoxCeleb
are not present in the current workspace and were not downloaded in this
iteration. RAVDESS can be expanded with the existing resumable downloader:

```powershell
.\.venv\Scripts\python.exe scripts\download_ravdess_negative.py `
  --actors 1,2,3,4,5,6 `
  --max-videos 240
```

After adding negatives, run the identity training script again. Other-person
videos are identity negatives only; they are never added to Wang Xing's
expression support domain. Seedance videos are Wang Xing identity positives,
generated-domain samples, and automatic expression pseudo-label candidates;
only high-confidence pseudo-labels enter the expression support domain.

The generated identity profile stores calibration metrics including ROC-AUC,
PR-AUC, EER, and FPR at 1% and 5% false-positive rates. These are calibration
summaries, not a substitute for a held-out cross-batch evaluation.

## Redesign Contract

The specialization follows two serial decisions:

1. Identity: decide `wangxing`, `not_wangxing`, or `uncertain` from a
   quality-weighted multi-frame ArcFace track, real and generated Wang Xing
   prototypes, and external negative prototypes.
2. Expression support: only after identity is `wangxing`, measure the maximum
   compatibility with the real Wang Xing expression support domains. The
   result exposes the top two profiles and can report that the exact emotion
   class is uncertain.

The generated Seedance source profile is secondary real-versus-generated
evidence. It does not gate identity and does not turn unknown or ambiguous
Seedance emotion labels into expression ground truth. Only
`high_confidence` pseudo labels can be included as auxiliary expression
samples.

The ordinary five category scores are kept separate. The specialization is
reported under `wangxing_au` / `expression_evidence` and must not rewrite the
ordinary expression score or weighted total.

Run the implementation/data audit with:

```powershell
.\.venv\Scripts\python.exe scripts\check_wangxing_redesign.py `
  --output outputs\wangxing_redesign_check.json
```

Use `--strict` in CI or before deployment. The current workspace is expected
to remain `partial` until the texture forensic profile is expanded beyond its
small preliminary calibration set.

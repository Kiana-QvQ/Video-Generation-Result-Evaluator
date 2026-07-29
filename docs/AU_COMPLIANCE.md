# AU Compliance Pipeline

This pipeline evaluates whether a generated expression is consistent with
Wang Xing's real facial-action patterns.

## 1. Extract AU CSV

Install the official LibreFace CLI in its supported environment first. The
CLI writes one row per video frame.

```powershell
.\.venv\Scripts\python.exe scripts\extract_libreface_au.py `
    --manifest data\video\expression_reference_manifest.json `
    --output-root data\au\libreface `
    --only-emotions `
    --device cpu
```

For a generated video or a driver video outside the dataset:

```powershell
.\.venv\Scripts\python.exe scripts\extract_libreface_au.py `
    --input .\generated.mp4 `
    --output-root data\au\generated `
    --device cpu
```

For a complete directory tree such as `data\MD_CL`, use `--input-root`.
The output keeps the same relative directory structure, so files with the
same name in different `CL_*` directories do not overwrite each other:

```powershell
.\.venv\Scripts\python.exe scripts\extract_libreface_au.py `
    --input-root data\MD_CL `
    --output-root data\au\MD_CL `
    --device cuda `
    --batch-size 64 `
    --num-workers 2
```

Existing CSV files are skipped by default. Add `--force` to rebuild them.
Use `--limit 1 --device cpu` first if you want to smoke-test the LibreFace
environment before starting the full extraction.

## 2. Build Wang Xing's AU profile

The profile uses the six canonical emotion classes:

```text
smile, anger, surprise, fear, annoyance, sadness
```

`anger` is `FenNu` and means explosive anger. `annoyance` is `ShengQi`
and means suppressed displeasure. They are not merged.

The trained feature contract keeps LibreFace tasks separate:

- Intensity: AU 1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26, normalized from
  LibreFace's 0-5 range to 0-1;
- Presence: AU 1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24, stored as
  auxiliary activation evidence.

These are not concatenated into one continuous 24- or 17-dimensional AU
vector. Profile and evaluation reports record supported and missing AU ids.
Missing columns remain `NaN` internally and reduce evidence confidence; they
are never silently interpreted as zero activation.

```powershell
.\.venv\Scripts\python.exe scripts\build_au_profile.py `
    --manifest data\video\expression_reference_manifest.json `
    --au-root data\au\libreface `
    --output data\au\wangxing_au_profile.json
```

## 3. Optional leakage classifier

The positive directory must contain real Wang Xing AU CSV files. The negative
directory must contain other-person or known-bad AI AU CSV files.

```powershell
.\.venv\Scripts\python.exe scripts\fit_au_leakage_classifier.py `
    --positive-root data\au\libreface `
    --negative-root data\au\negative `
    --output data\au\au_leakage_classifier.json
```

Without negative data, the evaluator uses target-profile anomaly and
driver-style overlap proxies. It does not pretend that a supervised leakage
classifier exists.

## 3.1 One-click RAVDESS negative-data pipeline

RAVDESS is the default public negative set. The script downloads only the
selected actor ZIP archives from the official Zenodo record, extracts at most
the requested number of videos, and writes a reusable manifest. It does not
download the complete dataset.

The default is two actors and 48 videos, balanced over the eight RAVDESS
emotion codes:

```powershell
.\scripts\run_au_training_pipeline.ps1 `
    -NegativeDataset RAVDESS `
    -RavdessActors 1,2 `
    -MaxNegativeVideos 48 `
    -Device cuda
```

For a one-command launcher without opening PowerShell parameters:

```powershell
.\.venv\Scripts\python.exe start.py `
    --train-au `
    --negative-dataset RAVDESS `
    --ravdess-actors 1,2 `
    --ravdess-source HUGGINGFACE `
    --ravdess-cache-root data/cache/ravdess/huggingface `
    --max-negative-videos 48 `
    --au-device cuda
```

`start.py --train-au` calls the shared Python runner
`scripts/run_au_training_pipeline.py`. The normal HTTP webpage remains
independent from this training command.

`HUGGINGFACE` is an optional mirror source for the actor ZIP files. Keep its
cache directory separate from a partially downloaded Zenodo archive. Use
`--ravdess-source ZENODO` when the official source is reachable.

To use a slightly broader but still bounded subset:

```powershell
.\scripts\run_au_training_pipeline.ps1 `
    -NegativeDataset RAVDESS `
    -RavdessActors 1,2,3,4 `
    -RavdessEmotions 1,3,4,5,6,7,8 `
    -MaxNegativeVideos 96 `
    -Device cuda
```

The downloaded actor archives are cached in `data/cache/ravdess`; selected
videos are placed in `data/negative/ravdess/videos`, and the manifest is
`data/negative/ravdess/negative_manifest.json`.

The training command then:

1. Extracts AU CSV files for Wang Xing's emotion clips.
2. Extracts AU CSV files for the selected RAVDESS clips.
3. Builds `data/au/wangxing_au_profile.json`.
4. Builds `data/au/au_leakage_classifier.json`.

RAVDESS is a cross-identity real-expression negative set. It should not be
interpreted as a ground-truth expression set for Wang Xing.

## 3.2 One-click licensed Synthesized MetaHuman negative-data pipeline

The Synthesized MetaHuman dataset is access-controlled. Complete its
agreement process and use the official ZIP received from EURECOM:

```powershell
.\scripts\run_au_training_pipeline.ps1 `
    -NegativeDataset MetaHuman `
    -MetaHumanArchive .\licenses\SynthesizedMetaHuman.zip `
    -Device cuda
```

The one-click script:

1. Samples a bounded negative subset from the licensed archive.
2. Extracts AU CSV files for Wang Xing's emotion clips.
3. Extracts AU CSV files for the MetaHuman negative clips.
4. Builds `data/au/wangxing_au_profile.json`.
5. Builds `data/au/au_leakage_classifier.json`.

It does not bypass the dataset agreement or download an unapproved archive.

## 4.1 One-command evaluation

After the AU profile and leakage classifier have been trained, evaluate a
generated video with one command. The wrapper extracts the generated video's
AU CSV automatically, loads the two JSON model artifacts, and writes the
final evaluation report.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_generated_video.py `
    --generated-video .\generated.mp4 `
    --au-profile data\au\wangxing_au_profile.json `
    --leakage-classifier data\au\au_leakage_classifier.json `
    --expected-class smile `
    --target-image .\target.png `
    --output outputs\wangxing_au_compliance.json `
    --device cuda
```

If a driver video is available, add `--driver-video .\driver.mp4`.
The wrapper extracts its AU CSV and includes the driver-expression and
event-timing scores. Add `--cache-root data\au\cache` to enable a
content-addressed cache; rerunning the command reuses the same video's AU
CSV unless `--force` is supplied.

## 4. Evaluate one generated result

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_au_compliance.py `
    --generated-video .\generated.mp4 `
    --generated-au data\au\generated\generated.csv `
    --driver-au data\au\driver\driver.csv `
    --au-profile data\au\wangxing_au_profile.json `
    --expected-class smile `
    --target-video data\video\Neutral\124071016307_clip0001.mp4 `
    --leakage-classifier data\au\au_leakage_classifier.json `
    --output outputs\wangxing_au_compliance.json
```

The output contains:

- ArcFace identity preservation;
- AU personal-pattern compliance;
- Intensity and Presence AU coverage as separate evidence;
- AU DTW driver-expression preservation;
- constrained AU DTW plus first-order velocity similarity;
- AU time curves for generated and optional driver sequences;
- event-level AU records with start/end frame, duration, peak frame, and peak
  intensity;
- driver temporal-event alignment;
- face-quality gate status and usable-frame ratio;
- driver identity leakage risk;
- combined person-likeness score;
- anomalous AU frame indices.

For the Wang Xing-specific objective, use the `wangxing_targeted` section in
the report as the primary decision. It uses the Wang Xing AU profile as the
main evidence; identity images and driver videos are optional. The general
`fusion` section is a broader person-likeness decision and may remain
`review` when identity or driver evidence is not supplied.

The general fusion currently records the implemented weights explicitly:
identity 40%, personal AU 40%, and driver expression 20%. Missing components
are renormalized over the available evidence. The Wang Xing-targeted score is
an arithmetic mean of available personal-AU, driver-expression, and temporal
alignment evidence; it has no fixed identity weight. These weights are
implementation defaults and should be calibrated on a held-out validation set.

The current automatic mode also reports an `evidence_quality_status`,
`evidence_confidence_0_1`, and `evaluation_meta.evaluator_version`. A low
face-quality or low usable-frame ratio forces the targeted decision to
`review` instead of silently treating the AU score as reliable. These
decisions are deterministic model evidence and should not be interpreted as
human ground truth. AU dynamics are an individualized behavioral prior and
evidence of pattern drift; they are not, by themselves, an identity verdict.

Automatic expression selection uses both intensity AU and the auxiliary
presence AU representation when it is available. The personal AU score uses
55% intensity evidence and 45% presence evidence. Class selection additionally
uses a cross-class relative score, rather than comparing each class against
its own distance threshold; this prevents a small, broad class profile from
winning over a closer expression merely because its fitted threshold is
larger.

The scoring path now keeps two separate representations of the generated
sequence:

- Personal AU and identity evidence use only frames passing the face-quality
  mask.
- AU event timing keeps the original frame indices and full clip duration,
  marking low-quality frames as invalid instead of concatenating the valid
  frames.

The main evaluator's temporal-stability category remains independent from the
Wang Xing AU path and reads the original result video. Do not replace the
result video with a filtered-face video before calculating optical flow,
warping, or jitter.

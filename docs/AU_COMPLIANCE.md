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
    --input D:\path\generated.mp4 `
    --output-root data\au\generated `
    --device cpu
```

## 2. Build Wang Xing's AU profile

The profile uses the six canonical emotion classes:

```text
smile, anger, surprise, fear, annoyance, sadness
```

`anger` is `FenNu` and means explosive anger. `annoyance` is `ShengQi`
and means suppressed displeasure. They are not merged.

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
    -MetaHumanArchive D:\licensed\SynthesizedMetaHuman.zip `
    -Device cuda
```

The one-click script:

1. Samples a bounded negative subset from the licensed archive.
2. Extracts AU CSV files for Wang Xing's emotion clips.
3. Extracts AU CSV files for the MetaHuman negative clips.
4. Builds `data/au/wangxing_au_profile.json`.
5. Builds `data/au/au_leakage_classifier.json`.

It does not bypass the dataset agreement or download an unapproved archive.

## 4. Evaluate one generated result

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_au_compliance.py `
    --generated-video D:\path\generated.mp4 `
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
- AU DTW driver-expression preservation;
- driver identity leakage risk;
- combined person-likeness score;
- anomalous AU frame indices.

Hard thresholds should be calibrated on held-out human annotations before
using the result as an automatic block/allow decision.

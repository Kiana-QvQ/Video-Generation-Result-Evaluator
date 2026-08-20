# Joint Forensics Model

This is an isolated foundation for the Wang Xing video assessment plan. It
does not replace the ordinary five-category evaluator and it does not modify
the existing profile builder.

## Outputs

The model has one shared temporal representation and five heads:

- `identity`: Wang Xing identity evidence.
- `expression`: expression-domain class distribution.
- `expression_support`: whether the motion is supported by real Wang Xing
  expression data.
- `quality`: learned quality label when human quality labels exist.
- `artifact`: real-versus-Seedance source evidence.

These outputs are intentionally independent. A generated video can have a
natural Wang Xing expression and good visual quality while still carrying
Seedance artifacts.

## Feature Contract

The training entry point consumes one `.npz` file per video. Each available
modality is a two-dimensional array with shape `[frames, feature_dim]`:

```text
visual       optional RGB/video encoder frame embeddings
facial       AU and Face Mesh temporal features
texture      texture, edge, residual, or frame-quality features
audio        optional audio or audio-visual synchronization features
frame_mask   optional boolean mask with shape [frames]
```

The current profile rebuild does not produce these files. This separation is
deliberate: the profile rebuild can continue without the joint model reading
the same high-dimensional CSV corpus.

## Manifest And Labels

Create a path-only manifest:

```powershell
& .\.venv\Scripts\python.exe `
  scripts\data_build\build_joint_forensics_manifest.py
```

The manifest records the current real/Seedance source label and marks the
existing source holdout. It does not invent quality or expression-support
labels. The directory-derived real expression class is a support-domain
label, not a claim that every recording is perfect.

Create a manual annotation template for the Seedance videos:

```powershell
& .\.venv\Scripts\python.exe `
  scripts\archive\create_joint_forensics_annotation_template.py
```

The important labels are:

- expression naturalness for Wang Xing;
- expression support yes/no;
- texture/detail quality;
- temporal quality;
- visible Seedance artifact strength;
- audio-visual synchronization when audio exists.

The three existing high-confidence pseudo labels are useful for review, but
they are not enough to supervise a reliable generated-expression model.

## Training Boundary

Training accepts only records in `profile_train` by default. The
`source_holdout` records are excluded. It also requires pre-extracted NPZ
features, so it cannot accidentally start another expensive CSV/video
extraction job:

```powershell
& .\.venv\Scripts\python.exe `
  scripts\web_forensics\train_joint_forensics.py `
  --manifest outputs\forensics\joint_forensics_manifest.json `
  --feature-root outputs\forensics\joint_features `
  --device cuda
```

Do not run the training command until feature files and manual labels exist.
The current Wang Xing manifest has no non-Wang Xing identity negatives, so
the training entry point automatically disables the identity head instead of
training an all-positive identity classifier. The existing ArcFace identity
profile remains the identity authority. The source head can use the known
real/Seedance domain label, but that label should be described as source
classification rather than a universal artifact label.

Evaluate one feature file after a checkpoint exists:

```powershell
& .\.venv\Scripts\python.exe `
  scripts\web_forensics\evaluate_joint_forensics.py `
  --checkpoint outputs\forensics\joint_forensics_model.pt `
  --features outputs\forensics\joint_features\data\WangXing_Seedance\candidate.npz
```

## Integration Order

1. Finish the current facial-motion and texture profiles.
2. Use them as the interpretable baseline.
3. Add extracted NPZ features and manual labels.
4. Train the small shared temporal model without the source holdout.
5. Calibrate each output on its own validation data.
6. Keep ordinary five-category scores unchanged and attach the joint report
   as Wang Xing/forensics evidence.

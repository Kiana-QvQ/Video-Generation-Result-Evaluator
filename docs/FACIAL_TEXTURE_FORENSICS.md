# Facial Motion and Texture Forensics

This is the initial implementation for separating real-capture evidence from
Seedance-like evidence. It deliberately keeps two branches independent.

## Branches

### Facial Motion

`evaluator/forensics/facial_motion.py` reads AU and Face Mesh CSV files and
extracts:

- AU intensity summaries, velocity, acceleration, and jerk;
- activation events and duration statistics;
- normalized landmark-group motion for brow, eye, mouth, cheek, and jaw;
- AU co-activation correlations;
- cross-region phase coherence and estimated lag.

The branch can build a real-only profile or a two-domain
`real_vs_seedance` profile. Without both domains it reports features only and
does not claim to identify the source.

### Texture Detail

`evaluator/forensics/texture_detail.py` reads RGB frames or a video path and
extracts:

- local high-frequency and Laplacian detail;
- gradient and intensity-distribution statistics;
- DCT high-frequency energy;
- optical-flow-aligned temporal residuals;
- texture flicker and frame-to-frame stability.

These are not a replacement for MANIQA/MUSIQ. Those metrics remain useful for
perceived quality, while this branch is intended to provide evidence about
local texture persistence and temporal consistency.

## Data Protocol

Profiles should be trained with the same identity, framing, frame rate,
resolution, codec, and comparable expression content where possible. Split
train and test sets by source video or generation batch, not by individual
frames. Record the Seedance version, generation mode, prompt or driver id,
codec, and resolution in the surrounding manifest.

The first baseline uses profile distance rather than a learned deep video
classifier. This keeps the result inspectable and reduces the chance that the
detector learns a trivial resolution or codec shortcut. It should be replaced
or supplemented with a held-out classifier after enough matched Seedance
videos are available.

## Current Boundary

The new package is not automatically included in the existing five-category
score. The existing expression score still represents expression/reference
fit, and the existing texture score still represents visual quality or
ground-truth similarity. The new report adds separate forensic evidence so
these meanings are not mixed accidentally.

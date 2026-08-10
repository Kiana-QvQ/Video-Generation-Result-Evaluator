from pathlib import Path
import csv
import json
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator.forensics.facial_motion import extract_facial_motion_features

base = Path(__file__).resolve().parents[1]
md = base / "data/au/MD_CL/CL_beishang01/040061620398.csv"
cache = (
    base
    / "outputs/au_cache/wangxing_specialization_v1"
    / "8a73a61c6e8542f3b2051923a57904d7c1b85b392f6233bad3d480f83eef5a76.csv"
)
seedance = next((base / "data/au/WangXing_Seedance").glob("*.csv"))


def time_stats(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = [float(r["frame_time_in_ms"]) for r in rows if r.get("frame_time_in_ms")]
    fields = list(rows[0].keys()) if rows else []
    extra = [
        c
        for c in fields
        if c.startswith("face_") or c in {"facial_expression", "gaze_yaw", "gaze_pitch"}
    ]
    return {
        "path": str(path.name),
        "n": len(times),
        "t0": times[0] if times else None,
        "t1": times[1] if len(times) > 1 else None,
        "tmax": times[-1] if times else None,
        "median_dt": float(np.median(np.diff(times))) if len(times) > 1 else None,
        "extra_cols": extra,
    }


print("MD_time", time_stats(md))
print("CACHE_time", time_stats(cache))
print("SEEDANCE_sample_time", time_stats(seedance))

# Compare a few velocity features that depend on timestamps
md_f = extract_facial_motion_features(md, time_aware_derivatives=True)["features"]
wx_f = extract_facial_motion_features(cache, time_aware_derivatives=True)["features"]
keys = [
    "au_01_velocity_p95",
    "au_12_velocity_p95",
    "au_25_velocity_p95",
    "au_01_acceleration_p95",
    "landmark_mouth_velocity_p95",
    "landmark_mouth_jerk_p95",
    "au_event_active_ratio",
    "au_01_mean",
    "au_12_mean",
    "motion_coherence_0_1",
]
print("feature_delta")
for key in keys:
    print(
        key,
        "MD",
        round(float(md_f.get(key, 0.0)), 6),
        "WX",
        round(float(wx_f.get(key, 0.0)), 6),
        "ratio",
        round(
            float(wx_f.get(key, 0.0)) / max(float(md_f.get(key, 0.0)), 1e-12),
            4,
        ),
    )

# How many MD_CL files have second-scale timestamps?
bad = good = 0
examples = []
for path in sorted((base / "data/au/MD_CL").rglob("*.csv"))[:200]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "frame_time_in_ms" not in rows[0]:
        continue
    tmax = float(rows[-1]["frame_time_in_ms"])
    if tmax < 100:
        bad += 1
        if len(examples) < 3:
            examples.append((str(path.relative_to(base)), tmax, len(rows)))
    else:
        good += 1
print("md_sample200_tmax_lt_100", bad, "ge_100", good, "examples", examples)

# Seedance domain time units
sbad = sgood = 0
for path in sorted((base / "data/au/WangXing_Seedance").glob("*.csv")):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "frame_time_in_ms" not in rows[0]:
        continue
    tmax = float(rows[-1]["frame_time_in_ms"])
    if tmax < 100:
        sbad += 1
    else:
        sgood += 1
print("seedance_tmax_lt_100", sbad, "ge_100", sgood)

# Also check if non-time-aware scoring would align
from evaluator.forensics.facial_motion import score_facial_motion

profiles = json.loads(
    (base / "outputs/forensics/forensics_profiles.json").read_text(
        encoding="utf-8-sig"
    )
)
# Build temporary no-time-aware profile from existing means is hard;
# instead compare feature extract without time awareness.
md_nt = extract_facial_motion_features(md, time_aware_derivatives=False)["features"]
wx_nt = extract_facial_motion_features(cache, time_aware_derivatives=False)["features"]
print("no_time_aware_velocity")
for key in ["au_01_velocity_p95", "landmark_mouth_velocity_p95", "au_01_mean"]:
    print(key, "MD", md_nt.get(key), "WX", wx_nt.get(key))

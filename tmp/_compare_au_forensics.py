from pathlib import Path
import json, hashlib, csv
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator.forensics.facial_motion import (
    score_facial_motion,
    extract_facial_motion_features,
)

base = Path(__file__).resolve().parents[1]
md = base / "data/au/MD_CL/CL_beishang01/040061620398.csv"
cache = (
    base
    / "outputs/au_cache/wangxing_specialization_v1"
    / "8a73a61c6e8542f3b2051923a57904d7c1b85b392f6233bad3d480f83eef5a76.csv"
)
profiles = json.loads(
    (base / "outputs/forensics/forensics_profiles.json").read_text(
        encoding="utf-8-sig"
    )
)
fm = profiles["facial_motion"]


def head_info(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    au_int = [
        c
        for c in fields
        if "intensity" in c.lower() or c.lower().startswith("au_")
    ]
    lm = [c for c in fields if c.lower().startswith("lm_mp_")]
    times = [float(r.get("frame_time_in_ms") or "nan") for r in rows[:5]]
    last_t = (
        float(rows[-1].get("frame_time_in_ms") or "nan") if rows else None
    )
    align = set()
    for row in rows[:80]:
        if "face_alignment_method" in row and row.get("face_alignment_method"):
            align.add(row.get("face_alignment_method"))
    return {
        "rows": len(rows),
        "ncols": len(fields),
        "au_cols": au_int,
        "lm_count": len(lm),
        "has_z": any(c.endswith("_z") for c in lm),
        "first_times": times,
        "last_time": last_t,
        "align_methods_sample": sorted(a for a in align if a),
        "size": path.stat().st_size,
        "sha16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
    }


print("MD", json.dumps(head_info(md), ensure_ascii=False))
print("CACHE_exists", cache.is_file())
if cache.is_file():
    print("CACHE", json.dumps(head_info(cache), ensure_ascii=False))
    print("same_bytes", md.read_bytes() == cache.read_bytes())

for label, path in [("MD_CL", md), ("WX_CACHE", cache)]:
    if not path.is_file():
        continue
    feat = extract_facial_motion_features(path, time_aware_derivatives=True)
    scored = score_facial_motion(path, fm)
    metrics = scored["metrics"]
    print(
        label,
        json.dumps(
            {
                "frames": feat["frame_count"],
                "timebase": feat["timebase"],
                "landmark_available": feat["landmark_available"],
                "supported_au": feat["supported_au_ids"],
                "raw": metrics.get("raw_real_domain_evidence_0_1"),
                "real_fit": metrics.get("real_domain_fit_0_1"),
                "seedance_fit": metrics.get("seedance_domain_fit_0_1"),
                "landmark_ratio": metrics.get("landmark_valid_frame_ratio"),
                "feature_mode": metrics.get("feature_mode"),
                "motion_coherence": metrics.get("motion_coherence_0_1"),
                "source": feat.get("source"),
            },
            ensure_ascii=False,
        ),
    )

hold = json.loads(
    (base / "data/forensics/holdout_split.json").read_text(encoding="utf-8")
)
hits = [
    record
    for record in hold["real"]
    if "040061620398" in record.get("au", "")
    or "040061620398" in record.get("video", "")
]
print("holdout_hits", hits)
srcs = fm.get("provenance", {}).get("real_au_sources", [])
print("in_profile_sources_beishang01", "CL_beishang01/040061620398.csv" in srcs)
print("sources_with_398", [s for s in srcs if "040061620398" in s])
print("feature_protocol", fm.get("feature_protocol"))
print(
    "counts",
    fm["real"]["sample_count"],
    fm["seedance"]["sample_count"],
)
print("seedance_au_root", fm.get("provenance", {}).get("seedance_au_root"))
print("real_au_root", fm.get("provenance", {}).get("real_au_root"))

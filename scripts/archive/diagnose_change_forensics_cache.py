from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from evaluator.modules.forensics import analyze_forensics
from evaluator.modules.forensics.facial_motion import (
    extract_facial_motion_features,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min


def _video_name_from_status(payload: dict[str, Any]) -> str | None:
    original = payload.get("original_files", {})
    if isinstance(original, dict):
        value = original.get("result_video")
        if isinstance(value, str) and value.strip():
            return Path(value).name
    value = payload.get("name")
    if isinstance(value, str) and value.strip():
        return Path(value).name
    return None


def _latest_web_runs(web_runs_root: Path) -> dict[str, Path]:
    latest: dict[str, tuple[datetime, Path]] = {}
    for status_path in web_runs_root.glob("*/status.json"):
        try:
            payload = _read_json(status_path)
        except (OSError, json.JSONDecodeError):
            continue
        video_name = _video_name_from_status(payload)
        if not video_name or not video_name.endswith("_Change.mp4"):
            continue
        run_dir = status_path.parent
        stamp = max(
            _timestamp(payload.get("finished_at")),
            _timestamp(payload.get("updated_at")),
            _timestamp(payload.get("started_at")),
            _timestamp(payload.get("created_at")),
        )
        current = latest.get(video_name)
        if current is None or stamp >= current[0]:
            latest[video_name] = (stamp, run_dir)
    return {name: run_dir for name, (_, run_dir) in latest.items()}


def _sniff_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first_line = handle.readline()
    return "\t" if "\t" in first_line else ","


def _csv_basic_stats(path: Path) -> dict[str, Any]:
    delimiter = _sniff_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = list(reader.fieldnames or [])
        row_count = 0
        for _ in reader:
            row_count += 1
    landmark_x = [
        name for name in fieldnames if name.lower().startswith("lm_mp_") and name.lower().endswith("_x")
    ]
    landmark_y = [
        name for name in fieldnames if name.lower().startswith("lm_mp_") and name.lower().endswith("_y")
    ]
    landmark_z = [
        name for name in fieldnames if name.lower().startswith("lm_mp_") and name.lower().endswith("_z")
    ]
    return {
        "row_count": row_count,
        "field_count": len(fieldnames),
        "landmark_x_count": len(landmark_x),
        "landmark_y_count": len(landmark_y),
        "landmark_z_count": len(landmark_z),
        "has_insightface_bbox": any(
            name.lower().startswith("insightface_bbox") for name in fieldnames
        ),
    }


def _forensics_summary(report: dict[str, Any]) -> dict[str, Any]:
    authenticity = report.get("authenticity", {})
    scores = report.get("scores", {})
    branches = report.get("branches", {})
    facial = branches.get("facial_motion", {}) if isinstance(branches, dict) else {}
    texture = branches.get("texture_detail", {}) if isinstance(branches, dict) else {}
    facial_metrics = facial.get("metrics", {}) if isinstance(facial, dict) else {}
    texture_metrics = texture.get("metrics", {}) if isinstance(texture, dict) else {}
    return {
        "status": report.get("status"),
        "decision": authenticity.get("decision"),
        "raw_real_domain_evidence_0_1": authenticity.get(
            "raw_real_domain_evidence_0_1"
        ),
        "calibrated_real_probability_0_1": authenticity.get(
            "calibrated_real_probability_0_1"
        ),
        "confidence_0_1": authenticity.get("confidence_0_1"),
        "facial_raw_real_domain_evidence_0_1": facial_metrics.get(
            "raw_real_domain_evidence_0_1"
        ),
        "texture_raw_real_domain_evidence_0_1": texture_metrics.get(
            "raw_real_domain_evidence_0_1"
        ),
        "motion_coherence_0_1": facial_metrics.get("motion_coherence_0_1"),
        "landmark_valid_frame_ratio": facial_metrics.get(
            "landmark_valid_frame_ratio"
        ),
        "pose_normalized_frame_ratio": facial_metrics.get(
            "pose_normalized_frame_ratio"
        ),
        "input_quality_gate_0_1": facial_metrics.get("input_quality_gate_0_1"),
        "feature_mode": facial_metrics.get("feature_mode"),
        "texture_flicker_0_1": texture_metrics.get("texture_flicker_0_1"),
        "temporal_stability_proxy_0_1": texture_metrics.get(
            "temporal_stability_proxy_0_1"
        ),
        "nr_vqa_score_0_1": texture_metrics.get("nr_vqa_score_0_1"),
        "freq_forensics_score_0_1": texture_metrics.get(
            "freq_forensics_score_0_1"
        ),
        "branch_weights": authenticity.get("branch_weights"),
    }


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _delta(target: dict[str, Any], source: dict[str, Any], key: str) -> float | None:
    left = _finite_float(target.get(key))
    right = _finite_float(source.get(key))
    if left is None or right is None:
        return None
    return left - right


def _feature_std_lookup(profile: dict[str, Any]) -> dict[str, float]:
    names = list(profile.get("feature_names", []))
    real = profile.get("real", {})
    seedance = profile.get("seedance", {})
    real_std = list(real.get("std", []))
    seedance_std = list(seedance.get("std", []))
    lookup: dict[str, float] = {}
    for index, name in enumerate(names):
        values: list[float] = []
        if index < len(real_std):
            values.append(abs(float(real_std[index])))
        if index < len(seedance_std):
            values.append(abs(float(seedance_std[index])))
        lookup[name] = max(sum(values) / len(values), 0.05) if values else 0.05
    return lookup


def _top_feature_drifts(
    cached_features: dict[str, Any],
    current_features: dict[str, Any],
    std_lookup: dict[str, float],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, cached_value in cached_features.items():
        current_value = current_features.get(name)
        left = _finite_float(cached_value)
        right = _finite_float(current_value)
        if left is None or right is None:
            continue
        delta = right - left
        if abs(delta) < 1e-9:
            continue
        scale = std_lookup.get(name, 0.05)
        rows.append(
            {
                "feature": name,
                "cached": left,
                "current": right,
                "delta": delta,
                "abs_delta": abs(delta),
                "std_scale": scale,
                "abs_standardized_delta": abs(delta) / max(scale, 0.05),
            }
        )
    rows.sort(key=lambda item: item["abs_standardized_delta"], reverse=True)
    return rows[:limit]


def _stored_web_forensics(result_payload: dict[str, Any]) -> dict[str, Any] | None:
    wangxing = result_payload.get("wangxing_au")
    if isinstance(wangxing, dict) and isinstance(wangxing.get("forensics"), dict):
        return wangxing["forensics"]
    specialized = result_payload.get("specialized")
    if isinstance(specialized, dict):
        wangxing = specialized.get("wangxing_au")
        if isinstance(wangxing, dict) and isinstance(
            wangxing.get("forensics"),
            dict,
        ):
            return wangxing["forensics"]
    return None


def _run_forensics(
    *,
    profiles: dict[str, Any],
    au_path: Path,
    video_path: Path,
    max_frames: int,
    sample_fps: float,
) -> dict[str, Any]:
    return analyze_forensics(
        facial_motion=au_path,
        facial_motion_profile=profiles.get("facial_motion"),
        texture_detail=video_path,
        texture_detail_profile=profiles.get("texture_detail"),
        authenticity_calibrator=profiles.get("authenticity_calibrator"),
        max_frames=max_frames,
        sample_fps=sample_fps,
        detect_faces=True,
    )


def _video_diagnosis(
    *,
    video_name: str,
    run_dir: Path,
    current_video_root: Path,
    current_au_root: Path,
    profiles: dict[str, Any],
    facial_std_lookup: dict[str, float],
    sample_fps: float,
) -> dict[str, Any]:
    stem = Path(video_name).stem
    video_path = current_video_root / video_name
    current_au_path = current_au_root / f"{stem}.csv"
    wangxing_path = run_dir / "wangxing_au_result.json"
    result_path = run_dir / "result.json"
    run_status_path = run_dir / "status.json"
    run_video_path = run_dir / "result.mp4"

    wangxing_payload = _read_json(wangxing_path)
    result_payload = _read_json(result_path)
    status_payload = _read_json(run_status_path)
    cached_au_path = Path(wangxing_payload["evaluation_meta"]["generated_au"])

    cached_motion = extract_facial_motion_features(
        cached_au_path,
        time_aware_derivatives=True,
    )
    current_motion = extract_facial_motion_features(
        current_au_path,
        time_aware_derivatives=True,
    )

    stored_web_report = _stored_web_forensics(result_payload)
    reproduced_web_report = _run_forensics(
        profiles=profiles,
        au_path=cached_au_path,
        video_path=run_video_path,
        max_frames=16,
        sample_fps=sample_fps,
    )
    current_same_video_16 = _run_forensics(
        profiles=profiles,
        au_path=current_au_path,
        video_path=run_video_path,
        max_frames=16,
        sample_fps=sample_fps,
    )
    current_same_video_32 = _run_forensics(
        profiles=profiles,
        au_path=current_au_path,
        video_path=run_video_path,
        max_frames=32,
        sample_fps=sample_fps,
    )
    current_original_video_32 = _run_forensics(
        profiles=profiles,
        au_path=current_au_path,
        video_path=video_path,
        max_frames=32,
        sample_fps=sample_fps,
    )

    stored_summary = (
        _forensics_summary(stored_web_report)
        if isinstance(stored_web_report, dict)
        else None
    )
    reproduced_summary = _forensics_summary(reproduced_web_report)
    current16_summary = _forensics_summary(current_same_video_16)
    current32_summary = _forensics_summary(current_same_video_32)
    original32_summary = _forensics_summary(current_original_video_32)

    cached_sha = _sha256_file(cached_au_path)
    current_sha = _sha256_file(current_au_path)
    cached_stats = _csv_basic_stats(cached_au_path)
    current_stats = _csv_basic_stats(current_au_path)

    calibrated_mean = _finite_float(
        profiles.get("authenticity_calibrator", {}).get("mean")
    )
    reproduced_raw = _finite_float(
        reproduced_summary.get("raw_real_domain_evidence_0_1")
    )
    current16_raw = _finite_float(
        current16_summary.get("raw_real_domain_evidence_0_1")
    )

    return {
        "video_name": video_name,
        "run_dir": str(run_dir),
        "run_status": {
            "finished_at": status_payload.get("finished_at"),
            "updated_at": status_payload.get("updated_at"),
            "max_frames": status_payload.get("parameters", {}).get("max_frames"),
            "device": status_payload.get("parameters", {}).get("device"),
        },
        "paths": {
            "current_video": str(video_path),
            "current_au": str(current_au_path),
            "web_run_video": str(run_video_path),
            "cached_au": str(cached_au_path),
        },
        "hashes": {
            "cached_au_sha256": cached_sha,
            "current_au_sha256": current_sha,
            "cached_matches_current": cached_sha == current_sha,
        },
        "timestamps": {
            "cached_au_last_write": cached_au_path.stat().st_mtime,
            "current_au_last_write": current_au_path.stat().st_mtime,
        },
        "csv_stats": {
            "cached": cached_stats,
            "current": current_stats,
        },
        "stored_web_result": stored_summary,
        "reproduced_web_cached_16": reproduced_summary,
        "current_au_same_video_16": current16_summary,
        "current_au_same_video_32": current32_summary,
        "current_au_original_video_32": original32_summary,
        "effect_deltas": {
            "au_only_probability_delta": _delta(
                current16_summary,
                reproduced_summary,
                "calibrated_real_probability_0_1",
            ),
            "frame_budget_probability_delta": _delta(
                current32_summary,
                current16_summary,
                "calibrated_real_probability_0_1",
            ),
            "video_path_probability_delta": _delta(
                original32_summary,
                current32_summary,
                "calibrated_real_probability_0_1",
            ),
            "total_probability_delta_vs_web": _delta(
                original32_summary,
                reproduced_summary,
                "calibrated_real_probability_0_1",
            ),
        },
        "facial_feature_drifts": _top_feature_drifts(
            cached_motion.get("features", {}),
            current_motion.get("features", {}),
            facial_std_lookup,
        ),
        "flags": {
            "stale_au_cache": cached_sha != current_sha,
            "calibrator_cliff_zone_web": (
                calibrated_mean is not None
                and reproduced_raw is not None
                and abs(reproduced_raw - calibrated_mean) <= 0.015
            ),
            "calibrator_cliff_zone_current_16": (
                calibrated_mean is not None
                and current16_raw is not None
                and abs(current16_raw - calibrated_mean) <= 0.015
            ),
            "stored_vs_reproduced_match": (
                stored_summary is not None
                and _delta(
                    reproduced_summary,
                    stored_summary,
                    "calibrated_real_probability_0_1",
                )
                is not None
                and abs(
                    _delta(
                        reproduced_summary,
                        stored_summary,
                        "calibrated_real_probability_0_1",
                    )
                )
                <= 1e-6
            ),
        },
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Change Forensics Cache Diagnosis",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Forensics profile: `{report['profile_path']}`",
        f"- Profile domains: `{report['profile_domains']['facial_motion']}` / `{report['profile_domains']['texture_detail']}`",
        f"- Calibrator mean/scale/slope/intercept: `{report['calibrator']['mean']}` / `{report['calibrator']['scale']}` / `{report['calibrator']['slope']}` / `{report['calibrator']['intercept']}`",
        "",
    ]
    for item in report["videos"]:
        lines.extend(
            [
                f"## {item['video_name']}",
                "",
                f"- Stale AU cache: `{item['flags']['stale_au_cache']}`",
                f"- Cached AU == current AU hash: `{item['hashes']['cached_matches_current']}`",
                f"- Stored web decision/probability: `{item['stored_web_result']['decision'] if item['stored_web_result'] else None}` / `{item['stored_web_result']['calibrated_real_probability_0_1'] if item['stored_web_result'] else None}`",
                f"- Reproduced web decision/probability: `{item['reproduced_web_cached_16']['decision']}` / `{item['reproduced_web_cached_16']['calibrated_real_probability_0_1']}`",
                f"- Current AU on same web video (16f): `{item['current_au_same_video_16']['decision']}` / `{item['current_au_same_video_16']['calibrated_real_probability_0_1']}`",
                f"- Current AU on same web video (32f): `{item['current_au_same_video_32']['decision']}` / `{item['current_au_same_video_32']['calibrated_real_probability_0_1']}`",
                f"- Current AU on original video (32f): `{item['current_au_original_video_32']['decision']}` / `{item['current_au_original_video_32']['calibrated_real_probability_0_1']}`",
                f"- AU-only probability delta: `{item['effect_deltas']['au_only_probability_delta']}`",
                f"- Frame-budget probability delta: `{item['effect_deltas']['frame_budget_probability_delta']}`",
                f"- Video-path probability delta: `{item['effect_deltas']['video_path_probability_delta']}`",
                f"- Web raw near calibrator mean: `{item['flags']['calibrator_cliff_zone_web']}`",
                "",
                "| Top facial drift | Cached | Current | Delta | Std delta |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for drift in item["facial_feature_drifts"][:5]:
            lines.append(
                f"| `{drift['feature']}` | `{drift['cached']:.6f}` | `{drift['current']:.6f}` | `{drift['delta']:.6f}` | `{drift['abs_standardized_delta']:.3f}` |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Change-video web AU cache against current re-extracted AU "
            "and quantify its impact on the default forensics pipeline."
        )
    )
    parser.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles.json",
    )
    parser.add_argument("--current-video-root", default="data/test/AI")
    parser.add_argument("--current-au-root", default="data/au/test/AI")
    parser.add_argument("--web-runs-root", default="outputs/web_runs")
    parser.add_argument(
        "--output-json",
        default="outputs/forensics/change_forensics_cache_diagnosis.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/forensics/change_forensics_cache_diagnosis.md",
    )
    parser.add_argument("--sample-fps", type=float, default=8.0)
    args = parser.parse_args()

    profile_path = project_path(args.forensics_profile)
    current_video_root = project_path(args.current_video_root)
    current_au_root = project_path(args.current_au_root)
    web_runs_root = project_path(args.web_runs_root)
    output_json = project_path(args.output_json)
    output_md = project_path(args.output_md)

    profiles = _read_json(profile_path)
    facial_profile = profiles.get("facial_motion", {})
    facial_std_lookup = _feature_std_lookup(facial_profile)
    web_runs = _latest_web_runs(web_runs_root)

    target_videos = sorted(
        path.name for path in current_video_root.glob("*_Change.mp4") if path.is_file()
    )
    video_reports: list[dict[str, Any]] = []
    for video_name in target_videos:
        run_dir = web_runs.get(video_name)
        if run_dir is None:
            video_reports.append(
                {
                    "video_name": video_name,
                    "error": "missing_web_run",
                }
            )
            continue
        video_reports.append(
            _video_diagnosis(
                video_name=video_name,
                run_dir=run_dir,
                current_video_root=current_video_root,
                current_au_root=current_au_root,
                profiles=profiles,
                facial_std_lookup=facial_std_lookup,
                sample_fps=args.sample_fps,
            )
        )

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "profile_path": str(profile_path),
        "profile_domains": {
            "facial_motion": profiles.get("facial_motion", {}).get("domain"),
            "texture_detail": profiles.get("texture_detail", {}).get("domain"),
        },
        "calibrator": profiles.get("authenticity_calibrator", {}),
        "videos": video_reports,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(_markdown_report(report) + "\n", encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

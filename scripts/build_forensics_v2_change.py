from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from evaluator.modules.core.hardware_policy import resolve_policy
from evaluator.modules.forensics import (
    analyze_forensics,
    build_texture_detail_profile,
    build_two_domain_facial_motion_profile,
    extract_texture_detail_features,
    fit_probability_calibrator,
)
from evaluator.modules.forensics.holdout import holdout_paths, load_holdout_manifest
from evaluator.modules.forensics.seedance_authenticity import (
    apply_probability_calibrator,
    fuse_authenticity_evidence,
)

VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".webm",
    ".wmv",
}
AU_SUFFIXES = {".csv", ".tsv"}
CHANGE_TRAIN_STEMS = (
    "BaiJunZhiJiang_Change",
    "Happy_Change",
    "LeJiShengBei_Change",
)
CHANGE_EVAL_STEMS = (
    "YanWu_Change",
    "ImissU_Change",
)
CHANGE_ALL_STEMS = (*CHANGE_TRAIN_STEMS, *CHANGE_EVAL_STEMS)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _ensure_file(path: Path, *, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _change_record(stem: str, *, split: str) -> dict[str, Any]:
    video = _ensure_file(project_path(f"data/test/AI/{stem}.mp4"), label="Change video")
    au = _ensure_file(project_path(f"data/au/test/AI/{stem}.csv"), label="Change AU")
    return {
        "name": f"{stem}.mp4",
        "stem": stem,
        "split": split,
        "video": _rel(video),
        "au": _rel(au),
    }


def build_protocol_payload() -> dict[str, Any]:
    return {
        "schema_version": "forensics_v2_change_protocol_v1",
        "note": (
            "Default forensics v2 protocol: keep current facial_motion + "
            "texture + calibrator architecture, but add Change negatives "
            "into the generated domain without leaking Change eval clips "
            "into profile fitting."
        ),
        "real_domain": {
            "au_root": "data/au/MD_CL",
            "video_root": "data/MD_CL",
        },
        "generated_domain": {
            "base_au_root": "data/au/WangXing_Seedance",
            "base_video_root": "data/WangXing_Seedance",
            "change_train": [
                _change_record(stem, split="train")
                for stem in CHANGE_TRAIN_STEMS
            ],
            "change_eval": [
                _change_record(stem, split="eval")
                for stem in CHANGE_EVAL_STEMS
            ],
        },
    }


def build_extra_generated_manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "forensics_v2_extra_generated_manifest_v1",
        "note": (
            "Additional generated-domain training samples for default "
            "forensics v2. These are Change negatives that may join the "
            "generated domain profile."
        ),
        "records": [
            {
                "name": item["name"],
                "stem": item["stem"],
                "source": "change_train",
                "video_path": item["video"],
                "au_path": item["au"],
            }
            for item in protocol["generated_domain"]["change_train"]
        ],
    }


def build_combined_holdout_manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    base = load_holdout_manifest(project_path("data/forensics/holdout_split.json"))
    real = list(base.get("real", []))
    generated = list(base.get("seedance", []))
    for item in protocol["generated_domain"]["change_eval"]:
        generated.append(
            {
                "name": item["name"],
                "video": item["video"],
                "au": item["au"],
                "source": "change_eval",
                "change_eval": True,
            }
        )
    return {
        "schema_version": "forensics_holdout_split_v1",
        "note": (
            "Default forensics v2 holdout: official real holdout + official "
            "Seedance holdout + Change eval negatives. Use this both to "
            "exclude evaluation clips from profile fitting and to fit the "
            "32-frame v2 calibrator."
        ),
        "summary": {
            "real": len(real),
            "seedance": len(generated),
            "official_seedance_holdout": len(base.get("seedance", [])),
            "change_eval": len(protocol["generated_domain"]["change_eval"]),
        },
        "real": real,
        "seedance": generated,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _files(root: Path, suffixes: set[str]) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _limit(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    if limit == 1:
        return [paths[0]]
    indexes = [
        round(index * (len(paths) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [paths[index] for index in indexes]


def _load_generated_manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"records must be a list: {manifest_path}")
    return [record for record in records if isinstance(record, dict)]


def _resolved_path(value: str | None) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _collect_training_paths(
    *,
    root: Path,
    suffixes: set[str],
    excluded: set[str],
    extra_records: list[dict[str, Any]],
    extra_key: str,
) -> list[Path]:
    paths = [
        path
        for path in _files(root, suffixes)
        if str(path.resolve()) not in excluded
    ]
    seen = {str(path.resolve()) for path in paths}
    for record in extra_records:
        path = _resolved_path(record.get(extra_key))
        if path is None:
            continue
        resolved = str(path.resolve())
        if resolved in excluded or resolved in seen:
            continue
        _ensure_file(path, label=extra_key)
        paths.append(path.resolve())
        seen.add(resolved)
    return sorted(paths)


def _build_texture_domain(
    paths: list[Path],
    *,
    domain: str,
    max_frames: int,
    sample_fps: float,
    nr_vqa_backends: Sequence[str] | None = None,
    device: str = "auto",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        if index == 1 or index == total or index % 5 == 0:
            print(
                f"[texture:{domain}] {index}/{total} {path.name}",
                flush=True,
            )
        try:
            records.append(
                extract_texture_detail_features(
                    path,
                    max_frames=max_frames,
                    sample_fps=sample_fps,
                    nr_vqa_backends=nr_vqa_backends,
                    device=device,
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            skipped.append({"path": str(path), "error": str(exc)})
    if not records:
        return None, {
            "domain": domain,
            "processed_count": 0,
            "skipped_count": len(skipped),
            "skipped": skipped[:50],
        }
    return build_texture_detail_profile(records, domain=domain), {
        "domain": domain,
        "processed_count": len(records),
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
    }


def _holdout_samples(payload: dict[str, Any], domain: str) -> list[dict[str, Any]]:
    records = payload.get(domain, [])
    if not isinstance(records, list):
        raise ValueError(f"holdout domain must be a list: {domain}")
    samples: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        video_path = _resolved_path(record.get("video") or record.get("video_path"))
        au_path = _resolved_path(record.get("au") or record.get("au_path"))
        if video_path is None and au_path is None:
            continue
        subgroup = (
            "change_eval"
            if record.get("change_eval") or record.get("source") == "change_eval"
            else "official_seedance_holdout"
            if domain == "seedance"
            else "official_real_holdout"
        )
        samples.append(
            {
                "domain": domain,
                "name": record.get("name"),
                "video_path": video_path.resolve() if video_path else None,
                "au_path": au_path.resolve() if au_path else None,
                "subgroup": subgroup,
            }
        )
    return samples


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    labels_array = np.asarray(labels, dtype=np.int32)
    scores_array = np.asarray(scores, dtype=np.float64)
    positive = scores_array[labels_array == 1]
    negative = scores_array[labels_array == 0]
    if positive.size == 0 or negative.size == 0:
        return None
    differences = positive[:, None] - negative[None, :]
    return float(
        (
            np.sum(differences > 0.0)
            + 0.5 * np.sum(differences == 0.0)
        )
        / differences.size
    )


def _brier_score(labels: list[int], probabilities: list[float]) -> float | None:
    if not labels or len(labels) != len(probabilities):
        return None
    targets = np.asarray(labels, dtype=np.float64)
    values = np.asarray(probabilities, dtype=np.float64)
    return float(np.mean((values - targets) ** 2))


def _expected_calibration_error(
    labels: list[int],
    probabilities: list[float],
    *,
    bins: int = 10,
) -> float | None:
    if not labels or len(labels) != len(probabilities):
        return None
    targets = np.asarray(labels, dtype=np.float64)
    values = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    error = 0.0
    for index in range(len(edges) - 1):
        lower = edges[index]
        upper = edges[index + 1]
        selected = (
            (values >= lower)
            & (
                (values < upper)
                if index < len(edges) - 2
                else (values <= upper)
            )
        )
        if not np.any(selected):
            continue
        weight = float(np.mean(selected))
        error += weight * abs(
            float(np.mean(values[selected]))
            - float(np.mean(targets[selected]))
        )
    return float(error)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _score_holdout_sample(
    sample: dict[str, Any],
    *,
    profile_payload: dict[str, Any],
    max_frames: int,
    sample_fps: float,
    nr_vqa_backends: Sequence[str] | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    report = analyze_forensics(
        facial_motion=sample.get("au_path"),
        facial_motion_profile=profile_payload.get("facial_motion"),
        texture_detail=sample.get("video_path"),
        texture_detail_profile=profile_payload.get("texture_detail"),
        authenticity_calibrator=None,
        max_frames=max_frames,
        sample_fps=sample_fps,
        nr_vqa_backends=nr_vqa_backends,
        device=device,
    )
    branches = report.get("branches", {})
    facial = branches.get("facial_motion", {}) if isinstance(branches, dict) else {}
    texture = branches.get("texture_detail", {}) if isinstance(branches, dict) else {}
    facial_metrics = facial.get("metrics", {}) if isinstance(facial, dict) else {}
    texture_metrics = texture.get("metrics", {}) if isinstance(texture, dict) else {}
    raw = report.get("authenticity", {}).get("raw_real_domain_evidence_0_1")
    if raw is None:
        raise RuntimeError(f"raw authenticity evidence unavailable for {sample}")
    return {
        "name": sample.get("name"),
        "domain": sample["domain"],
        "subgroup": sample["subgroup"],
        "au_path": str(sample.get("au_path")) if sample.get("au_path") else None,
        "video_path": (
            str(sample.get("video_path")) if sample.get("video_path") else None
        ),
        "raw_real_domain_evidence_0_1": float(raw),
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
        "texture_flicker_0_1": texture_metrics.get("texture_flicker_0_1"),
        "report": report,
    }


def _attach_calibrated_outputs(
    rows: list[dict[str, Any]],
    *,
    calibrator: dict[str, Any],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        report = row["report"]
        authenticity = fuse_authenticity_evidence(
            report.get("branches", {}).get("facial_motion"),
            report.get("branches", {}).get("texture_detail"),
            calibrator=calibrator,
        )
        enriched.append(
            {
                **{key: value for key, value in row.items() if key != "report"},
                "calibrated_real_probability_0_1": authenticity.get(
                    "calibrated_real_probability_0_1"
                ),
                "decision": authenticity.get("decision"),
                "confidence_0_1": authenticity.get("confidence_0_1"),
                "branch_weights": authenticity.get("branch_weights"),
                "uncertainty_reasons": authenticity.get("uncertainty_reasons"),
            }
        )
    return enriched


def _branch_auc(rows: list[dict[str, Any]], key: str) -> float | None:
    labels: list[int] = []
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        labels.append(1 if row["domain"] == "real" else 0)
        values.append(float(value))
    return _roc_auc(labels, values)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [1 if row["domain"] == "real" else 0 for row in rows]
    raw_scores = [float(row["raw_real_domain_evidence_0_1"]) for row in rows]
    probabilities = [
        float(row["calibrated_real_probability_0_1"])
        for row in rows
        if row.get("calibrated_real_probability_0_1") is not None
    ]
    probability_labels = [
        1 if row["domain"] == "real" else 0
        for row in rows
        if row.get("calibrated_real_probability_0_1") is not None
    ]
    generated_rows = [row for row in rows if row["domain"] != "real"]
    return {
        "count": len(rows),
        "real_count": sum(1 for row in rows if row["domain"] == "real"),
        "generated_count": sum(1 for row in rows if row["domain"] != "real"),
        "mean_raw_real_domain_evidence_0_1": _mean(raw_scores),
        "roc_auc_real_vs_generated_raw": _roc_auc(labels, raw_scores),
        "brier_score": _brier_score(probability_labels, probabilities),
        "expected_calibration_error_10": _expected_calibration_error(
            probability_labels,
            probabilities,
        ),
        "branch_auc": {
            "fused_raw": _branch_auc(rows, "raw_real_domain_evidence_0_1"),
            "facial_raw": _branch_auc(rows, "facial_raw_real_domain_evidence_0_1"),
            "texture_raw": _branch_auc(rows, "texture_raw_real_domain_evidence_0_1"),
        },
        "generated_decisions": {
            "real_capture": sum(
                1 for row in generated_rows if row.get("decision") == "real_capture"
            ),
            "seedance_like": sum(
                1 for row in generated_rows if row.get("decision") == "seedance_like"
            ),
            "uncertain": sum(
                1 for row in generated_rows if row.get("decision") == "uncertain"
            ),
        },
    }


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["subgroup"], []).append(row)
    return {name: _summarize_rows(group) for name, group in groups.items()}


def _score_change_batch(
    *,
    profile_payload: dict[str, Any],
    calibrator: dict[str, Any],
    max_frames: int,
    sample_fps: float,
    nr_vqa_backends: Sequence[str] | None = None,
    device: str = "auto",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stem in CHANGE_ALL_STEMS:
        split = "train" if stem in CHANGE_TRAIN_STEMS else "eval"
        sample = {
            "domain": "seedance",
            "subgroup": f"change_{split}",
            "name": f"{stem}.mp4",
            "video_path": project_path(f"data/test/AI/{stem}.mp4").resolve(),
            "au_path": project_path(f"data/au/test/AI/{stem}.csv").resolve(),
        }
        scored = _score_holdout_sample(
            sample,
            profile_payload=profile_payload,
            max_frames=max_frames,
            sample_fps=sample_fps,
            nr_vqa_backends=nr_vqa_backends,
            device=device,
        )
        row = _attach_calibrated_outputs([scored], calibrator=calibrator)[0]
        row["split"] = split
        rows.append(row)
    return rows


def prepare_outputs(args: argparse.Namespace) -> int:
    protocol = build_protocol_payload()
    generated_manifest = build_extra_generated_manifest(protocol)
    holdout_manifest = build_combined_holdout_manifest(protocol)
    _write_json(project_path(args.protocol_output), protocol)
    _write_json(project_path(args.generated_manifest_output), generated_manifest)
    _write_json(project_path(args.holdout_output), holdout_manifest)
    print(f"Wrote {project_path(args.protocol_output)}")
    print(f"Wrote {project_path(args.generated_manifest_output)}")
    print(f"Wrote {project_path(args.holdout_output)}")
    return 0


def train_outputs(args: argparse.Namespace) -> int:
    protocol_path = project_path(args.protocol)
    generated_manifest_path = project_path(args.extra_generated_manifest)
    holdout_manifest_path = project_path(args.holdout_manifest)
    if not protocol_path.is_file() or not generated_manifest_path.is_file() or not holdout_manifest_path.is_file():
        raise SystemExit(
            "Missing protocol/manifest files. Run the prepare subcommand first."
        )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    extra_generated_records = _load_generated_manifest_records(generated_manifest_path)
    holdout_manifest = load_holdout_manifest(holdout_manifest_path)
    nr_vqa_backends = (
        tuple(
            item.strip()
            for item in str(args.nr_vqa_backends).split(",")
            if item.strip()
        )
        if args.nr_vqa_backends
        else None
    )
    resolved_device = resolve_policy(args.device).resolved_device

    real_au_root = project_path(args.real_au_root)
    real_video_root = project_path(args.real_video_root)
    generated_au_root = project_path(args.generated_au_root)
    generated_video_root = project_path(args.generated_video_root)

    excluded_real_au = holdout_paths(holdout_manifest_path, domain="real", kind="au")
    excluded_generated_au = holdout_paths(
        holdout_manifest_path,
        domain="seedance",
        kind="au",
    )
    excluded_real_videos = holdout_paths(
        holdout_manifest_path,
        domain="real",
        kind="video",
    )
    excluded_generated_videos = holdout_paths(
        holdout_manifest_path,
        domain="seedance",
        kind="video",
    )

    real_au_paths = [
        path
        for path in _files(real_au_root, AU_SUFFIXES)
        if str(path.resolve()) not in excluded_real_au
    ]
    generated_au_paths = _collect_training_paths(
        root=generated_au_root,
        suffixes=AU_SUFFIXES,
        excluded=excluded_generated_au,
        extra_records=extra_generated_records,
        extra_key="au_path",
    )
    real_video_paths = _limit(
        [
            path
            for path in _files(real_video_root, VIDEO_SUFFIXES)
            if str(path.resolve()) not in excluded_real_videos
        ],
        args.max_videos_per_domain,
    )
    generated_video_paths = _limit(
        _collect_training_paths(
            root=generated_video_root,
            suffixes=VIDEO_SUFFIXES,
            excluded=excluded_generated_videos,
            extra_records=extra_generated_records,
            extra_key="video_path",
        ),
        args.max_videos_per_domain,
    )

    if not real_au_paths or not generated_au_paths:
        raise SystemExit("Both real and generated AU training sets are required.")
    if not real_video_paths or not generated_video_paths:
        raise SystemExit("Both real and generated video training sets are required.")

    print(
        "Building facial-motion profile: "
        f"real_au={len(real_au_paths)} generated_au={len(generated_au_paths)}",
        flush=True,
    )
    facial_motion_profile = build_two_domain_facial_motion_profile(
        real_au_paths,
        generated_au_paths,
        min_landmark_ratio=args.min_landmark_ratio,
        min_pose_ratio=args.min_pose_ratio,
    )
    real_texture, real_texture_report = _build_texture_domain(
        real_video_paths,
        domain="real",
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
        nr_vqa_backends=nr_vqa_backends,
        device=resolved_device,
    )
    generated_texture, generated_texture_report = _build_texture_domain(
        generated_video_paths,
        domain="seedance",
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
        nr_vqa_backends=nr_vqa_backends,
        device=resolved_device,
    )
    if real_texture is None or generated_texture is None:
        raise SystemExit("Texture profile could not be built for both domains.")

    profile_payload: dict[str, Any] = {
        "schema_version": "forensics_profiles_v2_change_v1",
        "facial_motion": {
            **facial_motion_profile,
            "provenance": {
                "real_au_root": str(real_au_root),
                "generated_au_root": str(generated_au_root),
                "extra_generated_manifest": str(generated_manifest_path),
                "holdout_manifest": str(holdout_manifest_path),
                "real_au_count": len(real_au_paths),
                "generated_au_count": len(generated_au_paths),
                "generated_domain_note": "WangXing_Seedance + Change train negatives",
                "min_landmark_ratio": float(args.min_landmark_ratio),
                "min_pose_ratio": float(args.min_pose_ratio),
            },
        },
        "texture_detail": {
            "schema_version": "texture_detail_forensics_v1",
            "domain": "real_vs_seedance",
            "feature_names": real_texture["feature_names"],
            "real": {
                key: real_texture[key]
                for key in ("sample_count", "mean", "std", "source_records")
            },
            "seedance": {
                key: generated_texture[key]
                for key in ("sample_count", "mean", "std", "source_records")
            },
        },
        "texture_provenance": {
            "real_video_root": str(real_video_root),
            "generated_video_root": str(generated_video_root),
            "extra_generated_manifest": str(generated_manifest_path),
            "holdout_manifest": str(holdout_manifest_path),
            "real_video_count": len(real_video_paths),
            "generated_video_count": len(generated_video_paths),
            "max_videos_per_domain": int(args.max_videos_per_domain),
            "max_frames_per_video": int(args.max_frames),
            "sample_fps": float(args.sample_fps),
            "nr_vqa_backends": (
                list(nr_vqa_backends) if nr_vqa_backends else None
            ),
            "device": resolved_device,
            "real_report": real_texture_report,
            "generated_report": generated_texture_report,
        },
        "warnings": [
            "Default forensics v2 keeps the existing architecture and expands the generated domain with Change negatives.",
            "Do not promote this profile to the web until you review the generated holdout and Change diagnostics in the report JSON.",
        ],
    }

    real_holdout_samples = _holdout_samples(holdout_manifest, "real")
    generated_holdout_samples = _holdout_samples(holdout_manifest, "seedance")
    holdout_scored_raw: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    holdout_samples = [*real_holdout_samples, *generated_holdout_samples]
    total_holdout = len(holdout_samples)
    for index, sample in enumerate(holdout_samples, start=1):
        if index == 1 or index == total_holdout or index % 5 == 0:
            print(
                f"[holdout] {index}/{total_holdout} "
                f"{sample.get('name') or sample.get('video_path')}",
                flush=True,
            )
        try:
            holdout_scored_raw.append(
                _score_holdout_sample(
                    sample,
                    profile_payload=profile_payload,
                    max_frames=args.max_frames,
                    sample_fps=args.sample_fps,
                    nr_vqa_backends=nr_vqa_backends,
                    device=resolved_device,
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(
                {
                    "sample": str(sample.get("au_path") or sample.get("video_path")),
                    "error": str(exc),
                }
            )

    real_scores = [
        float(row["raw_real_domain_evidence_0_1"])
        for row in holdout_scored_raw
        if row["domain"] == "real"
    ]
    generated_scores = [
        float(row["raw_real_domain_evidence_0_1"])
        for row in holdout_scored_raw
        if row["domain"] != "real"
    ]
    if len(real_scores) < args.min_samples_per_domain or len(generated_scores) < args.min_samples_per_domain:
        raise SystemExit(
            "Not enough scored holdout samples to fit the v2 calibrator. "
            f"real={len(real_scores)} generated={len(generated_scores)}"
        )

    calibrator = fit_probability_calibrator(real_scores, generated_scores)
    calibrator["status"] = "ready"
    calibrator["note"] = (
        "32-frame default forensics v2 calibrator fitted on official real "
        "holdout + official Seedance holdout + Change eval negatives."
    )
    profile_payload["authenticity_calibrator"] = calibrator

    holdout_scored = _attach_calibrated_outputs(
        holdout_scored_raw,
        calibrator=calibrator,
    )
    change_batch_rows = _score_change_batch(
        profile_payload=profile_payload,
        calibrator=calibrator,
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
        nr_vqa_backends=nr_vqa_backends,
        device=resolved_device,
    )
    labels = [1 if row["domain"] == "real" else 0 for row in holdout_scored]
    probabilities = [
        float(row["calibrated_real_probability_0_1"])
        for row in holdout_scored
        if row.get("calibrated_real_probability_0_1") is not None
    ]
    probability_labels = [
        1 if row["domain"] == "real" else 0
        for row in holdout_scored
        if row.get("calibrated_real_probability_0_1") is not None
    ]

    report_payload = {
        "schema_version": "forensics_v2_change_training_report_v1",
        "protocol": str(protocol_path),
        "holdout_manifest": str(holdout_manifest_path),
        "extra_generated_manifest": str(generated_manifest_path),
        "output_profile": str(project_path(args.output_profile)),
        "training": {
            "real_au_count": len(real_au_paths),
            "generated_au_count": len(generated_au_paths),
            "real_video_count": len(real_video_paths),
            "generated_video_count": len(generated_video_paths),
            "max_frames": int(args.max_frames),
            "sample_fps": float(args.sample_fps),
            "max_videos_per_domain": int(args.max_videos_per_domain),
            "min_landmark_ratio": float(args.min_landmark_ratio),
            "min_pose_ratio": float(args.min_pose_ratio),
            "nr_vqa_backends": (
                list(nr_vqa_backends) if nr_vqa_backends else None
            ),
            "device": resolved_device,
        },
        "validation": _summarize_rows(holdout_scored),
        "validation_by_group": _group_rows(holdout_scored),
        "change_batch_diagnostics": change_batch_rows,
        "calibrator": calibrator,
        "failures": failures[:100],
        "holdout_samples": holdout_scored,
        "promotion_checklist": [
            "Verify that change_eval rows no longer lean real_capture on the new profile.",
            "Compare validation_by_group. change_eval should improve without collapsing official_real_holdout.",
            "Only after review should you promote output_profile to outputs/forensics/forensics_profiles.json.",
        ],
        "aggregate_metrics": {
            "roc_auc_real_vs_generated_prob": _roc_auc(labels, probabilities)
            if len(probabilities) == len(labels)
            else None,
            "brier_score": _brier_score(probability_labels, probabilities),
            "expected_calibration_error_10": _expected_calibration_error(
                probability_labels,
                probabilities,
            ),
        },
    }

    _write_json(project_path(args.output_profile), profile_payload)
    _write_json(project_path(args.output_report), report_payload)
    print(f"Wrote {project_path(args.output_profile)}")
    print(f"Wrote {project_path(args.output_report)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and train default forensics v2: keep the current "
            "facial_motion + texture + calibrator architecture, but add "
            "Change negatives into the generated domain."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--protocol-output",
        default="data/forensics/forensics_v2_change_protocol.json",
    )
    prepare.add_argument(
        "--generated-manifest-output",
        default="data/forensics/forensics_v2_change_generated_train_manifest.json",
    )
    prepare.add_argument(
        "--holdout-output",
        default="data/forensics/holdout_split_forensics_v2_change.json",
    )

    train = subparsers.add_parser("train")
    train.add_argument(
        "--protocol",
        default="data/forensics/forensics_v2_change_protocol.json",
    )
    train.add_argument(
        "--extra-generated-manifest",
        default="data/forensics/forensics_v2_change_generated_train_manifest.json",
    )
    train.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split_forensics_v2_change.json",
    )
    train.add_argument("--real-au-root", default="data/au/MD_CL")
    train.add_argument("--generated-au-root", default="data/au/WangXing_Seedance")
    train.add_argument("--real-video-root", default="data/MD_CL")
    train.add_argument("--generated-video-root", default="data/WangXing_Seedance")
    train.add_argument(
        "--max-videos-per-domain",
        type=int,
        default=50,
        help="Texture profile videos per domain; 0 means all.",
    )
    train.add_argument("--max-frames", type=int, default=32)
    train.add_argument("--sample-fps", type=float, default=8.0)
    train.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for optional MUSIQ/pyiqa inference.",
    )
    train.add_argument(
        "--nr-vqa-backends",
        default=None,
        help=(
            "Comma-separated NR-VQA backend order. Default uses the package "
            "preference order; use builtin_nr_vqa for a fast smoke test."
        ),
    )
    train.add_argument("--min-landmark-ratio", type=float, default=0.0)
    train.add_argument("--min-pose-ratio", type=float, default=0.0)
    train.add_argument("--min-samples-per-domain", type=int, default=25)
    train.add_argument(
        "--output-profile",
        default="outputs/forensics/forensics_profiles_v2_change.json",
    )
    train.add_argument(
        "--output-report",
        default="outputs/forensics/forensics_v2_change_report.json",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare_outputs(args)
    if args.command == "train":
        return train_outputs(args)
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

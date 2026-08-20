"""One-click web-forensics v2 profile, fusion training, and evaluation.

This path does not use any video .pt model. It trains a compact web-only
fusion head over:
- Wang Xing source AU evidence;
- facial-motion AU/SSL/physiology/quality evidence;
- texture/frequency/NR-VQA evidence;
- window-level forensic summaries.

Commands:
    prepare
    build-profiles
    train
    evaluate
    all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path, resolve_profile
from evaluator.modules.forensics import (
    analyze_forensics,
    extract_texture_detail_features,
)
from evaluator.modules.forensics.learned_fusion_head import (
    extract_fusion_feature_dict,
)
from evaluator.modules.wangxing.wangxing_specialization import (
    build_source_profile,
)
from scripts.web_forensics.evaluate_single_video_forensics_dataset import (
    _build_web_card,
    _run_one,
)
from scripts.web_forensics.web_authenticity_policy import (
    apply_policy,
)

FEATURE_NAMES = (
    "wx_real_probability_0_1",
    "wx_generated_probability_0_1",
    "wx_margin_0_1",
    "wx_real_distance",
    "wx_generated_distance",
    "wx_real_score_0_1",
    "wx_generated_score_0_1",
    "wx_valid_frame_ratio",
    "fm_real_domain_fit_0_1",
    "fm_seedance_domain_fit_0_1",
    "fm_raw_real_domain_evidence_0_1",
    "fm_motion_coherence_0_1",
    "fm_au_relation_consistency_0_1",
    "fm_au_dynamics_naturalness_0_1",
    "fm_training_free_motion_prior_0_1",
    "fm_ssl_au_score_0_1",
    "fm_ssl_backbone_score_0_1",
    "fm_ssl_temporal_consistency_0_1",
    "fm_physio_rhythm_score_0_1",
    "fm_input_quality_gate_0_1",
    "fm_landmark_valid_frame_ratio",
    "fm_pose_normalized_frame_ratio",
    "branch_gap_wx_minus_fm",
    "branch_mean_real",
    "quality_min",
    "texture_raw_real_domain_evidence_0_1",
    "texture_real_domain_fit_0_1",
    "texture_seedance_domain_fit_0_1",
    "texture_stability_0_1",
    "texture_flicker_0_1",
    "texture_micro_temporal_0_1",
    "texture_frequency_0_1",
    "texture_nr_vqa_0_1",
    "texture_detail_quality_0_1",
    "texture_face_box_coverage_0_1",
    "fusion_confidence_0_1",
    "fusion_training_free_prior_0_1",
    "facial_window_mean_0_1",
    "facial_window_worst_0_1",
    "facial_window_aggregate_0_1",
    "texture_window_mean_0_1",
    "texture_window_worst_0_1",
    "texture_window_aggregate_0_1",
    "facial_window_p90_0_1",
    "facial_window_p95_0_1",
    "facial_window_std_0_1",
    "facial_window_anomaly_ratio_0_1",
    "facial_window_longest_anomaly_run_0_1",
    "texture_window_p90_0_1",
    "texture_window_p95_0_1",
    "texture_window_std_0_1",
    "texture_window_anomaly_ratio_0_1",
    "texture_window_longest_anomaly_run_0_1",
    "fm_landmark_missing_mask",
    "fm_pose_missing_mask",
    "fm_au_missing_mask",
    "fm_timestamp_irregularity_0_1",
    "fm_face_detection_confidence_0_1",
    "fm_face_detection_missing_mask",
    "texture_full_frame_high_frequency_0_1",
    "texture_face_crop_high_frequency_0_1",
    "texture_full_face_high_frequency_gap_0_1",
    "texture_full_frame_temporal_residual_0_1",
    "texture_face_crop_temporal_residual_0_1",
    "texture_full_face_temporal_residual_gap_0_1",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _finite(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _profile_signature(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _run(command: list[str]) -> None:
    print(">>", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _paths_from_exclusion(path: Path) -> set[str]:
    payload = _load_json(path)
    values: set[str] = set()
    for domain in ("real", "seedance"):
        for item in payload.get(domain, []):
            if isinstance(item, dict) and item.get("au"):
                values.add(str(project_path(item["au"]).resolve()).casefold())
    return values


def _build_profiles(args: argparse.Namespace) -> None:
    exclusion = project_path(args.profile_exclusion)
    output = project_path(args.forensics_profile)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_forensics_profiles.py"),
        "--real-au-root",
        "data/au/MD_CL",
        "--seedance-au-root",
        "data/au/WangXing_Seedance",
        "--real-video-root",
        "data/MD_CL",
        "--seedance-video-root",
        "data/WangXing_Seedance",
        "--holdout-manifest",
        str(exclusion),
        "--max-videos",
        str(args.profile_max_videos),
        "--max-frames",
        "32",
        "--sample-fps",
        "8",
        "--output",
        str(output),
    ]
    _run(command)

    source_profile = project_path(args.source_profile)
    source_manifest = project_path(
        "data/au/WangXing_Seedance/pseudo_expression_manifest.json"
    )
    source = build_source_profile(
        real_au_root=project_path("data/au/MD_CL"),
        seedance_label_manifest=source_manifest,
        output_path=source_profile,
        exclude_au_paths=_paths_from_exclusion(exclusion),
    )
    print(
        "source profile counts:",
        json.dumps(
            source.get("provenance", {}).get("sample_counts", {}),
            ensure_ascii=False,
        ),
        flush=True,
    )

    calibrator_output = project_path(args.calibrator)
    calibrator_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "calibrate_forensics.py"),
        "--profile",
        str(output),
        "--holdout-manifest",
        "data/forensics/holdout_split.json",
        "--output",
        str(calibrator_output),
        "--update-profile",
        str(output),
    ]
    _run(calibrator_command)


def _window_distribution(
    report: dict[str, Any],
    branch_name: str,
) -> dict[str, float]:
    branch = (report.get("branches") or {}).get(branch_name) or {}
    records = branch.get("window_records") or []
    scores = []
    for record in records:
        value = record.get(
            "evidence_score_0_1",
            record.get("anomaly_score_0_1"),
        )
        if value is not None:
            scores.append(_finite(value))
    if not scores:
        scores = [0.5]
    values = np.asarray(scores, dtype=np.float64)
    anomaly = values >= 0.60
    longest = 0
    current = 0
    for value in anomaly:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return {
        "mean": float(np.mean(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "std": float(np.std(values)),
        "anomaly_ratio": float(np.mean(anomaly)),
        "longest_anomaly_run": float(longest / max(len(values), 1)),
        "worst": float(np.max(values)),
        "aggregate": float(
            0.5 * np.mean(values) + 0.5 * np.max(values)
        ),
    }


def _quality_features(
    au_path: Path,
    report: dict[str, Any],
) -> dict[str, float]:
    facial_result = (report.get("branches") or {}).get(
        "facial_motion"
    ) or {}
    feature_record = facial_result.get("feature_record") or {}
    feature_map = feature_record.get("features") or {}
    au_ids = feature_record.get("au_ids") or []
    supported = feature_record.get("supported_au_ids") or []
    landmark_ratio = _finite(
        feature_map.get("landmark_valid_frame_ratio"),
        0.0,
    )
    pose_ratio = _finite(
        feature_map.get("pose_normalized_frame_ratio"),
        0.0,
    )
    landmark_available = bool(
        feature_record.get("landmark_available")
    )
    timestamps = feature_record.get("timestamps_seconds") or []
    timestamp_irregularity = 1.0
    if len(timestamps) >= 3:
        diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
        median = float(np.median(diffs))
        if median > 0 and np.all(np.isfinite(diffs)):
            timestamp_irregularity = min(
                1.0,
                float(np.std(diffs) / median),
            )
    detection_scores: list[float] = []
    try:
        with au_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                value = row.get("face_detection_score")
                if value is None:
                    continue
                parsed = _finite(value, math.nan)
                if math.isfinite(parsed):
                    detection_scores.append(
                        max(0.0, min(1.0, parsed))
                    )
    except (OSError, UnicodeError):
        detection_scores = []
    return {
        "fm_landmark_missing_mask": float(
            not landmark_available or landmark_ratio < 0.5
        ),
        "fm_pose_missing_mask": float(pose_ratio < 0.5),
        "fm_au_missing_mask": float(
            len(au_ids) == 0
            or len(supported) < len(au_ids)
        ),
        "fm_timestamp_irregularity_0_1": timestamp_irregularity,
        "fm_face_detection_confidence_0_1": (
            float(np.mean(detection_scores))
            if detection_scores
            else 0.5
        ),
        "fm_face_detection_missing_mask": float(
            not bool(detection_scores)
        ),
    }


def _full_frame_texture_features(
    video_path: Path,
    report: dict[str, Any],
    *,
    device: str,
) -> dict[str, float]:
    texture_result = (report.get("branches") or {}).get(
        "texture_detail"
    ) or {}
    face_features = (
        texture_result.get("feature_record") or {}
    ).get("features") or {}
    full = extract_texture_detail_features(
        video_path,
        max_frames=32,
        sample_fps=8.0,
        detect_faces=False,
        include_nr_vqa=False,
        include_frequency_forensics=True,
        device=device,
    )
    full_features = full.get("features") or {}
    full_high = _finite(
        full_features.get("high_frequency_ratio_mean"),
        0.5,
    )
    face_high = _finite(
        face_features.get("high_frequency_ratio_mean"),
        0.5,
    )
    full_residual = _finite(
        full_features.get("temporal_warp_residual_mean"),
        0.5,
    )
    face_residual = _finite(
        face_features.get("temporal_warp_residual_mean"),
        0.5,
    )
    return {
        "texture_full_frame_high_frequency_0_1": full_high,
        "texture_face_crop_high_frequency_0_1": face_high,
        "texture_full_face_high_frequency_gap_0_1": abs(
            full_high - face_high
        ),
        "texture_full_frame_temporal_residual_0_1": full_residual,
        "texture_face_crop_temporal_residual_0_1": face_residual,
        "texture_full_face_temporal_residual_gap_0_1": abs(
            full_residual - face_residual
        ),
    }


def _extra_features(
    report: dict[str, Any],
    *,
    au_path: Path,
    video_path: Path,
    device: str,
) -> dict[str, float]:
    branches = report.get("branches") or {}
    facial = branches.get("facial_motion") or {}
    texture = branches.get("texture_detail") or {}
    fusion = report.get("fusion") or {}
    summaries = report.get("window_summaries") or {}
    facial_distribution = _window_distribution(
        report,
        "facial_motion",
    )
    texture_distribution = _window_distribution(
        report,
        "texture_detail",
    )

    def branch_value(name: str, key: str, default: float = 0.5) -> float:
        value = (branches.get(name) or {}).get(key)
        if value is None:
            value = (branches.get(name) or {}).get("metrics", {}).get(key)
        return _finite(value, default)

    def window_value(name: str, key: str) -> float:
        return _finite(
            ((summaries.get(name) or {}).get("summary") or {}).get(key),
            0.5,
        )

    return {
        "texture_raw_real_domain_evidence_0_1": branch_value(
            "texture_detail",
            "raw_real_domain_evidence_0_1",
        ),
        "texture_real_domain_fit_0_1": branch_value(
            "texture_detail",
            "real_domain_fit_0_1",
        ),
        "texture_seedance_domain_fit_0_1": branch_value(
            "texture_detail",
            "seedance_domain_fit_0_1",
        ),
        "texture_stability_0_1": branch_value(
            "texture_detail",
            "temporal_stability_proxy_0_1",
        ),
        "texture_flicker_0_1": branch_value(
            "texture_detail",
            "texture_flicker_0_1",
        ),
        "texture_micro_temporal_0_1": branch_value(
            "texture_detail",
            "micro_temporal_naturalness_0_1",
        ),
        "texture_frequency_0_1": branch_value(
            "texture_detail",
            "freq_forensics_score_0_1",
        ),
        "texture_nr_vqa_0_1": branch_value(
            "texture_detail",
            "nr_vqa_score_0_1",
        ),
        "texture_detail_quality_0_1": branch_value(
            "texture_detail",
            "detail_quality_proxy_0_1",
        ),
        "texture_face_box_coverage_0_1": branch_value(
            "texture_detail",
            "face_box_coverage",
        ),
        "fusion_confidence_0_1": _finite(
            fusion.get("confidence_0_1")
        ),
        "fusion_training_free_prior_0_1": _finite(
            fusion.get("training_free_prior_0_1")
        ),
        "facial_window_mean_0_1": window_value(
            "facial_expression_muscle",
            "mean_evidence_score_0_1",
        ),
        "facial_window_worst_0_1": window_value(
            "facial_expression_muscle",
            "worst_evidence_score_0_1",
        ),
        "facial_window_aggregate_0_1": window_value(
            "facial_expression_muscle",
            "aggregate_evidence_score_0_1",
        ),
        "texture_window_mean_0_1": window_value(
            "texture_detail",
            "mean_evidence_score_0_1",
        ),
        "texture_window_worst_0_1": window_value(
            "texture_detail",
            "worst_evidence_score_0_1",
        ),
        "texture_window_aggregate_0_1": window_value(
            "texture_detail",
            "aggregate_evidence_score_0_1",
        ),
        "facial_window_p90_0_1": facial_distribution["p90"],
        "facial_window_p95_0_1": facial_distribution["p95"],
        "facial_window_std_0_1": facial_distribution["std"],
        "facial_window_anomaly_ratio_0_1": facial_distribution[
            "anomaly_ratio"
        ],
        "facial_window_longest_anomaly_run_0_1": facial_distribution[
            "longest_anomaly_run"
        ],
        "texture_window_p90_0_1": texture_distribution["p90"],
        "texture_window_p95_0_1": texture_distribution["p95"],
        "texture_window_std_0_1": texture_distribution["std"],
        "texture_window_anomaly_ratio_0_1": texture_distribution[
            "anomaly_ratio"
        ],
        "texture_window_longest_anomaly_run_0_1": texture_distribution[
            "longest_anomaly_run"
        ],
        **_quality_features(au_path, report),
        **_full_frame_texture_features(
            video_path,
            report,
            device=device,
        ),
    }


def _feature_vector(
    *,
    au_path: Path,
    video_path: Path,
    source_profile: dict[str, Any],
    profiles: dict[str, Any],
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    report = analyze_forensics(
        facial_motion=au_path,
        facial_motion_profile=profiles.get("facial_motion"),
        texture_detail=video_path,
        texture_detail_profile=profiles.get("texture_detail"),
        authenticity_calibrator=profiles.get("authenticity_calibrator"),
        max_frames=32,
        sample_fps=8.0,
        device=device,
    )
    au_features = extract_fusion_feature_dict(
        au_path=au_path,
        wangxing_source_profile=source_profile,
        forensics_profiles=profiles,
    )
    merged = {
        **au_features,
        **_extra_features(
            report,
            au_path=au_path,
            video_path=video_path,
            device=device,
        ),
    }
    vector = np.asarray(
        [_finite(merged.get(name)) for name in FEATURE_NAMES],
        dtype=np.float64,
    )
    return vector, report


def _train_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    groups: list[str],
    seed: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    if len(groups) != len(labels):
        raise ValueError("Fusion labels and groups are misaligned.")
    splitter = GroupShuffleSplit(
        n_splits=20,
        test_size=0.20,
        random_state=seed,
    )
    fit_idx = val_idx = None
    for candidate_fit, candidate_val in splitter.split(
        features,
        labels,
        groups,
    ):
        if (
            len(np.unique(labels[candidate_fit])) == 2
            and len(np.unique(labels[candidate_val])) == 2
        ):
            fit_idx, val_idx = candidate_fit, candidate_val
            break
    if fit_idx is None or val_idx is None:
        raise ValueError(
            "Unable to create a two-class source-group validation split."
        )
    x_fit, x_val = features[fit_idx], features[val_idx]
    y_fit, y_val = labels[fit_idx], labels[val_idx]
    scaler = StandardScaler()
    x_fit_scaled = scaler.fit_transform(x_fit)
    x_val_scaled = scaler.transform(x_val)
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(x_fit_scaled, y_fit)
    val_p_gen = model.predict_proba(x_val_scaled)[:, 1]
    best = None
    for threshold_step in range(20, 81):
        threshold = threshold_step / 100.0
        predicted = (val_p_gen >= threshold).astype(np.int32)
        tp = int(((y_val == 1) & (predicted == 1)).sum())
        tn = int(((y_val == 0) & (predicted == 0)).sum())
        fp = int(((y_val == 0) & (predicted == 1)).sum())
        fn = int(((y_val == 1) & (predicted == 0)).sum())
        generated_recall = tp / (tp + fn) if tp + fn else 0.0
        real_recall = tn / (tn + fp) if tn + fp else 0.0
        accuracy = (tp + tn) / len(y_val)
        score = min(generated_recall, real_recall)
        candidate = (score, accuracy, generated_recall, threshold)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return {
        "schema_version": "web_forensics_fusion_v2",
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "coef": model.coef_.reshape(-1).astype(float).tolist(),
        "intercept": float(model.intercept_[0]),
        "threshold_generated": float(best[3]),
        "validation_metrics": {
            "min_class_recall": float(best[0]),
            "accuracy": float(best[1]),
            "generated_recall": float(best[2]),
            "sample_count": int(len(y_val)),
            "train_group_count": int(len(set(groups[index] for index in fit_idx))),
            "validation_group_count": int(len(set(groups[index] for index in val_idx))),
            "split_protocol": "source_group_shuffle_split",
        },
        "train_counts": {
            "real": int((y_fit == 0).sum()),
            "generated": int((y_fit == 1).sum()),
        },
        "manual_scores_required": False,
        "uncertain_band_used": False,
    }


def _predict_fusion(vector: np.ndarray, head: dict[str, Any]) -> dict[str, Any]:
    mean = np.asarray(head["scaler_mean"], dtype=np.float64)
    scale = np.maximum(
        np.asarray(head["scaler_scale"], dtype=np.float64),
        1e-8,
    )
    scaled = (vector - mean) / scale
    logit = float(
        np.dot(scaled, np.asarray(head["coef"], dtype=np.float64))
        + float(head["intercept"])
    )
    p_gen = float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit)))))
    threshold = float(head["threshold_generated"])
    return {
        "logit": logit,
        "generated_probability": p_gen,
        "real_probability": 1.0 - p_gen,
        "threshold_generated": threshold,
        "prediction": "generated" if p_gen >= threshold else "real",
    }


def _train_paths(args: argparse.Namespace) -> list[tuple[int, Path, Path]]:
    split = _load_json(project_path(args.split_manifest))
    excluded = {
        str(project_path(item["video"]).resolve()).casefold()
        for domain in ("real", "seedance")
        for item in _load_json(
            project_path(args.profile_exclusion)
        ).get(domain, [])
        if item.get("video")
    }
    rows: list[tuple[int, Path, Path]] = []
    for label, key, au_root in (
        (0, "real", project_path("data/au/MD_CL")),
        (1, "fake", project_path("data/au/WangXing_Seedance")),
    ):
        for value in split.get("train", {}).get(key, []):
            video = project_path(value)
            if video.stem.endswith("_le1024") or str(video.resolve()).casefold() in excluded:
                continue
            au = (
                au_root / video.name
            ).with_suffix(".csv") if label else (
                au_root
                / video.relative_to(project_path("data/MD_CL"))
            ).with_suffix(".csv")
            if video.is_file() and au.is_file():
                rows.append((label, video, au))
    return rows


def cmd_prepare(args: argparse.Namespace) -> None:
    _run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "scripts"
                / "数据构建"
                / "build_web_forensics_v2_dataset.py"
            ),
            "--output-root",
            args.dataset_root,
            "--split-manifest",
            args.split_manifest,
            "--seed",
            str(args.seed),
        ]
    )


def cmd_profiles(args: argparse.Namespace) -> None:
    _build_profiles(args)


def cmd_train(args: argparse.Namespace) -> None:
    profile_path = project_path(args.forensics_profile)
    source_path = project_path(args.source_profile)
    profiles = _load_json(profile_path)
    source = _load_json(source_path)
    rows = _train_paths(args)
    if len(rows) < 8 or len({label for label, _, _ in rows}) < 2:
        raise SystemExit("Insufficient web-forensics fusion training rows.")
    cache_path = project_path(args.feature_cache)
    cached: dict[str, np.ndarray] = {}
    if cache_path.is_file():
        with np.load(str(cache_path), allow_pickle=True) as payload:
            cached = {
                str(path): payload["features"][index]
                for index, path in enumerate(payload["paths"].tolist())
            }
    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    groups: list[str] = []
    for index, (label, video, au) in enumerate(rows, start=1):
        key = str(au.resolve())
        if key in cached:
            vector = np.asarray(cached[key], dtype=np.float64)
        else:
            vector, _ = _feature_vector(
                au_path=au,
                video_path=video,
                source_profile=source,
                profiles=profiles,
                device=args.device,
            )
            cached[key] = vector
        features.append(vector)
        labels.append(label)
        paths.append(key)
        groups.append(video.stem.casefold())
        if index % 10 == 0 or index == len(rows):
            print(f"[web fusion features] {index}/{len(rows)}", flush=True)
    matrix = np.stack(features)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(cache_path),
        paths=np.asarray(paths, dtype=object),
        features=matrix,
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
    )
    head = _train_logistic(
        matrix,
        np.asarray(labels),
        groups=groups,
        seed=args.seed,
    )
    head.update(
        {
            "forensics_profile": str(profile_path),
            "source_profile": str(source_path),
            "profile_exclusion": str(project_path(args.profile_exclusion)),
            "feature_cache": str(cache_path),
            "device": args.device,
        }
    )
    _write_json(project_path(args.fusion_head), head)
    print(json.dumps(head["validation_metrics"], ensure_ascii=False, indent=2))
    print(f"Wrote {project_path(args.fusion_head)}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    manifest_path = project_path(args.manifest)
    manifest = _load_json(manifest_path)
    profiles_path = project_path(args.forensics_profile)
    source_path = project_path(args.source_profile)
    profiles = _load_json(profiles_path)
    source = _load_json(source_path)
    head = _load_json(project_path(args.fusion_head))
    authenticity_policy = None
    if args.authenticity_policy:
        authenticity_policy = _load_json(
            project_path(args.authenticity_policy)
        )
    identity, expression, _source_identity = (
        resolve_profile("wangxing_identity_profile.json", required=True),
        resolve_profile("wangxing_expression_profile.json", required=True),
        resolve_profile("wangxing_source_profile.json", required=False),
    )
    results: list[dict[str, Any]] = []
    for index, sample in enumerate(manifest.get("samples", []), start=1):
        print(f"[web fusion eval] {index}/{len(manifest['samples'])}", flush=True)
        video = manifest_path.parent / sample["video"]
        au = manifest_path.parent / sample["au"]
        vector, report = _feature_vector(
            au_path=au,
            video_path=video,
            source_profile=source,
            profiles=profiles,
            device=args.device,
        )
        fusion = _predict_fusion(vector, head)
        fusion["decision"] = fusion["prediction"]
        fusion["conclusion"] = (
            "偏向 AI 生成"
            if fusion["prediction"] == "generated"
            else "偏向真实拍摄"
        )
        fusion["detail"] = (
            f"网页融合器真实拍摄概率为 "
            f"{fusion['real_probability'] * 100.0:.1f}%。"
        )
        # Reuse the webpage-shaped identity/forensics payload, with the new
        # web fusion decision attached separately.
        card_result = _run_one(
            sample=sample,
            manifest_root=manifest_path.parent,
            forensics_profiles=profiles,
            identity_profile=identity,
            expression_profile=expression,
            source_profile=source_path,
            max_frames=32,
            sample_fps=8.0,
            forensics_device=args.device,
            wangxing_device=args.wangxing_device,
            include_wangxing=True,
            precomputed_forensics=report,
        )
        card_result["web_fusion"] = fusion
        card_result["web_fusion"]["raw_feature_count"] = len(vector)
        if authenticity_policy:
            policy_result = apply_policy(
                card_result,
                authenticity_policy["policy"],
            )
            card_result["web_policy"] = policy_result
        card_result.setdefault("web_card", {})[
            "optimized_forensics"
        ] = {
            "conclusion": (
                "偏向 AI 生成"
                if (
                    card_result.get("web_policy", fusion)["prediction"]
                    == "generated"
                )
                else "偏向真实拍摄"
            ),
            "detail": (
                f"网页真实性策略真实拍摄概率为 "
                f"{card_result.get('web_policy', fusion)['real_probability'] * 100.0:.1f}%。"
            ),
            "real_probability": card_result.get(
                "web_policy", fusion
            )["real_probability"],
            "generated_probability": card_result.get(
                "web_policy", fusion
            )["generated_probability"],
            "threshold_generated": card_result.get(
                "web_policy", fusion
            )["threshold_generated"],
        }
        results.append(card_result)
    output = project_path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    labels = [int(row.get("label_generated", 0)) for row in results]
    predictions = [
        int(
            row.get("web_policy", row["web_fusion"])["prediction"]
            == "generated"
        )
        for row in results
    ]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    summary = {
        "sample_count": len(results),
        "real_count": sum(label == 0 for label in labels),
        "ai_count": sum(label == 1 for label in labels),
        "generated_recall": tp / (tp + fn) if tp + fn else None,
        "real_recall": tn / (tn + fp) if tn + fp else None,
        "overall_accuracy": (tp + tn) / len(labels) if labels else None,
        "confusion": {
            "tp_generated": tp,
            "tn_real": tn,
            "fp_real_as_generated": fp,
            "fn_generated_as_real": fn,
        },
    }
    _write_json(
        output / "all_results.json",
        {
            "schema_version": "web_forensics_v2_results_v1",
            "fusion_head": str(project_path(args.fusion_head)),
            "authenticity_policy": (
                str(project_path(args.authenticity_policy))
                if args.authenticity_policy
                else None
            ),
            "forensics_profile": str(profiles_path),
            "source_profile": str(source_path),
            "summary": summary,
            "results": results,
        },
    )
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"All results: {output / 'all_results.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Web-only forensics v2 prepare/train/evaluate pipeline."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--dataset-root",
            default="data/test/web_forensics_v2",
        )
        command.add_argument(
            "--split-manifest",
            default="outputs/vedio_pred/wangxing_dual_pt_split_res1k.json",
        )
        command.add_argument("--seed", type=int, default=42)

    prepare = sub.add_parser("prepare")
    common(prepare)
    prepare.set_defaults(func=cmd_prepare)

    profiles = sub.add_parser("build-profiles")
    profiles.add_argument(
        "--profile-exclusion",
        default="data/forensics/web_forensics_v2_profile_exclusion.json",
    )
    profiles.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles_web_v2_test_excluded.json",
    )
    profiles.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_web_v2_test_excluded.json",
    )
    profiles.add_argument(
        "--calibrator",
        default="outputs/forensics/forensics_authenticity_calibrator_web_v2.json",
    )
    profiles.add_argument("--profile-max-videos", type=int, default=120)
    profiles.set_defaults(func=cmd_profiles)

    train = sub.add_parser("train")
    train.add_argument(
        "--split-manifest",
        default="outputs/vedio_pred/wangxing_dual_pt_split_res1k.json",
    )
    train.add_argument(
        "--profile-exclusion",
        default="data/forensics/web_forensics_v2_profile_exclusion.json",
    )
    train.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles_web_v2_test_excluded.json",
    )
    train.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_web_v2_test_excluded.json",
    )
    train.add_argument(
        "--fusion-head",
        default="outputs/forensics/web_forensics_fusion_v2.json",
    )
    train.add_argument(
        "--feature-cache",
        default="outputs/forensics/web_forensics_v2_feature_cache.npz",
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument(
        "--manifest",
        default="data/test/web_forensics_v2/single_video/manifest.json",
    )
    evaluate.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles_web_v2_test_excluded.json",
    )
    evaluate.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_web_v2_test_excluded.json",
    )
    evaluate.add_argument(
        "--fusion-head",
        default="outputs/forensics/web_forensics_fusion_v2.json",
    )
    evaluate.add_argument(
        "--output-root",
        default="outputs/forensics/web_forensics_v2_results",
    )
    evaluate.add_argument(
        "--authenticity-policy",
        default=None,
        help=(
            "Optional development-fitted policy that keeps identity out of "
            "the authenticity probability."
        ),
    )
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--wangxing-device", choices=("cpu", "cuda"), default="cuda")
    evaluate.set_defaults(func=cmd_evaluate)

    all_command = sub.add_parser("all")
    common(all_command)
    all_command.add_argument(
        "--manifest",
        default="data/test/web_forensics_v2/single_video/manifest.json",
    )
    all_command.add_argument(
        "--profile-exclusion",
        default="data/forensics/web_forensics_v2_profile_exclusion.json",
    )
    all_command.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles_web_v2_test_excluded.json",
    )
    all_command.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_web_v2_test_excluded.json",
    )
    all_command.add_argument(
        "--calibrator",
        default="outputs/forensics/forensics_authenticity_calibrator_web_v2.json",
    )
    all_command.add_argument(
        "--profile-max-videos",
        type=int,
        default=120,
    )
    all_command.add_argument(
        "--fusion-head",
        default="outputs/forensics/web_forensics_fusion_v2.json",
    )
    all_command.add_argument(
        "--feature-cache",
        default="outputs/forensics/web_forensics_v2_feature_cache.npz",
    )
    all_command.add_argument("--device", default="cuda")
    all_command.add_argument("--wangxing-device", choices=("cpu", "cuda"), default="cuda")
    all_command.add_argument(
        "--output-root",
        default="outputs/forensics/web_forensics_v2_results",
    )
    all_command.add_argument(
        "--authenticity-policy",
        default=None,
    )

    def cmd_all(args: argparse.Namespace) -> None:
        cmd_prepare(args)
        cmd_profiles(args)
        cmd_train(args)
        cmd_evaluate(args)

    all_command.set_defaults(func=cmd_all)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

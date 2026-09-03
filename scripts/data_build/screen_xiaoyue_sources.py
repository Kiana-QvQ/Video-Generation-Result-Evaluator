"""Screen XiaoYue sources without modifying the original media.

The script samples video frames for frontal-face completeness and normalizes
BlendShape images into an ASCII-path RGB reference directory. It writes
manifests only; downstream training must use the accepted video paths and
must keep test/reference material excluded.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bbox(face: Any) -> tuple[float, float, float, float] | None:
    values = np.asarray(getattr(face, "bbox", None), dtype=np.float32).reshape(-1)
    if len(values) < 4 or not np.isfinite(values[:4]).all():
        return None
    return tuple(float(value) for value in values[:4])


def _keypoints(face: Any) -> np.ndarray | None:
    values = np.asarray(getattr(face, "kps", None), dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 5 or values.shape[1] < 2:
        return None
    values = values[:5, :2]
    return values if np.isfinite(values).all() else None


def _frame_quality(frame: np.ndarray, detector: Any) -> dict[str, Any]:
    height, width = frame.shape[:2]
    faces = detector.get(frame)
    if not faces:
        return {
            "keep": False,
            "reason": "no_face",
            "detection_score": 0.0,
            "keypoint_coverage": 0.0,
            "face_area_ratio": 0.0,
            "border_margin_ratio": 0.0,
            "symmetry_score": 0.0,
            "eye_separation_ratio": 0.0,
        }
    face = max(
        faces,
        key=lambda item: max(
            0.0,
            _finite(np.asarray(getattr(item, "bbox", [0, 0, 0, 0]))[2])
            - _finite(np.asarray(getattr(item, "bbox", [0, 0, 0, 0]))[0]),
        )
        * max(
            0.0,
            _finite(np.asarray(getattr(item, "bbox", [0, 0, 0, 0]))[3])
            - _finite(np.asarray(getattr(item, "bbox", [0, 0, 0, 0]))[1]),
        ),
    )
    box = _bbox(face)
    points = _keypoints(face)
    if box is None or points is None:
        return {
            "keep": False,
            "reason": "invalid_face_geometry",
            "detection_score": _clamp(_finite(getattr(face, "det_score", 0.0))),
            "keypoint_coverage": 0.0,
            "face_area_ratio": 0.0,
            "border_margin_ratio": 0.0,
            "symmetry_score": 0.0,
            "eye_separation_ratio": 0.0,
        }
    x0, y0, x1, y1 = box
    box_width = max(0.0, x1 - x0)
    box_height = max(0.0, y1 - y0)
    face_area_ratio = box_width * box_height / max(width * height, 1)
    border_pixels = min(x0, y0, width - x1, height - y1)
    border_margin_ratio = max(
        0.0,
        border_pixels / max(box_width, box_height, 1.0),
    )
    inside = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < float(width))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < float(height))
    )
    keypoint_coverage = float(np.mean(inside))
    nose = points[2]
    left = float(
        np.mean(
            [
                np.linalg.norm(nose - points[0]),
                np.linalg.norm(nose - points[3]),
            ]
        )
    )
    right = float(
        np.mean(
            [
                np.linalg.norm(nose - points[1]),
                np.linalg.norm(nose - points[4]),
            ]
        )
    )
    symmetry_score = (
        _clamp(min(left, right) / max(left, right))
        if min(left, right) > 1e-6
        else 0.0
    )
    eye_separation_ratio = _clamp(
        float(np.linalg.norm(points[0] - points[1]))
        / max(box_width, 1.0)
    )
    detection_score = _clamp(_finite(getattr(face, "det_score", 0.0)))
    checks = (
        ("low_detection_score", detection_score >= 0.60),
        ("incomplete_keypoints", keypoint_coverage >= 1.0),
        ("face_too_small", face_area_ratio >= 0.01),
        ("face_touches_border", border_margin_ratio >= 0.01),
        (
            "face_too_profile",
            symmetry_score >= 0.75 and eye_separation_ratio >= 0.30,
        ),
    )
    reasons = [name for name, passed in checks if not passed]
    return {
        "keep": not reasons,
        "reason": reasons[0] if reasons else None,
        "reasons": reasons,
        "detection_score": detection_score,
        "keypoint_coverage": keypoint_coverage,
        "face_area_ratio": face_area_ratio,
        "border_margin_ratio": border_margin_ratio,
        "symmetry_score": symmetry_score,
        "eye_separation_ratio": eye_separation_ratio,
    }


def _sample_indices(frame_count: int, samples: int) -> list[int]:
    if frame_count <= 0:
        return list(range(samples))
    count = min(max(1, samples), frame_count)
    return sorted(
        {
            int(round(index))
            for index in np.linspace(0, frame_count - 1, count)
        }
    )


def _video_audit(path: Path, detector: Any, samples: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {
            "status": "error",
            "error": "video_open_failed",
            "video": str(path),
        }
    fps = _finite(capture.get(cv2.CAP_PROP_FPS), 0.0)
    frame_count = int(round(_finite(capture.get(cv2.CAP_PROP_FRAME_COUNT), 0.0)))
    width = int(round(_finite(capture.get(cv2.CAP_PROP_FRAME_WIDTH), 0.0)))
    height = int(round(_finite(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), 0.0)))
    selected = _sample_indices(frame_count, samples)
    audits: list[dict[str, Any]] = []
    for index in selected:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            audits.append({"keep": False, "reason": "frame_read_failed"})
            continue
        audits.append(_frame_quality(frame, detector))
    capture.release()
    kept = sum(bool(item.get("keep")) for item in audits)
    reasons: dict[str, int] = {}
    for item in audits:
        for reason in item.get("reasons") or [item.get("reason")]:
            if reason:
                reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    ratio = kept / max(len(audits), 1)
    decision = (
        "keep_candidate"
        if ratio >= 0.80
        else "manual_review"
        if ratio >= 0.50
        else "exclude_candidate"
    )
    numeric_fields = (
        "detection_score",
        "keypoint_coverage",
        "face_area_ratio",
        "border_margin_ratio",
        "symmetry_score",
        "eye_separation_ratio",
    )
    means = {
        field: float(
            np.mean(
                [
                    _finite(item.get(field), 0.0)
                    for item in audits
                ]
            )
        )
        for field in numeric_fields
    }
    return {
        "status": "ok",
        "video": str(path),
        "video_meta": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
        },
        "sample_count": len(audits),
        "kept_sample_count": kept,
        "sample_keep_ratio": ratio,
        "decision": decision,
        "mean_metrics": means,
        "reason_counts": reasons,
        "sampled_frames": audits,
    }


def _iter_video_roots(root: Path) -> Iterable[tuple[Path, str]]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        parts = {part.casefold() for part in path.parts}
        if "reference" in parts and not any(
            part.casefold() in {"real", "real2"} for part in path.parts
        ):
            continue
        source_kind = "hk_real" if "hk" in parts else "reference_real"
        yield path, source_kind


def _normalize_images(
    source: Path,
    destination: Path,
    detector: Any,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    image_paths = [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    for index, image_path in enumerate(image_paths, start=1):
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        if index == 1 or index % 25 == 0 or index == len(image_paths):
            print(
                f"[xiaoyue blendshape] {index}/{len(image_paths)}",
                flush=True,
            )
        relative = image_path.relative_to(source)
        target = destination / relative.parent / f"{relative.stem}.jpg"
        try:
            image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
            frame = np.asarray(image)[:, :, ::-1].copy()
            audit = _frame_quality(frame, detector)
            if not audit.get("keep"):
                rows.append(
                    {
                        "source": str(image_path),
                        "status": "exclude_candidate",
                        "reason_counts": audit.get("reasons") or [
                            audit.get("reason")
                        ],
                    }
                )
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, format="JPEG", quality=95, optimize=True)
            rows.append(
                {
                    "source": str(image_path),
                    "normalized": str(target),
                    "width": image.width,
                    "height": image.height,
                    "status": "keep_candidate",
                    "metrics": {
                        key: audit.get(key)
                        for key in (
                            "detection_score",
                            "face_area_ratio",
                            "symmetry_score",
                            "eye_separation_ratio",
                        )
                    },
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "source": str(image_path),
                    "normalized": str(target),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "source": str(source),
        "destination": str(destination),
        "count": len(rows),
        "success_count": sum(
            row["status"] == "keep_candidate" for row in rows
        ),
        "excluded_count": sum(
            row["status"] == "exclude_candidate" for row in rows
        ),
        "error_count": sum(row["status"] == "error" for row in rows),
        "rows": rows,
    }


def _build_detector(model_root: Path, det_size: int) -> Any:
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise SystemExit("insightface is required for XiaoYue screening.") from exc
    detector = FaceAnalysis(
        name="buffalo_l",
        root=str(model_root),
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
    )
    detector.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return detector


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="data/xiaoyue/source")
    parser.add_argument("--blendshape-root", default="data/xiaoyue/source/blendshape")
    parser.add_argument(
        "--normalized-blendshape-root",
        default="data/xiaoyue/processed/blendshape_front",
    )
    parser.add_argument(
        "--output",
        default="outputs/xiaoyue_screening/source_quality_manifest.json",
    )
    parser.add_argument("--samples-per-video", type=int, default=8)
    parser.add_argument("--det-size", type=int, default=320)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--skip-blendshape",
        action="store_true",
        help="Only audit videos; skip BlendShape image screening.",
    )
    parser.add_argument(
        "--blendshape-only",
        action="store_true",
        help="Only screen/normalize BlendShape images; skip videos.",
    )
    parser.add_argument("--model-root", default="model_cache/insightface")
    args = parser.parse_args(argv)
    if args.samples_per_video < 1:
        raise SystemExit("--samples-per-video must be positive.")
    source_root = PROJECT_ROOT / args.source_root
    videos = [] if args.blendshape_only else list(_iter_video_roots(source_root))
    if args.limit > 0:
        videos = videos[: args.limit]
    if not videos and not args.blendshape_only:
        raise SystemExit(f"No source videos found under {source_root}")
    detector = _build_detector(PROJECT_ROOT / args.model_root, args.det_size)
    rows: list[dict[str, Any]] = []
    for index, (video, source_kind) in enumerate(videos, start=1):
        print(f"[xiaoyue screen] {index}/{len(videos)} {video}", flush=True)
        audit = _video_audit(video, detector, args.samples_per_video)
        audit["source_kind"] = source_kind
        rows.append(audit)
    blendshape = (
        {
            "status": "skipped",
            "count": 0,
            "success_count": 0,
            "excluded_count": 0,
            "error_count": 0,
        }
        if args.skip_blendshape
        else _normalize_images(
            source_root / "blendshape"
            if args.blendshape_root == "data/xiaoyue/source/blendshape"
            else PROJECT_ROOT / args.blendshape_root,
            PROJECT_ROOT / args.normalized_blendshape_root,
            detector,
        )
    )
    counts = {
        decision: sum(row.get("decision") == decision for row in rows)
        for decision in ("keep_candidate", "manual_review", "exclude_candidate")
    }
    payload = {
        "schema_version": "xiaoyue_source_quality_manifest_v1",
        "screening_policy": {
            "backend": "insightface_buffalo_l_cpu",
            "samples_per_video": args.samples_per_video,
            "det_size": args.det_size,
            "keep_threshold": 0.80,
            "manual_review_threshold": 0.50,
            "originals_modified": False,
        },
        "source_root": str(source_root.resolve()),
        "video_count": len(rows),
        "decision_counts": counts,
        "videos": rows,
        "blendshape": {
            "count": blendshape["count"],
            "success_count": blendshape["success_count"],
            "error_count": blendshape["error_count"],
            "normalized_root": str(
                (PROJECT_ROOT / args.normalized_blendshape_root).resolve()
            ),
        },
        "training_note": (
            "This is a quality manifest only. Test/reference videos are not "
            "included because the source scan is restricted to hk and "
            "reference/real or reference/real2."
        ),
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "video_count": len(rows),
                "decision_counts": counts,
                "blendshape": payload["blendshape"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

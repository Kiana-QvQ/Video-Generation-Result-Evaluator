"""Probe frozen V3 dependence on face versus full-frame evidence.

This experiment does not train or alter the production checkpoint. It uses
the same sampled frames and AU vector as V3 inference, then transforms only
the video frames before V3 feature extraction:

- original: unchanged frame sequence;
- face_only: retain the landmark-derived face region, neutralize background;
- background_only: neutralize the landmark-derived face region;
- low_texture: blur each frame while preserving global composition.

The report is diagnostic, not a replacement decision policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from evaluator.modules.forensics.learned_fusion_head import extract_fusion_features
from evaluator.vedio_pred.wangxing_dual_pt import SCALE_A, SCALE_B
from evaluator.vedio_pred.real_video_detector import _frame_feature, _read_sampled_frames
from wangxing_project.joint_au_pt_v3 import (
    TEMPORAL_DIM,
    _model_from_checkpoint,
    _normalize_features,
)

MODES = (
    "original",
    "face_only",
    "background_only",
    "low_texture",
    "face_blur",
    "background_blur",
)


def _finite(value: str | None) -> float | None:
    try:
        parsed = float(value or "")
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _sample_rows(au_path: Path, count: int) -> list[dict[str, str]]:
    with au_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return [{} for _ in range(count)]
    indexes = np.linspace(0, len(rows) - 1, count).round().astype(int)
    return [rows[int(index)] for index in indexes]


def _face_box(
    row: dict[str, str],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    points: list[tuple[float, float]] = []
    for name, value in row.items():
        if not name.startswith("lm_mp_") or not name.endswith("_x"):
            continue
        prefix = name[:-2]
        x = _finite(value)
        y = _finite(row.get(f"{prefix}_y"))
        if x is not None and y is not None:
            points.append((x * width, y * height))
    if len(points) < 12:
        return None
    values = np.asarray(points, dtype=np.float32)
    low = values.min(axis=0)
    high = values.max(axis=0)
    span = max(float(np.max(high - low)), 8.0)
    margin = max(0.30 * span, 12.0)
    x0 = max(0, int(low[0] - margin))
    y0 = max(0, int(low[1] - margin))
    x1 = min(width, int(high[0] + margin))
    y1 = min(height, int(high[1] + margin))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def _neutral_canvas(frame: np.ndarray) -> np.ndarray:
    value = np.mean(frame, axis=(0, 1), keepdims=True)
    return np.broadcast_to(value, frame.shape).astype(frame.dtype).copy()


def _transform(
    frame: np.ndarray,
    row: dict[str, str],
    mode: str,
) -> tuple[np.ndarray, bool]:
    if mode == "original":
        return frame, True
    if mode == "low_texture":
        return cv2.GaussianBlur(frame, (0, 0), sigmaX=5.0), True
    height, width = frame.shape[:2]
    box = _face_box(row, width=width, height=height)
    if box is None:
        return frame, False
    x0, y0, x1, y1 = box
    if mode == "face_only":
        transformed = _neutral_canvas(frame)
        transformed[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
        return transformed, True
    if mode == "background_only":
        transformed = frame.copy()
        transformed[y0:y1, x0:x1] = _neutral_canvas(frame)[y0:y1, x0:x1]
        return transformed, True
    if mode == "face_blur":
        transformed = frame.copy()
        transformed[y0:y1, x0:x1] = cv2.GaussianBlur(
            frame[y0:y1, x0:x1],
            (0, 0),
            sigmaX=5.0,
        )
        return transformed, True
    if mode == "background_blur":
        transformed = cv2.GaussianBlur(frame, (0, 0), sigmaX=5.0)
        transformed[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
        return transformed, True
    raise ValueError(f"Unsupported mode: {mode}")


def _temporal_features(frames: list[np.ndarray]) -> np.ndarray:
    values: list[np.ndarray] = []
    previous: np.ndarray | None = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if previous is not None:
            difference = np.abs(gray - previous)
            values.append(
                np.asarray(
                    [
                        float(np.mean(difference)),
                        float(np.std(difference)),
                        float(np.percentile(difference, 95)),
                        float(np.mean(difference > 0.12)),
                        float(np.mean(np.abs(np.diff(difference, axis=0)))),
                        float(np.mean(np.abs(np.diff(difference, axis=1)))),
                    ],
                    dtype=np.float32,
                )
            )
        previous = gray
    return np.stack(values).astype(np.float32)


def _sequence(
    *,
    video: Path,
    au: Path,
    count: int,
    frame_size: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    frames = _read_sampled_frames(
        video_path=video,
        num_frames=count,
        frame_size=frame_size,
    )
    rows = _sample_rows(au, len(frames))
    transformed: list[np.ndarray] = []
    usable = 0
    for frame, row in zip(frames, rows):
        output, valid = _transform(frame, row, mode)
        transformed.append(output)
        usable += int(valid)
    return (
        np.stack([_frame_feature(frame) for frame in transformed]).astype(np.float32),
        _temporal_features(transformed),
        usable / max(len(transformed), 1),
    )


def _predict(
    *,
    checkpoint: dict[str, Any],
    video: Path,
    au: Path,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    frame_a, temporal_a, coverage_a = _sequence(
        video=video,
        au=au,
        count=int(SCALE_A["num_frames"]),
        frame_size=int(SCALE_A["frame_size"]),
        mode=mode,
    )
    frame_b, temporal_b, coverage_b = _sequence(
        video=video,
        au=au,
        count=int(SCALE_B["num_frames"]),
        frame_size=int(SCALE_B["frame_size"]),
        mode=mode,
    )
    au_vector, _ = extract_fusion_features(
        au_path=au,
        wangxing_source_profile=source_profile,
        forensics_profiles=forensics_profiles,
    )
    raw = {
        "frame_a": frame_a[None, ...],
        "temporal_a": temporal_a[None, ...],
        "frame_b": frame_b[None, ...],
        "temporal_b": temporal_b[None, ...],
        "au": np.asarray(au_vector, dtype=np.float32)[None, ...],
    }
    stats = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in checkpoint["stats"].items()
    }
    normalized = _normalize_features(raw, stats)
    model = _model_from_checkpoint(checkpoint)
    tensors = [
        torch.from_numpy(normalized[name])
        for name in ("frame_a", "temporal_a", "frame_b", "temporal_b", "au")
    ]
    with torch.no_grad():
        logit = float(model(*tensors)[0].item())
    temperature = float(checkpoint.get("temperature", 1.0))
    p_gen = float(1.0 / (1.0 + math.exp(-logit / max(temperature, 1e-6))))
    return {
        "generated_probability": p_gen,
        "prediction": "generated" if p_gen >= 0.5 else "real",
        "face_transform_coverage": (coverage_a + coverage_b) / 2.0,
    }


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(project_path(path).read_text(encoding="utf-8-sig"))


def _samples(manifest: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in manifest.get("samples") or []:
        rows.append(
            {
                "sample_id": str(item.get("sample_id")),
                "label": str(item.get("label")),
                "label_generated": int(item.get("label_generated", 0)),
                "video": (root / str(item["video"])).resolve(),
                "au": (root / str(item["au"])).resolve(),
            }
        )
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in MODES:
        deltas = [
            float(row["modes"][mode]["generated_probability"])
            - float(row["modes"]["original"]["generated_probability"])
            for row in rows
            if mode != "original"
        ]
        predictions = [
            int(row["modes"][mode]["prediction"] == "generated")
            for row in rows
        ]
        labels = [int(row["label_generated"]) for row in rows]
        tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
        tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
        fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
        fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
        result[mode] = {
            "mean_delta_p_gen_vs_original": (
                0.0 if mode == "original" else float(np.mean(deltas))
            ),
            "mean_abs_delta_p_gen_vs_original": (
                0.0 if mode == "original" else float(np.mean(np.abs(deltas)))
            ),
            "decision_flip_count_vs_original": sum(
                row["modes"][mode]["prediction"]
                != row["modes"]["original"]["prediction"]
                for row in rows
            ),
            "generated_recall": tp / (tp + fn) if tp + fn else None,
            "real_recall": tn / (tn + fp) if tn + fp else None,
            "overall_accuracy": (tp + tn) / len(rows) if rows else None,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/dev/wangxing_hard_cases/single_video/manifest.json",
    )
    parser.add_argument(
        "--model",
        default="outputs/vedio_pred/models/wangxing_v3_res1k.pt",
    )
    parser.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_holdout_excluded.json",
    )
    parser.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "experiments/20260907_authenticity_contract/"
            "artifacts/v3_shortcut_diagnosis_hard_dev.json"
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Repeat to run only named manifest samples.",
    )
    args = parser.parse_args()

    manifest_path = project_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    items = _samples(manifest, manifest_path.parent)
    if args.sample_id:
        selected = set(args.sample_id)
        items = [item for item in items if item["sample_id"] in selected]
    if args.limit > 0:
        items = items[: args.limit]
    checkpoint = torch.load(
        str(project_path(args.model)),
        map_location="cpu",
        weights_only=False,
    )
    source_profile = _load_json(args.source_profile)
    forensics_profiles = _load_json(args.forensics_profile)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not item["video"].is_file() or not item["au"].is_file():
            raise FileNotFoundError(f"Missing inputs for {item['sample_id']}")
        modes = {
            mode: _predict(
                checkpoint=checkpoint,
                video=item["video"],
                au=item["au"],
                source_profile=source_profile,
                forensics_profiles=forensics_profiles,
                mode=mode,
            )
            for mode in MODES
        }
        rows.append(
            {
                "sample_id": item["sample_id"],
                "label": item["label"],
                "label_generated": item["label_generated"],
                "modes": modes,
            }
        )
        print(
            f"[{index}/{len(items)}] {item['sample_id']} "
            f"original={modes['original']['generated_probability']:.4f} "
            f"face={modes['face_only']['generated_probability']:.4f} "
            f"background={modes['background_only']['generated_probability']:.4f}",
            flush=True,
        )
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "v3_shortcut_diagnosis_v1",
        "manifest": str(manifest_path),
        "model": str(project_path(args.model)),
        "modes": list(MODES),
        "sample_count": len(rows),
        "rows": rows,
        "summary": _summary(rows),
        "interpretation": (
            "Large decision flips or probability shifts after background_only "
            "or low_texture indicate dependence on non-face or low-frequency "
            "full-frame evidence. This is diagnostic only."
        ),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

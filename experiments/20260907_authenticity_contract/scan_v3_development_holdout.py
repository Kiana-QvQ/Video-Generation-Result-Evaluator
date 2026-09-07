"""Scan the frozen V3 development holdout and surface failure cases.

This is an experiment-only scorer. It uses the exact V3 feature extraction,
normalization, model architecture, and stored temperature, but loads the
checkpoint once for the whole scan. It never trains, writes model artifacts,
or changes production configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from wangxing_project.joint_au_pt_v3 import (
    SCALE_A,
    SCALE_B,
    V3_MODEL_TYPE,
    _extract_sequence,
    _model_from_checkpoint,
    _normalize_features,
)
from wangxing_project.joint_au_pt import (
    extract_fusion_features,
    resolve_au_csv_for_video,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _project_path(value: str) -> Path:
    return project_path(value)


def _load_profiles() -> tuple[dict[str, Any], dict[str, Any]]:
    source = _load_json(
        _project_path(
            "outputs/forensics/wangxing_source_profile_holdout_excluded.json"
        )
    )
    forensics = _load_json(
        _project_path("outputs/forensics/forensics_profiles.json")
    )
    return source, forensics


def _score(
    *,
    video: Path,
    au: Path,
    model: torch.nn.Module,
    stats: dict[str, np.ndarray],
    temperature: float,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
) -> dict[str, Any]:
    frame_a, temporal_a = _extract_sequence(
        video,
        num_frames=int(SCALE_A["num_frames"]),
        frame_size=int(SCALE_A["frame_size"]),
    )
    frame_b, temporal_b = _extract_sequence(
        video,
        num_frames=int(SCALE_B["num_frames"]),
        frame_size=int(SCALE_B["frame_size"]),
    )
    au_vector, au_details = extract_fusion_features(
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
    normalized = _normalize_features(raw, stats)
    tensors = [
        torch.from_numpy(normalized[name])
        for name in ("frame_a", "temporal_a", "frame_b", "temporal_b", "au")
    ]
    with torch.no_grad():
        logit = float(model(*tensors)[0].item())
    probability = float(
        1.0 / (1.0 + math.exp(-logit / max(temperature, 1e-6)))
    )
    return {
        "prediction": "generated" if probability >= 0.5 else "real",
        "generated_probability": probability,
        "real_probability": 1.0 - probability,
        "logit": logit,
        "au_quality_min": float(au_details.get("quality_min", 0.5)),
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["status"] == "ok"]
    labels = np.asarray([row["label_generated"] for row in valid], dtype=int)
    preds = np.asarray(
        [int(row["prediction"] == "generated") for row in valid],
        dtype=int,
    )
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    return {
        "evaluated_count": len(valid),
        "missing_count": len(rows) - len(valid),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "generated_recall": tp / max(1, tp + fn),
        "real_recall": tn / max(1, tn + fp),
        "accuracy": (tp + tn) / max(1, len(valid)),
    }


def _failure_cases(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    false_positives = [
        row
        for row in rows
        if row["status"] == "ok"
        and row["label_generated"] == 0
        and row["prediction"] == "generated"
    ]
    false_negatives = [
        row
        for row in rows
        if row["status"] == "ok"
        and row["label_generated"] == 1
        and row["prediction"] == "real"
    ]
    boundary = sorted(
        (row for row in rows if row["status"] == "ok"),
        key=lambda row: abs(float(row["generated_probability"]) - 0.5),
    )[:20]
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "nearest_threshold": boundary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan V3's frozen development holdout without training."
    )
    parser.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
    )
    parser.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_v3_res1k.pt",
    )
    parser.add_argument(
        "--output",
        default=(
            "experiments/20260907_authenticity_contract/artifacts/"
            "v3_development_holdout_scan.json"
        ),
    )
    args = parser.parse_args(argv)

    holdout_path = _project_path(args.holdout_manifest)
    checkpoint_path = _project_path(args.model_path)
    output_path = _project_path(args.output)
    holdout = _load_json(holdout_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_type") != V3_MODEL_TYPE:
        raise ValueError(f"Unsupported V3 model: {checkpoint.get('model_type')}")
    model = _model_from_checkpoint(checkpoint)
    stats = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in checkpoint["stats"].items()
    }
    temperature = float(checkpoint.get("temperature", 1.0))
    source_profile, forensics_profiles = _load_profiles()

    samples: list[tuple[int, str, dict[str, Any]]] = []
    for item in holdout.get("real", []):
        samples.append((0, "real", item))
    for item in holdout.get("seedance", holdout.get("fake", [])):
        samples.append((1, "generated", item))

    rows: list[dict[str, Any]] = []
    for index, (label, source_label, item) in enumerate(samples, start=1):
        video = _project_path(str(item["video"]))
        au = resolve_au_csv_for_video(video, au_hint=item.get("au"))
        base = {
            "index": index,
            "source_label": source_label,
            "label_generated": label,
            "video": str(video),
            "au": None if au is None else str(au),
        }
        if not video.is_file() or au is None:
            rows.append({**base, "status": "missing_inputs"})
            print(f"[{index}/{len(samples)}] missing {video.name}", flush=True)
            continue
        result = _score(
            video=video,
            au=au,
            model=model,
            stats=stats,
            temperature=temperature,
            source_profile=source_profile,
            forensics_profiles=forensics_profiles,
        )
        rows.append({**base, "status": "ok", **result})
        print(
            f"[{index}/{len(samples)}] {source_label} "
            f"pred={result['prediction']} "
            f"p_gen={result['generated_probability']:.4f}",
            flush=True,
        )

    payload = {
        "schema_version": "v3_development_holdout_scan_v1",
        "purpose": (
            "Failure diagnosis only. This official V3 holdout must not be "
            "used as a final acceptance set."
        ),
        "holdout_manifest": str(holdout_path),
        "model_path": str(checkpoint_path),
        "temperature": temperature,
        "rows": rows,
        "metrics": _metrics(rows),
        "failure_cases": _failure_cases(rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read frozen V3 joint/video/AU evidence on a small diagnostic manifest.

The output is for candidate design only. Auxiliary heads and zeroed AU inputs
must not be used as an online decision rule unless a separately trained
candidate passes the frozen acceptance suite.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from wangxing_project.joint_au_pt import extract_fusion_features
from wangxing_project.joint_au_pt_v3 import (
    SCALE_A,
    SCALE_B,
    V3_MODEL_TYPE,
    _extract_sequence,
    _model_from_checkpoint,
    _normalize_features,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sigmoid(logit: float, temperature: float) -> float:
    return float(1.0 / (1.0 + math.exp(-logit / max(temperature, 1e-6))))


def _resolve_input(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    local = (root / path).resolve()
    return local if local.is_file() else project_path(path).resolve()


def _predict(
    *,
    video: Path,
    au: Path,
    model: torch.nn.Module,
    stats: dict[str, np.ndarray],
    temperature: float,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
) -> dict[str, float]:
    frame_a, temporal_a = _extract_sequence(
        video, num_frames=int(SCALE_A["num_frames"]), frame_size=int(SCALE_A["frame_size"])
    )
    frame_b, temporal_b = _extract_sequence(
        video, num_frames=int(SCALE_B["num_frames"]), frame_size=int(SCALE_B["frame_size"])
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
    zeroed = [*tensors[:4], torch.zeros_like(tensors[4])]
    with torch.no_grad():
        joint, video, au_head = model(*tensors, return_aux=True)
        joint_zero_au = model(*zeroed)
    return {
        "joint_p_gen": _sigmoid(float(joint.item()), temperature),
        "video_aux_p_gen": _sigmoid(float(video.item()), temperature),
        "au_aux_p_gen": _sigmoid(float(au_head.item()), temperature),
        "joint_with_zeroed_au_p_gen": _sigmoid(
            float(joint_zero_au.item()), temperature
        ),
        "au_quality_min": float(au_details.get("quality_min", 0.5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=(
            "experiments/20260907_authenticity_contract/manifests/"
            "v3_development_failures.json"
        ),
    )
    parser.add_argument(
        "--model",
        default="outputs/vedio_pred/models/wangxing_v3_res1k.pt",
    )
    parser.add_argument(
        "--output",
        default=(
            "experiments/20260907_authenticity_contract/artifacts/"
            "v3_branch_evidence_development_failures.json"
        ),
    )
    args = parser.parse_args()

    manifest_path = project_path(args.manifest)
    manifest = _load_json(manifest_path)
    checkpoint = torch.load(project_path(args.model), map_location="cpu")
    if checkpoint.get("model_type") != V3_MODEL_TYPE:
        raise ValueError("Unexpected model type")
    model = _model_from_checkpoint(checkpoint)
    stats = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in checkpoint["stats"].items()
    }
    temperature = float(checkpoint.get("temperature", 1.0))
    source_profile = _load_json(
        project_path("outputs/forensics/wangxing_source_profile_holdout_excluded.json")
    )
    forensics_profiles = _load_json(
        project_path("outputs/forensics/forensics_profiles.json")
    )

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(manifest["samples"], start=1):
        video = _resolve_input(str(item["video"]), manifest_path.parent)
        au = _resolve_input(str(item["au"]), manifest_path.parent)
        evidence = _predict(
            video=video,
            au=au,
            model=model,
            stats=stats,
            temperature=temperature,
            source_profile=source_profile,
            forensics_profiles=forensics_profiles,
        )
        rows.append(
            {
                "sample_id": item["sample_id"],
                "label": item["label"],
                "label_generated": int(item["label_generated"]),
                **evidence,
            }
        )
        print(
            f"[{index}/{len(manifest['samples'])}] {item['sample_id']} "
            f"joint={evidence['joint_p_gen']:.4f} "
            f"video={evidence['video_aux_p_gen']:.4f} "
            f"au={evidence['au_aux_p_gen']:.4f} "
            f"zero_au={evidence['joint_with_zeroed_au_p_gen']:.4f}",
            flush=True,
        )

    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": "v3_branch_evidence_probe_v1",
                "purpose": "Diagnostic only; no replacement rule is implied.",
                "model": str(project_path(args.model)),
                "temperature": temperature,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

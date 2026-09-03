"""Build XiaoYue identity/source assets without inventing emotion labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from evaluator.modules.wangxing.wangxing_specialization import (
    build_identity_profile,
    build_source_profile,
    extract_sequence_features,
    sequence_feature_names,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _make_generated_manifest(
    manifest: dict[str, Any],
    output: Path,
    generated_au_root: Path,
) -> Path:
    records = []
    for item in manifest.get("generated") or []:
        video = Path(str(item["video"])).resolve()
        au_path = (generated_au_root / f"{video.stem}.csv").resolve()
        if not au_path.is_file():
            continue
        records.append(
            {
                "video_path": str(video),
                "au_path": str(au_path),
                "pseudo_label": "unknown",
                "label_status": "high_confidence",
                "source_type": "generated_xiaoyue",
            }
        )
    payload = {
        "schema_version": "xiaoyue_generated_au_manifest_v1",
        "subject": "xiaoyue",
        "records": records,
        "note": (
            "These are known generated-domain AU files for source-profile "
            "fitting; no emotion pseudo-label is asserted."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _pending_expression_profile(
    manifest: dict[str, Any],
    output: Path,
    *,
    real_au_root: Path,
) -> None:
    payload = {
        "schema_version": "xiaoyue_expression_profile_v1",
        "subject": "xiaoyue",
        "status": "pending_semantic_labels",
        "classes": {},
        "feature_names": [],
        "provenance": {
            "real_au_root": str(real_au_root),
            "candidate_real_video_count": len(manifest.get("real") or []),
            "reason": (
                "Available folders describe capture protocols "
                "(FACS/BiaoQing/ShengYin/TaiCi), not verified emotion labels."
            ),
            "required_next_input": (
                "A mapping from source folders or clips to semantic emotion "
                "classes before expression compatibility training."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _expression_class(relative: Path) -> str | None:
    parts = [part.casefold() for part in relative.parts]
    if len(parts) >= 2 and parts[0] == "reference":
        return parts[1]
    if len(parts) >= 3 and parts[0] == "hk":
        return parts[-2]
    return None


def _build_expression_profile(
    manifest: dict[str, Any],
    real_au_root: Path,
    output: Path,
    *,
    candidate_manifest: Path,
) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = {}
    metadata: list[dict[str, Any]] = []
    for au_path in sorted(real_au_root.rglob("*.csv")):
        relative = au_path.relative_to(real_au_root)
        expression_class = _expression_class(relative)
        if expression_class is None:
            continue
        try:
            vector, quality = extract_sequence_features(au_path)
        except (OSError, ValueError, RuntimeError):
            continue
        grouped.setdefault(expression_class, []).append(vector)
        metadata.append(
            {
                "path": str(au_path),
                "class": expression_class,
                "frame_count": quality["frame_count"],
                "valid_frame_ratio": quality["valid_frame_ratio"],
            }
        )
    if len(grouped) < 2:
        raise ValueError("XiaoYue expression profile needs at least two source groups.")
    classes: dict[str, Any] = {}
    feature_names = list(sequence_feature_names())
    for name, vectors in sorted(grouped.items()):
        matrix = np.stack(vectors).astype(float)
        location = np.median(matrix, axis=0)
        scale = np.maximum(
            (np.quantile(matrix, 0.75, axis=0)
             - np.quantile(matrix, 0.25, axis=0)) / 1.349,
            1e-3,
        )
        distances = np.sqrt(
            np.mean(np.square((matrix - location) / scale), axis=1)
        )
        classes[name] = {
            "display_name": f"{name} / XiaoYue source group",
            "sample_count": len(vectors),
            "location": location.tolist(),
            "scale": scale.tolist(),
            "distance_threshold": float(
                max(0.75, float(np.quantile(distances, 0.95)) * 1.20)
            ),
            "training_distance_summary": {
                "median": float(np.median(distances)),
                "p95": float(np.quantile(distances, 0.95)),
                "max": float(np.max(distances)),
            },
        }
    payload = {
        "schema_version": "xiaoyue_expression_profile_v1",
        "subject": "xiaoyue",
        "status": "available",
        "semantic_labels_confirmed": False,
        "feature_names": feature_names,
        "classes": classes,
        "class_mapping_note": (
            "Classes are capture/action source groups (FACS0-5, TaiCi, "
            "BiaoQing, ShengYin, YaoTou), not verified emotion names."
        ),
        "provenance": {
            "real_au_root": str(real_au_root),
            "sample_count": len(metadata),
            "class_counts": {
                name: len(values) for name, values in sorted(grouped.items())
            },
            "metadata": metadata,
            "candidate_manifest": str(candidate_manifest.resolve()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/xiaoyue/processed/specialization_manifest.json",
    )
    parser.add_argument(
        "--real-video-root",
        default="data/xiaoyue/processed/real_candidates",
    )
    parser.add_argument(
        "--generated-video-root",
        default="data/xiaoyue/processed/ai_candidates",
    )
    parser.add_argument(
        "--negative-root",
        default="data/negative/ravdess/videos",
    )
    parser.add_argument(
        "--real-au-root",
        default="data/au/xiaoyue/real",
    )
    parser.add_argument(
        "--generated-au-root",
        default="data/au/xiaoyue/generated",
    )
    parser.add_argument(
        "--profile-root",
        default="data/xiaoyue/profiles",
    )
    parser.add_argument("--identity-frames", type=int, default=8)
    parser.add_argument("--identity-limit", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    manifest = _load(project_path(args.manifest))
    profile_root = project_path(args.profile_root)
    profile_root.mkdir(parents=True, exist_ok=True)
    generated_manifest = _make_generated_manifest(
        manifest,
        profile_root / "xiaoyue_generated_au_manifest.json",
        project_path(args.generated_au_root),
    )

    identity = build_identity_profile(
        real_root=project_path(args.real_video_root),
        generated_root=project_path(args.generated_video_root),
        negative_root=project_path(args.negative_root),
        output_path=profile_root / "xiaoyue_identity_profile.json",
        device=args.device,
        max_frames=args.identity_frames,
        limit=args.identity_limit if args.identity_limit > 0 else None,
        subject="xiaoyue",
    )
    identity["subject"] = "xiaoyue"
    identity["source_manifest"] = str(project_path(args.manifest).resolve())
    (profile_root / "xiaoyue_identity_profile.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    source = build_source_profile(
        real_au_root=project_path(args.real_au_root),
        seedance_label_manifest=generated_manifest,
        output_path=profile_root / "xiaoyue_source_profile.json",
    )
    source["subject"] = "xiaoyue"
    source["source_semantics"] = {
        "real": "xiaoyue_real",
        "generated": "xiaoyue_generated",
    }
    source_models = source.get("sources", {})
    if "real_wangxing" in source_models:
        source_models["real_xiaoyue"] = source_models.pop("real_wangxing")
    if "generated_wangxing" in source_models:
        source_models["generated_xiaoyue"] = source_models.pop(
            "generated_wangxing"
        )
    source["sources"] = source_models
    source_counts = Counter(
        str(item.get("source_type") or "")
        for item in source["provenance"].get("metadata", [])
    )
    for item in source["provenance"].get("metadata", []):
        source_type = item.get("source_type")
        if source_type == "real_wangxing":
            item["source_type"] = "real_xiaoyue"
            source_counts["real_xiaoyue"] += 1
            source_counts["real_wangxing"] -= 1
        elif source_type == "generated_wangxing":
            item["source_type"] = "generated_xiaoyue"
            source_counts["generated_xiaoyue"] += 1
            source_counts["generated_wangxing"] -= 1
    source["provenance"]["sample_counts"] = {
        key: count
        for key, count in source_counts.items()
        if key and count > 0
    }
    source["schema_version"] = "xiaoyue_source_profile_v1"
    (profile_root / "xiaoyue_source_profile.json").write_text(
        json.dumps(source, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    expression_path = profile_root / "xiaoyue_expression_profile.json"
    expression = _build_expression_profile(
        manifest,
        project_path(args.real_au_root),
        expression_path,
        candidate_manifest=project_path(args.manifest),
    )
    result = {
        "subject": "xiaoyue",
        "identity_profile": str(
            (profile_root / "xiaoyue_identity_profile.json").resolve()
        ),
        "source_profile": str(
            (profile_root / "xiaoyue_source_profile.json").resolve()
        ),
        "expression_profile": str(expression_path.resolve()),
        "identity_counts": {
            "positive": identity["calibration"]["positive_count"],
            "negative": identity["calibration"]["negative_count"],
        },
        "source_counts": source["provenance"]["sample_counts"],
        "expression_status": expression["status"],
        "expression_semantic_labels_confirmed": expression[
            "semantic_labels_confirmed"
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

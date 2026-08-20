from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.forensics.holdout import holdout_paths
from evaluator.modules.core.paths import project_path

VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".wmv"}
AU_SUFFIXES = {".csv", ".tsv"}
EXPRESSION_CLASSES = (
    "anger",
    "disgust",
    "fear",
    "sadness",
    "smile",
    "surprise",
    "neutral",
)
EXPRESSION_ALIASES = {
    "fennu": "anger",
    "shengqi": "anger",
    "yanwu": "disgust",
    "kongju": "fear",
    "beishang": "sadness",
    "kaixin": "smile",
    "jingya": "surprise",
}


def _files(root: Path, suffixes: set[str]) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _matched_pairs(video_root: Path, au_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for au_path in _files(au_root, AU_SUFFIXES):
        relative = au_path.relative_to(au_root)
        video_path = next(
            (
                candidate
                for suffix in sorted(VIDEO_SUFFIXES)
                if (
                    candidate := video_root / relative.with_suffix(suffix)
                ).is_file()
            ),
            None,
        )
        if video_path is not None:
            pairs.append((video_path, au_path))
    return pairs


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _expression_class(au_path: Path) -> str | None:
    for part in au_path.parts:
        key = part.lower()
        for alias, expression in EXPRESSION_ALIASES.items():
            if alias in key:
                return expression
    return None


def _load_pseudo_labels(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        video_path = record.get("video_path")
        if isinstance(video_path, str):
            result[str(Path(video_path).resolve())] = {
                "pseudo_expression_class": record.get("pseudo_label"),
                "pseudo_expression_status": record.get("label_status"),
                "pseudo_expression_confidence": record.get(
                    "confidence_0_1"
                ),
                "pseudo_expression_training_allowed": bool(
                    record.get("use_for_training", False)
                ),
            }
    return result


def _record(
    *,
    domain: str,
    video_path: Path,
    au_path: Path,
    holdout_videos: set[str],
    pseudo_labels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    resolved_video = str(video_path.resolve())
    real_expression = (
        _expression_class(au_path) if domain == "real" else None
    )
    record: dict[str, Any] = {
        "video_path": _relative(video_path),
        "au_path": _relative(au_path),
        "feature_path": None,
        "domain": domain,
        "source_label": 1 if domain == "seedance" else 0,
        "identity_label": 1,
        "identity_label_source": "dataset_scope_assumption",
        "expression_class": real_expression,
        "expression_support_label": None,
        "quality_label": None,
        "artifact_label": None,
        "split": (
            "source_holdout"
            if resolved_video in holdout_videos
            else "profile_train"
        ),
        "labels_require_review": domain == "seedance",
        "label_notes": (
            "Source label is known from the dataset root. Expression and "
            "quality labels still require annotation."
            if domain == "seedance"
            else (
                "Expression class is inferred from the real-data directory; "
                "quality is not assumed to be perfect."
            )
        ),
    }
    if domain == "seedance":
        record.update(pseudo_labels.get(resolved_video, {}))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a label-safe manifest for the shared-input Wang Xing "
            "forensics model. This reads paths only; it does not read CSV "
            "contents or video frames."
        )
    )
    parser.add_argument("--real-video-root", default="data/MD_CL")
    parser.add_argument("--real-au-root", default="data/au/MD_CL")
    parser.add_argument(
        "--seedance-video-root",
        default="data/WangXing_Seedance",
    )
    parser.add_argument(
        "--seedance-au-root",
        default="data/au/WangXing_Seedance",
    )
    parser.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
    )
    parser.add_argument(
        "--pseudo-label-manifest",
        default="data/au/WangXing_Seedance/pseudo_expression_manifest.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/forensics/joint_forensics_manifest.json",
    )
    args = parser.parse_args()

    real_video_root = project_path(args.real_video_root)
    real_au_root = project_path(args.real_au_root)
    seedance_video_root = project_path(args.seedance_video_root)
    seedance_au_root = project_path(args.seedance_au_root)
    holdout_manifest = project_path(args.holdout_manifest)
    pseudo_manifest = project_path(args.pseudo_label_manifest)

    holdout_videos = set()
    if holdout_manifest.is_file():
        holdout_videos = holdout_paths(
            holdout_manifest,
            domain="real",
            kind="video",
        ) | holdout_paths(
            holdout_manifest,
            domain="seedance",
            kind="video",
        )
    pseudo_labels = _load_pseudo_labels(pseudo_manifest)

    real_pairs = _matched_pairs(real_video_root, real_au_root)
    seedance_pairs = _matched_pairs(seedance_video_root, seedance_au_root)
    records = [
        _record(
            domain="real",
            video_path=video_path,
            au_path=au_path,
            holdout_videos=holdout_videos,
            pseudo_labels=pseudo_labels,
        )
        for video_path, au_path in real_pairs
    ]
    records.extend(
        _record(
            domain="seedance",
            video_path=video_path,
            au_path=au_path,
            holdout_videos=holdout_videos,
            pseudo_labels=pseudo_labels,
        )
        for video_path, au_path in seedance_pairs
    )

    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "joint_forensics_manifest_v1",
        "feature_contract": {
            "file_format": "npz",
            "keys": [
                "visual",
                "facial",
                "texture",
                "audio",
                "frame_mask",
            ],
            "shape": "[frames, feature_dim] per modality",
            "audio_optional": True,
        },
        "expression_classes": list(EXPRESSION_CLASSES),
        "label_policy": {
            "source_label": "known dataset source, not a generic artifact truth",
            "identity_label": "weak dataset-scope assumption; verify with ArcFace",
            "expression_class": "real directory support label only",
            "expression_support_label": "manual annotation required",
            "quality_label": "manual annotation required",
            "artifact_label": "manual or source-held-out evaluation label",
            "pseudo_labels": "diagnostic only unless explicitly reviewed",
        },
        "records": records,
        "summary": {
            "record_count": len(records),
            "real_count": len(real_pairs),
            "seedance_count": len(seedance_pairs),
            "source_holdout_count": sum(
                record["split"] == "source_holdout" for record in records
            ),
            "real_expression_class_count": sum(
                record["expression_class"] is not None
                for record in records
                if record["domain"] == "real"
            ),
            "pseudo_training_allowed_count": sum(
                record.get("pseudo_expression_training_allowed", False)
                for record in records
            ),
        },
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

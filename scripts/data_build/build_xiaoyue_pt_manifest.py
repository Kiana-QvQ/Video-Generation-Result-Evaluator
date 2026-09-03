"""Build isolated XiaoYue PT train/test manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(value: str | Path) -> Path:
    target = Path(value).expanduser()
    return (target if target.is_absolute() else PROJECT_ROOT / target).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _train_pairs(
    source_manifest: dict[str, Any],
    real_au_root: Path,
    generated_au_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    real: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    for index, item in enumerate(source_manifest.get("real") or []):
        video = _path(item["video"])
        relative = video.relative_to(_path("data/xiaoyue/processed/real_candidates"))
        au = real_au_root / relative.with_suffix(".csv")
        if video.is_file() and au.is_file():
            group = "/".join(relative.parts[: min(3, len(relative.parts))])
            real.append(
                {
                    "video": str(video),
                    "au": str(au.resolve()),
                    "label_generated": 0,
                    "base_label": 0,
                    "group_id": f"real_{group}",
                    "source_id": f"real_{index:04d}",
                }
            )
    for index, item in enumerate(source_manifest.get("generated") or []):
        video = _path(item["video"])
        au = generated_au_root / f"{video.stem}.csv"
        if video.is_file() and au.is_file():
            generated.append(
                {
                    "video": str(video),
                    "au": str(au.resolve()),
                    "label_generated": 1,
                    "base_label": 1,
                    "group_id": f"generated_{video.stem}",
                    "source_id": f"generated_{index:04d}",
                }
            )
    return real, generated


def _test_item(
    video: Path,
    au: Path,
    *,
    label: int,
    group_id: str,
    sample_id: str,
) -> dict[str, Any]:
    return {
        "video": str(video.resolve()),
        "au": str(au.resolve()),
        "label_generated": label,
        "base_label": label,
        "group_id": group_id,
        "sample_id": sample_id,
        "sha256": _sha256(video),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        default="data/xiaoyue/processed/specialization_manifest.json",
    )
    parser.add_argument("--real-au-root", default="data/au/xiaoyue/real")
    parser.add_argument(
        "--generated-au-root",
        default="data/au/xiaoyue/generated",
    )
    parser.add_argument(
        "--test-root",
        default="data/xiaoyue/processed/test_reference",
    )
    parser.add_argument(
        "--test-au-root",
        default="data/au/xiaoyue/test",
    )
    parser.add_argument(
        "--output",
        default="data/xiaoyue/processed/pt_manifest.json",
    )
    parser.add_argument(
        "--test-output",
        default="data/xiaoyue/processed/pt_test_manifest.json",
    )
    args = parser.parse_args(argv)
    source = _load(_path(args.source_manifest))
    real, generated = _train_pairs(
        source,
        _path(args.real_au_root),
        _path(args.generated_au_root),
    )
    test_root = _path(args.test_root)
    test_au_root = _path(args.test_au_root)
    test_specs = (
        ("gaussian_reference.mp4", 0, "xiaoyue_test_gaussian_real", "real_gaussian"),
        ("test1_real_reference.mp4", 0, "xiaoyue_test_1_real", "real_01"),
        ("test2_real_reference.mp4", 0, "xiaoyue_test_2_real", "real_02"),
        ("test1_seedance25.mp4", 1, "xiaoyue_test_1_ai", "ai_01"),
        ("test2_seedance25.mp4", 1, "xiaoyue_test_2_ai", "ai_02"),
    )
    test_real: list[dict[str, Any]] = []
    test_fake: list[dict[str, Any]] = []
    seen_test_hashes: set[str] = set()
    duplicate_test_samples: list[str] = []
    for filename, label, group_id, sample_id in test_specs:
        video = test_root / filename
        au = test_au_root / Path(filename).with_suffix(".csv").name
        if not video.is_file() or not au.is_file():
            continue
        video_hash = _sha256(video)
        if video_hash in seen_test_hashes:
            duplicate_test_samples.append(sample_id)
            continue
        item = _test_item(
            video,
            au,
            label=label,
            group_id=group_id,
            sample_id=sample_id,
        )
        seen_test_hashes.add(video_hash)
        (test_fake if label else test_real).append(item)
    train_hashes = {
        _sha256(_path(item["video"]))
        for item in [*real, *generated]
    }
    test_hashes = {item["sha256"] for item in [*test_real, *test_fake]}
    overlap = train_hashes & test_hashes
    if overlap:
        raise SystemExit("XiaoYue train/test SHA-256 overlap detected.")
    train_payload = {
        "schema_version": "xiaoyue_temporal_au_video_v3_manifest_v1",
        "subject": "xiaoyue",
        "training_allowed": True,
        "pairs": {
            "train": {"real": real, "fake": generated},
            "test": {"real": test_real, "fake": test_fake},
        },
        "counts": {
            "train_real": len(real),
            "train_fake": len(generated),
            "test_real": len(test_real),
            "test_fake": len(test_fake),
        },
        "excluded_test_paths": [
            "data/xiaoyue/test",
            "data/xiaoyue/processed/test_reference",
        ],
        "notes": [
            "Only accepted source-quality candidates enter training.",
            "The four AI candidates are an initial generated domain and are not sufficient for broad generator generalization.",
            "All test/reference files remain evaluation-only.",
        ],
    }
    test_payload = {
        "schema_version": "xiaoyue_temporal_au_video_v3_test_manifest_v1",
        "subject": "xiaoyue",
        "training_allowed": False,
        "real": test_real,
        "fake": test_fake,
        "seedance": test_fake,
        "counts": {
            "real": len(test_real),
            "fake": len(test_fake),
        },
        "train_test_sha256_overlap": False,
        "duplicate_samples_excluded": duplicate_test_samples,
    }
    output = _path(args.output)
    test_output = _path(args.test_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(train_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    test_output.write_text(json.dumps(test_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train_manifest": str(output), "test_manifest": str(test_output), "train": train_payload["counts"], "test": test_payload["counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

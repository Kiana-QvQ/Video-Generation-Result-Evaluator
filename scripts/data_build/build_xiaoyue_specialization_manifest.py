"""Build the isolated XiaoYue specialization training manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(value: str | Path) -> Path:
    target = Path(value).expanduser()
    return target if target.is_absolute() else PROJECT_ROOT / target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_real_candidates(
    screening: dict[str, Any],
    source_root: Path,
    destination_root: Path,
) -> list[dict[str, Any]]:
    destination_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row in screening.get("videos") or []:
        if row.get("decision") != "keep_candidate":
            continue
        source = Path(str(row["video"])).expanduser().resolve()
        if not source.is_file():
            continue
        try:
            relative = source.relative_to(source_root)
        except ValueError as exc:
            raise SystemExit(
                f"Accepted source is outside source root: {source}"
            ) from exc
        if any(part.casefold() in {"test", "test_reference"} for part in relative.parts):
            raise SystemExit(f"Test path entered training set: {source}")
        target = destination_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)
        records.append(
            {
                "video": str(target.resolve()),
                "source_video": str(source),
                "source_kind": row.get("source_kind"),
                "screening_decision": row.get("decision"),
                "sample_keep_ratio": row.get("sample_keep_ratio"),
                "sha256": _sha256(target),
            }
        )
    return records


def _copy_ai_candidates(
    source_root: Path,
    destination_root: Path,
) -> list[dict[str, Any]]:
    destination_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for source in sorted(source_root.rglob("*.mp4")):
        target = destination_root / source.name
        if not target.is_file() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)
        records.append(
            {
                "video": str(target.resolve()),
                "source_video": str(source.resolve()),
                "label": "generated_xiaoyue",
                "sha256": _sha256(target),
            }
        )
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--screening-manifest",
        default="outputs/xiaoyue_screening/source_quality_manifest.json",
    )
    parser.add_argument("--source-root", default="data/xiaoyue/source")
    parser.add_argument(
        "--real-output-root",
        default="data/xiaoyue/processed/real_candidates",
    )
    parser.add_argument(
        "--ai-root",
        default="data/xiaoyue/processed/ai_candidates",
    )
    parser.add_argument(
        "--output",
        default="data/xiaoyue/processed/specialization_manifest.json",
    )
    args = parser.parse_args(argv)
    screening_path = _path(args.screening_manifest)
    screening = json.loads(
        screening_path.read_text(encoding="utf-8-sig")
    )
    source_root = _path(args.source_root).resolve()
    real_records = _copy_real_candidates(
        screening,
        source_root,
        _path(args.real_output_root).resolve(),
    )
    ai_records = _copy_ai_candidates(
        _path(args.ai_root).resolve(),
        _path(args.ai_root).resolve(),
    )
    real_hashes = {item["sha256"] for item in real_records}
    ai_records = [
        item for item in ai_records if item["sha256"] not in real_hashes
    ]
    result = {
        "schema_version": "xiaoyue_specialization_manifest_v1",
        "subject": "xiaoyue",
        "screening_manifest": str(screening_path.resolve()),
        "training_allowed": True,
        "test_paths_forbidden": [
            "data/xiaoyue/test",
            "data/xiaoyue/processed/test_reference",
        ],
        "real": real_records,
        "generated": ai_records,
        "counts": {
            "real": len(real_records),
            "generated": len(ai_records),
            "total": len(real_records) + len(ai_records),
        },
        "notes": [
            "Only keep_candidate videos from the source quality manifest enter the real set.",
            "manual_review and exclude_candidate videos remain outside training.",
            "All test/reference videos remain excluded.",
            "AI candidates are a small initial domain and do not support generator ranking.",
        ],
    }
    output = _path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "counts": result["counts"],
                "test_paths_forbidden": result["test_paths_forbidden"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

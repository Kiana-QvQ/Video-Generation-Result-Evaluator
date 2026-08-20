"""Build a Wang Xing hard-example development set from a frozen web result.

The source result is treated as a diagnostic benchmark. The generated set is
development-only and must never replace the final 32+32 test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "forensics"
    / "web_forensics_v2_results_full"
    / "all_results.json"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "dev" / "wangxing_hard_cases"
DEFAULT_PROFILE_EXCLUSION = (
    PROJECT_ROOT / "data" / "forensics" / "wangxing_hard_dev_exclusion.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256(source) == _sha256(destination):
        return
    if destination.is_file():
        destination.unlink()
    try:
        destination.hardlink_to(source)
    except (OSError, NotImplementedError):
        shutil.copy2(source, destination)


def _pick_rows(
    results: list[dict[str, Any]],
    *,
    ai_count: int,
    real_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    missed_ai = [
        row
        for row in results
        if row.get("label") == "ai"
        and (row.get("web_fusion") or {}).get("prediction") != "generated"
    ]
    if len(missed_ai) < ai_count:
        raise ValueError(
            f"Expected at least {ai_count} missed AI rows, got {len(missed_ai)}."
        )
    missed_ai.sort(
        key=lambda row: (
            (row.get("web_fusion") or {}).get("generated_probability", 1.0),
            row.get("sample_id", ""),
        )
    )

    hard_real = [
        row
        for row in results
        if row.get("label") == "real"
        and (row.get("web_fusion") or {}).get("prediction") == "real"
    ]
    if len(hard_real) < real_count:
        raise ValueError(
            f"Expected at least {real_count} real rows, got {len(hard_real)}."
        )
    hard_real.sort(
        key=lambda row: (
            -float((row.get("web_fusion") or {}).get("generated_probability", 0.0)),
            row.get("sample_id", ""),
        )
    )
    return missed_ai[:ai_count], hard_real[:real_count]


def _make_sample(
    *,
    output_root: Path,
    row: dict[str, Any],
    index: int,
    hard_reason: str,
) -> dict[str, Any]:
    label = str(row["label"])
    sample_id = f"{label}_{index:02d}"
    source_video = Path(str(row["video"])).resolve()
    source_au = Path(str(row["au"])).resolve()
    if not source_video.is_file() or not source_au.is_file():
        raise FileNotFoundError(
            f"Missing hard-dev pair: {source_video} / {source_au}"
        )
    sample_root = output_root / "single_video" / label / sample_id
    _link_or_copy(source_video, sample_root / "video.mp4")
    _link_or_copy(source_au, sample_root / "au.csv")
    return {
        "sample_id": sample_id,
        "label": label,
        "label_generated": int(label == "ai"),
        "video": f"{label}/{sample_id}/video.mp4",
        "au": f"{label}/{sample_id}/au.csv",
        "source_video": str(source_video),
        "source_au": str(source_au),
        "source_sample_id": row.get("sample_id"),
        "hard_reason": hard_reason,
        "development_only": True,
        "final_test": False,
        "training_allowed": False,
        "profile_training_allowed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the hard-example development set from frozen web results."
    )
    parser.add_argument("--source-results", default=str(DEFAULT_SOURCE_RESULTS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--profile-exclusion",
        default=str(DEFAULT_PROFILE_EXCLUSION),
    )
    parser.add_argument("--ai-count", type=int, default=6)
    parser.add_argument("--real-count", type=int, default=6)
    args = parser.parse_args(argv)

    source = _load(Path(args.source_results).expanduser().resolve())
    ai_rows, real_rows = _pick_rows(
        list(source.get("results") or []),
        ai_count=int(args.ai_count),
        real_count=int(args.real_count),
    )
    output_root = Path(args.output_root).expanduser().resolve()
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(ai_rows, start=1):
        samples.append(
            _make_sample(
                output_root=output_root,
                row=row,
                index=index,
                hard_reason="web_v2_missed_ai",
            )
        )
    for index, row in enumerate(real_rows, start=1):
        samples.append(
            _make_sample(
                output_root=output_root,
                row=row,
                index=index,
                hard_reason="web_v2_high_real_probability",
            )
        )

    manifest = {
        "schema_version": "wangxing_hard_dev_v1",
        "sample_count": len(samples),
        "real_count": int(args.real_count),
        "ai_count": int(args.ai_count),
        "development_only": True,
        "final_test": False,
        "training_allowed": False,
        "profile_training_allowed": False,
        "source_results": str(Path(args.source_results).expanduser().resolve()),
        "samples": samples,
    }
    _write(output_root / "single_video" / "manifest.json", manifest)
    for sample in samples:
        _write(
            output_root
            / "single_video"
            / sample["label"]
            / sample["sample_id"]
            / "sample.json",
            sample,
        )

    exclusion = {
        "schema_version": "wangxing_hard_dev_exclusion_v1",
        "note": (
            "Development-only hard cases. Exclude from profile fitting while "
            "tuning the web authenticity policy; never use as final test."
        ),
        "real": [
            {"video": row["source_video"], "au": row["source_au"]}
            for row in samples
            if row["label"] == "real"
        ],
        "seedance": [
            {"video": row["source_video"], "au": row["source_au"]}
            for row in samples
            if row["label"] == "ai"
        ],
    }
    _write(Path(args.profile_exclusion).expanduser().resolve(), exclusion)
    (output_root / "README.md").write_text(
        """# Wang Xing Hard Development Set

This set contains six AI cases missed by the frozen web v2 result and six
hard real controls near the web decision boundary.

It is development-only. It must not replace or overwrite the final
`data/test/wangxing_32x32` set.
""",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "manifest": str(output_root / "single_video" / "manifest.json"),
                "profile_exclusion": str(
                    Path(args.profile_exclusion).expanduser().resolve()
                ),
                "counts": {
                    "ai": int(args.ai_count),
                    "real": int(args.real_count),
                },
                "training_allowed": False,
                "final_test": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

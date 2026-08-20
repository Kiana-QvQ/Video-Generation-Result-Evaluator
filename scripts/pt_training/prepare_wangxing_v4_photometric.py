"""Prepare train-only photometric variants for the face-aware PT v4 model."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRESETS = (
    ("bright", "eq=brightness=0.08:contrast=1.05:saturation=1.05:gamma=1.05"),
    ("dim", "eq=brightness=-0.08:contrast=0.95:saturation=0.95:gamma=0.95"),
    ("warm", "eq=brightness=0.02:contrast=1.0:saturation=1.12:gamma=1.0"),
    ("flat", "eq=brightness=0.0:contrast=0.88:saturation=0.90:gamma=1.0"),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _make_variant(
    source: Path,
    destination: Path,
    filter_graph: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        filter_graph,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(destination),
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or not destination.is_file():
        detail = (result.stderr or result.stdout).decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(f"ffmpeg photometric variant failed: {detail[-500:]}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build train-only photometric variants for v4."
    )
    parser.add_argument(
        "--manifest",
        default="outputs/vedio_pred/wangxing_v3_generalization_manifest_res1k.json",
    )
    parser.add_argument(
        "--output-manifest",
        default="outputs/vedio_pred/wangxing_v4_generalization_manifest_res1k.json",
    )
    parser.add_argument(
        "--output-root",
        default="data/_aug/wangxing_v4_photometric",
    )
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    manifest = _load(PROJECT_ROOT / args.manifest)
    output_root = PROJECT_ROOT / args.output_root
    rng = random.Random(args.seed)
    train = manifest["pairs"]["train"]
    records: dict[str, list[dict[str, Any]]] = {
        "real": list(train["real"]),
        "fake": list(train["fake"]),
    }
    for label in ("real", "fake"):
        candidates = list(train[label])
        rng.shuffle(candidates)
        for index, item in enumerate(
            candidates[: max(0, int(args.max_per_class))]
        ):
            source = PROJECT_ROOT / item["video"]
            preset_name, filter_graph = PRESETS[index % len(PRESETS)]
            destination = (
                output_root
                / label
                / f"{source.stem}_{preset_name}_{index:04d}.mp4"
            )
            _make_variant(source, destination, filter_graph)
            records[label].append(
                {
                    **item,
                    "video": _relative(destination),
                    "augmentation": f"photometric_{preset_name}",
                    "source_video": item["video"],
                }
            )
    output = {
        "schema_version": "wangxing_v4_photometric_manifest_v1",
        "protocol": {
            "base_manifest": args.manifest,
            "photometric_presets": [name for name, _ in PRESETS],
            "change_clips_in_train": False,
            "group_split_required": True,
            "training_allowed": True,
        },
        "pairs": {
            "train": records,
            "test": manifest["pairs"]["test"],
        },
        "counts": {
            "train_real": len(records["real"]),
            "train_fake": len(records["fake"]),
            "test_real": len(manifest["pairs"]["test"]["real"]),
            "test_fake": len(manifest["pairs"]["test"]["fake"]),
            "photometric_real": sum(
                item.get("augmentation", "").startswith("photometric_")
                for item in records["real"]
            ),
            "photometric_fake": sum(
                item.get("augmentation", "").startswith("photometric_")
                for item in records["fake"]
            ),
        },
    }
    _write(PROJECT_ROOT / args.output_manifest, output)
    print(json.dumps(output["counts"], ensure_ascii=False, indent=2))
    print(f"Wrote {PROJECT_ROOT / args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

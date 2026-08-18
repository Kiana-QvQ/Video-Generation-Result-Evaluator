"""Prepare train-only domain-generalization data for the v3 detector.

The five ``data/test/AI`` Change clips are never read by this script.
It creates:
- label-preserving media variants for both real and generated videos;
- local frame-wise warp/blend pseudo-fakes from real training videos.

The output manifest stores explicit video/AU pairs and source groups so the
v3 trainer can keep an original clip and all of its variants in one split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from wangxing_project.joint_au_pt import (
    attach_au_pairs,
    is_augmented_video,
    is_forbidden_train_video,
)

MEDIA_LONG_EDGE = 1280
MEDIA_FPS = 24
DEFAULT_CRFS = (23, 28)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _group_id(video: Path) -> str:
    digest = hashlib.sha1(str(video.resolve()).encode("utf-8")).hexdigest()
    return digest[:16]


def _safe_stem(video: Path) -> str:
    digest = hashlib.sha1(str(video.resolve()).encode("utf-8")).hexdigest()
    return f"{video.stem[:32]}_{digest[:10]}"


def _run_ffmpeg(
    source: Path,
    destination: Path,
    *,
    crf: int,
    long_edge: int,
    fps: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    vf = (
        f"scale=if(gt(iw,ih),{long_edge},-2):"
        f"if(gt(iw,ih),-2,{long_edge}),"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,"
        f"fps={fps}"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(int(crf)),
        str(destination),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = (
            completed.stderr or completed.stdout or b"ffmpeg failed"
        ).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ffmpeg failed for {source.name}: {message[-500:]}"
        )
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg produced an empty file: {destination}")


def _blend_video(
    source: Path,
    destination: Path,
    *,
    seed: int,
    alpha: float = 0.18,
) -> None:
    """Create a conservative local warp/blend pseudo-fake.

    The AU sidecar stays tied to the source clip because the operation targets
    visual facial feature drift while preserving the original motion labels.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid source video dimensions: {source}")

    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create pseudo-fake video: {destination}")

    rng = np.random.default_rng(int(seed))
    mask = np.zeros((height, width), dtype=np.float32)
    center = (int(width * 0.50), int(height * 0.47))
    axes = (max(2, int(width * 0.37)), max(2, int(height * 0.30)))
    cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(width, height) * 0.035)
    mask = np.clip(mask * float(alpha), 0.0, 1.0)[..., None]

    written = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        scale_x = 1.0 + float(rng.uniform(-0.018, 0.018))
        scale_y = 1.0 + float(rng.uniform(-0.018, 0.018))
        shift_x = float(rng.uniform(-0.018, 0.018) * width)
        shift_y = float(rng.uniform(-0.018, 0.018) * height)
        matrix = np.asarray(
            [
                [scale_x, 0.0, shift_x],
                [0.0, scale_y, shift_y],
            ],
            dtype=np.float32,
        )
        warped = cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        frame_float = frame.astype(np.float32)
        warped_float = warped.astype(np.float32)
        blended = (
            frame_float * (1.0 - mask)
            + warped_float * mask
        ).clip(0.0, 255.0).astype(np.uint8)
        writer.write(blended)
        written += 1

    capture.release()
    writer.release()
    if written < 2 or not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError(f"Pseudo-fake video is incomplete: {destination}")


def _record(
    *,
    video: Path,
    au: Path,
    label: int,
    base_label: int,
    group_id: str,
    augmentation: str,
    source_video: Path,
) -> dict[str, Any]:
    if is_forbidden_train_video(video):
        raise ValueError(f"Forbidden Change video entered v3 train: {video}")
    return {
        "video": str(video.resolve()),
        "au": str(au.resolve()),
        "label_generated": int(label),
        "base_label": int(base_label),
        "group_id": str(group_id),
        "augmentation": augmentation,
        "source_video": str(source_video.resolve()),
    }


def _original_pairs(
    pairs: list[dict[str, Any]],
    *,
    label: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in pairs:
        video = _resolve(str(item["video"]))
        au = _resolve(str(item["au"]))
        if is_augmented_video(video):
            continue
        result.append(
            _record(
                video=video,
                au=au,
                label=label,
                base_label=label,
                group_id=_group_id(video),
                augmentation="original",
                source_video=video,
            )
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare train-only v3 domain-generalization data. "
            "data/test/AI is never read."
        )
    )
    parser.add_argument(
        "--manifest",
        default="outputs/vedio_pred/wangxing_dual_pt_split_res1k.json",
    )
    parser.add_argument(
        "--output-manifest",
        default=(
            "outputs/vedio_pred/"
            "wangxing_v3_generalization_manifest_res1k.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="data/_aug/wangxing_v3_generalization",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-pseudo-fakes",
        type=int,
        default=120,
        help="Maximum real training clips used to create local warp/blend fakes.",
    )
    parser.add_argument(
        "--max-media-per-class",
        type=int,
        default=120,
        help="Maximum original clips per class receiving media variants.",
    )
    parser.add_argument(
        "--skip-pseudo-fakes",
        action="store_true",
    )
    parser.add_argument(
        "--skip-media-variants",
        action="store_true",
    )
    args = parser.parse_args(argv)

    base_path = _resolve(args.manifest)
    if not base_path.is_file():
        raise SystemExit(f"Base manifest not found: {base_path}")
    manifest = _load_json(base_path)
    if "pairs" not in manifest:
        manifest = attach_au_pairs(
            manifest,
            project_root=PROJECT_ROOT,
            holdout_manifest=project_path("data/forensics/holdout_split.json"),
        )

    train_real = _original_pairs(
        list(manifest["pairs"]["train"]["real"]),
        label=0,
    )
    train_fake = _original_pairs(
        list(manifest["pairs"]["train"]["fake"]),
        label=1,
    )
    if not train_real or not train_fake:
        raise SystemExit("Need original real and generated training pairs.")

    output_root = _resolve(args.output_root)
    rng = np.random.default_rng(int(args.seed))
    pseudo_records: list[dict[str, Any]] = []
    if not args.skip_pseudo_fakes:
        count = min(max(0, int(args.max_pseudo_fakes)), len(train_real))
        selected = rng.choice(len(train_real), size=count, replace=False)
        print(
            f"[pseudo] generating {count} train-only warp/blend videos...",
            flush=True,
        )
        for offset, selected_index in enumerate(selected):
            source = train_real[int(selected_index)]
            source_video = Path(source["video"])
            destination = (
                output_root
                / "pseudo_fake"
                / f"{_safe_stem(source_video)}_blend.mp4"
            )
            print(
                f"[pseudo] {offset + 1}/{count} {source_video.name}",
                flush=True,
            )
            try:
                _blend_video(
                    source_video,
                    destination,
                    seed=int(args.seed) + offset,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                print(
                    f"[pseudo] WARN skip {source_video.name}: {exc}",
                    flush=True,
                )
                continue
            pseudo_records.append(
                _record(
                    video=destination,
                    au=Path(source["au"]),
                    label=1,
                    base_label=0,
                    group_id=source["group_id"],
                    augmentation="real_local_warp_blend_pseudofake",
                    source_video=source_video,
                )
            )

    media_records: list[dict[str, Any]] = []
    if not args.skip_media_variants:
        media_jobs: list[tuple[int, dict[str, Any], int, int]] = []
        for label, originals in ((0, train_real), (1, train_fake)):
            count = min(max(0, int(args.max_media_per_class)), len(originals))
            selected = rng.choice(len(originals), size=count, replace=False)
            for offset, selected_index in enumerate(selected):
                media_jobs.append(
                    (
                        label,
                        originals[int(selected_index)],
                        offset,
                        count,
                    )
                )
        print(
            f"[media] generating {len(media_jobs)} label-preserving variants...",
            flush=True,
        )
        for job_index, (label, source, offset, count) in enumerate(
            media_jobs,
            start=1,
        ):
            source_video = Path(source["video"])
            crf = DEFAULT_CRFS[offset % len(DEFAULT_CRFS)]
            destination = (
                output_root
                / "media"
                / ("real" if label == 0 else "generated")
                / f"{_safe_stem(source_video)}_le1280_crf{crf}.mp4"
            )
            print(
                f"[media] {job_index}/{len(media_jobs)} "
                f"{source_video.name} crf={crf}",
                flush=True,
            )
            try:
                _run_ffmpeg(
                    source_video,
                    destination,
                    crf=crf,
                    long_edge=MEDIA_LONG_EDGE,
                    fps=MEDIA_FPS,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                print(
                    f"[media] WARN skip {source_video.name}: {exc}",
                    flush=True,
                )
                continue
            media_records.append(
                _record(
                    video=destination,
                    au=Path(source["au"]),
                    label=label,
                    base_label=label,
                    group_id=source["group_id"],
                    augmentation=f"media_le{MEDIA_LONG_EDGE}_"
                    f"fps{MEDIA_FPS}_crf{crf}",
                    source_video=source_video,
                )
            )

    test_pairs = {
        "real": [
            {
                **dict(item),
                "label_generated": 0,
                "base_label": 0,
                "group_id": _group_id(_resolve(str(item["video"]))),
                "augmentation": "official_holdout",
            }
            for item in manifest["pairs"]["test"]["real"]
        ],
        "fake": [
            {
                **dict(item),
                "label_generated": 1,
                "base_label": 1,
                "group_id": _group_id(_resolve(str(item["video"]))),
                "augmentation": "official_holdout",
            }
            for item in manifest["pairs"]["test"]["fake"]
        ],
    }
    output = {
        "schema_version": "wangxing_v3_generalization_manifest_v1",
        "protocol": {
            "train_sources": [
                "data/MD_CL",
                "data/WangXing_Seedance",
                "data/_aug/wangxing_v3_generalization",
            ],
            "change_clips_in_train": False,
            "change_clips_path": "data/test/AI",
            "media_long_edge": MEDIA_LONG_EDGE,
            "media_fps": MEDIA_FPS,
            "media_crfs": list(DEFAULT_CRFS),
            "pseudo_fake_method": "local_frame_warp_blend",
            "group_split_required": True,
        },
        "pairs": {
            "train": {
                "real": train_real
                + [item for item in media_records if item["label_generated"] == 0],
                "fake": train_fake
                + [item for item in media_records if item["label_generated"] == 1]
                + pseudo_records,
            },
            "test": test_pairs,
        },
        "counts": {
            "train_real": len(train_real)
            + sum(item["label_generated"] == 0 for item in media_records),
            "train_fake": len(train_fake)
            + sum(item["label_generated"] == 1 for item in media_records)
            + len(pseudo_records),
            "test_real": len(test_pairs["real"]),
            "test_fake": len(test_pairs["fake"]),
            "pseudo_fake": len(pseudo_records),
            "media_variants": len(media_records),
        },
        "source_manifest": str(base_path),
    }
    output_path = _resolve(args.output_manifest)
    _write_json(output_path, output)
    print(json.dumps(output["counts"], ensure_ascii=False, indent=2))
    print(f"Wrote manifest: {output_path}")
    print("Change clips were not read and are not in the training pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

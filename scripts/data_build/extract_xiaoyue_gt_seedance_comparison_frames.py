"""Extract same-normalized-time comparison frames for XiaoYue videos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIME_FRACTIONS = (0.0, 0.14, 0.28, 0.42, 0.57, 0.71, 0.85, 1.0)
MOUTH_LANDMARKS = (61, 291, 13, 14, 78, 308)
DISPLAY_TITLES = {
    "GT": "GT实拍视频",
    "test1_seedance25": "test1:seedance2.5+gaussian+提示词",
    "test2_seedance25": "test2:seedance2.5+提示词",
}


def _path(value: str | Path) -> Path:
    target = Path(value).expanduser()
    return (target if target.is_absolute() else PROJECT_ROOT / target).resolve()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_at(rows: list[dict[str, str]], fraction: float) -> dict[str, str]:
    if not rows:
        return {}
    index = int(round(fraction * (len(rows) - 1)))
    return rows[max(0, min(index, len(rows) - 1))]


def _points(row: dict[str, str]) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    for index in range(478):
        try:
            x = float(row.get(f"lm_mp_{index}_x", ""))
            y = float(row.get(f"lm_mp_{index}_y", ""))
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            result[index] = (float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0)))
    return result


def _crop(
    frame: np.ndarray,
    points: dict[int, tuple[float, float]],
    indexes: Sequence[int] | None,
    *,
    margin_ratio: float,
    fallback: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    height, width = frame.shape[:2]
    selected = (
        [points[index] for index in indexes if index in points]
        if indexes is not None
        else list(points.values())
    )
    if len(selected) < 2:
        if fallback is None:
            return frame.copy()
        x0, y0, x1, y1 = fallback
    else:
        values = np.asarray(selected, dtype=np.float32)
        low = values.min(axis=0)
        high = values.max(axis=0)
        span = max(float(np.max(high - low)), 0.015)
        margin = span * margin_ratio
        x0 = int((low[0] - margin) * width)
        y0 = int((low[1] - margin) * height)
        x1 = int((high[0] + margin) * width)
        y1 = int((high[1] + margin) * height)
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    crop = frame[y0:y1, x0:x1]
    return crop if crop.size else frame.copy()


def _chinese_font(size: int):
    from PIL import ImageFont

    for candidate in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _label(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    """Draw Chinese-capable title bar over the frame."""
    from PIL import Image, ImageDraw

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bar_h = 52 if subtitle else 36
    canvas = Image.new("RGB", (rgb.shape[1], rgb.shape[0] + bar_h), (20, 20, 20))
    canvas.paste(Image.fromarray(rgb), (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    max_size = 24 if canvas.width > 420 else 16
    font = None
    for size in range(max_size, 11, -1):
        candidate = _chinese_font(size)
        if candidate.getbbox(title)[2] <= canvas.width - 16:
            font = candidate
            break
    font = font or _chinese_font(12)
    draw.text((10, 6 if subtitle else 8), title, fill=(255, 255, 255), font=font)
    if subtitle:
        sub_font = _chinese_font(13)
        draw.text((10, 30), subtitle, fill=(200, 200, 200), font=sub_font)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    source_height, source_width = image.shape[:2]
    scale = min(width / max(source_width, 1), height / max(source_height, 1))
    resized = cv2.resize(
        image,
        (max(1, int(round(source_width * scale))), max(1, int(round(source_height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y0 = (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def _contact_sheet(
    images: list[list[np.ndarray]],
    output: Path,
    *,
    tile_width: int,
    tile_height: int,
) -> None:
    rows: list[np.ndarray] = []
    for row in images:
        tiles = [
            _letterbox(image, tile_width, tile_height)
            for image in row
        ]
        rows.append(cv2.hconcat(tiles))
    sheet = cv2.vconcat(rows)
    _write_png(output, sheet)


def _write_png(path: Path, image: np.ndarray) -> None:
    """Write through bytes so OpenCV does not choke on Unicode parent paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"OpenCV PNG encoding failed: {path}")
    path.write_bytes(encoded.tobytes())


def _read_frame(
    capture: cv2.VideoCapture,
    frame_count: int,
    fraction: float,
) -> tuple[np.ndarray, int]:
    frame_index = int(round(fraction * max(frame_count - 1, 0)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to read frame {frame_index}.")
    return frame, frame_index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-video",
        default="data/xiaoyue/test/test1/GT.mp4",
    )
    parser.add_argument(
        "--gt-au",
        default="data/au/xiaoyue/test/test1_real_reference.csv",
    )
    parser.add_argument(
        "--test1-video",
        default="data/xiaoyue/test/test1/seedance2.5.mp4",
    )
    parser.add_argument(
        "--test1-au",
        default="data/au/xiaoyue/test/test1_seedance25.csv",
    )
    parser.add_argument(
        "--test2-video",
        default="data/xiaoyue/test/test2/seedance2.5.mp4",
    )
    parser.add_argument(
        "--test2-au",
        default="data/au/xiaoyue/test/test2_seedance25.csv",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/xiaoyue/gt_test1_test2_comparison_frames",
    )
    args = parser.parse_args(argv)

    specs = [
        ("GT", _path(args.gt_video), _path(args.gt_au)),
        ("test1_seedance25", _path(args.test1_video), _path(args.test1_au)),
        ("test2_seedance25", _path(args.test2_video), _path(args.test2_au)),
    ]
    output_root = _path(args.output_root)
    for _, video, au in specs:
        if not video.is_file() or not au.is_file():
            raise SystemExit(f"Missing comparison input: {video} / {au}")

    sheet_inputs: dict[str, list[list[np.ndarray]]] = {
        "full_frame": [],
        "face_crop": [],
        "mouth_crop": [],
    }
    timeline: list[dict[str, Any]] = []
    for fraction in TIME_FRACTIONS:
        full_row: list[np.ndarray] = []
        face_row: list[np.ndarray] = []
        mouth_row: list[np.ndarray] = []
        time_record: dict[str, Any] = {
            "fraction": fraction,
            "videos": {},
        }
        for label, video, au in specs:
            rows = _load_rows(au)
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise SystemExit(f"Unable to open comparison video: {video}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame, frame_index = _read_frame(capture, frame_count, fraction)
            capture.release()
            row = _row_at(rows, fraction)
            points = _points(row)
            timestamp = frame_index / fps if fps > 1e-6 else None
            time_tag = f"t{int(round(fraction * 100)):03d}"
            display_title = DISPLAY_TITLES.get(label, label)
            subtitle = (
                f"{time_tag} {timestamp:.2f}s"
                if timestamp is not None
                else time_tag
            )
            full = _label(frame, display_title, subtitle)
            face = _label(
                _crop(frame, points, None, margin_ratio=0.22),
                display_title,
                subtitle,
            )
            mouth = _label(
                _crop(frame, points, MOUTH_LANDMARKS, margin_ratio=1.8),
                display_title,
                subtitle,
            )
            for kind, image in (
                ("full_frame", full),
                ("face_crop", face),
                ("mouth_crop", mouth),
            ):
                target = output_root / kind / f"{time_tag}_{label}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                _write_png(target, image)
            full_row.append(full)
            face_row.append(face)
            mouth_row.append(mouth)
            time_record["videos"][label] = {
                "video": str(video),
                "au": str(au),
                "frame_index": frame_index,
                "frame_count": frame_count,
                "fps": fps,
                "timestamp_seconds": timestamp,
                "au_row_index": int(round(fraction * max(len(rows) - 1, 0))),
            }
        sheet_inputs["full_frame"].append(full_row)
        sheet_inputs["face_crop"].append(face_row)
        sheet_inputs["mouth_crop"].append(mouth_row)
        timeline.append(time_record)

    _contact_sheet(
        sheet_inputs["full_frame"],
        output_root / "contact_sheets" / "full_frame_timeline.png",
        tile_width=360,
        tile_height=500,
    )
    _contact_sheet(
        sheet_inputs["face_crop"],
        output_root / "contact_sheets" / "face_crop_timeline.png",
        tile_width=360,
        tile_height=360,
    )
    _contact_sheet(
        sheet_inputs["mouth_crop"],
        output_root / "contact_sheets" / "mouth_crop_timeline.png",
        tile_width=360,
        tile_height=260,
    )
    report = {
        "schema_version": "xiaoyue_comparison_frames_v1",
        "subject": "xiaoyue",
        "time_alignment": {
            "mode": "normalized_video_duration_fraction",
            "fractions": list(TIME_FRACTIONS),
            "gt_source": "test1/GT.mp4; test2/GT.mp4 is duplicate and omitted",
        },
        "videos": {
            label: {
                "video": str(video),
                "au": str(au),
            }
            for label, video, au in specs
        },
        "timeline": timeline,
        "outputs": {
            "full_frame": str((output_root / "full_frame").resolve()),
            "face_crop": str((output_root / "face_crop").resolve()),
            "mouth_crop": str((output_root / "mouth_crop").resolve()),
            "contact_sheets": str((output_root / "contact_sheets").resolve()),
        },
    }
    (output_root / "timeline_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["time_alignment"], ensure_ascii=False, indent=2))
    print(f"Output root: {output_root}")
    print(f"Timeline manifest: {output_root / 'timeline_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

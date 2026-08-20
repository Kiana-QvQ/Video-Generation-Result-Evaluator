"""Automatic perturbation robustness probes (no human scoring).

Injects blur / noise / flicker / frame-drop / temporal shuffle, and optionally
landmark jitter on an AU CSV, then checks that automatic scores decrease.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from evaluator.modules.core.paths import project_path
    from evaluator.modules.core.video_metrics import sample_video_frames
    from evaluator.modules.forensics.facial_motion import score_facial_motion
    from evaluator.modules.forensics.nr_vqa import extract_nr_vqa_features
    from evaluator.modules.forensics.perturbation import (
        run_frame_perturbation_battery,
        run_landmark_jitter_probe,
    )
    from evaluator.modules.forensics.texture_detail import (
        extract_texture_detail_features,
    )
except ImportError:
    # Also run directly from a flat collaborator host containing modules/.
    from modules.core.paths import project_path
    from modules.core.video_metrics import sample_video_frames
    from modules.forensics.facial_motion import score_facial_motion
    from modules.forensics.nr_vqa import extract_nr_vqa_features
    from modules.forensics.perturbation import (
        run_frame_perturbation_battery,
        run_landmark_jitter_probe,
    )
    from modules.forensics.texture_detail import extract_texture_detail_features


def _score_nr_vqa(frames: Sequence[Any]) -> float:
    result = extract_nr_vqa_features(frames, prefer_backends=("builtin_nr_vqa",))
    return float(result["score_0_1"])


def _score_texture(frames: Sequence[Any]) -> float:
    result = extract_texture_detail_features(
        frames,
        detect_faces=False,
        include_nr_vqa=True,
        nr_vqa_backends=("builtin_nr_vqa",),
    )
    features = result.get("features", {})
    return float(
        features.get(
            "training_free_texture_prior_0_1",
            features.get("nr_vqa_score_0_1", 0.5),
        )
    )


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No CSV rows to write.")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _landmark_score_fn(
    profile: dict[str, Any] | None,
) -> Any:
    def score_rows(rows: list[dict[str, str]]) -> float:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "jitter.csv"
            _write_csv_rows(csv_path, rows)
            scored = score_facial_motion(csv_path, profile or {})
            metrics = scored.get("metrics", {})
            value = metrics.get("ssl_au_score_0_1")
            if value is None:
                value = metrics.get("motion_coherence_0_1", 0.5)
            return float(value)

    return score_rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run automatic perturbation probes: blur/noise/flicker/drop/"
            "shuffle (+ optional landmark jitter). No human scores required."
        )
    )
    parser.add_argument("--video", help="Candidate video for frame perturbations.")
    parser.add_argument(
        "--au-csv",
        help="Optional AU/landmark CSV for landmark-jitter probe.",
    )
    parser.add_argument(
        "--profile",
        help="Optional forensics_profiles.json for facial-motion scoring.",
    )
    parser.add_argument(
        "--score-mode",
        choices=("nr_vqa", "texture"),
        default="nr_vqa",
        help="Frame score function used by the perturbation battery.",
    )
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--sample-fps", type=float, default=8.0)
    parser.add_argument("--min-drop", type=float, default=0.02)
    parser.add_argument(
        "--landmark-sigma",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--output",
        default="outputs/forensics/perturbation.json",
    )
    args = parser.parse_args(argv)

    if not args.video and not args.au_csv:
        raise SystemExit("Specify at least one of --video or --au-csv.")

    payload: dict[str, Any] = {
        "manual_scores_required": False,
        "proves_human_mos_equivalence": False,
        "video": str(project_path(args.video)) if args.video else None,
        "au_csv": str(project_path(args.au_csv)) if args.au_csv else None,
    }

    if args.video:
        video_path = project_path(args.video)
        if not video_path.is_file():
            raise SystemExit(f"Video not found: {video_path}")
        _, _, _, frames = sample_video_frames(
            video_path,
            args.max_frames,
            args.sample_fps,
        )
        if len(frames) < 2:
            raise SystemExit("Need at least two sampled frames.")
        score_fn = _score_texture if args.score_mode == "texture" else _score_nr_vqa
        payload["frame_battery"] = run_frame_perturbation_battery(
            frames,
            score_fn,
            min_drop=args.min_drop,
        )
        payload["score_mode"] = args.score_mode

    if args.au_csv:
        csv_path = project_path(args.au_csv)
        if not csv_path.is_file():
            raise SystemExit(f"AU CSV not found: {csv_path}")
        profile = None
        if args.profile:
            profile_path = project_path(args.profile)
            profiles = json.loads(profile_path.read_text(encoding="utf-8-sig"))
            profile = (
                profiles.get("facial_motion")
                if isinstance(profiles, dict)
                else None
            )
        rows = _load_csv_rows(csv_path)
        payload["landmark_jitter"] = run_landmark_jitter_probe(
            rows,
            _landmark_score_fn(profile),
            sigma=args.landmark_sigma,
            min_drop=min(args.min_drop, 0.01),
        )

    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    output.write_text(serialized, encoding="utf-8")
    print(serialized)

    frame_battery = payload.get("frame_battery")
    if isinstance(frame_battery, dict) and frame_battery.get("pass_ratio", 1.0) < 0.6:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

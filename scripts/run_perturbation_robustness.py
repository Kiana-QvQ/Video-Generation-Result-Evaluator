#!/usr/bin/env python
"""Run automatic perturbation robustness probes on texture / NR-VQA scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.video_metrics import sample_video_frames
from evaluator.modules.forensics.nr_vqa import extract_nr_vqa_features
from evaluator.modules.forensics.perturbation import run_frame_perturbation_battery
from evaluator.modules.forensics.texture_detail import extract_texture_detail_features


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatic degradation probes for no-reference scores.",
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--sample-fps", type=float, default=8.0)
    parser.add_argument("--min-drop", type=float, default=0.02)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    _, _, _, frames = sample_video_frames(
        args.video,
        args.max_frames,
        args.sample_fps,
    )
    if len(frames) < 2:
        raise SystemExit("Need at least two sampled frames.")

    def score_fn(candidate_frames):
        texture = extract_texture_detail_features(
            candidate_frames,
            max_frames=len(candidate_frames),
            sample_fps=args.sample_fps,
            detect_faces=False,
            include_nr_vqa=True,
        )
        nr = float(texture["features"].get("nr_vqa_score_0_1", 0.0))
        clarity = float(
            texture["features"].get("high_frequency_ratio_mean", 0.0)
        )
        return 0.7 * nr + 0.3 * min(1.0, clarity * 8.0)

    report = run_frame_perturbation_battery(
        frames,
        score_fn,
        min_drop=args.min_drop,
    )
    report["video"] = str(Path(args.video).resolve())
    report["builtin_nr_vqa"] = extract_nr_vqa_features(
        frames,
        prefer_backends=("builtin_nr_vqa",),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["pass_ratio"] >= 0.6 else 2


if __name__ == "__main__":
    raise SystemExit(main())

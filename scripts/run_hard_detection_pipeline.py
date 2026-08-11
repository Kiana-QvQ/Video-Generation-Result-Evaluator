"""Hard-detection training / evaluation pipeline (you run the training steps).

Steps:
1) rebuild quality-filtered forensics facial-motion profile (excludes holdout)
2) recalibrate fused score on non-holdout AU
3) evaluate holdout with hard labels only (coverage=100%)

This script never invents uncertain labels for the headline metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import profile_path, project_path
from evaluator.modules.forensics.authenticity_decision import metrics_from_decisions
from evaluator.modules.forensics.fused_hard_detector import score_fused_hard_detector
from evaluator.modules.forensics.seedance_authenticity import (
    fit_probability_calibrator,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def cmd_build_profile(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_forensics_profiles.py"),
        "--real-au-root",
        args.real_au_root,
        "--seedance-au-root",
        args.seedance_au_root,
        "--holdout-manifest",
        args.holdout_manifest,
        "--motion-only",
        "--min-landmark-ratio",
        str(args.min_landmark_ratio),
        "--min-pose-ratio",
        str(args.min_pose_ratio),
        "--output",
        args.forensics_profile_out,
    ]
    if args.include_texture:
        command.extend(
            [
                "--real-video-root",
                args.real_video_root,
                "--seedance-video-root",
                args.seedance_video_root,
                "--max-frames",
                str(args.max_frames),
                "--sample-fps",
                str(args.sample_fps),
            ]
        )
        # rebuild with texture: drop --motion-only
        command = [item for item in command if item != "--motion-only"]
    _run(command)
    return 0


def cmd_recalibrate(args: argparse.Namespace) -> int:
    holdout = _load_json(project_path(args.holdout_manifest))
    holdout_au = {
        str(project_path(item["au"]).resolve())
        for key in ("real", "seedance")
        for item in holdout.get(key, [])
    }
    forensics_profiles = _load_json(project_path(args.forensics_profile))
    source_profile = _load_json(
        project_path(args.source_profile)
        if args.source_profile
        else profile_path("wangxing_source_profile", required=True)
    )

    real_paths = [
        path
        for path in sorted(Path(args.real_au_root).rglob("*.csv"))
        if str(path.resolve()) not in holdout_au
    ][: args.limit]
    gen_paths = [
        path
        for path in sorted(Path(args.seedance_au_root).glob("*.csv"))
        if str(path.resolve()) not in holdout_au
    ][: args.limit]
    print(f"Recalibrate train sizes real={len(real_paths)} generated={len(gen_paths)}")

    real_scores: list[float] = []
    gen_scores: list[float] = []
    for index, path in enumerate(real_paths, start=1):
        scored = score_fused_hard_detector(
            au_path=path,
            wangxing_source_profile=source_profile,
            forensics_profiles=forensics_profiles,
            include_texture=False,
            hard_threshold=args.threshold,
        )
        value = scored.get("fused_real_0_1")
        if value is not None:
            real_scores.append(float(value))
        if index % 20 == 0:
            print(f"  real {index}/{len(real_paths)}", flush=True)
    for index, path in enumerate(gen_paths, start=1):
        scored = score_fused_hard_detector(
            au_path=path,
            wangxing_source_profile=source_profile,
            forensics_profiles=forensics_profiles,
            include_texture=False,
            hard_threshold=args.threshold,
        )
        value = scored.get("fused_real_0_1")
        if value is not None:
            gen_scores.append(float(value))
        if index % 20 == 0:
            print(f"  generated {index}/{len(gen_paths)}", flush=True)

    if len(real_scores) < 4 or len(gen_scores) < 4:
        raise SystemExit("Need >=4 real and generated non-holdout scores.")
    calibrator = fit_probability_calibrator(real_scores, gen_scores)
    calibrator.update(
        {
            "feature": "fused_real_0_1",
            "manual_scores_required": False,
            "uncertain_band_used": False,
            "train_real_count": len(real_scores),
            "train_generated_count": len(gen_scores),
            "forensics_profile": str(project_path(args.forensics_profile)),
            "source_profile": str(
                project_path(args.source_profile)
                if args.source_profile
                else profile_path("wangxing_source_profile", required=True)
            ),
        }
    )
    output = project_path(args.calibrator_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(calibrator, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.update_forensics_profile:
        profile_path_obj = project_path(args.forensics_profile)
        profiles = _load_json(profile_path_obj)
        profiles["fused_hard_calibrator"] = calibrator
        profile_path_obj.write_text(
            json.dumps(profiles, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Updated forensics profile calibrator block: {profile_path_obj}")
    print(f"Wrote calibrator: {output}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    holdout = _load_json(project_path(args.holdout_manifest))
    forensics_profiles = _load_json(project_path(args.forensics_profile))
    source_profile = _load_json(
        project_path(args.source_profile)
        if args.source_profile
        else profile_path("wangxing_source_profile", required=True)
    )
    calibrator = None
    calibrator_path = project_path(args.calibrator) if args.calibrator else None
    if calibrator_path and calibrator_path.is_file():
        calibrator = _load_json(calibrator_path)
    elif isinstance(forensics_profiles.get("fused_hard_calibrator"), dict):
        calibrator = forensics_profiles["fused_hard_calibrator"]

    samples: list[dict[str, Any]] = []
    for item in holdout.get("real", []):
        samples.append({"label_generated": 0, "source_label": "real", **item})
    for item in holdout.get("seedance", []):
        samples.append(
            {"label_generated": 1, "source_label": "generated", **item}
        )

    labels: list[int] = []
    decisions: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        au_path = project_path(sample["au"])
        video_path = project_path(sample["video"]) if sample.get("video") else None
        if not au_path.is_file():
            continue
        scored = score_fused_hard_detector(
            au_path=au_path,
            video_path=video_path,
            wangxing_source_profile=source_profile,
            forensics_profiles=forensics_profiles,
            fused_calibrator=calibrator,
            include_texture=bool(args.include_texture),
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
            hard_threshold=args.threshold,
        )
        decision = scored["hard_decision"]
        labels.append(int(sample["label_generated"]))
        decisions.append(decision)
        rows.append(
            {
                "index": index,
                "source_label": sample["source_label"],
                "label_generated": int(sample["label_generated"]),
                "au": str(au_path),
                "decision": decision,
                "decision_score_0_1": scored.get("decision_score_0_1"),
                "fusion": scored.get("fusion"),
            }
        )
        print(
            f"[{index}/{len(samples)}] {sample['source_label']} "
            f"score={scored.get('decision_score_0_1')} "
            f"pred={decision.get('decision')}",
            flush=True,
        )

    metrics = metrics_from_decisions(labels, decisions)
    payload = {
        "schema_version": "fused_hard_detection_metrics_v1",
        "uncertain_band_used": False,
        "coverage_expected": 1.0,
        "threshold": args.threshold,
        "include_texture": bool(args.include_texture),
        "max_frames": args.max_frames,
        "headline": {
            "generated_recall": metrics.get("generated_recall"),
            "overall_accuracy": metrics.get("accuracy"),
            "generated_precision": metrics.get("generated_precision"),
            "real_recall": metrics.get("real_recall"),
            "coverage": metrics.get("coverage"),
        },
        "metrics": metrics,
        "rows": rows,
        "note": (
            "Hard fused detector (Wang Xing source + forensics). "
            "No uncertain class in headline metrics."
        ),
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["headline"], ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hard fused detector pipeline (build / recalibrate / evaluate)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "build-profile",
        help="Rebuild quality-filtered forensics profile (excludes holdout).",
    )
    build.add_argument("--real-au-root", default="data/au/MD_CL")
    build.add_argument("--seedance-au-root", default="data/au/WangXing_Seedance")
    build.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
    )
    build.add_argument("--min-landmark-ratio", type=float, default=0.45)
    build.add_argument("--min-pose-ratio", type=float, default=0.35)
    build.add_argument(
        "--forensics-profile-out",
        default="outputs/forensics/forensics_profiles_quality_filtered.json",
    )
    build.add_argument("--include-texture", action="store_true")
    build.add_argument("--real-video-root", default="data/MD_CL")
    build.add_argument("--seedance-video-root", default="data/WangXing_Seedance")
    build.add_argument("--max-frames", type=int, default=24)
    build.add_argument("--sample-fps", type=float, default=8.0)
    build.set_defaults(func=cmd_build_profile)

    recal = sub.add_parser(
        "recalibrate",
        help="Fit fused hard calibrator on non-holdout AU.",
    )
    recal.add_argument("--real-au-root", default="data/au/MD_CL")
    recal.add_argument("--seedance-au-root", default="data/au/WangXing_Seedance")
    recal.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
    )
    recal.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles_quality_filtered.json",
    )
    recal.add_argument("--source-profile", default="")
    recal.add_argument("--limit", type=int, default=80)
    recal.add_argument("--threshold", type=float, default=0.5)
    recal.add_argument(
        "--calibrator-out",
        default="outputs/forensics/fused_hard_calibrator.json",
    )
    recal.add_argument(
        "--update-forensics-profile",
        action="store_true",
        help="Embed calibrator into the forensics profile JSON.",
    )
    recal.set_defaults(func=cmd_recalibrate)

    evaluate = sub.add_parser(
        "evaluate",
        help="Evaluate holdout with hard fused detector (coverage=100%%).",
    )
    evaluate.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
    )
    evaluate.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles_quality_filtered.json",
    )
    evaluate.add_argument("--source-profile", default="")
    evaluate.add_argument(
        "--calibrator",
        default="outputs/forensics/fused_hard_calibrator.json",
    )
    evaluate.add_argument("--threshold", type=float, default=0.5)
    evaluate.add_argument("--include-texture", action="store_true")
    evaluate.add_argument("--max-frames", type=int, default=24)
    evaluate.add_argument("--sample-fps", type=float, default=8.0)
    evaluate.add_argument(
        "--output",
        default="outputs/forensics/fused_hard_detection_metrics.json",
    )
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

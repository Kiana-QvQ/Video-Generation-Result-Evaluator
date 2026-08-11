"""Compute generated-video recall and overall accuracy on the holdout split.

Uses the current forensics pipeline (facial-motion profile + authenticity
calibrator + SSL/physio features). Positive class = generated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from evaluator.modules.forensics import analyze_forensics


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _decision_score(report: dict[str, Any]) -> tuple[float | None, str]:
    scores = report.get("scores", {})
    calibrated = _finite(scores.get("calibrated_real_probability_0_1"))
    if calibrated is not None:
        return calibrated, "calibrated_real_probability_0_1"
    raw = _finite(scores.get("raw_real_domain_evidence_0_1"))
    if raw is not None:
        return raw, "raw_real_domain_evidence_0_1"
    facial = _finite(scores.get("facial_expression_muscle_score_0_1"))
    if facial is not None:
        return facial, "facial_expression_muscle_score_0_1"
    return None, "unavailable"


def _metrics(
    labels: Sequence[int],
    preds: Sequence[int],
) -> dict[str, Any]:
    # label/pred: 1 = generated, 0 = real
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    generated = tp + fn
    real = tn + fp
    total = len(labels)
    return {
        "generated_recall": tp / generated if generated else None,
        "generated_precision": tp / (tp + fp) if (tp + fp) else None,
        "real_recall": tn / real if real else None,
        "accuracy": (tp + tn) / total if total else None,
        "confusion": {
            "tp_generated": tp,
            "tn_real": tn,
            "fp_real_as_generated": fp,
            "fn_generated_as_real": fn,
        },
        "counts": {
            "generated": generated,
            "real": real,
            "total": total,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated-video recall and overall accuracy on holdout."
        )
    )
    parser.add_argument(
        "--profile",
        default="evaluator/modules/assets/profiles/forensics_profiles.json",
    )
    parser.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Predict generated when real-score < threshold.",
    )
    parser.add_argument(
        "--include-texture",
        action="store_true",
        help="Also score texture branch (slower).",
    )
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Optional cap per class (0 = all holdout).",
    )
    parser.add_argument(
        "--output",
        default="outputs/forensics/holdout_detection_metrics.json",
    )
    args = parser.parse_args(argv)

    profile_path = project_path(args.profile)
    holdout_path = project_path(args.holdout_manifest)
    profiles = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8-sig"))

    real_items = list(holdout.get("real", []))
    gen_items = list(holdout.get("seedance", []))
    if args.limit_per_class > 0:
        real_items = real_items[: args.limit_per_class]
        gen_items = gen_items[: args.limit_per_class]

    samples: list[dict[str, Any]] = []
    for item in real_items:
        samples.append({"label": 0, "source_label": "real", **item})
    for item in gen_items:
        samples.append({"label": 1, "source_label": "generated", **item})

    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    preds: list[int] = []
    score_key_used = None
    for index, sample in enumerate(samples, start=1):
        au_path = project_path(sample["au"])
        video_path = project_path(sample["video"]) if sample.get("video") else None
        if not au_path.is_file():
            rows.append(
                {
                    "index": index,
                    "source_label": sample["source_label"],
                    "au": str(au_path),
                    "status": "missing_au",
                }
            )
            continue
        texture_input = None
        if args.include_texture and video_path is not None and video_path.is_file():
            texture_input = video_path
        try:
            report = analyze_forensics(
                facial_motion=au_path,
                facial_motion_profile=profiles.get("facial_motion"),
                texture_detail=texture_input,
                texture_detail_profile=profiles.get("texture_detail"),
                authenticity_calibrator=profiles.get("authenticity_calibrator"),
                max_frames=args.max_frames,
                sample_fps=args.sample_fps,
                detect_faces=False,
            )
        except Exception as exc:  # noqa: BLE001 - batch evaluation must continue
            rows.append(
                {
                    "index": index,
                    "source_label": sample["source_label"],
                    "au": str(au_path),
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue

        score, score_key = _decision_score(report)
        score_key_used = score_key
        if score is None:
            rows.append(
                {
                    "index": index,
                    "source_label": sample["source_label"],
                    "au": str(au_path),
                    "status": "no_score",
                }
            )
            continue
        pred = 1 if score < args.threshold else 0
        labels.append(int(sample["label"]))
        preds.append(pred)
        scores = report.get("scores", {})
        rows.append(
            {
                "index": index,
                "source_label": sample["source_label"],
                "label_generated": int(sample["label"]),
                "pred_generated": pred,
                "correct": int(pred == int(sample["label"])),
                "decision_score_real": score,
                "decision_score_key": score_key,
                "au": str(au_path),
                "ssl_au_score_0_1": scores.get("ssl_au_score_0_1"),
                "ssl_backbone_score_0_1": scores.get("ssl_backbone_score_0_1"),
                "physio_rhythm_score_0_1": scores.get("physio_rhythm_score_0_1"),
                "nr_vqa_score_0_1": scores.get("nr_vqa_score_0_1"),
                "freq_forensics_score_0_1": scores.get("freq_forensics_score_0_1"),
                "status": "ok",
            }
        )
        print(
            f"[{index}/{len(samples)}] {sample['source_label']} "
            f"score={score:.3f} pred={'generated' if pred else 'real'} "
            f"{'OK' if pred == int(sample['label']) else 'MISS'}"
        )

    metrics = _metrics(labels, preds)
    payload = {
        "schema_version": "holdout_detection_metrics_v1",
        "profile": str(profile_path),
        "holdout_manifest": str(holdout_path),
        "threshold": args.threshold,
        "decision_rule": (
            f"predict generated if {score_key_used or 'real_score'} < {args.threshold}"
        ),
        "positive_class": "generated",
        "include_texture": bool(args.include_texture),
        "scored_count": len(labels),
        "metrics": metrics,
        "headline": {
            "generated_recall": metrics.get("generated_recall"),
            "overall_accuracy": metrics.get("accuracy"),
        },
        "rows": rows,
        "note": (
            "Holdout real/generated source labels only; no manual MOS scores. "
            "Recall is for the generated class; accuracy is over real+generated."
        ),
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["headline"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return 0 if metrics.get("accuracy") is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())

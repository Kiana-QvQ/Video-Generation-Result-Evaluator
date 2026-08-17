"""Evaluate Wang Xing specialization authenticity (AU learned head + video .pt).

Project-side only. Peer evaluator host code is not modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import profile_path, project_path
from evaluator.modules.forensics.authenticity_decision import metrics_from_decisions
from evaluator.modules.forensics.learned_fusion_head import load_learned_head
from wangxing_project.model_slots import list_model_slots
from wangxing_project.specialization_fused import (
    score_wangxing_specialization_authenticity,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _is_unscorable(
    decision: dict[str, Any],
    min_quality: float,
    *,
    quality_gate: bool,
) -> bool:
    if not quality_gate:
        return bool(
            decision.get("decision") == "uncertain"
            and decision.get("manual_scores_required")
        )
    if decision.get("manual_scores_required"):
        return True
    quality = decision.get("quality_0_1")
    try:
        return quality is not None and float(quality) < float(min_quality)
    except (TypeError, ValueError):
        return False


def cmd_slots(_: argparse.Namespace) -> int:
    print(json.dumps(list_model_slots(), ensure_ascii=False, indent=2))
    return 0


def _resolve_source_profile(args: argparse.Namespace) -> Path:
    if args.source_profile:
        path = project_path(args.source_profile)
        if not path.is_file():
            raise SystemExit(f"Source profile not found: {path}")
        return path
    preferred = project_path(
        "outputs/forensics/wangxing_source_profile_holdout_excluded.json"
    )
    if preferred.is_file():
        return preferred
    bundled = profile_path("wangxing_source_profile", required=False)
    if bundled is not None and bundled.is_file():
        return bundled
    raise SystemExit(
        "No source profile found. Pass --source-profile or build "
        "outputs/forensics/wangxing_source_profile_holdout_excluded.json"
    )


def cmd_score_one(args: argparse.Namespace) -> int:
    forensics = _load_json(project_path(args.forensics_profile))
    source = _load_json(_resolve_source_profile(args))
    head = load_learned_head(project_path(args.learned_head))
    video_arg = (args.video or "").strip()
    scored = score_wangxing_specialization_authenticity(
        au_path=project_path(args.au),
        video_path=project_path(video_arg) if video_arg else None,
        wangxing_source_profile=source,
        forensics_profiles=forensics,
        learned_head=head,
        pt_model_path=args.pt_model or None,
        pt_cache_dir=project_path(args.pt_cache_dir),
        use_pt=not args.no_pt,
        au_weight=args.au_weight,
        pt_weight=args.pt_weight,
        hard_threshold=args.threshold,
        min_quality=args.min_quality,
        allow_uncertain=args.quality_gate,
    )
    print(json.dumps(scored, ensure_ascii=False, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    holdout = _load_json(project_path(args.holdout_manifest))
    forensics = _load_json(project_path(args.forensics_profile))
    source_path = _resolve_source_profile(args)
    source = _load_json(source_path)
    head = load_learned_head(project_path(args.learned_head))

    real_samples = [
        {"label_generated": 0, "source_label": "real", **item}
        for item in holdout.get("real", [])
    ]
    gen_samples = [
        {"label_generated": 1, "source_label": "generated", **item}
        for item in holdout.get("seedance", [])
    ]
    if args.limit and args.limit > 0:
        # Balanced smoke: half real / half generated when possible.
        half = max(1, args.limit // 2)
        samples = real_samples[:half] + gen_samples[: args.limit - half]
    else:
        samples = real_samples + gen_samples

    labels: list[int] = []
    decisions: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    unscorable_samples: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        au_path = project_path(sample["au"])
        video_path = project_path(sample["video"]) if sample.get("video") else None
        if not au_path.is_file():
            print(f"[{index}/{len(samples)}] skip missing AU {au_path}", flush=True)
            continue
        scored = score_wangxing_specialization_authenticity(
            au_path=au_path,
            video_path=video_path,
            wangxing_source_profile=source,
            forensics_profiles=forensics,
            learned_head=head,
            pt_model_path=args.pt_model or None,
            pt_cache_dir=project_path(args.pt_cache_dir),
            use_pt=not args.no_pt,
            au_weight=args.au_weight,
            pt_weight=args.pt_weight,
            hard_threshold=args.threshold,
            min_quality=args.min_quality,
            allow_uncertain=args.quality_gate,
        )
        decision = scored["hard_decision"]
        unscorable = _is_unscorable(
            decision,
            args.min_quality,
            quality_gate=args.quality_gate,
        )
        labels.append(int(sample["label_generated"]))
        decisions.append(decision)
        row = {
            "index": index,
            "source_label": sample["source_label"],
            "label_generated": int(sample["label_generated"]),
            "au": str(au_path),
            "video": str(video_path) if video_path else None,
            "decision_score_0_1": scored.get("decision_score_0_1"),
            "hard_decision": decision,
            "unscorable": unscorable,
            "branches": {
                "au": scored["branches"]["au_learned_head"].get(
                    "real_probability_0_1"
                ),
                "pt": scored["branches"]["video_dual_pt"].get(
                    "real_probability_0_1"
                ),
                "pt_status": scored["branches"]["video_dual_pt"].get("status"),
            },
            "fusion": scored.get("fusion"),
        }
        rows.append(row)
        if unscorable:
            unscorable_samples.append(
                {
                    "index": index,
                    "source_label": sample["source_label"],
                    "label_generated": int(sample["label_generated"]),
                    "au": str(au_path),
                    "video": str(video_path) if video_path else None,
                    "quality_0_1": decision.get("quality_0_1"),
                    "reasons": decision.get("reasons", []),
                }
            )
        print(
            f"[{index}/{len(samples)}] {sample['source_label']} "
            f"score={scored.get('decision_score_0_1')} "
            f"pred={decision.get('decision')} "
            f"unscorable={unscorable} "
            f"pt={scored['branches']['video_dual_pt'].get('status')}",
            flush=True,
        )

    metrics = metrics_from_decisions(labels, decisions)
    strict_metrics = metrics_from_decisions(
        labels,
        decisions,
        include_uncertain_as_error=True,
    )
    payload = {
        "schema_version": "wangxing_specialization_fused_holdout_metrics_v2",
        "source_profile": str(source_path),
        "learned_head": str(project_path(args.learned_head)),
        "forensics_profile": str(project_path(args.forensics_profile)),
        "use_pt": not args.no_pt,
        "au_weight": args.au_weight,
        "pt_weight": args.pt_weight,
        "threshold": args.threshold,
        "min_quality": args.min_quality,
        "quality_gate": {
            "enabled": bool(args.quality_gate),
            "min_quality": args.min_quality,
            "unscorable_count": len(unscorable_samples),
        },
        "model_slots": list_model_slots(),
        "headline": {
            "generated_recall": metrics.get("generated_recall"),
            "overall_accuracy": metrics.get("accuracy"),
            "generated_precision": metrics.get("generated_precision"),
            "real_recall": metrics.get("real_recall"),
            "coverage": metrics.get("coverage"),
            "evaluable_count": metrics.get("decided_count"),
            "unscorable_count": len(unscorable_samples),
        },
        "metrics": metrics,
        "metrics_if_unscorable_counted_as_error": strict_metrics,
        "unscorable_samples": unscorable_samples,
        "rows": rows,
        "note": (
            "Project-side Wang Xing specialization authenticity fusion "
            "(AU learned multi-technique head + dual-scale video .pt). "
            "Low-quality samples are reported as unscorable/manual review "
            "and excluded from evaluable headline metrics."
        ),
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["headline"], ensure_ascii=False, indent=2))
    print(
        f"Unscorable/manual review: {len(unscorable_samples)} "
        f"(min_quality={args.min_quality})"
    )
    print(f"Wrote {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wang Xing specialization authenticity (AU + .pt) — project side."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    slots = sub.add_parser("slots", help="List reserved model slots.")
    slots.set_defaults(func=cmd_slots)

    one = sub.add_parser("score-one", help="Score one AU/video into specialization block.")
    one.add_argument("--au", required=True)
    one.add_argument("--video", default="")
    one.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles_quality_filtered.json",
    )
    one.add_argument("--source-profile", default="")
    one.add_argument(
        "--learned-head",
        default="outputs/forensics/learned_fusion_head_logistic_noleak.json",
    )
    one.add_argument("--pt-model", default="")
    one.add_argument("--pt-cache-dir", default="outputs/vedio_pred/cache")
    one.add_argument("--no-pt", action="store_true")
    one.add_argument("--au-weight", type=float, default=0.65)
    one.add_argument("--pt-weight", type=float, default=0.35)
    one.add_argument("--threshold", type=float, default=None)
    one.add_argument("--min-quality", type=float, default=0.45)
    one.add_argument(
        "--quality-gate",
        action="store_true",
        help="Refuse hard labels for samples below --min-quality.",
    )
    one.set_defaults(func=cmd_score_one)

    evaluate = sub.add_parser(
        "evaluate",
        help="Holdout metrics for fused Wang Xing specialization authenticity.",
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
        "--learned-head",
        default="outputs/forensics/learned_fusion_head_logistic_noleak.json",
    )
    evaluate.add_argument("--pt-model", default="")
    evaluate.add_argument("--pt-cache-dir", default="outputs/vedio_pred/cache")
    evaluate.add_argument("--no-pt", action="store_true")
    evaluate.add_argument("--au-weight", type=float, default=0.65)
    evaluate.add_argument("--pt-weight", type=float, default=0.35)
    evaluate.add_argument("--threshold", type=float, default=None)
    evaluate.add_argument(
        "--min-quality",
        type=float,
        default=0.45,
        help="Samples below this quality are uncertain/manual review.",
    )
    evaluate.add_argument(
        "--quality-gate",
        action="store_true",
        help="Refuse hard labels for samples below --min-quality.",
    )
    evaluate.add_argument("--limit", type=int, default=0)
    evaluate.add_argument(
        "--output",
        default="outputs/forensics/wangxing_specialization_fused_holdout_metrics.json",
    )
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "min_quality") and not 0.0 <= args.min_quality <= 1.0:
        raise SystemExit("--min-quality must be between 0 and 1.")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

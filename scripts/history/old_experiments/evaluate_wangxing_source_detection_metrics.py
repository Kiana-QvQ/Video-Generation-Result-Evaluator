"""Wang Xing specialization: generated recall + overall accuracy.

Uses the bundled Wang Xing source profile (real_wangxing vs generated_wangxing)
on the forensics holdout split. Optionally applies uncertainty band + quality
gate. Identity frame count (8 vs 24) is reported separately when enabled.
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
from evaluator.modules.forensics.authenticity_decision import (
    decide_real_vs_generated,
    metrics_from_decisions,
)
from evaluator.modules.forensics.seedance_authenticity import (
    fit_probability_calibrator,
)
from evaluator.modules.wangxing.wangxing_specialization import (
    evaluate_identity_profile,
    score_source_profile,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _metrics_table(rows: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    return {name: metrics for name, metrics in rows}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Wang Xing source specialization on holdout: "
            "generated recall and overall accuracy."
        )
    )
    parser.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
    )
    parser.add_argument(
        "--source-profile",
        default="",
        help="Defaults to bundled wangxing_source_profile.json",
    )
    parser.add_argument(
        "--identity-profile",
        default="",
        help="Optional; enables identity branch with --identity-frames.",
    )
    parser.add_argument(
        "--identity-frames",
        type=int,
        nargs="+",
        default=[],
        help="E.g. 8 24 to compare identity frame sampling.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--uncertain-low", type=float, default=0.35)
    parser.add_argument("--uncertain-high", type=float, default=0.65)
    parser.add_argument("--min-quality", type=float, default=0.45)
    parser.add_argument(
        "--recalibrate-non-holdout",
        action="store_true",
        help="Fit Platt calibrator on non-holdout AU before scoring holdout.",
    )
    parser.add_argument("--recalibrate-limit", type=int, default=60)
    parser.add_argument(
        "--output",
        default="outputs/forensics/wangxing_source_detection_metrics.json",
    )
    args = parser.parse_args(argv)

    holdout = _load_json(project_path(args.holdout_manifest))
    source_profile_path = (
        project_path(args.source_profile)
        if args.source_profile
        else profile_path("wangxing_source_profile", required=True)
    )
    source_profile = _load_json(source_profile_path)

    samples: list[dict[str, Any]] = []
    for item in holdout.get("real", []):
        samples.append({"label_generated": 0, "source_label": "real", **item})
    for item in holdout.get("seedance", []):
        samples.append(
            {"label_generated": 1, "source_label": "generated", **item}
        )

    holdout_au = {
        str(project_path(sample["au"]).resolve()) for sample in samples
    }

    calibrator = None
    if args.recalibrate_non_holdout:
        real_train = [
            path
            for path in sorted(Path("data/au/MD_CL").rglob("*.csv"))
            if str(path.resolve()) not in holdout_au
        ][: args.recalibrate_limit]
        gen_train = [
            path
            for path in sorted(Path("data/au/WangXing_Seedance").glob("*.csv"))
            if str(path.resolve()) not in holdout_au
        ][: args.recalibrate_limit]
        real_scores: list[float] = []
        gen_scores: list[float] = []
        for path in real_train:
            scored = score_source_profile(path, source_profile)
            value = scored.get("real_probability_0_1")
            if value is not None:
                real_scores.append(float(value))
        for path in gen_train:
            scored = score_source_profile(path, source_profile)
            value = scored.get("real_probability_0_1")
            if value is not None:
                gen_scores.append(float(value))
        if len(real_scores) >= 4 and len(gen_scores) >= 4:
            calibrator = fit_probability_calibrator(real_scores, gen_scores)
            print(
                f"Recalibrated on non-holdout: real={len(real_scores)} "
                f"generated={len(gen_scores)}"
            )
        else:
            print("Recalibration skipped: insufficient non-holdout scores.")

    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    hard_decisions: list[dict[str, Any]] = []
    band_decisions: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        au_path = project_path(sample["au"])
        if not au_path.is_file():
            continue
        scored = score_source_profile(au_path, source_profile)
        real_prob = scored.get("real_probability_0_1")
        quality = None
        quality_blob = scored.get("quality") or {}
        if isinstance(quality_blob, dict):
            quality = quality_blob.get("valid_frame_ratio")
        hard = decide_real_vs_generated(
            real_score_0_1=real_prob,
            quality_0_1=quality,
            calibrator=calibrator,
            hard_threshold=0.5,
            allow_uncertain=False,
        )
        band = decide_real_vs_generated(
            real_score_0_1=real_prob,
            quality_0_1=quality,
            calibrator=calibrator,
            hard_threshold=0.5,
            uncertain_low=args.uncertain_low,
            uncertain_high=args.uncertain_high,
            min_quality=args.min_quality,
            allow_uncertain=True,
        )
        labels.append(int(sample["label_generated"]))
        hard_decisions.append(hard)
        band_decisions.append(band)
        rows.append(
            {
                "index": index,
                "source_label": sample["source_label"],
                "label_generated": int(sample["label_generated"]),
                "au": str(au_path),
                "wangxing_source": scored,
                "hard_decision": hard,
                "band_decision": band,
            }
        )
        print(
            f"[{index}/{len(samples)}] {sample['source_label']} "
            f"real_p={real_prob} hard={hard['decision']} band={band['decision']}"
        )

    hard_metrics = metrics_from_decisions(labels, hard_decisions)
    band_metrics = metrics_from_decisions(labels, band_decisions)
    band_as_error = metrics_from_decisions(
        labels,
        band_decisions,
        include_uncertain_as_error=True,
    )

    identity_compare: dict[str, Any] = {}
    if args.identity_frames:
        identity_path = (
            project_path(args.identity_profile)
            if args.identity_profile
            else profile_path("wangxing_identity_profile", required=True)
        )
        identity_profile = _load_json(identity_path)
        from evaluator.modules.core.holistic_evaluator import (
            _FaceDetector,
            _IdentityBackend,
        )

        backend = _IdentityBackend(_FaceDetector(), device=args.device)
        # Compare only on a small balanced subset for runtime.
        subset = samples[:10] + samples[-10:]
        for frame_count in args.identity_frames:
            id_labels: list[int] = []
            id_decisions: list[dict[str, Any]] = []
            for sample in subset:
                video = project_path(sample["video"])
                if not video.is_file():
                    continue
                try:
                    identity = evaluate_identity_profile(
                        video,
                        identity_profile,
                        backend,
                        max_frames=int(frame_count),
                    )
                except Exception as exc:  # noqa: BLE001
                    id_decisions.append(
                        {
                            "predicted_generated": None,
                            "decision": "uncertain",
                            "error": str(exc),
                        }
                    )
                    id_labels.append(int(sample["label_generated"]))
                    continue
                # Higher generated prototype similarity => more generated-like.
                gen_sim = float(
                    identity.get("metrics", {}).get(
                        "generated_prototype_similarity",
                        0.0,
                    )
                    if isinstance(identity.get("metrics"), dict)
                    else 0.0
                )
                real_sim = float(
                    identity.get("metrics", {}).get(
                        "real_prototype_similarity",
                        0.0,
                    )
                    if isinstance(identity.get("metrics"), dict)
                    else 0.0
                )
                # Convert to real_score in [0,1]-ish via softmax-like ratio.
                total = abs(real_sim) + abs(gen_sim) + 1e-8
                real_score = (real_sim + 1.0) / (
                    (real_sim + 1.0) + (gen_sim + 1.0)
                )
                del total
                decision = decide_real_vs_generated(
                    real_score_0_1=real_score,
                    quality_0_1=identity.get("valid_frame_ratio"),
                    allow_uncertain=False,
                )
                id_labels.append(int(sample["label_generated"]))
                id_decisions.append(decision)
            identity_compare[str(frame_count)] = {
                "subset_size": len(id_labels),
                "metrics_hard": metrics_from_decisions(id_labels, id_decisions),
                "note": (
                    "Identity frame comparison on a 20-clip subset; "
                    "source-profile metrics above are the primary detection numbers."
                ),
            }

    payload = {
        "schema_version": "wangxing_source_detection_metrics_v1",
        "source_profile": str(source_profile_path),
        "holdout_manifest": str(project_path(args.holdout_manifest)),
        "scored_count": len(labels),
        "recalibrator_used": calibrator is not None,
        "headline": {
            "method": "wangxing_source_profile",
            "hard_threshold_0_5": {
                "generated_recall": hard_metrics.get("generated_recall"),
                "overall_accuracy": hard_metrics.get("accuracy"),
                "generated_precision": hard_metrics.get("generated_precision"),
                "real_recall": hard_metrics.get("real_recall"),
                "coverage": hard_metrics.get("coverage"),
            },
            "uncertain_band_plus_quality_gate": {
                "generated_recall": band_metrics.get("generated_recall"),
                "overall_accuracy": band_metrics.get("accuracy"),
                "generated_precision": band_metrics.get("generated_precision"),
                "real_recall": band_metrics.get("real_recall"),
                "coverage": band_metrics.get("coverage"),
            },
        },
        "metrics": _metrics_table(
            [
                ("hard_threshold_0_5", hard_metrics),
                ("uncertain_band_quality_gate", band_metrics),
                (
                    "uncertain_counted_as_error_full_coverage",
                    band_as_error,
                ),
            ]
        ),
        "identity_frame_compare": identity_compare,
        "policy": {
            "uncertain_low": args.uncertain_low,
            "uncertain_high": args.uncertain_high,
            "min_quality": args.min_quality,
            "explanation": (
                "Hard mode labels every clip. Band+quality refuses mid-score "
                "or low face-mesh coverage clips; coverage then drops and "
                "accuracy is only over decided clips."
            ),
        },
        "rows": rows,
        "note": (
            "Primary detector is Wang Xing source specialization "
            "(AU sequence domain profile), not forensics-only. "
            "Identity 8-vs-24 is optional and subset-only."
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


if __name__ == "__main__":
    raise SystemExit(main())

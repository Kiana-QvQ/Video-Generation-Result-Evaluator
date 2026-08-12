"""Train / evaluate learned fusion head for hard real-vs-generated detection.

Uses non-holdout AU only for training. Holdout is evaluation-only.
Aims for generated_recall and overall_accuracy in the 0.75-0.85 band.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import profile_path, project_path
from evaluator.modules.forensics.authenticity_decision import metrics_from_decisions
from evaluator.modules.forensics.learned_fusion_head import (
    extract_fusion_features,
    fit_learned_fusion_head,
    load_learned_head,
    save_learned_head,
    score_with_learned_head,
    select_threshold,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _holdout_au_set(manifest: dict[str, Any]) -> set[str]:
    return {
        str(project_path(item["au"]).resolve())
        for key in ("real", "seedance")
        for item in manifest.get(key, [])
    }


def _collect_train_paths(
    *,
    real_au_root: Path,
    seedance_au_root: Path,
    holdout_au: set[str],
    real_limit: int,
    seedance_limit: int,
    real_per_generated: float,
    random_state: int,
) -> tuple[list[Path], list[Path]]:
    gen_paths = [
        path
        for path in sorted(seedance_au_root.glob("*.csv"))
        if str(path.resolve()) not in holdout_au
    ][:seedance_limit]
    real_pool = [
        path
        for path in sorted(real_au_root.rglob("*.csv"))
        if str(path.resolve()) not in holdout_au
    ]
    target_real = min(
        len(real_pool),
        real_limit,
        max(int(round(len(gen_paths) * real_per_generated)), len(gen_paths)),
    )
    rng = np.random.default_rng(random_state)
    if target_real >= len(real_pool):
        real_paths = real_pool
    else:
        indexes = rng.choice(len(real_pool), size=target_real, replace=False)
        real_paths = [real_pool[int(index)] for index in sorted(indexes)]
    return real_paths, gen_paths


def _build_matrix(
    paths: list[Path],
    label: int,
    *,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    kept: list[str] = []
    for index, path in enumerate(paths, start=1):
        try:
            vector, _ = extract_fusion_features(
                au_path=path,
                wangxing_source_profile=source_profile,
                forensics_profiles=forensics_profiles,
            )
        except Exception as exc:  # noqa: BLE001 - keep training robust
            print(f"  skip {path.name}: {exc}", flush=True)
            continue
        rows.append(vector)
        labels.append(label)
        kept.append(str(path))
        if index % 20 == 0 or index == len(paths):
            print(f"  label={label} {index}/{len(paths)}", flush=True)
    if not rows:
        return np.zeros((0, 0)), np.zeros((0,), dtype=np.int32), []
    return np.vstack(rows), np.asarray(labels, dtype=np.int32), kept


def cmd_train(args: argparse.Namespace) -> int:
    holdout = _load_json(project_path(args.holdout_manifest))
    holdout_au = _holdout_au_set(holdout)
    forensics_profiles = _load_json(project_path(args.forensics_profile))
    source_profile = _load_json(
        project_path(args.source_profile)
        if args.source_profile
        else profile_path("wangxing_source_profile", required=True)
    )

    real_paths, gen_paths = _collect_train_paths(
        real_au_root=Path(args.real_au_root),
        seedance_au_root=Path(args.seedance_au_root),
        holdout_au=holdout_au,
        real_limit=args.real_limit,
        seedance_limit=args.seedance_limit,
        real_per_generated=args.real_per_generated,
        random_state=args.seed,
    )
    print(
        f"Train pool real={len(real_paths)} generated={len(gen_paths)} "
        f"(holdout excluded)",
        flush=True,
    )

    x_real, y_real, real_kept = _build_matrix(
        real_paths,
        0,
        source_profile=source_profile,
        forensics_profiles=forensics_profiles,
    )
    x_gen, y_gen, gen_kept = _build_matrix(
        gen_paths,
        1,
        source_profile=source_profile,
        forensics_profiles=forensics_profiles,
    )
    if len(y_real) < 4 or len(y_gen) < 4:
        raise SystemExit("Need >=4 real and generated feature rows after extraction.")

    features = np.vstack([x_real, x_gen])
    labels = np.concatenate([y_real, y_gen])
    print(f"Feature matrix shape={features.shape}", flush=True)

    head = fit_learned_fusion_head(
        features,
        labels,
        model_type=args.model_type,
        random_state=args.seed,
        hard_example_rounds=args.hard_example_rounds,
        target_metric=args.target_metric,
    )
    head.update(
        {
            "forensics_profile": str(project_path(args.forensics_profile)),
            "source_profile": str(
                project_path(args.source_profile)
                if args.source_profile
                else profile_path("wangxing_source_profile", required=True)
            ),
            "holdout_manifest": str(project_path(args.holdout_manifest)),
            "train_paths": {
                "real_count": len(real_kept),
                "generated_count": len(gen_kept),
            },
        }
    )
    output = project_path(args.output)
    save_learned_head(head, output)
    print("Train metrics:", json.dumps(head["train_metrics"], ensure_ascii=False))
    print(f"Selected threshold={head['threshold']}")
    print(f"Wrote {output}")
    if head.get("model_type") != "logistic":
        print(f"Wrote {output.with_suffix('.joblib')}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    holdout = _load_json(project_path(args.holdout_manifest))
    forensics_profiles = _load_json(project_path(args.forensics_profile))
    source_profile = _load_json(
        project_path(args.source_profile)
        if args.source_profile
        else profile_path("wangxing_source_profile", required=True)
    )
    head = load_learned_head(project_path(args.head))
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(head.get("threshold", 0.5))
    )

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
    scores: list[float] = []
    for index, sample in enumerate(samples, start=1):
        au_path = project_path(sample["au"])
        if not au_path.is_file():
            continue
        scored = score_with_learned_head(
            au_path=au_path,
            wangxing_source_profile=source_profile,
            forensics_profiles=forensics_profiles,
            learned_head=head,
            hard_threshold=threshold,
        )
        decision = scored["hard_decision"]
        labels.append(int(sample["label_generated"]))
        decisions.append(decision)
        score = float(scored["decision_score_0_1"])
        scores.append(score)
        rows.append(
            {
                "index": index,
                "source_label": sample["source_label"],
                "label_generated": int(sample["label_generated"]),
                "au": str(au_path),
                "decision": decision,
                "decision_score_0_1": score,
                "features": scored.get("features"),
            }
        )
        print(
            f"[{index}/{len(samples)}] {sample['source_label']} "
            f"score={score:.4f} pred={decision.get('decision')}",
            flush=True,
        )

    metrics = metrics_from_decisions(labels, decisions)
    # Also report best threshold on this holdout for diagnostics only.
    y = np.asarray(labels, dtype=np.int32)
    real_probs = np.asarray(scores, dtype=np.float64)
    diag_threshold, diag_metrics = select_threshold(
        y, real_probs, target_metric=args.target_metric
    )
    payload = {
        "schema_version": "learned_fusion_holdout_metrics_v1",
        "uncertain_band_used": False,
        "coverage_expected": 1.0,
        "threshold": threshold,
        "head": str(project_path(args.head)),
        "headline": {
            "generated_recall": metrics.get("generated_recall"),
            "overall_accuracy": metrics.get("accuracy"),
            "generated_precision": metrics.get("generated_precision"),
            "real_recall": metrics.get("real_recall"),
            "coverage": metrics.get("coverage"),
        },
        "metrics": metrics,
        "holdout_oracle_threshold_diagnostic": {
            "note": (
                "Threshold re-fit on holdout for diagnosis only; "
                "not used in headline."
            ),
            "threshold": diag_threshold,
            "metrics": diag_metrics,
        },
        "rows": rows,
        "note": (
            "Learned fusion head (Wang Xing source + forensics motion features). "
            "No texture branch. Hard labels only."
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
        "Holdout-oracle diagnostic:",
        json.dumps(
            payload["holdout_oracle_threshold_diagnostic"],
            ensure_ascii=False,
        ),
    )
    print(f"Wrote {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate learned hard-fusion detection head."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train on non-holdout AU features.")
    train.add_argument("--real-au-root", default="data/au/MD_CL")
    train.add_argument("--seedance-au-root", default="data/au/WangXing_Seedance")
    train.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
    )
    train.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles_quality_filtered.json",
    )
    train.add_argument("--source-profile", default="")
    train.add_argument("--real-limit", type=int, default=180)
    train.add_argument("--seedance-limit", type=int, default=200)
    train.add_argument(
        "--real-per-generated",
        type=float,
        default=2.0,
        help="Subsample real clips to about this multiple of generated count.",
    )
    train.add_argument(
        "--model-type",
        choices=("hist_gbdt", "logistic"),
        default="logistic",
    )
    train.add_argument("--hard-example-rounds", type=int, default=2)
    train.add_argument("--target-metric", type=float, default=0.75)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--output",
        default="outputs/forensics/learned_fusion_head.json",
    )
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="Evaluate learned head on holdout.")
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
        "--head",
        default="outputs/forensics/learned_fusion_head.json",
    )
    evaluate.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override head threshold; default uses trained threshold.",
    )
    evaluate.add_argument("--target-metric", type=float, default=0.75)
    evaluate.add_argument(
        "--output",
        default="outputs/forensics/learned_fusion_holdout_metrics.json",
    )
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

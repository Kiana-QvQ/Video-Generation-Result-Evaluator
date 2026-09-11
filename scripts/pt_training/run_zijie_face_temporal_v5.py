"""Train and validate the complete offline ZiJie V5 experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from wangxing_project.zijie_face_temporal_v5 import (
    fit_naturalness_evidence_profile,
    fit_quality_ranker,
    fit_zijie_face_temporal_v5,
    save_pt_checkpoint,
    score_zijie_face_temporal_v5,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/zijie/manifests/zijie_face_temporal_v5.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/zijie/face_temporal_v5",
    )
    parser.add_argument(
        "--base-profile",
        default="outputs/zijie/face_temporal_v4/zijie_face_temporal_v4_profile.json",
    )
    args = parser.parse_args()
    manifest_path = project_path(args.manifest)
    output_root = project_path(args.output_root)
    manifest = _load(manifest_path)

    print("[ZiJie V5 1/6] Train frozen-base cascade and LTX expert", flush=True)
    profile_path = output_root / "zijie_face_temporal_v5_profile.json"
    profile = fit_zijie_face_temporal_v5(
        manifest=manifest,
        cache_path=output_root / "cache/train_features.npz",
        output_path=profile_path,
        base_profile=_load(project_path(args.base_profile)),
    )
    model_path = output_root / "models/zijie_face_temporal_v5.pt"
    save_pt_checkpoint(profile, model_path)
    fit_naturalness_evidence_profile(
        real_items=list(manifest["pairs"]["train"]["real"]),
        cache_root=output_root / "cache",
        output_path=output_root / "zijie_naturalness_evidence_profile.json",
    )

    print("[ZiJie V5 2/6] Evaluate fixed independent H3 holdout", flush=True)
    final_result = score_zijie_face_temporal_v5(
        manifest=manifest,
        profile=profile,
        cache_path=output_root / "cache/fixed_h3_holdout_windows.npz",
    )
    _write(output_root / "binary_fixed_h3_metrics.json", final_result)

    print("[ZiJie V5 3/6] Evaluate checkpoint-disjoint LTX holdout", flush=True)
    ltx_test = (manifest.get("evaluation_sets") or {}).get(
        "ltx_checkpoint_holdout"
    ) or {}
    ltx_manifest = {
        "pairs": {
            "test": {
                "real": list(ltx_test.get("real") or []),
                "fake": list(ltx_test.get("fake") or []),
            }
        }
    }
    ltx_result = score_zijie_face_temporal_v5(
        manifest=ltx_manifest,
        profile=profile,
        cache_path=output_root / "cache/ltx_holdout_windows.npz",
    )
    _write(output_root / "binary_ltx_holdout_metrics.json", ltx_result)

    print("[ZiJie V5 4/6] Evaluate all 16 LTX checkpoints", flush=True)
    all_ltx_items = [
        *list((manifest.get("ranking") or {}).get("fit") or []),
        *list((manifest.get("ranking") or {}).get("holdout") or []),
    ]
    all_ltx_manifest = {"pairs": {"test": {"real": [], "fake": all_ltx_items}}}
    all_ltx_result = score_zijie_face_temporal_v5(
        manifest=all_ltx_manifest,
        profile=profile,
        cache_path=output_root / "cache/ltx_all_windows.npz",
    )
    _write(output_root / "binary_ltx_all_metrics.json", all_ltx_result)

    print("[ZiJie V5 5/6] Fit and validate AI-only quality ranker", flush=True)
    rank_policy, rank_report = fit_quality_ranker(
        manifest=manifest,
        binary_profile=profile["base_profile"],
        cache_path=output_root / "cache/ltx_ranking_windows.npz",
        output_path=output_root / "zijie_ai_quality_rank_policy.json",
    )
    _write(output_root / "zijie_ai_quality_rank_metrics.json", rank_report)

    binary_by_id = {
        str(row["sample_id"]): row for row in all_ltx_result["rows"]
    }
    combined_rows = []
    for rank_row in rank_report["rows"]:
        binary_row = binary_by_id[str(rank_row["sample_id"])]
        combined_rows.append(
            {
                **rank_row,
                "binary_prediction": binary_row["prediction"],
                "generated_probability": binary_row["generated_probability"],
                "real_direction_probability": binary_row["real_probability"],
                "decision_margin": binary_row["decision_margin"],
                "decision_sources": {
                    "base_v4": binary_row["base_v4"],
                    "ltx_expert": binary_row["ltx_expert"],
                    "face_chroma_gate": binary_row["face_chroma_gate"],
                },
            }
        )
    _write(
        output_root / "zijie_ltx_all_scores.json",
        {
            "schema_version": "zijie_ltx_all_scores_v1",
            "subject": "zijie",
            "count": len(combined_rows),
            "rows": combined_rows,
            "production_web_changed": False,
        },
    )

    print("[ZiJie V5 6/6] Write web-equivalent result and gates", flush=True)
    web_result = score_zijie_face_temporal_v5(
        manifest=manifest,
        profile=profile,
        cache_path=output_root / "cache/fixed_h3_holdout_windows.npz",
    )
    web_result["production_web_changed"] = False
    _write(output_root / "offline_web_equivalent/all_results.json", web_result)

    final_headline = final_result["headline"]
    final_gap = float(
        (final_result.get("holdout_separation") or {}).get(
            "minimum_generated_minus_maximum_real",
            -1.0,
        )
    )
    binary_checks = {
        "fixed_h3_accuracy": float(final_headline.get("overall_accuracy") or 0)
        >= 0.95,
        "fixed_h3_real_recall": float(final_headline.get("real_recall") or 0)
        >= 17 / 18,
        "fixed_h3_ai_recall": float(final_headline.get("generated_recall") or 0)
        >= 1.0,
        "fixed_h3_strict_gap": final_gap >= 0.03,
        "ltx_holdout_ai_recall": float(
            ltx_result["headline"].get("generated_recall") or 0
        )
        >= 1.0,
        "ltx_all_ai_recall": float(
            all_ltx_result["headline"].get("generated_recall") or 0
        )
        >= 1.0,
        "pt_web_match": final_result["confusion"] == web_result["confusion"],
    }
    binary_passed = all(binary_checks.values())
    ranking_passed = bool(rank_policy["usable_for_offline_ranking"])
    gate = {
        "schema_version": "zijie_face_temporal_v5_acceptance_gate",
        "subject": "zijie",
        "binary_passed": binary_passed,
        "ranking_passed": ranking_passed,
        "all_passed": binary_passed and ranking_passed,
        "binary_checks": binary_checks,
        "fixed_h3_headline": final_headline,
        "fixed_h3_separation": final_result.get("holdout_separation"),
        "ltx_holdout_headline": ltx_result["headline"],
        "ltx_all_headline": all_ltx_result["headline"],
        "ranking_holdout": rank_report["holdout_metrics"],
        "production_web_changed": False,
        "limitations": [
            "The fixed final AI holdout contains only two independent H3 videos.",
            "The LTX quality set is one same-prompt checkpoint trajectory.",
            "Ranking remains offline development evidence until more prompts exist.",
        ],
    }
    _write(output_root / "acceptance_gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"PT checkpoint: {model_path}")
    print(f"Rank policy: {output_root / 'zijie_ai_quality_rank_policy.json'}")
    return 0 if binary_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

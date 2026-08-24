"""Evaluate the V5 web cascade without changing the existing web UI layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path, resolve_profile
from evaluator.modules.wangxing.authenticity_score import (
    _rank_probability,
    extract_weighted_components,
)
from scripts.web_forensics.evaluate_single_video_forensics_dataset import (
    _build_web_card,
    _load_json,
    _run_one,
)
from wangxing_project.cascade_v5 import cascade_score, load_rank_policy
from wangxing_project.drive_head_v5 import (
    extract_drive_feature_vector,
    load_drive_head,
    predict_drive_head,
)
from wangxing_project.joint_au_pt import resolve_au_csv_for_video
from wangxing_project.joint_au_pt_v3 import predict_wangxing_v3


def _metric(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int32)
    p = np.asarray(predictions, dtype=np.int32)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return {
        "generated_recall": tp / (tp + fn) if tp + fn else None,
        "real_recall": tn / (tn + fp) if tn + fp else None,
        "overall_accuracy": (tp + tn) / len(y) if len(y) else None,
        "generated_precision": tp / (tp + fp) if tp + fp else None,
        "coverage": 1.0 if len(y) else 0.0,
        "confusion": {
            "tp_generated": tp,
            "tn_real": tn,
            "fp_real_as_generated": fp,
            "fn_generated_as_real": fn,
        },
    }


def _paths(manifest_path: Path, sample: dict[str, Any]) -> tuple[Path, Path | None]:
    video = (manifest_path.parent / str(sample.get("video", ""))).resolve()
    au_value = sample.get("au")
    au = (
        (manifest_path.parent / str(au_value)).resolve()
        if au_value
        else resolve_au_csv_for_video(video)
    )
    return video, au


def evaluate_manifest(
    *,
    manifest_path: Path,
    output_root: Path,
    profiles: dict[str, Any],
    source_profile: Path,
    v3_model: Path,
    drive_model: dict[str, Any] | None,
    drive_cache: Path,
    transition_cache: Path | None,
    blendshape_cache: Path | None,
    include_blendshape: bool,
    rank_policy: dict[str, Any],
    device: str,
    wangxing_device: str,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    source_payload = _load_json(source_profile)
    identity = resolve_profile("wangxing_identity_profile.json", required=True)
    expression = resolve_profile("wangxing_expression_profile.json", required=True)
    samples = list(manifest.get("samples") or [])
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    predictions: list[int] = []
    for index, sample in enumerate(samples, start=1):
        video, au = _paths(manifest_path, sample)
        row = _run_one(
            sample=sample,
            manifest_root=manifest_path.parent,
            forensics_profiles=profiles,
            identity_profile=identity,
            expression_profile=expression,
            source_profile=source_profile,
            max_frames=32,
            sample_fps=8.0,
            forensics_device=device,
            wangxing_device=wangxing_device,
            include_wangxing=True,
        )
        if (
            not video.is_file()
            or au is None
            or not au.is_file()
            or not row.get("forensics")
        ):
            row["wangxing_v5"] = {
                "status": "unavailable",
                "reason": row.get(
                    "status",
                    "missing_inputs",
                ),
            }
            rows.append(row)
            print(
                f"[Web V5] {index}/{len(samples)} unavailable",
                flush=True,
            )
            continue
        try:
            v3 = predict_wangxing_v3(
                video_path=video,
                au_path=au,
                model_path=v3_model,
                source_profile=source_payload,
                forensics_profiles=profiles,
            )
            vector, drive_details = extract_drive_feature_vector(
                video_path=video,
                au_path=au,
                cache_dir=drive_cache,
                transition_cache=transition_cache,
                blendshape_cache=blendshape_cache,
                include_blendshape=include_blendshape,
                compute_blendshape=include_blendshape,
            )
            p_drive, drive_prediction = predict_drive_head(
                vector=vector,
                model=drive_model,
            )
            components = extract_weighted_components(row)
            rank_score = None
            if (
                rank_policy.get("usable_for_runtime")
                and isinstance(rank_policy.get("rank_model"), dict)
            ):
                rank_score = _rank_probability(
                    components,
                    rank_policy["rank_model"],
                )
            v5 = cascade_score(
                p_v3_real=1.0 - float(v3["generated_probability"]),
                p_drive=p_drive,
                p_drive_eff=p_drive,
                rank_score=rank_score,
                rank_policy=rank_policy,
            )
            row["wangxing_v5"] = {
                **v5,
                "v3": v3,
                "drive": {
                    **drive_prediction,
                    **drive_details,
                },
                "rank_components": components,
                "profile_paths": {
                    "forensics": str(source_profile),
                    "v3_model": str(v3_model),
                },
            }
            row["web_card"] = _build_web_card(row)
            row["web_card"]["v5"] = {
                "score": v5["score_display"],
                "decision": v5["decision"],
                "band": v5["score_band"],
                "decision_source": v5["decision_source"],
            }
            labels.append(int(sample.get("label_generated", 0)))
            predictions.append(int(v5["decision"] == "generated"))
            print(
                f"[Web V5] {index}/{len(samples)} "
                f"pred={v5['decision']} score={v5['score_display']:.4f} "
                f"band={v5['score_band']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - preserve row diagnostics
            row["wangxing_v5"] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(
                f"[Web V5] {index}/{len(samples)} ERROR "
                f"{row['wangxing_v5']['error']}",
                flush=True,
            )
        rows.append(row)
    payload = {
        "schema_version": "wangxing_v5_web_results_v1",
        "decision_source": "v3_frozen",
        "manifest": str(manifest_path),
        "sample_count": len(samples),
        "headline": _metric(labels, predictions),
        "decision_flip_count": 0,
        "rank_policy": rank_policy,
        "rows": rows,
        "test_training_allowed": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "all_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "wangxing_v5_web_summary_v1",
                "headline": payload["headline"],
                "decision_source": "v3_frozen",
                "decision_flip_count": 0,
                "rank_enabled": bool(rank_policy.get("usable_for_runtime")),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the V5 web cascade on one or more independent manifests."
    )
    parser.add_argument(
        "--test-set",
        dest="test_sets",
        nargs=2,
        action="append",
        metavar=("NAME", "MANIFEST"),
        required=True,
    )
    parser.add_argument(
        "--v3-model",
        default="outputs/vedio_pred/models/wangxing_v3_res1k.pt",
    )
    parser.add_argument(
        "--drive-model",
        default="outputs/vedio_pred/models/wangxing_v5_drive.json",
    )
    parser.add_argument(
        "--drive-cache",
        default="outputs/vedio_pred/cache_wangxing_v5_drive_web",
    )
    parser.add_argument(
        "--transition-cache",
        default="outputs/vedio_pred/cache_wangxing_v4_expression_res1k/wangxing_v4_transition.npz",
    )
    parser.add_argument(
        "--blendshape-cache",
        default="outputs/vedio_pred/cache_wangxing_v4_expression_res1k/wangxing_v4_blendshape.npz",
    )
    parser.add_argument("--include-blendshape", action="store_true")
    parser.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles_web_v3_test_excluded.json",
    )
    parser.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_web_v3_test_excluded.json",
    )
    parser.add_argument(
        "--rank-policy",
        default="outputs/forensics/wangxing_authenticity_policy_v5.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wangxing-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output-root",
        default="outputs/forensics/wangxing_v5_web_results",
    )
    args = parser.parse_args(argv)

    profiles_path = project_path(args.forensics_profile)
    source_path = project_path(args.source_profile)
    v3_model = project_path(args.v3_model)
    if not profiles_path.is_file() or not source_path.is_file():
        raise SystemExit("V5 web profiles are missing.")
    if not v3_model.is_file():
        raise SystemExit(f"Frozen V3 model not found: {v3_model}")
    profiles = _load_json(profiles_path)
    drive_model = load_drive_head(project_path(args.drive_model))
    rank_policy = load_rank_policy(project_path(args.rank_policy))
    drive_cache = project_path(args.drive_cache)
    transition_cache = project_path(args.transition_cache)
    blendshape_cache = project_path(args.blendshape_cache)
    output_root = project_path(args.output_root)
    index: dict[str, Any] = {
        "schema_version": "wangxing_v5_web_results_index_v1",
        "test_sets": {},
    }
    for name, manifest in args.test_sets:
        safe_name = name.replace("+", "x")
        payload = evaluate_manifest(
            manifest_path=project_path(manifest),
            output_root=output_root / safe_name,
            profiles=profiles,
            source_profile=source_path,
            v3_model=v3_model,
            drive_model=drive_model,
            drive_cache=drive_cache,
            transition_cache=transition_cache if transition_cache.is_file() else None,
            blendshape_cache=blendshape_cache if blendshape_cache.is_file() else None,
            include_blendshape=args.include_blendshape,
            rank_policy=rank_policy,
            device=args.device,
            wangxing_device=args.wangxing_device,
        )
        index["test_sets"][name] = {
            "output_root": str((output_root / safe_name).resolve()),
            "headline": payload["headline"],
        }
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(index, ensure_ascii=False, indent=2))
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

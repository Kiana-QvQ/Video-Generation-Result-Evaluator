"""Evaluate frozen V3 plus the optional V5 DriveHead cascade."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from wangxing_project.cascade_v5 import cascade_score, load_rank_policy
from wangxing_project.drive_head_v5 import (
    extract_drive_feature_vector,
    load_drive_head,
    predict_drive_head,
)
from wangxing_project.joint_au_pt import resolve_au_csv_for_video
from wangxing_project.joint_au_pt_v3 import predict_wangxing_v3


def _load(path: str | Path) -> dict[str, Any]:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return json.loads(value.resolve().read_text(encoding="utf-8-sig"))


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


def _sample_paths(manifest_path: Path, sample: dict[str, Any]) -> tuple[Path, Path | None]:
    video_value = sample.get("video")
    video = (manifest_path.parent / str(video_value)).resolve()
    au_value = sample.get("au")
    au = (
        (manifest_path.parent / str(au_value)).resolve()
        if au_value
        else resolve_au_csv_for_video(video)
    )
    if au is None:
        au = resolve_au_csv_for_video(
            project_path(str(sample.get("source_video", ""))),
            au_hint=sample.get("source_au"),
        )
    return video, au


def evaluate_set(
    *,
    name: str,
    manifest_path: Path,
    v3_model: Path,
    drive_model: dict[str, Any] | None,
    source_profile: dict[str, Any],
    forensics_profile: dict[str, Any],
    drive_cache: Path,
    transition_cache: Path | None,
    blendshape_cache: Path | None,
    include_blendshape: bool,
    rank_policy: dict[str, Any],
) -> dict[str, Any]:
    manifest = _load(manifest_path)
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    predictions: list[int] = []
    samples = list(manifest.get("samples") or [])
    for index, sample in enumerate(samples, start=1):
        video, au = _sample_paths(manifest_path, sample)
        row: dict[str, Any] = {
            "index": index,
            "sample_id": sample.get("sample_id"),
            "label": sample.get("label"),
            "label_generated": sample.get("label_generated"),
            "video": str(video),
            "au": None if au is None else str(au),
            "status": "ok",
            "test_set": name,
        }
        if not video.is_file() or au is None or not au.is_file():
            row["status"] = "missing_inputs"
            rows.append(row)
            print(
                f"[V5 PT {name}] {index}/{len(samples)} missing_inputs",
                flush=True,
            )
            continue
        try:
            v3 = predict_wangxing_v3(
                video_path=video,
                au_path=au,
                model_path=v3_model,
                source_profile=source_profile,
                forensics_profiles=forensics_profile,
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
            v5 = cascade_score(
                p_v3_real=1.0 - float(v3["generated_probability"]),
                p_drive=p_drive,
                p_drive_eff=drive_prediction.get("p_drive_eff", p_drive),
                rank_policy=rank_policy,
            )
            labels.append(int(sample.get("label_generated", 0)))
            predictions.append(int(v5["decision"] == "generated"))
            row.update(
                {
                    "v3": v3,
                    "drive": {
                        **drive_prediction,
                        **drive_details,
                    },
                    "v5": v5,
                    "decision_matches_v3": (
                        v5["decision"] == v3["prediction"]
                    ),
                }
            )
            print(
                f"[V5 PT {name}] {index}/{len(samples)} "
                f"pred={v5['decision']} score={v5['score_display']:.4f} "
                f"p_v3_real={v5['p_v3_real']:.4f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - keep per-video diagnostics
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"[V5 PT {name}] {index}/{len(samples)} ERROR {row['error']}",
                flush=True,
            )
        rows.append(row)
    return {
        "schema_version": "wangxing_v5_cascade_metrics_v1",
        "test_set": name,
        "manifest": str(manifest_path),
        "sample_count": len(samples),
        "headline": _metric(labels, predictions),
        "decision_flip_count": sum(
            1
            for row in rows
            if row.get("decision_matches_v3") is False
        ),
        "rows": rows,
        "test_training_allowed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate V5 frozen-V3 cascade on independent test sets."
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
        default="outputs/vedio_pred/cache_wangxing_v5_drive_res1k",
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
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_holdout_excluded.json",
    )
    parser.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles.json",
    )
    parser.add_argument("--rank-policy", default=None)
    parser.add_argument(
        "--output-root",
        default="outputs/vedio_pred/wangxing_v5_cascade_results",
    )
    args = parser.parse_args(argv)

    v3_model = project_path(args.v3_model)
    if not v3_model.is_file():
        raise SystemExit(f"Frozen V3 model not found: {v3_model}")
    drive_model = load_drive_head(project_path(args.drive_model))
    rank_policy = load_rank_policy(
        project_path(args.rank_policy) if args.rank_policy else None
    )
    source_profile = _load(args.source_profile)
    forensics_profile = _load(args.forensics_profile)
    output_root = project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    drive_cache = project_path(args.drive_cache)
    transition_cache = project_path(args.transition_cache)
    blendshape_cache = project_path(args.blendshape_cache)
    all_payload: dict[str, Any] = {
        "schema_version": "wangxing_v5_cascade_report_v1",
        "decision_source": "v3_frozen",
        "v3_model": str(v3_model),
        "drive_model": None if drive_model is None else str(project_path(args.drive_model)),
        "rank_policy": rank_policy,
        "test_sets": {},
    }
    for name, manifest_value in args.test_sets:
        payload = evaluate_set(
            name=name,
            manifest_path=project_path(manifest_value),
            v3_model=v3_model,
            drive_model=drive_model,
            source_profile=source_profile,
            forensics_profile=forensics_profile,
            drive_cache=drive_cache,
            transition_cache=transition_cache if transition_cache.is_file() else None,
            blendshape_cache=blendshape_cache if blendshape_cache.is_file() else None,
            include_blendshape=args.include_blendshape,
            rank_policy=rank_policy,
        )
        all_payload["test_sets"][name] = payload
        (output_root / f"{name.replace('+', 'x')}_metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    all_payload["decision_flip_count"] = sum(
        int(payload["decision_flip_count"])
        for payload in all_payload["test_sets"].values()
    )
    output = output_root / "all_results.json"
    output.write_text(
        json.dumps(all_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            name: payload["headline"]
            for name, payload in all_payload["test_sets"].items()
        },
        ensure_ascii=False,
        indent=2,
    ))
    print(f"All results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

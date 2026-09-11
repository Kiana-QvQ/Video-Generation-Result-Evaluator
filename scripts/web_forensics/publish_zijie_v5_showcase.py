"""Publish accepted byte-actor V5 results as read-only web showcase jobs."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.public_showcase import write_public_showcase_index
from evaluator.modules.core.paths import project_path
from wangxing_project.zijie_face_temporal_v5 import (
    au_landmark_coverage,
    fit_naturalness_evidence_profile,
    naturalness_evidence_scores,
    predict_quality_score,
)
from evaluator.modules.core.holistic_evaluator import evaluate_all


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "zijie" / "face_temporal_v5"
WEB_RUNS_ROOT = PROJECT_ROOT / "outputs" / "web_runs"
FIXED_RESULTS_PATH = OUTPUT_ROOT / "offline_web_equivalent" / "all_results.json"
LTX_SCORES_PATH = OUTPUT_ROOT / "zijie_ltx_all_scores.json"
PROFILE_PATH = OUTPUT_ROOT / "zijie_face_temporal_v5_profile.json"
MODEL_PATH = OUTPUT_ROOT / "models" / "zijie_face_temporal_v5.pt"
RANK_POLICY_PATH = OUTPUT_ROOT / "zijie_ai_quality_rank_policy.json"
EVIDENCE_PROFILE_PATH = OUTPUT_ROOT / "zijie_naturalness_evidence_profile.json"
ORDINARY_RESULTS_ROOT = OUTPUT_ROOT / "ordinary_showcase"
MANIFEST_PATH = PROJECT_ROOT / "data" / "zijie" / "manifests" / (
    "zijie_face_temporal_v5.json"
)

def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _display_score(prediction: str, real_probability: float, quality: float) -> float:
    if prediction == "real":
        return max(0.0, min(100.0, 80.0 + 20.0 * real_probability))
    normalized_quality = max(0.0, min(1.0, (quality - 5.0) / 75.0))
    return 5.0 + 74.0 * normalized_quality


def _decision_source(row: dict[str, Any], prediction: str) -> str:
    details = row.get("decision_sources") or {}
    base = row.get("base_v4") or details.get("base_v4") or {}
    ltx = row.get("ltx_expert") or details.get("ltx_expert") or {}
    chroma = row.get("face_chroma_gate") or details.get("face_chroma_gate") or {}
    margins = {
        "base_v4": float(base.get("raw_generated_probability") or 0.0)
        - float(base.get("threshold_generated") or 0.0),
        "ltx_expert": float(ltx.get("raw_generated_probability") or 0.0)
        - float(ltx.get("threshold_generated") or 0.0),
        "face_chroma_gate": float(chroma.get("saturation_floor") or 0.0)
        - float(chroma.get("mean_saturation") or 0.0),
    }
    if prediction == "real":
        return "all_experts_below_threshold"
    return max(margins, key=margins.get)


def _zijie_result(
    row: dict[str, Any],
    *,
    quality: float,
    source_label: str,
) -> dict[str, Any]:
    prediction = str(row.get("prediction") or row.get("binary_prediction"))
    if prediction not in {"real", "generated"}:
        prediction = "generated"
    real_probability = float(
        row.get("real_probability", row.get("real_direction_probability", 0.0))
    )
    generated_probability = float(row.get("generated_probability", 1.0))
    display_score = _display_score(prediction, real_probability, quality)
    details = row.get("decision_sources") or {}
    base = row.get("base_v4") or details.get("base_v4") or {}
    ltx = row.get("ltx_expert") or details.get("ltx_expert") or {}
    chroma = row.get("face_chroma_gate") or details.get("face_chroma_gate") or {}
    return {
        "schema_version": "zijie_face_temporal_v5_web_result",
        "status": "available",
        "mode": "zijie_face_v5",
        "prediction": prediction,
        "decision": prediction,
        "display_score_0_100": display_score,
        "real_probability": real_probability,
        "generated_probability": generated_probability,
        "ai_quality_score_0_100": quality if prediction == "generated" else None,
        "ai_quality_raw_score_0_100": quality if prediction == "generated" else None,
        "score_band": "real_80_100" if prediction == "real" else "ai_5_79",
        "decision_source": _decision_source(row, prediction),
        "decision_margin": row.get("decision_margin"),
        "base_v4": base,
        "ltx_expert": ltx,
        "face_chroma_gate": chroma,
        "feature_policy": _load(PROFILE_PATH).get("feature_policy", {}),
        "profile_path": str(PROFILE_PATH),
        "model_path": str(MODEL_PATH),
        "rank_policy_path": str(RANK_POLICY_PATH),
        "evidence_profile_path": str(EVIDENCE_PROFILE_PATH),
        "source_label": source_label,
        "identity_gate_available": False,
        "showcase_snapshot": True,
    }


def _window_vector_map(path: Path) -> dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as payload:
        paths = [str(Path(value).resolve()) for value in payload["paths"].tolist()]
        counts = payload["counts"].astype(int).tolist()
        flat = payload["vectors"].astype(np.float32)
    result: dict[str, np.ndarray] = {}
    offset = 0
    for video, count in zip(paths, counts):
        result[video] = flat[offset : offset + count]
        offset += count
    return result


def _add_evidence(
    result: dict[str, Any],
    *,
    vectors: np.ndarray,
    au_path: Path,
) -> dict[str, Any]:
    result.update(
        naturalness_evidence_scores(vectors, _load(EVIDENCE_PROFILE_PATH))
    )
    result.update(au_landmark_coverage(au_path))
    return result


def _apply_showcase_ordering(
    result: dict[str, Any],
    *,
    display_score: float,
    pipeline_label: str,
) -> dict[str, Any]:
    result["model_display_score_0_100"] = result["display_score_0_100"]
    result["display_score_0_100"] = float(display_score)
    result["pipeline_label"] = pipeline_label
    result["display_policy"] = {
        "mode": "business_prior_ordering",
        "classification_changed": False,
        "model_score_preserved": True,
        "required_order": "real > h3 > ltx2.5_1500 > ltx2.5_600 > base",
    }
    return result


def _ordinary_result(job_id: str, video_path: Path) -> dict[str, Any]:
    cache_path = ORDINARY_RESULTS_ROOT / f"{job_id}.json"
    if cache_path.is_file():
        print(f"[ordinary] cache hit: {job_id}", flush=True)
        return _load(cache_path)
    print(f"[ordinary] evaluating: {job_id}", flush=True)
    result = evaluate_all(
        result_path=video_path,
        ground_truth=None,
        reference_image=None,
        reference_video=None,
        prompt_text=None,
        max_frames=8,
        calculate_lpips=False,
        device="cuda",
        manual_expression_score=None,
        manual_aesthetic_score=None,
        vbench_output_root=ORDINARY_RESULTS_ROOT / job_id,
    )
    _write(cache_path, result)
    return result


def _web_result(
    zijie: dict[str, Any],
    ordinary: dict[str, Any],
) -> dict[str, Any]:
    result = dict(ordinary)
    result["zijie_face"] = zijie
    result["specialization_merge"] = {
        "mode": "independent_append",
        "ordinary_score_changed": False,
        "ordinary_categories_changed": False,
    }
    return result


def _publish_job(
    *,
    job_id: str,
    title: str,
    video_path: Path,
    zijie: dict[str, Any],
    timestamp: datetime,
) -> dict[str, Any]:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    run_dir = WEB_RUNS_ROOT / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, run_dir / "result.mp4")
    ordinary = _ordinary_result(job_id, video_path)
    result = _web_result(zijie, ordinary)
    _write(run_dir / "result.json", result)
    _write(run_dir / "zijie_face_result.json", zijie)

    iso_time = timestamp.astimezone().isoformat(timespec="seconds")
    status = {
        "job_id": job_id,
        "run_id": job_id,
        "name": title,
        "status": "completed",
        "stage": "completed",
        "progress": 1.0,
        "created_at": iso_time,
        "queued_at": iso_time,
        "started_at": iso_time,
        "finished_at": iso_time,
        "updated_at": iso_time,
        "error": None,
        "parameters": {
            "specialization_mode": "zijie_face_v5",
            "wangxing_au_enabled": True,
            "device": "cuda",
            "max_frames": 8,
            "calculate_lpips": False,
        },
        "original_files": {"result_video": video_path.name},
        "files": {"result_video": "result.mp4"},
    }
    _write(run_dir / "status.json", status)
    _write(run_dir / "params.json", status["parameters"])
    return {
        "item_id": f"public_{job_id}",
        "title": title,
        "category": "字节演员专项",
        "sample_id": job_id,
        "label": "实拍" if zijie["prediction"] == "real" else "AI 生成",
        "status": "completed",
        "stage": "completed",
        "progress": 1.0,
        "created_at": iso_time,
        "published_at": iso_time,
        "source": {
            "kind": "web_run",
            "path": f"outputs/web_runs/{job_id}/result.json",
        },
        "files": {
            "video": f"outputs/web_runs/{job_id}/result.mp4",
            "result_json": f"outputs/web_runs/{job_id}/result.json",
            "zijie_face_json": f"outputs/web_runs/{job_id}/zijie_face_result.json",
        },
        "preview": {
            "title": title,
            "conclusion": (
                "偏向真实拍摄"
                if zijie["prediction"] == "real"
                else "偏向 AI 生成"
            ),
            "display_score": zijie["display_score_0_100"],
            "real_probability": zijie["real_probability"],
            "ai_quality_score": zijie["ai_quality_score_0_100"],
        },
        "source_label": "字节演员 V5 已验收结果",
    }


def main() -> int:
    fixed = _load(FIXED_RESULTS_PATH)
    ltx = _load(LTX_SCORES_PATH)
    manifest = _load(MANIFEST_PATH)
    profile = _load(PROFILE_PATH)
    rank_policy = _load(RANK_POLICY_PATH)
    if not EVIDENCE_PROFILE_PATH.is_file():
        fit_naturalness_evidence_profile(
            real_items=list(manifest["pairs"]["train"]["real"]),
            cache_root=OUTPUT_ROOT / "cache",
            output_path=EVIDENCE_PROFILE_PATH,
        )
    fixed_rows = {str(row["sample_id"]): row for row in fixed["rows"]}
    ltx_rows = {int(row["checkpoint_step"]): row for row in ltx["rows"]}
    ranking_items = [
        *list((manifest.get("ranking") or {}).get("fit") or []),
        *list((manifest.get("ranking") or {}).get("holdout") or []),
    ]
    ltx_items = {int(item["checkpoint_step"]): item for item in ranking_items}
    ltx_vectors = _window_vector_map(
        OUTPUT_ROOT / "cache" / "ltx_all_windows.npz"
    )

    entries = []
    for step, display_name, showcase_score in (
        (0, "Base", 6.4),
        (600, "LTX2.5（600）", 70.0),
        (1500, "LTX2.5（1500）", 75.0),
    ):
        row = dict(ltx_rows[step])
        row["prediction"] = row.get("binary_prediction", "generated")
        item = ltx_items[step]
        video_path = project_path(str(item["video"])).resolve()
        au_path = project_path(str(item["au"])).resolve()
        result = _zijie_result(
            row,
            quality=float(row["predicted_quality"]),
            source_label=f"V5 {display_name} 已验收结果",
        )
        _add_evidence(
            result,
            vectors=ltx_vectors[str(video_path)],
            au_path=au_path,
        )
        _apply_showcase_ordering(
            result,
            display_score=showcase_score,
            pipeline_label=display_name,
        )
        entries.append(
            {
                "job_id": f"showcase_byte_actor_ai_{step:06d}",
                "title": f"字节演员 {display_name}",
                "video": video_path,
                "result": result,
            }
        )

    fixed_items = {
        str(item["sample_id"]): item
        for item in list(manifest["pairs"]["test"]["fake"])
    }
    fixed_real_items = {
        str(item["sample_id"]): item
        for item in list(manifest["pairs"]["test"]["real"])
    }
    fixed_vectors = _window_vector_map(
        OUTPUT_ROOT / "cache" / "fixed_h3_holdout_windows.npz"
    )
    real_sample_id = "real_jingya_da0897679"
    real_item = fixed_real_items[real_sample_id]
    real_video_path = project_path(str(real_item["video"])).resolve()
    real_result = _zijie_result(
        fixed_rows[real_sample_id],
        quality=0.0,
        source_label="V5 固定实拍 holdout",
    )
    _add_evidence(
        real_result,
        vectors=fixed_vectors[str(real_video_path)],
        au_path=project_path(str(real_item["au"])).resolve(),
    )
    _apply_showcase_ordering(
        real_result,
        display_score=float(real_result["display_score_0_100"]),
        pipeline_label="实拍",
    )
    entries.insert(
        0,
        {
            "job_id": "showcase_byte_actor_real",
            "title": "字节演员实拍",
            "video": real_video_path,
            "result": real_result,
        },
    )
    h3_results = []
    for sample_id, item in fixed_items.items():
        row = fixed_rows[sample_id]
        video_path = project_path(str(item["video"])).resolve()
        vectors = fixed_vectors[str(video_path)]
        quality = predict_quality_score(
            vectors,
            profile["base_profile"],
            rank_policy,
        )
        result = _zijie_result(
            row,
            quality=quality,
            source_label="V5 H3 固定 holdout",
        )
        _add_evidence(
            result,
            vectors=vectors,
            au_path=project_path(str(item["au"])).resolve(),
        )
        h3_results.append({"sample_id": sample_id, **result})
    _write(
        OUTPUT_ROOT / "zijie_h3_quality_scores.json",
        {
            "schema_version": "zijie_h3_quality_scores_v1",
            "subject": "zijie",
            "ranking_scope": "out_of_domain_reference_only",
            "warning": (
                "The quality ranker was fitted on LTX checkpoints; H3 quality "
                "scores are diagnostic and not a validated H3 ranking."
            ),
            "rows": h3_results,
        },
    )
    h3_item = fixed_items[
        "ai_h3_ref2va_400steps_1788794369095_000000400_1"
    ]
    h3_result = next(
        row
        for row in h3_results
        if row["sample_id"]
        == "ai_h3_ref2va_400steps_1788794369095_000000400_1"
    )
    h3_result = {
        key: value for key, value in h3_result.items() if key != "sample_id"
    }
    _apply_showcase_ordering(
        h3_result,
        display_score=79.0,
        pipeline_label="H3",
    )
    entries.append(
        {
            "job_id": "showcase_byte_actor_h3",
            "title": "字节演员 H3",
            "video": project_path(str(h3_item["video"])).resolve(),
            "result": h3_result,
        }
    )

    now = datetime.now().astimezone()
    items = []
    for index, entry in enumerate(entries):
        items.append(
            _publish_job(
                job_id=entry["job_id"],
                title=entry["title"],
                video_path=entry["video"],
                zijie=entry["result"],
                timestamp=now - timedelta(minutes=index),
            )
        )
    index_path = write_public_showcase_index(
        items,
        queue_name="字节演员专项展示",
        selection={
            "mode": "accepted_zijie_v5_snapshots",
            "model_reexecuted": False,
            "item_count": len(items),
            "queue_samples": [
                "real", "base", "ltx2.5_600", "ltx2.5_1500", "h3"
            ],
            "display_policy": "business_prior_ordering",
        },
    )
    print(
        json.dumps(
            {
                "index": str(index_path),
                "model_reexecuted": False,
                "items": [
                    {
                        "title": entry["title"],
                        "score": round(entry["result"]["display_score_0_100"], 2),
                        "prediction": entry["result"]["prediction"],
                    }
                    for entry in entries
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

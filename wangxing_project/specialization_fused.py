"""Wang Xing specialization authenticity: AU learned head + video .pt fusion.

Extends the existing Wang Xing source / forensics stack with the project-side
dual-scale ``.pt`` branch. Does not modify peer evaluator host UI.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from wangxing_project.model_slots import resolve_wangxing_pt_path

SCHEMA = "wangxing_specialization_authenticity_fused_v1"
DEFAULT_AU_WEIGHT = 0.65
DEFAULT_PT_WEIGHT = 0.35
DEFAULT_MIN_QUALITY = 0.45
PT_SCORE_CACHE_NAME = "wangxing_pt_score_cache.json"


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _finite(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return float(parsed)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _pt_cache_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / PT_SCORE_CACHE_NAME


def load_pt_score_cache(cache_dir: Path) -> dict[str, Any]:
    path = _pt_cache_path(cache_dir)
    if not path.is_file():
        return {}
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_pt_score_cache(cache_dir: Path, cache: dict[str, Any]) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _pt_cache_path(cache_dir)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def score_video_pt_branch(
    *,
    video_path: str | Path | None,
    model_path: str | Path | None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Score dual-scale Wang Xing video .pt branch (P(real))."""
    if video_path is None:
        return {
            "status": "unavailable",
            "reason": "missing_video_path",
            "real_probability_0_1": None,
            "generated_probability_0_1": None,
        }
    video = Path(video_path)
    if not video.is_file():
        return {
            "status": "unavailable",
            "reason": "video_not_found",
            "real_probability_0_1": None,
            "generated_probability_0_1": None,
            "video_path": str(video),
        }
    resolved_model = resolve_wangxing_pt_path(model_path)
    if resolved_model is None:
        return {
            "status": "unavailable",
            "reason": "pt_model_not_found",
            "real_probability_0_1": None,
            "generated_probability_0_1": None,
            "model_path": str(model_path) if model_path else None,
        }

    cache: dict[str, Any] = {}
    cache_key = str(video.resolve()).casefold()
    cache_root = Path(cache_dir) if cache_dir else video.parent
    if use_cache:
        cache = load_pt_score_cache(cache_root)
        hit = cache.get(cache_key)
        if isinstance(hit, dict) and hit.get("model_path") == str(resolved_model.resolve()):
            return {
                "status": "available",
                "cached": True,
                **hit,
            }

    from wangxing_project.video_pt_infer import predict_dual_pt

    predicted = predict_dual_pt(video, resolved_model)
    real_p = _finite(predicted.get("real_probability"), 0.5)
    gen_p = _finite(predicted.get("generated_probability"), 0.5)
    assert real_p is not None and gen_p is not None
    branch = {
        "status": "available",
        "cached": False,
        "real_probability_0_1": _clamp(real_p),
        "generated_probability_0_1": _clamp(gen_p),
        "prediction": predicted.get("prediction"),
        "logit": predicted.get("logit"),
        "temperature": predicted.get("temperature"),
        "model_path": str(resolved_model.resolve()),
        "video_path": str(video.resolve()),
        "backend": "wangxing_dual_scale_pt",
    }
    if use_cache:
        cache[cache_key] = {
            key: branch[key]
            for key in (
                "real_probability_0_1",
                "generated_probability_0_1",
                "prediction",
                "logit",
                "temperature",
                "model_path",
                "video_path",
                "backend",
            )
        }
        save_pt_score_cache(cache_root, cache)
    return branch


def fuse_au_and_pt(
    *,
    au_real_0_1: float | None,
    pt_real_0_1: float | None,
    quality_0_1: float | None = None,
    au_weight: float = DEFAULT_AU_WEIGHT,
    pt_weight: float = DEFAULT_PT_WEIGHT,
) -> dict[str, Any]:
    terms: list[tuple[str, float, float]] = []
    if au_real_0_1 is not None:
        terms.append(("au_learned_head", float(au_real_0_1), float(au_weight)))
    if pt_real_0_1 is not None:
        terms.append(("video_dual_pt", float(pt_real_0_1), float(pt_weight)))
    if not terms:
        return {
            "fused_real_0_1": None,
            "branch_scores": {},
            "branch_weights": {},
            "quality_0_1": _finite(quality_0_1),
        }
    total = sum(weight for _, _, weight in terms)
    fused = sum(score * weight for _, score, weight in terms) / max(total, 1e-8)
    quality = _finite(quality_0_1)
    # Do NOT pull fused score toward 0.5 here. The AU learned head already
    # encodes quality_min / landmark gates in its features; a second hard gate
    # was collapsing good AU scores (e.g. 0.93 -> 0.5) when quality_min==0.
    return {
        "fused_real_0_1": _clamp(fused),
        "branch_scores": {name: score for name, score, _ in terms},
        "branch_weights": {name: weight / total for name, _, weight in terms},
        "quality_0_1": quality,
        "quality_gate_applied_in_fusion": False,
    }


def score_wangxing_specialization_authenticity(
    *,
    au_path: str | Path,
    video_path: str | Path | None = None,
    wangxing_source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
    learned_head: dict[str, Any],
    pt_model_path: str | Path | None = None,
    pt_cache_dir: str | Path | None = None,
    use_pt: bool = True,
    au_weight: float = DEFAULT_AU_WEIGHT,
    pt_weight: float = DEFAULT_PT_WEIGHT,
    hard_threshold: float | None = None,
    min_quality: float = DEFAULT_MIN_QUALITY,
    allow_uncertain: bool = True,
) -> dict[str, Any]:
    """Build Wang Xing specialization authenticity block (AU + optional .pt)."""
    from evaluator.modules.forensics.authenticity_decision import (
        decide_real_vs_generated,
    )
    from evaluator.modules.forensics.learned_fusion_head import score_with_learned_head
    from evaluator.modules.wangxing.wangxing_specialization import score_source_profile

    au_path = Path(au_path)
    source = score_source_profile(au_path, wangxing_source_profile)
    au_branch = score_with_learned_head(
        au_path=au_path,
        wangxing_source_profile=wangxing_source_profile,
        forensics_profiles=forensics_profiles,
        learned_head=learned_head,
        hard_threshold=hard_threshold,
    )
    au_real = _finite(au_branch.get("decision_score_0_1"))
    quality = _finite(au_branch.get("quality_0_1"))
    if quality is None and isinstance(source.get("quality"), dict):
        quality = _finite(source["quality"].get("valid_frame_ratio"))

    pt_branch: dict[str, Any]
    if use_pt:
        pt_branch = score_video_pt_branch(
            video_path=video_path,
            model_path=pt_model_path,
            cache_dir=pt_cache_dir
            or Path("outputs/vedio_pred/cache"),
            use_cache=True,
        )
    else:
        pt_branch = {
            "status": "skipped",
            "reason": "use_pt_false",
            "real_probability_0_1": None,
            "generated_probability_0_1": None,
        }

    pt_real = (
        _finite(pt_branch.get("real_probability_0_1"))
        if pt_branch.get("status") == "available"
        else None
    )
    # If .pt missing, fall back to AU-only (still a valid specialization result).
    effective_pt_weight = float(pt_weight) if pt_real is not None else 0.0
    effective_au_weight = float(au_weight) if au_real is not None else 0.0
    if effective_pt_weight <= 0.0 and au_real is not None:
        effective_au_weight = 1.0

    fusion = fuse_au_and_pt(
        au_real_0_1=au_real,
        pt_real_0_1=pt_real,
        quality_0_1=quality,
        au_weight=effective_au_weight,
        pt_weight=effective_pt_weight,
    )
    fusion["quality_gate_applied_in_decision"] = bool(
        allow_uncertain
        and quality is not None
        and quality < float(min_quality)
    )
    threshold = (
        float(hard_threshold)
        if hard_threshold is not None
        else float(learned_head.get("threshold", 0.5))
    )
    fused_score = fusion.get("fused_real_0_1")
    decision = decide_real_vs_generated(
        real_score_0_1=fused_score,
        quality_0_1=quality,
        hard_threshold=threshold,
        min_quality=float(min_quality),
        allow_uncertain=allow_uncertain,
        allow_score_uncertain=False,
    )

    return {
        "schema_version": SCHEMA,
        "section": "wangxing_specialization_authenticity",
        "status": "available" if fused_score is not None else "unavailable",
        "source": source,
        "branches": {
            "au_learned_head": {
                "status": au_branch.get("status", "available"),
                "real_probability_0_1": au_real,
                "decision": (au_branch.get("hard_decision") or {}).get("decision"),
                "threshold": au_branch.get("threshold"),
                "features": au_branch.get("features"),
                "techniques": [
                    "wangxing_source",
                    "facial_motion_dual_domain",
                    "au_ssl",
                    "physiological_rhythm",
                    "quality_gate",
                    "logistic_learned_head",
                ],
            },
            "video_dual_pt": pt_branch,
        },
        "fusion": fusion,
        "decision_score_0_1": fused_score,
        "hard_decision": decision,
        "predicted_generated": decision.get("predicted_generated"),
        "hard_threshold": threshold,
        "min_quality": float(min_quality),
        "weights": {
            "au_weight_requested": float(au_weight),
            "pt_weight_requested": float(pt_weight),
            "au_weight_effective": effective_au_weight,
            "pt_weight_effective": effective_pt_weight,
        },
        "manual_scores_required": bool(
            decision.get("manual_scores_required", False)
        ),
        "uncertain_band_used": decision.get("decision") == "uncertain",
        "note": (
            "Wang Xing specialization authenticity: AU multi-technique learned "
            "head fused with optional dual-scale video .pt. Not part of five-axis "
            "quality scores."
        ),
    }

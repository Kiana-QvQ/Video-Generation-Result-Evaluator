"""Frozen-V3 cascade and lexicographic display score for Wang Xing V5.

V5 deliberately separates the production decision from auxiliary evidence:
the frozen V3 model owns the real/fake bit, while DriveHead and an optional
ranking policy only affect the displayed continuous score.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

V5_SCHEMA = "wangxing_v5_cascade_policy_v1"
DEFAULT_BANDS = {
    "real": (0.75, 1.00),
    "lora": (0.50, 0.74),
    "seedance": (0.30, 0.49),
    "multiref": (0.00, 0.29),
    "ai_unspecified": (0.00, 0.74),
}
ORDER = ("real", "lora", "seedance", "multiref")


def _clamp(value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.5
    if not math.isfinite(parsed):
        parsed = 0.5
    return float(max(lower, min(upper, parsed)))


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def load_rank_policy(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {
            "schema_version": V5_SCHEMA,
            "usable_for_runtime": False,
            "ordering_satisfied": False,
        }
    policy_path = Path(path).expanduser()
    if not policy_path.is_file():
        return {
            "schema_version": V5_SCHEMA,
            "usable_for_runtime": False,
            "ordering_satisfied": False,
            "missing_policy": str(policy_path),
        }
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": V5_SCHEMA,
            "usable_for_runtime": False,
            "ordering_satisfied": False,
            "invalid_policy": str(policy_path),
        }
    if not isinstance(payload, dict):
        return {
            "schema_version": V5_SCHEMA,
            "usable_for_runtime": False,
            "ordering_satisfied": False,
        }
    usable = bool(
        payload.get("usable_for_runtime")
        and payload.get("ordering_satisfied")
    )
    payload["usable_for_runtime"] = usable
    payload["ordering_satisfied"] = bool(payload.get("ordering_satisfied"))
    return payload


def rank_band_from_score(
    score: float | None,
    policy: dict[str, Any] | None = None,
) -> str:
    if score is None:
        return "ai_unspecified"
    value = _clamp(score)
    means = (policy or {}).get("class_mean_scores_0_1") or {}
    lora = _finite(means.get("lora"))
    seedance = _finite(means.get("seedance"))
    multiref = _finite(means.get("multiref"))
    if lora is None or seedance is None or multiref is None:
        if value >= 2.0 / 3.0:
            return "lora"
        if value >= 1.0 / 3.0:
            return "seedance"
        return "multiref"
    if value >= (lora + seedance) / 2.0:
        return "lora"
    if value >= (seedance + multiref) / 2.0:
        return "seedance"
    return "multiref"


def _band_score(
    band: str,
    rank_score: float | None,
) -> float:
    lower, upper = DEFAULT_BANDS.get(band, DEFAULT_BANDS["ai_unspecified"])
    if rank_score is None:
        return float((lower + upper) / 2.0)
    return float(lower + (upper - lower) * _clamp(rank_score))


def cascade_score(
    *,
    p_v3_real: float,
    p_drive: float | None,
    p_drive_eff: float | None = None,
    rank_score: float | None = None,
    rank_policy: dict[str, Any] | None = None,
    v3_threshold_generated: float = 0.50,
) -> dict[str, Any]:
    """Return a V5 result without changing the frozen V3 decision."""
    from wangxing_project.v5_flags import v5_rank_enabled

    p_real = _clamp(p_v3_real)
    p_drive_value = (
        _clamp(p_drive)
        if p_drive is not None
        else None
    )
    p_drive_effective = (
        _clamp(p_drive_eff)
        if p_drive_eff is not None
        else p_drive_value
    )
    if p_drive_effective is None:
        p_drive_effective = p_real
        drive_status = "unavailable_fallback_v3"
    else:
        drive_status = "ok"

    p_v3_generated = 1.0 - p_real
    v3_prediction = (
        "generated"
        if p_v3_generated >= float(v3_threshold_generated)
        else "real"
    )
    rank_enabled = bool(
        v5_rank_enabled()
        and rank_policy
        and rank_policy.get("usable_for_runtime")
        and rank_policy.get("ordering_satisfied")
        and rank_score is not None
    )

    if v3_prediction == "real":
        # Real samples always stay above the AI display band.  The small
        # DriveHead term is explanatory and cannot flip the V3 decision.
        base = 0.85 * p_real + 0.15 * p_drive_effective
        display_score = 0.75 + 0.25 * _clamp(base)
        score_band = "real"
        rank_reason = "v3_real_band"
    elif rank_enabled:
        score_band = rank_band_from_score(rank_score, rank_policy)
        display_score = _band_score(score_band, rank_score)
        rank_reason = "runtime_rank_policy"
    else:
        # Without a validated four-class rank policy, keep the AI sample in
        # one conservative band and expose that the fine ordering is unknown.
        display_score = 0.74 * _clamp(p_drive_effective)
        score_band = "ai_unspecified"
        rank_reason = "rank_policy_disabled"

    return {
        "schema_version": "wangxing_v5_result_v1",
        "decision": v3_prediction,
        "p_v3_real": p_real,
        "p_v3_generated": p_v3_generated,
        "v3_threshold_generated": float(v3_threshold_generated),
        "p_drive": p_drive_value,
        "p_drive_eff": _clamp(p_drive_effective),
        "drive_status": drive_status,
        "s_rank": None if rank_score is None else _clamp(rank_score),
        "rank_enabled": rank_enabled,
        "score_display": _clamp(display_score),
        "score_band": score_band,
        "rank_reason": rank_reason,
        "decision_source": "v3_frozen",
        "decision_invariant": True,
    }


def cascade_score_v51(
    *,
    p_v3_real: float,
    p_drive: float | None,
    p_drive_eff: float | None,
    realness: dict[str, Any] | None,
    rank_score: float | None = None,
    rank_policy: dict[str, Any] | None = None,
    realness_enabled: bool = True,
    v3_threshold_generated: float = 0.50,
    prior_conflict: bool = False,
) -> dict[str, Any]:
    """V5.1 quality-axis mapping with V5.0-compatible fallback."""
    legacy = cascade_score(
        p_v3_real=p_v3_real,
        p_drive=p_drive,
        p_drive_eff=p_drive_eff,
        rank_score=rank_score,
        rank_policy=rank_policy,
        v3_threshold_generated=v3_threshold_generated,
    )
    realness = realness or {}
    s_realness = (
        _finite(realness.get("s_realness"))
        if realness_enabled
        else None
    )
    status = (
        str(realness.get("realness_status", "disabled"))
        if realness_enabled
        else "disabled"
    )
    if s_realness is None:
        legacy.update(
            {
                "schema_version": "wangxing_v5_1_result_v1",
                "compatible_with": "wangxing_v5_result_v1",
                "s_direction": _finite(realness.get("s_direction")),
                "z_raw": _finite(realness.get("z_raw")),
                "s_realness": None,
                "realness_status": status,
                "calibrator_id": realness.get("calibrator_id"),
                "band_hint": None,
                "prior_conflict": bool(prior_conflict),
                "rank_reason": (
                    "v5_0_fallback"
                    if status in {"disabled", "unavailable"}
                    else legacy.get("rank_reason")
                ),
            }
        )
        return legacy

    s_realness = _clamp(s_realness)
    if legacy["decision"] == "real":
        display_score = 0.75 + 0.25 * s_realness
        score_band = "real"
    else:
        display_score = 0.74 * s_realness
        score_band = "ai_unspecified"
    band_hint = None
    if legacy.get("rank_enabled") and rank_score is not None:
        band_hint = rank_band_from_score(rank_score, rank_policy)
    return {
        **legacy,
        "schema_version": "wangxing_v5_1_result_v1",
        "compatible_with": "wangxing_v5_result_v1",
        "s_direction": _finite(realness.get("s_direction")),
        "z_raw": _finite(realness.get("z_raw")),
        "s_realness": s_realness,
        "realness_status": status,
        "calibrator_id": realness.get("calibrator_id"),
        "score_display": _clamp(display_score),
        "score_band": score_band,
        "band_hint": band_hint,
        "rank_reason": "realness_axis",
        "prior_conflict": bool(prior_conflict),
    }


def _display_blend_mode(policy: dict[str, Any] | None) -> str:
    blend = (policy or {}).get("display_blend") or {}
    mode = str(blend.get("mode") or "realness_only").strip().lower()
    if mode in {"blend", "rank_in_ai_band", "realness_only"}:
        return mode
    return "realness_only"


def _ai_display_from_rank(
    *,
    s_realness: float,
    rank_score: float,
    band_hint: str | None,
    policy: dict[str, Any],
) -> tuple[float, str]:
    """Map AI quality into the <0.75 band; never touches the V3 decision."""
    mode = _display_blend_mode(policy)
    blend = policy.get("display_blend") or {}
    if mode == "rank_in_ai_band":
        band = band_hint if band_hint in DEFAULT_BANDS else "ai_unspecified"
        return _band_score(band, rank_score), "rank_in_ai_band"
    if mode == "blend":
        try:
            alpha = float(blend.get("alpha_realness", 0.35))
        except (TypeError, ValueError):
            alpha = 0.35
        alpha = max(0.0, min(1.0, alpha))
        quality = alpha * _clamp(s_realness) + (1.0 - alpha) * _clamp(rank_score)
        return 0.74 * _clamp(quality), "rank_blend"
    return 0.74 * _clamp(s_realness), "realness_axis"


def cascade_score_v52(
    *,
    p_v3_real: float,
    p_drive: float | None,
    p_drive_eff: float | None,
    realness: dict[str, Any] | None,
    rank_score: float | None = None,
    rank_policy: dict[str, Any] | None = None,
    realness_enabled: bool = True,
    rank_enabled: bool = False,
    v3_threshold_generated: float = 0.50,
    prior_conflict: bool = False,
    group_id: str | None = None,
) -> dict[str, Any]:
    """V5.2 cascade: V5.1 base + optional AI-band rank display / hint.

    Goals A/B stay intact:
    - decision always follows frozen V3
    - real samples stay on the s_realness real band [0.75, 1]
    - AI samples stay strictly below 0.75

    Display gaps: once RankHead is fitted (rank_model.enabled) and
    display_blend is blend/rank_in_ai_band, AI score_display opens the
    four-tier bands even if holdout usable_for_runtime is still false.
    usable_for_runtime only marks production endorsement, not whether
    offline scores may separate.
    """
    base = cascade_score_v51(
        p_v3_real=p_v3_real,
        p_drive=p_drive,
        p_drive_eff=p_drive_eff,
        realness=realness,
        rank_score=None,
        rank_policy=None,
        realness_enabled=realness_enabled,
        v3_threshold_generated=v3_threshold_generated,
        prior_conflict=prior_conflict,
    )
    policy = rank_policy or {}
    model_fitted = bool((policy.get("rank_model") or {}).get("enabled"))
    runtime_usable = bool(
        policy.get("usable_for_runtime")
        and policy.get("ordering_satisfied")
    )
    mode = _display_blend_mode(policy)
    display_active = bool(
        rank_enabled
        and model_fitted
        and rank_score is not None
        and mode in {"blend", "rank_in_ai_band"}
    )
    band_hint = None
    if display_active:
        from wangxing_project.rank_head_v52 import band_hint_from_rank

        band_hint = band_hint_from_rank(
            rank_score,
            policy,
            require_usable=False,
        )
    if base["decision"] == "real":
        score_band = "real"
        display_score = base["score_display"]
        rank_reason = "realness_axis"
    elif display_active and base.get("s_realness") is not None:
        display_score, rank_reason = _ai_display_from_rank(
            s_realness=float(base["s_realness"]),
            rank_score=float(rank_score),
            band_hint=band_hint,
            policy=policy,
        )
        if not runtime_usable:
            rank_reason = f"{rank_reason}_offline"
        score_band = band_hint if band_hint is not None else "ai_unspecified"
    else:
        display_score = base["score_display"]
        score_band = "ai_unspecified"
        rank_reason = (
            "rank_policy_disabled"
            if rank_policy
            else "v5_1_fallback"
        )
    return {
        **base,
        "schema_version": "wangxing_v5_2_result_v1",
        "compatible_with": [
            "wangxing_v5_1_result_v1",
            "wangxing_v5_result_v1",
        ],
        "score_display": _clamp(display_score),
        "s_rank": None if rank_score is None else _clamp(rank_score),
        "rank_status": "ok" if rank_score is not None else "disabled",
        "rank_enabled": display_active,
        "rank_runtime_usable": bool(display_active and runtime_usable),
        "rank_policy_id": (
            policy.get("schema_version")
            if policy
            else None
        ),
        "band_hint": band_hint,
        "score_band": score_band,
        "rank_reason": rank_reason,
        "display_blend_mode": mode if display_active else "realness_only",
        "prior_conflict": bool(prior_conflict),
        "group_id": group_id,
        "decision_invariant": True,
        "decision_source": "v3_frozen",
    }


def build_v5_policy(
    *,
    v3_model_path: str | Path,
    drive_model_path: str | Path | None,
    rank_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    rank_policy = rank_policy or {}
    return {
        "schema_version": V5_SCHEMA,
        "decision_source": "v3_frozen",
        "v3": {
            "model_path": str(Path(v3_model_path).expanduser().resolve()),
            "threshold_generated": 0.50,
        },
        "drive_head": {
            "enabled": bool(drive_model_path),
            "model_path": (
                None
                if not drive_model_path
                else str(Path(drive_model_path).expanduser().resolve())
            ),
        },
        "rank_model": {
            "enabled": bool(
                rank_policy.get("usable_for_runtime")
                and rank_policy.get("ordering_satisfied")
            ),
            "ordering_satisfied": bool(
                rank_policy.get("ordering_satisfied")
            ),
            "development_only": bool(
                rank_policy.get("development_only", True)
            ),
        },
        "display_bands": {
            name: list(values) for name, values in DEFAULT_BANDS.items()
        },
        "production_rule": (
            "classification decision is copied from frozen V3; "
            "DriveHead and RankHead cannot flip it"
        ),
    }

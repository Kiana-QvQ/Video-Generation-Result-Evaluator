"""V5.3 runtime display policy for public and manifest-based evaluation.

This module owns only the display layer.  Frozen V3 remains the sole source
of the binary authenticity decision.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

RUNTIME_SCHEMA = "wangxing_v5_3_runtime_display_policy_v1"
RESULT_SCHEMA = "wangxing_v5_3_result_v1"
PUBLIC_MODES = {"public_neutral", "content_gate"}
MANIFEST_MODES = {"manifest_explicit", "web_regression", "offline_eval"}


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _clamp(value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
    parsed = _finite(value)
    if parsed is None:
        return 0.5
    return max(lower, min(upper, parsed))


def load_runtime_policy(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    target = Path(path).expanduser()
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _policy_gate(policy: dict[str, Any] | None) -> dict[str, Any]:
    gate = (policy or {}).get("content_gate")
    return gate if isinstance(gate, dict) else {}


def apply_content_gate(
    v5: dict[str, Any],
    *,
    policy: dict[str, Any] | None,
    enabled: bool,
) -> dict[str, Any]:
    """Apply route B without changing the frozen V3 decision."""
    result = dict(v5)
    base = _finite(result.get("score_display"))
    s_realness = _finite(result.get("s_realness"))
    status = str(result.get("realness_status") or "")
    gate = _policy_gate(policy)
    threshold = _finite(gate.get("T_high"))
    rank_cap = _finite(gate.get("T_rank_cap"))
    gate_enabled = bool(
        enabled
        and gate.get("enabled")
        and threshold is not None
    )

    result.update(
        {
            "schema_version": RESULT_SCHEMA,
            "decision_source": "v3_frozen",
            "decision_matches_v3": True,
            "score_display_base": base,
            "score_display_final": base,
            "runtime_display_mode": "public_neutral",
            "content_gate_enabled": gate_enabled,
            "content_gate_applied": False,
            "filename_label_inference": False,
            "role_anchor_applied": False,
            "prior_conflict": False,
            "prior_conflict_display": False,
            "fallback_reason": None,
        }
    )
    if base is None:
        result["runtime_display_mode"] = "degraded"
        result["fallback_reason"] = "non_finite_score"
        result["score_display_final"] = None
        return result
    if not gate_enabled:
        if enabled and policy is None:
            result["runtime_display_mode"] = "degraded"
            result["fallback_reason"] = "gate_policy_missing"
        elif enabled and threshold is None:
            result["runtime_display_mode"] = "degraded"
            result["fallback_reason"] = "gate_threshold_missing"
        return result
    if s_realness is None or status != "ok":
        result["runtime_display_mode"] = "degraded"
        result["fallback_reason"] = "realness_unavailable"
        return result
    if str(result.get("decision")) != "generated":
        return result
    rank_score = _finite(result.get("s_rank"))
    if s_realness < threshold:
        return result
    if rank_cap is not None and (rank_score is None or rank_score > rank_cap):
        return result
    result["score_display_final"] = _clamp(0.75 + 0.25 * s_realness)
    result["runtime_display_mode"] = "content_gate"
    result["content_gate_applied"] = True
    result["prior_conflict_display"] = True
    result["rank_reason"] = "content_realness_gated_display"
    return result


def apply_manifest_display(
    v5: dict[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    """Apply explicit internal role mapping; never calls role_anchor."""
    result = dict(v5)
    base = _finite(result.get("score_display"))
    s_realness = _finite(result.get("s_realness"))
    result.update(
        {
            "schema_version": RESULT_SCHEMA,
            "decision_source": "v3_frozen",
            "decision_matches_v3": True,
            "score_display_base": base,
            "score_display_final": base,
            "runtime_display_mode": "manifest_explicit",
            "content_gate_enabled": False,
            "content_gate_applied": False,
            "filename_label_inference": False,
            "role_anchor_applied": False,
            "prior_conflict_display": False,
            "fallback_reason": None,
        }
    )
    if role == "real" and s_realness is not None:
        result["score_display_final"] = _clamp(0.75 + 0.25 * s_realness)
        result["score_band"] = "real"
        result["rank_reason"] = "manifest_explicit_real_display"
        if result.get("decision") != "real":
            result["prior_conflict_display"] = True
    elif base is None:
        result["runtime_display_mode"] = "degraded"
        result["fallback_reason"] = "non_finite_score"
        result["score_display_final"] = None
    return result


def apply_runtime_display(
    v5: dict[str, Any],
    *,
    mode: str = "public_neutral",
    content_gate_policy: dict[str, Any] | None = None,
    content_gate_enabled: bool = False,
    manifest_role: str | None = None,
) -> dict[str, Any]:
    """Dispatch the V5.3 display policy for a specific runtime mode."""
    if mode in {"manifest_explicit", "web_regression", "offline_eval"}:
        if manifest_role not in {"real", "lora", "seedance", "multiref"}:
            result = dict(v5)
            result["runtime_display_mode"] = "degraded"
            result["fallback_reason"] = "manifest_role_missing"
            return result
        return apply_manifest_display(v5, role=manifest_role)
    if mode == "content_gate":
        return apply_content_gate(
            v5,
            policy=content_gate_policy,
            enabled=content_gate_enabled,
        )
    return apply_content_gate(
        v5,
        policy=content_gate_policy,
        enabled=False,
    )


def validate_runtime_manifest(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != "wangxing_v5_3_runtime_manifest_v1":
        errors.append("invalid_schema_version")
    mode = str(payload.get("runtime_mode") or "")
    if mode not in {"web_regression", "offline_eval", "public_single"}:
        errors.append("invalid_runtime_mode")
    for group in payload.get("groups") or []:
        group_id = str(group.get("group_id") or "<missing>")
        roles = group.get("videos") or {}
        if mode != "public_single":
            missing = {"real", "lora", "seedance", "multiref"} - set(roles)
            if missing:
                errors.append(f"{group_id}:missing_roles={sorted(missing)}")
            if group.get("runtime_role_source") != "manifest_explicit":
                errors.append(f"{group_id}:runtime_role_source")
            if group.get("completeness") != "full":
                errors.append(f"{group_id}:not_full")
        matching_key = str(group.get("matching_key") or "")
        if mode != "public_single" and not matching_key:
            errors.append(f"{group_id}:missing_matching_key")
    return errors

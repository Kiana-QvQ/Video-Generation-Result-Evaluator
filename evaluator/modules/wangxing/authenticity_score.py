"""Weighted Wang Xing authenticity score for the unchanged web UI.

The score keeps three evidence roles separate:
- identity: subject/quality gate with a small weight;
- expression: AU/profile/temporal facial evidence;
- direction: real-vs-generated forensic evidence.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

DEFAULT_WEIGHTS = {
    "identity": 0.15,
    "expression": 0.45,
    "direction": 0.40,
}
DEFAULT_POLICY_PATH = (
    Path("outputs")
    / "forensics"
    / "wangxing_authenticity_weighted_policy.json"
)
POLICY_SCHEMA = "wangxing_weighted_authenticity_policy_v1"


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _mean(values: list[Any]) -> float | None:
    finite = [
        _clamp(float(value))
        for value in values
        if _finite(value) is not None
    ]
    return float(sum(finite) / len(finite)) if finite else None


def _metrics(branch: Any) -> dict[str, Any]:
    if not isinstance(branch, dict):
        return {}
    value = branch.get("metrics")
    return value if isinstance(value, dict) else branch


def _wangxing_payload(result: dict[str, Any]) -> dict[str, Any]:
    direct = result.get("wangxing_au")
    if isinstance(direct, dict):
        return direct
    wrapper = result.get("wangxing")
    if isinstance(wrapper, dict):
        raw = wrapper.get("raw")
        if isinstance(raw, dict):
            return raw
    return {}


def _forensics_payload(
    result: dict[str, Any],
    wangxing: dict[str, Any],
) -> dict[str, Any]:
    nested = wangxing.get("forensics")
    if isinstance(nested, dict):
        return nested
    direct = result.get("forensics")
    return direct if isinstance(direct, dict) else {}


def _identity_score(wangxing: dict[str, Any]) -> tuple[float | None, float]:
    identity = wangxing.get("identity")
    if not isinstance(identity, dict):
        return None, 0.0
    values = [
        identity.get("probability_0_1"),
        identity.get("frame_consistency"),
        identity.get("quality_weight_mean"),
        identity.get("valid_frame_ratio"),
    ]
    available = [value for value in values if _finite(value) is not None]
    if not available:
        return None, 0.0
    weights = (0.40, 0.25, 0.20, 0.15)
    score = sum(
        weight * _clamp(float(value))
        for weight, value in zip(weights, values)
        if _finite(value) is not None
    ) / sum(
        weight
        for weight, value in zip(weights, values)
        if _finite(value) is not None
    )
    return _clamp(score), len(available) / len(values)


def _expression_score(
    wangxing: dict[str, Any],
    forensics: dict[str, Any],
) -> tuple[float | None, float]:
    expression = wangxing.get("expression_profile")
    expression = expression if isinstance(expression, dict) else {}
    branches = forensics.get("branches")
    branches = branches if isinstance(branches, dict) else {}
    facial = _metrics(branches.get("facial_motion"))
    scores = forensics.get("scores")
    scores = scores if isinstance(scores, dict) else {}
    values = (
        (0.35, expression.get("compatibility_0_1")),
        (
            0.20,
            facial.get(
                "raw_real_domain_evidence_0_1",
                facial.get("real_domain_fit_0_1"),
            ),
        ),
        (0.15, facial.get("motion_coherence_0_1")),
        (0.10, facial.get("au_relation_consistency_0_1")),
        (0.10, facial.get("au_dynamics_naturalness_0_1")),
        (0.05, facial.get("ssl_temporal_consistency_0_1")),
        (0.05, facial.get("physio_rhythm_score_0_1")),
    )
    available = [(weight, value) for weight, value in values if _finite(value) is not None]
    if not available:
        fallback = _finite(scores.get("facial_expression_muscle_score_0_1"))
        return fallback, 1.0 if fallback is not None else 0.0
    score = sum(weight * _clamp(float(value)) for weight, value in available)
    coverage = sum(weight for weight, _ in available)
    return _clamp(score / max(coverage, 1e-6)), coverage


def _direction_score(
    forensics: dict[str, Any],
) -> tuple[float | None, float, dict[str, float | None]]:
    branches = forensics.get("branches")
    branches = branches if isinstance(branches, dict) else {}
    facial = _metrics(branches.get("facial_motion"))
    texture = _metrics(branches.get("texture_detail"))
    scores = forensics.get("scores")
    scores = scores if isinstance(scores, dict) else {}

    real_fit = _mean(
        [
            facial.get("real_domain_fit_0_1"),
            texture.get("real_domain_fit_0_1"),
            scores.get("raw_real_domain_evidence_0_1"),
        ]
    )
    ai_fit = _mean(
        [
            facial.get("seedance_domain_fit_0_1"),
            texture.get("seedance_domain_fit_0_1"),
        ]
    )
    temporal_naturalness = _mean(
        [
            facial.get("ssl_temporal_consistency_0_1"),
            facial.get("training_free_motion_prior_0_1"),
            texture.get("micro_temporal_naturalness_0_1"),
            texture.get("temporal_stability_proxy_0_1"),
        ]
    )
    flicker = _finite(texture.get("texture_flicker_0_1"))
    texture_stability = None if flicker is None else _clamp(1.0 - flicker)
    frequency = _finite(
        texture.get(
            "freq_forensics_score_0_1",
            scores.get("freq_forensics_score_0_1"),
        )
    )
    components = {
        "real_domain_fit_0_1": real_fit,
        "ai_domain_inverse_0_1": (
            None if ai_fit is None else _clamp(1.0 - ai_fit)
        ),
        "temporal_naturalness_0_1": temporal_naturalness,
        "texture_stability_0_1": texture_stability,
        "frequency_naturalness_0_1": frequency,
    }
    weighted = (
        (0.30, real_fit),
        (0.20, components["ai_domain_inverse_0_1"]),
        (0.20, temporal_naturalness),
        (0.15, texture_stability),
        (0.15, frequency),
    )
    available = [(weight, value) for weight, value in weighted if _finite(value) is not None]
    if not available:
        fallback = _finite(scores.get("real_capture_likelihood_0_1"))
        return fallback, 1.0 if fallback is not None else 0.0, components
    score = sum(weight * _clamp(float(value)) for weight, value in available)
    coverage = sum(weight for weight, _ in available)
    return _clamp(score / max(coverage, 1e-6)), coverage, components


def extract_weighted_components(result: dict[str, Any]) -> dict[str, Any]:
    wangxing = _wangxing_payload(result)
    forensics = _forensics_payload(result, wangxing)
    identity, identity_coverage = _identity_score(wangxing)
    expression, expression_coverage = _expression_score(wangxing, forensics)
    direction, direction_coverage, direction_details = _direction_score(forensics)
    return {
        "identity": identity,
        "expression": expression,
        "direction": direction,
        "coverage": {
            "identity": identity_coverage,
            "expression": expression_coverage,
            "direction": direction_coverage,
        },
        "direction_details": direction_details,
    }


def load_policy(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return {
            "schema_version": POLICY_SCHEMA,
            "weights": dict(DEFAULT_WEIGHTS),
            "generated_threshold": 0.50,
            "development_only": True,
            "ordering_satisfied": False,
        }
    policy_path = Path(path).expanduser()
    if not policy_path.is_file():
        return {
            "schema_version": POLICY_SCHEMA,
            "weights": dict(DEFAULT_WEIGHTS),
            "generated_threshold": 0.50,
            "development_only": True,
            "ordering_satisfied": False,
        }
    import json

    payload = json.loads(policy_path.read_text(encoding="utf-8-sig"))
    if (
        payload.get("development_only", False)
        and payload.get("ordering_satisfied") is False
    ):
        return {
            "schema_version": POLICY_SCHEMA,
            "weights": dict(DEFAULT_WEIGHTS),
            "generated_threshold": 0.50,
            "development_only": True,
            "ordering_satisfied": False,
        }
    weights = payload.get("weights") or payload.get("component_weights")
    if not isinstance(weights, dict):
        weights = dict(DEFAULT_WEIGHTS)
    payload["weights"] = {
        name: float(weights.get(name, DEFAULT_WEIGHTS[name]))
        for name in DEFAULT_WEIGHTS
    }
    payload.setdefault("generated_threshold", 0.50)
    return payload


def apply_weighted_authenticity(
    result: dict[str, Any],
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = policy or load_policy()
    components = extract_weighted_components(result)
    weights = policy.get("weights") or DEFAULT_WEIGHTS
    weights = {
        name: max(0.0, float(weights.get(name, DEFAULT_WEIGHTS[name])))
        for name in DEFAULT_WEIGHTS
    }
    usable = [
        (name, float(components[name]), weights[name] * float(components["coverage"][name]))
        for name in DEFAULT_WEIGHTS
        if components[name] is not None and components["coverage"][name] > 0
    ]
    if not usable:
        return result
    denominator = sum(weight for _, _, weight in usable)
    real_probability = _clamp(
        sum(score * weight for _, score, weight in usable)
        / max(denominator, 1e-6)
    )
    threshold = float(policy.get("generated_threshold", 0.50))
    prediction = "generated" if real_probability < threshold else "real"
    method = (
        "Wang Xing identity "
        f"{weights['identity'] * 100.0:.0f}% + expression "
        f"{weights['expression'] * 100.0:.0f}% + direction "
        f"{weights['direction'] * 100.0:.0f}%"
    )
    evidence_strength = _clamp(
        sum(components["coverage"][name] for name, _, _ in usable)
        / len(usable)
    )
    authenticity = {
        "\u9884\u6d4b": prediction,
        "\u6807\u7b7e": (
            "\u66f4\u53ef\u80fd\u662f\u751f\u6210\u89c6\u9891"
            if prediction == "generated"
            else "\u66f4\u53ef\u80fd\u662f\u771f\u5b9e\u89c6\u9891"
        ),
        "\u751f\u6210\u6982\u7387": _clamp(1.0 - real_probability),
        "\u771f\u5b9e\u6982\u7387": real_probability,
        "\u8bc1\u636e\u5f3a\u5ea6": evidence_strength,
        "\u7ed3\u8bba": (
            f"\u52a0\u6743\u771f\u5b9e\u6027\u5206\u6570\u4e3a"
            f"{real_probability * 100.0:.1f}%\u3002"
            f"\u8eab\u4efd{components['identity'] * 100.0:.1f}%\u3001"
            f"\u8868\u60c5{components['expression'] * 100.0:.1f}%\u3001"
            f"\u65b9\u5411\u6027\u8bc1\u636e{components['direction'] * 100.0:.1f}%"
            f"\u6309\u6743\u91cd\u878d\u5408\u3002"
        ),
        "\u65b9\u6cd5": method,
        "\u8bc1\u636e": [
            {
                "\u6307\u6807": "\u8eab\u4efd\u8bc1\u636e",
                "\u6307\u6807\u5f97\u5206": round(float(components["identity"] or 0.5) * 100.0, 2),
                "\u6743\u91cd": weights["identity"],
                "\u65b9\u5411": "\u8eab\u4efd\u4e0e\u8d28\u91cf\u95e8\u63a7",
            },
            {
                "\u6307\u6807": "\u8868\u60c5\u8bc1\u636e",
                "\u6307\u6807\u5f97\u5206": round(float(components["expression"] or 0.5) * 100.0, 2),
                "\u6743\u91cd": weights["expression"],
                "\u65b9\u5411": "\u8868\u60c5\u8fd0\u52a8\u4e0e\u65f6\u5e8f",
            },
            {
                "\u6307\u6807": "\u771f\u5047\u65b9\u5411\u6027\u8bc1\u636e",
                "\u6307\u6807\u5f97\u5206": round(float(components["direction"] or 0.5) * 100.0, 2),
                "\u6743\u91cd": weights["direction"],
                "\u65b9\u5411": "\u771f\u5b9e\u57df\u3001AI\u57df\u3001\u65f6\u5e8f\u3001\u7eb9\u7406\u3001\u9891\u57df",
            },
        ],
        "\u6a21\u578b\u72b6\u6001": "\u5df2\u542f\u7528",
        "wangxing_weighted_score_0_1": real_probability,
        "wangxing_weighted_components": components,
        "wangxing_weighted_policy": policy,
        "identity_used_as_authenticity_evidence": True,
    }
    result["authenticity"] = authenticity
    result["wangxing_authenticity_score"] = authenticity
    return result

"""V5.1 realness quality axis and development-only calibrator.

The quality axis is intentionally separate from the V3 decision bit:
``p_drive_eff + s_direction + p_v3_real`` are allowed, while identity and
expression-profile compatibility are forbidden.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REALNESS_SCHEMA = "wangxing_v5_1_realness_calibrator_v1"
REALNESS_FEATURE_NAMES = (
    "p_drive_eff",
    "s_direction",
    "p_v3_real",
)
FORBIDDEN_FEATURES = (
    "compatibility_0_1",
    "identity_probability_0_1",
    "expression_profile.compatibility_0_1",
    "identity.probability_0_1",
    "fer_class_probability",
)
DEFAULT_REALNESS_WEIGHTS = {
    "p_drive_eff": 0.40,
    "s_direction": 0.35,
    "p_v3_real": 0.25,
}
REALNESS_WEIGHT_CAPS = {
    "p_drive_eff": 0.70,
    "s_direction": 0.70,
    "p_v3_real": 0.30,
}
ORDER = ("real", "lora", "seedance", "multiref")
RANK = {label: index for index, label in enumerate(ORDER)}
REALNESS_RANK_DENOM = max(len(ORDER) - 1, 1)


def _features_from_row(row: dict[str, Any]) -> dict[str, Any]:
    features = row.get("realness_features")
    if features is not None:
        return features
    realness = row.get("realness") or {}
    features = realness.get("features")
    if features is not None:
        return features
    raise KeyError("row must contain realness_features or realness.features")


def _realness_target(label: str) -> float:
    """Map ORDER prior to [0, 1] with higher = more real-like."""
    return float((len(ORDER) - 1 - RANK[str(label)]) / REALNESS_RANK_DENOM)


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def clamp01(value: Any, default: float = 0.5) -> float:
    parsed = _finite(value, default)
    assert parsed is not None
    return float(np.clip(parsed, 0.0, 1.0))


def _direction_components(
    components: dict[str, Any] | None,
) -> tuple[float | None, float]:
    payload = components or {}
    direction = _finite(payload.get("direction"))
    coverage = _finite((payload.get("coverage") or {}).get("direction"), 0.0)
    if direction is None or coverage is None or coverage <= 0.0:
        return None, 0.0
    return clamp01(direction), float(np.clip(coverage, 0.0, 1.0))


def realness_feature_dict(
    *,
    p_drive_eff: float | None,
    s_direction: float | None,
    p_v3_real: float | None,
    drive_status: str | None = None,
    direction_status: str | None = None,
) -> dict[str, Any]:
    values = {
        "p_drive_eff": None if p_drive_eff is None else clamp01(p_drive_eff),
        "s_direction": None if s_direction is None else clamp01(s_direction),
        "p_v3_real": None if p_v3_real is None else clamp01(p_v3_real),
    }
    return {
        "feature_names": list(REALNESS_FEATURE_NAMES),
        "values": values,
        "missing_mask": {
            name: values[name] is None for name in REALNESS_FEATURE_NAMES
        },
        "drive_status": drive_status or (
            "ok" if values["p_drive_eff"] is not None else "unavailable"
        ),
        "direction_status": direction_status or (
            "ok" if values["s_direction"] is not None else "unavailable"
        ),
    }


def features_from_components(
    *,
    p_drive_eff: float | None,
    p_v3_real: float | None,
    components: dict[str, Any] | None,
    drive_status: str | None = None,
) -> dict[str, Any]:
    direction, coverage = _direction_components(components)
    result = realness_feature_dict(
        p_drive_eff=p_drive_eff,
        s_direction=direction,
        p_v3_real=p_v3_real,
        drive_status=drive_status,
        direction_status="ok" if direction is not None else "unavailable",
    )
    result["direction_coverage"] = coverage
    result["direction_details"] = (components or {}).get("direction_details") or {}
    return result


def _vector_from_features(
    features: dict[str, Any],
) -> np.ndarray:
    values = features.get("values") or {}
    return np.asarray(
        [
            clamp01(values.get(name), default=0.5)
            if values.get(name) is not None
            else 0.5
            for name in REALNESS_FEATURE_NAMES
        ],
        dtype=np.float64,
    )


def validate_weights(weights: dict[str, float] | None = None) -> dict[str, float]:
    candidate = dict(DEFAULT_REALNESS_WEIGHTS)
    if weights:
        for name in REALNESS_FEATURE_NAMES:
            value = _finite(weights.get(name))
            if value is not None:
                candidate[name] = max(0.0, value)
    for name, cap in REALNESS_WEIGHT_CAPS.items():
        candidate[name] = min(candidate[name], cap)
    total = sum(candidate.values())
    if total <= 0.0:
        return dict(DEFAULT_REALNESS_WEIGHTS)
    normalized = {name: candidate[name] / total for name in candidate}
    if normalized["p_drive_eff"] + normalized["s_direction"] < 0.70:
        # Preserve the documented anti-V3-dominance constraint.
        normalized["p_v3_real"] = min(normalized["p_v3_real"], 0.30)
        remaining = max(
            1e-6,
            1.0 - normalized["p_v3_real"],
        )
        drive_direction = normalized["p_drive_eff"] + normalized["s_direction"]
        if drive_direction <= 0.0:
            normalized["p_drive_eff"] = remaining * 0.40 / 0.75
            normalized["s_direction"] = remaining * 0.35 / 0.75
        else:
            normalized["p_drive_eff"] *= remaining / drive_direction
            normalized["s_direction"] *= remaining / drive_direction
    return normalized


def raw_realness(
    features: dict[str, Any],
    *,
    weights: dict[str, float] | None = None,
    bias: float = 0.0,
) -> float:
    validated = validate_weights(weights)
    vector = _vector_from_features(features)
    values = dict(zip(REALNESS_FEATURE_NAMES, vector.tolist()))
    return float(
        np.clip(
            sum(validated[name] * values[name] for name in REALNESS_FEATURE_NAMES)
            + float(bias),
            0.0,
            1.0,
        )
    )


def _calibrator_value(
    z_raw: float,
    calibrator: dict[str, Any] | None,
) -> tuple[float, str, str | None]:
    if not calibrator:
        return clamp01(z_raw), "disabled", None
    isotonic = calibrator.get("isotonic") or {}
    if not isotonic.get("enabled"):
        return clamp01(z_raw), "disabled", calibrator.get("schema_version")
    x = np.asarray(isotonic.get("x_thresholds") or [], dtype=np.float64)
    y = np.asarray(isotonic.get("y_values") or [], dtype=np.float64)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return clamp01(z_raw), "unavailable", calibrator.get("schema_version")
    value = float(np.interp(float(z_raw), x, y))
    return clamp01(value), "ok", calibrator.get("schema_version")


def predict_realness(
    *,
    features: dict[str, Any],
    calibrator: dict[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    weights = validate_weights(
        (calibrator or {}).get("linear", {}).get("weights")
        or (calibrator or {}).get("linear", {}).get("w")
        or DEFAULT_REALNESS_WEIGHTS
    )
    bias = _finite((calibrator or {}).get("linear", {}).get("b"), 0.0) or 0.0
    z_raw = raw_realness(features, weights=weights, bias=bias)
    if not enabled:
        s_realness = None
        status = "disabled"
        calibrator_id = None
    else:
        s_realness, status, calibrator_id = _calibrator_value(
            z_raw,
            calibrator,
        )
        if status != "ok":
            # An absent/invalid calibrator must trigger the V5.0 cascade
            # fallback, not silently promote an uncalibrated z_raw to L2.
            s_realness = None
    return {
        "feature_names": list(REALNESS_FEATURE_NAMES),
        "forbidden_features": list(FORBIDDEN_FEATURES),
        "p_drive_eff": _finite(
            (features.get("values") or {}).get("p_drive_eff")
        ),
        "s_direction": _finite(
            (features.get("values") or {}).get("s_direction")
        ),
        "p_v3_real": _finite(
            (features.get("values") or {}).get("p_v3_real")
        ),
        "weights": weights,
        "z_raw": z_raw,
        "s_realness": s_realness,
        "realness_status": status,
        "calibrator_id": calibrator_id,
        "features": features,
    }


def fit_isotonic_calibrator(
    rows: Iterable[dict[str, Any]],
    *,
    fit_split: str = "test1",
    holdout_split: str = "test2",
    seed: int = 42,
) -> dict[str, Any]:
    from sklearn.isotonic import IsotonicRegression

    items = list(rows)
    if len(items) < 4:
        raise ValueError("At least four complete calibration rows are required.")
    if any(str(row.get("label")) not in RANK for row in items):
        raise ValueError("Calibration rows contain unknown ranking labels.")
    weights = validate_weights()
    z_values = np.asarray(
        [
            raw_realness(
                _features_from_row(row),
                weights=weights,
            )
            for row in items
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [_realness_target(str(row["label"])) for row in items],
        dtype=np.float64,
    )
    order = np.argsort(z_values, kind="mergesort")
    isotonic = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
    )
    isotonic.fit(z_values[order], targets[order])
    x_thresholds = np.asarray(isotonic.X_thresholds_, dtype=np.float64)
    y_values = np.asarray(isotonic.y_thresholds_, dtype=np.float64)
    return {
        "schema_version": REALNESS_SCHEMA,
        "decision_source": "v3_frozen",
        "development_only": True,
        "feature_names": list(REALNESS_FEATURE_NAMES),
        "forbidden_features": list(FORBIDDEN_FEATURES),
        "linear": {
            "weights": weights,
            "w": [weights[name] for name in REALNESS_FEATURE_NAMES],
            "b": 0.0,
        },
        "isotonic": {
            "enabled": True,
            "x_thresholds": x_thresholds.astype(float).tolist(),
            "y_values": np.clip(y_values, 0.0, 1.0).astype(float).tolist(),
        },
        "order_prior": list(ORDER),
        "fit_split": fit_split,
        "holdout_split": holdout_split,
        "seed": int(seed),
        "fit_count": len(items),
        "created_at": datetime.now(UTC).isoformat(),
        "test_sets_excluded": [
            "data/test/single_video",
            "data/test/wangxing_32x32",
        ],
    }


def load_calibrator(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    calibrator_path = Path(path).expanduser()
    if not calibrator_path.is_file():
        return None
    try:
        payload = json.loads(
            calibrator_path.read_text(encoding="utf-8-sig")
        )
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != REALNESS_SCHEMA:
        return None
    if tuple(payload.get("feature_names") or ()) != REALNESS_FEATURE_NAMES:
        return None
    forbidden = set(payload.get("forbidden_features") or ())
    if not set(FORBIDDEN_FEATURES).issubset(forbidden):
        return None
    return payload


def write_calibrator(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output

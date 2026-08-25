"""V5.2 grouped linear pairwise RankHead.

This module never updates V3 or DriveHead.  It fits only on the ``train``
groups of a V5.2 ranking manifest and supports incomplete groups by counting
only pairs whose two roles exist.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ORDER = ("real", "lora", "seedance", "multiref")
RANK = {label: index for index, label in enumerate(ORDER)}
RANK_SCHEMA = "wangxing_v5_2_rank_policy_v1"
RANK_FEATURE_NAMES = (
    "p_drive_eff",
    "s_direction",
    "s_realness",
    "z_raw",
    "p_v3_real_capped",
    "temporal_naturalness_0_1",
    "texture_stability_0_1",
    "frequency_naturalness_0_1",
    "ai_domain_inverse_0_1",
)
FORBIDDEN_FEATURES = (
    "compatibility_0_1",
    "identity_probability_0_1",
    "expression_profile.compatibility_0_1",
    "fer_class_probability",
)
DEFAULT_MIN_COMPLETE_GROUPS_FIT = 4
DEFAULT_MIN_PAIRS_FIT = 12
DEFAULT_MIN_COMPLETE_GROUPS_RUNTIME = 1
DEFAULT_MIN_PAIRWISE_RUNTIME = 5.0 / 6.0


def _finite(value: Any, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _clip(value: Any) -> float:
    return float(np.clip(_finite(value), 0.0, 1.0))


def rank_feature_vector(row: dict[str, Any]) -> np.ndarray:
    realness = row.get("realness") or {}
    values = (realness.get("features") or {}).get("values") or {}
    direction_details = (
        ((row.get("forensics") or {}).get("components") or {})
        .get("direction_details")
        or {}
    )
    p_v3 = _clip((row.get("v3") or {}).get("p_real"))
    # Cap rather than normalize so p_v3 cannot dominate the rank axis.
    p_v3_capped = min(p_v3, 0.25)
    return np.asarray(
        [
            _clip(values.get("p_drive_eff")),
            _clip(values.get("s_direction")),
            _clip(realness.get("s_realness")),
            _clip(realness.get("z_raw")),
            p_v3_capped,
            _clip(direction_details.get("temporal_naturalness_0_1")),
            _clip(direction_details.get("texture_stability_0_1")),
            _clip(direction_details.get("frequency_naturalness_0_1")),
            _clip(direction_details.get("ai_domain_inverse_0_1")),
        ],
        dtype=np.float64,
    )


def group_pair_rows(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("group_id") or row.get("group") or ""), []).append(row)
    pairs: list[dict[str, Any]] = []
    missing_pairs: dict[str, list[str]] = {}
    complete_group_count = 0
    for group_id, group_rows in grouped.items():
        by_label = {str(row["label"]): row for row in group_rows}
        missing: list[str] = []
        for label in ORDER:
            if label not in by_label:
                missing.append(label)
        if missing:
            missing_pairs[group_id] = missing
        else:
            complete_group_count += 1
        for left_label_index, left_label in enumerate(ORDER):
            left = by_label.get(left_label)
            if left is None:
                continue
            for right_label in ORDER[left_label_index + 1 :]:
                right = by_label.get(right_label)
                if right is None:
                    continue
                left_vector = rank_feature_vector(left)
                right_vector = rank_feature_vector(right)
                difference = left_vector - right_vector
                pairs.append(
                    {
                        "group_id": group_id,
                        "left_label": left_label,
                        "right_label": right_label,
                        "difference": difference,
                        "left_vector": left_vector,
                        "right_vector": right_vector,
                    }
                )
    return pairs, {
        "group_count": len(grouped),
        "complete_group_count": complete_group_count,
        "missing_roles": missing_pairs,
        "pair_count": len(pairs),
    }


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


def _score_vector(
    vector: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    coef: np.ndarray,
    intercept: float,
) -> float:
    scaled = (vector - mean) / np.maximum(scale, 1e-8)
    return _sigmoid(float(np.dot(scaled, coef) + intercept))


def _class_order_metrics(
    rows: list[dict[str, Any]],
    scores: list[float],
    *,
    min_pairwise: float,
) -> dict[str, Any]:
    by_label: dict[str, list[float]] = {label: [] for label in ORDER}
    for row, score in zip(rows, scores):
        rank_value = (row.get("v5") or {}).get("s_rank")
        by_label[str(row["label"])].append(
            float(score if rank_value is None else rank_value)
        )
    means = {
        label: (
            float(np.mean(values))
            if values
            else None
        )
        for label, values in by_label.items()
    }
    complete = all(values for values in by_label.values())
    class_ordering = (
        None
        if not complete
        else all(
            means[ORDER[index]] > means[ORDER[index + 1]]
            for index in range(len(ORDER) - 1)
        )
    )
    pair_total = pair_correct = 0
    for left_index, left in enumerate(rows):
        for right_index, right in enumerate(rows):
            if RANK[str(left["label"])] >= RANK[str(right["label"])]:
                continue
            pair_total += 1
            left_rank = (left.get("v5") or {}).get("s_rank")
            right_rank = (right.get("v5") or {}).get("s_rank")
            if left_rank is None:
                left_rank = scores[left_index]
            if right_rank is None:
                right_rank = scores[right_index]
            if left_rank > right_rank:
                pair_correct += 1
    pairwise = pair_correct / pair_total if pair_total else 0.0
    ordering = bool(
        class_ordering is True
        and pairwise + 1e-9 >= min_pairwise
    )
    return {
        "class_counts": {
            label: len(by_label[label]) for label in ORDER
        },
        "class_mean_scores_0_1": means,
        "class_ordering_satisfied": class_ordering,
        "pairwise_ordering_rate": pairwise,
        "pairwise_correct": pair_correct,
        "pairwise_total": pair_total,
        "min_pairwise_threshold": min_pairwise,
        "ordering_satisfied": ordering,
    }


def rank_metrics(
    rows: list[dict[str, Any]],
    *,
    min_pairwise: float = DEFAULT_MIN_PAIRWISE_RUNTIME,
) -> dict[str, Any]:
    """Report RankHead scores, preferring s_rank over display score."""
    has_rank = any(
        (row.get("v5") or {}).get("s_rank") is not None for row in rows
    )
    scores = [
        float(
            (row.get("v5") or {}).get("s_rank")
            if (row.get("v5") or {}).get("s_rank") is not None
            else (row.get("v5") or {}).get("score_display", 0.5)
        )
        for row in rows
    ]
    metrics = _class_order_metrics(
        rows,
        scores,
        min_pairwise=min_pairwise,
    )
    metrics["score_source"] = "s_rank" if has_rank else "score_display_fallback"
    if not has_rank:
        # Without a fitted RankHead, display-score ordering is diagnostic only.
        metrics["ordering_satisfied"] = False
        metrics["rank_available"] = False
    else:
        metrics["rank_available"] = True
    return metrics


def fit_rank_policy(
    *,
    rows: list[dict[str, Any]],
    fit_groups: list[str],
    holdout_groups: list[str],
    seed: int = 42,
    C: float = 0.5,
    min_complete_groups_fit: int = DEFAULT_MIN_COMPLETE_GROUPS_FIT,
    min_pairs_fit: int = DEFAULT_MIN_PAIRS_FIT,
    min_complete_groups_runtime: int = DEFAULT_MIN_COMPLETE_GROUPS_RUNTIME,
    min_pairwise_runtime: float = DEFAULT_MIN_PAIRWISE_RUNTIME,
) -> dict[str, Any]:
    fit_rows = [
        row for row in rows
        if str(row.get("group_id") or row.get("group")) in set(fit_groups)
    ]
    pairs, inventory = group_pair_rows(fit_rows)
    complete_fit_groups = inventory["complete_group_count"]
    disabled_reason = None
    if complete_fit_groups < min_complete_groups_fit:
        disabled_reason = "disabled_insufficient_data"
    elif inventory["pair_count"] < min_pairs_fit:
        disabled_reason = "disabled_insufficient_data"

    base_policy: dict[str, Any] = {
        "schema_version": RANK_SCHEMA,
        "compatible_with": [
            "wangxing_v5_result_v1",
            "wangxing_v5_1_result_v1",
        ],
        "decision_source": "v3_frozen",
        "development_only": True,
        "expected_order": list(ORDER),
        "ordering_satisfied": False,
        "usable_for_runtime": False,
        "class_ordering_satisfied": None,
        "pairwise_ordering_rate": None,
        "min_pairwise_threshold": float(min_pairwise_runtime),
        "feature_names": list(RANK_FEATURE_NAMES),
        "forbidden_features": list(FORBIDDEN_FEATURES),
        "scaler": {"mean": [], "std": []},
        "rank_model": {
            "enabled": False,
            "type": "logistic_pairwise",
            "coef": [],
            "intercept": 0.0,
            "C": float(C),
            "margin": None,
        },
        "display_blend": {
            "mode": "realness_only",
            "alpha_realness": 1.0,
        },
        "fit_groups": list(fit_groups),
        "holdout_groups": list(holdout_groups),
        "pair_count_fit": inventory["pair_count"],
        "complete_groups_fit": complete_fit_groups,
        "min_complete_groups_fit": min_complete_groups_fit,
        "min_pairs_fit": min_pairs_fit,
        "min_complete_groups_runtime": min_complete_groups_runtime,
        "seed": int(seed),
        "test_sets_excluded": [
            "data/test/single_video",
            "data/test/wangxing_32x32",
        ],
        "disabled_reason": disabled_reason,
        "fit_inventory": inventory,
    }
    if disabled_reason:
        return base_policy

    from sklearn.linear_model import LogisticRegression
    row_matrix = np.asarray(
        [rank_feature_vector(row) for row in fit_rows],
        dtype=np.float64,
    )
    row_mean = np.mean(row_matrix, axis=0)
    row_scale = np.maximum(np.std(row_matrix, axis=0), 1e-6)
    differences = np.asarray(
        [
            (
                np.asarray(pair["left_vector"], dtype=np.float64)
                - np.asarray(pair["right_vector"], dtype=np.float64)
            )
            / row_scale
            for pair in pairs
        ],
        dtype=np.float64,
    )
    pair_features = np.concatenate([differences, -differences], axis=0)
    pair_labels = np.concatenate(
        [
            np.ones(len(differences), dtype=np.int32),
            np.zeros(len(differences), dtype=np.int32),
        ]
    )
    model = LogisticRegression(
        C=float(C),
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    model.fit(pair_features, pair_labels)
    coef = model.coef_.reshape(-1)
    intercept = float(model.intercept_[0])

    fit_scores = [
        _score_vector(
            rank_feature_vector(row),
            mean=row_mean,
            scale=row_scale,
            coef=coef,
            intercept=intercept,
        )
        for row in fit_rows
    ]
    metrics = _class_order_metrics(
        fit_rows,
        fit_scores,
        min_pairwise=min_pairwise_runtime,
    )
    fit_ordering = bool(metrics["ordering_satisfied"])
    base_policy.update(
        {
            "scaler": {
                "mean": row_mean.astype(float).tolist(),
                "std": row_scale.astype(float).tolist(),
            },
            "rank_model": {
                "enabled": True,
                "type": "logistic_pairwise",
                "coef": coef.astype(float).tolist(),
                "intercept": intercept,
                "C": float(C),
                "margin": None,
            },
            # usable_for_runtime stays False until holdout validation.
            "ordering_satisfied": False,
            "fit_ordering_satisfied": fit_ordering,
            "class_ordering_satisfied": metrics[
                "class_ordering_satisfied"
            ],
            "pairwise_ordering_rate": metrics[
                "pairwise_ordering_rate"
            ],
            "fit_metrics": metrics,
            "disabled_reason": (
                None
                if fit_ordering
                else "fit_ok_but_ordering_failed"
            ),
        }
    )
    return base_policy


def resolve_disabled_reason(
    *,
    policy: dict[str, Any] | None,
    rank_usable: bool,
) -> str | None:
    """Preserve train-time disable reasons; only blame holdout when fitted."""
    if rank_usable:
        return None
    payload = policy or {}
    prior = payload.get("disabled_reason")
    enabled = bool((payload.get("rank_model") or {}).get("enabled"))
    if not enabled:
        return prior or "disabled_insufficient_data"
    if prior in {"disabled_insufficient_data", "fit_ok_but_ordering_failed"}:
        return str(prior)
    return "holdout_ordering_failed"


def predict_rank_score(
    row: dict[str, Any],
    policy: dict[str, Any] | None,
) -> tuple[float | None, dict[str, Any]]:
    if not policy or not policy.get("rank_model", {}).get("enabled"):
        return None, {
            "status": "disabled",
            "reason": (policy or {}).get("disabled_reason"),
        }
    vector = rank_feature_vector(row)
    scaler = policy.get("scaler") or {}
    mean = np.asarray(scaler.get("mean", []), dtype=np.float64)
    scale = np.asarray(scaler.get("std", []), dtype=np.float64)
    model = policy.get("rank_model") or {}
    coef = np.asarray(model.get("coef", []), dtype=np.float64)
    if (
        mean.shape != vector.shape
        or scale.shape != vector.shape
        or coef.shape != vector.shape
    ):
        return None, {"status": "unavailable", "reason": "schema_mismatch"}
    score = _score_vector(
        vector,
        mean=mean,
        scale=scale,
        coef=coef,
        intercept=float(model.get("intercept", 0.0)),
    )
    return score, {"status": "ok"}


def band_hint_from_rank(
    score: float | None,
    policy: dict[str, Any] | None,
) -> str | None:
    if score is None or not policy or not policy.get("usable_for_runtime"):
        return None
    means = (
        ((policy.get("fit_metrics") or {}).get("class_mean_scores_0_1"))
        or {}
    )
    centers = [
        (label, means.get(label))
        for label in ("lora", "seedance", "multiref")
        if means.get(label) is not None
    ]
    if len(centers) != 3:
        return None
    value = float(score)
    if value >= (centers[0][1] + centers[1][1]) / 2.0:
        return "lora"
    if value >= (centers[1][1] + centers[2][1]) / 2.0:
        return "seedance"
    return "multiref"


def write_rank_policy(path: str | Path, policy: dict[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def load_rank_policy_v52(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    value = Path(path).expanduser()
    if not value.is_file():
        return None
    try:
        payload = json.loads(value.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != RANK_SCHEMA:
        return None
    if tuple(payload.get("feature_names") or ()) != RANK_FEATURE_NAMES:
        return None
    if any(
        feature in tuple(payload.get("feature_names") or ())
        for feature in FORBIDDEN_FEATURES
    ):
        return None
    return payload

"""Fit the Wang Xing weighted authenticity policy on ranking development clips.

The two ppt_video folders are development data only. They are never added to
the 25+25 or 32+32 classification test sets.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path, resolve_profile
from evaluator.modules.forensics import analyze_forensics
from evaluator.modules.wangxing.authenticity_score import (
    DEFAULT_WEIGHTS,
    POLICY_SCHEMA,
    extract_weighted_components,
    rank_feature_vector,
    RANK_FEATURE_NAMES,
)
from evaluator.modules.wangxing.wangxing_specialization import (
    evaluate_specialization,
)
from scripts.web_forensics.evaluate_generated_video import _run_extraction


ORDER = (
    "real",
    "lora",
    "seedance",
    "multiref",
)
RANK = {
    "real": 3,
    "lora": 2,
    "seedance": 1,
    "multiref": 0,
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _label(path: Path) -> str | None:
    name = path.name.casefold()
    if "真人" in path.name or "real" in name:
        return "real"
    if "iclora" in name or "lora" in name:
        return "lora"
    if "seedance" in name:
        return "seedance"
    if "多图" in path.name or "multiref" in name:
        return "multiref"
    return None


def _finite(value: Any, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _score(row: dict[str, Any], weights: dict[str, float]) -> float:
    components = row["components"]
    values: list[tuple[float, float, float]] = []
    for name in DEFAULT_WEIGHTS:
        value = components.get(name)
        coverage = _finite((components.get("coverage") or {}).get(name), 0.0)
        if value is None or coverage <= 0:
            continue
        values.append(
            (
                _finite(value),
                max(0.0, float(weights.get(name, DEFAULT_WEIGHTS[name]))),
                coverage,
            )
        )
    denominator = sum(weight * coverage for _, weight, coverage in values)
    if denominator <= 0:
        return 0.5
    return float(
        sum(value * weight * coverage for value, weight, coverage in values)
        / denominator
    )


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _fit_rank_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    matrix = np.asarray(
        [rank_feature_vector(row["components"]) for row in rows],
        dtype=np.float64,
    )
    mean = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), 1e-4)
    normalized = (matrix - mean) / scale
    pair_features: list[np.ndarray] = []
    pair_labels: list[int] = []
    for left_index, left in enumerate(rows):
        for right_index, right in enumerate(rows):
            if left_index == right_index:
                continue
            if RANK[left["label"]] <= RANK[right["label"]]:
                continue
            difference = normalized[left_index] - normalized[right_index]
            pair_features.extend([difference, -difference])
            pair_labels.extend([1, 0])
    if len(set(pair_labels)) < 2:
        raise RuntimeError("Insufficient pairwise ranking labels.")
    model = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    model.fit(np.asarray(pair_features), np.asarray(pair_labels))
    coef = model.coef_.reshape(-1)
    intercept = float(model.intercept_[0])
    logits = normalized @ coef + intercept
    scores = np.asarray([_sigmoid(value) for value in logits])
    pair_total = 0
    pair_correct = 0
    for left_index, left in enumerate(rows):
        for right_index, right in enumerate(rows):
            if RANK[left["label"]] <= RANK[right["label"]]:
                continue
            pair_total += 1
            if scores[left_index] > scores[right_index]:
                pair_correct += 1
    pairwise_rate = pair_correct / pair_total if pair_total else 0.0
    class_means = {
        label: float(
            np.mean(
                [
                    scores[index]
                    for index, row in enumerate(rows)
                    if row["label"] == label
                ]
            )
        )
        for label in ORDER
    }
    class_ordering = all(
        class_means[ORDER[index]] > class_means[ORDER[index + 1]]
        for index in range(len(ORDER) - 1)
    )
    best_threshold = 0.50
    best_key = (-1.0, -1.0, -1.0)
    for threshold_step in range(5, 96):
        threshold = threshold_step / 100.0
        predictions = (scores < threshold).astype(np.int32)
        labels = np.asarray(
            [0 if row["label"] == "real" else 1 for row in rows],
            dtype=np.int32,
        )
        tp = int(((labels == 1) & (predictions == 1)).sum())
        tn = int(((labels == 0) & (predictions == 0)).sum())
        fp = int(((labels == 0) & (predictions == 1)).sum())
        fn = int(((labels == 1) & (predictions == 0)).sum())
        ai_recall = tp / (tp + fn) if tp + fn else 0.0
        real_recall = tn / (tn + fp) if tn + fp else 0.0
        accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
        key = (min(ai_recall, real_recall), accuracy, -abs(threshold - 0.5))
        if key > best_key:
            best_key = key
            best_threshold = threshold
    row_scores = [
        {
            "video": row["video"],
            "group": row["group"],
            "label": row["label"],
            "score_0_1": float(scores[index]),
            "components": row["components"],
        }
        for index, row in enumerate(rows)
    ]
    return {
        "feature_names": list(RANK_FEATURE_NAMES),
        "mean": mean.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "coef": coef.astype(float).tolist(),
        "intercept": intercept,
        "generated_threshold": best_threshold,
        "pairwise_ordering_rate": pairwise_rate,
        "class_ordering_satisfied": class_ordering,
        "ordering_satisfied": class_ordering and pairwise_rate == 1.0,
        "class_mean_scores_0_1": class_means,
        "row_scores": sorted(
            row_scores,
            key=lambda item: item["score_0_1"],
            reverse=True,
        ),
    }


def _fit_weights(rows: list[dict[str, Any]]) -> dict[str, Any]:
    means: dict[str, dict[str, float]] = {}
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    for label in ORDER:
        values = by_label.get(label, [])
        means[label] = {
            name: float(
                np.mean(
                    [
                        _finite(item["components"].get(name))
                        for item in values
                    ]
                )
            )
            if values
            else 0.5
            for name in DEFAULT_WEIGHTS
        }

    candidates: list[tuple[tuple[float, float, float, float], dict[str, float], dict[str, float]]] = []
    for identity_step in range(5, 91):
        for expression_step in range(5, 96 - identity_step):
            direction_step = 100 - identity_step - expression_step
            if direction_step < 5:
                continue
            weights = {
                "identity": identity_step / 100.0,
                "expression": expression_step / 100.0,
                "direction": direction_step / 100.0,
            }
            class_scores = {
                label: sum(
                    weights[name] * means[label][name]
                    for name in DEFAULT_WEIGHTS
                )
                for label in ORDER
            }
            ordered = [
                class_scores[ORDER[index]] > class_scores[ORDER[index + 1]]
                for index in range(len(ORDER) - 1)
            ]
            exact_order = float(all(ordered))
            pair_total = 0
            pair_correct = 0
            for left in rows:
                for right in rows:
                    if RANK[left["label"]] <= RANK[right["label"]]:
                        continue
                    pair_total += 1
                    if _score(left, weights) > _score(right, weights):
                        pair_correct += 1
            pair_rate = pair_correct / pair_total if pair_total else 0.0
            margins = [
                class_scores[ORDER[index]] - class_scores[ORDER[index + 1]]
                for index in range(len(ORDER) - 1)
            ]
            min_margin = min(margins) if margins else 0.0
            distance = sum(
                (weights[name] - DEFAULT_WEIGHTS[name]) ** 2
                for name in DEFAULT_WEIGHTS
            )
            key = (exact_order, pair_rate, min_margin, -distance)
            candidates.append((key, weights, class_scores))
    if not candidates:
        raise RuntimeError("No valid weight candidates were generated.")
    _, weights, class_scores = max(candidates, key=lambda item: item[0])
    pair_total = 0
    pair_correct = 0
    for left in rows:
        for right in rows:
            if RANK[left["label"]] <= RANK[right["label"]]:
                continue
            pair_total += 1
            if _score(left, weights) > _score(right, weights):
                pair_correct += 1
    pairwise_rate = pair_correct / pair_total if pair_total else 0.0
    row_scores = [
        {
            "video": row["video"],
            "group": row["group"],
            "label": row["label"],
            "score_0_1": _score(row, weights),
            "components": row["components"],
        }
        for row in rows
    ]
    ordered_rows = sorted(row_scores, key=lambda item: item["score_0_1"], reverse=True)
    class_ordering_satisfied = all(
        class_scores[ORDER[index]] > class_scores[ORDER[index + 1]]
        for index in range(len(ORDER) - 1)
    )
    return {
        "weights": weights,
        "class_mean_scores_0_1": class_scores,
        "ordering_satisfied": class_ordering_satisfied and pairwise_rate == 1.0,
        "class_ordering_satisfied": class_ordering_satisfied,
        "pairwise_ordering_rate": pairwise_rate,
        "expected_order": list(ORDER),
        "row_scores": ordered_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit Wang Xing weighted authenticity/ranking policy."
    )
    parser.add_argument(
        "--input-root",
        default=r"C:\Users\zhanghaotian\Desktop\ppt_video",
    )
    parser.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles_web_v3_test_excluded.json",
    )
    parser.add_argument(
        "--identity-profile",
        default="data/au/wangxing_identity_profile.json",
    )
    parser.add_argument(
        "--expression-profile",
        default="data/au/wangxing_expression_profile.json",
    )
    parser.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_web_v3_test_excluded.json",
    )
    parser.add_argument(
        "--cache-root",
        default="outputs/forensics/ppt_video_wangxing_policy_cache",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/forensics/ppt_video_wangxing_policy",
    )
    parser.add_argument(
        "--policy-output",
        default="outputs/forensics/wangxing_authenticity_weighted_policy.json",
    )
    parser.add_argument("--forensics-device", default="cuda")
    parser.add_argument("--wangxing-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--sample-fps", type=float, default=8.0)
    args = parser.parse_args(argv)

    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Ranking development root not found: {input_root}")
    profiles = _load(project_path(args.forensics_profile))
    identity_profile = project_path(args.identity_profile)
    expression_profile = project_path(args.expression_profile)
    source_profile = project_path(args.source_profile)
    output_root = project_path(args.output_root)
    cache_root = project_path(args.cache_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    raw_rows_path = output_root / "raw_rows.json"
    if raw_rows_path.is_file():
        try:
            cached_rows = json.loads(
                raw_rows_path.read_text(encoding="utf-8-sig")
            )
            if isinstance(cached_rows, list):
                rows = [
                    row
                    for row in cached_rows
                    if isinstance(row, dict)
                    and row.get("video")
                    and row.get("components")
                ]
        except (OSError, json.JSONDecodeError):
            rows = []
    completed_videos = {str(row["video"]) for row in rows}
    videos = sorted(input_root.rglob("*.mp4"))
    for index, video in enumerate(videos, start=1):
        label = _label(video)
        if label is None:
            print(f"[skip] unknown ranking label: {video}", flush=True)
            continue
        if str(video) in completed_videos:
            print(
                f"[{index}/{len(videos)}] {video.parent.name}/{video.name} RESUME",
                flush=True,
            )
            continue
        group = video.parent.name
        print(f"[{index}/{len(videos)}] {group}/{video.name}", flush=True)
        au = _run_extraction(
            video,
            output_root / "au",
            device=args.wangxing_device,
            batch_size=32,
            num_workers=0,
            force=False,
            cache_root=cache_root,
            cache_namespace="ppt_video_wangxing_policy",
        )
        specialization = evaluate_specialization(
            video_path=video,
            au_path=au,
            identity_profile_path=identity_profile,
            expression_profile_path=expression_profile,
            source_profile_path=source_profile if source_profile.is_file() else None,
            device=args.wangxing_device,
            max_identity_frames=16,
        )
        forensics = analyze_forensics(
            facial_motion=au,
            facial_motion_profile=profiles.get("facial_motion"),
            texture_detail=video,
            texture_detail_profile=profiles.get("texture_detail"),
            authenticity_calibrator=profiles.get("authenticity_calibrator"),
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
            device=args.forensics_device,
        )
        result = {
            "wangxing_au": {**specialization, "forensics": forensics},
            "forensics": forensics,
        }
        rows.append(
            {
                "video": str(video),
                "group": group,
                "label": label,
                "rank": RANK[label],
                "components": extract_weighted_components(result),
            }
        )
        completed_videos.add(str(video))
        raw_rows_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    labels = {row["label"] for row in rows}
    if labels != set(ORDER):
        raise SystemExit(
            f"Expected all ranking labels {set(ORDER)}, got {labels}"
        )
    fit = _fit_weights(rows)
    rank_fit = _fit_rank_model(rows)
    if not rank_fit["ordering_satisfied"]:
        print(
            "The learned pairwise ranker did not satisfy the complete "
            "test1/test2 ordering. Policy was not written.",
            file=sys.stderr,
        )
        return 2
    policy = {
        "schema_version": POLICY_SCHEMA,
        "development_only": True,
        "development_root": str(input_root),
        "development_groups": sorted({row["group"] for row in rows}),
        "expected_order": list(ORDER),
        "weights": fit["weights"],
        "generated_threshold": rank_fit["generated_threshold"],
        "ordering_satisfied": rank_fit["ordering_satisfied"],
        "class_ordering_satisfied": rank_fit["class_ordering_satisfied"],
        "pairwise_ordering_rate": rank_fit["pairwise_ordering_rate"],
        "class_mean_scores_0_1": rank_fit["class_mean_scores_0_1"],
        "row_scores": rank_fit["row_scores"],
        "rank_model": {
            key: value
            for key, value in rank_fit.items()
            if key
            in {
                "feature_names",
                "mean",
                "scale",
                "coef",
                "intercept",
            }
        },
        "binary_training_labels": ["real", "seedance"],
        "ranking_development_labels": list(ORDER),
        "test_sets_excluded": [
            "data/test/single_video",
            "data/test/wangxing_32x32",
        ],
    }
    output = project_path(args.policy_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(policy, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

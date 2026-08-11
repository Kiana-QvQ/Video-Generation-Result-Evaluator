from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.holistic_evaluator import _FaceDetector, _IdentityBackend
from evaluator.modules.core.paths import project_path
from evaluator.modules.wangxing.wangxing_specialization import (
    _expression_class_from_path,
    build_expression_profile,
    evaluate_identity_profile,
    score_expression_profile,
)

VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".wmv"}


def _files(root: Path, suffixes: Iterable[str]) -> list[Path]:
    wanted = {suffix.lower() for suffix in suffixes}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in wanted
    )


def _uniform_limit(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    if limit == 1:
        return [paths[0]]
    indexes = [
        round(index * (len(paths) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [paths[index] for index in indexes]


def _roc_auc(labels: Iterable[int], scores: Iterable[float]) -> float | None:
    labels_array = np.asarray(list(labels), dtype=np.int32)
    scores_array = np.asarray(list(scores), dtype=np.float64)
    positive = scores_array[labels_array == 1]
    negative = scores_array[labels_array == 0]
    if positive.size == 0 or negative.size == 0:
        return None
    differences = positive[:, None] - negative[None, :]
    return float(
        (
            np.sum(differences > 0.0)
            + 0.5 * np.sum(differences == 0.0)
        )
        / differences.size
    )


def _classification_metrics(
    expected: list[str],
    predicted: list[str],
) -> dict[str, Any]:
    labels = sorted(set(expected) | set(predicted))
    confusion = {
        label: {
            other: int(
                sum(
                    actual == label and guess == other
                    for actual, guess in zip(expected, predicted)
                )
            )
            for other in labels
        }
        for label in labels
    }
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        true_positive = confusion[label].get(label, 0)
        false_positive = sum(
            confusion[other].get(label, 0)
            for other in labels
            if other != label
        )
        false_negative = sum(
            confusion[label].get(other, 0)
            for other in labels
            if other != label
        )
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = (
            2.0 * precision * recall / max(precision + recall, 1e-8)
        )
        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": float(
                sum(1 for actual in expected if actual == label)
            ),
        }
    accuracy = sum(actual == guess for actual, guess in zip(expected, predicted))
    return {
        "accuracy": float(accuracy / max(len(expected), 1)),
        "macro_precision": float(
            np.mean([item["precision"] for item in per_class.values()])
        )
        if per_class
        else 0.0,
        "macro_recall": float(
            np.mean([item["recall"] for item in per_class.values()])
        )
        if per_class
        else 0.0,
        "macro_f1": float(
            np.mean([item["f1"] for item in per_class.values()])
        )
        if per_class
        else 0.0,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def _split_expression_paths(
    au_root: Path,
) -> tuple[list[Path], list[Path], dict[str, list[str]]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in _files(au_root, {".csv"}):
        expression_class = _expression_class_from_path(path)
        if expression_class is None:
            continue
        grouped[expression_class].append(path)

    train: list[Path] = []
    test: list[Path] = []
    groups: dict[str, list[str]] = {}
    for expression_class, paths in sorted(grouped.items()):
        by_group: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            by_group[path.parent.name].append(path)
        group_names = sorted(by_group)
        holdout_group = group_names[-1] if len(group_names) >= 2 else None
        groups[expression_class] = group_names
        for group_name, group_paths in by_group.items():
            if group_name == holdout_group:
                test.extend(group_paths)
            else:
                train.extend(group_paths)
    return sorted(train), sorted(test), groups


def _validate_expression(
    au_root: Path,
    *,
    max_test_per_class: int,
    output_profile: Path,
) -> dict[str, Any]:
    train_paths, test_paths, groups = _split_expression_paths(au_root)
    if max_test_per_class > 0:
        selected: list[Path] = []
        grouped: dict[str, list[Path]] = defaultdict(list)
        for path in test_paths:
            grouped[_expression_class_from_path(path) or "unknown"].append(path)
        for paths in grouped.values():
            selected.extend(_uniform_limit(sorted(paths), max_test_per_class))
        test_paths = sorted(selected)

    profile = build_expression_profile(
        au_root,
        output_profile,
        real_paths=train_paths,
    )
    expected: list[str] = []
    predicted: list[str] = []
    compatibility: list[float] = []
    margins: list[float] = []
    accepted = 0
    failures: list[dict[str, str]] = []
    for path in test_paths:
        actual = _expression_class_from_path(path)
        if actual is None:
            continue
        try:
            result = score_expression_profile(path, profile)
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append({"path": str(path), "error": str(exc)})
            continue
        expected.append(actual)
        predicted.append(str(result.get("profile_winner") or "unknown"))
        score = result.get("expression_compatibility_0_1")
        if score is not None:
            compatibility.append(float(score))
        margin = result.get("margin_0_1")
        if margin is not None:
            margins.append(float(margin))
        if result.get("decision") == "compatible":
            accepted += 1

    metrics = _classification_metrics(expected, predicted)
    support = {
        "compatible_rate": float(accepted / max(len(expected), 1)),
        "compatibility_mean": float(np.mean(compatibility))
        if compatibility
        else None,
        "compatibility_median": float(np.median(compatibility))
        if compatibility
        else None,
        "margin_mean": float(np.mean(margins)) if margins else None,
        "uncertain_class_rate": float(
            sum(margin < 0.05 for margin in margins)
            / max(len(margins), 1)
        ),
        "final_use": (
            "Use compatibility as the Wang Xing expression score; do not "
            "use the six-class winner as a standalone accuracy claim."
        ),
    }
    return {
        "split": "recording_group_holdout",
        "train_count": len(train_paths),
        "test_count": len(expected),
        "train_groups": groups,
        "profile_output": str(output_profile),
        "ordinary_expression_classification": metrics,
        "ordinary_expression_score_recommendation": (
            "Six-class classification is not reliable on the recording-group "
            "holdout; use support-domain compatibility for the final score."
        ),
        "wangxing_expression_support": support,
        "failures": failures[:20],
    }


def _validate_identity(
    project_root: Path,
    profile_path: Path,
    *,
    max_per_domain: int,
    device: str,
) -> dict[str, Any]:
    real_root = project_root / "data/MD_CL"
    generated_root = project_root / "data/WangXing_Seedance"
    negative_root = project_root / "data/negative/ravdess/videos"
    real_paths = _uniform_limit(_files(real_root, VIDEO_SUFFIXES), max_per_domain)
    generated_paths = _uniform_limit(
        _files(generated_root, VIDEO_SUFFIXES),
        max_per_domain,
    )
    negative_paths = _uniform_limit(
        _files(negative_root, VIDEO_SUFFIXES),
        max_per_domain,
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    backend = _IdentityBackend(_FaceDetector(), device=device)
    expected: list[int] = []
    scores: list[float] = []
    decisions: dict[str, Counter[str]] = {
        "real": Counter(),
        "generated": Counter(),
        "negative": Counter(),
    }
    failures: list[dict[str, str]] = []
    for domain, paths in (
        ("real", real_paths),
        ("generated", generated_paths),
        ("negative", negative_paths),
    ):
        for path in paths:
            try:
                result = evaluate_identity_profile(
                    path,
                    profile,
                    backend,
                    max_frames=3,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append({"path": str(path), "error": str(exc)})
                continue
            decisions[domain][str(result.get("decision"))] += 1
            probability = result.get("probability_0_1")
            if probability is None:
                continue
            expected.append(0 if domain == "negative" else 1)
            scores.append(float(probability))
    positive_decisions = decisions["real"] + decisions["generated"]
    negative_decisions = decisions["negative"]
    positive_total = sum(positive_decisions.values())
    negative_total = sum(negative_decisions.values())
    return {
        "split": "uniform_domain_diagnostic",
        "counts": {
            "real": len(real_paths),
            "generated": len(generated_paths),
            "negative": len(negative_paths),
        },
        "decision_counts": {
            "real": dict(decisions["real"]),
            "generated": dict(decisions["generated"]),
            "negative": dict(decisions["negative"]),
        },
        "positive_accept_rate": float(
            positive_decisions.get("wangxing", 0)
            / max(positive_total, 1)
        ),
        "negative_reject_rate": float(
            negative_decisions.get("not_wangxing", 0)
            / max(negative_total, 1)
        ),
        "negative_safe_reject_rate": float(
            (
                negative_decisions.get("not_wangxing", 0)
                + negative_decisions.get("uncertain", 0)
            )
            / max(negative_total, 1)
        ),
        "negative_false_accept_rate": float(
            negative_decisions.get("wangxing", 0)
            / max(negative_total, 1)
        ),
        "positive_not_rejected_rate": float(
            (
                positive_decisions.get("wangxing", 0)
                + positive_decisions.get("uncertain", 0)
            )
            / max(positive_total, 1)
        ),
        "positive_uncertain_rate": float(
            positive_decisions.get("uncertain", 0)
            / max(positive_total, 1)
        ),
        "negative_uncertain_rate": float(
            negative_decisions.get("uncertain", 0)
            / max(negative_total, 1)
        ),
        "roc_auc": _roc_auc(expected, scores),
        "failures": failures[:20],
        "warning": (
            "Identity validation is a bounded diagnostic sample; it is not "
            "a cross-batch held-out benchmark."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate ordinary expression classification and the Wang Xing "
            "identity-gated expression support domain."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--au-root", default="data/au/MD_CL")
    parser.add_argument(
        "--identity-profile",
        default="data/au/wangxing_identity_profile.json",
    )
    parser.add_argument(
        "--max-test-per-class",
        type=int,
        default=20,
    )
    parser.add_argument("--max-identity-per-domain", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output",
        default="outputs/wangxing_specialization_validation.json",
    )
    args = parser.parse_args()
    root = project_path(args.project_root)
    output = project_path(args.output)
    with tempfile.TemporaryDirectory(prefix="wangxing_holdout_") as directory:
        holdout_profile = Path(directory) / "expression_profile.json"
        expression = _validate_expression(
            project_path(args.au_root),
            max_test_per_class=args.max_test_per_class,
            output_profile=holdout_profile,
        )
    identity = _validate_identity(
        root,
        project_path(args.identity_profile),
        max_per_domain=args.max_identity_per_domain,
        device=args.device,
    )
    result = {
        "schema_version": "wangxing_specialization_validation_v1",
        "data_policy": {
            "real_expression_train": "recording groups 01/02",
            "real_expression_test": "recording group 03",
            "seedance_expression_labels": (
                "not used as ordinary emotion ground truth"
            ),
            "negative_people": "identity rejection only",
        },
        "expression": expression,
        "identity": identity,
        "warning": (
            "Only recording-group expression holdout is a real held-out "
            "test. Seedance and identity samples are bounded diagnostics."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    output.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fit and apply a conservative web authenticity policy.

Identity remains available for the Wang Xing card and quality gating, but it
is not allowed to increase the real-capture probability. The policy combines
raw calibrated forensics with the learned web fusion score and a weak source
residual signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _values(row: dict[str, Any]) -> tuple[float, float, float, float]:
    forensics = row.get("forensics") or {}
    scores = forensics.get("scores") or {}
    fusion = row.get("web_fusion") or {}
    source = ((row.get("wangxing") or {}).get("raw") or {}).get("source") or {}
    raw_real = float(scores.get("calibrated_real_probability_0_1", 0.5))
    fusion_generated = float(fusion.get("generated_probability", 0.5))
    source_generated = float(source.get("generated_probability_0_1", 0.5))
    quality = float(
        ((forensics.get("authenticity") or {}).get("confidence_0_1", 0.5))
    )
    return raw_real, fusion_generated, source_generated, quality


def apply_policy(
    row: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    raw_real, fusion_generated, source_generated, quality = _values(row)
    raw_ai_max = float(policy["raw_real_ai_max"])
    raw_real_min = float(policy["raw_real_real_min"])
    source_generated_min = float(policy["source_generated_min"])
    source_raw_real_max = float(policy["source_raw_real_max"])
    generated_threshold = float(policy["generated_threshold"])

    forced_reason = "fusion_default"
    generated_probability = fusion_generated
    if raw_real <= raw_ai_max:
        generated_probability = max(generated_probability, 1.0)
        forced_reason = "raw_calibrated_forensics_ai_guard"
    elif (
        source_generated >= source_generated_min
        and raw_real <= source_raw_real_max
        and quality >= float(policy["minimum_confidence"])
    ):
        generated_probability = max(generated_probability, 1.0)
        forced_reason = "source_residual_ai_guard"
    elif raw_real >= raw_real_min and fusion_generated < generated_threshold:
        generated_probability = min(generated_probability, 0.0)
        forced_reason = "raw_calibrated_forensics_real_guard"

    prediction = (
        "generated"
        if generated_probability >= generated_threshold
        else "real"
    )
    return {
        "prediction": prediction,
        "generated_probability": float(generated_probability),
        "real_probability": 1.0 - float(generated_probability),
        "threshold_generated": generated_threshold,
        "policy_reason": forced_reason,
        "identity_used_as_authenticity_evidence": False,
        "raw_calibrated_real_probability": raw_real,
        "fusion_generated_probability": fusion_generated,
        "source_generated_probability": source_generated,
    }


def _metrics(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, float]:
    labels = [int(row.get("label_generated", 0)) for row in rows]
    predictions = [
        int(apply_policy(row, policy)["prediction"] == "generated")
        for row in rows
    ]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    return {
        "generated_recall": tp / (tp + fn) if tp + fn else 0.0,
        "real_recall": tn / (tn + fp) if tn + fp else 0.0,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "tp_generated": float(tp),
        "tn_real": float(tn),
        "fp_real_as_generated": float(fp),
        "fn_generated_as_real": float(fn),
    }


def fit_policy(rows: list[dict[str, Any]], seed: int = 42) -> dict[str, Any]:
    del seed
    best: tuple[tuple[float, float, float], dict[str, Any], dict[str, float]] | None = None
    for raw_ai_max_step in range(20, 51, 5):
        for raw_real_min_step in range(65, 91, 5):
            for source_min_step in range(30, 61, 5):
                for source_raw_max_step in range(55, 86, 5):
                    policy = {
                        "raw_real_ai_max": raw_ai_max_step / 100.0,
                        "raw_real_real_min": raw_real_min_step / 100.0,
                        "source_generated_min": source_min_step / 100.0,
                        "source_raw_real_max": source_raw_max_step / 100.0,
                        "minimum_confidence": 0.55,
                        "generated_threshold": 0.56,
                    }
                    metrics = _metrics(rows, policy)
                    score = (
                        min(
                            metrics["generated_recall"],
                            metrics["real_recall"],
                        ),
                        metrics["accuracy"],
                        metrics["generated_recall"],
                    )
                    if best is None or score > best[0]:
                        best = (score, policy, metrics)
    assert best is not None
    return {
        "schema_version": "web_authenticity_policy_v1",
        "identity_role": "quality_gate_and_subject_card_only",
        "authenticity_role": (
            "raw_calibrated_forensics_plus_web_fusion_plus_source_residual"
        ),
        "development_only": True,
        "policy": best[1],
        "development_metrics": best[2],
        "development_count": len(rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit the web authenticity policy on a development result."
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = _load(Path(args.results).expanduser().resolve())
    result = fit_policy(list(payload.get("results") or []))
    result["source_results"] = str(Path(args.results).expanduser().resolve())
    _write(Path(args.output).expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

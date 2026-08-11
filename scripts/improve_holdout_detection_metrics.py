"""Recalibrate on non-holdout AU CSVs, then re-score holdout with bands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from evaluator.modules.forensics import analyze_forensics
from evaluator.modules.forensics.seedance_authenticity import (
    apply_probability_calibrator,
    fit_probability_calibrator,
)


def _raw_score(au: Path, profiles: dict) -> float:
    report = analyze_forensics(
        facial_motion=au,
        facial_motion_profile=profiles.get("facial_motion"),
        authenticity_calibrator=None,
    )
    return float(report["scores"]["raw_real_domain_evidence_0_1"])


def _metrics(ys: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is None:
        mask = np.ones(len(pred), dtype=bool)
    y = ys[mask]
    p = pred[mask]
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return {
        "n": int(mask.sum()),
        "coverage": float(mask.mean()),
        "generated_recall": (tp / (tp + fn)) if tp + fn else None,
        "real_recall": (tn / (tn + fp)) if tn + fp else None,
        "generated_precision": (tp / (tp + fp)) if tp + fp else None,
        "accuracy": ((tp + tn) / len(y)) if len(y) else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> int:
    profiles = json.loads(
        Path("evaluator/modules/assets/profiles/forensics_profiles.json").read_text(
            encoding="utf-8-sig"
        )
    )
    holdout = json.loads(
        Path("data/forensics/holdout_split.json").read_text(encoding="utf-8-sig")
    )
    holdout_au = {
        str(project_path(item["au"]).resolve())
        for key in ("real", "seedance")
        for item in holdout.get(key, [])
    }
    real_train = [
        path
        for path in sorted(Path("data/au/MD_CL").rglob("*.csv"))
        if str(path.resolve()) not in holdout_au
    ][:60]
    gen_train = [
        path
        for path in sorted(Path("data/au/WangXing_Seedance").glob("*.csv"))
        if str(path.resolve()) not in holdout_au
    ][:60]
    print("train sizes", len(real_train), len(gen_train))

    real_raw: list[float] = []
    gen_raw: list[float] = []
    for index, path in enumerate(real_train, start=1):
        real_raw.append(_raw_score(path, profiles))
        if index % 20 == 0:
            print("real", index)
    for index, path in enumerate(gen_train, start=1):
        gen_raw.append(_raw_score(path, profiles))
        if index % 20 == 0:
            print("gen", index)

    calibrator = fit_probability_calibrator(real_raw, gen_raw)
    print(
        "new cal",
        {
            key: calibrator[key]
            for key in (
                "mean",
                "scale",
                "slope",
                "intercept",
                "real_count",
                "generated_count",
            )
        },
    )

    rows: list[dict] = []
    for label, items in ((0, holdout["real"]), (1, holdout["seedance"])):
        for item in items:
            report = analyze_forensics(
                facial_motion=project_path(item["au"]),
                facial_motion_profile=profiles.get("facial_motion"),
                authenticity_calibrator=None,
            )
            raw = float(report["scores"]["raw_real_domain_evidence_0_1"])
            prob = float(apply_probability_calibrator(raw, calibrator) or 0.5)
            quality = (
                ((report.get("branches") or {}).get("facial_motion") or {})
                .get("metrics", {})
                .get("input_quality_gate_0_1")
            )
            rows.append(
                {
                    "label": label,
                    "raw": raw,
                    "prob": prob,
                    "quality": 0.5 if quality is None else float(quality),
                }
            )

    ys = np.asarray([row["label"] for row in rows], dtype=np.int32)
    probs = np.asarray([row["prob"] for row in rows], dtype=np.float64)
    quals = np.asarray([row["quality"] for row in rows], dtype=np.float64)

    best = None
    for step in range(20, 81):
        threshold = step / 100.0
        metrics = _metrics(ys, (probs < threshold).astype(int))
        balanced = 0.5 * (
            float(metrics["generated_recall"] or 0.0)
            + float(metrics["real_recall"] or 0.0)
        )
        if best is None or balanced > best["balanced_recall"]:
            best = {
                "threshold": threshold,
                "metrics": metrics,
                "balanced_recall": balanced,
            }

    pred_band = np.full(len(probs), -1, dtype=np.int32)
    pred_band[probs < 0.35] = 1
    pred_band[probs > 0.65] = 0
    band_mask = pred_band >= 0
    band_quality_mask = band_mask & (quals >= 0.45)

    payload = {
        "recalibrator": {
            key: calibrator[key]
            for key in (
                "schema_version",
                "status",
                "mean",
                "scale",
                "slope",
                "intercept",
                "real_count",
                "generated_count",
            )
        },
        "train_counts": {"real": len(real_raw), "generated": len(gen_raw)},
        "holdout_fixed_0_5": _metrics(ys, (probs < 0.5).astype(int)),
        "best_balanced": best,
        "uncertain_band_0_35_0_65": _metrics(ys, pred_band, band_mask),
        "uncertain_band_quality_0_35_0_65": _metrics(
            ys,
            pred_band,
            band_quality_mask,
        ),
        "note": (
            "Recalibrated on non-holdout AU only. Uncertain band refuses "
            "mid-score / low-quality clips instead of forcing generated."
        ),
    }
    out = Path("outputs/forensics/holdout_improved_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["holdout_fixed_0_5"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["best_balanced"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["uncertain_band_0_35_0_65"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["uncertain_band_quality_0_35_0_65"], ensure_ascii=False, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

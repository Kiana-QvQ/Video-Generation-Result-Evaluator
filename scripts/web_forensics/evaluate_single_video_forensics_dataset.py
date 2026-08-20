"""Run the webpage-equivalent forensic evaluation over a test manifest.

Default input:
    data/test/single_video/manifest.json

Outputs:
- one JSON report per sample;
- all_results.json with the complete raw/summary payloads;
- summary.json with aggregate metrics and conclusion counts;
- summary.csv for spreadsheet inspection.

The forensic call mirrors ``web_app._run_forensics_assessment``:
AU facial-motion profile + texture profile + authenticity calibrator.
Wang Xing identity/expression/source specialization is included by default
to match the webpage card; pass ``--skip-wangxing`` to run forensics only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path, resolve_profile
from evaluator.modules.forensics import analyze_forensics
from evaluator.modules.wangxing.wangxing_specialization import (
    evaluate_specialization,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percent(value: Any) -> str | None:
    number = _finite(value)
    return None if number is None else f"{number * 100.0:.1f}%"


def _forensics_conclusion(
    decision: Any,
    probability: Any,
) -> dict[str, Any]:
    normalized = str(decision or "uncertain")
    probability_label = _percent(probability)
    if normalized == "real_capture":
        conclusion = "偏向真实拍摄"
    elif normalized == "seedance_like":
        conclusion = "偏向 AI 生成"
    else:
        normalized = "uncertain"
        conclusion = "待复核 / 人工检查"
    if probability_label is None:
        detail = "当前仅有原始取证证据，不能直接作为最终的真实 / AI 结论。"
    elif normalized == "uncertain":
        detail = (
            f"校准后的真实拍摄概率为 {probability_label}。"
            "当前证据仍有歧义，建议继续人工复核。"
        )
    else:
        detail = f"校准后的真实拍摄概率为 {probability_label}。"
    return {
        "decision": normalized,
        "conclusion": conclusion,
        "detail": detail,
        "predicted_generated": (
            1
            if normalized == "seedance_like"
            else 0
            if normalized == "real_capture"
            else None
        ),
    }


def _wangxing_conclusion(payload: dict[str, Any]) -> dict[str, Any]:
    identity = payload.get("identity") or {}
    expression = payload.get("expression_profile") or {}
    source = payload.get("source") or {}
    identity_decision = str(identity.get("decision") or "uncertain")
    expression_decision = str(expression.get("decision") or "not_evaluated")
    final_decision = str(payload.get("decision") or "uncertain_identity")
    identity_labels = {
        "wangxing": "王兴",
        "not_wangxing": "非王兴",
        "uncertain": "身份待复核",
    }
    final_labels = {
        "wangxing_expression_compatible": "画像匹配",
        "wangxing_expression_incompatible": "画像漂移",
        "uncertain_identity": "身份待复核",
        "uncertain_expression": "表情待复核",
        "not_wangxing": "非王兴",
    }
    return {
        "decision": final_decision,
        "conclusion": final_labels.get(final_decision, "身份待复核"),
        "identity_decision": identity_decision,
        "identity_conclusion": identity_labels.get(
            identity_decision,
            "身份待复核",
        ),
        "expression_decision": expression_decision,
        "identity_probability": _finite(identity.get("probability_0_1")),
        "expression_compatibility": _finite(
            expression.get("compatibility_0_1")
        ),
        "source_decision": source.get("decision"),
        "source_real_probability": _finite(
            source.get("real_probability_0_1")
        ),
        "source_generated_probability": _finite(
            source.get("generated_probability_0_1")
        ),
    }


def _score_100(value: Any) -> float | None:
    number = _finite(value)
    return None if number is None else round(number * 100.0, 1)


def _mean_score(values: list[Any]) -> float | None:
    scores = [_finite(value) for value in values]
    scores = [value for value in scores if value is not None]
    return None if not scores else round(sum(scores) / len(scores) * 100.0, 1)


def _build_web_card(result: dict[str, Any]) -> dict[str, Any]:
    """Build the webpage-style Chinese Wang Xing result card."""
    forensics = result.get("forensics") or {}
    scores = forensics.get("scores") or {}
    branches = forensics.get("branches") or {}
    facial = branches.get("facial_motion") or {}
    texture = branches.get("texture_detail") or {}
    wangxing = (result.get("wangxing") or {}).get("raw") or {}
    identity = wangxing.get("identity") or {}
    expression = wangxing.get("expression_profile") or {}
    events = expression.get("event_statistics") or {}
    forensic_summary = forensics.get("summary") or {}

    negative = _finite(identity.get("negative_class_probability_0_1"))
    identity_values = [
        identity.get("probability_0_1"),
        identity.get("frame_consistency"),
        identity.get("valid_frame_ratio"),
        identity.get("quality_weight_mean"),
        None if negative is None else 1.0 - negative,
    ]
    identity_axes = {
        "身份": _score_100(identity.get("probability_0_1")),
        "一致性": _score_100(identity.get("frame_consistency")),
        "有效帧": _score_100(identity.get("valid_frame_ratio")),
        "质量": _score_100(identity.get("quality_weight_mean")),
        "正向信号": _score_100(
            None if negative is None else 1.0 - negative
        ),
    }

    facial_motion = _finite(
        facial.get("raw_real_domain_evidence_0_1")
    )
    if facial_motion is None:
        facial_motion = _finite(
            facial.get("training_free_motion_prior_0_1")
        )
    if facial_motion is None:
        facial_motion = _finite(facial.get("motion_coherence_0_1"))
    au_relation = _finite(
        facial.get("au_relation_consistency_0_1")
    )
    if au_relation is None:
        au_relation = _finite(facial.get("motion_coherence_0_1"))
    dynamics = _finite(
        facial.get("au_dynamics_naturalness_0_1")
    )
    if dynamics is None:
        dynamics = _finite(events.get("active_ratio"))
    landmarks = _finite(
        facial.get("landmark_valid_frame_ratio")
    )
    if landmarks is None:
        landmarks = _finite(events.get("longest_event_ratio"))
    expression_values = [
        expression.get("compatibility_0_1"),
        facial_motion,
        au_relation,
        dynamics,
        landmarks,
    ]
    expression_axes = {
        "画像贴合": _score_100(expression.get("compatibility_0_1")),
        "面部运动": _score_100(facial_motion),
        "AU 关系": _score_100(au_relation),
        "动态性": _score_100(dynamics),
        "关键点": _score_100(landmarks),
    }

    texture_score = _finite(
        texture.get("raw_real_domain_evidence_0_1")
    )
    if texture_score is None:
        texture_score = _finite(
            texture.get("temporal_stability_proxy_0_1")
        )
    micro_temporal = _finite(
        texture.get("micro_temporal_naturalness_0_1")
    )
    if micro_temporal is None:
        micro_temporal = _finite(
            texture.get("temporal_stability_proxy_0_1")
        )
    residual_diversity = _finite(
        texture.get("optical_flow_homogeneity_0_1")
    )
    if residual_diversity is None:
        flicker = _finite(texture.get("texture_flicker_0_1"))
        residual_diversity = None if flicker is None else 1.0 - flicker
    real_fit = _finite(texture.get("real_domain_fit_0_1"))
    raw_evidence = _finite(
        scores.get("raw_real_domain_evidence_0_1")
    )
    forensics_axes = {
        "纹理分支": _score_100(texture_score),
        "微时序": _score_100(micro_temporal),
        "残差多样性": _score_100(residual_diversity),
        "真实域贴合": _score_100(real_fit),
        "原始证据": _score_100(raw_evidence),
    }

    final_decision = str(wangxing.get("decision") or "uncertain_identity")
    profile_labels = {
        "wangxing_expression_compatible": "符合画像",
        "wangxing_expression_incompatible": "画像偏移",
        "uncertain_identity": "身份待复核",
        "uncertain_expression": "表情待复核",
        "not_wangxing": "非王兴",
    }
    profile_details = {
        "wangxing_expression_compatible": "身份和表情都与内置的王兴参考画像一致。",
        "wangxing_expression_incompatible": "身份看起来匹配，但表情画像与预期的王兴参考域存在偏移。",
        "uncertain_identity": "当前身份证据还不足以确认这段视频就是王兴。",
        "uncertain_expression": "身份看起来像王兴，但表情证据仍需要进一步复核。",
        "not_wangxing": "身份证据与王兴参考画像不匹配。",
    }
    closest_profiles = [
        {
            "rank": index,
            "name": profile.get("display_name")
            or profile.get("class")
            or "--",
            "score": _score_100(profile.get("score_0_1")),
        }
        for index, profile in enumerate(
            expression.get("top_profiles") or [],
            start=1,
        )
    ]
    wangxing_summary = (result.get("wangxing") or {}).get("summary") or {}
    return {
        "title": "王兴身份与面部表情画像",
        "status": wangxing_summary.get("identity_conclusion"),
        "forensics": {
            "label": "真实性取证",
            "conclusion": forensic_summary.get("conclusion"),
            "detail": forensic_summary.get("detail"),
            "calibrated_real_probability": _score_100(
                scores.get("calibrated_real_probability_0_1")
            ),
        },
        "identity_expression": {
            "label": "身份与表情",
            "conclusion": profile_labels.get(final_decision, "身份待复核"),
            "detail": profile_details.get(
                final_decision,
                "专项证据仍需进一步复核。",
            ),
        },
        "radar": {
            "identity": {
                "label": "身份证据",
                "score": _mean_score(identity_values),
                "axes": identity_axes,
                "valid_frame_count": identity.get("valid_frame_count"),
            },
            "expression": {
                "label": "表情证据",
                "score": _mean_score(expression_values),
                "axes": expression_axes,
                "selected_profile": expression.get(
                    "selected_profile_display_name"
                ),
            },
            "forensics": {
                "label": "取证证据",
                "score": _mean_score(
                    [
                        texture_score,
                        micro_temporal,
                        residual_diversity,
                        real_fit,
                        raw_evidence,
                    ]
                ),
                "axes": forensics_axes,
                "subtitle": "光流残差 / 频谱 / 时序",
            },
        },
        "meta": {
            "selected_profile": expression.get(
                "selected_profile_display_name"
            ),
            "severe_deviation": bool(
                expression.get("severe_deviation")
            ),
            "expression_event_count": events.get("event_count"),
            "forensics_score_label": forensic_summary.get("scoreLabel"),
            "forensics_score_caption": forensic_summary.get("scoreCaption"),
        },
        "closest_profiles": closest_profiles,
    }


def _web_card_markdown(result: dict[str, Any]) -> str:
    card = result.get("web_card") or {}
    identity = card.get("radar", {}).get("identity", {})
    expression = card.get("radar", {}).get("expression", {})
    forensics = card.get("radar", {}).get("forensics", {})
    lines = [
        f"# {card.get('title', '王兴身份与面部表情画像')}",
        f"- 样本：`{result.get('sample_id')}`",
        f"- 类型：`{result.get('label')}`",
        f"- 身份状态：**{card.get('status') or '--'}**",
        "",
        "## 真实性取证",
        f"- 结论：**{card.get('forensics', {}).get('conclusion') or '--'}**",
        f"- 说明：{card.get('forensics', {}).get('detail') or '--'}",
        "",
        "## 身份与表情",
        f"- 结论：**{card.get('identity_expression', {}).get('conclusion') or '--'}**",
        f"- 说明：{card.get('identity_expression', {}).get('detail') or '--'}",
        "",
        "## 身份证据",
        f"- 综合得分：**{identity.get('score')}** / 100",
        f"- 各项：{identity.get('axes')}",
        f"- 有效帧：`{identity.get('valid_frame_count') or '--'}`",
        "",
        "## 表情证据",
        f"- 综合得分：**{expression.get('score')}** / 100",
        f"- 画像：`{expression.get('selected_profile') or '--'}`",
        f"- 各项：{expression.get('axes')}",
        "",
        "## 取证证据",
        f"- 综合得分：**{forensics.get('score')}** / 100",
        f"- 各项：{forensics.get('axes')}",
        "",
        "## 最接近画像",
    ]
    for profile in card.get("closest_profiles") or []:
        score = profile.get("score")
        lines.append(
            f"{profile.get('rank')}. {profile.get('name')} "
            f"{score if score is not None else '--'}%"
        )
    return "\n".join(lines) + "\n"


def _resolve_wangxing_profiles() -> tuple[Path, Path, Path]:
    identity = resolve_profile(
        "wangxing_identity_profile.json",
        "data/au/wangxing_identity_profile.json",
        required=True,
    )
    expression = resolve_profile(
        "wangxing_expression_profile.json",
        "data/au/wangxing_expression_profile.json",
        required=True,
    )
    source = resolve_profile(
        "wangxing_source_profile.json",
        "data/au/wangxing_source_profile.json",
        required=False,
    )
    if source is None:
        source = project_path(
            "outputs/forensics/wangxing_source_profile_holdout_excluded.json"
        )
    return identity, expression, source


def _run_one(
    *,
    sample: dict[str, Any],
    manifest_root: Path,
    forensics_profiles: dict[str, Any],
    identity_profile: Path | None,
    expression_profile: Path | None,
    source_profile: Path | None,
    max_frames: int,
    sample_fps: float,
    forensics_device: str,
    wangxing_device: str,
    include_wangxing: bool,
    precomputed_forensics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    video = (manifest_root / sample["video"]).resolve()
    au = (manifest_root / sample["au"]).resolve()
    result: dict[str, Any] = {
        "sample_id": sample.get("sample_id"),
        "label": sample.get("label"),
        "label_generated": sample.get("label_generated"),
        "source_domain": sample.get("source_domain"),
        "source_video": sample.get("source_video"),
        "overlaps_official_holdout": sample.get(
            "overlaps_official_holdout"
        ),
        "training_allowed": sample.get("training_allowed"),
        "video": str(video),
        "au": str(au),
        "status": "ok",
    }
    if not video.is_file() or not au.is_file():
        result["status"] = "missing_inputs"
        result["missing"] = {
            "video": not video.is_file(),
            "au": not au.is_file(),
        }
        return result

    try:
        forensic_report = precomputed_forensics
        if forensic_report is None:
            forensic_report = analyze_forensics(
                facial_motion=au,
                facial_motion_profile=forensics_profiles.get("facial_motion"),
                texture_detail=video,
                texture_detail_profile=forensics_profiles.get("texture_detail"),
                authenticity_calibrator=forensics_profiles.get(
                    "authenticity_calibrator"
                ),
                max_frames=int(max_frames),
                sample_fps=float(sample_fps),
                device=forensics_device,
            )
        authenticity = forensic_report.get("authenticity") or {}
        scores = forensic_report.get("scores") or {}
        summary = _forensics_conclusion(
            authenticity.get("decision"),
            scores.get("calibrated_real_probability_0_1"),
        )
        result["forensics"] = {
            "status": forensic_report.get("status"),
            "summary": summary,
            "scores": scores,
            "fusion": forensic_report.get("fusion"),
            "branches": {
                "facial_motion": (
                    forensic_report.get("branches", {}).get("facial_motion")
                    or {}
                ).get("metrics"),
                "texture_detail": (
                    forensic_report.get("branches", {}).get("texture_detail")
                    or {}
                ).get("metrics"),
            },
            "authenticity": authenticity,
            "auto_pipeline": forensic_report.get("auto_pipeline"),
        }
    except Exception as exc:  # noqa: BLE001 - retain all sample results
        result["status"] = "forensics_error"
        result["forensics_error"] = f"{type(exc).__name__}: {exc}"

    if include_wangxing:
        if not (
            identity_profile
            and expression_profile
            and identity_profile.is_file()
            and expression_profile.is_file()
        ):
            result["wangxing_error"] = "Wang Xing profiles are unavailable."
            if result["status"] == "ok":
                result["status"] = "wangxing_error"
        else:
            try:
                specialization = evaluate_specialization(
                    video_path=video,
                    au_path=au,
                    identity_profile_path=identity_profile,
                    expression_profile_path=expression_profile,
                    source_profile_path=(
                        source_profile
                        if source_profile and source_profile.is_file()
                        else None
                    ),
                    device=wangxing_device,
                    max_identity_frames=16,
                )
                result["wangxing"] = {
                    "summary": _wangxing_conclusion(specialization),
                    "raw": specialization,
                }
            except Exception as exc:  # noqa: BLE001
                result["wangxing_error"] = f"{type(exc).__name__}: {exc}"
                if result["status"] == "ok":
                    result["status"] = "wangxing_error"
    result["web_card"] = _build_web_card(result)
    result["web_card_markdown"] = _web_card_markdown(result)
    return result


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    decision_counts: Counter[str] = Counter()
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        label = str(result.get("label") or "unknown")
        by_label[label].append(result)
        decision = (
            result.get("forensics", {})
            .get("summary", {})
            .get("decision")
        )
        if decision:
            decision_counts[str(decision)] += 1

    def label_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [
            row
            for row in rows
            if row.get("status") == "ok"
            and row.get("forensics", {}).get("summary")
        ]
        probabilities = [
            _finite(
                row["forensics"]["scores"].get(
                    "calibrated_real_probability_0_1"
                )
            )
            for row in valid
        ]
        probabilities = [value for value in probabilities if value is not None]
        predictions = [
            row["forensics"]["summary"].get("predicted_generated")
            for row in valid
        ]
        target = [
            int(row.get("label_generated"))
            for row in valid
            if row.get("label_generated") is not None
        ]
        return {
            "count": len(rows),
            "valid_forensics": len(valid),
            "mean_calibrated_real_probability": (
                sum(probabilities) / len(probabilities)
                if probabilities
                else None
            ),
            "predicted_generated_count": sum(
                value == 1 for value in predictions
            ),
            "target_generated_count": sum(value == 1 for value in target),
            "correct_count": sum(
                prediction == target_value
                for prediction, target_value in zip(predictions, target)
                if prediction is not None
            ),
        }

    return {
        "sample_count": len(results),
        "status_counts": dict(Counter(str(row.get("status")) for row in results)),
        "forensics_decision_counts": dict(decision_counts),
        "by_label": {
            label: label_metrics(rows)
            for label, rows in sorted(by_label.items())
        },
    }


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "label",
        "label_generated",
        "source_domain",
        "status",
        "forensics_status",
        "forensics_decision",
        "forensics_conclusion",
        "calibrated_real_probability",
        "raw_real_domain_evidence",
        "facial_motion_score",
        "texture_detail_score",
        "wangxing_decision",
        "wangxing_identity_decision",
        "wangxing_expression_decision",
        "wangxing_source_real_probability",
        "video",
        "au",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            forensic = row.get("forensics") or {}
            forensic_summary = forensic.get("summary") or {}
            scores = forensic.get("scores") or {}
            wangxing = row.get("wangxing", {}).get("summary") or {}
            writer.writerow(
                {
                    "sample_id": row.get("sample_id"),
                    "label": row.get("label"),
                    "label_generated": row.get("label_generated"),
                    "source_domain": row.get("source_domain"),
                    "status": row.get("status"),
                    "forensics_status": forensic.get("status"),
                    "forensics_decision": forensic_summary.get("decision"),
                    "forensics_conclusion": forensic_summary.get(
                        "conclusion"
                    ),
                    "calibrated_real_probability": scores.get(
                        "calibrated_real_probability_0_1"
                    ),
                    "raw_real_domain_evidence": scores.get(
                        "raw_real_domain_evidence_0_1"
                    ),
                    "facial_motion_score": scores.get(
                        "facial_expression_muscle_score_0_1"
                    ),
                    "texture_detail_score": scores.get(
                        "texture_detail_score_0_1"
                    ),
                    "wangxing_decision": wangxing.get("decision"),
                    "wangxing_identity_decision": wangxing.get(
                        "identity_decision"
                    ),
                    "wangxing_expression_decision": wangxing.get(
                        "expression_decision"
                    ),
                    "wangxing_source_real_probability": wangxing.get(
                        "source_real_probability"
                    ),
                    "video": row.get("video"),
                    "au": row.get("au"),
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run webpage-equivalent forensics over the single-video test set."
        )
    )
    parser.add_argument(
        "--manifest",
        default="data/test/single_video/manifest.json",
    )
    parser.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/forensics/single_video_forensics_test",
    )
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--sample-fps", type=float, default=8.0)
    parser.add_argument(
        "--forensics-device",
        default="auto",
        help="Device passed to analyze_forensics texture branch.",
    )
    parser.add_argument(
        "--wangxing-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--skip-wangxing",
        action="store_true",
        help="Only run analyze_forensics; skip identity/expression specialization.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-sample JSON files under --output-root.",
    )
    args = parser.parse_args(argv)

    manifest_path = project_path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    manifest_root = manifest_path.parent
    forensics_path = project_path(args.forensics_profile)
    if not forensics_path.is_file():
        raise SystemExit(f"Forensics profile not found: {forensics_path}")
    forensics_profiles = _load_json(forensics_path)

    identity_profile = expression_profile = source_profile = None
    if not args.skip_wangxing:
        identity_profile, expression_profile, source_profile = (
            _resolve_wangxing_profiles()
        )

    output_root = project_path(args.output_root)
    per_sample_root = output_root / "per_sample"
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    samples = list(manifest.get("samples", []))
    for index, sample in enumerate(samples, start=1):
        label = str(sample.get("label") or "unknown")
        sample_path = (
            per_sample_root
            / label
            / f"{sample['sample_id']}.json"
        )
        if args.resume and sample_path.is_file():
            try:
                existing = _load_json(sample_path)
            except (OSError, json.JSONDecodeError):
                existing = {}
            reusable = (
                existing.get("status") == "ok"
                and isinstance(existing.get("forensics"), dict)
                and (
                    args.skip_wangxing
                    or isinstance(existing.get("web_card"), dict)
                )
            )
            if reusable:
                print(
                    f"[{index}/{len(samples)}] "
                    f"{sample.get('sample_id')} RESUME",
                    flush=True,
                )
                results.append(existing)
                continue
        print(
            f"[{index}/{len(samples)}] {sample.get('sample_id')}",
            flush=True,
        )
        result = _run_one(
            sample=sample,
            manifest_root=manifest_root,
            forensics_profiles=forensics_profiles,
            identity_profile=identity_profile,
            expression_profile=expression_profile,
            source_profile=source_profile,
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
            forensics_device=args.forensics_device,
            wangxing_device=args.wangxing_device,
            include_wangxing=not args.skip_wangxing,
        )
        results.append(result)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    aggregate = _aggregate(results)
    metadata = {
        "manifest": str(manifest_path),
        "forensics_profile": str(forensics_path),
        "max_frames": int(args.max_frames),
        "sample_fps": float(args.sample_fps),
        "forensics_device": args.forensics_device,
        "wangxing_enabled": not args.skip_wangxing,
        "wangxing_device": args.wangxing_device,
        "training_allowed": manifest.get("training_allowed", False),
        "official_holdout_overlap": manifest.get(
            "official_holdout_overlap"
        ),
    }
    all_payload = {
        "schema_version": "single_video_web_forensics_results_v1",
        "metadata": metadata,
        "aggregate": aggregate,
        "results": results,
    }
    (output_root / "all_results.json").write_text(
        json.dumps(all_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "single_video_web_forensics_summary_v1",
                "metadata": metadata,
                "aggregate": aggregate,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(output_root / "summary.csv", results)
    card_report = [
        "# 单视频网页专项结果汇总",
        "",
        "该报告按网页王兴专项卡片格式整理每条样本。",
        "",
    ]
    for result in results:
        card_report.append(result.get("web_card_markdown", ""))
        card_report.append("\n---\n")
    (output_root / "web_card_report.md").write_text(
        "\n".join(card_report),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"All results: {output_root / 'all_results.json'}")
    print(f"Summary JSON: {output_root / 'summary.json'}")
    print(f"Summary CSV: {output_root / 'summary.csv'}")
    print(f"Web card report: {output_root / 'web_card_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import inspect
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REDESIGN_CHECK_SCHEMA = "wangxing_specialization_redesign_check_v1"
EXPECTED_CLASSES = {
    "smile",
    "anger",
    "surprise",
    "fear",
    "sadness",
    "disgust",
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _count_files(root: Path, suffixes: Iterable[str]) -> int:
    suffix_set = {suffix.lower() for suffix in suffixes}
    if not root.is_dir():
        return 0
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffix_set
    )


def _check(
    check_id: str,
    status: str,
    details: str,
    *,
    required: bool = True,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": status,
        "required": required,
        "details": details,
    }


def _profile_check(
    checks: list[dict[str, Any]],
    *,
    check_id: str,
    path: Path,
    schema: str,
) -> dict[str, Any] | None:
    payload = _load_json(path)
    if payload is None:
        checks.append(
            _check(
                check_id,
                "missing",
                f"Missing or invalid JSON profile: {path}",
            )
        )
        return None
    actual_schema = payload.get("schema_version")
    if actual_schema != schema:
        checks.append(
            _check(
                check_id,
                "partial",
                f"Expected schema {schema}, found {actual_schema!r}: {path}",
            )
        )
    else:
        checks.append(
            _check(check_id, "complete", f"Profile exists: {path}")
        )
    return payload


def inspect_wangxing_redesign(
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Audit implementation and artifacts against the Wang Xing redesign."""
    root = Path(project_root).resolve()
    checks: list[dict[str, Any]] = []

    real_video_count = _count_files(
        root / "data/MD_CL",
        {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"},
    )
    seedance_video_count = _count_files(
        root / "data/WangXing_Seedance",
        {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"},
    )
    checks.append(
        _check(
            "data.real_video_scale",
            "complete" if real_video_count >= 600 else "partial",
            f"Found {real_video_count} real videos; target is at least 600.",
        )
    )
    checks.append(
        _check(
            "data.seedance_video_scale",
            "complete" if seedance_video_count >= 100 else "partial",
            f"Found {seedance_video_count} Seedance videos; target is at least 100.",
        )
    )

    identity_path = root / "data/au/wangxing_identity_profile.json"
    expression_path = root / "data/au/wangxing_expression_profile.json"
    source_path = root / "data/au/wangxing_source_profile.json"
    identity = _profile_check(
        checks,
        check_id="artifact.identity_profile",
        path=identity_path,
        schema="wangxing_identity_profile_v2",
    )
    expression = _profile_check(
        checks,
        check_id="artifact.expression_profile",
        path=expression_path,
        schema="wangxing_expression_profile_v2",
    )
    source = _profile_check(
        checks,
        check_id="artifact.source_profile",
        path=source_path,
        schema="wangxing_source_profile_v1",
    )
    from evaluator.wangxing_specialization import (
        SPECIALIZATION_EVALUATOR_VERSION,
    )

    profile_versions = {
        name: payload.get("evaluator_version")
        for name, payload in (
            ("identity", identity),
            ("expression", expression),
            ("source", source),
        )
        if payload is not None
    }
    checks.append(
        _check(
            "artifact.profile_version_alignment",
            "complete"
            if profile_versions
            and all(
                version == SPECIALIZATION_EVALUATOR_VERSION
                for version in profile_versions.values()
            )
            else "partial",
            f"expected={SPECIALIZATION_EVALUATOR_VERSION}; "
            f"actual={profile_versions}",
        )
    )

    if identity is not None:
        calibration = identity.get("calibration_metrics", {})
        thresholds = identity.get("thresholds", {})
        required_metrics = {
            "roc_auc",
            "pr_auc",
            "eer",
            "recall_at_1pct_fpr",
            "recall_at_5pct_fpr",
        }
        missing_metrics = sorted(required_metrics - set(calibration))
        required_thresholds = {
            "min_valid_frame_count",
            "min_valid_frame_ratio",
            "min_frame_consistency",
        }
        missing_thresholds = sorted(required_thresholds - set(thresholds))
        status = "complete" if not missing_metrics and not missing_thresholds else "partial"
        checks.append(
            _check(
                "identity.open_set_calibration",
                status,
                f"missing_metrics={missing_metrics}; "
                f"missing_thresholds={missing_thresholds}; "
                f"negative_prototypes={len(identity.get('negative_prototypes', []))}",
            )
        )

    if expression is not None:
        classes = set(expression.get("classes", {}))
        provenance = expression.get("provenance", {})
        real_count = int(provenance.get("real_sample_count", 0))
        pseudo_count = int(provenance.get("pseudo_sample_count", 0))
        status = (
            "complete"
            if EXPECTED_CLASSES <= classes and real_count >= 600
            else "partial"
        )
        checks.append(
            _check(
                "expression.real_support_domain",
                status,
                f"classes={sorted(classes)}; real_samples={real_count}; "
                f"trusted_pseudo_samples={pseudo_count}",
            )
        )
        checks.append(
            _check(
                "expression.open_set_conclusion",
                "complete"
                if "most_compatible_profiles" in inspect.getsource(
                    __import__(
                        "evaluator.wangxing_specialization",
                        fromlist=["score_expression_profile"],
                    ).score_expression_profile
                )
                else "missing",
                "Expression output is based on maximum support-domain "
                "compatibility and exposes the top two profiles.",
            )
        )

    if source is not None:
        counts = source.get("provenance", {}).get("sample_counts", {})
        status = (
            "complete"
            if counts.get("real_wangxing", 0) >= 600
            and counts.get("generated_wangxing", 0) >= 100
            else "partial"
        )
        checks.append(
            _check(
                "source.secondary_domain_evidence",
                status,
                f"sample_counts={counts}; source profile is secondary "
                "evidence and must not gate identity.",
                required=False,
            )
        )

    manifest_path = root / "data/au/WangXing_Seedance/pseudo_expression_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest is None:
        checks.append(
            _check(
                "labels.seedance_manifest",
                "missing",
                f"Missing pseudo-label manifest: {manifest_path}",
            )
        )
    else:
        summary = manifest.get("summary", {})
        statuses = summary.get("status_counts", {})
        checks.append(
            _check(
                "labels.seedance_unknown_rejected",
                "complete"
                if statuses.get("unknown", 0) > 0
                and statuses.get("ambiguous", 0) > 0
                else "partial",
                f"status_counts={statuses}; only high_confidence records "
                "may enter expression training.",
            )
        )

    forensic_path = root / "outputs/forensics/forensics_profiles.json"
    forensic = _load_json(forensic_path)
    if forensic is None:
        checks.append(
            _check(
                "artifact.forensics_profiles",
                "missing",
                f"Missing forensic profile: {forensic_path}",
                required=False,
            )
        )
    else:
        motion = forensic.get("facial_motion", {})
        texture = forensic.get("texture_detail")
        motion_complete = (
            motion.get("real", {}).get("sample_count", 0) >= 600
            and motion.get("seedance", {}).get("sample_count", 0) >= 100
        )
        texture_samples = min(
            texture.get("real", {}).get("sample_count", 0),
            texture.get("seedance", {}).get("sample_count", 0),
        ) if isinstance(texture, dict) else 0
        checks.append(
            _check(
                "forensics.parallel_branches",
                "complete" if motion_complete and texture else "partial",
                f"motion_real={motion.get('real', {}).get('sample_count', 0)}; "
                f"motion_seedance={motion.get('seedance', {}).get('sample_count', 0)}; "
                f"texture_min_samples={texture_samples}",
                required=False,
            )
        )
        checks.append(
            _check(
                "forensics.texture_calibration_size",
                "complete" if texture_samples >= 50 else "partial",
                f"Texture profile has {texture_samples} samples per domain; "
                "at least 50 per domain is recommended before production use.",
                required=False,
            )
        )
        calibrator = forensic.get("authenticity_calibrator")
        calibrator_status = (
            calibrator.get("status")
            if isinstance(calibrator, dict)
            else None
        )
        checks.append(
            _check(
                "forensics.probability_calibrator",
                "complete"
                if calibrator_status in {"ready", "calibrated"}
                else "partial",
                "A ready held-out calibrator is required before source "
                f"probabilities are emitted; status={calibrator_status!r}.",
                required=False,
            )
        )

    authenticity_layer = root / "evaluator/forensics/seedance_authenticity.py"
    authenticity_source = (
        authenticity_layer.read_text(encoding="utf-8-sig")
        if authenticity_layer.is_file()
        else ""
    )
    checks.append(
        _check(
            "forensics.independent_authenticity_layer",
            "complete"
            if authenticity_layer.is_file()
            and "fuse_authenticity_evidence" in authenticity_source
            else "missing",
            "Independent fusion, confidence, and optional calibration layer.",
            required=False,
        )
    )
    manifest_path = root / "data/forensics/forensics_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest is None:
        checks.append(
            _check(
                "forensics.data_manifest",
                "missing",
                f"Missing forensic data manifest: {manifest_path}",
                required=False,
            )
        )
    else:
        summary = manifest.get("summary", {})
        checks.append(
            _check(
                "forensics.data_manifest",
                "complete"
                if summary.get("record_count", 0) >= 800
                else "partial",
                f"records={summary.get('record_count')}; "
                f"au_linked={summary.get('au_linked_count')}; "
                f"generated_metadata_complete="
                f"{summary.get('metadata_complete_generated_count')}",
                required=False,
            )
        )
        checks.append(
            _check(
                "forensics.seedance_metadata",
                "complete"
                if summary.get("metadata_complete_generated_count", 0)
                >= summary.get("generated_count", 0)
                else "partial",
                "Seedance version/mode/input metadata completeness.",
                required=False,
            )
        )

    specialization_source = (
        root / "evaluator/wangxing_specialization.py"
    ).read_text(encoding="utf-8-sig")
    holistic_source = (root / "evaluator/holistic_evaluator.py").read_text(
        encoding="utf-8-sig"
    )
    web_source = (root / "web_app.py").read_text(encoding="utf-8-sig")
    from evaluator.wangxing_specialization import (
        evaluate_specialization,
    )

    specialization_function_source = inspect.getsource(
        evaluate_specialization
    )
    gate_ok = (
        'if identity["decision"] == "wangxing":'
        in specialization_function_source
        and "score_expression_profile(" in specialization_function_source
        and specialization_function_source.index(
            'if identity["decision"] == "wangxing":'
        )
        < specialization_function_source.index("score_expression_profile(")
    )
    checks.append(
        _check(
            "decision.identity_before_expression",
            "complete" if gate_ok else "missing",
            "Expression evaluation is gated by the Wang Xing identity decision.",
        )
    )
    checks.append(
        _check(
            "decision.uncertain_identity_supported",
            "complete"
            if '"decision": "uncertain"' in specialization_source
            and "uncertain_identity" in specialization_source
            else "missing",
            "Low-quality or low-margin identity evidence returns uncertainty.",
        )
    )
    checks.append(
        _check(
            "scope.normal_five_scores_unchanged",
            "complete"
            if '"normal_evaluation_unchanged": True' in specialization_source
            and '"normal_expression_unchanged": True' in web_source
            and all(
                marker in holistic_source
                for marker in (
                    '"identity": 35',
                    '"texture": 15',
                    '"expression": 15',
                    '"temporal": 25',
                    '"aesthetics": 10',
                )
            )
            else "partial",
            "Specialization is reported separately from ordinary category scores.",
        )
    )
    checks.append(
        _check(
            "scope.no_body_action_conclusion",
            "complete"
            if "action_compliance" not in inspect.getsource(
                __import__(
                    "evaluator.wangxing_specialization",
                    fromlist=["evaluate_specialization"],
                ).evaluate_specialization
            )
            else "partial",
            "Specialization conclusion contains identity and facial-expression "
            "evidence, not body-action compliance.",
        )
    )
    checks.append(
        _check(
            "scope.source_profile_not_identity_gate",
            "complete"
            if "not_used_for_identity_gate" in specialization_source
            else "missing",
            "Generated-domain source evidence is explicitly secondary.",
        )
    )
    checks.append(
        _check(
            "forensics.window_evidence",
            "complete"
            if "window_records" in (
                root / "evaluator/forensics/facial_motion.py"
            ).read_text(encoding="utf-8-sig")
            and "window_records" in (
                root / "evaluator/forensics/texture_detail.py"
            ).read_text(encoding="utf-8-sig")
            else "missing",
            "Both branches expose window-level evidence records.",
            required=False,
        )
    )
    checks.append(
        _check(
            "forensics.uncalibrated_decision_guard",
            "complete"
            if (
                "raw_real_domain_evidence_0_1" in authenticity_source
                and 'decision = "uncertain"' in authenticity_source
            )
            else "missing",
            "Uncalibrated profile evidence cannot produce a source decision.",
            required=False,
        )
    )
    checks.append(
        _check(
            "integration.web_forensics_auto_call",
            "complete"
            if "_run_forensics_assessment" in web_source
            and "auto_invoked_by" in web_source
            else "missing",
            "Web Wang Xing flow invokes the same forensic scorer.",
            required=False,
        )
    )

    counts = {
        status: sum(item["status"] == status for item in checks)
        for status in ("complete", "partial", "missing")
    }
    required_failures = [
        item
        for item in checks
        if item["required"] and item["status"] != "complete"
    ]
    if any(item["status"] == "missing" for item in checks):
        overall = "missing"
    elif any(item["status"] == "partial" for item in checks):
        overall = "partial"
    else:
        overall = "complete"
    return {
        "schema_version": REDESIGN_CHECK_SCHEMA,
        "overall_status": overall,
        "project_root": str(root),
        "summary": counts,
        "required_failures": required_failures,
        "checks": checks,
    }

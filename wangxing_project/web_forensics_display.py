"""Map V5.2 cascade outputs onto legacy forensics fields for the unchanged web UI.

The public Wang Xing dashboard reads only two authenticity slots from
``wangxing_au.forensics`` (no UI layout changes):

- ``scores.calibrated_real_probability_0_1`` → top-right "真实拍摄概率"
- ``authenticity.binary_decision`` → "真实性取证" conclusion

Contract:
- conclusion always follows frozen V3 (never Rank / filename)
- probability always follows V5.2 ``score_display`` (V3 + RankHead)
- web inference never infers ranking labels from filenames
- web inference never applies role_anchor
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from evaluator.modules.core.paths import project_path
from wangxing_project.v5_flags import v5_display_cascade_enabled

WEB_V52_RANK_POLICY_CANDIDATES = (
    "outputs/vedio_pred/wangxing_v5_2_results/rank_policy_validated.json",
    "outputs/forensics/wangxing_v5_2_rank_policy.json",
)
WEB_V52_CALIBRATOR = "outputs/forensics/wangxing_v5_realness_calibrator.json"
WEB_V52_V3_MODEL = "outputs/vedio_pred/models/wangxing_v3_res1k.pt"
WEB_V52_DRIVE_MODEL = "outputs/vedio_pred/models/wangxing_v5_drive.json"
WEB_V52_FORENSICS_PROFILE = (
    "outputs/forensics/forensics_profiles_web_v3_test_excluded.json"
)
WEB_V52_SOURCE_PROFILE = (
    "outputs/forensics/wangxing_source_profile_web_v3_test_excluded.json"
)
WEB_V52_CACHE_DIR = "outputs/forensics/cache_wangxing_v5_2_web"
# Neutral AI label for single-upload web inference; never inferred from names.
WEB_V52_NEUTRAL_LABEL = "seedance"


def _first_existing_path(candidates: tuple[str, ...]) -> Path | None:
    for relative in candidates:
        path = project_path(relative)
        if path.is_file():
            return path
    return None


def v52_web_assets_available() -> bool:
    return (
        project_path(WEB_V52_V3_MODEL).is_file()
        and project_path(WEB_V52_CALIBRATOR).is_file()
        and project_path(WEB_V52_FORENSICS_PROFILE).is_file()
        and project_path(WEB_V52_SOURCE_PROFILE).is_file()
    )


@lru_cache(maxsize=1)
def _load_web_v52_context() -> dict[str, Any] | None:
    from wangxing_project.drive_head_v5 import load_drive_head
    from wangxing_project.rank_head_v52 import load_rank_policy_v52
    from wangxing_project.realness_v5 import load_calibrator
    from wangxing_project.v51_runtime import load_json

    if not v52_web_assets_available():
        return None
    calibrator = load_calibrator(project_path(WEB_V52_CALIBRATOR))
    if calibrator is None:
        return None
    rank_policy_path = _first_existing_path(WEB_V52_RANK_POLICY_CANDIDATES)
    rank_policy = (
        load_rank_policy_v52(rank_policy_path)
        if rank_policy_path is not None
        else None
    )
    drive_model = load_drive_head(project_path(WEB_V52_DRIVE_MODEL))
    return {
        "profiles": load_json(WEB_V52_FORENSICS_PROFILE),
        "source_profile": load_json(WEB_V52_SOURCE_PROFILE),
        "calibrator": calibrator,
        "v3_model": project_path(WEB_V52_V3_MODEL),
        "drive_model": drive_model,
        "cache_dir": project_path(WEB_V52_CACHE_DIR),
        "rank_policy": rank_policy,
    }


def _binary_decision_from_v3(decision: str) -> str:
    return "real_capture" if decision == "real" else "seedance_like"


def apply_v52_forensics_display(
    forensics: dict[str, Any],
    v5: dict[str, Any],
) -> dict[str, Any]:
    """Patch legacy forensics payload consumed by ``web/app.js``."""
    decision = str(v5.get("decision") or "generated")
    score_display = float(v5.get("score_display", 0.0))
    binary = _binary_decision_from_v3(decision)
    conclusion = (
        "偏向真实拍摄" if binary == "real_capture" else "偏向 AI 生成"
    )

    scores = forensics.setdefault("scores", {})
    scores["calibrated_real_probability_0_1"] = score_display
    scores["real_capture_likelihood_0_1"] = score_display
    scores["wangxing_v5_score_display_0_1"] = score_display
    scores["wangxing_v5_p_v3_real_0_1"] = v5.get("p_v3_real")

    fusion = forensics.setdefault("fusion", {})
    fusion["real_capture_likelihood_0_1"] = score_display
    # Keep binary_decision authoritative; do not let UI fall back to
    # probability>=0.5 when score_display is high but V3 says AI.
    fusion["decision"] = binary
    fusion["wangxing_v5_display_source"] = "score_display"

    authenticity = forensics.setdefault("authenticity", {})
    authenticity["binary_decision"] = binary
    authenticity["decision"] = binary
    authenticity["binary_conclusion"] = conclusion
    authenticity["calibrated_real_probability_0_1"] = score_display
    authenticity["wangxing_v5_decision"] = decision
    authenticity["wangxing_v5_score_display_0_1"] = score_display
    authenticity["wangxing_v5_display_note"] = (
        "顶部数值为 V5.2 质量/档位展示分（score_display）；"
        "真实性取证结论仍跟冻结 V3 真伪决策，不跟分数阈值。"
    )

    forensics["wangxing_v5_display"] = {
        "schema_version": "wangxing_v5_2_web_forensics_display_v1",
        "score_display": score_display,
        "decision": decision,
        "p_v3_real": v5.get("p_v3_real"),
        "score_band": v5.get("score_band"),
        "band_hint": v5.get("band_hint"),
        "rank_reason": v5.get("rank_reason"),
        "display_blend_mode": v5.get("display_blend_mode"),
        "prior_conflict": v5.get("prior_conflict"),
        "filename_label_inference": False,
        "role_anchor_applied": False,
    }
    return forensics


def infer_v52_for_web(
    *,
    video_path: str | Path,
    au_path: str | Path,
    device: str,
    wangxing_device: str | None = None,
) -> dict[str, Any] | None:
    from wangxing_project.cascade_v5 import cascade_score_v52
    from wangxing_project.rank_head_v52 import predict_rank_score
    from wangxing_project.v51_runtime import build_feature_row

    context = _load_web_v52_context()
    if context is None:
        return None
    wangxing_device = wangxing_device or device
    row = build_feature_row(
        video=Path(video_path),
        label=WEB_V52_NEUTRAL_LABEL,
        group="web_single",
        au_path=Path(au_path),
        v3_model=context["v3_model"],
        drive_model=context["drive_model"],
        drive_cache=context["cache_dir"],
        source_profile=context["source_profile"],
        forensics_profile=context["profiles"],
        device=device,
        wangxing_device=wangxing_device,
        calibrator=context["calibrator"],
        realness_enabled=True,
        rank_policy=context["rank_policy"],
    )
    row["label"] = WEB_V52_NEUTRAL_LABEL
    # Web uploads have no ground-truth label; never mark prior_conflict.
    row["prior_conflict"] = False
    rank_policy = context["rank_policy"]
    rank_score = None
    if rank_policy is not None:
        rank_score, _rank_status = predict_rank_score(row, rank_policy)
    v5 = cascade_score_v52(
        p_v3_real=float(row["v5"]["p_v3_real"]),
        p_drive=row["v5"].get("p_drive"),
        p_drive_eff=row["v5"].get("p_drive_eff"),
        realness=row.get("realness"),
        rank_score=rank_score,
        rank_policy=rank_policy,
        realness_enabled=True,
        rank_enabled=bool(rank_policy and rank_score is not None),
        prior_conflict=False,
        group_id="web_single",
    )
    return v5


def should_apply_v52_web_forensics_display() -> bool:
    """Apply V5.2 by default when assets exist; legacy requires opt-out."""
    return bool(
        v5_display_cascade_enabled()
        and v52_web_assets_available()
    )


def patch_wangxing_au_forensics_for_v52(
    payload: dict[str, Any],
    *,
    video_path: str | Path,
    au_path: str | Path,
    device: str,
    wangxing_device: str | None = None,
) -> dict[str, Any]:
    """Run V5.2 and patch ``payload['forensics']`` for the legacy web UI."""
    if str(payload.get("status") or "") != "available":
        return payload
    if not should_apply_v52_web_forensics_display():
        return payload
    forensics = payload.get("forensics")
    if not isinstance(forensics, dict):
        return payload
    try:
        v5 = infer_v52_for_web(
            video_path=video_path,
            au_path=au_path,
            device=device,
            wangxing_device=wangxing_device,
        )
    except Exception as exc:
        payload["wangxing_v5_display_error"] = {
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        return payload
    if not isinstance(v5, dict):
        return payload
    apply_v52_forensics_display(forensics, v5)
    payload["forensics"] = forensics
    payload["wangxing_v5"] = {
        "schema_version": "wangxing_v5_result_v1",
        "status": "available",
        **v5,
    }
    return payload


def resync_result_forensics_from_v5(result: dict[str, Any]) -> None:
    """Re-apply display mapping after other authenticity writers run."""
    wangxing = result.get("wangxing_au")
    if not isinstance(wangxing, dict):
        return
    v5 = wangxing.get("wangxing_v5")
    forensics = wangxing.get("forensics")
    if not isinstance(v5, dict) or not isinstance(forensics, dict):
        return
    if str(v5.get("status") or "") == "unavailable":
        return
    apply_v52_forensics_display(forensics, v5)
    wangxing["forensics"] = forensics
    result["wangxing_au"] = wangxing
    result["wangxing_v5"] = v5


def clear_web_v52_context_cache() -> None:
    _load_web_v52_context.cache_clear()

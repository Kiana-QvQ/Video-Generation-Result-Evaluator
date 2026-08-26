"""Runtime feature flags for Wang Xing V5.

Production defaults keep the public web UI on the legacy V3/specialization
path. Offline PT/Web V5 scripts remain independent.
"""

from __future__ import annotations

import os
from typing import Any


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return bool(default)
    value = str(raw).strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def v5_drive_enabled() -> bool:
    """Allow DriveHead evidence to be attached to live job results."""
    return _env_flag("V5_DRIVE_ENABLED", default=False)


def v5_rank_enabled() -> bool:
    """Allow RankHead fine bands when ordering_satisfied is also true."""
    return _env_flag("V5_RANK_ENABLED", default=False)


def v5_display_cascade_enabled() -> bool:
    """Allow V5.2 score_display only when explicitly enabled."""
    return _env_flag("V5_DISPLAY_CASCADE", default=False)


def v5_realness_enabled() -> bool:
    """Allow the offline/publicly guarded V5.1 quality-axis mapping."""
    return _env_flag("V5_REALNESS_ENABLED", default=False)


def v5_runtime_flags() -> dict[str, Any]:
    return {
        "schema_version": "wangxing_v5_runtime_flags_v1",
        "V5_DRIVE_ENABLED": v5_drive_enabled(),
        "V5_RANK_ENABLED": v5_rank_enabled(),
        "V5_DISPLAY_CASCADE": v5_display_cascade_enabled(),
        "V5_REALNESS_ENABLED": v5_realness_enabled(),
        "production_default": "legacy_wangxing_au",
        "note": (
            "Public web keeps the legacy Wang Xing specialization unless "
            "V5_DISPLAY_CASCADE=true is explicitly set."
        ),
    }

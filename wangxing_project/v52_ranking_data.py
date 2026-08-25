"""Helpers for V5.2 ranking manifests and pool-based group completion.

LTX groups currently ship LoRA + multiref only.  To enable linear RankHead
training (complete train groups >= 4), missing real / seedance slots can be
filled from existing project pools that are outside final binary holdouts.

Filled pairs are marked ``pool_fill_dev``: they are same-identity / same-domain
proxies, not guaranteed same-prompt matched quadruples.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

ORDER = ("real", "lora", "seedance", "multiref")
FINAL_TEST_MANIFESTS = (
    "data/test/single_video/manifest.json",
    "data/test/wangxing_32x32/single_video/manifest.json",
)
DEFAULT_REAL_POOLS = (
    "data/MD_CL",
    "data/video",
)
DEFAULT_SEEDANCE_POOLS = (
    "data/WangXing_Seedance",
)


def is_complete(videos: dict[str, Any] | None) -> bool:
    payload = videos or {}
    return all(payload.get(role) for role in ORDER)


def load_forbidden_basenames(project_root: Path) -> set[str]:
    """Basenames that appear in final 25+25 / 32+32 source videos."""
    forbidden: set[str] = set()
    for relative in FINAL_TEST_MANIFESTS:
        path = project_root / relative
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for sample in payload.get("samples") or []:
            for key in ("source_video", "video"):
                value = sample.get(key)
                if value:
                    forbidden.add(Path(str(value)).name.casefold())
    return forbidden


def _iter_mp4(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.rglob("*.mp4")
        if path.is_file() and "(1)" not in path.name
    )


def collect_pool_videos(
    *,
    project_root: Path,
    pool_roots: Iterable[str | Path],
    forbidden_basenames: set[str],
) -> list[Path]:
    videos: list[Path] = []
    seen: set[str] = set()
    for root_value in pool_roots:
        root = Path(root_value)
        if not root.is_absolute():
            root = project_root / root
        root = root.expanduser().resolve()
        for path in _iter_mp4(root):
            key = str(path.resolve())
            if key in seen:
                continue
            if path.name.casefold() in forbidden_basenames:
                continue
            seen.add(key)
            videos.append(path.resolve())
    return videos


def _stable_pick(
    candidates: list[Path],
    *,
    key: str,
    used: set[str],
) -> Path | None:
    if not candidates:
        return None
    available = [path for path in candidates if str(path) not in used]
    if not available:
        return None
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(available)
    chosen = available[index]
    used.add(str(chosen))
    return chosen


def complete_partial_groups(
    groups: list[dict[str, Any]],
    *,
    project_root: Path,
    real_pools: Iterable[str | Path] = DEFAULT_REAL_POOLS,
    seedance_pools: Iterable[str | Path] = DEFAULT_SEEDANCE_POOLS,
    min_complete_train: int = 5,
) -> dict[str, Any]:
    """Fill missing real/seedance for partial train groups from local pools."""
    forbidden = load_forbidden_basenames(project_root)
    real_pool = collect_pool_videos(
        project_root=project_root,
        pool_roots=real_pools,
        forbidden_basenames=forbidden,
    )
    seedance_pool = collect_pool_videos(
        project_root=project_root,
        pool_roots=seedance_pools,
        forbidden_basenames=forbidden,
    )
    used: set[str] = set()
    for group in groups:
        videos = group.get("videos") or {}
        for role in ORDER:
            path = videos.get(role)
            if path:
                used.add(str(Path(str(path)).expanduser().resolve()))

    completions: list[dict[str, Any]] = []
    for group in groups:
        if str(group.get("split")) == "holdout":
            # Never mutate the holdout quadruple.
            continue
        videos = dict(group.get("videos") or {})
        if is_complete(videos):
            group["completeness"] = "full"
            group.setdefault("completion_mode", "native_full")
            continue
        filled: dict[str, str] = {}
        if not videos.get("real"):
            picked = _stable_pick(
                real_pool,
                key=f"{group.get('group_id')}:real",
                used=used,
            )
            if picked is None:
                raise RuntimeError(
                    "Real pool exhausted while completing ranking groups."
                )
            videos["real"] = str(picked)
            filled["real"] = str(picked)
        if not videos.get("seedance"):
            picked = _stable_pick(
                seedance_pool,
                key=f"{group.get('group_id')}:seedance",
                used=used,
            )
            if picked is None:
                raise RuntimeError(
                    "Seedance pool exhausted while completing ranking groups."
                )
            videos["seedance"] = str(picked)
            filled["seedance"] = str(picked)
        group["videos"] = {
            role: videos.get(role) for role in ORDER
        }
        if is_complete(group["videos"]):
            group["completeness"] = "full"
            group["completion_mode"] = (
                "pool_fill_dev" if filled else "native_full"
            )
            group["same_prompt_matched"] = False if filled else True
            group["filled_roles"] = filled
            completions.append(
                {
                    "group_id": group.get("group_id"),
                    "filled_roles": filled,
                    "completion_mode": group["completion_mode"],
                }
            )
        else:
            group["completeness"] = "partial"
            group["completion_mode"] = "incomplete"
            group["same_prompt_matched"] = False

    train_complete = sum(
        1
        for group in groups
        if str(group.get("split")) == "train"
        and group.get("completeness") == "full"
    )
    if train_complete < int(min_complete_train):
        raise RuntimeError(
            "After pool completion, train complete groups = "
            f"{train_complete} < required {min_complete_train}. "
            "Add more real/seedance pool videos or lower --min-complete-train."
        )
    return {
        "forbidden_basename_count": len(forbidden),
        "real_pool_size": len(real_pool),
        "seedance_pool_size": len(seedance_pool),
        "completions": completions,
        "train_complete_groups": train_complete,
        "min_complete_train": int(min_complete_train),
        "note": (
            "pool_fill_dev uses unpaired real/seedance proxies from "
            "MD_CL/video + WangXing_Seedance; not same-prompt matched."
        ),
    }

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = "wangxing_expression_reference_v1"
MANIFEST_NAME = "slice_manifest.json"
LABELS_NAME = "arkit52_chinese_motion_labels.json"
PATH_MARKER = "wangxing_arkit52_front_89f"


EXPRESSION_TAXONOMY: dict[str, dict[str, Any]] = {
    "smile": {
        "display_name_zh": "笑",
        "description_zh": "喜悦、开心",
        "valence": "positive",
        "arousal": "medium_high",
        "source_performances": ["Xiao"],
        "is_emotion": True,
    },
    "anger": {
        "display_name_zh": "愤怒",
        "description_zh": "愤怒与生气反应",
        "valence": "negative",
        "arousal": "high",
        "source_performances": ["FenNu", "ShengQi"],
        "is_emotion": True,
    },
    "surprise": {
        "display_name_zh": "惊讶",
        "description_zh": "突发、外显的惊讶反应",
        "valence": "mixed",
        "arousal": "high",
        "source_performances": ["JingYa"],
        "is_emotion": True,
    },
    "fear": {
        "display_name_zh": "恐惧",
        "description_zh": "紧张、回避和警觉",
        "valence": "negative",
        "arousal": "high",
        "source_performances": ["KongJu"],
        "is_emotion": True,
    },
    "annoyance": {
        "display_name_zh": "生气",
        "description_zh": "闷气、不悦、压抑型生气",
        "valence": "negative",
        "arousal": "medium",
        "source_performances": [],
        "is_emotion": True,
    },
    "sadness": {
        "display_name_zh": "悲伤",
        "description_zh": "悲伤、低落",
        "valence": "negative",
        "arousal": "low_medium",
        "source_performances": ["BeiShang", "BeiShang2"],
        "is_emotion": True,
    },
}


SUPPORT_TAXONOMY: dict[str, dict[str, Any]] = {
    "neutral": {
        "display_name_zh": "中性",
        "description_zh": "无明显情绪变化的目标人物基准",
        "source_performances": ["Neutral"],
        "is_emotion": False,
    },
    "speech": {
        "display_name_zh": "说话",
        "description_zh": "日常说话时的王兴面部动态",
        "source_performances": [
            "XinWenGao1",
            "XinWenGao2",
            "YingWenZiMu",
            "YanWu",
        ],
        "is_emotion": False,
    },
    "articulation": {
        "display_name_zh": "发音与口型",
        "description_zh": "嘴部、唇形和发音相关动态",
        "source_performances": [
            "FaYin1",
            "FaYin2",
            "FuYin",
            "GouYi",
            "GouYi2",
            "RaoKouLing1",
            "RaoKouLing2",
            "YuanYin",
        ],
        "is_emotion": False,
    },
    "facial_action": {
        "display_name_zh": "FACS 面部动作",
        "description_zh": "细粒度眉眼嘴面部动作，不直接等同于情绪",
        "source_performances": ["FACS0", "FACS1", "FACS2"],
        "is_emotion": False,
    },
}


ALL_TAXONOMY = {**EXPRESSION_TAXONOMY, **SUPPORT_TAXONOMY}
PERFORMANCE_TO_CLASS = {
    performance.casefold(): class_name
    for class_name, definition in ALL_TAXONOMY.items()
    for performance in definition["source_performances"]
}


def classify_performance(performance: str) -> str:
    """Map a source directory name to the canonical expression class."""
    normalized = str(performance).strip().casefold()
    return PERFORMANCE_TO_CLASS.get(normalized, "unclassified")


def _relative_clip_path(clip_path: str) -> str:
    normalized = str(clip_path).replace("\\", "/")
    for marker in (f"{PATH_MARKER}/", "data/video/"):
        if marker in normalized:
            return normalized.split(marker, 1)[1]
    return Path(normalized).name


def _local_clip_path(root: Path, relative_path: str) -> Path:
    return root / Path(relative_path.replace("/", "\\"))


def _clip_index_from_name(path: Path) -> int | None:
    match = re.search(r"clip(\d+)", path.stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def build_expression_manifest(root: str | Path) -> dict[str, Any]:
    """Build a target-specific reference manifest from the existing slices."""
    root = Path(root)
    manifest = _load_json(root / MANIFEST_NAME)
    labels_path = root / LABELS_NAME
    labels = _load_json(labels_path) if labels_path.exists() else []
    labels_by_path = {
        str(item.get("relative_path", "")).replace("\\", "/").casefold(): item
        for item in labels
    }

    records: list[dict[str, Any]] = []
    for item in manifest:
        relative_path = _relative_clip_path(item["clip_path"])
        local_path = _local_clip_path(root, relative_path)
        performance = str(item["performance"])
        class_name = classify_performance(performance)
        label = labels_by_path.get(relative_path.casefold(), {})
        definition = ALL_TAXONOMY.get(class_name)
        records.append(
            {
                "clip_id": (
                    f"{performance}/{Path(relative_path).stem}"
                ),
                "person": item.get("person", "wangxing"),
                "relative_path": relative_path,
                "local_path": relative_path,
                "exists": local_path.is_file(),
                "phase1_usable": (
                    local_path.is_file()
                    and class_name in ALL_TAXONOMY
                ),
                "performance": performance,
                "expression_class": class_name,
                "display_name_zh": (
                    definition.get("display_name_zh")
                    if definition
                    else "未分类"
                ),
                "is_emotion": (
                    bool(definition.get("is_emotion"))
                    if definition
                    else False
                ),
                "clip_index": item.get("clip_index"),
                "start_frame": item.get("start_frame"),
                "end_frame_exclusive": item.get("end_frame_exclusive"),
                "clip_len": item.get("clip_len"),
                "source_fps": item.get("source_fps"),
                "output_fps": item.get("output_fps"),
                "width": item.get("width"),
                "height": item.get("height"),
                "source_performance": performance,
                "metadata_source": "slice_manifest",
                "label_status": (
                    "manifest_label" if label else "taxonomy_only"
                ),
                "label_task": label.get("label_task"),
                "basic_emotion": label.get("basic_emotion"),
                "expression_change": label.get("expression_change"),
                "brow_change": label.get("brow_change"),
                "eye_change": label.get("eye_change"),
                "head_change": label.get("head_change"),
                "controller_prompt": label.get("controller_prompt"),
            }
        )

    known_paths = {
        str(record["relative_path"]).replace("\\", "/").casefold()
        for record in records
    }
    for local_path in sorted(root.rglob("*.mp4")):
        relative_path = local_path.relative_to(root).as_posix()
        if relative_path.casefold() in known_paths:
            continue
        performance = (
            relative_path.split("/", 1)[0]
            if "/" in relative_path
            else local_path.parent.name
        )
        class_name = classify_performance(performance)
        definition = ALL_TAXONOMY.get(class_name)
        records.append(
            {
                "clip_id": f"{performance}/{local_path.stem}",
                "person": "wangxing",
                "relative_path": relative_path,
                "local_path": relative_path,
                "exists": True,
                "phase1_usable": class_name in ALL_TAXONOMY,
                "performance": performance,
                "expression_class": class_name,
                "display_name_zh": (
                    definition.get("display_name_zh")
                    if definition
                    else "未分类"
                ),
                "is_emotion": (
                    bool(definition.get("is_emotion"))
                    if definition
                    else False
                ),
                "clip_index": _clip_index_from_name(local_path),
                "start_frame": None,
                "end_frame_exclusive": None,
                "clip_len": None,
                "source_fps": None,
                "output_fps": None,
                "width": None,
                "height": None,
                "source_performance": performance,
                "metadata_source": "filesystem",
                "label_status": "taxonomy_only",
                "label_task": None,
                "basic_emotion": None,
                "expression_change": None,
                "brow_change": None,
                "eye_change": None,
                "head_change": None,
                "controller_prompt": None,
            }
        )

    manifest_records = records[: len(manifest)]
    emotion_records = [
        record for record in records
        if record["phase1_usable"] and record["is_emotion"]
    ]
    usable_records = [
        record for record in records if record["phase1_usable"]
    ]

    def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in rows:
            key = str(row["expression_class"])
            result[key] = result.get(key, 0) + 1
        return dict(sorted(result.items()))

    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "王兴目标人物表情参考库：先建立身份与表情基准，"
            "再用于生成结果的表情神似度评价。"
        ),
        "person": "wangxing",
        "taxonomy": ALL_TAXONOMY,
        "source": {
            "manifest": MANIFEST_NAME,
            "labels": LABELS_NAME if labels_path.exists() else None,
            "manifest_rows": len(manifest_records),
            "label_rows": len(labels),
            "filesystem_rows": len(
                [
                    record
                    for record in records
                    if record["metadata_source"] == "filesystem"
                ]
            ),
            "actual_video_rows": len(
                [record for record in records if record["exists"]]
            ),
            "usable_rows": len(usable_records),
            "emotion_rows": len(emotion_records),
            "missing_video_rows": sum(
                not bool(record["exists"]) for record in manifest_records
            ),
        },
        "counts": {
            "manifest": counts(manifest_records),
            "all_records": counts(records),
            "usable": counts(usable_records),
            "emotion": counts(emotion_records),
        },
        "records": records,
    }


def validate_expression_manifest(
    payload: dict[str, Any],
) -> list[str]:
    """Return validation errors without changing any dataset files."""
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("Unexpected expression manifest schema version.")

    records = payload.get("records")
    if not isinstance(records, list) or not records:
        errors.append("Expression manifest contains no records.")
        return errors

    for record in records:
        class_name = record.get("expression_class")
        if class_name not in ALL_TAXONOMY:
            errors.append(
                f"Unknown expression class: {class_name!r} "
                f"for {record.get('relative_path')!r}"
            )
        if record.get("phase1_usable") and not record.get("exists"):
            errors.append(
                f"Usable record does not exist: {record.get('relative_path')}"
            )

    return errors

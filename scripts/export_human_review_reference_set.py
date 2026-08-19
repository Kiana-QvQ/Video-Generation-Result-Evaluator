"""Export original reference-content experiment batches.

The source is the normalized Confluence archive, not the later
``performance_v8`` voting pairs. The export is an experiment archive:
prompt, reference inputs and generated videos are kept when present.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "human_review" / "data" / "raw_archive" / "experiments_20260811"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "test" / "with_reference"
DEFAULT_DESKTOP = Path(r"C:\Users\zhanghaotian\Desktop\test_video\with_reference")
CONFLUENCE_SOURCE = (
    "http://confluence.digisky.com/pages/viewpage.action?pageId=175507358"
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac"}

README_TEXT = f"""# 带参考生成实验

这里保存原始参考内容生成实验的可读导出，不是 `performance_v8` 的人工投票题库。
源归档共 18 组，其中 14 组有提示词并被导出；另外 4 组没有提示词，因此只保留在
`human_review/data/raw_archive/experiments_20260811`，没有复制到这里。

每组实验中，提示词、参考输入和生成结果有什么就保留什么，不要求每一种参考媒介都存在。

数据来源：{CONFLUENCE_SOURCE}

```text
.
├── experiments/
│   └── exp_001__helmet_identity_views/
│       ├── prompt.txt
│       ├── reference_inputs/
│       │   ├── images/
│       │   ├── audio/
│       │   └── videos/
│       ├── generated_videos/
│       └── experiment.json
├── manifest.json
└── README.md
```

命名约定：

- `reference_inputs/` 表示生成时使用的参考输入，不再使用容易和评审题混淆的 `references/`。
- `generated_videos/` 表示模型生成结果，不再使用暗示二选一的 `candidates/`。
- 实验目录使用 `exp_###__slug`；前半段是稳定 ID，后半段只用于人工浏览。
- 生成结果使用 `run_XX__model_id.mp4`，不把单个结果称为 `candidate`。
- `experiment.json` 记录稳定实验 ID、短名称、相对路径、媒介类型、模型信息和源归档位置。
"""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size == source.stat().st_size:
            return
        destination.unlink()
    try:
        destination.hardlink_to(source)
    except (OSError, NotImplementedError):
        shutil.copy2(source, destination)


def _media_kind(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def _list_media(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    files = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and _media_kind(path)
    ]
    return files


def _experiment_identity(batch_name: str) -> tuple[str, str, str]:
    match = re.match(r"^(exp_\d{3})_(?P<slug>.+)$", batch_name)
    if not match:
        return batch_name, batch_name, batch_name
    experiment_id = match.group(1)
    slug = match.group("slug")
    return experiment_id, slug, f"{experiment_id}__{slug}"


def _reference_role(source: Path) -> str:
    stem = source.stem.lower()
    if stem.startswith("reference_identity_"):
        return "identity"
    if stem.startswith("reference_appearance_"):
        return "appearance"
    if stem.startswith("reference_motion_"):
        return "motion"
    if stem.startswith("reference_audio_"):
        return "audio"
    if stem.startswith("reference_image_"):
        return "image"
    return "reference"


def _reference_destination(source: Path) -> tuple[str, str]:
    media_type = _media_kind(source)
    if media_type == "image":
        folder = "images"
    elif media_type == "audio":
        folder = "audio"
    elif media_type == "video":
        folder = "videos"
    else:
        folder = "other"

    prefix_by_role = {
        "identity": "reference_identity_",
        "appearance": "reference_appearance_",
        "motion": "reference_motion_",
        "audio": "reference_audio_",
        "image": "reference_image_",
    }
    stem = source.stem
    role = _reference_role(source)
    prefix = prefix_by_role.get(role)
    if prefix and stem.startswith(prefix):
        stem = f"{role}_{stem.removeprefix(prefix)}"
    elif stem == source.stem:
        stem = f"{role}_{stem}"
    return folder, f"{stem}{source.suffix.lower()}"


def _generated_destination(source: Path) -> str:
    match = re.match(r"^candidate_(?P<index>\d+)_(?P<model>.+)$", source.stem)
    if match:
        return (
            f"run_{match.group('index')}__{match.group('model')}"
            f"{source.suffix.lower()}"
        )
    return f"run__{source.stem}{source.suffix.lower()}"


def _generated_model_id(source: Path) -> str | None:
    match = re.match(r"candidate_\d+_(?P<model>.+)$", source.stem)
    return match.group("model") if match else None


def _read_prompt(batch: Path) -> str:
    prompt_path = batch / "prompt.txt"
    if prompt_path.is_file():
        return prompt_path.read_text(encoding="utf-8-sig").strip()
    txt_files = sorted(batch.glob("*.txt"))
    if not txt_files:
        return ""
    return txt_files[0].read_text(encoding="utf-8-sig").strip()


def _export_batch(batch: Path, output_root: Path) -> dict[str, Any] | None:
    prompt = _read_prompt(batch)
    references = _list_media(batch / "references")
    candidates = _list_media(batch / "candidates")
    if not prompt:
        return None

    experiment_id, experiment_slug, directory_name = _experiment_identity(batch.name)
    dest = output_root / "experiments" / directory_name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    prompt_file = None
    if prompt:
        prompt_file = "prompt.txt"
        (dest / prompt_file).write_text(prompt + "\n", encoding="utf-8")

    reference_items: list[dict[str, str]] = []
    for source in references:
        folder, filename = _reference_destination(source)
        destination = dest / "reference_inputs" / folder / filename
        _copy_or_link(source, destination)
        reference_items.append(
            {
                "path": destination.relative_to(dest).as_posix(),
                "type": _media_kind(source) or "file",
                "role": _reference_role(source),
                "source_name": source.name,
            }
        )

    generated_items: list[dict[str, str]] = []
    for source in candidates:
        destination = dest / "generated_videos" / _generated_destination(source)
        _copy_or_link(source, destination)
        generated_items.append(
            {
                "path": destination.relative_to(dest).as_posix(),
                "type": "video",
                "model_id": _generated_model_id(source),
                "source_name": source.name,
            }
        )

    payload = {
        "schema_version": "with_reference_experiment_v5",
        "experiment_id": experiment_id,
        "experiment_slug": experiment_slug,
        "directory_name": directory_name,
        "source_batch_name": batch.name,
        "source_batch": str(batch),
        "source_page": CONFLUENCE_SOURCE,
        "prompt": prompt,
        "prompt_file": prompt_file,
        "reference_inputs": reference_items,
        "generated_videos": generated_items,
        "content_flags": {
            "has_prompt": bool(prompt),
            "has_image_reference": any(
                item["type"] == "image" for item in reference_items
            ),
            "has_audio_reference": any(
                item["type"] == "audio" for item in reference_items
            ),
            "has_video_reference": any(
                item["type"] == "video" for item in reference_items
            ),
            "reference_count": len(reference_items),
            "generated_video_count": len(generated_items),
        },
        "training_allowed": False,
    }
    _write_json(dest / "experiment.json", payload)
    return payload


def _sync_desktop(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export original Confluence reference-content batches."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--desktop-output", default=str(DEFAULT_DESKTOP))
    args = parser.parse_args(argv)

    source_root = Path(args.source).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    if not source_root.is_dir():
        raise SystemExit(f"Invalid experiment archive: {source_root}")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    experiments: list[dict[str, Any]] = []
    skipped_batches: list[dict[str, str]] = []
    for batch in sorted(path for path in source_root.iterdir() if path.is_dir()):
        payload = _export_batch(batch, output_root)
        if payload:
            experiments.append(payload)
        else:
            experiment_id, experiment_slug, directory_name = _experiment_identity(
                batch.name
            )
            skipped_batches.append(
                {
                    "experiment_id": experiment_id,
                    "experiment_slug": experiment_slug,
                    "directory_name": directory_name,
                    "source_batch_name": batch.name,
                    "source_batch": str(batch),
                    "reason": "missing_prompt",
                }
            )

    prompt_count = sum(
        1 for item in experiments if item["content_flags"]["has_prompt"]
    )
    reference_count = sum(
        1
        for item in experiments
        if item["content_flags"]["reference_count"] > 0
    )
    manifest = {
        "schema_version": "with_reference_experiment_export_v5",
        "source_page": CONFLUENCE_SOURCE,
        "source_dataset": str(source_root),
        "source_experiment_count": len(experiments) + len(skipped_batches),
        "experiment_count": len(experiments),
        "with_prompt_count": prompt_count,
        "with_reference_input_count": reference_count,
        "skipped_batches": skipped_batches,
        "training_allowed": False,
        "experiments": experiments,
    }
    _write_json(output_root / "manifest.json", manifest)
    (output_root / "README.md").write_text(README_TEXT, encoding="utf-8")

    desktop_output = str(args.desktop_output or "").strip()
    desktop_path = None
    if desktop_output:
        desktop_path = Path(desktop_output).expanduser().resolve()
        _sync_desktop(output_root, desktop_path)

    print(
        json.dumps(
            {
                "output": str(output_root),
                "desktop_output": str(desktop_path) if desktop_path else None,
                "experiments": len(experiments),
                "with_prompt": prompt_count,
                "with_reference_inputs": reference_count,
                "skipped_without_prompt": len(skipped_batches),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

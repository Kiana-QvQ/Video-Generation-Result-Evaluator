from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE_DIR = PROJECT_ROOT / "model_cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


PYIQA_CHECKPOINTS = {
    "maniqa-pipal": {
        "filename": "MANIQA_PIPAL-ae6d356b.pth",
        "size": 543_335_435,
    },
    "musiq": {
        "filename": "musiq_koniq_ckpt-e95806b9.pth",
        "size": 108_610_983,
    },
}


def configure_project_environment() -> dict[str, str]:
    """Keep model, framework, and temporary caches inside this project."""
    cache_dirs = {
        "TORCH_HOME": MODEL_CACHE_DIR,
        "INSIGHTFACE_HOME": MODEL_CACHE_DIR / "insightface",
        "PYIQA_HOME": MODEL_CACHE_DIR / "pyiqa",
        "HF_HOME": MODEL_CACHE_DIR / "huggingface",
        "HF_HUB_CACHE": MODEL_CACHE_DIR / "huggingface" / "hub",
        "HF_DATASETS_CACHE": MODEL_CACHE_DIR / "huggingface" / "datasets",
        "TRANSFORMERS_CACHE": MODEL_CACHE_DIR / "huggingface" / "transformers",
        "XDG_CACHE_HOME": MODEL_CACHE_DIR / "xdg",
        "VBENCH_CACHE_DIR": MODEL_CACHE_DIR / "vbench",
        "TORCH_EXTENSIONS_DIR": MODEL_CACHE_DIR / "torch_extensions",
        "MPLCONFIGDIR": MODEL_CACHE_DIR / "matplotlib",
        "PIP_CACHE_DIR": MODEL_CACHE_DIR / "pip",
        "DOCKER_CONFIG": PROJECT_ROOT / ".docker",
    }
    for key, value in cache_dirs.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    return {key: str(value) for key, value in cache_dirs.items()}


configure_project_environment()


def _is_complete_checkpoint(path: Path, expected_size: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size == expected_size
    except OSError:
        return False


def _remove_pyiqa_partials(filename: str) -> None:
    """Remove abandoned torch.hub download fragments after a full file exists."""
    for directory in (
        MODEL_CACHE_DIR / "hub" / "pyiqa",
        MODEL_CACHE_DIR / "hub" / "checkpoints",
    ):
        for partial in directory.glob(f"{filename}.*.partial"):
            try:
                partial.unlink()
            except OSError:
                pass


def prepare_pyiqa_checkpoint(metric_name: str) -> Path | None:
    """Bridge the project IQA cache into torch.hub without triggering downloads."""
    spec = PYIQA_CHECKPOINTS.get(metric_name)
    if spec is None:
        return None

    filename = str(spec["filename"])
    expected_size = int(spec["size"])
    source = MODEL_CACHE_DIR / "hub" / "pyiqa" / filename
    target = MODEL_CACHE_DIR / "hub" / "checkpoints" / filename

    if _is_complete_checkpoint(target, expected_size):
        _remove_pyiqa_partials(filename)
        return target
    if not _is_complete_checkpoint(source, expected_size):
        return None

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.bridge")
    try:
        shutil.copyfile(source, temporary)
        if not _is_complete_checkpoint(temporary, expected_size):
            raise OSError(f"Incomplete IQA checkpoint bridge: {source}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)

    if _is_complete_checkpoint(target, expected_size):
        _remove_pyiqa_partials(filename)
        return target
    return None

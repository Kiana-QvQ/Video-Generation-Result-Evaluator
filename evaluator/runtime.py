from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = PROJECT_ROOT / "model_cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


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
        "GRADIO_TEMP_DIR": OUTPUT_DIR / "gradio_temp",
        "DOCKER_CONFIG": PROJECT_ROOT / ".docker",
    }
    for key, value in cache_dirs.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    return {key: str(value) for key, value in cache_dirs.items()}


configure_project_environment()

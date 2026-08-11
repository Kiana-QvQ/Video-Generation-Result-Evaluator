from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from importlib import metadata

from evaluator.modules.core.runtime import MODEL_CACHE_DIR, PROJECT_ROOT


VBENCH_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]
VBENCH_DOCKER_IMAGE = os.environ.get(
    "VBENCH_DOCKER_IMAGE",
    "video-evaluator-vbench:cu118",
)
DINO_SOURCE_ROOT = (
    MODEL_CACHE_DIR / "vbench" / "dino_model" / "facebookresearch_dino_main"
)
DINO_COMPAT_ROOT = PROJECT_ROOT / "tools" / "dino_compat"


def _as_root(value: str | Path | None) -> Path | None:
    if not value:
        return None
    root = Path(value).expanduser().resolve()
    return root if root.exists() else None


def _launch_script(root: Path) -> Path | None:
    candidates = [
        root / "vbench" / "launch" / "evaluate.py",
        root / "launch" / "evaluate.py",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _distributed_command(script: Path) -> list[str]:
    if os.name == "nt":
        return [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "vbench_windows_runner.py"),
            "--script",
            str(script),
        ]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "1",
        str(script),
    ]


def _docker_image_available(image: str) -> bool:
    docker = shutil.which("docker")
    if not docker:
        return False
    try:
        completed = subprocess.run(
            [docker, "image", "inspect", image],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _ensure_dino_compat_source() -> bool:
    """Install the offline DINO loader when the original repo is unavailable."""
    hubconf = DINO_SOURCE_ROOT / "hubconf.py"
    checkpoint = (
        MODEL_CACHE_DIR
        / "vbench"
        / "dino_model"
        / "dino_vitbase16_pretrain.pth"
    )
    compat_hubconf = DINO_COMPAT_ROOT / "hubconf.py"
    compat_vit = DINO_COMPAT_ROOT / "vision_transformer.py"
    if not checkpoint.exists() or not compat_hubconf.exists() or not compat_vit.exists():
        return False
    if not hubconf.exists():
        DINO_SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(compat_hubconf, hubconf)
        shutil.copy2(compat_vit, DINO_SOURCE_ROOT / "vision_transformer.py")
    return hubconf.exists() and (DINO_SOURCE_ROOT / "vision_transformer.py").exists()


def _dino_available() -> bool:
    if not _ensure_dino_compat_source():
        return False
    return importlib.util.find_spec("timm") is not None


def _missing_local_assets(dimensions: list[str]) -> list[str]:
    assets: dict[str, list[Path]] = {
        "aesthetic_quality": [
            MODEL_CACHE_DIR / "vbench" / "clip_model" / "ViT-L-14.pt",
            MODEL_CACHE_DIR
            / "vbench"
            / "aesthetic_model"
            / "emb_reader"
            / "sa_0_4_vit_l_14_linear.pth",
        ],
    }
    missing: list[str] = []
    for dimension in dimensions:
        for path in assets.get(dimension, []):
            if not path.is_file():
                missing.append(str(path))
    return missing


def discover_vbench() -> dict[str, Any]:
    """Prefer the active project environment before external backends."""
    dino_available = _dino_available()
    package_spec = importlib.util.find_spec("vbench")
    if package_spec and package_spec.submodule_search_locations:
        package_root = Path(
            next(iter(package_spec.submodule_search_locations))
        ).resolve()
        launch_script = _launch_script(package_root)
        if launch_script:
            return {
                "available": True,
                "ready": _vbench_runtime_compatible(),
                "backend": "package",
                "root": str(package_root.parent),
                "command": _distributed_command(launch_script),
                "dino_available": dino_available,
            }

    configured_root = _as_root(os.environ.get("VBENCH_ROOT"))
    if configured_root:
        launch_script = _launch_script(configured_root)
        if launch_script:
            return {
                "available": True,
                "ready": _vbench_runtime_compatible(),
                "backend": "source",
                "root": str(configured_root),
                "command": _distributed_command(launch_script),
                "dino_available": dino_available,
            }

    executable = shutil.which("vbench")
    if executable:
        return {
            "available": True,
            "ready": _vbench_runtime_compatible(),
            "backend": "package-cli",
            "root": None,
            "command": [executable, "evaluate"],
            "dino_available": dino_available,
        }

    sibling_executable = Path(sys.executable).resolve().parent / "vbench.exe"
    if sibling_executable.exists():
        return {
            "available": True,
            "ready": _vbench_runtime_compatible(),
            "backend": "package-cli",
            "root": None,
            "command": [str(sibling_executable), "evaluate"],
            "dino_available": dino_available,
        }

    docker = shutil.which("docker")
    if docker and _docker_image_available(VBENCH_DOCKER_IMAGE):
        return {
            "available": True,
            "ready": True,
            "backend": "docker",
            "root": None,
            "docker": docker,
            "image": VBENCH_DOCKER_IMAGE,
            "command": None,
            "dino_available": dino_available,
        }

    return {
        "available": False,
        "ready": False,
        "backend": None,
        "root": None,
        "command": None,
        "dino_available": dino_available,
    }


def _vbench_runtime_compatible() -> dict[str, Any]:
    """Check the package contract instead of treating an entry point as ready."""
    expected_transformers = "4.33.2"
    try:
        transformers_version = metadata.version("transformers")
    except metadata.PackageNotFoundError:
        transformers_version = None
    try:
        vbench_version = metadata.version("vbench")
    except metadata.PackageNotFoundError:
        vbench_version = None
    compatible = transformers_version == expected_transformers
    return {
        "compatible": compatible,
        "required_transformers": expected_transformers,
        "transformers_version": transformers_version,
        "vbench_version": vbench_version,
        "reason": (
            None
            if compatible
            else (
                "VBench requires transformers==4.33.2; "
                f"active environment has {transformers_version or 'missing'}."
            )
        ),
    }


def _scalar_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
        return score if math.isfinite(score) else None
    return None


def _matching_score(value: Any, dimension: str) -> float | None:
    """Find the named dimension, never a generic top-level score."""
    target = dimension.casefold().replace("_", "")

    def visit(node: Any) -> float | None:
        if isinstance(node, dict):
            for key, child in node.items():
                normalized = str(key).casefold().replace("_", "").replace(" ", "")
                if normalized == target:
                    direct = _scalar_score(child)
                    if direct is not None:
                        return direct
                    if isinstance(child, dict):
                        for score_key in ("score", "value", "mean"):
                            candidate = _scalar_score(child.get(score_key))
                            if candidate is not None:
                                return candidate
                if (
                    normalized in {"dimension", "name", "metric"}
                    and str(child).casefold().replace("_", "").replace(" ", "")
                    == target
                    and isinstance(node, dict)
                ):
                    for score_key in ("score", "value", "mean"):
                        candidate = _scalar_score(node.get(score_key))
                        if candidate is not None:
                            return candidate
            for child in node.values():
                found = visit(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = visit(child)
                if found is not None:
                    return found
        return None

    return visit(value)


def _parse_output(output_dir: Path, dimensions: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    json_files = sorted(output_dir.rglob("*.json"))
    for dimension in dimensions:
        score: float | None = None
        source_file: str | None = None
        for json_file in json_files:
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            score = _matching_score(payload, dimension)
            if score is not None:
                source_file = json_file.name
                break
        records.append(
            {
                "dimension": dimension,
                "score": score,
                "direction": "higher_is_better",
                "source_file": source_file,
            }
        )
    return records


def run_vbench(
    video_path: str | Path,
    dimensions: list[str],
    output_root: str | Path,
) -> dict[str, Any]:
    video_path = Path(video_path).resolve()
    dimensions = [
        dimension for dimension in dimensions if dimension in VBENCH_DIMENSIONS
    ]
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not dimensions:
        raise ValueError("At least one supported VBench dimension is required.")

    installation = discover_vbench()
    blocked_dimensions = (
        ["subject_consistency"]
        if "subject_consistency" in dimensions
        and not installation.get("dino_available", False)
        else []
    )
    runnable_dimensions = [
        dimension for dimension in dimensions if dimension not in blocked_dimensions
    ]
    if not installation["available"]:
        return {
            "available": False,
            "ready": False,
            "status": "not_installed",
            "installation": (
                "未检测到 VBench。请先安装 requirements/vbench.txt，或设置 "
                "VBENCH_ROOT 指向 VBench 源码目录。"
            ),
            "dimensions": dimensions,
            "records": [],
            "blocked_dimensions": blocked_dimensions,
        }
    compatibility = installation.get("ready")
    if isinstance(compatibility, dict) and not compatibility.get("compatible"):
        return {
            "available": True,
            "ready": False,
            "status": "incompatible",
            "backend": installation["backend"],
            "dimensions": dimensions,
            "records": [],
            "blocked_dimensions": blocked_dimensions,
            "compatibility": compatibility,
        }
    if compatibility is False:
        return {
            "available": True,
            "ready": False,
            "status": "incompatible",
            "backend": installation["backend"],
            "dimensions": dimensions,
            "records": [],
            "blocked_dimensions": blocked_dimensions,
        }
    missing_assets = _missing_local_assets(runnable_dimensions)
    if missing_assets:
        return {
            "available": True,
            "ready": False,
            "status": "not_ready",
            "backend": installation["backend"],
            "dimensions": dimensions,
            "records": [
                {
                    "dimension": dimension,
                    "score": None,
                    "direction": "higher_is_better",
                    "source_file": None,
                    "status": "unavailable",
                    "reason": (
                        "Missing local VBench assets: "
                        + ", ".join(missing_assets)
                    ),
                }
                for dimension in dimensions
            ],
            "blocked_dimensions": blocked_dimensions,
            "missing_assets": missing_assets,
        }
    if not runnable_dimensions:
        return {
            "available": True,
            "ready": False,
            "status": "not_ready",
            "backend": installation["backend"],
            "dimensions": dimensions,
            "records": [
                {
                    "dimension": dimension,
                    "score": None,
                    "direction": "higher_is_better",
                    "source_file": None,
                    "status": "unavailable",
                    "reason": "DINO source/checkpoint is unavailable.",
                }
                for dimension in blocked_dimensions
            ],
            "blocked_dimensions": blocked_dimensions,
        }

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root).resolve() / "vbench" / run_id
    input_dir = run_dir / "input"
    result_dir = run_dir / "result"
    input_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    staged_video = input_dir / video_path.name
    shutil.copy2(video_path, staged_video)

    if installation["backend"] == "docker":
        host_cache = (MODEL_CACHE_DIR / "vbench").resolve()
        command = [
            installation["docker"],
            "run",
            "--rm",
            "--gpus",
            "all",
            "-e",
            "VBENCH_CACHE_DIR=/root/.cache/vbench",
            "-e",
            "TORCH_HOME=/root/.cache/torch",
            "-v",
            f"{input_dir.resolve()}:/workspace/input:ro",
            "-v",
            f"{result_dir.resolve()}:/workspace/result",
            "-v",
            f"{host_cache}:/root/.cache/vbench",
            installation["image"],
            "--videos_path",
            "/workspace/input",
            "--mode",
            "custom_input",
            "--output_path",
            "/workspace/result",
            "--dimension",
            *runnable_dimensions,
            "--load_ckpt_from_local",
            "True",
        ]
    else:
        command = [
            *installation["command"],
            "--videos_path",
            str(input_dir),
            "--mode",
            "custom_input",
            "--output_path",
            str(result_dir),
            "--dimension",
            *runnable_dimensions,
            "--load_ckpt_from_local",
            "True",
        ]
    process_env = os.environ.copy()
    if os.name == "nt":
        process_env["USE_LIBUV"] = "0"
        process_env["TORCH_USE_LIBUV"] = "0"
    completed = subprocess.run(
        command,
        cwd=installation["root"] or str(run_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=process_env,
        check=False,
    )
    parsed_records = _parse_output(result_dir, runnable_dimensions)
    parsed_by_dimension = {
        record["dimension"]: record for record in parsed_records
    }
    records = []
    for dimension in dimensions:
        if dimension in blocked_dimensions:
            records.append(
                {
                    "dimension": dimension,
                    "score": None,
                    "direction": "higher_is_better",
                    "source_file": None,
                    "status": "unavailable",
                    "reason": "DINO source/checkpoint is unavailable.",
                }
            )
        else:
            records.append(parsed_by_dimension[dimension])
    windows_distributed_error = (
        installation["backend"] != "docker"
        and
        os.name == "nt"
        and (
            "use_libuv" in completed.stderr
            or "RendezvousConnectionError" in completed.stderr
        )
    )
    return {
        "available": True,
        "ready": completed.returncode == 0,
        "status": "completed" if completed.returncode == 0 else "failed",
        "backend": installation["backend"],
        "platform_note": (
            "VBench 官方分布式脚本在当前 Windows/PyTorch 构建上无法启动。"
            "请将 VBench 后端放到 Linux、WSL 或 Docker 中运行。"
            if windows_distributed_error
            else None
        ),
        "command": command,
        "return_code": completed.returncode,
        "dimensions": dimensions,
        "records": records,
        "blocked_dimensions": blocked_dimensions,
        "output_dir": str(result_dir),
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
    }

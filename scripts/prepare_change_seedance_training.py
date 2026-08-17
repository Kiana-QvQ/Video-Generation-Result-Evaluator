"""Prepare opt-in Change/1k Seedance hard-example training assets.

Does NOT overwrite default holdout, dual-.pt model, or noleak fusion head.
Creates parallel manifests and a command checklist for local retrain.

Protocol (fixed, seed=42):
- Train Change (has lm_mp_*): BaiJunZhiJiang, Happy, LeJiShengBei
- Eval Change (never train): YanWu, ImissU
  - ImissU: MediaPipe landmarks unavailable (extreme profile); still scored

Resolution policy:
- Default long-edge target is **1024 (1k)**, never below.
- Native Change clips are 720p; for .pt training they are resampled up to 1k.
- High-res Seedance train clips are downsampled to 1k (never upscaled).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from evaluator.vedio_pred.wangxing_dual_pt import build_wangxing_split_manifest

TRAIN_STEMS = (
    "BaiJunZhiJiang_Change",
    "Happy_Change",
    "LeJiShengBei_Change",
)
EVAL_STEMS = (
    "YanWu_Change",
    "ImissU_Change",
)
NO_LANDMARK_STEMS = frozenset({"ImissU_Change"})
MIN_LONG_EDGE = 1024


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _item(stem: str) -> dict[str, Any]:
    video = project_path(f"data/test/AI/{stem}.mp4")
    au = project_path(f"data/au/test/AI/{stem}.csv")
    if not video.is_file():
        raise FileNotFoundError(f"Missing Change video: {video}")
    if not au.is_file():
        raise FileNotFoundError(f"Missing Change AU CSV: {au}")
    return {
        "name": f"{stem}.mp4",
        "stem": stem,
        "video": _rel(video),
        "au": _rel(au),
        "landmark_expected": stem not in NO_LANDMARK_STEMS,
        "note": (
            "MediaPipe landmarks unavailable (extreme profile)."
            if stem in NO_LANDMARK_STEMS
            else "LibreFace AU with lm_mp_* (MD_CL-compatible)."
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _video_dimensions(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, 0
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return width, height


def _video_max_edge(path: Path) -> int:
    width, height = _video_dimensions(path)
    return max(width, height)


def build_protocol(*, long_edge: int, seed: int) -> dict[str, Any]:
    train = [_item(stem) for stem in TRAIN_STEMS]
    eval_items = [_item(stem) for stem in EVAL_STEMS]
    return {
        "schema_version": "change_seedance_protocol_v1",
        "engine": "Seedance",
        "batch": "data/test/AI *_Change",
        "seed": int(seed),
        "min_long_edge": int(long_edge),
        "purpose": (
            "Same-engine hard examples (Change export / extreme lighting or "
            f"profile). .pt uses >= {long_edge}px long-edge. "
            "Train subset may join AU+.pt; eval subset never trains."
        ),
        "train": train,
        "eval": eval_items,
        "defaults_preserved": {
            "holdout": "data/forensics/holdout_split.json",
            "dual_pt_model": (
                "outputs/vedio_pred/models/wangxing_dual_scale_classifier.pt"
            ),
            "fusion_head": (
                "outputs/forensics/learned_fusion_head_logistic_noleak.json"
            ),
        },
    }


def build_change_eval_holdout(
    protocol: dict[str, Any],
    *,
    real_support: int = 10,
) -> dict[str, Any]:
    """Holdout-shaped manifest for Change OOD eval (+ few reals for FP check)."""
    base = json.loads(
        project_path("data/forensics/holdout_split.json").read_text(
            encoding="utf-8-sig"
        )
    )
    real = list(base.get("real", []))[: max(0, int(real_support))]
    seedance = [
        {
            "video": item["video"],
            "au": item["au"],
            "name": item["name"],
            "landmark_expected": item.get("landmark_expected", True),
            "note": item.get("note", ""),
        }
        for item in protocol["eval"]
    ]
    return {
        "schema_version": "forensics_holdout_split_v1",
        "note": (
            "Change Seedance OOD eval only. Not a replacement for the main "
            "MD_CL vs WangXing_Seedance holdout."
        ),
        "summary": {
            "real": len(real),
            "seedance": len(seedance),
            "source_protocol": "change_seedance_protocol_v1",
        },
        "real": real,
        "seedance": seedance,
    }


def build_exclude_holdout(protocol: dict[str, Any]) -> dict[str, Any]:
    """Main holdout + Change eval AUs/videos — for noleak training exclusion."""
    base = json.loads(
        project_path("data/forensics/holdout_split.json").read_text(
            encoding="utf-8-sig"
        )
    )
    seedance = list(base.get("seedance", []))
    for item in protocol["eval"]:
        seedance.append(
            {
                "video": item["video"],
                "au": item["au"],
                "name": item["name"],
                "change_eval": True,
            }
        )
    return {
        "schema_version": "forensics_holdout_split_v1",
        "note": (
            "Union of official holdout and Change eval. Use only as "
            "--holdout-manifest exclusion when fitting change-aug profiles/heads."
        ),
        "summary": {
            "real": len(base.get("real", [])),
            "seedance": len(seedance),
        },
        "real": base.get("real", []),
        "seedance": seedance,
    }


def build_pseudo_manifest_change_aug(protocol: dict[str, Any]) -> dict[str, Any]:
    src = project_path(
        "data/au/WangXing_Seedance/pseudo_expression_manifest.json"
    )
    payload = json.loads(src.read_text(encoding="utf-8-sig"))
    records = list(payload.get("records", []))
    existing = {
        str(Path(str(rec.get("au_path", ""))).resolve()).casefold()
        for rec in records
        if isinstance(rec, dict) and rec.get("au_path")
    }
    added = 0
    for item in protocol["train"]:
        au = project_path(item["au"])
        video = project_path(item["video"])
        key = str(au.resolve()).casefold()
        if key in existing:
            continue
        records.append(
            {
                "video_path": str(video.resolve()),
                "au_path": str(au.resolve()),
                "source_type": "generated_wangxing",
                "pseudo_label": "change_seedance_hard",
                "label_status": "change_aug_train",
                "use_for_training": True,
                "confidence_0_1": 1.0,
                "compatibility_0_1": 1.0,
                "margin_0_1": 1.0,
                "valid_frame_ratio": 1.0,
                "note": "Injected Change Seedance hard example for source profile.",
            }
        )
        existing.add(key)
        added += 1
    payload = dict(payload)
    payload["records"] = records
    payload["change_aug"] = {
        "added_train_records": added,
        "protocol": "change_seedance_protocol_v1",
        "base_manifest": _rel(src),
    }
    return payload


def _resample_long_edge(
    src: Path,
    dst: Path,
    *,
    long_edge: int,
) -> None:
    """Resample so max(w,h)==long_edge (even dims)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 0:
        existing_edge = _video_max_edge(dst)
        if existing_edge == long_edge:
            return
    vf = (
        f"scale="
        f"'if(gt(iw,ih),{long_edge},-2)':"
        f"'if(gt(iw,ih),-2,{long_edge})',"
        f"scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-an",
        str(dst),
    ]
    completed = subprocess.run(cmd, capture_output=True, check=False)
    if completed.returncode != 0:
        err = (completed.stderr or completed.stdout or b"").decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            f"ffmpeg failed ({completed.returncode}): {err[-500:]}"
        )
    if not dst.is_file() or dst.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg produced empty output: {dst}")
    width, height = _video_dimensions(dst)
    if max(width, height) != long_edge:
        raise RuntimeError(
            f"Unexpected output size for {dst.name}: got {width}x{height}, "
            f"expected long edge {long_edge}"
        )


def build_dual_split_change_aug(
    protocol: dict[str, Any],
    *,
    downsample_count: int,
    long_edge: int,
    seed: int,
) -> dict[str, Any]:
    if long_edge < MIN_LONG_EDGE:
        raise ValueError(f"long_edge must be >= {MIN_LONG_EDGE}, got {long_edge}")

    base = build_wangxing_split_manifest(
        project_root=PROJECT_ROOT,
        real_root=project_path("data/MD_CL"),
        fake_root=project_path("data/WangXing_Seedance"),
        holdout_manifest=project_path("data/forensics/holdout_split.json"),
        real_train_count=120,
        seed=seed,
    )
    train_fake: list[str] = list(base["train"]["fake"])

    # Change train: force >=1k for .pt (native files stay 720 for AU pairing).
    change_1k_paths: list[str] = []
    change_out = project_path(f"data/_aug/change_le{long_edge}")
    for item in protocol["train"]:
        src = project_path(item["video"])
        dst = change_out / f"{item['stem']}_le{long_edge}.mp4"
        try:
            _resample_long_edge(src, dst, long_edge=long_edge)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"Mandatory Change {long_edge}px asset failed for {src.name}: {exc}"
            ) from exc
        change_1k_paths.append(str(dst.resolve()))
        train_fake.append(str(dst.resolve()))

    # High-res Seedance -> 1k (downscale only; never upscale small clips).
    down_paths: list[str] = []
    if downsample_count > 0:
        rng = np.random.default_rng(seed)
        pool: list[Path] = []
        for path_str in base["train"]["fake"]:
            path = Path(path_str)
            if not path.is_file():
                continue
            if _video_max_edge(path) > long_edge:
                pool.append(path)
        if pool:
            picks = [
                pool[int(i)]
                for i in rng.choice(
                    len(pool),
                    size=min(downsample_count, len(pool)),
                    replace=False,
                )
            ]
            out_root = project_path(f"data/_aug/seedance_le{long_edge}")
            for src in picks:
                dst = out_root / f"{src.stem}_le{long_edge}.mp4"
                try:
                    _resample_long_edge(src, dst, long_edge=long_edge)
                except (OSError, RuntimeError) as exc:
                    print(f"WARN downsample skip {src.name}: {exc}", flush=True)
                    continue
                if dst.is_file():
                    down_paths.append(str(dst.resolve()))
            train_fake.extend(down_paths)
        else:
            print(
                f"WARN no train Seedance clips with max edge > {long_edge} "
                "for downscale augmentation.",
                flush=True,
            )

    seen: set[str] = set()
    deduped: list[str] = []
    for path in train_fake:
        key = str(Path(path).resolve()).casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(str(Path(path).resolve()))

    manifest = dict(base)
    manifest["schema_version"] = "wangxing_dual_pt_split_change_aug_v1"
    manifest["train"] = {
        "real": list(base["train"]["real"]),
        "fake": deduped,
    }
    manifest["test"] = dict(base["test"])
    manifest["change_aug"] = {
        "protocol": "change_seedance_protocol_v1",
        "min_long_edge": long_edge,
        "change_train_videos_native": [item["video"] for item in protocol["train"]],
        "change_train_videos_1k": change_1k_paths,
        "downsampled_long_edge": long_edge,
        "downsampled_count": len(down_paths),
        "downsampled_videos": down_paths,
        "note": (
            "Test remains official Seedance holdout. Evaluate Change OOD "
            "separately with holdout_change_eval.json. "
            f".pt Change train uses resampled >= {long_edge}px long-edge."
        ),
    }
    manifest["counts"] = {
        "train_real": len(manifest["train"]["real"]),
        "train_fake": len(manifest["train"]["fake"]),
        "test_real": len(manifest["test"]["real"]),
        "test_fake": len(manifest["test"]["fake"]),
        "change_train_fake_1k": len(change_1k_paths),
        "downsampled_fake": len(down_paths),
    }
    return manifest


def _write_commands_file_legacy(path: Path) -> None:
    text = f"""# Change Seedance 增强重训命令（默认产物不覆盖，长边 >= {MIN_LONG_EDGE}）

在仓库根目录、使用 `.venv`。先跑准备（默认把 Change 与高清 Seedance 重采样到 **1k**）：

```powershell
.\\.venv\\Scripts\\python.exe scripts\\prepare_change_seedance_training.py
# 仅跳过「高清 Seedance→1k」批量增强时（Change→1k 仍会做）：
# .\\.venv\\Scripts\\python.exe scripts\\prepare_change_seedance_training.py --skip-downsample
```

## 1) Source 画像（排除官方 holdout + Change eval）

```powershell
.\\.venv\\Scripts\\python.exe scripts\\train_wangxing_specialization.py --skip-identity `
  --seedance-label-manifest data\\au\\WangXing_Seedance\\pseudo_expression_manifest_change_aug.json `
  --holdout-manifest data\\forensics\\holdout_split_plus_change_eval.json `
  --source-profile-output outputs\\forensics\\wangxing_source_profile_change_aug_noleak.json `
  --expression-output outputs\\forensics\\wangxing_expression_profile_change_aug_tmp.json
```

## 2) AU 学习头（注入 Change train AU，输出新 JSON）

```powershell
.\\.venv\\Scripts\\python.exe scripts\\train_learned_fusion_head.py train `
  --forensics-profile outputs\\forensics\\forensics_profiles_quality_filtered.json `
  --source-profile outputs\\forensics\\wangxing_source_profile_change_aug_noleak.json `
  --holdout-manifest data\\forensics\\holdout_split_plus_change_eval.json `
  --extra-generated-au-manifest data\\forensics\\change_seedance_protocol.json `
  --model-type logistic `
  --output outputs\\forensics\\learned_fusion_head_logistic_change_aug.json
```

## 3) 双尺度 .pt（新模型路径，不覆盖旧 dual）

```powershell
.\\.venv\\Scripts\\python.exe scripts\\train_wangxing_video_pt.py train `
  --manifest outputs\\vedio_pred\\wangxing_dual_pt_split_change_aug.json `
  --cache-dir outputs\\vedio_pred\\cache_change_aug `
  --model-path outputs\\vedio_pred\\models\\wangxing_dual_scale_classifier_change_aug.pt `
  --metrics-output outputs\\vedio_pred\\wangxing_dual_pt_holdout_metrics_change_aug.json `
  --epochs 80 --batch-size 16 --learning-rate 3e-4 --seed 42
```

## 4) 评估：官方 holdout（防回退）

```powershell
.\\.venv\\Scripts\\python.exe scripts\\train_learned_fusion_head.py evaluate `
  --forensics-profile outputs\\forensics\\forensics_profiles_quality_filtered.json `
  --source-profile outputs\\forensics\\wangxing_source_profile_change_aug_noleak.json `
  --holdout-manifest data\\forensics\\holdout_split.json `
  --head outputs\\forensics\\learned_fusion_head_logistic_change_aug.json `
  --output outputs\\forensics\\learned_fusion_holdout_metrics_change_aug.json

.\\.venv\\Scripts\\python.exe scripts\\run_wangxing_specialization_fused.py evaluate `
  --holdout-manifest data\\forensics\\holdout_split.json `
  --source-profile outputs\\forensics\\wangxing_source_profile_change_aug_noleak.json `
  --learned-head outputs\\forensics\\learned_fusion_head_logistic_change_aug.json `
  --pt-model outputs\\vedio_pred\\models\\wangxing_dual_scale_classifier_change_aug.pt `
  --pt-cache-dir outputs\\vedio_pred\\cache_change_aug `
  --output outputs\\forensics\\wangxing_specialization_fused_holdout_metrics_change_aug.json
```

## 5) 评估：Change OOD

```powershell
.\\.venv\\Scripts\\python.exe scripts\\train_learned_fusion_head.py evaluate `
  --forensics-profile outputs\\forensics\\forensics_profiles_quality_filtered.json `
  --source-profile outputs\\forensics\\wangxing_source_profile_change_aug_noleak.json `
  --holdout-manifest data\\forensics\\holdout_change_eval.json `
  --head outputs\\forensics\\learned_fusion_head_logistic_change_aug.json `
  --output outputs\\forensics\\learned_fusion_change_ood_metrics.json

.\\.venv\\Scripts\\python.exe scripts\\run_wangxing_specialization_fused.py evaluate `
  --holdout-manifest data\\forensics\\holdout_change_eval.json `
  --source-profile outputs\\forensics\\wangxing_source_profile_change_aug_noleak.json `
  --learned-head outputs\\forensics\\learned_fusion_head_logistic_change_aug.json `
  --pt-model outputs\\vedio_pred\\models\\wangxing_dual_scale_classifier_change_aug.pt `
  --pt-cache-dir outputs\\vedio_pred\\cache_change_aug `
  --output outputs\\forensics\\wangxing_specialization_fused_change_ood_metrics.json
```

说明：
- 未换输出路径时，原 MD_CL↔Seedance 默认逻辑与旧模型文件不变。
- Change 原生 720p 仅作 AU；`.pt` 训练用 `data/_aug/change_le{MIN_LONG_EDGE}/` 的 1k 版。
- ImissU 无 MediaPipe landmark，Change OOD 上可能仍偏弱。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        text.replace("{MIN_LONG_EDGE}", str(MIN_LONG_EDGE)),
        encoding="utf-8",
    )


def write_commands_file(path: Path, *, long_edge: int, seed: int) -> None:
    prepare_cmd = ".\\.venv\\Scripts\\python.exe scripts\\prepare_change_seedance_training.py"
    skip_downsample_cmd = (
        ".\\.venv\\Scripts\\python.exe scripts\\prepare_change_seedance_training.py "
        "--skip-downsample"
    )
    if long_edge != MIN_LONG_EDGE or seed != 42:
        suffix_parts: list[str] = []
        if long_edge != MIN_LONG_EDGE:
            suffix_parts.append(f"--long-edge {long_edge}")
        if seed != 42:
            suffix_parts.append(f"--seed {seed}")
        suffix = " " + " ".join(suffix_parts)
        prepare_cmd += suffix
        skip_downsample_cmd += suffix

    text = rf"""# Change Seedance 增强重训命令（默认产物不覆盖，长边 >= {long_edge}）

在仓库根目录、使用 `.venv`。先跑准备（默认把 Change 与高分辨率 Seedance 重采样到 1k）：

```powershell
{prepare_cmd}
# 仅跳过“高分辨率 Seedance→1k”批量增强时（Change→1k 仍会做）：
# {skip_downsample_cmd}
```

## 1) Source 画像（排除官方 holdout + Change eval）

```powershell
.\.venv\Scripts\python.exe scripts\train_wangxing_specialization.py --skip-identity `
  --seedance-label-manifest data\au\WangXing_Seedance\pseudo_expression_manifest_change_aug.json `
  --holdout-manifest data\forensics\holdout_split_plus_change_eval.json `
  --source-profile-output outputs\forensics\wangxing_source_profile_change_aug_noleak.json `
  --expression-output outputs\forensics\wangxing_expression_profile_change_aug_tmp.json
```

## 2) AU 学习头（注入 Change train AU，输出新 JSON）

```powershell
.\.venv\Scripts\python.exe scripts\train_learned_fusion_head.py train `
  --forensics-profile outputs\forensics\forensics_profiles_quality_filtered.json `
  --source-profile outputs\forensics\wangxing_source_profile_change_aug_noleak.json `
  --holdout-manifest data\forensics\holdout_split_plus_change_eval.json `
  --extra-generated-au-manifest data\forensics\change_seedance_protocol.json `
  --model-type logistic `
  --seed {seed} `
  --output outputs\forensics\learned_fusion_head_logistic_change_aug.json
```

## 3) 双尺度 `.pt`（新模型路径，不覆盖旧 dual）

```powershell
.\.venv\Scripts\python.exe scripts\train_wangxing_video_pt.py train `
  --manifest outputs\vedio_pred\wangxing_dual_pt_split_change_aug.json `
  --cache-dir outputs\vedio_pred\cache_change_aug `
  --model-path outputs\vedio_pred\models\wangxing_dual_scale_classifier_change_aug.pt `
  --metrics-output outputs\vedio_pred\wangxing_dual_pt_holdout_metrics_change_aug.json `
  --epochs 80 --batch-size 16 --learning-rate 3e-4 --seed {seed}
```

## 4) 评估：官方 holdout（防回退）

```powershell
.\.venv\Scripts\python.exe scripts\train_learned_fusion_head.py evaluate `
  --forensics-profile outputs\forensics\forensics_profiles_quality_filtered.json `
  --source-profile outputs\forensics\wangxing_source_profile_change_aug_noleak.json `
  --holdout-manifest data\forensics\holdout_split.json `
  --head outputs\forensics\learned_fusion_head_logistic_change_aug.json `
  --output outputs\forensics\learned_fusion_holdout_metrics_change_aug.json

.\.venv\Scripts\python.exe scripts\run_wangxing_specialization_fused.py evaluate `
  --holdout-manifest data\forensics\holdout_split.json `
  --source-profile outputs\forensics\wangxing_source_profile_change_aug_noleak.json `
  --learned-head outputs\forensics\learned_fusion_head_logistic_change_aug.json `
  --pt-model outputs\vedio_pred\models\wangxing_dual_scale_classifier_change_aug.pt `
  --pt-cache-dir outputs\vedio_pred\cache_change_aug `
  --output outputs\forensics\wangxing_specialization_fused_holdout_metrics_change_aug.json
```

## 5) 评估：Change OOD

```powershell
.\.venv\Scripts\python.exe scripts\train_learned_fusion_head.py evaluate `
  --forensics-profile outputs\forensics\forensics_profiles_quality_filtered.json `
  --source-profile outputs\forensics\wangxing_source_profile_change_aug_noleak.json `
  --holdout-manifest data\forensics\holdout_change_eval.json `
  --head outputs\forensics\learned_fusion_head_logistic_change_aug.json `
  --output outputs\forensics\learned_fusion_change_ood_metrics.json

.\.venv\Scripts\python.exe scripts\run_wangxing_specialization_fused.py evaluate `
  --holdout-manifest data\forensics\holdout_change_eval.json `
  --source-profile outputs\forensics\wangxing_source_profile_change_aug_noleak.json `
  --learned-head outputs\forensics\learned_fusion_head_logistic_change_aug.json `
  --pt-model outputs\vedio_pred\models\wangxing_dual_scale_classifier_change_aug.pt `
  --pt-cache-dir outputs\vedio_pred\cache_change_aug `
  --output outputs\forensics\wangxing_specialization_fused_change_ood_metrics.json
```

说明：
- 未换输出路径时，原 MD_CL→Seedance 默认逻辑与旧模型文件不变。
- Change 原生 720p 仅作 AU；`.pt` 训练用 `data/_aug/change_le{long_edge}/` 的 1k 版。
- ImissU 无 MediaPipe landmark，Change OOD 上可能仍偏弱。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Change Seedance hard-example training assets "
            f"(long-edge >= {MIN_LONG_EDGE})."
        )
    )
    parser.add_argument(
        "--downsample-count",
        type=int,
        default=24,
        help=(
            "How many high-res train Seedance clips to also keep as "
            f"long-edge {MIN_LONG_EDGE}+ downscales."
        ),
    )
    parser.add_argument(
        "--long-edge",
        type=int,
        default=MIN_LONG_EDGE,
        help=f"Target long-edge pixels (minimum {MIN_LONG_EDGE}).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-downsample",
        action="store_true",
        help=(
            "Skip high-res Seedance→1k batch. Change→1k for .pt is still done."
        ),
    )
    args = parser.parse_args(argv)

    long_edge = int(args.long_edge)
    if long_edge < MIN_LONG_EDGE:
        raise SystemExit(
            f"--long-edge must be >= {MIN_LONG_EDGE}, got {long_edge}"
        )

    protocol = build_protocol(long_edge=long_edge, seed=int(args.seed))
    protocol_path = project_path("data/forensics/change_seedance_protocol.json")
    _write_json(protocol_path, protocol)

    eval_holdout = build_change_eval_holdout(protocol)
    _write_json(
        project_path("data/forensics/holdout_change_eval.json"),
        eval_holdout,
    )

    exclude = build_exclude_holdout(protocol)
    _write_json(
        project_path("data/forensics/holdout_split_plus_change_eval.json"),
        exclude,
    )

    pseudo = build_pseudo_manifest_change_aug(protocol)
    _write_json(
        project_path(
            "data/au/WangXing_Seedance/pseudo_expression_manifest_change_aug.json"
        ),
        pseudo,
    )

    dual = build_dual_split_change_aug(
        protocol,
        downsample_count=(
            0 if args.skip_downsample else max(0, args.downsample_count)
        ),
        long_edge=long_edge,
        seed=int(args.seed),
    )
    _write_json(
        project_path("outputs/vedio_pred/wangxing_dual_pt_split_change_aug.json"),
        dual,
    )

    commands = project_path("docs/CHANGE_SEEDANCE_RETRAIN.md")
    write_commands_file(
        commands,
        long_edge=long_edge,
        seed=int(args.seed),
    )

    summary = {
        "protocol": str(protocol_path),
        "min_long_edge": long_edge,
        "train_change": [item["name"] for item in protocol["train"]],
        "eval_change": [item["name"] for item in protocol["eval"]],
        "dual_split_counts": dual["counts"],
        "change_1k": dual.get("change_aug", {}).get("change_train_videos_1k"),
        "commands": str(commands),
        "note": "Default dual .pt / noleak head files were not modified.",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

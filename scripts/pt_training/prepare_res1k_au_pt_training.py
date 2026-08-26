"""Prepare resolution-augmented AU+.pt training assets (test/AI never trains).

Does NOT overwrite:
- data/forensics/holdout_split.json
- outputs/vedio_pred/models/wangxing_dual_scale_classifier.pt
- outputs/forensics/learned_fusion_head_logistic_noleak.json
- outputs/forensics/forensics_profiles.json

Protocol:
- Train: MD_CL + WangXing_Seedance only (official holdout excluded)
- Augment: high-res train clips resampled to long-edge >= 1024 (default 1024)
- Test/OOD: all five data/test/AI *_Change.mp4 — evaluation only
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from evaluator.vedio_pred.wangxing_dual_pt import build_wangxing_split_manifest
from wangxing_project.joint_au_pt import attach_au_pairs

MIN_LONG_EDGE = 1024
TEST_AI_STEMS = (
    "BaiJunZhiJiang_Change",
    "Happy_Change",
    "ImissU_Change",
    "LeJiShengBei_Change",
    "YanWu_Change",
)
NO_LANDMARK_STEMS = frozenset({"ImissU_Change"})


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _video_max_edge(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return max(width, height)


def _resample_long_edge(src: Path, dst: Path, *, long_edge: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 0:
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


def _dedupe(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in paths:
        key = str(Path(item).resolve()).casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(str(Path(item).resolve()))
    return out


def _downsample_pool(
    pool: list[Path],
    *,
    out_root: Path,
    long_edge: int,
    count: int,
    seed: int,
    label: str,
) -> list[str]:
    eligible = [
        path for path in pool if path.is_file() and _video_max_edge(path) > long_edge
    ]
    if count <= 0:
        return []
    if not eligible:
        print(
            f"WARN {label}: no clips with max edge > {long_edge} to downscale.",
            flush=True,
        )
        return []
    rng = np.random.default_rng(seed)
    picks = [
        eligible[int(index)]
        for index in rng.choice(
            len(eligible),
            size=min(count, len(eligible)),
            replace=False,
        )
    ]
    written: list[str] = []
    for src in picks:
        dst = out_root / f"{src.stem}_le{long_edge}.mp4"
        try:
            _resample_long_edge(src, dst, long_edge=long_edge)
        except (OSError, RuntimeError) as exc:
            print(f"WARN {label} skip {src.name}: {exc}", flush=True)
            continue
        if dst.is_file():
            written.append(str(dst.resolve()))
    return written


def build_test_ai_holdout(*, real_support: int = 10) -> dict[str, Any]:
    holdout_path = project_path("data/forensics/holdout_split.json")
    if not holdout_path.is_file():
        raise FileNotFoundError(f"Official holdout not found: {holdout_path}")
    base = json.loads(holdout_path.read_text(encoding="utf-8-sig"))
    real = list(base.get("real", []))[: max(0, int(real_support))]
    seedance: list[dict[str, Any]] = []
    for stem in TEST_AI_STEMS:
        video = project_path(f"data/test/AI/{stem}.mp4")
        au = project_path(f"data/au/test/AI/{stem}.csv")
        if not video.is_file() or not au.is_file():
            raise FileNotFoundError(f"Missing test/AI pair: {stem}")
        seedance.append(
            {
                "video": _rel(video),
                "au": _rel(au),
                "name": f"{stem}.mp4",
                "landmark_expected": stem not in NO_LANDMARK_STEMS,
                "note": (
                    "Test-only; never used for AU+.pt training."
                    + (
                        " MediaPipe landmarks unavailable (extreme profile)."
                        if stem in NO_LANDMARK_STEMS
                        else ""
                    )
                ),
            }
        )
    return {
        "schema_version": "forensics_holdout_split_v1",
        "note": (
            "OOD eval for data/test/AI Change clips. "
            "Not a replacement for MD_CL vs WangXing_Seedance holdout. "
            "None of these Change clips enter training."
        ),
        "summary": {
            "real": len(real),
            "seedance": len(seedance),
        },
        "real": real,
        "seedance": seedance,
    }


def write_commands_file(path: Path) -> None:
    text = f"""# AU+.pt 中期融合重训（test/AI 五条只测不训，长边 >= {MIN_LONG_EDGE}）

网页 forensics 默认文件不覆盖。不要跑 `prepare_change_seedance_training.py`
（那套会把 Change 写进训练集）。

**主路径**：把 AU 25 维证据与双尺度视频特征 **拼接后进同一个 MLP**（early concat），
不再用 0.65/0.35 事后加权。产物路径全部带 `joint_au_pt`，不覆盖旧 dual / noleak。

在仓库根目录、`.venv`：

```powershell
.\\.venv\\Scripts\\python.exe scripts\\pt_training\\prepare_res1k_au_pt_training.py
```

## 1) 联合 AU+视频 .pt（新路径）

```powershell
.\\.venv\\Scripts\\python.exe scripts\\pt_training\\train_wangxing_joint_au_pt.py train `
  --manifest outputs\\vedio_pred\\wangxing_dual_pt_split_res1k.json `
  --cache-dir outputs\\vedio_pred\\cache_joint_au_pt_res1k `
  --model-path outputs\\vedio_pred\\models\\wangxing_joint_au_pt_res1k.pt `
  --metrics-output outputs\\vedio_pred\\wangxing_joint_au_pt_holdout_metrics_res1k.json `
  --source-profile outputs\\forensics\\wangxing_source_profile_holdout_excluded.json `
  --forensics-profile outputs\\forensics\\forensics_profiles.json `
  --epochs 80 --batch-size 16 --learning-rate 3e-4 --seed 42 `
  --device cuda
```

特征抽取（24f@1024 + 8f@2048 + AU 25 维）在 CPU；只有 MLP 训练上 GPU。
默认 `--device cuda`，若本机没有 CUDA 再改 `--device cpu`。

## 2) 评估：官方 holdout（防回退）

```powershell
.\\.venv\\Scripts\\python.exe scripts\\pt_training\\train_wangxing_joint_au_pt.py evaluate `
  --holdout-manifest data\\forensics\\holdout_split.json `
  --model-path outputs\\vedio_pred\\models\\wangxing_joint_au_pt_res1k.pt `
  --source-profile outputs\\forensics\\wangxing_source_profile_holdout_excluded.json `
  --forensics-profile outputs\\forensics\\forensics_profiles.json `
  --output outputs\\forensics\\wangxing_joint_au_pt_official_holdout_metrics.json
```

## 3) 评估：test/AI 五条（只测）

```powershell
.\\.venv\\Scripts\\python.exe scripts\\pt_training\\train_wangxing_joint_au_pt.py evaluate `
  --holdout-manifest data\\forensics\\holdout_test_AI.json `
  --model-path outputs\\vedio_pred\\models\\wangxing_joint_au_pt_res1k.pt `
  --source-profile outputs\\forensics\\wangxing_source_profile_holdout_excluded.json `
  --forensics-profile outputs\\forensics\\forensics_profiles.json `
  --output outputs\\forensics\\wangxing_joint_au_pt_test_AI_metrics.json
```

## （可选）仅视频 dual .pt，不含 AU

若还想单独训纯视频 res1k dual（对比基线）：

```powershell
.\\.venv\\Scripts\\python.exe scripts\\pt_training\\train_wangxing_video_pt.py train `
  --manifest outputs\\vedio_pred\\wangxing_dual_pt_split_res1k.json `
  --cache-dir outputs\\vedio_pred\\cache_res1k `
  --model-path outputs\\vedio_pred\\models\\wangxing_dual_scale_classifier_res1k.pt `
  --metrics-output outputs\\vedio_pred\\wangxing_dual_pt_holdout_metrics_res1k.json `
  --epochs 80 --batch-size 16 --learning-rate 3e-4 --seed 42
```

说明：
- 训练假/真的 1k 副本在 `data/_aug/seedance_le{MIN_LONG_EDGE}/` 与 `data/_aug/mdcl_le{MIN_LONG_EDGE}/`
- `*_le1024.mp4` 复用原片 AU（去掉 `_le1024` 后按 stem 匹配）
- 评估 Change 用原生 720p，不要升采样后再测
- ImissU 无 MediaPipe landmark，AU 支路可能偏弱
- 未改默认 dual `.pt` / noleak 头 / `forensics_profiles.json`；网页暂不接线
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        text.replace("{MIN_LONG_EDGE}", str(MIN_LONG_EDGE)),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare 1k resolution-augmented AU+.pt split. "
            "data/test/AI is evaluation-only."
        )
    )
    parser.add_argument("--fake-downsample-count", type=int, default=24)
    parser.add_argument("--real-downsample-count", type=int, default=24)
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
        help="Only write manifests; do not run ffmpeg.",
    )
    args = parser.parse_args(argv)

    long_edge = int(args.long_edge)
    if long_edge < MIN_LONG_EDGE:
        raise SystemExit(
            f"--long-edge must be >= {MIN_LONG_EDGE}, got {long_edge}"
        )

    holdout = project_path("data/forensics/holdout_split.json")
    if not holdout.is_file():
        raise SystemExit(f"Official holdout not found: {holdout}")

    protocol = {
        "schema_version": "res1k_au_pt_protocol_v1",
        "min_long_edge": long_edge,
        "train_sources": ["data/MD_CL", "data/WangXing_Seedance"],
        "eval_only": [f"{stem}.mp4" for stem in TEST_AI_STEMS],
        "never_train": [f"data/test/AI/{stem}.mp4" for stem in TEST_AI_STEMS],
        "defaults_preserved": {
            "holdout": "data/forensics/holdout_split.json",
            "dual_pt_model": (
                "outputs/vedio_pred/models/wangxing_dual_scale_classifier.pt"
            ),
            "fusion_head": (
                "outputs/forensics/learned_fusion_head_logistic_noleak.json"
            ),
            "forensics_profiles": "outputs/forensics/forensics_profiles.json",
        },
    }
    _write_json(project_path("data/forensics/res1k_au_pt_protocol.json"), protocol)

    test_holdout = build_test_ai_holdout()
    _write_json(project_path("data/forensics/holdout_test_AI.json"), test_holdout)

    base = build_wangxing_split_manifest(
        project_root=PROJECT_ROOT,
        real_root=project_path("data/MD_CL"),
        fake_root=project_path("data/WangXing_Seedance"),
        holdout_manifest=holdout,
        real_train_count=120,
        seed=int(args.seed),
    )
    train_real = list(base["train"]["real"])
    train_fake = list(base["train"]["fake"])

    down_real: list[str] = []
    down_fake: list[str] = []
    if not args.skip_downsample:
        down_fake = _downsample_pool(
            [Path(item) for item in base["train"]["fake"]],
            out_root=project_path(f"data/_aug/seedance_le{long_edge}"),
            long_edge=long_edge,
            count=max(0, int(args.fake_downsample_count)),
            seed=int(args.seed),
            label="seedance",
        )
        down_real = _downsample_pool(
            [Path(item) for item in base["train"]["real"]],
            out_root=project_path(f"data/_aug/mdcl_le{long_edge}"),
            long_edge=long_edge,
            count=max(0, int(args.real_downsample_count)),
            seed=int(args.seed) + 1,
            label="mdcl",
        )
        train_fake.extend(down_fake)
        train_real.extend(down_real)

    forbidden = {
        str(project_path(f"data/test/AI/{stem}.mp4").resolve()).casefold()
        for stem in TEST_AI_STEMS
    }
    train_real = [
        path
        for path in _dedupe(train_real)
        if str(Path(path).resolve()).casefold() not in forbidden
    ]
    train_fake = [
        path
        for path in _dedupe(train_fake)
        if str(Path(path).resolve()).casefold() not in forbidden
    ]

    manifest = dict(base)
    manifest["schema_version"] = "wangxing_dual_pt_split_res1k_v1"
    manifest["train"] = {"real": train_real, "fake": train_fake}
    manifest["test"] = dict(base["test"])
    manifest["res1k_aug"] = {
        "min_long_edge": long_edge,
        "downsampled_real": down_real,
        "downsampled_fake": down_fake,
        "test_ai_never_in_train": True,
        "note": (
            "Official holdout remains the .pt test set. "
            "Score data/test/AI with holdout_test_AI.json."
        ),
    }
    manifest["counts"] = {
        "train_real": len(train_real),
        "train_fake": len(train_fake),
        "test_real": len(manifest["test"]["real"]),
        "test_fake": len(manifest["test"]["fake"]),
        "downsampled_real": len(down_real),
        "downsampled_fake": len(down_fake),
    }
    manifest = attach_au_pairs(
        manifest,
        project_root=PROJECT_ROOT,
        holdout_manifest=holdout,
    )
    manifest["counts"]["downsampled_real"] = len(down_real)
    manifest["counts"]["downsampled_fake"] = len(down_fake)
    _write_json(
        project_path("outputs/vedio_pred/wangxing_dual_pt_split_res1k.json"),
        manifest,
    )

    commands = project_path("docs/01_algorithm/RES1K_AU_PT_RETRAIN.md")
    write_commands_file(commands)
    summary = {
        "protocol": "data/forensics/res1k_au_pt_protocol.json",
        "min_long_edge": long_edge,
        "eval_only": protocol["eval_only"],
        "counts": manifest["counts"],
        "au_pair_missing": {
            key: len(value)
            for key, value in (manifest.get("au_pair_missing") or {}).items()
        },
        "commands": str(commands),
        "joint_train": (
            "scripts/pt_training/train_wangxing_joint_au_pt.py train "
            "(AU 25-d + dual video features → one MLP)"
        ),
        "note": (
            "Default dual .pt / noleak head / forensics_profiles were not modified. "
            "test/AI Change clips are not in the train lists."
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

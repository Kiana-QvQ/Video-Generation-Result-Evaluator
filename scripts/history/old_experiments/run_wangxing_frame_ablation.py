"""Wang Xing video .pt ablation: frames x resolution on the same holdout.

Default grid: 8/24/32 frames x 1024(1k)/2048(2k).
Reports generated_recall and overall_accuracy for report comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from evaluator.vedio_pred.wangxing_dual_pt import build_wangxing_split_manifest
from wangxing_project.frame_ablation import train_and_eval_single_scale


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _res_label(frame_size: int) -> str:
    if int(frame_size) == 1024:
        return "1k"
    if int(frame_size) == 2048:
        return "2k"
    return f"{frame_size}px"


def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * float(value):.1f}%"


def _write_report_md(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# 视频 `.pt` 单尺度消融：帧数 × 分辨率",
        "",
        "## 协议",
        "",
        "- 训练：非 holdout 真拍均匀 120 + 非 holdout Seedance",
        "- 测试：forensics holdout 50 真 + 50 生成",
        "- 指标：生成召回 = TP/(TP+FN)；整体准确率 = (TP+TN)/N",
        "- 说明：本表为 **单尺度视频 `.pt` 支路**对比，不含 AU 融合",
        "",
        "## 对比表",
        "",
        "| 帧数 | 分辨率 | 生成召回 | 整体准确率 | 真拍召回 | 生成精确率 |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["table"]:
        lines.append(
            f"| {row['num_frames']} | {row['resolution']} "
            f"({row['frame_size']}) | {_pct(row['generated_recall'])} "
            f"| {_pct(row['overall_accuracy'])} "
            f"| {_pct(row['real_recall'])} "
            f"| {_pct(row['generated_precision'])} |"
        )
    lines.extend(
        [
            "",
            "## 口径备注",
            "",
            "- 线上双尺度模型仍是 **24f@1k + 8f@2k**，与本表单尺度网格分开看",
            "- 原始 JSON：" + str(summary.get("output_json", "")),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Wang Xing video ablation: 8/24/32 x 1k/2k."
    )
    parser.add_argument("--real-root", default="data/MD_CL")
    parser.add_argument("--fake-root", default="data/WangXing_Seedance")
    parser.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
    )
    parser.add_argument("--real-train-count", type=int, default=120)
    parser.add_argument(
        "--manifest",
        default="outputs/vedio_pred/wangxing_dual_pt_split.json",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[8, 24, 32],
        help="Frame counts (default: 8 24 32).",
    )
    parser.add_argument(
        "--frame-sizes",
        type=int,
        nargs="+",
        default=[1024, 2048],
        help="Resolutions: 1024=1k, 2048=2k (default: both).",
    )
    parser.add_argument("--cache-dir", default="outputs/vedio_pred/cache")
    parser.add_argument("--model-dir", default="outputs/vedio_pred/models")
    parser.add_argument(
        "--output",
        default="outputs/vedio_pred/wangxing_frame_res_ablation_8_24_32_1k_2k.json",
    )
    parser.add_argument(
        "--report-md",
        default="outputs/vedio_pred/wangxing_frame_res_ablation_8_24_32_1k_2k.md",
        help="Markdown table for pasting into reports.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse already-trained single-scale .pt if present.",
    )
    args = parser.parse_args(argv)

    manifest_path = project_path(args.manifest)
    if not manifest_path.is_file():
        print(f"Building split manifest: {manifest_path}", flush=True)
        manifest = build_wangxing_split_manifest(
            project_root=PROJECT_ROOT,
            real_root=project_path(args.real_root),
            fake_root=project_path(args.fake_root),
            holdout_manifest=project_path(args.holdout_manifest),
            real_train_count=args.real_train_count,
            seed=args.seed,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        manifest = _load_json(manifest_path)

    grid = [
        (int(num_frames), int(frame_size))
        for frame_size in args.frame_sizes
        for num_frames in args.frames
    ]
    print("Split counts:", json.dumps(manifest.get("counts"), ensure_ascii=False))
    print(
        f"Grid ({len(grid)} runs): "
        + ", ".join(f"{f}f@{_res_label(s)}" for f, s in grid),
        flush=True,
    )

    runs: list[dict[str, Any]] = []
    for index, (num_frames, frame_size) in enumerate(grid, start=1):
        print(
            f"\n######## [{index}/{len(grid)}] "
            f"{num_frames}f @ {_res_label(frame_size)} ########",
            flush=True,
        )
        result = train_and_eval_single_scale(
            manifest=manifest,
            num_frames=num_frames,
            frame_size=frame_size,
            cache_dir=project_path(args.cache_dir),
            model_dir=project_path(args.model_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            skip_existing=bool(args.skip_existing),
        )
        runs.append(result)

    table = [
        {
            "num_frames": run["num_frames"],
            "frame_size": run["frame_size"],
            "resolution": _res_label(run["frame_size"]),
            "generated_recall": run["headline"]["generated_recall"],
            "overall_accuracy": run["headline"]["overall_accuracy"],
            "real_recall": run["headline"].get("real_recall"),
            "generated_precision": run["headline"].get("generated_precision"),
        }
        for run in runs
    ]
    output = project_path(args.output)
    report_md = project_path(args.report_md)
    summary = {
        "schema_version": "wangxing_frame_res_ablation_v1",
        "manifest": str(manifest_path),
        "output_json": str(output),
        "report_md": str(report_md),
        "grid": {
            "frames": [int(x) for x in args.frames],
            "frame_sizes": [int(x) for x in args.frame_sizes],
        },
        "protocol": {
            "train_real": "even 120 non-holdout",
            "train_fake": "non-holdout Seedance",
            "test": "forensics holdout 50+50",
            "branch": "single-scale video .pt only (not AU fusion)",
            "metric_definitions": {
                "generated_recall": "TP/(TP+FN) on generated",
                "overall_accuracy": "(TP+TN)/N on real+generated",
            },
        },
        "runs": runs,
        "table": table,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report_md(summary, report_md)

    print("\n======== 报告对比表 ========")
    print(
        f"{'配置':>12}  {'生成召回':>10}  {'整体准确率':>10}  {'真拍召回':>10}"
    )
    for row in table:
        label = f"{row['num_frames']}f@{row['resolution']}"
        print(
            f"{label:>12}  "
            f"{_pct(row['generated_recall']):>10}  "
            f"{_pct(row['overall_accuracy']):>10}  "
            f"{_pct(row['real_recall']):>10}"
        )
    print(f"\nWrote JSON: {output}")
    print(f"Wrote MD:   {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

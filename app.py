from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import pandas as pd

from evaluator.holistic_evaluator import WEIGHTS, evaluate_all
from evaluator.runtime import OUTPUT_DIR
from evaluator.video_metrics import is_video_path, probe_video, resolve_path
from evaluator.vbench_runner import VBENCH_DIMENSIONS, run_vbench_ui


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _video_component(label: str) -> Any:
    kwargs = {"label": label, "format": "mp4"}
    try:
        return gr.Video(sources=["upload"], type="filepath", **kwargs)
    except TypeError:
        return gr.Video(**kwargs)


def _file_component(label: str, file_types: list[str]) -> Any:
    try:
        return gr.File(
            label=label,
            file_types=file_types,
            type="filepath",
        )
    except TypeError:
        return gr.File(label=label, file_types=file_types)


def _reference_summary(
    reference_image: Any,
    reference_video: Any,
    gt_video: Any,
) -> list[str]:
    messages: list[str] = []
    image_path = resolve_path(reference_image)
    video_path = resolve_path(reference_video)
    gt_path = resolve_path(gt_video)
    if image_path:
        image = cv2.imread(image_path)
        if image is not None:
            height, width = image.shape[:2]
            messages.append(f"参考图：{Path(image_path).name}（{width} × {height}）")
        else:
            messages.append(f"参考图：读取失败（{Path(image_path).name}）")
    if video_path:
        try:
            info = probe_video(video_path)
            messages.append(
                f"参考视频：{Path(video_path).name}（{info.width} × {info.height}，"
                f"{info.fps:.3f} FPS，{info.duration_seconds:.3f}s）"
            )
        except Exception as exc:
            messages.append(f"参考视频：读取失败（{exc}）")
    if gt_path:
        messages.append(
            f"GT 视频：{Path(gt_path).name}（第 2 类 PSNR / SSIM / LPIPS，"
            "并可作为其他类别的同步参考）"
        )
    if not messages:
        messages.append(
            "未上传额外参考素材；身份指标可能不可用，"
            "第 2 类全参考指标不可用，表情和美学使用人工评分。"
        )
    return messages


def _build_summary(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(result["summary"])


def _build_frame_table(result: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "sample_index",
        "result_frame",
        "gt_frame",
        "transition_index",
        "from_frame",
        "to_frame",
        "timestamp_seconds",
        "face_found",
        "identity_backend",
        "identity_similarity",
        "expression_motion_similarity",
        "text_video_similarity",
        "psnr_db",
        "ssim",
        "lpips",
        "high_frequency_ratio",
        "warping_error",
        "reference_flow_endpoint_error",
        "face_bbox",
    ]
    records = result.get("frame_records", [])
    if not records:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(records)
    for column in columns:
        if column not in table:
            table[column] = None
    table = table[columns]
    if "face_bbox" in table:
        table["face_bbox"] = table["face_bbox"].apply(
            lambda value: json.dumps(value, ensure_ascii=False)
            if isinstance(value, list)
            else value
        )
    return table


def _format_status(
    result: dict[str, Any],
    reference_messages: list[str],
) -> str:
    complete = result["status"] == "complete"
    title = "### 五类评估完成" if complete else "### 五类评估完成（部分指标不可用）"
    lines = [
        title,
        "",
        f"覆盖情况：**{result['coverage']}**，权重合计 100%（"
        + " / ".join(f"{key} {value}%" for key, value in WEIGHTS.items())
        + "）。",
        f"评估模式：**{result.get('evaluation_mode', 'auto')}**。",
        (
            f"加权分数：**{result['weighted_score_0_100']:.2f}/100**，"
            f"参与权重 {result['weighted_score_weight_coverage']}%。"
            if result["weighted_score_0_100"] is not None
            else "加权分数：不可用。"
        ),
        "",
        "**输入与评估路径**",
    ]
    if result.get("prompt_text"):
        lines.append(f"- 文本 Prompt：{result['prompt_text']}")
    lines.extend(f"- {message}" for message in reference_messages)
    lines.extend(
        [
            "",
            "**逻辑说明**",
            "- 第 2 类只有上传逐帧对应的 GT 视频时才计算 PSNR、SSIM、LPIPS；普通参考图/参考视频不替代 GT。",
            "- 参考图用于角色外观和身份一致性；参考视频用于表情、动作和时间稳定性。",
            "- 无 GT 但有 Prompt 时，第 3 类加入文本-视频语义对齐；没有可用参考时回退人工 1~5 分。",
            "- 第 4 类优先比较参考视频/GT 的运动，否则计算结果视频自身的 warping error。",
            "- 第 5 类以人工美学评分为主，技术质量代理仅作辅助。",
        ]
    )
    warnings = result.get("warnings", [])
    if warnings:
        lines.extend(["", "**注意事项**"])
        lines.extend(f"- {warning}" for warning in warnings[:12])
    return "\n".join(lines)


def evaluate(
    result_video: Any,
    prompt_text: str | None,
    gt_video: Any,
    reference_image: Any,
    reference_video: Any,
    max_frames: int,
    calculate_lpips: bool,
    device: str,
    manual_expression_score: float | None,
    manual_aesthetic_score: float | None,
) -> tuple[str, pd.DataFrame, pd.DataFrame, str | None, str]:
    result_path = resolve_path(result_video)
    gt_path = resolve_path(gt_video)
    image_path = resolve_path(reference_image)
    reference_path = resolve_path(reference_video)
    reference_messages = _reference_summary(reference_image, reference_video, gt_video)

    if not result_path:
        empty = pd.DataFrame()
        return (
            "### 请输入生成结果视频\n\n结果视频是必填项。",
            empty,
            empty,
            None,
            json.dumps({"error": "result_video is required"}, ensure_ascii=False, indent=2),
        )
    if not is_video_path(result_path):
        empty = pd.DataFrame()
        return (
            "### 结果视频格式不支持\n\n请上传 MP4、MOV、AVI、MKV、WEBM 或 M4V。",
            empty,
            empty,
            None,
            json.dumps({"error": "unsupported result video format"}, ensure_ascii=False, indent=2),
        )
    if gt_path and not is_video_path(gt_path):
        empty = pd.DataFrame()
        return (
            "### GT 文件格式不支持\n\nGT 必须是视频文件。",
            empty,
            empty,
            None,
            json.dumps({"error": "unsupported GT video format"}, ensure_ascii=False, indent=2),
        )
    if reference_path and not is_video_path(reference_path):
        empty = pd.DataFrame()
        return (
            "### 参考视频格式不支持\n\n参考视频必须是视频文件。",
            empty,
            empty,
            None,
            json.dumps({"error": "unsupported reference video format"}, ensure_ascii=False, indent=2),
        )

    try:
        result = evaluate_all(
            result_path=result_path,
            ground_truth=gt_path,
            reference_image=image_path,
            reference_video=reference_path,
            prompt_text=prompt_text,
            max_frames=int(max_frames),
            calculate_lpips=calculate_lpips,
            device=device,
            manual_expression_score=(
                float(manual_expression_score)
                if manual_expression_score is not None
                else None
            ),
            manual_aesthetic_score=(
                float(manual_aesthetic_score)
                if manual_aesthetic_score is not None
                else None
            ),
        )
    except Exception as exc:
        empty = pd.DataFrame()
        return (
            f"### 评估失败\n\n`{type(exc).__name__}: {exc}`",
            empty,
            empty,
            None,
            json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2),
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"holistic_metrics_{timestamp}.csv"
    frame_table = _build_frame_table(result)
    frame_table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    result["reference_material"] = reference_messages
    result["prompt_text"] = prompt_text
    result["result_video"] = probe_video(result_path).to_dict()
    if gt_path:
        result["ground_truth_video"] = probe_video(gt_path).to_dict()
    return (
        _format_status(result, reference_messages),
        _build_summary(result),
        frame_table,
        str(csv_path),
        json.dumps(result, ensure_ascii=False, indent=2),
    )

with gr.Blocks(title="视频生成模型结果评估器") as demo:
    gr.Markdown(
        """
# 视频生成模型结果评估器

首版支持“上传生成结果视频 + 上传参考素材”的评估流程。系统会根据素材类型
自动选择五类评估路径，并明确区分精确指标、代理指标和不可用指标。

**最小输入：** 生成结果视频。**推荐输入：** 结果视频 + Prompt + 参考图或参考视频。
如果有与结果逐帧对应的 GT 视频，请单独上传到 GT 输入框；只有 GT 才会启用第 2 类
的 PSNR、SSIM、LPIPS。
"""
    )

    with gr.Row():
        result_video = _video_component("生成结果视频（必填）")
        gt_video = _video_component(
            "GT 参考视频（可选：第 2 类 PSNR / SSIM / LPIPS）"
        )

    prompt_text = gr.Textbox(
        label="文本 Prompt（可选：用于文本-视频语义对齐）",
        placeholder="例如：人物自然地微笑并眨眼，镜头保持稳定",
        lines=2,
    )

    with gr.Row():
        reference_image = _file_component(
            "参考素材：参考图（可选：角色外观/身份基准）",
            [".png", ".jpg", ".jpeg", ".webp"],
        )
        reference_video = _video_component(
            "参考素材：参考动作视频（可选：表情/动作/时间稳定性）"
        )

    with gr.Row():
        manual_expression_score = gr.Slider(
            minimum=1,
            maximum=5,
            value=None,
            step=1,
            label="无可用参考视频时：表情/动作人工评分（1~5）",
        )
        manual_aesthetic_score = gr.Slider(
            minimum=1,
            maximum=5,
            value=None,
            step=1,
            label="美学人工评分（1~5）",
        )

    with gr.Row():
        max_frames = gr.Slider(
            minimum=2,
            maximum=256,
            value=64,
            step=1,
            label="最多采样帧数",
        )
        device = gr.Dropdown(
            choices=["cpu", "auto", "cuda"],
            value="auto",
            label="LPIPS / 可选模型推理设备",
        )
        calculate_lpips = gr.Checkbox(
            value=True,
            label="有 GT 时计算 LPIPS（第 2 类）",
        )

    gr.Markdown(
        """
**五类首版逻辑：**

1. 角色一致性（35%）：参考图/参考视频/GT 提供身份基准，ArcFace（可选）或人脸特征代理输出平均相似度、尾部 10% 相似度和方差。
2. 质感和细节（15%）：**仅有 GT 视频时**计算 PSNR、SSIM、LPIPS；无 GT 时使用 MANIQA/MUSIQ（可选）和高频细节代理，不能把普通参考素材当作 GT。
3. 表情准确（15%）：参考视频使用运动轨迹相似度，Prompt 使用逐帧 CLIP 语义对齐，二者都不可用时使用人工 1~5 分。
4. 时间稳定性（25%）：身份变化、landmark/人脸框抖动和 warping error；有参考视频或 GT 时额外比较参考光流。
5. 美学质量（10%）：人工 1~5 分为主，曝光、清晰度和色彩技术代理为辅助。
"""
    )

    evaluate_button = gr.Button("开始五类评估", variant="primary")
    status = gr.Markdown()

    with gr.Tab("五类汇总"):
        summary = gr.Dataframe(
            headers=["类别", "权重", "状态", "标准化分数", "核心结果", "后端"],
            datatype=["str", "str", "str", "str", "str", "str"],
            label="截图五类评估汇总",
            interactive=False,
        )

    with gr.Tab("逐帧明细"):
        frame_table = gr.Dataframe(
            label="逐帧 GT 指标（有 GT 时）/ 身份 / 运动 / 稳定性明细",
            interactive=False,
        )
        csv_file = gr.File(label="下载逐帧 CSV")

    with gr.Tab("原始 JSON"):
        raw_json = gr.Code(
            language="json",
            label="机器可读完整结果",
            interactive=False,
        )

    with gr.Tab("VBench（可选）"):
        gr.Markdown(
            """
VBench 作为额外后端，不替代五类主流程。首次运行前请安装
`requirements-vbench.txt`；部分官方维度在 Windows 上可能需要 Linux、WSL 或 Docker。
"""
        )
        vbench_dimensions = gr.CheckboxGroup(
            choices=VBENCH_DIMENSIONS,
            value=["motion_smoothness", "aesthetic_quality"],
            label="VBench 维度",
        )
        vbench_button = gr.Button("运行 VBench", variant="secondary")
        vbench_status = gr.Markdown()
        vbench_table = gr.Dataframe(
            headers=["dimension", "score", "direction", "source_file"],
            datatype=["str", "str", "str", "str"],
            label="VBench 结果",
            interactive=False,
        )
        vbench_json = gr.Code(
            language="json",
            label="VBench 原始 JSON",
            interactive=False,
        )

    evaluate_button.click(
        fn=evaluate,
        inputs=[
            result_video,
            prompt_text,
            gt_video,
            reference_image,
            reference_video,
            max_frames,
            calculate_lpips,
            device,
            manual_expression_score,
            manual_aesthetic_score,
        ],
        outputs=[status, summary, frame_table, csv_file, raw_json],
    )
    vbench_button.click(
        fn=lambda video, dimensions: run_vbench_ui(
            video,
            dimensions,
            OUTPUT_DIR,
        ),
        inputs=[result_video, vbench_dimensions],
        outputs=[vbench_status, vbench_table, vbench_json],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
    )

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .video_metrics import (
    SEMANTIC_WINDOW_FRAMES,
    _read_frames,
    _sample_indices,
    sample_aligned_video_windows,
    sample_video_windows,
    probe_video,
)


ETVA_URL = os.environ.get(
    "ETVA_JUDGE_URL",
    "http://127.0.0.1:30000/v1/chat/completions",
)
ETVA_MODEL = os.environ.get("ETVA_JUDGE_MODEL", "qwen2-vl-2b-awq")
try:
    ETVA_MAX_FRAME_DIMENSION = max(
        224,
        int(os.environ.get("ETVA_MAX_FRAME_DIMENSION", "768")),
    )
except ValueError:
    ETVA_MAX_FRAME_DIMENSION = 768


def _enabled() -> bool:
    return os.environ.get("ETVA_JUDGE_ENABLED", "auto").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def etva_service_available(timeout_seconds: float = 0.25) -> bool:
    """Check whether the external VLM is already occupying the GPU."""
    if not _enabled():
        return False
    models_url = ETVA_URL.replace(
        "/v1/chat/completions",
        "/v1/models",
    )
    try:
        with urllib.request.urlopen(models_url, timeout=timeout_seconds) as response:
            if not 200 <= int(response.status) < 300:
                return False
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("data", []) if isinstance(payload, dict) else []
        return any(
            isinstance(item, dict) and bool(item.get("id"))
            for item in models
        )
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError):
        return False


def _data_uri(frame: np.ndarray) -> str:
    height, width = frame.shape[:2]
    max_dimension = max(height, width)
    if max_dimension > ETVA_MAX_FRAME_DIMENSION:
        scale = ETVA_MAX_FRAME_DIMENSION / max_dimension
        frame = cv2.resize(
            frame,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 82],
    )
    if not ok:
        raise ValueError("Unable to encode a frame for the ETVA judge.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _sample_video_frames(
    path: str | Path,
    max_frames: int,
) -> list[np.ndarray]:
    info = probe_video(path)
    count = min(max_frames, info.frame_count)
    indices = _sample_indices(info.frame_count, count)
    return _read_frames(info.path, indices)


def _request(
    content: list[dict[str, Any]],
    timeout_seconds: float,
) -> str:
    payload = {
        "model": ETVA_MODEL,
        "temperature": 0.0,
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }
    request = urllib.request.Request(
        ETVA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        result = json.loads(response.read().decode("utf-8"))
    message = result["choices"][0]["message"]["content"]
    if isinstance(message, list):
        return "\n".join(
            str(item.get("text", item)) if isinstance(item, dict) else str(item)
            for item in message
        )
    return str(message)


def _parse_result(text: str) -> tuple[float | None, list[float]]:
    values: list[float] = []
    try:
        payload = json.loads(text)
        candidates = payload.get("scores", []) if isinstance(payload, dict) else []
        if isinstance(candidates, list):
            values = [
                float(value)
                for value in candidates
                if float(value) in {0.0, 0.5, 1.0}
            ]
        overall = payload.get("overall") if isinstance(payload, dict) else None
        if overall is not None and float(overall) in {0.0, 0.5, 1.0}:
            return float(overall), values
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

    values = [
        float(value)
        for value in re.findall(r"(?<![\d.])(0(?:\.5)?|1(?:\.0)?)(?![\d.])", text)
    ]
    values = [value for value in values if value in {0.0, 0.5, 1.0}]
    return (float(np.mean(values)) if values else None), values


def _feedback_items(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = [
            item.get("text", "")
            if isinstance(item, dict)
            else str(item)
            for item in value
        ]
    else:
        values = []
    return [
        item.strip()
        for item in values
        if isinstance(item, str) and item.strip()
    ]


def _parse_feedback(text: str) -> tuple[list[str], list[str]]:
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return [], []
    if not isinstance(payload, dict):
        return [], []
    problems = _feedback_items(
        payload.get("problems")
        or payload.get("issues")
        or payload.get("problem"),
    )
    suggestions = _feedback_items(
        payload.get("suggestions")
        or payload.get("recommendations")
        or payload.get("adjustments")
        or payload.get("suggestion"),
    )
    return problems[:3], suggestions[:3]


def _unique_feedback(items: list[str], limit: int = 6) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = " ".join(item.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized[:240])
        if len(unique) >= limit:
            break
    return unique


def _unavailable_result(
    reason: str,
    warnings: list[str] | None = None,
    service_active: bool | None = None,
    failure_kind: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "unavailable",
        "backend": "qwen2_vl_2b_awq_http",
        "model": ETVA_MODEL,
        "url": ETVA_URL,
        "score_0_1": None,
        "reason": reason,
        "warnings": warnings or [reason],
    }
    if service_active is not None:
        result["service_active"] = service_active
    if failure_kind:
        result["failure_kind"] = failure_kind
    return result


def _aggregate_window_scores(
    scores: list[float],
    tail_weight: float = 0.2,
) -> tuple[float | None, float | None]:
    if not scores:
        return None, None
    ordered = sorted(float(score) for score in scores)
    tail_count = max(1, int(np.ceil(len(ordered) * 0.1)))
    mean_score = float(np.mean(ordered))
    tail_score = float(np.mean(ordered[:tail_count]))
    weight = max(0.0, min(1.0, float(tail_weight)))
    return (
        float((1.0 - weight) * mean_score + weight * tail_score),
        tail_score,
    )


def evaluate_etva_judge(
    result_path: str | Path,
    prompt_text: str | None,
    reference_path: str | Path | None,
    max_frames: int,
    window_frames: int = SEMANTIC_WINDOW_FRAMES,
    service_available: bool | None = None,
) -> dict[str, Any]:
    """Ask the local OpenAI-compatible Qwen VLM to score semantic alignment."""
    prompt = (prompt_text or "").strip()
    if not _enabled():
        return {
            "status": "disabled",
            "backend": "qwen2_vl_2b_awq_http",
            "model": ETVA_MODEL,
            "url": ETVA_URL,
            "score_0_1": None,
            "reason": "ETVA_JUDGE_ENABLED disabled.",
            "warnings": [],
        }
    if service_available is False:
        return _unavailable_result(
            "Qwen weights are cached, but the ETVA Judge HTTP service is not connected. "
            f"Start the service and make sure {ETVA_URL} is reachable.",
            service_active=False,
            failure_kind="service_unavailable",
        )
    reference_only = os.environ.get(
        "ETVA_JUDGE_REFERENCE_ONLY",
        "0",
    ).lower() in {"1", "true", "yes", "on"}
    if not prompt and (not reference_path or not reference_only):
        return {
            "status": "unavailable",
            "backend": "qwen2_vl_2b_awq_http",
            "model": ETVA_MODEL,
            "url": ETVA_URL,
            "score_0_1": None,
            "reason": (
                "A prompt is required for ETVA judging unless "
                "ETVA_JUDGE_REFERENCE_ONLY is enabled."
            ),
            "warnings": [],
        }

    try:
        frame_count = max(1, min(int(window_frames), SEMANTIC_WINDOW_FRAMES))
        if reference_path:
            _, _, windows = sample_aligned_video_windows(
                result_path,
                reference_path,
                max_frames,
                window_frames=frame_count,
            )
        else:
            _, windows = sample_video_windows(
                result_path,
                max_frames,
                window_frames=frame_count,
            )

        window_scores: list[float] = []
        question_scores: list[float] = []
        raw_responses: list[str] = []
        window_records: list[dict[str, Any]] = []
        feedback_problems: list[str] = []
        feedback_suggestions: list[str] = []
        warnings: list[str] = []
        for window in windows:
            result_frames = window.get("result_frames", window.get("frames", []))
            reference_frames = window.get("reference_frames", [])
            content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": (
                        "You are an evaluation judge for a generated human video. "
                        "Each score must be exactly 0, 0.5, or 1. "
                        "Score whether the generated frames match the requested "
                        "prompt and preserve the intended expression/action. "
                        "Also inspect visible identity/appearance drift, facial "
                        "expression, motion timing, flicker/deformation, framing, "
                        "lighting, and sharpness. "
                        "Use 1 for clear match, 0.5 for partial/uncertain, and 0 "
                        "for mismatch. Also return up to 3 concise problems and "
                        "up to 3 practical suggestions in Simplified Chinese. "
                        "Only mention issues visible in the frames or supported "
                        "by the prompt/reference; return an empty list when there "
                        "is no clear issue. "
                        "Return exactly this JSON schema: "
                        '{"scores":[0,0.5,1],"overall":0.5,'
                        '"problems":[],"suggestions":[]}. '
                        f"Prompt: {prompt or '(compare against the reference video)'}"
                    ),
                },
                {
                    "type": "text",
                    "text": "Generated video frames follow in temporal order.",
                },
            ]
            content.extend(
                {"type": "image_url", "image_url": {"url": _data_uri(frame)}}
                for frame in result_frames
            )
            if reference_frames:
                content.append(
                    {
                        "type": "text",
                        "text": "Reference video frames follow. Compare expression and action.",
                    }
                )
                content.extend(
                    {"type": "image_url", "image_url": {"url": _data_uri(frame)}}
                    for frame in reference_frames
                )

            try:
                raw = _request(content, timeout_seconds=4.0)
                score, scores = _parse_result(raw)
                if score is None:
                    raise ValueError(
                        f"Could not parse a 0/0.5/1 score from: {raw[:500]}"
                    )
            except (
                OSError,
                ValueError,
                KeyError,
                json.JSONDecodeError,
                urllib.error.URLError,
            ) as exc:
                warnings.append(
                    f"ETVA window {window['window_index']} failed: {exc}"
                )
                continue

            normalized_score = float(max(0.0, min(1.0, score)))
            problems, suggestions = _parse_feedback(raw)
            window_scores.append(normalized_score)
            question_scores.extend(scores)
            raw_responses.append(raw)
            feedback_problems.extend(problems)
            feedback_suggestions.extend(suggestions)
            window_records.append(
                {
                    "window_index": int(window["window_index"]),
                    "window_start_seconds": window["start_seconds"],
                    "window_end_seconds": window["end_seconds"],
                    "score_0_1": normalized_score,
                    "question_scores": scores,
                    "problems": problems,
                    "suggestions": suggestions,
                }
            )

        aggregate_score, tail_score = _aggregate_window_scores(window_scores)
        if aggregate_score is None:
            raise ValueError("ETVA did not produce a usable score for any window.")
        return {
            "status": "available",
            "backend": "qwen2_vl_2b_awq_http",
            "model": ETVA_MODEL,
            "url": ETVA_URL,
            "score_0_1": aggregate_score,
            "question_scores": question_scores,
            "raw_response": raw_responses[-1] if raw_responses else None,
            "raw_responses": raw_responses,
            "metrics": {
                "window_mean_score": float(np.mean(window_scores)),
                "window_lower_10pct_score": tail_score,
                "window_count": len(window_scores),
                "requested_window_count": len(windows),
            },
            "window_records": window_records,
            "feedback": {
                "problems": _unique_feedback(feedback_problems),
                "suggestions": _unique_feedback(feedback_suggestions),
            },
            "warnings": warnings,
        }
    except (OSError, ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError) as exc:
        if service_available is True:
            reason = (
                "ETVA Judge service is connected, but it did not return a usable "
                f"result: {exc}"
            )
            failure_kind = "invalid_response"
        else:
            reason = f"ETVA judge is not reachable or returned invalid output: {exc}"
            failure_kind = "request_failed"
        return _unavailable_result(
            reason,
            [str(exc)],
            service_active=service_available,
            failure_kind=failure_kind,
        )

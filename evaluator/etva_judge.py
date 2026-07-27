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

from .video_metrics import _read_frames, _sample_indices, probe_video


ETVA_URL = os.environ.get(
    "ETVA_JUDGE_URL",
    "http://127.0.0.1:30000/v1/chat/completions",
)
ETVA_MODEL = os.environ.get("ETVA_JUDGE_MODEL", "qwen2-vl-2b-awq")


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
            return 200 <= int(response.status) < 300
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _data_uri(frame: np.ndarray) -> str:
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


def evaluate_etva_judge(
    result_path: str | Path,
    prompt_text: str | None,
    reference_path: str | Path | None,
    max_frames: int,
) -> dict[str, Any]:
    """Ask the local OpenAI-compatible Qwen VLM to score semantic alignment."""
    prompt = (prompt_text or "").strip()
    if not _enabled():
        return {
            "status": "disabled",
            "backend": "qwen2_vl_2b_awq_http",
            "score_0_1": None,
            "reason": "ETVA_JUDGE_ENABLED disabled.",
            "warnings": [],
        }
    reference_only = os.environ.get(
        "ETVA_JUDGE_REFERENCE_ONLY",
        "0",
    ).lower() in {"1", "true", "yes", "on"}
    if not prompt and (not reference_path or not reference_only):
        return {
            "status": "unavailable",
            "backend": "qwen2_vl_2b_awq_http",
            "score_0_1": None,
            "reason": (
                "A prompt is required for ETVA judging unless "
                "ETVA_JUDGE_REFERENCE_ONLY is enabled."
            ),
            "warnings": [],
        }

    try:
        result_frames = _sample_video_frames(result_path, min(max_frames, 8))
        reference_frames = (
            _sample_video_frames(reference_path, min(max_frames, 8))
            if reference_path
            else []
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are an evaluation judge for a generated human video. "
                    "Return JSON only with this schema: "
                    '{"scores":[0,0.5,1],"overall":0.5}. '
                    "Each score must be exactly 0, 0.5, or 1. "
                    "Score whether the generated frames match the requested "
                    "prompt and preserve the intended expression/action. "
                    "Use 1 for clear match, 0.5 for partial/uncertain, and 0 "
                    "for mismatch. "
                    f"Prompt: {prompt or '(compare against the reference video)'}"
                ),
            }
        ]
        if result_frames:
            content.append(
                {
                    "type": "text",
                    "text": "Generated video frames follow in temporal order.",
                }
            )
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

        raw = _request(content, timeout_seconds=4.0)
        score, scores = _parse_result(raw)
        if score is None:
            raise ValueError(f"Could not parse a 0/0.5/1 score from: {raw[:500]}")
        return {
            "status": "available",
            "backend": "qwen2_vl_2b_awq_http",
            "model": ETVA_MODEL,
            "url": ETVA_URL,
            "score_0_1": float(max(0.0, min(1.0, score))),
            "question_scores": scores,
            "raw_response": raw,
            "warnings": [],
        }
    except (OSError, ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError) as exc:
        return {
            "status": "unavailable",
            "backend": "qwen2_vl_2b_awq_http",
            "score_0_1": None,
            "reason": f"ETVA judge is not reachable or returned invalid output: {exc}",
            "warnings": [str(exc)],
        }

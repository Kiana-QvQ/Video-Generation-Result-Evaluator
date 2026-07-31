from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException


class LocalQwenJudge:
    def __init__(self, model_path: Path, served_model_name: str) -> None:
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Local Qwen Judge dependencies are missing. "
                "Install requirements/vlm_local.txt first."
            ) from exc

        self._torch = torch
        self._process_vision_info = process_vision_info
        self.served_model_name = served_model_name
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype="auto",
            device_map="auto",
        ).eval()

    @staticmethod
    def _image_url_to_qwen_image(url: str) -> str:
        if url.startswith("data:"):
            header, encoded = url.split(",", 1)
            if ";base64" not in header:
                raise ValueError("Only base64 data URI images are supported.")
            image_bytes = base64.b64decode(encoded)
            # qwen-vl-utils accepts a data URI and avoids writing user uploads
            # into a persistent directory.
            mime = header[5:].split(";", 1)[0] or "image/jpeg"
            encoded = base64.b64encode(image_bytes).decode("ascii")
            return f"data:{mime};base64,{encoded}"
        return url

    def _messages(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list.")

        converted: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be an object.")
            content = message.get("content", "")
            if isinstance(content, str):
                converted.append(
                    {
                        "role": str(message.get("role", "user")),
                        "content": content,
                    }
                )
                continue
            if not isinstance(content, list):
                raise ValueError("Message content must be text or a list.")

            items: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    items.append(
                        {
                            "type": "text",
                            "text": str(item.get("text", "")),
                        }
                    )
                elif item_type == "image_url":
                    image_url = item.get("image_url", {})
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url", "")
                    if image_url:
                        items.append(
                            {
                                "type": "image",
                                "image": self._image_url_to_qwen_image(
                                    str(image_url)
                                ),
                            }
                        )
            converted.append(
                {
                    "role": str(message.get("role", "user")),
                    "content": items,
                }
            )
        return converted

    def complete(self, payload: dict[str, Any]) -> str:
        messages = self._messages(payload)
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        max_new_tokens = min(
            max(16, int(payload.get("max_tokens", 256))),
            512,
        )
        with self._torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            generated_ids = [
                output_ids[len(input_ids[index]) :]
                for index, output_ids in enumerate(generated_ids)
            ]
        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""


app = FastAPI(title="Local Qwen Judge")
_judge: LocalQwenJudge | None = None
_served_model_name = ""


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok" if _judge is not None else "starting",
        "model": _served_model_name,
    }


@app.get("/v1/models")
def models() -> dict[str, Any]:
    if _judge is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    return {
        "object": "list",
        "data": [
            {
                "id": _judge.served_model_name,
                "object": "model",
                "owned_by": "local-transformers",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
    if _judge is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    try:
        content = _judge.complete(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "id": f"local-qwen-{time.time_ns()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _judge.served_model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def main() -> None:
    global _judge, _served_model_name
    parser = argparse.ArgumentParser(description="Run a local Qwen2-VL Judge.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Qwen model directory does not exist: {model_path}")
    _served_model_name = args.served_model_name
    _judge = LocalQwenJudge(model_path, args.served_model_name)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

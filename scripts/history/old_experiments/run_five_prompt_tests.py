from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "tests" / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

REFERENCE_IMAGES = ["front.png", "Left.png", "Right.png"]

TEST_CASES = [
    {
        "case_id": "01_jieguo_prompt_match",
        "label": "短发人物：匹配大笑 prompt",
        "result_video": "ltx2.3+关键的iclora+JieGuo.mp4",
        "gt_video": "ltx2.3+关键的iclora+wx不上班.mp4",
        "prompt_text": (
            "Close-up portrait of the same young East Asian man with short black hair. "
            "He starts with a neutral face and gradually makes a broad joyful smile "
            "with visible teeth. Front-facing, centered framing, stable head, black "
            "background, natural facial motion."
        ),
        "calculate_lpips": "true",
    },
    {
        "case_id": "02_jieguo_prompt_conflict",
        "label": "短发人物：与画面冲突的中性 prompt",
        "result_video": "ltx2.3+关键的iclora+JieGuo.mp4",
        "gt_video": "ltx2.3+关键的iclora+wx不上班.mp4",
        "prompt_text": (
            "Close-up portrait of the same young East Asian man with short black hair. "
            "Keep a completely neutral expression throughout: lips closed, no smile, "
            "no visible teeth, no eyebrow movement, stable head, centered framing, "
            "black background."
        ),
        "calculate_lpips": "true",
    },
    {
        "case_id": "03_wx_prompt_match",
        "label": "短发人物第二段：匹配逐渐大笑 prompt",
        "result_video": "ltx2.3+关键的iclora+wx不上班.mp4",
        "gt_video": "",
        "prompt_text": (
            "A front-facing close-up of a young East Asian man with short black hair "
            "changing from a calm neutral face to a broad happy smile with teeth. "
            "Keep the face centered, the head stable, and the black background unchanged."
        ),
        "calculate_lpips": "false",
    },
    {
        "case_id": "04_bald_prompt_match",
        "label": "光头人物：匹配中性 prompt",
        "result_video": "_root_ai-toolkit_output_LTX2_samples_1782210089339__000001250_0.mp4",
        "gt_video": "",
        "prompt_text": (
            "A bald adult man in a dark studio, front-facing with a neutral expression "
            "and closed lips. Subtle breathing only, stable head, unchanged lighting, "
            "centered portrait framing."
        ),
        "calculate_lpips": "false",
    },
    {
        "case_id": "05_bald_prompt_conflict",
        "label": "光头人物：与画面冲突的大笑 prompt",
        "result_video": "_root_ai-toolkit_output_LTX2_samples_1782217880727__000002000_0.mp4",
        "gt_video": "",
        "prompt_text": (
            "A bald adult man in a dark studio gradually makes a broad joyful smile "
            "with clearly visible teeth. Keep the head stable, front-facing, and the "
            "lighting unchanged throughout."
        ),
        "calculate_lpips": "false",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit and summarize five prompt/video evaluation jobs."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for all jobs and write the final report.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory relative to the project root.",
    )
    parser.add_argument(
        "--resume-manifest",
        default="",
        help="Resume polling an existing manifest without submitting new jobs.",
    )
    return parser.parse_args()


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _assert_local_files(case: dict[str, Any]) -> None:
    for key in ("result_video", "gt_video"):
        filename = case.get(key)
        if filename and not (DATA_ROOT / filename).is_file():
            raise FileNotFoundError(f"Missing {key}: {DATA_ROOT / filename}")
    for filename in REFERENCE_IMAGES:
        if not (DATA_ROOT / filename).is_file():
            raise FileNotFoundError(f"Missing reference image: {DATA_ROOT / filename}")


def _upload_job(base_url: str, case: dict[str, Any], max_frames: int) -> dict[str, Any]:
    _assert_local_files(case)
    opened: list[Any] = []
    files: list[tuple[str, tuple[str, Any, str]]] = []

    def add_file(field: str, filename: str) -> None:
        path = DATA_ROOT / filename
        handle = path.open("rb")
        opened.append(handle)
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        files.append((field, (filename, handle, content_type)))

    try:
        add_file("result_video", str(case["result_video"]))
        if case.get("gt_video"):
            add_file("gt_video", str(case["gt_video"]))
        for filename in REFERENCE_IMAGES:
            add_file("reference_images", filename)

        response = requests.post(
            _url(base_url, "/api/jobs"),
            files=files,
            data={
                "prompt_text": case["prompt_text"],
                "max_frames": str(max_frames),
                "calculate_lpips": str(case["calculate_lpips"]).lower(),
                "device": "auto",
            },
            timeout=180,
        )
        if response.status_code != 202:
            raise RuntimeError(
                f"Submitting {case['case_id']} failed with "
                f"{response.status_code}: {response.text[:1000]}"
            )
        payload = response.json()
        return {
            **case,
            "job_id": payload["job_id"],
            "submitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "queue_position": payload.get("queue_position"),
        }
    finally:
        for handle in opened:
            handle.close()


def _read_json(response: requests.Response, label: str) -> dict[str, Any]:
    if response.status_code != 200:
        raise RuntimeError(f"Reading {label} failed: {response.status_code} {response.text[:500]}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Reading {label} returned a non-object payload.")
    return payload


def _fetch_job(base_url: str, job_id: str) -> dict[str, Any]:
    return _read_json(
        requests.get(_url(base_url, f"/api/jobs/{job_id}"), timeout=60),
        f"job {job_id}",
    )


def _fetch_result(base_url: str, job_id: str) -> dict[str, Any]:
    response = requests.get(
        _url(base_url, f"/api/runs/{job_id}/result.json"),
        timeout=120,
    )
    return _read_json(response, f"result {job_id}")


def _score(result: dict[str, Any], category: str) -> Any:
    return result.get("categories", {}).get(category, {}).get("score_0_1")


def _text_alignment(result: dict[str, Any]) -> Any:
    return (
        result.get("categories", {})
        .get("expression", {})
        .get("metrics", {})
        .get("text_video_alignment")
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_report(output_dir: Path, submitted: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    rows: list[str] = [
        "# Five Prompt/Video Evaluation Test",
        "",
        f"Created: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "This batch intentionally compares matching and conflicting prompts, "
        "and uses the short-haired person's reference images for every case.",
        "",
        "| Case | Video | Prompt intent | Status | Overall | Identity | Texture | Expression | Text-video | Temporal | Aesthetics |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item, result in zip(submitted, results):
        overall = result.get("weighted_score_0_100")
        row = [
            item["case_id"],
            item["result_video"],
            item["label"],
            result.get("status", "unknown"),
            overall,
            _score(result, "identity"),
            _score(result, "texture"),
            _score(result, "expression"),
            _text_alignment(result),
            _score(result, "temporal"),
            _score(result, "aesthetics"),
        ]
        rows.append(
            "| "
            + " | ".join(
                "—" if value is None else str(round(value, 4)) if isinstance(value, float) else str(value)
                for value in row
            )
            + " |"
        )
    rows.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Cases 01 and 02 use the same video with a matching versus conflicting prompt; the text-video alignment and expression scores should separate them if prompt scoring is sensitive.",
            "- Cases 04 and 05 use neutral bald-person footage with a matching versus conflicting prompt; this is a second prompt-sensitivity check.",
            "- Because all cases use `front.png`, `Left.png`, and `Right.png` as identity references, the bald-person cases should also show an identity penalty relative to the short-haired cases.",
            "- Scores are normalized to 0-1 except the overall score, which is 0-100.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(rows), encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    if args.resume_manifest:
        manifest_path = Path(args.resume_manifest)
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path
        manifest_path = manifest_path.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        submitted = list(manifest.get("cases", []))
        if len(submitted) != len(TEST_CASES):
            raise ValueError(
                f"Expected {len(TEST_CASES)} cases in manifest, got {len(submitted)}."
            )
        output_dir = manifest_path.parent
    else:
        output_dir = (
            PROJECT_ROOT / args.output_dir
            if args.output_dir
            else OUTPUT_ROOT
            / f"five_prompt_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        submitted = []
    output_dir.mkdir(parents=True, exist_ok=True)

    health = _read_json(
        requests.get(_url(base_url, "/api/health"), timeout=60),
        "health",
    )
    if health.get("status") != "ok":
        raise RuntimeError(f"Service is not healthy: {health}")

    if not args.resume_manifest:
        for case in TEST_CASES:
            item = _upload_job(base_url, case, args.max_frames)
            submitted.append(item)
            print(
                f"submitted {item['case_id']} -> {item['job_id']} "
                f"(queue position {item.get('queue_position')})",
                flush=True,
            )

        _write_json(
            output_dir / "manifest.json",
            {
                "base_url": base_url,
                "max_frames": args.max_frames,
                "reference_images": REFERENCE_IMAGES,
                "submitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "cases": submitted,
            },
        )

    if not args.wait:
        print(f"submitted {len(submitted)} jobs; manifest={output_dir / 'manifest.json'}")
        return 0

    terminal = {"completed", "failed", "canceled"}
    latest: dict[str, dict[str, Any]] = {}
    while True:
        all_terminal = True
        statuses: list[str] = []
        for item in submitted:
            job = _fetch_job(base_url, item["job_id"])
            latest[item["job_id"]] = job
            statuses.append(f"{item['case_id']}={job.get('status')}")
            if job.get("status") not in terminal:
                all_terminal = False
        print(" | ".join(statuses), flush=True)
        if all_terminal:
            break
        time.sleep(max(args.poll_seconds, 5))

    _write_json(output_dir / "jobs.json", list(latest.values()))
    results: list[dict[str, Any]] = []
    for item in submitted:
        job = latest[item["job_id"]]
        if job.get("status") == "completed":
            result = _fetch_result(base_url, item["job_id"])
        else:
            result = {
                "status": job.get("status"),
                "error": job.get("error"),
                "weighted_score_0_100": None,
                "categories": {},
            }
        results.append(result)
        _write_json(output_dir / f"{item['case_id']}.json", result)
    _write_report(output_dir, submitted, results)
    print(f"completed batch; report={output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

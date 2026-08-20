from __future__ import annotations

import argparse
import json
import mimetypes
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:7860"
TERMINAL_STATUSES = {"completed", "failed", "canceled"}


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip()


def _case(
    case_id: str,
    cohort: str,
    result_video: str,
    *,
    gt_video: str | None = None,
    reference_video: str | None = None,
    reference_images: tuple[str, ...],
    prompt_file: str | None = None,
    complete_reference: bool,
    note: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "cohort": cohort,
        "result_video": result_video,
        "gt_video": gt_video,
        "reference_video": reference_video,
        "reference_images": reference_images,
        "prompt_file": prompt_file,
        "complete_reference": complete_reference,
        "note": note,
    }


def build_cases() -> list[dict[str, Any]]:
    three_views = ("front.png", "Left.png", "Right.png")
    test1_views = tuple(f"tests/test1/{name}" for name in three_views)
    test2_views = tuple(
        f"tests/test2/{name}"
        for name in (
            "front.png",
            "Left.png",
            "Right.png",
            "BS01.jpg",
            "BS02.jpg",
            "BS03.jpg",
            "BS04.jpg",
        )
    )
    test3_views = tuple(
        f"tests/test3/{name}"
        for name in (
            "7816c83b0ce01fc33d8066f0b369fc5d.jpg",
            "78aba561130d1fce71a797691c70ddc1.jpg",
            "bf985be31be4f29be89c29325d717570.jpg",
        )
    )
    test4_views = tuple(f"tests/test4/{name}" for name in three_views)
    real_views = tuple(f"tests/test1/{name}" for name in three_views)

    cases = [
        _case(
            "seedance_test1",
            "seedance",
            "tests/test1/生成.mp4",
            gt_video="tests/test1/gt.mp4",
            reference_video="tests/test1/gt.mp4",
            reference_images=test1_views,
            prompt_file="tests/test1/prompt.txt",
            complete_reference=True,
            note="Seedance result with the supplied GT/action video.",
        ),
        _case(
            "seedance_test2",
            "seedance",
            "tests/test2/BaiJunZhiZhong_Jieguo.mp4",
            gt_video="tests/test2/GT和表情.mp4",
            reference_video="tests/test2/GT和表情.mp4",
            reference_images=test2_views,
            prompt_file="tests/test2/prompt.txt",
            complete_reference=True,
            note="Seedance result using the GT/expression driver video.",
        ),
        _case(
            "seedance_test2_bs",
            "seedance",
            "tests/test2/BaiJunZhiZhong_Jieguo+BS.mp4",
            gt_video="tests/test2/GT和表情.mp4",
            reference_video="tests/test2/GT和表情.mp4",
            reference_images=test2_views,
            prompt_file="tests/test2/prompt.txt",
            complete_reference=True,
            note="Second Seedance result from the same GT/action setup.",
        ),
        _case(
            "seedance_test3",
            "seedance_incomplete_reference",
            "tests/test3/pianduan05.mp4",
            reference_images=test3_views,
            prompt_file="tests/test3/prompt.txt",
            complete_reference=False,
            note="No GT or reference action video is present; result-only control.",
        ),
        _case(
            "seedance_test4",
            "seedance_action_only",
            "tests/test4/pianduan07.mp4",
            reference_video="tests/test4/输入pianduan04_7.mp4",
            reference_images=test4_views,
            prompt_file="tests/test4/prompt.txt",
            complete_reference=False,
            note="Action reference is present, but no pixel-aligned GT is present.",
        ),
        _case(
            "real_xiao_control",
            "real_control",
            "data/video/Xiao/124071016307_clip0001.mp4",
            gt_video="data/video/Xiao/124071016307_clip0001.mp4",
            reference_video="data/video/Xiao/124071016307_clip0001.mp4",
            reference_images=real_views,
            complete_reference=True,
            note="Real Wang Xing clip self-paired as the upper-bound control.",
        ),
        _case(
            "real_fennu_control",
            "real_control",
            "data/video/FenNu/124071016307_clip0001.mp4",
            gt_video="data/video/FenNu/124071016307_clip0001.mp4",
            reference_video="data/video/FenNu/124071016307_clip0001.mp4",
            reference_images=real_views,
            complete_reference=True,
            note="Real Wang Xing anger clip self-paired as a control.",
        ),
        _case(
            "real_jingya_control",
            "real_control",
            "data/video/JingYa/124071016307_clip0001.mp4",
            gt_video="data/video/JingYa/124071016307_clip0001.mp4",
            reference_video="data/video/JingYa/124071016307_clip0001.mp4",
            reference_images=real_views,
            complete_reference=True,
            note="Real Wang Xing surprise clip self-paired as a control.",
        ),
        _case(
            "real_beishang_control",
            "real_control",
            "data/video/BeiShang2/124071016307_clip0001.mp4",
            gt_video="data/video/BeiShang2/124071016307_clip0001.mp4",
            reference_video="data/video/BeiShang2/124071016307_clip0001.mp4",
            reference_images=real_views,
            complete_reference=True,
            note="Real Wang Xing sadness clip self-paired as a control.",
        ),
    ]
    return cases


def _response_json(response: requests.Response, label: str) -> dict[str, Any]:
    if response.status_code not in {200, 202}:
        raise RuntimeError(
            f"{label} failed with {response.status_code}: "
            f"{response.text[:1000]}"
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} returned a non-object response.")
    return payload


def _validate_case(case: dict[str, Any]) -> None:
    required = [case["result_video"], *case["reference_images"]]
    for path in (case.get("gt_video"), case.get("reference_video")):
        if path:
            required.append(path)
    if case.get("prompt_file"):
        required.append(case["prompt_file"])
    for relative in required:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"{case['case_id']} input is missing: {path}"
            )


def submit_case(
    base_url: str,
    case: dict[str, Any],
    *,
    max_frames: int,
    device: str,
    calculate_lpips: bool,
    manual_aesthetic_score: float | None,
) -> dict[str, Any]:
    _validate_case(case)
    opened: list[Any] = []
    files: list[tuple[str, tuple[str, Any, str]]] = []

    def add_file(field: str, relative: str) -> None:
        path = PROJECT_ROOT / relative
        handle = path.open("rb")
        opened.append(handle)
        files.append(
            (
                field,
                (
                    path.name,
                    handle,
                    mimetypes.guess_type(path.name)[0]
                    or "application/octet-stream",
                ),
            )
        )

    try:
        add_file("result_video", case["result_video"])
        if case.get("gt_video"):
            add_file("gt_video", case["gt_video"])
        if case.get("reference_video"):
            add_file("reference_video", case["reference_video"])
        for image in case["reference_images"]:
            add_file("reference_images", image)

        prompt = (
            _read_text(PROJECT_ROOT / case["prompt_file"])
            if case.get("prompt_file")
            else ""
        )
        response = requests.post(
            _url(base_url, "/api/jobs"),
            files=files,
            data={
                "prompt_text": prompt,
                "max_frames": str(max_frames),
                "calculate_lpips": str(calculate_lpips).lower(),
                "device": device,
                "manual_expression_score": "",
                "manual_aesthetic_score": (
                    ""
                    if manual_aesthetic_score is None
                    else str(manual_aesthetic_score)
                ),
                "wangxing_au_enabled": "true",
                "wangxing_expected_class": "auto",
            },
            timeout=180,
        )
        payload = _response_json(response, f"submit {case['case_id']}")
        job_id = str(payload["job_id"])
        rename = requests.patch(
            _url(base_url, f"/api/jobs/{job_id}"),
            json={"name": f"reference-test/{case['case_id']}"},
            timeout=60,
        )
        _response_json(rename, f"rename {job_id}")
        return {
            **case,
            "job_id": job_id,
            "queue_position": payload.get("queue_position"),
            "submitted_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
        }
    finally:
        for handle in opened:
            handle.close()


def fetch_job(base_url: str, job_id: str) -> dict[str, Any]:
    return _get_json_with_retry(
        base_url,
        f"/api/jobs/{job_id}",
        f"job {job_id}",
        timeout=120,
    )


def fetch_result(base_url: str, job_id: str) -> dict[str, Any]:
    return _get_json_with_retry(
        base_url,
        f"/api/runs/{job_id}/result.json",
        f"result {job_id}",
        timeout=180,
    )


def _get_json_with_retry(
    base_url: str,
    path: str,
    label: str,
    *,
    timeout: int,
    attempts: int = 5,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return _response_json(
                requests.get(
                    _url(base_url, path),
                    timeout=timeout,
                ),
                label,
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(10, 2 + attempt * 2))
    assert last_error is not None
    raise last_error


def _score(result: dict[str, Any], category: str) -> Any:
    return result.get("categories", {}).get(category, {}).get("score_0_1")


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    au = result.get("wangxing_au", {})
    compliance = au.get("au_compliance", {})
    targeted = au.get("wangxing_targeted", {})
    fusion = au.get("fusion", {})
    return {
        "status": result.get("status"),
        "evaluation_mode": result.get("evaluation_mode"),
        "coverage": result.get("coverage"),
        "weighted_score_0_100": result.get("weighted_score_0_100"),
        "category_scores": {
            name: _score(result, name)
            for name in (
                "identity",
                "texture",
                "expression",
                "temporal",
                "aesthetics",
            )
        },
        "category_backends": {
            name: result.get("categories", {})
            .get(name, {})
            .get("backend")
            for name in (
                "identity",
                "texture",
                "expression",
                "temporal",
                "aesthetics",
            )
        },
        "texture_metrics": result.get("categories", {})
        .get("texture", {})
        .get("metrics", {}),
        "aesthetics_metrics": result.get("categories", {})
        .get("aesthetics", {})
        .get("metrics", {}),
        "au_status": au.get("status"),
        "au_selected_expression_class": compliance.get(
            "selected_expression_class"
        ),
        "au_personal_score_0_1": compliance.get(
            "personal_au_score_0_1"
        ),
        "au_driver_expression_score_0_1": compliance.get(
            "driver_expression_score_0_1"
        ),
        "au_driver_temporal_alignment_score_0_1": compliance.get(
            "driver_temporal_alignment_score_0_1"
        ),
        "au_leakage_risk_0_1": compliance.get(
            "driver_identity_leakage_risk_0_1"
        ),
        "au_evidence_quality_status": compliance.get(
            "evidence_quality_status"
        ),
        "au_evidence_confidence_0_1": compliance.get(
            "evidence_confidence_0_1"
        ),
        "au_wangxing_fit_score_0_1": targeted.get(
            "wangxing_expression_fit_score_0_1"
        ),
        "au_targeted_decision": targeted.get("decision"),
        "au_fusion_person_likeness_0_1": fusion.get(
            "person_likeness_score_0_1"
        ),
        "au_fusion_decision": fusion.get("decision"),
        "warnings": result.get("warnings", [])[:8],
    }


def _numeric(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def compare_groups(
    items: list[dict[str, Any]],
    left_cohort: str,
    right_cohort: str,
) -> dict[str, Any]:
    fields = (
        ("weighted_score_0_100", ("weighted_score_0_100",)),
        ("identity_score_0_1", ("category_scores", "identity")),
        ("texture_score_0_1", ("category_scores", "texture")),
        ("expression_score_0_1", ("category_scores", "expression")),
        ("temporal_score_0_1", ("category_scores", "temporal")),
        ("aesthetics_score_0_1", ("category_scores", "aesthetics")),
        ("au_personal_score_0_1", ("au_personal_score_0_1",)),
        ("au_leakage_risk_0_1", ("au_leakage_risk_0_1",)),
        ("au_wangxing_fit_score_0_1", ("au_wangxing_fit_score_0_1",)),
        (
            "au_evidence_confidence_0_1",
            ("au_evidence_confidence_0_1",),
        ),
    )
    comparison: dict[str, Any] = {
        "left_cohort": left_cohort,
        "right_cohort": right_cohort,
        "fields": {},
    }
    for name, path in fields:
        values: dict[str, list[float]] = {
            left_cohort: [],
            right_cohort: [],
        }
        for item in items:
            cohort = item["case"]["cohort"]
            if cohort not in values:
                continue
            value: Any = item.get("result", {})
            for part in path:
                value = value.get(part) if isinstance(value, dict) else None
            parsed = _numeric(value)
            if parsed is not None:
                values[cohort].append(parsed)
        left = values[left_cohort]
        right = values[right_cohort]
        if left and right:
            wins = sum(
                1.0 if left_value > right_value else 0.5
                if left_value == right_value
                else 0.0
                for left_value in left
                for right_value in right
            )
            auc = wins / (len(left) * len(right))
        else:
            auc = None
        comparison["fields"][name] = {
            "left_count": len(left),
            "right_count": len(right),
            "left_mean": statistics.mean(left) if left else None,
            "right_mean": statistics.mean(right) if right else None,
            "auc_left_higher": auc,
            "separation_auc": (
                max(auc, 1.0 - auc) if auc is not None else None
            ),
            "preferred_higher_cohort": (
                left_cohort
                if auc is not None and auc >= 0.5
                else right_cohort
                if auc is not None
                else None
            ),
        }
    return comparison


def write_outputs(
    output_dir: Path,
    manifest: dict[str, Any],
    items: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "reference_pair_queue_test_v1",
        "manifest": manifest,
        "items": items,
        "groups": summarize_groups(items),
        "comparisons": {
            "seedance_vs_real_control": compare_groups(
                items,
                "seedance",
                "real_control",
            ),
            "seedance_action_only_vs_real_control": compare_groups(
                items,
                "seedance_action_only",
                "real_control",
            ),
            "seedance_incomplete_reference_vs_real_control": compare_groups(
                items,
                "seedance_incomplete_reference",
                "real_control",
            ),
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = [
        "case_id,cohort,complete_reference,status,evaluation_mode,"
        "weighted_score_0_100,identity,texture,expression,temporal,"
        "aesthetics,au_personal,au_leakage,au_wangxing_fit,"
        "au_evidence_confidence,au_decision"
    ]
    for item in items:
        result = item.get("result", {})
        scores = result.get("category_scores", {})
        values = [
            item["case_id"],
            item["cohort"],
            item["complete_reference"],
            item.get("job", {}).get("status"),
            result.get("evaluation_mode"),
            result.get("weighted_score_0_100"),
            scores.get("identity"),
            scores.get("texture"),
            scores.get("expression"),
            scores.get("temporal"),
            scores.get("aesthetics"),
            result.get("au_personal_score_0_1"),
            result.get("au_leakage_risk_0_1"),
            result.get("au_wangxing_fit_score_0_1"),
            result.get("au_evidence_confidence_0_1"),
            result.get("au_targeted_decision"),
        ]
        rows.append(",".join("" if value is None else str(value) for value in values))
    (output_dir / "scores.csv").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8-sig",
    )


def summarize_groups(items: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "weighted_score_0_100",
        "identity",
        "texture",
        "expression",
        "temporal",
        "aesthetics",
        "au_personal",
        "au_leakage",
        "au_wangxing_fit",
        "au_evidence_confidence",
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(item["case"]["cohort"], []).append(item)

    result: dict[str, Any] = {}
    for cohort, cohort_items in groups.items():
        field_values: dict[str, list[float]] = {field: [] for field in fields}
        for item in cohort_items:
            compact = item.get("result", {})
            categories = compact.get("category_scores", {})
            raw_values = {
                "weighted_score_0_100": compact.get(
                    "weighted_score_0_100"
                ),
                "identity": categories.get("identity"),
                "texture": categories.get("texture"),
                "expression": categories.get("expression"),
                "temporal": categories.get("temporal"),
                "aesthetics": categories.get("aesthetics"),
                "au_personal": compact.get("au_personal_score_0_1"),
                "au_leakage": compact.get("au_leakage_risk_0_1"),
                "au_wangxing_fit": compact.get(
                    "au_wangxing_fit_score_0_1"
                ),
                "au_evidence_confidence": compact.get(
                    "au_evidence_confidence_0_1"
                ),
            }
            for field, value in raw_values.items():
                parsed = _numeric(value)
                if parsed is not None:
                    field_values[field].append(parsed)
        result[cohort] = {
            "sample_count": len(cohort_items),
            "completed_count": sum(
                1
                for item in cohort_items
                if item.get("job", {}).get("status") == "completed"
            ),
            "field_summary": {
                field: {
                    "count": len(values),
                    "mean": statistics.mean(values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                }
                for field, values in field_values.items()
            },
        }
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Queue paired GT/reference-action tests through the running "
            "web evaluator."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--calculate-lpips",
        action="store_true",
        help="Keep LPIPS enabled for a slower full-reference run.",
    )
    parser.add_argument(
        "--manual-aesthetic-score",
        type=float,
        default=3.0,
        help=(
            "Use a controlled manual aesthetic score to skip slow VBench. "
            "Set an empty value only by editing the script for full VBench."
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--resume-manifest", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--case",
        action="append",
        help="Submit only the named case; repeat for multiple cases.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_frames < 2:
        raise SystemExit("--max-frames must be at least 2.")
    if args.poll_seconds < 5:
        raise SystemExit("--poll-seconds must be at least 5.")

    all_cases = build_cases()
    if args.resume_manifest:
        manifest_path = Path(args.resume_manifest)
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path
        manifest_path = manifest_path.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        submitted = list(manifest["cases"])
        output_dir = manifest_path.parent
    else:
        selected = (
            [case for case in all_cases if case["case_id"] in args.case]
            if args.case
            else all_cases
        )
        selected_ids = {case["case_id"] for case in selected}
        unknown = set(args.case or ()) - selected_ids
        if unknown:
            raise SystemExit(f"Unknown cases: {', '.join(sorted(unknown))}")
        health = _response_json(
            requests.get(_url(args.base_url, "/api/health"), timeout=30),
            "health",
        )
        if health.get("status") != "ok":
            raise RuntimeError(f"Web service is not healthy: {health}")
        submitted = []
        for case in selected:
            item = submit_case(
                args.base_url,
                case,
                max_frames=args.max_frames,
                device=args.device,
                calculate_lpips=args.calculate_lpips,
                manual_aesthetic_score=args.manual_aesthetic_score,
            )
            submitted.append(item)
            print(
                f"submitted {item['case_id']} -> {item['job_id']} "
                f"(queue position {item.get('queue_position')})",
                flush=True,
            )
        output_dir = (
            PROJECT_ROOT / args.output_dir
            if args.output_dir
            else PROJECT_ROOT
            / "outputs"
            / f"reference_pair_queue_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        manifest = {
            "base_url": args.base_url,
            "max_frames": args.max_frames,
            "device": args.device,
            "calculate_lpips": args.calculate_lpips,
            "manual_aesthetic_score": args.manual_aesthetic_score,
            "wangxing_au_enabled": True,
            "wangxing_expected_class": "auto",
            "reference_fields": {
                "gt_video": "used when case has an aligned GT video",
                "reference_video": "used as motion/action reference",
            },
            "created_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "cases": submitted,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if not args.wait:
        print(
            f"submitted {len(submitted)} jobs; "
            f"manifest={output_dir / 'manifest.json'}"
        )
        return 0

    latest: dict[str, dict[str, Any]] = {}
    while True:
        all_terminal = True
        statuses: list[str] = []
        for item in submitted:
            job = fetch_job(args.base_url, item["job_id"])
            latest[item["job_id"]] = job
            statuses.append(
                f"{item['case_id']}={job.get('status')}/{job.get('stage')}"
            )
            if job.get("status") not in TERMINAL_STATUSES:
                all_terminal = False
        print(" | ".join(statuses), flush=True)
        if all_terminal:
            break
        time.sleep(args.poll_seconds)

    items: list[dict[str, Any]] = []
    for item in submitted:
        job = latest[item["job_id"]]
        if job.get("status") == "completed":
            result = compact_result(fetch_result(args.base_url, item["job_id"]))
        else:
            result = {
                "status": job.get("status"),
                "error": job.get("error"),
            }
        items.append({"case": item, "job": job, "result": result})
    write_outputs(output_dir, manifest, items)
    print(f"completed batch; results={output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

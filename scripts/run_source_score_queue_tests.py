from __future__ import annotations

import argparse
import json
import mimetypes
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:7860"
REFERENCE_IMAGES = (
    Path("tests/data/front.png"),
    Path("tests/data/Left.png"),
    Path("tests/data/Right.png"),
)
COHORT_ROOTS = {
    "real_video": Path("data/video"),
    "real_md_cl": Path("data/MD_CL"),
    "seedance": Path("data/WangXing_Seedance"),
    "tests_data": Path("tests/data"),
}
DEFAULT_PROMPT = (
    "A front-facing close-up portrait of the same young East Asian man "
    "with short black hair. Natural facial motion, stable head, clear face, "
    "and a consistent dark studio background."
)
TERMINAL_STATUSES = {"completed", "failed", "canceled"}


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _video_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.mp4")
        if path.is_file()
    )


def _round_robin(paths: Iterable[Path], limit: int) -> list[Path]:
    grouped: dict[Path, list[Path]] = {}
    for path in paths:
        grouped.setdefault(path.parent, []).append(path)
    for values in grouped.values():
        values.sort()

    selected: list[Path] = []
    keys = sorted(grouped)
    while keys and len(selected) < limit:
        next_keys: list[Path] = []
        for key in keys:
            values = grouped[key]
            if values:
                selected.append(values.pop(0))
                if len(selected) >= limit:
                    break
            if values:
                next_keys.append(key)
        keys = next_keys
    return selected


def discover_cases(
    project_root: Path,
    *,
    limit_per_cohort: int,
    cohorts: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    selected_cohorts = tuple(cohorts or COHORT_ROOTS)
    unknown = sorted(set(selected_cohorts) - set(COHORT_ROOTS))
    if unknown:
        raise ValueError(f"Unknown cohorts: {', '.join(unknown)}")

    cases: list[dict[str, Any]] = []
    for cohort in selected_cohorts:
        root = project_root / COHORT_ROOTS[cohort]
        if not root.is_dir():
            raise FileNotFoundError(f"Cohort directory is missing: {root}")
        for index, path in enumerate(
            _round_robin(_video_files(root), limit_per_cohort),
            start=1,
        ):
            relative_path = path.relative_to(project_root).as_posix()
            cases.append(
                {
                    "case_id": f"{cohort}_{index:02d}",
                    "cohort": cohort,
                    "label": path.parent.name,
                    "relative_path": relative_path,
                    "path": path,
                }
            )
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


def submit_case(
    base_url: str,
    case: dict[str, Any],
    *,
    project_root: Path,
    max_frames: int,
    prompt_text: str,
    device: str,
) -> dict[str, Any]:
    opened: list[Any] = []
    files: list[tuple[str, tuple[str, Any, str]]] = []

    def add_file(field: str, path: Path) -> None:
        handle = path.open("rb")
        opened.append(handle)
        content_type = mimetypes.guess_type(path.name)[0]
        files.append(
            (
                field,
                (
                    path.name,
                    handle,
                    content_type or "application/octet-stream",
                ),
            )
        )

    try:
        add_file("result_video", case["path"])
        for reference_image in REFERENCE_IMAGES:
            add_file("reference_images", project_root / reference_image)
        response = requests.post(
            _url(base_url, "/api/jobs"),
            files=files,
            data={
                "prompt_text": prompt_text,
                "max_frames": str(max_frames),
                "calculate_lpips": "false",
                "device": device,
                "manual_expression_score": "",
                "manual_aesthetic_score": "",
                "wangxing_au_enabled": "true",
                "wangxing_expected_class": "auto",
            },
            timeout=180,
        )
        payload = _response_json(response, f"submit {case['case_id']}")
        job_id = str(payload["job_id"])
        name = f"source-test/{case['case_id']}/{case['relative_path']}"
        rename = requests.patch(
            _url(base_url, f"/api/jobs/{job_id}"),
            json={"name": name[:120]},
            timeout=60,
        )
        _response_json(rename, f"rename {job_id}")
        return {
            **{
                key: value
                for key, value in case.items()
                if key != "path"
            },
            "path": str(case["path"]),
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
    response = requests.get(
        _url(base_url, f"/api/jobs/{job_id}"),
        timeout=60,
    )
    return _response_json(response, f"job {job_id}")


def fetch_result(base_url: str, job_id: str) -> dict[str, Any]:
    response = requests.get(
        _url(base_url, f"/api/runs/{job_id}/result.json"),
        timeout=120,
    )
    return _response_json(response, f"result {job_id}")


def _category_score(result: dict[str, Any], category: str) -> Any:
    return (
        result.get("categories", {})
        .get(category, {})
        .get("score_0_1")
    )


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    au = result.get("wangxing_au", {})
    au_compliance = au.get("au_compliance", {})
    targeted = au.get("wangxing_targeted", {})
    fusion = au.get("fusion", {})
    return {
        "status": result.get("status"),
        "coverage": result.get("coverage"),
        "weighted_score_0_100": result.get("weighted_score_0_100"),
        "category_scores": {
            category: _category_score(result, category)
            for category in (
                "identity",
                "texture",
                "expression",
                "temporal",
                "aesthetics",
            )
        },
        "category_backends": {
            category: result.get("categories", {})
            .get(category, {})
            .get("backend")
            for category in (
                "identity",
                "texture",
                "expression",
                "temporal",
                "aesthetics",
            )
        },
        "au_status": au.get("status"),
        "au_selected_expression_class": au_compliance.get(
            "selected_expression_class"
        ),
        "au_personal_score_0_1": au_compliance.get(
            "personal_au_score_0_1"
        ),
        "au_driver_expression_score_0_1": au_compliance.get(
            "driver_expression_score_0_1"
        ),
        "au_driver_temporal_alignment_score_0_1": au_compliance.get(
            "driver_temporal_alignment_score_0_1"
        ),
        "au_leakage_risk_0_1": au_compliance.get(
            "driver_identity_leakage_risk_0_1"
        ),
        "au_evidence_quality_status": au_compliance.get(
            "evidence_quality_status"
        ),
        "au_evidence_confidence_0_1": au_compliance.get(
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
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


def _field_values(
    items: list[dict[str, Any]],
    cohort: str,
    field: str,
) -> list[float]:
    values: list[float] = []
    for item in items:
        if item["case"]["cohort"] != cohort:
            continue
        value = _numeric(item.get("result", {}).get(field))
        if value is not None:
            values.append(value)
    return values


def _auc_left_higher(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    wins = sum(
        1.0 if left_value > right_value else 0.5
        if left_value == right_value
        else 0.0
        for left_value in left
        for right_value in right
    )
    return wins / (len(left) * len(right))


def compare_groups(
    items: list[dict[str, Any]],
    left_cohort: str,
    right_cohort: str,
) -> dict[str, Any]:
    fields = (
        "weighted_score_0_100",
        "category_scores",
        "au_personal_score_0_1",
        "au_leakage_risk_0_1",
        "au_wangxing_fit_score_0_1",
        "au_evidence_confidence_0_1",
    )
    expanded: list[tuple[str, str]] = [
        ("weighted_score_0_100", "weighted_score_0_100"),
        *[
            (f"category_{category}", "category_scores." + category)
            for category in (
                "identity",
                "texture",
                "expression",
                "temporal",
                "aesthetics",
            )
        ],
        ("au_personal_score_0_1", "au_personal_score_0_1"),
        ("au_leakage_risk_0_1", "au_leakage_risk_0_1"),
        ("au_wangxing_fit_score_0_1", "au_wangxing_fit_score_0_1"),
        (
            "au_evidence_confidence_0_1",
            "au_evidence_confidence_0_1",
        ),
    ]
    result: dict[str, Any] = {
        "left_cohort": left_cohort,
        "right_cohort": right_cohort,
        "fields": {},
    }
    for output_name, field in expanded:
        left: list[float] = []
        right: list[float] = []
        for item in items:
            if item["case"]["cohort"] not in {left_cohort, right_cohort}:
                continue
            source = item.get("result", {})
            value: Any = source
            for part in field.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            parsed = _numeric(value)
            if parsed is not None:
                (left if item["case"]["cohort"] == left_cohort else right).append(
                    parsed
                )
        auc = _auc_left_higher(left, right)
        result["fields"][output_name] = {
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
    return result


def _group_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "weighted_score_0_100",
        "category_scores",
        "au_personal_score_0_1",
        "au_leakage_risk_0_1",
        "au_wangxing_fit_score_0_1",
        "au_evidence_confidence_0_1",
    )
    expanded = [
        ("weighted_score_0_100", "weighted_score_0_100"),
        *[
            (f"category_{category}", f"category_scores.{category}")
            for category in (
                "identity",
                "texture",
                "expression",
                "temporal",
                "aesthetics",
            )
        ],
        ("au_personal_score_0_1", "au_personal_score_0_1"),
        ("au_leakage_risk_0_1", "au_leakage_risk_0_1"),
        ("au_wangxing_fit_score_0_1", "au_wangxing_fit_score_0_1"),
        (
            "au_evidence_confidence_0_1",
            "au_evidence_confidence_0_1",
        ),
    ]
    cohorts = sorted({item["case"]["cohort"] for item in items})
    summary: dict[str, Any] = {}
    for cohort in cohorts:
        cohort_items = [item for item in items if item["case"]["cohort"] == cohort]
        values_by_field: dict[str, list[float]] = {}
        for output_name, field in expanded:
            values: list[float] = []
            for item in cohort_items:
                value: Any = item.get("result", {})
                for part in field.split("."):
                    value = value.get(part) if isinstance(value, dict) else None
                parsed = _numeric(value)
                if parsed is not None:
                    values.append(parsed)
            values_by_field[output_name] = values
        summary[cohort] = {
            "sample_count": len(cohort_items),
            "completed_count": sum(
                1 for item in cohort_items if item.get("job", {}).get("status") == "completed"
            ),
            "field_summary": {
                field: {
                    "count": len(values),
                    "mean": statistics.mean(values) if values else None,
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0
                    if values
                    else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                }
                for field, values in values_by_field.items()
            },
        }
    return summary


def write_outputs(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    items: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "source_score_queue_test_v1",
        "manifest": manifest,
        "items": items,
        "groups": _group_summary(items),
        "comparisons": {
            f"{left}_vs_seedance": compare_groups(
                items,
                left,
                "seedance",
            )
            for left in ("real_video", "real_md_cl", "tests_data")
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = [
        "cohort,case_id,status,weighted_score_0_100,identity,texture,"
        "expression,temporal,aesthetics,au_personal,au_leakage,"
        "au_wangxing_fit,au_evidence_confidence,au_decision"
    ]
    for item in items:
        result = item.get("result", {})
        scores = result.get("category_scores", {})
        values = [
            item["case"]["cohort"],
            item["case"]["case_id"],
            item.get("job", {}).get("status"),
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Submit real-vs-Seedance score cases to the running web queue. "
            "The web app owns AU extraction and uses its existing AU cache."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--limit-per-cohort", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--resume-manifest", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--cohort",
        action="append",
        choices=tuple(COHORT_ROOTS),
        help="Restrict the submission; repeat for multiple cohorts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.limit_per_cohort <= 0:
        raise SystemExit("--limit-per-cohort must be positive.")
    if args.max_frames < 2:
        raise SystemExit("--max-frames must be at least 2.")
    if args.poll_seconds < 5:
        raise SystemExit("--poll-seconds must be at least 5.")

    if args.resume_manifest:
        manifest_path = Path(args.resume_manifest)
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path
        manifest_path = manifest_path.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases = list(manifest["cases"])
        output_dir = manifest_path.parent
        submitted = cases
    else:
        health = _response_json(
            requests.get(_url(args.base_url, "/api/health"), timeout=30),
            "health",
        )
        if health.get("status") != "ok":
            raise RuntimeError(f"Web service is not healthy: {health}")
        cases = discover_cases(
            PROJECT_ROOT,
            limit_per_cohort=args.limit_per_cohort,
            cohorts=args.cohort,
        )
        output_dir = (
            PROJECT_ROOT / args.output_dir
            if args.output_dir
            else PROJECT_ROOT
            / "outputs"
            / f"source_score_queue_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        submitted = []
        for case in cases:
            item = submit_case(
                args.base_url,
                case,
                project_root=PROJECT_ROOT,
                max_frames=args.max_frames,
                prompt_text=args.prompt_text,
                device=args.device,
            )
            submitted.append(item)
            print(
                f"submitted {item['case_id']} -> {item['job_id']} "
                f"(queue position {item.get('queue_position')})",
                flush=True,
            )
        manifest = {
            "base_url": args.base_url,
            "max_frames": args.max_frames,
            "device": args.device,
            "prompt_text": args.prompt_text,
            "wangxing_au_enabled": True,
            "wangxing_expected_class": "auto",
            "reference_images": [str(path) for path in REFERENCE_IMAGES],
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
        print(f"submitted {len(submitted)} jobs; manifest={output_dir / 'manifest.json'}")
        return 0

    latest: dict[str, dict[str, Any]] = {}
    while True:
        all_terminal = True
        status_text: list[str] = []
        for item in submitted:
            job = fetch_job(args.base_url, item["job_id"])
            latest[item["job_id"]] = job
            status_text.append(
                f"{item['case_id']}={job.get('status')}/{job.get('stage')}"
            )
            if job.get("status") not in TERMINAL_STATUSES:
                all_terminal = False
        print(" | ".join(status_text), flush=True)
        if all_terminal:
            break
        time.sleep(args.poll_seconds)

    items: list[dict[str, Any]] = []
    for item in submitted:
        job = latest[item["job_id"]]
        result = (
            compact_result(fetch_result(args.base_url, item["job_id"]))
            if job.get("status") == "completed"
            else {
                "status": job.get("status"),
                "error": job.get("error"),
            }
        )
        items.append({"case": item, "job": job, "result": result})
    write_outputs(output_dir, manifest=manifest, items=items)
    print(f"completed batch; results={output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

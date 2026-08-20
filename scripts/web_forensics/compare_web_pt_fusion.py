"""Compare webpage, PT, and optional fusion results per final-test video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(
        Path(path).expanduser().resolve().read_text(encoding="utf-8-sig")
    )


def _name(value: str | None) -> str:
    return Path(value or "").name.casefold()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _web_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in payload.get("results", []):
        card = row.get("web_card") or {}
        expression = ((card.get("radar") or {}).get("expression") or {})
        result[_name(row.get("source_video") or row.get("video"))] = {
            "decision": (row.get("forensics") or {})
            .get("summary", {})
            .get("decision"),
            "real_probability": (
                (row.get("forensics") or {})
                .get("scores", {})
                .get("calibrated_real_probability_0_1")
            ),
            "expression_score": expression.get("score"),
            "fusion_generated_probability": (
                (row.get("web_fusion") or {}).get("generated_probability")
            ),
            "policy_generated_probability": (
                (row.get("web_policy") or {}).get("generated_probability")
            ),
        }
    return result


def _pt_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        _name(row.get("video")): {
            "prediction": row.get("prediction"),
            "generated_probability": row.get("generated_probability"),
        }
        for row in payload.get("rows", [])
        if row.get("status") == "ok"
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a per-video web/PT/fusion comparison report."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--web-results", required=True)
    parser.add_argument("--pt-metrics", required=True)
    parser.add_argument("--fusion-results")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    manifest = _load(args.manifest)
    web = _web_map(_load(args.web_results))
    pt = _pt_map(_load(args.pt_metrics))
    fusion = (
        _web_map(_load(args.fusion_results))
        if args.fusion_results
        else {}
    )
    rows: list[dict[str, Any]] = []
    for sample in manifest.get("samples", []):
        key = _name(sample.get("source_video") or sample.get("video"))
        web_row = web.get(key, {})
        pt_row = pt.get(key, {})
        fusion_row = fusion.get(key, {})
        rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "label": sample.get("label"),
                "video": sample.get("source_video") or sample.get("video"),
                "web_decision": web_row.get("decision"),
                "web_real_probability": web_row.get("real_probability"),
                "expression_score": web_row.get("expression_score"),
                "pt_prediction": pt_row.get("prediction"),
                "pt_generated_probability": pt_row.get(
                    "generated_probability"
                ),
                "fusion_generated_probability": (
                    fusion_row.get("fusion_generated_probability")
                    or web_row.get("fusion_generated_probability")
                ),
                "policy_generated_probability": (
                    fusion_row.get("policy_generated_probability")
                    or web_row.get("policy_generated_probability")
                ),
            }
        )

    output = Path(args.output).expanduser().resolve()
    _write(
        output.with_suffix(".json"),
        {
            "schema_version": "web_pt_fusion_comparison_v1",
            "manifest": str(Path(args.manifest).expanduser().resolve()),
            "rows": rows,
        },
    )
    lines = [
        "# Web / PT / Fusion Comparison",
        "",
        f"Manifest: `{args.manifest}`",
        "",
        "| Sample | Label | Web | Web P(real) | Expression | PT P(AI) | PT | Fusion P(AI) | Policy P(AI) |",
        "|---|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        def fmt(value: Any, percent: bool = False) -> str:
            if value is None:
                return "-"
            return f"{float(value) * 100:.1f}%" if percent else f"{float(value):.1f}"

        lines.append(
            f"| `{row['sample_id']}` | {row['label']} | "
            f"{row['web_decision'] or '-'} | "
            f"{fmt(row['web_real_probability'], True)} | "
            f"{fmt(row['expression_score'])} | "
            f"{fmt(row['pt_generated_probability'], True)} | "
            f"{row['pt_prediction'] or '-'} | "
            f"{fmt(row['fusion_generated_probability'], True)} | "
            f"{fmt(row['policy_generated_probability'], True)} |"
        )
    output.with_suffix(".md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output.with_suffix('.md')}")
    print(f"Wrote {output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

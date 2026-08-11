"""Collaborator-ready Wang Xing specialization package.

Top-level layout shipped to collaborators:

```text
evaluator/
├── __init__.py
├── detail_expression_metrics.py
├── README.md
└── modules/
    ├── assets/
    ├── core/
    ├── forensics/
    └── wangxing/
```

Put the *parent* of this folder on ``PYTHONPATH``, then::

    from evaluator.detail_expression_metrics import compute_detail_metric
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MetricResult",
    "compute_detail_metric",
    "compute_face_expression_metric",
    "prepare_generated_video",
]


def __getattr__(name: str) -> Any:
    if name in {
        "MetricResult",
        "compute_detail_metric",
        "compute_face_expression_metric",
        "prepare_generated_video",
    }:
        from . import detail_expression_metrics as metrics

        return getattr(metrics, name)
    if name in {"WEIGHTS", "evaluate_all"}:
        from .modules.core.holistic_evaluator import WEIGHTS, evaluate_all

        return WEIGHTS if name == "WEIGHTS" else evaluate_all
    if name in {"evaluate_full_reference", "probe_video"}:
        from .modules.core.video_metrics import (
            evaluate_full_reference,
            probe_video,
        )

        return (
            evaluate_full_reference
            if name == "evaluate_full_reference"
            else probe_video
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name}")

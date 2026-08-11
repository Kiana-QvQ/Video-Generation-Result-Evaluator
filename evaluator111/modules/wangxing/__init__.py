"""Wang Xing specialization, AU datasets, and quality supplement."""

from __future__ import annotations

from typing import Any

__all__ = ["evaluate_quality_supplement"]


def __getattr__(name: str) -> Any:
    if name == "evaluate_quality_supplement":
        from .wangxing_quality_supplement import evaluate_quality_supplement

        return evaluate_quality_supplement
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

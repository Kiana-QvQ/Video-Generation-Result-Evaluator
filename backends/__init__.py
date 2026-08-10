"""Project-local optional model runners (not part of the collaborator evaluator package).

Modules:
- ``viclip_backend``: ViCLIP semantic similarity
- ``etva_judge``: external VLM judge HTTP client
- ``vbench_runner``: VBench aesthetic / consistency runner
- ``subst``: Windows subst helpers for long paths
"""

from __future__ import annotations

__all__ = [
    "etva_judge",
    "subst",
    "vbench_runner",
    "viclip_backend",
]

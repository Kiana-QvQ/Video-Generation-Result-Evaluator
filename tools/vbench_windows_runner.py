"""Run the official VBench launcher with Windows-safe process settings."""

from __future__ import annotations

import os
import sys
import importlib.util
from datetime import datetime as _Datetime
from pathlib import Path

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("TORCH_USE_LIBUV", "0")

def _load_evaluate_module():
    if "--script" not in sys.argv:
        from vbench.launch import evaluate

        return evaluate

    marker = sys.argv.index("--script")
    try:
        script_path = Path(sys.argv[marker + 1]).resolve()
    except IndexError as exc:
        raise SystemExit("--script requires a VBench evaluate.py path") from exc
    del sys.argv[marker : marker + 2]

    spec = importlib.util.spec_from_file_location(
        "_frame_audit_vbench_evaluate",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load VBench launcher: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _SafeNow:
    def __init__(self, value: _Datetime) -> None:
        self._value = value

    def strftime(self, format_string: str) -> str:
        return self._value.strftime(format_string).replace(":", "-")


class _SafeDatetime:
    @classmethod
    def now(cls) -> _SafeNow:
        return _SafeNow(_Datetime.now())


evaluate_module = _load_evaluate_module()
evaluate_module.datetime = _SafeDatetime
evaluate_module.main()

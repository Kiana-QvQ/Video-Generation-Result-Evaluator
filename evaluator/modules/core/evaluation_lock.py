from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar

from .runtime import OUTPUT_DIR

P = ParamSpec("P")
R = TypeVar("R")

_THREAD_LOCK = threading.Lock()
_LOCK_PATH = OUTPUT_DIR / ".evaluation.lock"
_WINDOWS_MUTEX_NAME = "Local\\FrameAuditEvaluationLock"


@contextmanager
def _windows_evaluation_lock() -> Iterator[None]:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [
        wintypes.LPVOID,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel32.ReleaseMutex.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.CreateMutexW(None, False, _WINDOWS_MUTEX_NAME)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    acquired = False
    try:
        result = kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        if result not in {0x00000000, 0x00000080}:
            raise ctypes.WinError(ctypes.get_last_error())
        acquired = True
        yield
    finally:
        if acquired:
            kernel32.ReleaseMutex(handle)
        kernel32.CloseHandle(handle)


@contextmanager
def evaluation_lock() -> Iterator[None]:
    """Serialize GPU-heavy evaluation across threads and processes."""
    with _THREAD_LOCK:
        if os.name == "nt":
            with _windows_evaluation_lock():
                yield
            return
        _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK_PATH.open("a+b") as handle:
            if _LOCK_PATH.stat().st_size == 0:
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def serialized_evaluation(
    function: Callable[P, R],
) -> Callable[P, R]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with evaluation_lock():
            return function(*args, **kwargs)

    return wrapped

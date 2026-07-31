from __future__ import annotations

import gc


def release_cuda_memory() -> None:
    """Release cached CUDA blocks after a heavyweight model phase."""
    gc.collect()
    try:
        import torch

        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass

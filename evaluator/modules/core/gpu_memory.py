from __future__ import annotations

import gc
import logging


LOGGER = logging.getLogger(__name__)


def release_cuda_memory() -> None:
    """Release cached CUDA blocks after a heavyweight model phase."""
    gc.collect()
    try:
        import torch

        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception as exc:
            LOGGER.debug("CUDA synchronize failed during cleanup: %s", exc)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception as exc:
        LOGGER.warning("CUDA cache cleanup unavailable: %s", exc)

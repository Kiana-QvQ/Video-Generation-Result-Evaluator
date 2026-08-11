"""Re-export predictors from the on-disk ``vedio_pred`` package."""

from vedio_pred.predict_real_video import main, predict_real_video

__all__ = ["main", "predict_real_video"]

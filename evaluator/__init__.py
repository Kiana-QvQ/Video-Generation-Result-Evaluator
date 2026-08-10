"""Video evaluation core package.

Public layout for collaborators:

- ``detail_expression_metrics``: yellow-box texture + facial muscle API
  (calls packaged forensics / wangxing implementations)
- ``core``: holistic five-category evaluation and video IO
- ``wangxing``: identity / expression specialization (AU / profiles)
- ``forensics``: real-capture vs Seedance authenticity branches

Optional ViCLIP / ETVA / VBench runners live in the repo-root ``backends``
package and are **not** part of this collaborator package.
"""

from .core.holistic_evaluator import WEIGHTS, evaluate_all
from .core.video_metrics import evaluate_full_reference, probe_video

__all__ = [
    "WEIGHTS",
    "evaluate_all",
    "evaluate_full_reference",
    "probe_video",
]

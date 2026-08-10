from __future__ import annotations

import shutil
import tempfile
import logging
from pathlib import Path

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


class FaceDetector:
    """Lightweight OpenCV face detector shared by evaluator components."""

    def __init__(self) -> None:
        cascade_path = (
            Path(cv2.data.haarcascades)
            / "haarcascade_frontalface_default.xml"
        )
        classifier_path = cascade_path
        # OpenCV on Windows may fail to open model paths containing Chinese
        # characters. Copy the bundled cascade to an ASCII temp path first.
        if cascade_path.exists() and any(
            ord(char) > 127 for char in str(cascade_path)
        ):
            try:
                temp_dir = Path(tempfile.gettempdir()) / "video_evaluator_models"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / cascade_path.name
                if (
                    not temp_path.exists()
                    or temp_path.stat().st_size != cascade_path.stat().st_size
                ):
                    shutil.copyfile(cascade_path, temp_path)
                classifier_path = temp_path
            except OSError as exc:
                LOGGER.warning(
                    "Unable to stage OpenCV face cascade at an ASCII path: %s",
                    exc,
                )
        self.classifier = cv2.CascadeClassifier(str(classifier_path))
        self.available = not self.classifier.empty()

    def detect(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        if not self.available:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )
        if len(faces) == 0:
            return None
        x, y, width, height = max(
            faces,
            key=lambda item: int(item[2]) * int(item[3]),
        )
        return int(x), int(y), int(width), int(height)

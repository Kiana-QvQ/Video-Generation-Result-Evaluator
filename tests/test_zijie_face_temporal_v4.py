from __future__ import annotations

import numpy as np
import pytest

from wangxing_project.zijie_face_temporal_v4 import (
    _margin_probability,
    _separation,
)


def test_separation_reports_robust_and_strict_gaps() -> None:
    result = _separation(
        np.asarray([0.10, 0.20, 0.30]),
        np.asarray([0.70, 0.80]),
    )

    assert result["minimum_generated_minus_maximum_real"] == pytest.approx(0.40)
    assert result["p10_generated_minus_p90_real"] > 0.40


def test_margin_probability_preserves_threshold_and_order() -> None:
    profile = {
        "threshold_generated": 0.75,
        "probability_margin_scale": 0.10,
    }

    assert _margin_probability(0.75, profile) == pytest.approx(0.50)
    assert _margin_probability(0.85, profile) > 0.70
    assert _margin_probability(0.65, profile) < 0.30

from __future__ import annotations

import numpy as np
import pytest

from wangxing_project.zijie_face_temporal_v5 import (
    _pairwise_rate,
    _rank_metrics,
    _spearman,
    au_landmark_coverage,
    naturalness_evidence_scores,
    predict_quality_score,
)


def test_rank_metrics_reward_matching_non_monotonic_quality() -> None:
    expected = np.asarray([5.0, 70.0, 90.0, 72.0])
    predicted = np.asarray([10.0, 68.0, 85.0, 70.0])

    metrics = _rank_metrics(expected, predicted)

    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["pairwise_ordering_rate"] == pytest.approx(1.0)
    assert metrics["score_range"] == pytest.approx(75.0)


def test_rank_metrics_detect_reversed_order() -> None:
    expected = np.asarray([10.0, 30.0, 80.0])
    reversed_prediction = np.asarray([80.0, 30.0, 10.0])

    assert _spearman(expected, reversed_prediction) == pytest.approx(-1.0)
    assert _pairwise_rate(expected, reversed_prediction) == pytest.approx(0.0)


def test_quality_prediction_uses_features_not_checkpoint_metadata() -> None:
    binary_profile = {
        "experts": [
            {
                "feature_indexes": [0],
                "scaler_mean": [0.0],
                "scaler_scale": [1.0],
                "coefficient": [1.0],
                "intercept": 0.0,
                "weight": 0.5,
            },
            {
                "feature_indexes": [0],
                "scaler_mean": [0.0],
                "scaler_scale": [1.0],
                "coefficient": [-1.0],
                "intercept": 0.0,
                "weight": 0.5,
            },
        ]
    }
    policy = {
        "feature_indexes": [0],
        "scaler_mean": [0.0],
        "scaler_scale": [1.0],
        "coefficient": [10.0],
        "intercept": 50.0,
        "score_clip": [5.0, 80.0],
    }

    score = predict_quality_score(
        np.asarray([[1.0], [2.0]], dtype=np.float32),
        binary_profile,
        policy,
    )

    assert 5.0 <= score <= 80.0


def test_naturalness_score_decreases_with_real_reference_distance() -> None:
    block = {
        "feature_indexes": [0, 1],
        "real_center": [0.0, 0.0],
        "real_scale": [1.0, 1.0],
        "distance_p90": 1.0,
        "temperature": 0.25,
    }
    profile = {"face": block, "mouth": block}

    close = naturalness_evidence_scores(
        np.asarray([[0.1, 0.1]], dtype=np.float32), profile
    )
    far = naturalness_evidence_scores(
        np.asarray([[2.0, 2.0]], dtype=np.float32), profile
    )

    assert close["face_naturalness_score_0_100"] > far[
        "face_naturalness_score_0_100"
    ]
    assert close["mouth_naturalness_score_0_100"] > far[
        "mouth_naturalness_score_0_100"
    ]


def test_au_landmark_coverage_uses_frame_span_and_valid_mouth(tmp_path) -> None:
    path = tmp_path / "au.csv"
    headers = [
        "frame_idx",
        "lm_mp_33_x", "lm_mp_33_y", "lm_mp_263_x", "lm_mp_263_y",
        "lm_mp_61_x", "lm_mp_61_y", "lm_mp_291_x", "lm_mp_291_y",
        "lm_mp_13_x", "lm_mp_13_y", "lm_mp_14_x", "lm_mp_14_y",
    ]
    valid = ["0.5"] * 12
    rows = [
        ["0", *valid],
        ["2", *valid[:4], "", *valid[5:]],
    ]
    path.write_text(
        ",".join(headers) + "\n" + "\n".join(",".join(row) for row in rows),
        encoding="utf-8",
    )

    coverage = au_landmark_coverage(path)

    assert coverage["sampled_frame_span"] == 3
    assert coverage["face_coverage_0_1"] == pytest.approx(2 / 3)
    assert coverage["mouth_coverage_0_1"] == pytest.approx(1 / 3)

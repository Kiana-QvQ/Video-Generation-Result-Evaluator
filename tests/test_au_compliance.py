from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluator.wangxing.au_compliance import (
    AU_PROFILE_SCHEMA,
    _add_auto_selection_scores,
    _combine_personal_au_scores,
    _face_mesh_action_features,
    _infer_observable_expression_class,
    compare_temporal_events,
    dtw_similarity,
    fit_au_profile,
    fuse_compliance_scores,
    fuse_wangxing_targeted_scores,
    load_au_table,
    load_au_profile_tables,
    sha256_file,
    score_au_compliance,
    temporal_event_features,
    _summary_feature_indices,
    _summary_pairs,
    au_summary,
)


def _write_au_csv(path: Path, sequence: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["AU01_r", "AU04_r", "AU06_r", "AU12_r", "AU15_r", "AU25_r"])
        writer.writerows(sequence.tolist())


def _write_libreface_au_csv(path: Path, sequence: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "au_1_intensity",
                "au_4_intensity",
                "au_6_intensity",
                "au_12_intensity",
                "au_15_intensity",
                "au_25_intensity",
            ]
        )
        writer.writerows((sequence * 5.0).tolist())


def _write_partial_au_csv(
    path: Path,
    sequence: np.ndarray,
    au_ids: tuple[int, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"AU{au_id:02d}_r" for au_id in au_ids])
        writer.writerows(sequence.tolist())


def _write_quality_au_csv(path: Path, sequence: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_idx",
        "pitch",
        "yaw",
        *[
            coordinate
            for index in range(20)
            for coordinate in (
                f"lm_mp_{index}_x",
                f"lm_mp_{index}_y",
            )
        ],
        "AU01_r",
        "AU04_r",
        "AU06_r",
        "AU12_r",
        "AU15_r",
        "AU25_r",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(sequence):
            payload = {
                "frame_idx": index,
                "pitch": 0.0,
                "yaw": 0.0,
                "AU01_r": float(row[0] * 5.0),
                "AU04_r": float(row[1] * 5.0),
                "AU06_r": float(row[2] * 5.0),
                "AU12_r": float(row[3] * 5.0),
                "AU15_r": float(row[4] * 5.0),
                "AU25_r": float(row[5] * 5.0),
            }
            for landmark_index in range(20):
                payload[f"lm_mp_{landmark_index}_x"] = (
                    0.2 + 0.03 * (landmark_index % 10)
                    if index == 0
                    else 0.0
                )
                payload[f"lm_mp_{landmark_index}_y"] = (
                    0.2 + 0.03 * (landmark_index // 10)
                    if index == 0
                    else 0.0
                )
            writer.writerow(payload)


class AUComplianceTests(unittest.TestCase):
    def test_face_mesh_gates_weak_mouth_motion(self) -> None:
        points = np.zeros((8, 478, 2), dtype=np.float32)
        points[:, 234] = [0.0, 0.5]
        points[:, 454] = [1.0, 0.5]
        points[:, 13] = [0.5, 0.50]
        points[:, 14] = [0.5, 0.505]
        points[:, 61] = [0.3, 0.5]
        points[:, 291] = [0.7, 0.5]
        points[:, 105] = [0.4, 0.35]
        points[:, 334] = [0.6, 0.35]
        points[:, 159] = [0.4, 0.42]
        points[:, 145] = [0.4, 0.46]
        points[:, 386] = [0.6, 0.42]
        points[:, 374] = [0.6, 0.46]

        result = _face_mesh_action_features(
            {
                "_landmarks_2d": points,
                "landmark_indices": list(range(478)),
            }
        )

        self.assertEqual(result["status"], "available")
        self.assertEqual(result["mouth_open"]["salient_event_count"], 0)
        self.assertEqual(
            result["metrics"]["expression_confidence_0_1"],
            0.0,
        )

    def test_missing_requested_aus_are_nan_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.csv"
            _write_partial_au_csv(
                path,
                np.ones((3, 2), dtype=np.float32) * 0.5,
                (1, 6),
            )
            sequence, au_ids, metadata = load_au_table(
                path,
                (1, 4, 6),
                feature_type="intensity",
            )

            with self.assertRaises(ValueError):
                load_au_table(
                    path,
                    (1, 4, 6),
                    feature_type="intensity",
                    strict=True,
                )

        self.assertEqual(au_ids, (1, 4, 6))
        self.assertEqual(metadata["supported_au_ids"], [1, 6])
        self.assertEqual(metadata["missing_au_ids"], [4])
        self.assertTrue(np.isnan(sequence[:, 1]).all())

    def test_loads_openface_style_intensity_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            _write_au_csv(path, np.ones((4, 6), dtype=np.float32) * 2.5)
            sequence, au_ids, _ = load_au_table(path)

        self.assertEqual(au_ids, (1, 4, 6, 12, 15, 25))
        self.assertEqual(sequence.shape, (4, 6))
        self.assertTrue(np.allclose(sequence, 0.5))

    def test_loads_libreface_style_intensity_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            _write_libreface_au_csv(
                path,
                np.ones((4, 6), dtype=np.float32) * 0.5,
            )
            sequence, au_ids, _ = load_au_table(path)

        self.assertEqual(au_ids, (1, 4, 6, 12, 15, 25))
        self.assertTrue(np.allclose(sequence, 0.5))

    def test_fallback_detection_score_contributes_to_quality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fallback.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "frame_idx",
                        "frame_time_in_ms",
                        "face_alignment_method",
                        "face_detection_score",
                        "au_1_intensity",
                    ]
                )
                writer.writerows(
                    [
                        [0, 0.0, "insightface_bbox", 0.2, 1.0],
                        [1, 0.0333, "insightface_bbox", 0.8, 2.0],
                    ]
                )
            _, _, metadata = load_au_table(
                path,
                (1,),
                feature_type="intensity",
            )

        quality = metadata["quality"]
        self.assertTrue(quality["available"])
        self.assertEqual(quality["source"], "insightface_detection")
        self.assertEqual(quality["status"], "partial")
        self.assertAlmostEqual(quality["mean_frame_quality"], 0.5)
        self.assertAlmostEqual(
            metadata["frame_times_seconds"][1],
            0.0333,
            places=4,
        )

    def test_mesh_quality_is_not_marked_mixed_when_fallback_is_unused(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh_with_fallback_fields.csv"
            fieldnames = [
                "frame_idx",
                "pitch",
                "yaw",
                "face_alignment_method",
                "face_detection_score",
                *[
                    coordinate
                    for index in range(20)
                    for coordinate in (
                        f"lm_mp_{index}_x",
                        f"lm_mp_{index}_y",
                    )
                ],
                "au_1_intensity",
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for frame_index in range(2):
                    payload = {
                        "frame_idx": frame_index,
                        "pitch": 0.0,
                        "yaw": 0.0,
                        "face_alignment_method": "mediapipe",
                        "face_detection_score": "",
                        "au_1_intensity": 0.5,
                    }
                    for landmark_index in range(20):
                        payload[f"lm_mp_{landmark_index}_x"] = (
                            0.2 + 0.08 * (landmark_index % 10)
                        )
                        payload[f"lm_mp_{landmark_index}_y"] = (
                            0.2 + 0.07 * (landmark_index // 10)
                        )
                    writer.writerow(payload)
            _, _, metadata = load_au_table(
                path,
                (1,),
                feature_type="intensity",
            )

        quality = metadata["quality"]
        self.assertEqual(quality["source"], "face_mesh")
        self.assertEqual(quality["status"], "pass")
        self.assertEqual(quality["valid_frame_ratio"], 1.0)

    def test_profile_loader_reads_intensity_and_presence_in_one_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "au_1_intensity",
                        "au_1",
                        "au_4_intensity",
                        "au_4",
                    ]
                )
                writer.writerow([2.5, 1, 0.5, 0])
                writer.writerow([5.0, 0, 2.5, 1])
            intensity, supported, presence, presence_supported = (
                load_au_profile_tables(
                    path,
                    intensity_au_ids=(1, 4),
                    presence_au_ids=(1, 4),
                )
            )

        self.assertEqual(supported, (1, 4))
        self.assertTrue(np.allclose(intensity, [[0.5, 0.1], [1.0, 0.5]]))
        self.assertEqual(presence_supported, (1, 4))
        self.assertIsNotNone(presence)
        self.assertTrue(np.array_equal(presence, [[1.0, 0.0], [0.0, 1.0]]))

    def test_profile_loader_keeps_missing_aus_as_nan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial-profile.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["au_1_intensity", "au_1"])
                writer.writerow([2.5, 1])
            intensity, _, presence, _ = load_au_profile_tables(
                path,
                intensity_au_ids=(1, 4),
                presence_au_ids=(1, 4),
            )

        self.assertTrue(np.isnan(intensity[:, 1]).all())
        self.assertIsNotNone(presence)
        self.assertTrue(np.isnan(presence[:, 1]).all())

    def test_dtw_identical_sequences_score_one(self) -> None:
        sequence = np.asarray([[0, 1, 0], [0.2, 0.8, 0.1]], dtype=np.float32)
        self.assertAlmostEqual(dtw_similarity(sequence, sequence), 1.0)

    def test_temporal_event_alignment_is_one_for_identical_sequences(self) -> None:
        sequence = np.asarray(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0.8, 0.4, 0, 0.7, 0, 0],
                [0.9, 0.5, 0, 0.8, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        features = temporal_event_features(sequence)
        alignment = compare_temporal_events(features, features)
        self.assertAlmostEqual(
            alignment["event_alignment_score_0_1"],
            1.0,
        )

    def test_temporal_events_include_frame_level_boundaries(self) -> None:
        sequence = np.asarray(
            [
                [0],
                [0],
                [1],
                [1],
                [0],
                [0],
                [1],
                [1],
                [0],
            ],
            dtype=np.float32,
        )
        result = temporal_event_features(
            sequence,
            au_ids=(1,),
            active_threshold=0.6,
        )
        events = result["per_au"]["1"]["events"]

        self.assertEqual(len(events), 2)
        self.assertEqual(
            (events[0]["start_frame"], events[0]["end_frame"]),
            (2, 3),
        )
        self.assertEqual(
            (events[1]["start_frame"], events[1]["end_frame"]),
            (6, 7),
        )
        self.assertGreater(events[0]["peak_intensity"], 0.5)

    def test_temporal_events_keep_original_timeline_around_invalid_frames(self) -> None:
        sequence = np.asarray(
            [[1], [1], [1], [1], [0]],
            dtype=np.float32,
        )
        result = temporal_event_features(
            sequence,
            au_ids=(1,),
            active_threshold=0.6,
            valid_mask=np.asarray(
                [True, True, False, True, True],
                dtype=bool,
            ),
            frame_indices=np.asarray([10, 11, 12, 13, 14], dtype=np.int64),
        )
        events = result["per_au"]["1"]["events"]

        self.assertEqual(result["frame_count"], 5)
        self.assertEqual(result["valid_frame_count"], 4)
        self.assertEqual(
            [(event["start_frame"], event["end_frame"]) for event in events],
            [(10, 11), (13, 13)],
        )
        self.assertAlmostEqual(events[0]["start_position"], 0.0)
        self.assertAlmostEqual(events[1]["start_position"], 0.75)

    def test_face_quality_gate_marks_low_quality_evidence_uncertain(self) -> None:
        base = np.asarray(
            [[0, 0, 1, 1, 0, 0], [0.1, 0, 0.9, 0.8, 0, 0.2]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            generated_path = root / "generated.csv"
            fit_au_profile([("smile", base), ("smile", base)], profile_path)
            _write_quality_au_csv(
                generated_path,
                np.vstack([base, base]),
            )
            result = score_au_compliance(
                profile_path,
                generated_path,
                expected_class="smile",
            )

        self.assertEqual(
            result["generated_au"]["quality"]["status"],
            "uncertain",
        )
        self.assertEqual(result["evidence_quality_status"], "uncertain")
        self.assertIn("face_quality_low", result["uncertainty_reasons"])

    def test_profile_scores_matching_class(self) -> None:
        base = np.asarray(
            [[0, 0, 1, 1, 0, 0], [0.1, 0, 0.9, 0.8, 0, 0.2]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            generated_path = root / "generated.csv"
            fit_au_profile([("smile", base), ("smile", base * 0.98)], profile_path)
            _write_au_csv(generated_path, base)
            result = score_au_compliance(
                profile_path,
                generated_path,
                expected_class="smile",
            )
            payload = json.loads(profile_path.read_text(encoding="utf-8"))

        self.assertEqual(result["selected_expression_class"], "smile")
        self.assertGreater(result["personal_au_score_0_1"], 0.5)
        self.assertEqual(payload["schema_version"], AU_PROFILE_SCHEMA)

    def test_profile_facial_dynamics_uses_profile_statistics_without_driver(
        self,
    ) -> None:
        base = np.asarray(
            [
                [0, 0, 1, 1, 0, 0],
                [0.1, 0, 0.9, 0.8, 0, 0.2],
                [0.2, 0, 0.8, 0.7, 0, 0.3],
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            training_path = root / "training.csv"
            generated_path = root / "generated.csv"
            _write_au_csv(training_path, base)
            _write_au_csv(generated_path, base * 0.98)
            fit_au_profile(
                [("smile", base)],
                profile_path,
                sample_metadata=[
                    {
                        "source_id": "training",
                        "au_path": str(training_path),
                    }
                ],
            )

            result = score_au_compliance(
                profile_path,
                generated_path,
                expected_class="smile",
            )

        self.assertEqual(
            result["wangxing_facial_dynamics_evidence"]["source"],
            "wangxing_training_profile_dynamic_statistics",
        )
        self.assertIsNotNone(
            result["facial_expression_dynamics_score_0_1"]
        )
        self.assertEqual(
            result["wangxing_action_compliance_score_0_1"],
            None,
        )
        self.assertEqual(
            result["wangxing_temporal_alignment_score_0_1"],
            None,
        )
        self.assertTrue(
            result["wangxing_facial_dynamics_evidence"]["uses_reference_video"]
            is False
        )

    def test_exact_training_sequence_uses_profile_control(self) -> None:
        base = np.asarray(
            [[0, 0, 1, 1, 0, 0], [0.1, 0, 0.9, 0.8, 0, 0.2]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            generated_path = root / "generated.csv"
            _write_au_csv(generated_path, base)
            fit_au_profile(
                [("smile", base)],
                profile_path,
                sample_metadata=[
                    {
                        "source_id": "self-control",
                        "au_sha256": sha256_file(generated_path),
                    }
                ],
            )

            result = score_au_compliance(
                profile_path,
                generated_path,
                expected_class="smile",
                driver_au_path=generated_path,
            )

        self.assertTrue(result["same_generated_driver_au"])
        self.assertTrue(
            result["class_scores"]["smile"]["exact_sequence_match"]
        )
        self.assertAlmostEqual(result["personal_au_score_0_1"], 1.0)
        self.assertEqual(
            result["class_scores"]["smile"]["personal_au_score_source"],
            "exact_training_sequence_control",
        )

    def test_exact_training_video_uses_profile_control_after_au_reextraction(
        self,
    ) -> None:
        base = np.asarray(
            [[0, 0, 1, 1, 0, 0], [0.1, 0, 0.9, 0.8, 0, 0.2]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            generated_path = root / "generated.csv"
            video_path = root / "training-video.mp4"
            video_path.write_bytes(b"same training video")
            _write_au_csv(generated_path, base * 0.9)
            fit_au_profile(
                [("smile", base)],
                profile_path,
                sample_metadata=[
                    {
                        "source_id": "same-video",
                        "video_sha256": sha256_file(video_path),
                    }
                ],
            )

            result = score_au_compliance(
                profile_path,
                generated_path,
                expected_class="smile",
                generated_video_path=video_path,
            )

        score = result["class_scores"]["smile"]
        self.assertTrue(score["exact_sequence_match"])
        self.assertEqual(score["exact_sequence_match_source"], "video_hash")
        self.assertAlmostEqual(result["personal_au_score_0_1"], 1.0)

    def test_auto_selection_returns_neutral_for_weak_expression_signal(self) -> None:
        expression = np.asarray(
            [[0, 0, 1, 1, 0, 0], [0.1, 0, 0.9, 0.8, 0, 0.2]],
            dtype=np.float32,
        )
        neutral = np.zeros((8, 6), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            generated_path = root / "generated.csv"
            fit_au_profile([("anger", expression)], profile_path)
            _write_au_csv(generated_path, neutral)
            result = score_au_compliance(profile_path, generated_path)

        self.assertEqual(result["selected_expression_class"], "neutral")
        self.assertIsNone(result["personal_au_score_0_1"])
        self.assertEqual(
            result["class_scores"]["neutral"]["selection_reason"],
            "no_clear_expression",
        )

    def test_auto_selection_does_not_favor_broader_class_threshold(self) -> None:
        class_scores = {
            "smile": {
                "personal_au_score_0_1": 0.33,
                "summary_distance": 4.00,
                "frame_anomaly_ratio": 0.28,
            },
            "annoyance": {
                "personal_au_score_0_1": 0.41,
                "summary_distance": 4.31,
                "frame_anomaly_ratio": 0.39,
            },
        }

        _add_auto_selection_scores(class_scores)

        self.assertGreater(
            class_scores["smile"]["auto_selection_score_0_1"],
            class_scores["annoyance"]["auto_selection_score_0_1"],
        )

    def test_presence_evidence_can_correct_auto_expression_class(self) -> None:
        class_scores = {
            "smile": {
                "personal_au_score_0_1": 0.33,
                "summary_distance": 4.00,
                "frame_anomaly_ratio": 0.28,
            },
            "annoyance": {
                "personal_au_score_0_1": 0.41,
                "summary_distance": 4.31,
                "frame_anomaly_ratio": 0.39,
            },
        }

        _combine_personal_au_scores(
            class_scores,
            {"smile": 0.97, "annoyance": 0.84},
        )
        _add_auto_selection_scores(class_scores)

        self.assertGreater(
            class_scores["smile"]["personal_au_score_0_1"],
            class_scores["annoyance"]["personal_au_score_0_1"],
        )
        self.assertGreater(
            class_scores["smile"]["auto_selection_score_0_1"],
            class_scores["annoyance"]["auto_selection_score_0_1"],
        )

    def test_observable_smile_cue_overrides_profile_classification(self) -> None:
        temporal = {
            "per_au": {
                "4": {"active_ratio": 0.05},
                "6": {"active_ratio": 0.55},
                "12": {"active_ratio": 0.68},
                "15": {"active_ratio": 0.0},
                "17": {"active_ratio": 0.02},
                "23": {"active_ratio": 0.0},
                "24": {"active_ratio": 0.0},
            }
        }
        presence = {
            "anger": {
                "activation_ratio": {
                    "4": 0.05,
                    "6": 0.74,
                    "12": 0.84,
                    "15": 0.0,
                    "17": 0.10,
                    "23": 0.0,
                    "24": 0.01,
                }
            }
        }

        result = _infer_observable_expression_class(
            generated_temporal=temporal,
            presence_reports=presence,
        )

        self.assertEqual(result["selected_class"], "smile")
        self.assertEqual(
            result["reason"],
            "strong_smile_au6_au12_coactivation",
        )
        self.assertGreater(result["smile_score_0_1"], 0.6)

    def test_observable_smile_requires_joint_au_evidence(self) -> None:
        temporal = {
            "per_au": {
                "4": {"active_ratio": 0.20},
                "6": {"active_ratio": 0.0},
                "12": {"active_ratio": 0.05},
                "15": {"active_ratio": 0.0},
                "17": {"active_ratio": 0.0},
                "23": {"active_ratio": 0.0},
                "24": {"active_ratio": 0.0},
            }
        }
        presence = {
            "anger": {
                "activation_ratio": {
                    "4": 0.60,
                    "6": 0.15,
                    "12": 0.84,
                    "15": 0.0,
                    "17": 0.0,
                    "23": 0.0,
                    "24": 0.0,
                }
            }
        }

        result = _infer_observable_expression_class(
            generated_temporal=temporal,
            presence_reports=presence,
        )

        self.assertIsNone(result["selected_class"])

    def test_partial_intensity_support_is_scored_without_zero_filling(self) -> None:
        base = np.asarray(
            [[0, 0, 1, 1, 0, 0], [0.1, 0, 0.9, 0.8, 0, 0.2]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            generated_path = root / "generated.csv"
            driver_path = root / "driver.csv"
            fit_au_profile(
                [("smile", base), ("smile", base * 0.98)],
                profile_path,
            )
            _write_partial_au_csv(
                generated_path,
                base[:, :5],
                (1, 4, 6, 12, 15),
            )
            _write_partial_au_csv(
                driver_path,
                base[:, :5],
                (1, 4, 6, 12, 15),
            )
            result = score_au_compliance(
                profile_path,
                generated_path,
                expected_class="smile",
                driver_au_path=driver_path,
            )

        self.assertEqual(result["supported_au_ids"], [1, 4, 6, 12, 15])
        self.assertEqual(result["missing_au_ids"], [25])
        self.assertEqual(result["evidence_quality_status"], "uncertain")
        self.assertIsNotNone(result["driver_expression_score_0_1"])
        self.assertIsNotNone(result["driver_temporal_alignment"])
        self.assertIsNone(result["generated_au"]["selected_columns"]["25"])

    def test_partial_summary_indices_keep_middle_au_alignment(self) -> None:
        full_au_ids = (1, 2, 4, 6, 12)
        supported_au_ids = (1, 2, 4, 12)
        pairs = [
            pair
            for pair in _summary_pairs(full_au_ids)
            if pair[0] in supported_au_ids and pair[1] in supported_au_ids
        ]
        summary = au_summary(
            np.ones((4, len(supported_au_ids)), dtype=np.float32),
            au_ids=supported_au_ids,
            coactivation_pairs=pairs,
        )
        indices = _summary_feature_indices(
            full_au_ids,
            supported_au_ids,
        )

        self.assertEqual(len(summary), len(indices))
        self.assertEqual(
            indices,
            [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18],
        )

    def test_fusion_reviews_high_leakage(self) -> None:
        result = fuse_compliance_scores(
            identity_score_0_1=0.9,
            personal_au_score_0_1=0.8,
            driver_expression_score_0_1=0.8,
            leakage_risk_0_1=0.7,
        )
        self.assertEqual(result["decision"], "review")
        self.assertIn("driver_identity_leakage", result["decision_reasons"])

    def test_wangxing_targeted_fit_reviews_missing_dynamic_evidence(self) -> None:
        result = fuse_wangxing_targeted_scores(
            personal_au_score_0_1=0.8,
            driver_expression_score_0_1=None,
            leakage_risk_0_1=0.1,
        )
        self.assertEqual(result["decision"], "review")
        self.assertEqual(result["status"], "partial")
        self.assertIn("facial_dynamics", result["missing_evidence"])
        self.assertAlmostEqual(result["score_weight_coverage"], 0.7)
        self.assertAlmostEqual(
            result["wangxing_expression_fit_score_0_1"],
            0.8,
        )

    def test_wangxing_targeted_fit_reviews_low_personal_au(self) -> None:
        result = fuse_wangxing_targeted_scores(
            personal_au_score_0_1=0.3,
            driver_expression_score_0_1=None,
            leakage_risk_0_1=0.1,
        )
        self.assertEqual(result["decision"], "review")
        self.assertIn(
            "wangxing_au_below_threshold",
            result["decision_reasons"],
        )

    def test_wangxing_targeted_fit_reviews_uncertain_evidence(self) -> None:
        result = fuse_wangxing_targeted_scores(
            personal_au_score_0_1=0.8,
            driver_expression_score_0_1=None,
            leakage_risk_0_1=0.1,
            evidence_quality_status="uncertain",
            evidence_confidence_0_1=0.2,
            uncertainty_reasons=["face_quality_low"],
        )
        self.assertEqual(result["decision"], "review")
        self.assertIn("evidence_quality_low", result["decision_reasons"])

    def test_wangxing_targeted_fit_reviews_partial_evidence(self) -> None:
        result = fuse_wangxing_targeted_scores(
            personal_au_score_0_1=0.8,
            driver_expression_score_0_1=None,
            leakage_risk_0_1=0.1,
            evidence_quality_status="partial",
            evidence_confidence_0_1=0.5,
            uncertainty_reasons=["face_quality_low"],
        )
        self.assertEqual(result["decision"], "review")
        self.assertIn("evidence_quality_low", result["decision_reasons"])

"""Joint AU(25-d) + dual-scale video features → one MLP.

Does not replace default dual-only ``wangxing_dual_scale_classifier.pt``.
AU features are the same 25 evidence dims used by the logistic fusion head
(Wang Xing source + facial_motion / SSL / physio / quality gates).
"""

from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from evaluator.modules.core.paths import PROJECT_ROOT, project_path
from evaluator.modules.forensics.learned_fusion_head import (
    FEATURE_NAMES,
    extract_fusion_features,
)
from evaluator.vedio_pred.real_video_detector import (
    FEATURE_VERSION,
    _classification_metrics,
    _fit_temperature,
    _predict_classifier_logits,
    _set_seed,
    _standardize,
)
from evaluator.vedio_pred.wangxing_dual_pt import (
    SCALE_A,
    SCALE_B,
    DualScaleClassifier,
    build_dual_feature_table,
    extract_dual_feature,
)

JOINT_MODEL_TYPE = "wangxing_joint_au_dual_pt_v1"
AU_DIM = len(FEATURE_NAMES)
_LE_SUFFIX = re.compile(r"_le\d+$", re.IGNORECASE)
TEST_AI_STEMS = frozenset(
    {
        "BaiJunZhiJiang_Change",
        "Happy_Change",
        "ImissU_Change",
        "LeJiShengBei_Change",
        "YanWu_Change",
    }
)


def resolve_torch_device(device: str | None = "cuda") -> torch.device:
    requested = (device or "cuda").strip()
    if requested.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested CUDA but torch.cuda.is_available() is False. "
            "Install a CUDA build of PyTorch, or pass --device cpu."
        )
    return resolved


def is_forbidden_train_video(path: str | Path) -> bool:
    video = Path(path)
    if video.stem in TEST_AI_STEMS or au_stem_from_video(video) in TEST_AI_STEMS:
        return True
    parts = [part.casefold() for part in video.parts]
    for index, part in enumerate(parts[:-1]):
        if part == "test" and parts[index + 1] == "ai":
            return True
    return False


def is_augmented_video(path: str | Path) -> bool:
    """Return whether a path is a resolution-augmented training copy."""
    video = Path(path)
    return any(part.casefold() == "_aug" for part in video.parts) or bool(
        _LE_SUFFIX.search(video.stem)
    )


def _predict_logits(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    array = np.asarray(features, dtype=np.float32)
    with torch.no_grad():
        tensor = torch.from_numpy(array).to(device)
        logits = model(tensor)
    return logits.detach().cpu().numpy().astype(np.float32)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _profile_signature(
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
) -> str:
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "source_profile": source_profile,
        "forensics_profiles": forensics_profiles,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


def _video_key(path: Path) -> str:
    return str(path.resolve()).casefold()


def au_stem_from_video(video: Path) -> str:
    return _LE_SUFFIX.sub("", video.stem)


def resolve_au_csv_for_video(
    video: str | Path,
    *,
    project_root: Path | None = None,
    au_hint: str | Path | None = None,
) -> Path | None:
    """Map a video (including ``*_le1024`` aug copies) to its LibreFace AU CSV."""
    root = Path(project_root or PROJECT_ROOT)
    if au_hint is not None:
        hint = Path(au_hint)
        if not hint.is_absolute():
            hint = root / hint
        if hint.is_file():
            return hint.resolve()

    video_path = Path(video)
    if not video_path.is_absolute():
        video_path = root / video_path
    video_path = video_path.resolve()
    stem = au_stem_from_video(video_path)

    for video_root, au_root in (
        (root / "data" / "MD_CL", root / "data" / "au" / "MD_CL"),
        (root / "data" / "WangXing_Seedance", root / "data" / "au" / "WangXing_Seedance"),
        (root / "data" / "test" / "AI", root / "data" / "au" / "test" / "AI"),
    ):
        try:
            rel = video_path.relative_to(video_root.resolve())
        except ValueError:
            continue
        mirrored = (au_root / rel).with_suffix(".csv")
        if mirrored.is_file():
            return mirrored.resolve()

    seedance = root / "data" / "au" / "WangXing_Seedance" / f"{stem}.csv"
    if seedance.is_file():
        return seedance.resolve()

    mdcl_hits = sorted((root / "data" / "au" / "MD_CL").rglob(f"{stem}.csv"))
    if len(mdcl_hits) == 1:
        return mdcl_hits[0].resolve()
    if len(mdcl_hits) > 1:
        parent_name = video_path.parent.name.casefold()
        for hit in mdcl_hits:
            if hit.parent.name.casefold() == parent_name:
                return hit.resolve()
        return mdcl_hits[0].resolve()

    test_ai = root / "data" / "au" / "test" / "AI" / f"{stem}.csv"
    if test_ai.is_file():
        return test_ai.resolve()
    return None


def attach_au_pairs(
    manifest: dict[str, Any],
    *,
    project_root: Path | None = None,
    holdout_manifest: Path | None = None,
) -> dict[str, Any]:
    """Fill ``pairs`` with video↔AU records; drop train rows without AU."""
    root = Path(project_root or PROJECT_ROOT)
    hint_by_video: dict[str, str] = {}
    if holdout_manifest is not None and holdout_manifest.is_file():
        holdout = _load_json(holdout_manifest)
        for key in ("real", "seedance"):
            for item in holdout.get(key, []):
                if not isinstance(item, dict) or not item.get("video"):
                    continue
                video = project_path(str(item["video"]))
                if item.get("au"):
                    hint_by_video[_video_key(video)] = str(item["au"])

    def _pair_list(videos: list[str]) -> tuple[list[dict[str, str]], list[str]]:
        pairs: list[dict[str, str]] = []
        missing: list[str] = []
        for video_str in videos:
            video = Path(video_str)
            if not video.is_absolute():
                video = root / video
            au = resolve_au_csv_for_video(
                video,
                project_root=root,
                au_hint=hint_by_video.get(_video_key(video)),
            )
            if au is None:
                missing.append(str(video))
                continue
            pairs.append({"video": str(video.resolve()), "au": str(au.resolve())})
        return pairs, missing

    out = dict(manifest)
    train_real, miss_tr = _pair_list(list(manifest.get("train", {}).get("real", [])))
    train_fake, miss_tf = _pair_list(list(manifest.get("train", {}).get("fake", [])))
    test_real, miss_er = _pair_list(list(manifest.get("test", {}).get("real", [])))
    test_fake, miss_ef = _pair_list(list(manifest.get("test", {}).get("fake", [])))
    dropped_forbidden: list[str] = []

    def _keep_trainable(pairs: list[dict[str, str]]) -> list[dict[str, str]]:
        kept: list[dict[str, str]] = []
        for item in pairs:
            if is_forbidden_train_video(item["video"]):
                dropped_forbidden.append(item["video"])
                continue
            kept.append(item)
        return kept

    train_real = _keep_trainable(train_real)
    train_fake = _keep_trainable(train_fake)
    out["pairs"] = {
        "train": {"real": train_real, "fake": train_fake},
        "test": {"real": test_real, "fake": test_fake},
    }
    out["train"] = {
        "real": [item["video"] for item in train_real],
        "fake": [item["video"] for item in train_fake],
    }
    out["test"] = {
        "real": [item["video"] for item in test_real],
        "fake": [item["video"] for item in test_fake],
    }
    out["au_pair_missing"] = {
        "train_real": miss_tr,
        "train_fake": miss_tf,
        "test_real": miss_er,
        "test_fake": miss_ef,
    }
    out["counts"] = {
        **dict(manifest.get("counts") or {}),
        "train_real": len(train_real),
        "train_fake": len(train_fake),
        "test_real": len(test_real),
        "test_fake": len(test_fake),
        "au_missing_train": len(miss_tr) + len(miss_tf),
        "au_missing_test": len(miss_er) + len(miss_ef),
        "dropped_forbidden_train": len(dropped_forbidden),
    }
    out["dropped_forbidden_train"] = dropped_forbidden
    out["schema_version"] = "wangxing_joint_au_pt_split_res1k_v1"
    out["fusion"] = {
        "mode": "early_concat_au25_dual_pt",
        "au_feature_names": list(FEATURE_NAMES),
        "au_dim": AU_DIM,
        "note": (
            "AU 25-d evidence is concatenated with dual-scale video features "
            "and fed into one MLP. No 0.65/0.35 late fusion."
        ),
    }
    return out


def _extract_au_matrix(
    pairs: list[dict[str, str]],
    *,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
    cache_path: Path,
) -> tuple[dict[str, np.ndarray], list[str]]:
    cache_path = Path(cache_path)
    profile_signature = _profile_signature(
        source_profile,
        forensics_profiles,
    )
    au_map: dict[str, np.ndarray] = {}
    if cache_path.is_file():
        try:
            with np.load(str(cache_path), allow_pickle=True) as payload:
                cached_signature = str(payload["profile_signature"].tolist()[0])
                cached_names = [
                    str(item)
                    for item in payload["feature_names"].tolist()
                ]
                if (
                    cached_signature == profile_signature
                    and cached_names == list(FEATURE_NAMES)
                ):
                    cached_paths = [
                        str(Path(item).resolve())
                        for item in payload["paths"].tolist()
                    ]
                    features = np.asarray(payload["features"], dtype=np.float32)
                    if features.ndim == 2 and len(cached_paths) == len(features):
                        au_map = {
                            path: features[index]
                            for index, path in enumerate(cached_paths)
                            if features[index].shape[0] == AU_DIM
                        }
        except (KeyError, OSError, ValueError):
            au_map = {}

    errors: list[str] = []
    needed = sorted(
        {
            str(Path(item["au"]).resolve())
            for item in pairs
            if item.get("au")
        }
    )
    pending = [path for path in needed if path not in au_map]
    for index, au_str in enumerate(pending, start=1):
        try:
            vector, _ = extract_fusion_features(
                au_path=Path(au_str),
                wangxing_source_profile=source_profile,
                forensics_profiles=forensics_profiles,
            )
            vector = np.asarray(vector, dtype=np.float32)
            if vector.shape != (AU_DIM,):
                raise ValueError(
                    f"Expected {AU_DIM} AU features, got shape {vector.shape}"
                )
            au_map[au_str] = vector
        except Exception as exc:  # noqa: BLE001 - keep batch training robust
            errors.append(f"{au_str}: {type(exc).__name__}: {exc}")
        if index % 20 == 0 or index == len(pending):
            print(f"  AU features {index}/{len(pending)}", flush=True)

    if au_map:
        ordered_paths = sorted(au_map)
        matrix = np.stack([au_map[path] for path in ordered_paths]).astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_path),
            paths=np.asarray(ordered_paths, dtype=object),
            features=matrix,
            feature_names=np.asarray(list(FEATURE_NAMES), dtype=object),
            profile_signature=np.asarray([profile_signature]),
        )
    return au_map, errors


def _collect_joint_matrix(
    pairs: list[dict[str, str]],
    label: int,
    *,
    video_features: dict[str, np.ndarray],
    au_features: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    missing: list[str] = []
    row_keys: list[str] = []
    for item in pairs:
        video_key = str(Path(item["video"]).resolve())
        au_key = str(Path(item["au"]).resolve())
        video_vec = video_features.get(video_key)
        au_vec = au_features.get(au_key)
        if video_vec is None or au_vec is None:
            missing.append(video_key)
            continue
        rows.append(np.concatenate([video_vec, au_vec], axis=0).astype(np.float32))
        labels.append(label)
        row_keys.append(video_key)
    if not rows:
        return (
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            missing,
            row_keys,
        )
    return (
        np.stack(rows).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        missing,
        row_keys,
    )


def _split_fit_validation(
    labels: np.ndarray,
    row_keys: list[str],
    *,
    seed: int,
    validation_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Split without putting resolution augmentations into validation."""
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) != len(row_keys):
        raise ValueError("Validation split labels and row keys are misaligned.")
    if len(labels) < 4 or len(np.unique(labels)) < 2:
        raise ValueError("Need at least four samples from both classes.")

    all_indices = np.arange(len(labels), dtype=np.int64)
    base_indices = np.asarray(
        [
            index
            for index, path in enumerate(row_keys)
            if not is_augmented_video(path)
        ],
        dtype=np.int64,
    )
    candidate_indices = base_indices
    candidate_counts = {
        int(label): int(np.sum(labels[candidate_indices] == label))
        for label in (0, 1)
    }
    if (
        len(candidate_indices) < 4
        or any(count < 2 for count in candidate_counts.values())
    ):
        candidate_indices = all_indices
        candidate_counts = {
            int(label): int(np.sum(labels[candidate_indices] == label))
            for label in (0, 1)
        }
    if any(count < 2 for count in candidate_counts.values()):
        raise ValueError(
            "Need at least two fit/validation candidates per class."
        )

    rng = np.random.default_rng(seed)
    validation_indices: list[int] = []
    for label in (0, 1):
        class_indices = candidate_indices[labels[candidate_indices] == label]
        validation_count = max(
            1,
            int(round(len(class_indices) * float(validation_ratio))),
        )
        validation_count = min(validation_count, len(class_indices) - 1)
        validation_indices.extend(
            int(index)
            for index in rng.choice(
                class_indices,
                size=validation_count,
                replace=False,
            )
        )

    val_idx = np.asarray(sorted(validation_indices), dtype=np.int64)
    val_set = set(int(index) for index in val_idx)
    fit_idx = np.asarray(
        [int(index) for index in all_indices if int(index) not in val_set],
        dtype=np.int64,
    )
    if len(np.unique(labels[fit_idx])) < 2 or len(np.unique(labels[val_idx])) < 2:
        raise ValueError("Unable to keep both classes in fit and validation.")
    return fit_idx, val_idx


def train_wangxing_joint_au_pt(
    *,
    manifest: dict[str, Any],
    cache_dir: Path,
    model_path: Path,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
    epochs: int = 80,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    seed: int = 42,
    device: str | torch.device | None = "cuda",
) -> dict[str, Any]:
    if "pairs" not in manifest:
        raise ValueError("Manifest is missing pairs; run prepare_res1k with AU attach.")

    torch_device = resolve_torch_device(
        str(device) if device is not None else "cuda"
    )
    _set_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device_name = (
        torch.cuda.get_device_name(torch_device)
        if torch_device.type == "cuda"
        else "cpu"
    )
    print(
        f"Joint MLP training device={torch_device} ({device_name}). "
        "Video/AU feature extraction stays on CPU.",
        flush=True,
    )

    cache_dir = Path(cache_dir)

    def _filter_train_pairs(
        pairs: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], list[str]]:
        kept: list[dict[str, str]] = []
        dropped: list[str] = []
        for item in pairs:
            if is_forbidden_train_video(item["video"]):
                dropped.append(item["video"])
                continue
            kept.append(item)
        return kept, dropped

    train_real_pairs, drop_tr = _filter_train_pairs(
        list(manifest["pairs"]["train"]["real"])
    )
    train_fake_pairs, drop_tf = _filter_train_pairs(
        list(manifest["pairs"]["train"]["fake"])
    )
    dropped_forbidden = drop_tr + drop_tf
    if dropped_forbidden:
        print(
            f"Dropped {len(dropped_forbidden)} forbidden train videos "
            "(data/test/AI Change clips never train).",
            flush=True,
        )
    pairs_test = list(manifest["pairs"]["test"]["real"]) + list(
        manifest["pairs"]["test"]["fake"]
    )
    pairs_train = train_real_pairs + train_fake_pairs
    all_pairs = pairs_train + pairs_test
    all_videos = [Path(item["video"]) for item in all_pairs]

    dual_cache = cache_dir / "wangxing_dual_f24s1024_f8s2048.npz"
    video_matrix, valid_videos, video_errors = build_dual_feature_table(
        all_videos,
        cache_path=dual_cache,
    )
    video_map = {
        str(path.resolve()): video_matrix[index]
        for index, path in enumerate(valid_videos)
    }

    au_map, au_errors = _extract_au_matrix(
        all_pairs,
        source_profile=source_profile,
        forensics_profiles=forensics_profiles,
        cache_path=cache_dir / "wangxing_joint_au25.npz",
    )

    x_tr, y_tr, miss_tr, keys_tr = _collect_joint_matrix(
        train_real_pairs,
        0,
        video_features=video_map,
        au_features=au_map,
    )
    x_tf, y_tf, miss_tf, keys_tf = _collect_joint_matrix(
        train_fake_pairs,
        1,
        video_features=video_map,
        au_features=au_map,
    )
    x_er, y_er, miss_er, _ = _collect_joint_matrix(
        list(manifest["pairs"]["test"]["real"]),
        0,
        video_features=video_map,
        au_features=au_map,
    )
    x_ef, y_ef, miss_ef, _ = _collect_joint_matrix(
        list(manifest["pairs"]["test"]["fake"]),
        1,
        video_features=video_map,
        au_features=au_map,
    )

    x_train = np.concatenate([x_tr, x_tf], axis=0)
    y_train = np.concatenate([y_tr, y_tf], axis=0)
    x_test = np.concatenate([x_er, x_ef], axis=0)
    y_test = np.concatenate([y_er, y_ef], axis=0)
    if len(x_train) < 8 or len(np.unique(y_train)) < 2:
        raise RuntimeError("联合训练集不足：需要同时包含带 AU 的真/假样本")
    if len(x_test) < 4 or len(np.unique(y_test)) < 2:
        raise RuntimeError("联合测试集不足：需要同时包含带 AU 的真/假样本")

    train_keys = keys_tr + keys_tf
    fit_idx, val_idx = _split_fit_validation(
        y_train,
        train_keys,
        seed=seed,
    )
    x_fit, mean, scale = _standardize(
        x_train[fit_idx],
        x_train[fit_idx],
    )
    x_fit = np.clip(x_fit, -8.0, 8.0).astype(np.float32)
    x_val = np.clip(
        (x_train[val_idx] - mean) / scale,
        -8.0,
        8.0,
    ).astype(np.float32)
    test_norm = np.clip(
        (x_test - mean) / scale,
        -8.0,
        8.0,
    ).astype(np.float32)
    y_fit = y_train[fit_idx].astype(np.float32)
    y_val = y_train[val_idx].astype(np.float32)

    expected_video_dim = int(x_fit.shape[1] - AU_DIM)
    if expected_video_dim <= 0:
        raise RuntimeError("Joint feature dim is invalid.")

    model = DualScaleClassifier(input_dim=int(x_fit.shape[1])).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=2e-4,
    )
    class_counts = np.bincount(y_fit.astype(np.int64), minlength=2)
    sample_weights = np.asarray(
        [1.0 / max(float(class_counts[int(label)]), 1.0) for label in y_fit],
        dtype=np.float64,
    )
    pin_memory = torch_device.type == "cuda"
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.ascontiguousarray(x_fit)),
            torch.from_numpy(np.ascontiguousarray(y_fit)),
        ),
        batch_size=max(1, int(batch_size)),
        sampler=WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        ),
        pin_memory=pin_memory,
    )
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_key = (-1.0, float("inf"))
    history: list[dict[str, Any]] = []

    for epoch in range(max(1, int(epochs))):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(torch_device, non_blocking=pin_memory)
            batch_y = batch_y.to(torch_device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        val_logits = _predict_logits(model, x_val, torch_device)
        val_prob = 1.0 / (1.0 + np.exp(-val_logits))
        val_metrics = _classification_metrics(y_val.astype(np.int64), val_prob)
        val_loss = float(
            criterion(
                torch.from_numpy(val_logits),
                torch.from_numpy(y_val),
            ).item()
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else math.inf,
                "validation_loss": val_loss,
                **val_metrics,
            }
        )
        if epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == int(epochs):
            print(
                f"epoch {epoch + 1}/{epochs} "
                f"train_loss={history[-1]['train_loss']:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_bacc={val_metrics['balanced_accuracy']:.4f}",
                flush=True,
            )
        key = (val_metrics["balanced_accuracy"], -val_loss)
        if key > best_key:
            best_key = key
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    val_logits = _predict_logits(model, x_val, torch_device)
    temperature = _fit_temperature(val_logits, y_val)
    test_logits = _predict_logits(model, test_norm, torch_device)
    test_prob_gen = 1.0 / (1.0 + np.exp(-test_logits / max(temperature, 1e-6)))
    pred_gen = (test_prob_gen >= 0.5).astype(np.int64)
    y_true = y_test.astype(np.int64)
    tp = int(((y_true == 1) & (pred_gen == 1)).sum())
    tn = int(((y_true == 0) & (pred_gen == 0)).sum())
    fp = int(((y_true == 0) & (pred_gen == 1)).sum())
    fn = int(((y_true == 1) & (pred_gen == 0)).sum())
    headline = {
        "generated_recall": tp / (tp + fn) if tp + fn else None,
        "overall_accuracy": (tp + tn) / len(y_true) if len(y_true) else None,
        "generated_precision": tp / (tp + fp) if tp + fp else None,
        "real_recall": tn / (tn + fp) if tn + fp else None,
        "coverage": (
            len(y_true)
            / max(
                len(manifest["pairs"]["test"]["real"])
                + len(manifest["pairs"]["test"]["fake"]),
                1,
            )
        ),
    }
    test_metrics = _classification_metrics(y_true, test_prob_gen)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    checkpoint = {
        "model_type": JOINT_MODEL_TYPE,
        "feature_version": FEATURE_VERSION,
        "config": {
            "scales": [SCALE_A, SCALE_B],
            "input_dim": int(x_fit.shape[1]),
            "video_dim": expected_video_dim,
            "au_dim": AU_DIM,
            "au_feature_names": list(FEATURE_NAMES),
            "threshold_generated": 0.5,
            "probability_target": "generated",
            "fusion_mode": "early_concat",
        },
        "model_state": cpu_state,
        "feature_mean": mean.astype(np.float32),
        "feature_scale": scale.astype(np.float32),
        "temperature": float(temperature),
        "device_used": str(torch_device),
        "dataset": {
            "train_real": int(len(y_tr)),
            "train_fake": int(len(y_tf)),
            "test_real": int(len(y_er)),
            "test_fake": int(len(y_ef)),
            "dropped_forbidden_train": dropped_forbidden,
            "missing": {
                "train_real": miss_tr,
                "train_fake": miss_tf,
                "test_real": miss_er,
                "test_fake": miss_ef,
            },
            "video_extract_errors_preview": video_errors[:20],
            "au_extract_errors_preview": au_errors[:20],
        },
        "validation": {
            "fit_count": int(len(fit_idx)),
            "validation_count": int(len(val_idx)),
            "validation_augmented_count": int(
                sum(
                    is_augmented_video(train_keys[index])
                    for index in val_idx
                )
            ),
            "normalization_fit": "fit_subset_only",
        },
        "train_val_metrics_tail": history[-10:],
        "test_headline": headline,
        "test_metrics": test_metrics,
        "confusion": {
            "tp_generated": tp,
            "tn_real": tn,
            "fp_real_as_generated": fp,
            "fn_generated_as_real": fn,
        },
    }
    torch.save(checkpoint, model_path)
    return {
        "model_path": str(model_path),
        "cache_dir": str(cache_dir),
        "headline": headline,
        "confusion": checkpoint["confusion"],
        "counts": checkpoint["dataset"],
        "temperature": float(temperature),
        "input_dim": int(x_fit.shape[1]),
        "video_dim": expected_video_dim,
        "au_dim": AU_DIM,
        "device": str(torch_device),
        "dropped_forbidden_train": dropped_forbidden,
        "validation": checkpoint["validation"],
    }


def predict_wangxing_joint_au_pt(
    *,
    video_path: str | Path,
    au_path: str | Path,
    model_path: str | Path,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
) -> dict[str, Any]:
    video_path = Path(video_path).expanduser().resolve()
    au_path = Path(au_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    checkpoint = torch.load(str(model_path), map_location="cpu")
    if checkpoint.get("model_type") != JOINT_MODEL_TYPE:
        raise ValueError(f"Unsupported model_type: {checkpoint.get('model_type')}")
    scales = checkpoint["config"]["scales"]
    video_vec = extract_dual_feature(
        video_path,
        scale_a=scales[0],
        scale_b=scales[1],
    )
    au_vec, au_dict = extract_fusion_features(
        au_path=au_path,
        wangxing_source_profile=source_profile,
        forensics_profiles=forensics_profiles,
    )
    feature = np.concatenate(
        [video_vec.astype(np.float32), np.asarray(au_vec, dtype=np.float32)],
        axis=0,
    )
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
    if feature.shape[0] != mean.shape[0]:
        raise ValueError(
            f"Feature dim mismatch: got {feature.shape[0]}, expected {mean.shape[0]}"
        )
    normalized = np.clip((feature - mean) / np.maximum(scale, 1e-4), -8.0, 8.0)
    model = DualScaleClassifier(input_dim=int(checkpoint["config"]["input_dim"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    logit = float(_predict_classifier_logits(model, normalized[None, :])[0])
    temperature = float(checkpoint.get("temperature", 1.0))
    p_gen = float(1.0 / (1.0 + math.exp(-logit / max(temperature, 1e-6))))
    p_gen = min(0.98, max(0.02, p_gen))
    decision = "generated" if p_gen >= 0.5 else "real"
    return {
        "prediction": decision,
        "generated_probability": round(p_gen, 4),
        "real_probability": round(1.0 - p_gen, 4),
        "logit": logit,
        "temperature": temperature,
        "model_path": str(model_path),
        "video_path": str(video_path),
        "au_path": str(au_path),
        "fusion_mode": "early_concat",
        "au_quality_min": float(au_dict.get("quality_min", 0.5)),
    }


def evaluate_holdout_joint_au_pt(
    *,
    holdout_manifest: Path,
    model_path: Path,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
) -> dict[str, Any]:
    holdout = _load_json(holdout_manifest)
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    preds: list[int] = []
    samples = []
    for item in holdout.get("real", []):
        samples.append((0, "real", item))
    for item in holdout.get("seedance", []):
        samples.append((1, "generated", item))

    for index, (label, source_label, item) in enumerate(samples, start=1):
        if not isinstance(item, dict) or not item.get("video"):
            continue
        video = project_path(str(item["video"]))
        au = resolve_au_csv_for_video(
            video,
            au_hint=item.get("au"),
        )
        if au is None or not video.is_file():
            rows.append(
                {
                    "index": index,
                    "source_label": source_label,
                    "label_generated": label,
                    "status": "missing_inputs",
                    "video": str(video),
                    "au": None if au is None else str(au),
                }
            )
            continue
        scored = predict_wangxing_joint_au_pt(
            video_path=video,
            au_path=au,
            model_path=model_path,
            source_profile=source_profile,
            forensics_profiles=forensics_profiles,
        )
        pred = 1 if scored["prediction"] == "generated" else 0
        labels.append(label)
        preds.append(pred)
        rows.append(
            {
                "index": index,
                "source_label": source_label,
                "label_generated": label,
                "status": "ok",
                "video": str(video),
                "au": str(au),
                **scored,
            }
        )
        print(
            f"[{index}/{len(samples)}] {source_label} "
            f"pred={scored['prediction']} "
            f"p_gen={scored['generated_probability']}",
            flush=True,
        )

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(preds, dtype=np.int64)
    if len(y) == 0:
        headline = {
            "generated_recall": None,
            "overall_accuracy": None,
            "generated_precision": None,
            "real_recall": None,
            "coverage": 0.0,
        }
        confusion = {
            "tp_generated": 0,
            "tn_real": 0,
            "fp_real_as_generated": 0,
            "fn_generated_as_real": 0,
        }
    else:
        tp = int(((y == 1) & (p == 1)).sum())
        tn = int(((y == 0) & (p == 0)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        headline = {
            "generated_recall": tp / (tp + fn) if tp + fn else None,
            "overall_accuracy": (tp + tn) / len(y),
            "generated_precision": tp / (tp + fp) if tp + fp else None,
            "real_recall": tn / (tn + fp) if tn + fp else None,
            "coverage": len(y) / len(samples) if samples else 0.0,
        }
        confusion = {
            "tp_generated": tp,
            "tn_real": tn,
            "fp_real_as_generated": fp,
            "fn_generated_as_real": fn,
        }
    return {
        "schema_version": "wangxing_joint_au_pt_holdout_metrics_v1",
        "model_path": str(model_path),
        "holdout_manifest": str(holdout_manifest),
        "headline": headline,
        "confusion": confusion,
        "rows": rows,
    }

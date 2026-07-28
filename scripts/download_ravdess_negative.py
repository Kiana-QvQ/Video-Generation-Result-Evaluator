from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAVDESS_RECORD_ID = "1188976"
RAVDESS_SOURCE_URLS = {
    "ZENODO": (
        "https://zenodo.org/records/{record_id}/files/"
        "Video_Speech_Actor_{actor:02d}.zip?download=1"
    ),
    "HUGGINGFACE": (
        "https://huggingface.co/datasets/HoangPhuc7679/RAVDESS/"
        "resolve/main/Video_Speech_Actor_{actor:02d}.zip?download=true"
    ),
}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
RAVDESS_MEMBER_RE = re.compile(
    r"(?P<modality>\d{2})-"
    r"(?P<vocal_channel>\d{2})-"
    r"(?P<emotion>\d{2})-"
    r"(?P<intensity>\d{2})-"
    r"(?P<statement>\d{2})-"
    r"(?P<repetition>\d{2})-"
    r"(?P<actor>\d{2})"
    r"(?P<suffix>\.[A-Za-z0-9]+)$",
    re.IGNORECASE,
)


def parse_int_list(value: str, *, name: str) -> list[int]:
    values: list[int] = []
    for token in re.split(r"[\s,;]+", str(value).strip()):
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid {name} value: {token!r}") from exc
    if not values:
        raise ValueError(f"{name} cannot be empty.")
    return list(dict.fromkeys(values))


def parse_ravdess_member(member_name: str) -> dict[str, str] | None:
    name = PurePosixPath(str(member_name).replace("\\", "/")).name
    match = RAVDESS_MEMBER_RE.fullmatch(name)
    if not match or match.group("suffix").lower() not in VIDEO_SUFFIXES:
        return None
    result = match.groupdict()
    result["filename"] = name
    result["emotion_name"] = RAVDESS_EMOTIONS.get(
        result["emotion"],
        "unknown",
    )
    return result


def _safe_member_name(member_name: str) -> str:
    name = PurePosixPath(str(member_name).replace("\\", "/")).name
    if name in {"", ".", ".."}:
        raise ValueError(f"Invalid archive member name: {member_name!r}")
    return name


def _download_with_resume_once(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    headers = {"User-Agent": "VideoEvaluator/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code != 416 or not existing:
            raise
        print(f"Using already complete archive: {destination}")
        return

    status = getattr(response, "status", response.getcode())
    if existing and status == 206:
        mode = "ab"
        downloaded = existing
    else:
        mode = "wb"
        downloaded = 0

    content_length = response.headers.get("Content-Length")
    expected = None
    if content_length:
        expected = downloaded + int(content_length)
    content_range = response.headers.get("Content-Range", "")
    range_match = re.search(r"/(\d+)$", content_range)
    if range_match:
        expected = int(range_match.group(1))

    print(
        f"Downloading {url} -> {destination}"
        + (f" (resume at {existing} bytes)" if existing else "")
    )
    with response, destination.open(mode) as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)

    if expected is not None and downloaded != expected:
        raise IOError(
            f"Incomplete download for {destination}: "
            f"{downloaded} of {expected} bytes."
        )


def _download_with_resume(
    url: str,
    destination: Path,
    *,
    max_attempts: int = 5,
) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            _download_with_resume_once(url, destination)
            return
        except (OSError, urllib.error.URLError) as exc:
            if attempt >= max_attempts:
                raise
            wait_seconds = min(30, 3 * attempt)
            print(
                f"Download interrupted ({type(exc).__name__}: {exc}). "
                f"Retrying in {wait_seconds}s "
                f"({attempt}/{max_attempts - 1})...",
                flush=True,
            )
            time.sleep(wait_seconds)


def ensure_archive(
    actor: int,
    *,
    cache_root: Path,
    record_id: str,
    source: str,
    force: bool,
) -> Path:
    archive = cache_root / f"Video_Speech_Actor_{actor:02d}.zip"
    partial = cache_root / f"Video_Speech_Actor_{actor:02d}.zip.part"
    if force and archive.exists():
        archive.unlink()
    if force and partial.exists():
        partial.unlink()
    if archive.exists():
        try:
            with zipfile.ZipFile(archive) as handle:
                handle.infolist()
            return archive
        except zipfile.BadZipFile:
            archive.unlink()

    try:
        url_template = RAVDESS_SOURCE_URLS[source]
    except KeyError as exc:
        raise ValueError(f"Unsupported RAVDESS source: {source}") from exc
    url = url_template.format(record_id=record_id, actor=actor)
    _download_with_resume(url, partial)
    try:
        with zipfile.ZipFile(partial) as handle:
            handle.infolist()
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"Downloaded file is not a valid ZIP: {partial}. "
            "The official Zenodo URL may have returned an error page."
        ) from exc
    partial.replace(archive)
    return archive


def list_video_members(
    archive: Path,
    *,
    allowed_emotions: Iterable[int],
) -> list[dict[str, str]]:
    allowed = {f"{value:02d}" for value in allowed_emotions}
    with zipfile.ZipFile(archive) as handle:
        records: list[dict[str, str]] = []
        for info in handle.infolist():
            parsed = parse_ravdess_member(info.filename)
            if parsed is None or parsed["emotion"] not in allowed:
                continue
            parsed["member_name"] = info.filename
            records.append(parsed)
    return records


def select_balanced_members(
    records: Iterable[dict[str, str]],
    *,
    max_videos: int,
    seed: int,
) -> list[dict[str, str]]:
    if max_videos <= 0:
        raise ValueError("max_videos must be positive.")

    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for record in records:
        key = (record["actor"], record["emotion"])
        groups.setdefault(key, []).append(record)
    if not groups:
        return []

    rng = random.Random(seed)
    queues = []
    for key in sorted(groups):
        values = list(groups[key])
        rng.shuffle(values)
        queues.append(values)

    selected: list[dict[str, str]] = []
    while len(selected) < max_videos:
        progressed = False
        for queue in queues:
            if len(selected) >= max_videos:
                break
            if queue:
                selected.append(queue.pop())
                progressed = True
        if not progressed:
            break
    return sorted(
        selected,
        key=lambda item: (
            item["actor"],
            item["emotion"],
            item["intensity"],
            item["filename"],
        ),
    )


def _copy_selected_members(
    archives: dict[int, Path],
    selected: Iterable[dict[str, str]],
    *,
    output_root: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, selected_record in enumerate(selected, start=1):
        actor = int(selected_record["actor"])
        source_archive = archives[actor]
        filename = _safe_member_name(selected_record["member_name"])
        target = output_root / "videos" / f"actor_{actor:02d}" / filename
        target.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(source_archive) as handle:
            with handle.open(selected_record["member_name"]) as source:
                with target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

        records.append(
            {
                "clip_id": f"ravdess_actor_{actor:02d}/{filename}",
                "person": f"ravdess_actor_{actor:02d}",
                "performance": "ravdess_speech",
                "relative_path": target.relative_to(output_root).as_posix(),
                "local_path": str(target.resolve()),
                "source_archive": source_archive.name,
                "source_member": selected_record["member_name"],
                "ravdess_actor": actor,
                "ravdess_emotion_code": int(selected_record["emotion"]),
                "ravdess_emotion": selected_record["emotion_name"],
                "ravdess_intensity": int(selected_record["intensity"]),
                "phase1_usable": True,
                "is_emotion": True,
                "expression_class": "negative_identity",
                "metadata_source": "RAVDESS",
            }
        )
    return records


def write_manifest(
    records: list[dict[str, object]],
    *,
    output_root: Path,
    actors: list[int],
    emotions: list[int],
    record_id: str,
    seed: int,
) -> Path:
    payload = {
        "schema_version": "negative_video_manifest_v1",
        "source": "RAVDESS",
        "source_record": f"https://zenodo.org/records/{record_id}",
        "selection": {
            "actors": actors,
            "emotions": emotions,
            "seed": seed,
            "download_policy": (
                "Only the selected actor ZIP archives are downloaded; "
                "only selected video members are extracted."
            ),
        },
        "records": records,
    }
    manifest_path = output_root / "negative_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download a bounded RAVDESS video subset for AU negative "
            "identity training."
        )
    )
    parser.add_argument(
        "--actors",
        default="1,2",
        help="Comma-separated actor IDs, for example 1,2,3.",
    )
    parser.add_argument(
        "--emotions",
        default="1,2,3,4,5,6,7,8",
        help="RAVDESS emotion codes to include.",
    )
    parser.add_argument("--max-videos", type=int, default=48)
    parser.add_argument(
        "--output-root",
        default="data/negative/ravdess",
    )
    parser.add_argument(
        "--cache-root",
        default="data/cache/ravdess",
    )
    parser.add_argument("--record-id", default=RAVDESS_RECORD_ID)
    parser.add_argument(
        "--source",
        choices=tuple(sorted(RAVDESS_SOURCE_URLS)),
        default="ZENODO",
        help="Download source. HUGGINGFACE is often faster than Zenodo.",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        actors = parse_int_list(args.actors, name="actors")
        emotions = parse_int_list(args.emotions, name="emotions")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    invalid_actors = [value for value in actors if not 1 <= value <= 24]
    invalid_emotions = [value for value in emotions if not 1 <= value <= 8]
    if invalid_actors:
        raise SystemExit(f"Actor IDs must be in 1..24: {invalid_actors}")
    if invalid_emotions:
        raise SystemExit(f"Emotion codes must be in 1..8: {invalid_emotions}")

    output_root = Path(args.output_root)
    cache_root = Path(args.cache_root)
    archives: dict[int, Path] = {}
    all_members: list[dict[str, str]] = []
    for actor in actors:
        try:
            archive = ensure_archive(
                actor,
                cache_root=cache_root,
                record_id=str(args.record_id),
                source=str(args.source),
                force=bool(args.force),
            )
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            expected_archive = (
                cache_root / f"Video_Speech_Actor_{actor:02d}.zip"
            )
            url_template = RAVDESS_SOURCE_URLS[str(args.source)]
            source_url = url_template.format(
                record_id=str(args.record_id),
                actor=actor,
            )
            raise SystemExit(
                "RAVDESS download failed. Check network access or download "
                "the official actor ZIP manually. "
                f"Expected local file: {expected_archive}. "
                f"Source URL: {source_url}. "
                f"Details: {exc}"
            ) from exc
        archives[actor] = archive
        all_members.extend(
            list_video_members(
                archive,
                allowed_emotions=emotions,
            )
        )

    selected = select_balanced_members(
        all_members,
        max_videos=int(args.max_videos),
        seed=int(args.seed),
    )
    if not selected:
        raise SystemExit(
            "No matching RAVDESS videos were found in the selected archives."
        )
    records = _copy_selected_members(
        archives,
        selected,
        output_root=output_root,
    )
    manifest_path = write_manifest(
        records,
        output_root=output_root,
        actors=actors,
        emotions=emotions,
        record_id=str(args.record_id),
        seed=int(args.seed),
    )
    print(
        json.dumps(
            {
                "source": "RAVDESS",
                "actors": actors,
                "selected_videos": len(records),
                "emotion_counts": {
                    name: sum(
                        item["ravdess_emotion"] == name for item in records
                    )
                    for name in sorted(RAVDESS_EMOTIONS.values())
                },
                "manifest": str(manifest_path),
                "output_root": str(output_root),
                "cache_root": str(cache_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

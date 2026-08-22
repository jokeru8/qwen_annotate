"""Read-only inspection of LeRobot v2.1 datasets."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from pydantic import BaseModel

from qwen_annotate.config import AnnotationConfig


class VideoProbe(BaseModel):
    frames: int
    fps: float
    width: int
    height: int


class EpisodeInfo(BaseModel):
    episode_index: int
    length: int
    task: str
    parquet: Path
    videos: dict[str, Path]


class DatasetIndex(BaseModel):
    root: Path
    version: str
    fps: float
    camera_keys: list[str]
    episodes: list[EpisodeInfo]


def probe_video(path: Path) -> VideoProbe:
    """Read basic metadata from a video without changing it."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to probe videos") from exc
    try:
        container = av.open(str(path))
    except Exception as exc:  # PyAV exposes several codec and I/O exceptions.
        raise ValueError(f"Unable to open video {path}: {exc}") from exc

    try:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"Video {path} has no video stream")
        if stream.average_rate is None or float(stream.average_rate) <= 0:
            raise ValueError(f"Video {path} has invalid fps")
        if stream.width <= 0 or stream.height <= 0:
            raise ValueError(f"Video {path} has invalid dimensions")
        frames = stream.frames
        if frames <= 0:
            frames = sum(1 for _ in container.decode(stream))
        if frames <= 0:
            raise ValueError(f"Video {path} has no frames")
        return VideoProbe(
            frames=frames,
            fps=float(stream.average_rate),
            width=stream.width,
            height=stream.height,
        )
    finally:
        container.close()


def inspect_dataset(
    config: AnnotationConfig,
    probe: Callable[[Path], VideoProbe] = probe_video,
) -> DatasetIndex:
    """Validate and index a local LeRobot v2.1 dataset without writing to it."""
    root = config.source
    info = _read_json_object(root / "meta" / "info.json")
    version = _required_string(info, "codebase_version", "info.json")
    if version != "v2.1":
        raise ValueError(f"LeRobot codebase_version must be v2.1, got {version!r}")

    fps = _positive_number(info, "fps", "info.json")
    chunks_size = _positive_int(info, "chunks_size", "info.json")
    total_episodes = _nonnegative_int(info, "total_episodes", "info.json")
    expected_total_frames = _nonnegative_int(info, "total_frames", "info.json")
    data_path_format = _required_string(info, "data_path", "info.json")
    video_path_format = _required_string(info, "video_path", "info.json")
    camera_keys = _camera_keys(info)
    _require_configured_cameras(config, camera_keys)
    task_texts = _task_texts(root / "meta" / "tasks.jsonl")
    episode_rows = _read_jsonl(root / "meta" / "episodes.jsonl")

    if len(episode_rows) != total_episodes:
        raise ValueError(
            f"info.total_episodes is {total_episodes}, but episodes.jsonl has {len(episode_rows)} rows"
        )

    episodes: list[EpisodeInfo] = []
    total_frames = 0
    for expected_index, row in enumerate(episode_rows):
        episode_index = _required_int(row, "episode_index", f"episodes.jsonl row {expected_index}")
        if episode_index != expected_index:
            raise ValueError("Episode indices must be contiguous from 0 through N-1")
        length = _positive_int(row, "length", f"episodes.jsonl row {expected_index}")
        task = _episode_task(row, expected_index)
        if task not in task_texts:
            raise ValueError(
                f"Episode {episode_index} references task {task!r}, absent from meta/tasks.jsonl"
            )
        values = {
            "episode_chunk": episode_index // chunks_size,
            "episode_index": episode_index,
        }
        parquet = root / _format_path(data_path_format, values, "data_path")
        _verify_parquet_rows(parquet, length, episode_index)
        videos: dict[str, Path] = {}
        for video_key in camera_keys:
            video = root / _format_path(
                video_path_format, values | {"video_key": video_key}, "video_path"
            )
            if not video.is_file():
                raise FileNotFoundError(f"Missing video for episode {episode_index}: {video}")
            video_probe = probe(video)
            if video_probe.frames != length:
                raise ValueError(
                    f"Video frame count for episode {episode_index}, camera {video_key!r} "
                    f"is {video_probe.frames}, expected {length}"
                )
            if abs(video_probe.fps - fps) > 0.01:
                raise ValueError(
                    f"Video fps for episode {episode_index}, camera {video_key!r} "
                    f"is {video_probe.fps}, expected {fps}"
                )
            videos[video_key] = video
        episodes.append(
            EpisodeInfo(
                episode_index=episode_index,
                length=length,
                task=task,
                parquet=parquet,
                videos=videos,
            )
        )
        total_frames += length

    if total_frames != expected_total_frames:
        raise ValueError(
            f"info.total_frames is {expected_total_frames}, but episode lengths total {total_frames}"
        )
    if "total_videos" in info:
        total_videos = _nonnegative_int(info, "total_videos", "info.json")
        expected_total_videos = total_episodes * len(camera_keys)
        if total_videos != expected_total_videos:
            raise ValueError(
                f"info.total_videos is {total_videos}, expected {expected_total_videos}"
            )
    return DatasetIndex(
        root=root,
        version=version,
        fps=fps,
        camera_keys=camera_keys,
        episodes=episodes,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required metadata file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Malformed JSON in {path}: expected an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required metadata file: {path}")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Unable to read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"Malformed JSONL in {path}: blank line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL in {path} line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Malformed JSONL in {path} line {line_number}: expected object")
        rows.append(value)
    return rows


def _camera_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("Malformed info.json: features must be an object")
    cameras = [
        key
        for key, value in features.items()
        if isinstance(key, str) and isinstance(value, dict) and value.get("dtype") == "video"
    ]
    if not cameras:
        raise ValueError("Malformed info.json: no video features found")
    return cameras


def _require_configured_cameras(config: AnnotationConfig, camera_keys: list[str]) -> None:
    for camera in [config.primary_camera, *config.refine_cameras]:
        if camera not in camera_keys:
            raise ValueError(f"Configured camera {camera!r} is not a video feature in info.json")


def _task_texts(path: Path) -> set[str]:
    tasks: set[str] = set()
    for row_number, row in enumerate(_read_jsonl(path), start=1):
        tasks.add(_required_string(row, "task", f"tasks.jsonl row {row_number}"))
    return tasks


def _episode_task(row: dict[str, Any], row_number: int) -> str:
    tasks = row.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"episodes.jsonl row {row_number}: tasks must be a nonempty list")
    task = tasks[0]
    if not isinstance(task, str) or not task:
        raise ValueError(f"episodes.jsonl row {row_number}: first task must be a nonempty string")
    return task


def _verify_parquet_rows(path: Path, length: int, episode_index: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing parquet for episode {episode_index}: {path}")
    try:
        row_count = pq.ParquetFile(path).metadata.num_rows
    except Exception as exc:
        raise ValueError(f"Unable to read parquet metadata for episode {episode_index}: {path}") from exc
    if row_count != length:
        raise ValueError(
            f"Parquet row count for episode {episode_index} is {row_count}, expected {length}"
        )


def _format_path(template: str, values: dict[str, Any], field: str) -> Path:
    try:
        return Path(template.format(**values))
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(f"Malformed {field} format in info.json: {exc}") from exc


def _required_string(value: dict[str, Any], key: str, context: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"Malformed {context}: {key} must be a nonempty string")
    return item


def _required_int(value: dict[str, Any], key: str, context: str) -> int:
    item = value.get(key)
    if type(item) is not int:
        raise ValueError(f"Malformed {context}: {key} must be an integer")
    return item


def _positive_int(value: dict[str, Any], key: str, context: str) -> int:
    item = _required_int(value, key, context)
    if item <= 0:
        raise ValueError(f"Malformed {context}: {key} must be positive")
    return item


def _nonnegative_int(value: dict[str, Any], key: str, context: str) -> int:
    item = _required_int(value, key, context)
    if item < 0:
        raise ValueError(f"Malformed {context}: {key} must be nonnegative")
    return item


def _positive_number(value: dict[str, Any], key: str, context: str) -> float:
    item = value.get(key)
    if type(item) not in (int, float) or item <= 0:
        raise ValueError(f"Malformed {context}: {key} must be positive")
    return float(item)

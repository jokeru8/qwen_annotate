"""Read-only inspection of LeRobot v2.1 datasets."""

import json
import math
from collections.abc import Callable
from pathlib import Path
from string import Formatter
from typing import Any

import pyarrow.parquet as pq
from pydantic import BaseModel, Field, field_validator

from qwen_annotate.config import AnnotationConfig


VIDEO_FPS_TOLERANCE = 0.01


def video_fps_matches(measured: float, expected: float) -> bool:
    """Use one import/release tolerance for finite positive video frame rates."""
    return (
        isinstance(measured, (int, float))
        and not isinstance(measured, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and math.isfinite(measured)
        and math.isfinite(expected)
        and measured > 0
        and expected > 0
        and abs(float(measured) - float(expected)) <= VIDEO_FPS_TOLERANCE
    )


class VideoProbe(BaseModel):
    frames: int = Field(ge=0)
    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @field_validator("fps")
    @classmethod
    def fps_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fps must be finite")
        return value


class EpisodeInfo(BaseModel):
    episode_index: int
    length: int
    task: str
    parquet: Path
    videos: dict[str, Path]


class DatasetIndex(BaseModel):
    root: Path
    version: str
    fps: float = Field(gt=0)
    camera_keys: list[str]
    episodes: list[EpisodeInfo]

    @field_validator("fps")
    @classmethod
    def fps_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fps must be finite")
        return value


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
        if stream.average_rate is None:
            raise ValueError(f"Video {path} has invalid fps")
        try:
            fps = float(stream.average_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Video {path} has invalid fps") from exc
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Video {path} has invalid fps")
        if stream.width <= 0 or stream.height <= 0:
            raise ValueError(f"Video {path} has invalid dimensions")
        frames = stream.frames
        if type(frames) is not int or frames <= 0:
            frames = sum(1 for _ in container.decode(stream))
        if frames <= 0:
            raise ValueError(f"Video {path} has no frames")
        return VideoProbe(
            frames=frames,
            fps=fps,
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
    root = config.source.resolve()
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
    _validate_path_template(data_path_format, "data_path", {"episode_index"})
    _validate_path_template(video_path_format, "video_path", {"episode_index", "video_key"})
    camera_keys = _camera_keys(info)
    _require_configured_cameras(config, camera_keys)
    task_texts = _task_texts(root / "meta" / "tasks.jsonl")
    total_tasks = _nonnegative_int(info, "total_tasks", "info.json")
    if len(task_texts) != total_tasks:
        raise ValueError(f"info.total_tasks is {total_tasks}, but tasks.jsonl has {len(task_texts)} rows")
    episode_rows = _read_jsonl(root / "meta" / "episodes.jsonl")

    if len(episode_rows) != total_episodes:
        raise ValueError(
            f"info.total_episodes is {total_episodes}, but episodes.jsonl has {len(episode_rows)} rows"
        )
    total_chunks = _nonnegative_int(info, "total_chunks", "info.json")
    expected_chunks = (total_episodes + chunks_size - 1) // chunks_size
    if total_chunks != expected_chunks:
        raise ValueError(f"info.total_chunks is {total_chunks}, expected {expected_chunks}")

    episodes: list[EpisodeInfo] = []
    total_frames = 0
    seen_parquets: set[Path] = set()
    seen_videos: set[Path] = set()
    for expected_index, row in enumerate(episode_rows):
        episode_index = _required_int(row, "episode_index", f"episodes.jsonl row {expected_index}")
        if episode_index != expected_index:
            raise ValueError("Episode indices must be contiguous from 0 through N-1")
        length = _positive_int(row, "length", f"episodes.jsonl row {expected_index}")
        episode_tasks = _episode_tasks(row, expected_index)
        for task in episode_tasks:
            if task not in task_texts.values():
                raise ValueError(
                    f"Episode {episode_index} references task {task!r}, absent from meta/tasks.jsonl"
                )
        if len(episode_tasks) != 1:
            raise ValueError(
                "First release supports exactly one task per episode; "
                f"episode {episode_index} has {len(episode_tasks)}"
            )
        task = episode_tasks[0]
        values = {
            "episode_chunk": episode_index // chunks_size,
            "episode_index": episode_index,
        }
        parquet = _resolve_dataset_path(root, _format_path(data_path_format, values, "data_path"), "data_path")
        if parquet in seen_parquets:
            raise ValueError(f"data_path resolves duplicate parquet path: {parquet}")
        seen_parquets.add(parquet)
        _verify_parquet_rows(parquet, length, episode_index)
        videos: dict[str, Path] = {}
        for video_key in camera_keys:
            video = _resolve_dataset_path(
                root,
                _format_path(video_path_format, values | {"video_key": video_key}, "video_path"),
                "video_path",
            )
            if video in seen_videos:
                raise ValueError(f"video_path resolves duplicate video path: {video}")
            seen_videos.add(video)
            if not video.is_file():
                raise FileNotFoundError(f"Missing video for episode {episode_index}: {video}")
            video_probe = probe(video)
            if video_probe.frames != length:
                raise ValueError(
                    f"Video frame count for episode {episode_index}, camera {video_key!r} "
                    f"is {video_probe.frames}, expected {length}"
                )
            if not video_fps_matches(video_probe.fps, fps):
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
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
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
            value = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
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


def _task_texts(path: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    for row_number, row in enumerate(_read_jsonl(path), start=1):
        task_index = _nonnegative_int(row, "task_index", f"tasks.jsonl row {row_number}")
        if task_index in tasks:
            raise ValueError(f"tasks.jsonl has duplicate task_index {task_index}")
        task = _required_string(row, "task", f"tasks.jsonl row {row_number}")
        if task in tasks.values():
            raise ValueError(f"tasks.jsonl has duplicate task text {task!r}")
        tasks[task_index] = task
    if set(tasks) != set(range(len(tasks))):
        raise ValueError("tasks.jsonl task_index values must be contiguous from 0 through N-1")
    return tasks


def _episode_tasks(row: dict[str, Any], row_number: int) -> list[str]:
    tasks = row.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"episodes.jsonl row {row_number}: tasks must be a nonempty list")
    if any(not isinstance(task, str) or not task for task in tasks):
        raise ValueError(f"episodes.jsonl row {row_number}: tasks must contain nonempty strings")
    return tasks


def _verify_parquet_rows(path: Path, length: int, episode_index: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing parquet for episode {episode_index}: {path}")
    parquet_file: pq.ParquetFile | None = None
    try:
        parquet_file = pq.ParquetFile(path)
        row_count = parquet_file.metadata.num_rows
    except Exception as exc:
        raise ValueError(f"Unable to read parquet metadata for episode {episode_index}: {path}") from exc
    finally:
        if parquet_file is not None:
            parquet_file.close()
    if row_count != length:
        raise ValueError(
            f"Parquet row count for episode {episode_index} is {row_count}, expected {length}"
        )


def _format_path(template: str, values: dict[str, Any], field: str) -> Path:
    try:
        return Path(template.format(**values))
    except Exception as exc:
        raise ValueError(f"Malformed {field} format in info.json: {exc}") from exc


def _validate_path_template(template: str, field: str, required_fields: set[str]) -> None:
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"Malformed {field} format in info.json: {exc}") from exc
    fields: set[str] = set()
    allowed_fields = {"episode_chunk", "episode_index", "video_key"}
    for _, field_name, format_spec, _ in parsed:
        if field_name is None:
            continue
        if field_name not in allowed_fields:
            raise ValueError(f"Malformed {field} format in info.json: unsupported field {field_name!r}")
        if "{" in format_spec or "}" in format_spec:
            raise ValueError(f"Malformed {field} format in info.json: nested format fields are not allowed")
        fields.add(field_name)
    missing = required_fields - fields
    if missing:
        raise ValueError(
            f"Malformed {field} format in info.json: missing required field(s) {sorted(missing)}"
        )


def _resolve_dataset_path(root: Path, formatted: Path, field: str) -> Path:
    if formatted.is_absolute():
        raise ValueError(f"Malformed {field} format in info.json: absolute paths are not allowed")
    target = (root / formatted).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Malformed {field} format in info.json: path escapes dataset root") from exc
    return target


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


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
    if type(item) not in (int, float) or not math.isfinite(item) or item <= 0:
        raise ValueError(f"Malformed {context}: {key} must be positive")
    return float(item)

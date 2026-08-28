"""Strict, read-only inspection of LeRobot v3.0 shared-shard datasets."""

from __future__ import annotations

import math
import os
import re
import stat
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from string import Formatter
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from robo_annotate.config import AnnotationConfig

from .lerobot import (
    DatasetIndex,
    EpisodeDataRef,
    EpisodeInfo,
    EpisodeVideoRef,
    VideoProbe,
    _camera_keys,
    _nonnegative_int,
    _positive_int,
    _positive_number,
    _require_configured_cameras,
    _required_string,
    data_timestamp_matches,
    video_fps_matches,
)
from .v30_depth import depth_metadata, is_depth_feature


DATA_FIELDS = {"chunk_index", "file_index"}
VIDEO_FIELDS = {"video_key", "chunk_index", "file_index"}
REQUIRED_EPISODE_COLUMNS = {
    "episode_index",
    "tasks",
    "length",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "dataset_to_index",
    "meta/episodes/chunk_index",
    "meta/episodes/file_index",
}

_DATA_ROW_COLUMNS = ("index", "episode_index", "frame_index", "timestamp", "task_index")
_VIDEO_COLUMN_PARTS = ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
_STAT_METRICS = {"min", "max", "mean", "std", "count"}
_CHUNK_PATTERN = re.compile(r"chunk-([0-9]{3})")
_FILE_PATTERN = re.compile(r"file-([0-9]{3})\.parquet")


def read_v30_tasks(root: Path) -> dict[int, str]:
    """Read the exact v3 task table into a contiguous index-to-text mapping."""
    dataset_root = _checked_root(root)
    path = dataset_root / "meta" / "tasks.parquet"
    _require_regular_file(dataset_root, path, "tasks metadata")
    table = _read_parquet(path, "tasks metadata")
    expected = {"task_index", "task"}
    actual = set(table.column_names)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Malformed tasks metadata columns in {path}: missing={missing}, extra={extra}"
        )

    tasks: dict[int, str] = {}
    seen_texts: set[str] = set()
    for row_number, row in enumerate(table.select(["task_index", "task"]).to_pylist()):
        context = f"tasks.parquet row {row_number}"
        task_index = _strict_nonnegative_int(row["task_index"], "task_index", context)
        task = _strict_nonempty_string(row["task"], "task", context)
        if task_index in tasks:
            raise ValueError(f"Malformed {context}: duplicate task_index {task_index}")
        if task in seen_texts:
            raise ValueError(f"Malformed {context}: duplicate task text {task!r}")
        tasks[task_index] = task
        seen_texts.add(task)
    if set(tasks) != set(range(len(tasks))):
        raise ValueError("tasks.parquet task_index values must be contiguous from 0 through N-1")
    return tasks


def read_v30_episode_table(root: Path) -> pa.Table:
    """Read only exact v3 episode-metadata shard names in chunk/file order."""
    shards = _read_v30_episode_shards(_checked_root(root))
    return pa.concat_tables([table for _, _, table in shards])


def inspect_v30_dataset(
    config: AnnotationConfig,
    info: dict[str, Any],
    probe: Callable[[Path], VideoProbe],
) -> DatasetIndex:
    """Validate and index a local LeRobot v3.0 dataset without writing to it."""
    root = _checked_root(config.source)
    _require_regular_file(root, root / "meta" / "info.json", "info metadata")
    version = _required_string(info, "codebase_version", "info.json")
    if version != "v3.0":
        raise ValueError(f"LeRobot codebase_version must be v3.0, got {version!r}")

    fps = _positive_number(info, "fps", "info.json")
    chunks_size = info.get("chunks_size", 1000)
    if type(chunks_size) is not int or chunks_size <= 0:
        raise ValueError("Malformed info.json: chunks_size must be a positive integer")
    total_episodes = _nonnegative_int(info, "total_episodes", "info.json")
    expected_total_frames = _nonnegative_int(info, "total_frames", "info.json")
    expected_total_tasks = _nonnegative_int(info, "total_tasks", "info.json")
    data_template = info.get(
        "data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    video_template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    if not isinstance(data_template, str) or not data_template:
        raise ValueError("Malformed info.json: data_path must be a nonempty string")
    if not isinstance(video_template, str) or not video_template:
        raise ValueError("Malformed info.json: video_path must be a nonempty string")
    _validate_template(data_template, "data_path", DATA_FIELDS)
    _validate_template(video_template, "video_path", VIDEO_FIELDS)

    camera_keys = _camera_keys(info)
    _require_configured_cameras(config, camera_keys)
    video_shapes = _validate_video_features(info, camera_keys, fps)
    tasks = read_v30_tasks(root)
    if len(tasks) != expected_total_tasks:
        raise ValueError(
            f"info.total_tasks is {expected_total_tasks}, but tasks.parquet has {len(tasks)} rows"
        )
    task_indices = {task: index for index, task in tasks.items()}

    shards = _read_v30_episode_shards(root)
    table = pa.concat_tables([shard for _, _, shard in shards])
    if table.num_rows != total_episodes:
        raise ValueError(
            f"info.total_episodes is {total_episodes}, but episode metadata has {table.num_rows} rows"
        )
    required_columns = REQUIRED_EPISODE_COLUMNS | {
        f"videos/{camera}/{part}" for camera in camera_keys for part in _VIDEO_COLUMN_PARTS
    }
    missing = sorted(required_columns - set(table.column_names))
    if missing:
        raise ValueError(f"Episode metadata is missing required column(s): {missing}")
    _reject_unknown_episode_columns(table.column_names, required_columns, info)

    episode_rows: list[tuple[int, int, dict[str, Any]]] = []
    for chunk_index, file_index, shard in shards:
        episode_rows.extend((chunk_index, file_index, row) for row in shard.to_pylist())

    data_tables: dict[Path, pa.Table] = {}
    video_probes: dict[Path, VideoProbe] = {}
    data_paths: set[Path] = set()
    video_paths: set[Path] = set()
    video_slices: dict[tuple[str, Path], list[tuple[float, float, int]]] = defaultdict(list)
    episodes: list[EpisodeInfo] = []
    total_frames = 0

    for expected_index, (metadata_chunk, metadata_file, row) in enumerate(episode_rows):
        context = f"episode {expected_index}"
        episode_index = _strict_nonnegative_int(row.get("episode_index"), "episode_index", context)
        if episode_index != expected_index:
            raise ValueError("Episode indices must be contiguous from 0 through N-1")
        _validate_metadata_location(row, metadata_chunk, metadata_file, episode_index)
        length = _strict_positive_int(row.get("length"), "length", context)
        task, task_index = _validate_episode_task(
            row.get("tasks"), task_indices, episode_index
        )

        dataset_from = _strict_nonnegative_int(
            row.get("dataset_from_index"), "dataset_from_index", context
        )
        dataset_to = _strict_positive_int(
            row.get("dataset_to_index"), "dataset_to_index", context
        )
        if dataset_to - dataset_from != length:
            raise ValueError(
                f"Malformed episode {episode_index}: dataset range "
                f"[{dataset_from}, {dataset_to}) does not match length {length}"
            )
        if dataset_from != total_frames:
            raise ValueError(
                f"Malformed episode {episode_index}: dataset range must start at contiguous "
                f"global index {total_frames}, got {dataset_from}"
            )

        data_chunk = _strict_nonnegative_int(
            row.get("data/chunk_index"), "data/chunk_index", context
        )
        data_file = _strict_nonnegative_int(
            row.get("data/file_index"), "data/file_index", context
        )
        data_path = _resolve_payload(
            root,
            data_template,
            {"chunk_index": data_chunk, "file_index": data_file},
            "data_path",
            root / "data",
            f"episode {episode_index} data shard",
        )
        data_paths.add(data_path)
        if data_path not in data_tables:
            data_tables[data_path] = _read_data_table(data_path)
        _validate_data_slice(
            data_tables[data_path],
            data_path,
            episode_index,
            length,
            dataset_from,
            dataset_to,
            task_index,
            fps,
        )

        videos: dict[str, EpisodeVideoRef] = {}
        for camera in camera_keys:
            camera_context = f"episode {episode_index}, camera {camera!r}"
            video_chunk = _strict_nonnegative_int(
                row.get(f"videos/{camera}/chunk_index"), "chunk_index", camera_context
            )
            video_file = _strict_nonnegative_int(
                row.get(f"videos/{camera}/file_index"), "file_index", camera_context
            )
            from_timestamp = _strict_nonnegative_number(
                row.get(f"videos/{camera}/from_timestamp"), "from_timestamp", camera_context
            )
            to_timestamp = _strict_positive_number(
                row.get(f"videos/{camera}/to_timestamp"), "to_timestamp", camera_context
            )
            if to_timestamp <= from_timestamp:
                raise ValueError(
                    f"Malformed {camera_context}: to_timestamp must exceed from_timestamp"
                )
            duration_frames = (to_timestamp - from_timestamp) * fps
            if round(duration_frames) != length or abs(duration_frames - length) > 1.0:
                raise ValueError(
                    f"Malformed {camera_context}: timestamp range represents "
                    f"{duration_frames} frames, expected {length}"
                )

            video_path = _resolve_payload(
                root,
                video_template,
                {
                    "video_key": camera,
                    "chunk_index": video_chunk,
                    "file_index": video_file,
                },
                "video_path",
                root / "videos",
                f"video shard for episode {episode_index}, camera {camera!r}",
            )
            video_paths.add(video_path)
            if video_path not in video_probes:
                video_probes[video_path] = probe(video_path)
            video_probe = video_probes[video_path]
            _validate_video_probe(
                video_probe,
                video_shapes[camera],
                fps,
                video_path,
                episode_index,
                camera,
                to_timestamp,
            )
            video_slices[(camera, video_path)].append(
                (from_timestamp, to_timestamp, episode_index)
            )
            videos[camera] = EpisodeVideoRef(
                path=video_path,
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
                fps=fps,
            )

        episodes.append(
            EpisodeInfo(
                episode_index=episode_index,
                length=length,
                task=task,
                data=EpisodeDataRef(
                    path=data_path,
                    dataset_from_index=dataset_from,
                    dataset_to_index=dataset_to,
                ),
                videos=videos,
            )
        )
        total_frames += length

    _validate_nonoverlapping_video_slices(video_slices)
    if total_frames != expected_total_frames:
        raise ValueError(
            f"info.total_frames is {expected_total_frames}, but episode lengths total {total_frames}"
        )
    return DatasetIndex(
        root=root,
        version="v3.0",
        fps=fps,
        camera_keys=camera_keys,
        episodes=episodes,
    )


def _checked_root(root: Path) -> Path:
    source = Path(os.path.abspath(root))
    if source.is_symlink():
        raise ValueError(f"Malformed dataset root {source}: symbolic links are not allowed")
    try:
        mode = source.stat(follow_symlinks=False).st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing dataset root: {source}") from exc
    if not stat.S_ISDIR(mode):
        raise ValueError(f"Malformed dataset root {source}: expected a directory")
    return source.resolve()


def _read_v30_episode_shards(root: Path) -> list[tuple[int, int, pa.Table]]:
    episode_root = root / "meta" / "episodes"
    _require_directory(root, episode_root, "episode metadata directory")
    paths: list[tuple[int, int, Path]] = []
    for chunk_path in episode_root.iterdir():
        chunk_match = _CHUNK_PATTERN.fullmatch(chunk_path.name)
        if chunk_match is None:
            continue
        _require_directory(root, chunk_path, "episode metadata chunk")
        chunk_index = int(chunk_match.group(1))
        for file_path in chunk_path.iterdir():
            file_match = _FILE_PATTERN.fullmatch(file_path.name)
            if file_match is None:
                continue
            _require_regular_file(root, file_path, "episode metadata shard")
            paths.append((chunk_index, int(file_match.group(1)), file_path))
    paths.sort(key=lambda item: (item[0], item[1]))
    if not paths:
        raise FileNotFoundError(
            "Missing v3 episode metadata shards matching "
            "meta/episodes/chunk-NNN/file-NNN.parquet"
        )

    shards: list[tuple[int, int, pa.Table]] = []
    reference_schema: pa.Schema | None = None
    for chunk_index, file_index, path in paths:
        table = _read_parquet(path, "episode metadata shard")
        missing = sorted(REQUIRED_EPISODE_COLUMNS - set(table.column_names))
        if missing:
            raise ValueError(
                f"Episode metadata shard {path} is missing required column(s): {missing}"
            )
        if reference_schema is None:
            reference_schema = table.schema
        elif table.schema != reference_schema:
            raise ValueError(
                f"Episode metadata shard {path} has extra, missing, reordered, or mismatched columns"
            )
        shards.append((chunk_index, file_index, table))
    return shards


def _validate_template(template: str, field: str, expected_fields: set[str]) -> None:
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"Malformed {field} format in info.json: {exc}") from exc
    fields: list[str] = []
    for _, name, format_spec, _ in parsed:
        if name is None:
            continue
        if name not in expected_fields:
            raise ValueError(
                f"Malformed {field} format in info.json: unsupported field {name!r}"
            )
        if "{" in format_spec or "}" in format_spec:
            raise ValueError(
                f"Malformed {field} format in info.json: nested format fields are not allowed"
            )
        fields.append(name)
    if set(fields) != expected_fields or len(fields) != len(expected_fields):
        raise ValueError(
            f"Malformed {field} format in info.json: fields must be exactly "
            f"{sorted(expected_fields)}"
        )


def _resolve_payload(
    root: Path,
    template: str,
    values: dict[str, Any],
    field: str,
    expected_root: Path,
    context: str,
) -> Path:
    try:
        formatted = Path(template.format(**values))
    except Exception as exc:
        raise ValueError(f"Malformed {field} format in info.json: {exc}") from exc
    if formatted.is_absolute():
        raise ValueError(f"Malformed {field} format in info.json: absolute paths are not allowed")
    lexical = root / formatted
    _reject_symlink_components(root, lexical, context)
    target = lexical.resolve()
    try:
        target.relative_to(expected_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Malformed {field} format in info.json: path must stay within "
            f"{expected_root.relative_to(root)}"
        ) from exc
    _require_regular_file(root, lexical, context)
    return target


def _require_directory(root: Path, path: Path, context: str) -> None:
    _reject_symlink_components(root, path, context)
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing {context}: {path}") from exc
    if not stat.S_ISDIR(mode):
        raise ValueError(f"Malformed {context}: expected directory at {path}")


def _require_regular_file(root: Path, path: Path, context: str) -> None:
    _reject_symlink_components(root, path, context)
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing {context}: {path}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"Malformed {context}: expected a regular file at {path}")


def _reject_symlink_components(root: Path, path: Path, context: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Malformed {context}: path escapes dataset root: {path}") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Malformed {context}: symbolic links are not allowed: {current}")


def _read_parquet(path: Path, context: str, columns: list[str] | None = None) -> pa.Table:
    try:
        return pq.read_table(path, columns=columns)
    except Exception as exc:
        raise ValueError(f"Unable to read {context} {path}: {exc}") from exc


def _read_data_table(path: Path) -> pa.Table:
    table = _read_parquet(path, "data shard", list(_DATA_ROW_COLUMNS))
    missing = sorted(set(_DATA_ROW_COLUMNS) - set(table.column_names))
    if missing:
        raise ValueError(f"Data shard {path} is missing required column(s): {missing}")
    return table


def _validate_data_slice(
    table: pa.Table,
    path: Path,
    episode_index: int,
    length: int,
    dataset_from: int,
    dataset_to: int,
    task_index: int,
    fps: float,
) -> None:
    indices = table.column("index").to_pylist()
    for row_number, value in enumerate(indices):
        if type(value) is not int:
            raise ValueError(
                f"Malformed data row for episode {episode_index}: index at shard row "
                f"{row_number} must be an integer, got {value!r}"
            )
    expected_indices = list(range(dataset_from, dataset_to))
    starts = [position for position, value in enumerate(indices) if value == dataset_from]
    matches = [
        position
        for position in starts
        if indices[position : position + length] == expected_indices
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Malformed data shard for episode {episode_index}: {path} does not contain exactly "
            f"one contiguous global index sequence [{dataset_from}, {dataset_to})"
        )
    rows = table.slice(matches[0], length).to_pylist()
    for local_index, row in enumerate(rows):
        if type(row["episode_index"]) is not int or row["episode_index"] != episode_index:
            raise ValueError(
                f"Malformed data row for episode {episode_index}: episode_index at local row "
                f"{local_index} is {row['episode_index']!r}"
            )
        if type(row["frame_index"]) is not int or row["frame_index"] != local_index:
            raise ValueError(
                f"Malformed data row for episode {episode_index}: frame_index at local row "
                f"{local_index} is {row['frame_index']!r}"
            )
        if type(row["task_index"]) is not int or row["task_index"] != task_index:
            raise ValueError(
                f"Malformed data row for episode {episode_index}: task_index at local row "
                f"{local_index} is {row['task_index']!r}"
            )
        timestamp = row["timestamp"]
        expected_timestamp = local_index / fps
        if not data_timestamp_matches(timestamp, local_index, fps):
            raise ValueError(
                f"Malformed data row for episode {episode_index}: timestamp at local row "
                f"{local_index} is {timestamp!r}, expected {expected_timestamp}"
            )


def _validate_video_features(
    info: dict[str, Any], camera_keys: list[str], fps: float
) -> dict[str, tuple[int, int]]:
    features = info.get("features")
    assert isinstance(features, dict)  # _camera_keys already validated this boundary.
    result: dict[str, tuple[int, int]] = {}
    for camera in camera_keys:
        feature = features[camera]
        if not isinstance(feature, dict):
            raise ValueError(f"Malformed video feature {camera!r}: expected an object")
        shape = feature.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(item) is not int or item <= 0 for item in shape)
        ):
            raise ValueError(
                f"Malformed video feature {camera!r}: shape must contain three positive integers"
            )
        video_info = feature.get("info")
        if not isinstance(video_info, dict):
            raise ValueError(f"Malformed video feature {camera!r}: info must be an object")
        video_fps = _strict_positive_number(video_info.get("video.fps"), "video.fps", camera)
        width = _strict_positive_int(video_info.get("video.width"), "video.width", camera)
        height = _strict_positive_int(video_info.get("video.height"), "video.height", camera)
        if not video_fps_matches(video_fps, fps):
            raise ValueError(
                f"Malformed video feature {camera!r}: video.fps is {video_fps}, expected {fps}"
            )
        depth = is_depth_feature(feature)
        channels = 1 if depth else 3
        if depth:
            depth_metadata(feature, camera)
        if shape not in (
            [channels, height, width],
            [height, width, channels],
        ):
            kind = "depth" if depth else "RGB"
            raise ValueError(
                f"Malformed video feature {camera!r}: shape {shape} disagrees with "
                f"official CHW/HWC shape for {height}x{width} {kind} video"
            )
        declared_channels = video_info.get("video.channels")
        if declared_channels is not None and declared_channels != channels:
            raise ValueError(
                f"Malformed video feature {camera!r}: video.channels is "
                f"{declared_channels!r}, expected {channels}"
            )
        result[camera] = (width, height)
    return result


def _validate_video_probe(
    video_probe: VideoProbe,
    shape: tuple[int, int],
    fps: float,
    path: Path,
    episode_index: int,
    camera: str,
    to_timestamp: float,
) -> None:
    if not video_fps_matches(video_probe.fps, fps):
        raise ValueError(
            f"Video fps for episode {episode_index}, camera {camera!r} in {path} "
            f"is {video_probe.fps}, expected {fps}"
        )
    if (video_probe.width, video_probe.height) != shape:
        raise ValueError(
            f"Video shape for episode {episode_index}, camera {camera!r} in {path} is "
            f"{video_probe.width}x{video_probe.height}, expected {shape[0]}x{shape[1]}"
        )
    duration = video_probe.frames / video_probe.fps
    tolerance = 1.0 / fps
    if to_timestamp > duration + tolerance + 1e-9:
        raise ValueError(
            f"Video range for episode {episode_index}, camera {camera!r} exceeds shard "
            f"duration {duration} in {path}"
        )


def _validate_nonoverlapping_video_slices(
    slices: dict[tuple[str, Path], list[tuple[float, float, int]]]
) -> None:
    for (camera, path), ranges in slices.items():
        ordered = sorted(ranges)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current[0] < previous[1] - 1e-9:
                raise ValueError(
                    f"Video ranges for camera {camera!r} overlap in shard {path}: "
                    f"episodes {previous[2]} and {current[2]}"
                )


def _validate_metadata_location(
    row: dict[str, Any], chunk_index: int, file_index: int, episode_index: int
) -> None:
    context = f"episode {episode_index}"
    declared_chunk = _strict_nonnegative_int(
        row.get("meta/episodes/chunk_index"), "meta/episodes/chunk_index", context
    )
    declared_file = _strict_nonnegative_int(
        row.get("meta/episodes/file_index"), "meta/episodes/file_index", context
    )
    if (declared_chunk, declared_file) != (chunk_index, file_index):
        raise ValueError(
            f"Malformed episode {episode_index}: metadata location declares "
            f"chunk/file {declared_chunk}/{declared_file}, loaded from {chunk_index}/{file_index}"
        )


def _validate_episode_task(
    value: Any,
    task_indices: dict[str, int],
    episode_index: int,
) -> tuple[str, int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Malformed episode {episode_index}: tasks must be a nonempty list")
    if any(not isinstance(task, str) or not task for task in value):
        raise ValueError(
            f"Malformed episode {episode_index}: tasks must contain nonempty strings"
        )
    for task in value:
        if task not in task_indices:
            raise ValueError(
                f"Episode {episode_index} references task {task!r}, absent from meta/tasks.parquet"
            )
    if len(value) != 1:
        raise ValueError(
            "First release supports exactly one task per episode; "
            f"episode {episode_index} has {len(value)}"
        )
    task = value[0]
    return task, task_indices[task]


def _reject_unknown_episode_columns(
    columns: list[str], required: set[str], info: dict[str, Any]
) -> None:
    features = info.get("features")
    assert isinstance(features, dict)
    for column in columns:
        if column in required:
            continue
        if column.startswith("stats/"):
            remainder = column[len("stats/") :]
            feature, separator, metric = remainder.rpartition("/")
            if separator and feature in features and (
                metric in _STAT_METRICS
                or (len(metric) == 3 and metric.startswith("q") and metric[1:].isdigit())
            ):
                continue
        raise ValueError(f"Unexpected episode metadata column {column!r}")


def _strict_nonempty_string(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Malformed {context}: {field} must be a nonempty string")
    return value


def _strict_nonnegative_int(value: Any, field: str, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Malformed {context}: {field} must be a nonnegative integer")
    return value


def _strict_positive_int(value: Any, field: str, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"Malformed {context}: {field} must be a positive integer")
    return value


def _strict_nonnegative_number(value: Any, field: str, context: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Malformed {context}: {field} must be a nonnegative finite number")
    return float(value)


def _strict_positive_number(value: Any, field: str, context: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Malformed {context}: {field} must be a positive finite number")
    return float(value)

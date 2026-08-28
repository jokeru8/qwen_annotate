"""Independent validation of publishable LeRobot v3.0 shared-shard releases.

This module intentionally does not import the v3 dataset inspection adapter.  A
publisher therefore cannot prove its own output through the same parser used at
ingest time.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from string import Formatter
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .constraints import validate_annotation
from .lerobot import EpisodeVideoRef, video_fps_matches
from .models import FinalAnnotation
from .release_validator import (
    BoundaryPreview,
    ReleaseReport,
    ReleaseServices,
    _aware_utc,
    _integer,
    _payload_files,
    _read_object,
    _regular_under,
    _reject_forbidden,
    _safe_root,
    _sha256,
    _string,
    _subtasks,
    _validate_splits,
    _validate_task_info,
    _walk_regular,
)


_TASK_COLUMNS = {"task_index", "task"}
_DATA_COLUMNS = {"index", "episode_index", "frame_index", "timestamp", "task_index"}
_EPISODE_COLUMNS = {
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
_VIDEO_PARTS = ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
_STAT_METRICS = ("min", "max", "mean", "std", "count")
_CHUNK = re.compile(r"chunk-([0-9]{3})\Z")
_FILE = re.compile(r"file-([0-9]{3})\.parquet\Z")


@dataclass(frozen=True)
class _VideoSlice:
    path: Path
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class _Episode:
    index: int
    length: int
    task: str
    data_path: Path
    data_offset: int
    videos: dict[str, _VideoSlice]
    metadata: dict[str, Any]


def validate_v30_release(
    root: Path,
    *,
    source_root: Path | None,
    services: ReleaseServices,
    expected_output_root: Path | None,
    deep_video_stats: bool,
) -> ReleaseReport:
    """Validate v3 directly while retaining the same safe public boundary."""
    release_root = _safe_root(root, "release")
    checked_source = (
        _safe_root(source_root, "source") if source_root is not None else None
    )
    _walk_regular(release_root)
    info = _read_object(release_root / "meta/info.json")
    return _validate_v30_release_with_info(
        release_root,
        source_root=checked_source,
        services=services,
        expected_output_root=expected_output_root,
        deep_video_stats=deep_video_stats,
        info=info,
    )


def _validate_v30_release_with_info(
    root: Path,
    *,
    source_root: Path | None,
    services: ReleaseServices,
    expected_output_root: Path | None,
    deep_video_stats: bool,
    info: dict[str, Any],
) -> ReleaseReport:
    """Validate v3 using the facade's single bounded info.json read."""
    if _string(info, "codebase_version") != "v3.0":
        raise ValueError("v3 release validator requires codebase_version 'v3.0'")
    if type(deep_video_stats) is not bool:
        raise TypeError("deep_video_stats must be a bool")

    total_episodes = _strict_int(info.get("total_episodes"), "total_episodes", 0)
    total_frames = _strict_int(info.get("total_frames"), "total_frames", 0)
    total_tasks = _strict_int(info.get("total_tasks"), "total_tasks", 0)
    total_videos = _strict_int(info.get("total_videos"), "total_videos", 0)
    total_data_files = _strict_int(
        info.get("total_data_files"), "total_data_files", 0
    )
    total_video_files = _strict_int(
        info.get("total_video_files"), "total_video_files", 0
    )
    chunks_size = _strict_int(info.get("chunks_size"), "chunks_size", 1)
    fps = _positive_number(info.get("fps"), "fps")
    _positive_number(info.get("data_files_size_in_mb"), "data_files_size_in_mb")
    _positive_number(info.get("video_files_size_in_mb"), "video_files_size_in_mb")
    data_template = _template(
        info.get("data_path"), "data_path", {"chunk_index", "file_index"}
    )
    video_template = _template(
        info.get("video_path"),
        "video_path",
        {"video_key", "chunk_index", "file_index"},
    )
    episodes_template = None
    if "episodes_path" in info:
        episodes_template = _template(
            info.get("episodes_path"),
            "episodes_path",
            {"chunk_index", "file_index"},
        )
    _validate_splits(info.get("splits"), total_episodes)

    features = info.get("features")
    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a nonempty object")
    numeric_shapes, camera_shapes = _feature_shapes(features, fps)
    cameras = list(camera_shapes)
    if not cameras:
        raise ValueError("v3 release must declare at least one video camera")
    if total_videos != total_episodes * len(cameras):
        raise ValueError("total_videos is inconsistent with episodes and cameras")

    tasks = _read_tasks(root / "meta/tasks.parquet")
    if len(tasks) != total_tasks:
        raise ValueError("total_tasks differs from tasks.parquet")
    task_by_text = {text: index for index, text in tasks.items()}

    episode_shards = _read_episode_shards(root)
    if episodes_template is not None:
        for chunk_index, file_index, _, path in episode_shards:
            relative = _render(
                episodes_template,
                {"chunk_index": chunk_index, "file_index": file_index},
                "episodes_path",
                "meta",
            )
            if relative.parts[:2] != ("meta", "episodes") or root / relative != path:
                raise ValueError(
                    "episodes_path must resolve each metadata shard under meta/episodes"
                )
    rows: list[tuple[int, int, dict[str, Any]]] = []
    for chunk_index, file_index, table, _ in episode_shards:
        rows.extend((chunk_index, file_index, row) for row in table.to_pylist())
    if len(rows) != total_episodes:
        raise ValueError("episode metadata row count differs from total_episodes")

    required_episode_columns = _EPISODE_COLUMNS | {
        f"videos/{camera}/{part}" for camera in cameras for part in _VIDEO_PARTS
    }
    expected_stat_columns = {
        f"stats/{feature}/{metric}"
        for feature in [*numeric_shapes, *cameras]
        for metric in _STAT_METRICS
    }
    for _, _, table, _ in episode_shards:
        actual = set(table.column_names)
        if actual != required_episode_columns | expected_stat_columns:
            raise ValueError(
                "episode metadata schema mismatch: "
                f"missing={sorted((required_episode_columns | expected_stat_columns) - actual)}, "
                f"extra={sorted(actual - required_episode_columns - expected_stat_columns)}"
            )

    data_tables: dict[Path, pa.Table] = {}
    data_pairs: set[tuple[int, int]] = set()
    data_pair_paths: dict[tuple[int, int], Path] = {}
    data_coverage: dict[Path, set[int]] = defaultdict(set)
    video_probes: dict[Path, Any] = {}
    video_ranges: dict[tuple[str, Path], list[tuple[float, float, int]]] = (
        defaultdict(list)
    )
    video_pairs: dict[str, set[tuple[int, int]]] = defaultdict(set)
    episodes: list[_Episode] = []
    expected_global = 0
    expected_payload: set[str] = set()
    expected_episode_shards = {
        path.relative_to(root).as_posix() for _, _, _, path in episode_shards
    }

    for expected_index, (meta_chunk, meta_file, row) in enumerate(rows):
        context = f"episode {expected_index}"
        episode_index = _strict_int(row.get("episode_index"), "episode_index", 0)
        if episode_index != expected_index:
            raise ValueError("episode indices must be contiguous from 0 through N-1")
        declared_meta_chunk = _strict_int(
            row.get("meta/episodes/chunk_index"),
            "meta/episodes/chunk_index",
            0,
        )
        declared_meta_file = _strict_int(
            row.get("meta/episodes/file_index"), "meta/episodes/file_index", 0
        )
        if (declared_meta_chunk, declared_meta_file) != (meta_chunk, meta_file):
            raise ValueError(f"{context} metadata shard location is inconsistent")
        if declared_meta_file >= chunks_size:
            raise ValueError(f"{context} metadata file_index exceeds chunks_size")
        length = _strict_int(row.get("length"), "length", 1)
        episode_task = row.get("tasks")
        if (
            not isinstance(episode_task, list)
            or len(episode_task) != 1
            or not isinstance(episode_task[0], str)
            or not episode_task[0]
            or episode_task[0] not in task_by_text
        ):
            raise ValueError(f"{context} must reference exactly one known task")
        task_index = task_by_text[episode_task[0]]

        dataset_from = _strict_int(
            row.get("dataset_from_index"), "dataset_from_index", 0
        )
        dataset_to = _strict_int(row.get("dataset_to_index"), "dataset_to_index", 1)
        if dataset_to - dataset_from != length:
            raise ValueError(
                f"episode {episode_index} dataset_to_index does not match length"
            )
        if dataset_from != expected_global:
            raise ValueError(
                f"episode {episode_index} dataset_from_index is not globally contiguous"
            )
        data_chunk = _strict_int(row.get("data/chunk_index"), "data/chunk_index", 0)
        data_file = _strict_int(row.get("data/file_index"), "data/file_index", 0)
        if data_file >= chunks_size:
            raise ValueError(f"{context} data file_index exceeds chunks_size")
        data_relative = _render(
            data_template,
            {"chunk_index": data_chunk, "file_index": data_file},
            "data_path",
            "data",
        )
        data_path = _regular_under(root, data_relative)
        data_pairs.add((data_chunk, data_file))
        prior_data_path = data_pair_paths.setdefault(
            (data_chunk, data_file), data_path
        )
        if prior_data_path != data_path:
            raise ValueError("one data shard number resolves to multiple paths")
        expected_payload.add(data_relative.as_posix())
        if data_path not in data_tables:
            data_tables[data_path] = _read_data(data_path, features, numeric_shapes)
        data_offset = _validate_data_slice(
            data_tables[data_path],
            episode_index,
            length,
            dataset_from,
            dataset_to,
            task_index,
            fps,
        )
        occupied = set(range(data_offset, data_offset + length))
        if data_coverage[data_path] & occupied:
            raise ValueError(f"{context} data slice overlaps another episode")
        data_coverage[data_path] |= occupied

        episode_videos: dict[str, _VideoSlice] = {}
        for camera in cameras:
            camera_context = f"episode {episode_index}, camera {camera}"
            video_chunk = _strict_int(
                row.get(f"videos/{camera}/chunk_index"), "chunk_index", 0
            )
            video_file = _strict_int(
                row.get(f"videos/{camera}/file_index"), "file_index", 0
            )
            if video_file >= chunks_size:
                raise ValueError(f"{camera_context} file_index exceeds chunks_size")
            from_timestamp = _nonnegative_number(
                row.get(f"videos/{camera}/from_timestamp"), "from_timestamp"
            )
            to_timestamp = _positive_number(
                row.get(f"videos/{camera}/to_timestamp"), "to_timestamp"
            )
            if to_timestamp <= from_timestamp:
                raise ValueError(f"{camera_context} timestamp range is reversed")
            duration_frames = (to_timestamp - from_timestamp) * fps
            if round(duration_frames) != length or abs(duration_frames - length) > 1.0:
                raise ValueError(
                    f"{camera_context} timestamp range does not match episode length"
                )
            video_relative = _render(
                video_template,
                {
                    "video_key": camera,
                    "chunk_index": video_chunk,
                    "file_index": video_file,
                },
                "video_path",
                "videos",
            )
            video_path = _regular_under(root, video_relative)
            video_pairs[camera].add((video_chunk, video_file))
            expected_payload.add(video_relative.as_posix())
            if video_path not in video_probes:
                video_probes[video_path] = services.probe_video(video_path)
            probe = video_probes[video_path]
            width, height = camera_shapes[camera]
            if (
                not video_fps_matches(probe.fps, fps)
                or probe.width != width
                or probe.height != height
            ):
                raise ValueError(f"{camera_context} video fps or shape mismatch")
            if to_timestamp > probe.frames / probe.fps + 1.0 / fps + 1e-9:
                raise ValueError(f"{camera_context} timestamp range exceeds video shard")
            video_ranges[(camera, video_path)].append(
                (from_timestamp, to_timestamp, episode_index)
            )
            episode_videos[camera] = _VideoSlice(
                video_path, from_timestamp, to_timestamp
            )

        episodes.append(
            _Episode(
                episode_index,
                length,
                episode_task[0],
                data_path,
                data_offset,
                episode_videos,
                row,
            )
        )
        expected_global = dataset_to

    if expected_global != total_frames:
        raise ValueError("episode lengths differ from total_frames")
    if len(data_tables) != total_data_files:
        raise ValueError("total_data_files differs from referenced data shards")
    if len(video_probes) != total_video_files:
        raise ValueError("total_video_files differs from referenced video shards")
    _require_canonical_file_pairs(data_pairs, chunks_size, "data shard numbering")
    physical_indices = [
        value
        for pair in sorted(data_pairs)
        for value in data_tables[data_pair_paths[pair]]["index"].to_pylist()
    ]
    if physical_indices != list(range(total_frames)):
        raise ValueError(
            "global index values must be contiguous in physical shard order"
        )
    for camera in cameras:
        _require_canonical_file_pairs(
            video_pairs[camera], chunks_size, f"video shard numbering for {camera}"
        )
    _require_canonical_file_pairs(
        {(chunk, file) for chunk, file, _, _ in episode_shards},
        chunks_size,
        "episode metadata shard numbering",
    )
    for path, table in data_tables.items():
        if data_coverage[path] != set(range(table.num_rows)):
            raise ValueError(f"data shard has unreferenced or multiply referenced rows: {path}")
    _validate_video_coverage(video_ranges, video_probes, fps)

    actual_payload = _payload_files(root)
    if actual_payload != expected_payload:
        raise ValueError("missing or extra v3 payload files")
    actual_episode_shards = {
        path.relative_to(root).as_posix()
        for path in (root / "meta/episodes").rglob("*.parquet")
        if path.is_file()
    }
    if actual_episode_shards != expected_episode_shards:
        raise ValueError("missing or extra episode metadata shards")

    published_stats = _read_object(root / "meta/stats.json")
    expected_stats_features = set(numeric_shapes) | set(cameras)
    if set(published_stats) != expected_stats_features:
        raise ValueError("aggregate stats feature coverage mismatch")
    aggregate_numeric, episode_numeric = _numeric_stats(
        data_tables, episodes, numeric_shapes
    )
    for feature, actual in aggregate_numeric.items():
        _compare_stats(
            published_stats[feature], actual, f"stats {feature}", tolerance=1e-6
        )
    for episode in episodes:
        for feature, actual in episode_numeric[episode.index].items():
            _compare_stats(
                _row_stats(episode.metadata, feature),
                actual,
                f"episode {episode.index} stats {feature}",
                tolerance=1e-6,
            )

    if deep_video_stats:
        aggregate_video, episode_video = _video_stats(
            episodes,
            cameras,
            camera_shapes,
            video_ranges,
            services,
            fps,
        )
        # Published v3 image statistics originate before lossy H.264 encoding.
        # A four-code-value envelope covers codec drift while still rejecting
        # semantic/statistical corruption.
        codec_tolerance = 4.0 / 255.0 + 1e-6
        for camera, actual in aggregate_video.items():
            _compare_stats(
                published_stats[camera],
                actual,
                f"stats {camera}",
                tolerance=codec_tolerance,
            )
        for episode in episodes:
            for camera, actual in episode_video[episode.index].items():
                _compare_stats(
                    _row_stats(episode.metadata, camera),
                    actual,
                    f"episode {episode.index} stats {camera}",
                    tolerance=codec_tolerance,
                )
    else:
        for camera in cameras:
            _validate_stats_shape(
                published_stats[camera], 3, total_frames, f"stats {camera}"
            )
        for episode in episodes:
            for camera in cameras:
                _validate_stats_shape(
                    _row_stats(episode.metadata, camera),
                    3,
                    episode.length,
                    f"episode {episode.index} stats {camera}",
                )

    annotations = _read_object(root / "meta/lerobot_annotations.json")
    template, mode, preview = _validate_annotations(
        root,
        source_root,
        expected_output_root,
        annotations,
        episodes,
        cameras,
        fps,
        services,
    )

    digests = {relative: _sha256(root / relative) for relative in sorted(expected_payload)}
    if source_root is not None:
        _walk_regular(source_root)
        release_core = _official_core_files(root)
        source_core = _official_core_files(source_root)
        if release_core != source_core:
            raise ValueError("official v3 source/release file inventory differs")
        for relative in sorted(release_core):
            if _sha256(root / relative) != _sha256(source_root / relative):
                raise ValueError(f"official v3 source byte mismatch: {relative}")
    aggregate_digest = hashlib.sha256(
        "".join(
            f"{name}\0{digests[name]}\n" for name in sorted(digests)
        ).encode()
    ).hexdigest()
    return ReleaseReport(
        path=root,
        dataset_version="v3.0",
        episode_count=total_episodes,
        frame_count=total_frames,
        mode=mode,
        subtask_template=template,
        payload_files=sorted(expected_payload),
        payload_digests=digests,
        payload_checksum=aggregate_digest,
        preview=preview,
        validation_level="strict_deep" if deep_video_stats else "strict_structural",
        skipped_checks=() if deep_video_stats else ("video_payload_stat_equality",),
        validated_at=datetime.now(UTC),
    )


def _feature_shapes(
    features: Mapping[str, Any], fps: float
) -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
    numeric: dict[str, int] = {}
    cameras: dict[str, tuple[int, int]] = {}
    for name, value in features.items():
        if not isinstance(name, str) or not name or not isinstance(value, dict):
            raise ValueError("feature definitions must be named objects")
        dtype = value.get("dtype")
        shape = value.get("shape")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not shape or any(
            type(item) is not int or item <= 0 for item in shape
        ):
            raise ValueError(f"feature dtype or shape is invalid: {name}")
        if dtype == "video":
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"video feature shape must be [3,height,width]: {name}")
            video_info = value.get("info")
            if not isinstance(video_info, dict):
                raise ValueError(f"video feature info is invalid: {name}")
            video_fps = _positive_number(video_info.get("video.fps"), "video.fps")
            width = _strict_int(video_info.get("video.width"), "video.width", 1)
            height = _strict_int(video_info.get("video.height"), "video.height", 1)
            if not video_fps_matches(video_fps, fps) or shape != [3, height, width]:
                raise ValueError(f"video feature fps or shape is inconsistent: {name}")
            cameras[name] = (width, height)
        elif dtype.startswith("int") or dtype.startswith("float"):
            numeric[name] = math.prod(shape)
        else:
            raise ValueError(f"unsupported non-video feature dtype: {name}")
    return numeric, cameras


def _read_tasks(path: Path) -> dict[int, str]:
    table = _read_parquet(path, "tasks.parquet")
    if set(table.column_names) != _TASK_COLUMNS:
        raise ValueError("tasks.parquet schema must contain exactly task_index and task")
    result: dict[int, str] = {}
    texts: set[str] = set()
    for row_number, row in enumerate(table.to_pylist()):
        index = _strict_int(row.get("task_index"), "task_index", 0)
        task = row.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError(f"tasks.parquet row {row_number} has invalid task text")
        if index in result or task in texts:
            raise ValueError("tasks.parquet contains duplicate task metadata")
        result[index] = task
        texts.add(task)
    if set(result) != set(range(len(result))):
        raise ValueError("tasks.parquet task_index values must be contiguous")
    return result


def _read_episode_shards(
    root: Path,
) -> list[tuple[int, int, pa.Table, Path]]:
    directory = root / "meta/episodes"
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("missing safe meta/episodes directory")
    result: list[tuple[int, int, pa.Table, Path]] = []
    for chunk_path in sorted(directory.iterdir()):
        match = _CHUNK.fullmatch(chunk_path.name)
        if match is None or not chunk_path.is_dir() or chunk_path.is_symlink():
            raise ValueError(f"unexpected episode metadata entry: {chunk_path.name}")
        chunk_index = int(match.group(1))
        for file_path in sorted(chunk_path.iterdir()):
            file_match = _FILE.fullmatch(file_path.name)
            if file_match is None:
                raise ValueError(f"unexpected episode metadata file: {file_path.name}")
            file_index = int(file_match.group(1))
            result.append(
                (
                    chunk_index,
                    file_index,
                    _read_parquet(file_path, "episode metadata"),
                    file_path,
                )
            )
    if not result:
        raise ValueError("episode metadata has no parquet shards")
    return result


def _read_parquet(path: Path, context: str) -> pa.Table:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or missing {context}: {path}")
    try:
        return pq.read_table(path)
    except Exception as exc:
        raise ValueError(f"unable to read {context}: {path}") from exc


def _read_data(
    path: Path,
    features: Mapping[str, Any],
    numeric_shapes: Mapping[str, int],
) -> pa.Table:
    table = _read_parquet(path, "data shard")
    required = _DATA_COLUMNS | set(numeric_shapes)
    actual = set(table.column_names)
    if actual != required:
        raise ValueError(
            "data schema feature inventory mismatch: "
            f"missing={sorted(required - actual)}, extra={sorted(actual - required)}"
        )
    for name in ("index", "episode_index", "frame_index", "task_index"):
        field = table.schema.field(name)
        if not pa.types.is_integer(field.type) or table[name].null_count:
            raise ValueError(f"data schema requires a non-null integer {name} column")
    timestamp_field = table.schema.field("timestamp")
    if not pa.types.is_floating(timestamp_field.type) or table["timestamp"].null_count:
        raise ValueError("data schema requires a non-null floating timestamp column")
    for feature, width in numeric_shapes.items():
        field = table.schema.field(feature)
        declared = features[feature]["dtype"]
        base = field.type.value_type if pa.types.is_fixed_size_list(field.type) else field.type
        try:
            declared_type = pa.type_for_alias(declared)
        except ValueError as exc:
            raise ValueError(f"unsupported declared feature dtype: {feature}") from exc
        compatible = base.equals(declared_type)
        actual_width = field.type.list_size if pa.types.is_fixed_size_list(field.type) else 1
        if not compatible or actual_width != width or table[feature].null_count:
            raise ValueError(f"data schema disagrees with feature declaration: {feature}")
    return table


def _validate_data_slice(
    table: pa.Table,
    episode_index: int,
    length: int,
    dataset_from: int,
    dataset_to: int,
    task_index: int,
    fps: float,
) -> int:
    indices = table["index"].to_pylist()
    expected = list(range(dataset_from, dataset_to))
    matches = [
        offset
        for offset in range(max(0, len(indices) - length + 1))
        if indices[offset : offset + length] == expected
    ]
    if len(matches) != 1:
        raise ValueError(
            f"episode {episode_index} data shard does not contain one exact global index slice"
        )
    offset = matches[0]
    rows = table.slice(offset, length).select(sorted(_DATA_COLUMNS)).to_pylist()
    for local_index, row in enumerate(rows):
        if row["episode_index"] != episode_index:
            raise ValueError(f"episode {episode_index} data episode_index mismatch")
        if row["frame_index"] != local_index:
            raise ValueError(f"episode {episode_index} data frame_index mismatch")
        if row["task_index"] != task_index:
            raise ValueError(f"episode {episode_index} data task_index mismatch")
        timestamp = row["timestamp"]
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
            or not math.isclose(float(timestamp), local_index / fps, abs_tol=1e-6)
        ):
            raise ValueError(f"episode {episode_index} data timestamp mismatch")
    return offset


def _validate_video_coverage(
    ranges: Mapping[tuple[str, Path], list[tuple[float, float, int]]],
    probes: Mapping[Path, Any],
    fps: float,
) -> None:
    tolerance = 1.0 / fps + 1e-9
    for (camera, path), items in ranges.items():
        ordered = sorted(items)
        expected = 0.0
        for start, stop, episode_index in ordered:
            if start < expected - 1e-9:
                raise ValueError(
                    f"video slice overlap for camera {camera}, episode {episode_index}"
                )
            if start > expected + tolerance:
                raise ValueError(
                    f"video coverage gap for camera {camera}, episode {episode_index}"
                )
            expected = stop
        duration = probes[path].frames / probes[path].fps
        if abs(expected - duration) > tolerance:
            raise ValueError(f"video shard coverage differs from decoded duration: {path}")


def _require_canonical_file_pairs(
    pairs: set[tuple[int, int]], chunks_size: int, context: str
) -> None:
    expected = {
        (number // chunks_size, number % chunks_size)
        for number in range(len(pairs))
    }
    if pairs != expected:
        raise ValueError(f"{context} must be contiguous from chunk 0, file 0")


def _numeric_stats(
    tables: Mapping[Path, pa.Table],
    episodes: Sequence[_Episode],
    shapes: Mapping[str, int],
) -> tuple[dict[str, dict[str, list[float]]], dict[int, dict[str, dict[str, list[float]]]]]:
    ordered_tables = [tables[path] for path in sorted(tables)]
    aggregate = {
        feature: _stats(
            np.concatenate(
                [_matrix(table[feature], width) for table in ordered_tables]
            )
        )
        for feature, width in shapes.items()
    }
    by_episode: dict[int, dict[str, dict[str, list[float]]]] = {}
    for episode in episodes:
        table = tables[episode.data_path].slice(episode.data_offset, episode.length)
        by_episode[episode.index] = {
            feature: _stats(_matrix(table[feature], width))
            for feature, width in shapes.items()
        }
    return aggregate, by_episode


def _video_stats(
    episodes: Sequence[_Episode],
    cameras: Sequence[str],
    shapes: Mapping[str, tuple[int, int]],
    ranges: Mapping[tuple[str, Path], list[tuple[float, float, int]]],
    services: ReleaseServices,
    fps: float,
) -> tuple[dict[str, dict[str, list[float]]], dict[int, dict[str, dict[str, list[float]]]]]:
    aggregate_values: dict[str, list[np.ndarray]] = defaultdict(list)
    episode_values: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    ordered_ranges = sorted(
        ranges.items(), key=lambda item: (item[0][0], item[0][1])
    )
    for (camera, path), slices in ordered_ranges:
        width, height = shapes[camera]
        decoded: list[np.ndarray] = []
        iterator = iter(services.iter_video_rgb_frames(path))
        try:
            for frame in iterator:
                array = np.asarray(frame)
                if array.dtype != np.uint8 or array.shape != (height, width, 3):
                    raise ValueError(f"decoded video frame shape or dtype mismatch: {path}")
                decoded.append(array.mean(axis=(0, 1), dtype=np.float64) / 255.0)
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        probe_frames = max(round(stop * fps) for _, stop, _ in slices)
        if len(decoded) != probe_frames:
            raise ValueError(f"decoded video frame count mismatch: {path}")
        values = np.asarray(decoded, dtype=np.float64)
        aggregate_values[camera].append(values)
        for start, stop, episode_index in slices:
            left, right = round(start * fps), round(stop * fps)
            expected = next(item.length for item in episodes if item.index == episode_index)
            if right - left != expected:
                raise ValueError(f"episode {episode_index} decoded video slice length mismatch")
            episode_values[episode_index][camera] = values[left:right]
    aggregate = {
        camera: _stats(np.concatenate(aggregate_values[camera])) for camera in cameras
    }
    by_episode = {
        episode.index: {
            camera: _stats(episode_values[episode.index][camera]) for camera in cameras
        }
        for episode in episodes
    }
    return aggregate, by_episode


def _matrix(array: pa.ChunkedArray, width: int) -> np.ndarray:
    combined = array.combine_chunks()
    if combined.null_count:
        raise ValueError("numeric feature contains null values")
    if pa.types.is_fixed_size_list(combined.type):
        values = np.asarray(
            combined.values.to_numpy(zero_copy_only=False), dtype=np.float64
        )
        return values.reshape(-1, width)
    return np.asarray(combined.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1, 1)


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("statistics require a nonempty finite matrix")
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [float(len(values))],
    }


def _row_stats(row: Mapping[str, Any], feature: str) -> dict[str, Any]:
    return {
        metric: row.get(f"stats/{feature}/{metric}") for metric in _STAT_METRICS
    }


def _compare_stats(
    published: Any,
    actual: Mapping[str, list[float]],
    context: str,
    *,
    tolerance: float,
) -> None:
    width = len(actual["min"])
    _validate_stats_shape(published, width, round(actual["count"][0]), context)
    assert isinstance(published, dict)
    for metric in _STAT_METRICS:
        left = _flatten(published[metric])
        right = actual[metric]
        if len(left) != len(right):
            raise ValueError(f"{context} stats shape mismatch")
        differs = any(
            abs(a - b) > tolerance * max(1.0, abs(b))
            for a, b in zip(left, right, strict=True)
        )
        if differs:
            raise ValueError(f"{context} differs from payload")


def _validate_stats_shape(
    value: Any, width: int, expected_count: int, context: str
) -> None:
    if not isinstance(value, dict) or set(value) != set(_STAT_METRICS):
        raise ValueError(f"{context} metric coverage mismatch")
    count = _flatten(value["count"])
    if len(count) != 1 or not math.isclose(count[0], expected_count):
        raise ValueError(f"{context} count differs from episode/frame count")
    for metric in _STAT_METRICS[:-1]:
        numbers = _flatten(value[metric])
        if len(numbers) != width or not all(math.isfinite(item) for item in numbers):
            raise ValueError(f"{context} metric shape or values are invalid")
    minimum = _flatten(value["min"])
    maximum = _flatten(value["max"])
    mean = _flatten(value["mean"])
    standard_deviation = _flatten(value["std"])
    if any(
        low > average or average > high or deviation < 0
        for low, average, high, deviation in zip(
            minimum, mean, maximum, standard_deviation, strict=True
        )
    ):
        raise ValueError(f"{context} statistic ordering is invalid")


def _flatten(value: Any) -> list[float]:
    if isinstance(value, list):
        result: list[float] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("statistics must contain finite numbers")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("statistics must contain finite numbers")
    return [number]


def _validate_annotations(
    root: Path,
    source_root: Path | None,
    expected_output_root: Path | None,
    annotations: dict[str, Any],
    episodes: Sequence[_Episode],
    cameras: Sequence[str],
    fps: float,
    services: ReleaseServices,
) -> tuple[list[Any], str, BoundaryPreview | None]:
    top_fields = {
        "source_root",
        "work_dir",
        "subtask_template",
        "episodes",
        "primary_camera",
        "updated_at",
    }
    if set(annotations) not in (top_fields, top_fields | {"augmentation"}):
        raise ValueError("annotation top-level schema mismatch")
    augmentation_language = None
    if "augmentation" in annotations:
        augmentation = annotations["augmentation"]
        fields = {"enabled", "language", "model_repo", "model_revision", "prompt_version"}
        if (
            not isinstance(augmentation, dict)
            or set(augmentation) != fields
            or augmentation.get("enabled") is not True
        ):
            raise ValueError("annotation augmentation schema mismatch")
        for field in fields - {"enabled"}:
            _string(augmentation, field)
        augmentation_language = augmentation["language"]
    _aware_utc(annotations.get("updated_at"), "updated_at")
    declared_source = Path(_string(annotations, "source_root"))
    if not declared_source.is_absolute():
        raise ValueError("annotation source_root must be absolute")
    if source_root is not None and declared_source.resolve(strict=False) != source_root:
        raise ValueError("annotation source_root does not match supplied source")
    declared_work = Path(_string(annotations, "work_dir"))
    if not declared_work.is_absolute() or declared_work.name != "meta":
        raise ValueError("annotation work_dir must be an absolute path ending in meta")
    if expected_output_root is not None:
        if not isinstance(expected_output_root, Path):
            raise TypeError("expected_output_root must be a Path")
        expected_work = expected_output_root.resolve(strict=False) / "meta"
        if declared_work.resolve(strict=False) != expected_work:
            raise ValueError("annotation work_dir does not match expected output/meta")
    primary = _string(annotations, "primary_camera")
    if primary not in cameras:
        raise ValueError("primary camera is absent from features")
    template = _subtasks(annotations.get("subtask_template"))
    entries = annotations.get("episodes")
    expected_keys = {str(episode.index) for episode in episodes}
    if not isinstance(entries, dict) or set(entries) != expected_keys:
        raise ValueError("annotation episodes must exactly match dataset episodes")
    has_starts = [
        isinstance(entry, dict) and "start_subtask_index" in entry
        for entry in entries.values()
    ]
    if all(has_starts):
        mode = "dagger_patch"
    elif not any(has_starts):
        mode = "complete"
    else:
        raise ValueError("all DAgger records must include start_subtask_index")
    _reject_forbidden(annotations)

    instruction_names: list[str] = []
    annotation_facts: list[tuple[int, list[int]]] = []
    preview: BoundaryPreview | None = None
    for episode in episodes:
        entry = entries[str(episode.index)]
        common = {"episode_index", "boundaries", "high_level_instruction", "saved_at"}
        allowed = common | ({"start_subtask_index"} if mode == "dagger_patch" else set())
        if not isinstance(entry, dict) or set(entry) != allowed:
            raise ValueError("annotation episode schema mismatch")
        if _integer(entry, "episode_index", 0) != episode.index:
            raise ValueError("annotation episode key/index mismatch")
        instruction_names.append(_string(entry, "high_level_instruction"))
        _aware_utc(entry.get("saved_at"), "saved_at")
        boundaries = entry.get("boundaries")
        if not isinstance(boundaries, list) or any(type(item) is not int for item in boundaries):
            raise ValueError("boundaries must be an integer list")
        start = 0 if mode == "complete" else _integer(entry, "start_subtask_index", 0)
        annotation = FinalAnnotation(start_subtask_index=start, boundaries=boundaries)
        issues = validate_annotation(annotation, mode, len(template), episode.length, 1)
        if issues:
            issue_codes = ",".join(issue.code for issue in issues)
            raise ValueError("invalid annotation boundaries: " + issue_codes)
        annotation_facts.append((start, boundaries))
        if preview is None and boundaries:
            boundary = boundaries[0]
            video = episode.videos[primary]
            ref = EpisodeVideoRef(
                path=video.path,
                from_timestamp=video.from_timestamp,
                to_timestamp=video.to_timestamp,
                fps=fps,
            )
            requested = [boundary - 1, boundary]
            samples = services.extract_frames(ref, primary, requested)
            labels_differ = (
                len(samples) != 2
                or [item.frame_index for item in samples] != requested
                or any(item.camera_key != primary for item in samples)
            )
            if labels_differ:
                raise ValueError(
                    "boundary preview labels do not match requested episode-local frames"
                )
            preview = BoundaryPreview(
                episode_index=episode.index,
                camera_key=primary,
                frame_indices=(boundary - 1, boundary),
            )
    _validate_task_info(
        root,
        len(episodes),
        instruction_names,
        [episode.length for episode in episodes],
        template,
        annotation_facts,
        augmentation_language,
    )
    return template, mode, preview


def _official_core_files(root: Path) -> set[str]:
    result = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "meta/lerobot_annotations.json" or relative.startswith("meta/task_info/"):
            continue
        result.add(relative)
    return result


def _template(value: Any, label: str, expected_fields: set[str]) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    try:
        parsed = list(Formatter().parse(value))
    except ValueError as exc:
        raise ValueError(f"invalid {label} template") from exc
    fields: list[str] = []
    for _, name, spec, conversion in parsed:
        if name is None:
            continue
        if name not in expected_fields or "{" in spec or "}" in spec or conversion is not None:
            raise ValueError(f"unsafe {label} template")
        fields.append(name)
    if len(fields) != len(expected_fields) or set(fields) != expected_fields:
        raise ValueError(f"{label} template fields must be exactly {sorted(expected_fields)}")
    return value


def _render(
    template: str,
    values: Mapping[str, Any],
    label: str,
    required_root: str,
) -> Path:
    try:
        relative = Path(template.format(**values))
    except Exception as exc:
        raise ValueError(f"invalid {label} template") from exc
    unsafe = (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != required_root
    )
    if unsafe:
        raise ValueError(f"{label} path escapes {required_root}")
    return relative


def _strict_int(value: Any, field: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _nonnegative_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number >= 0")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be a finite number >= 0")
    return result


def _positive_number(value: Any, field: str) -> float:
    result = _nonnegative_number(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result

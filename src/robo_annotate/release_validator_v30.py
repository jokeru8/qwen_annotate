"""Independent validation of publishable LeRobot v3.0 shared-shard releases.

This module intentionally does not import the v3 dataset inspection adapter.  A
publisher therefore cannot prove its own output through the same parser used at
ingest time.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from string import Formatter
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .constraints import validate_annotation
from .lerobot import EpisodeVideoRef, data_timestamp_matches, video_fps_matches
from .models import FinalAnnotation
from .release_validator import (
    BoundaryPreview,
    ReleaseReport,
    ReleaseServices,
    _aware_utc,
    _integer,
    _reject_forbidden,
    _string,
    _subtasks,
)
from .secure_tree import SecureFile, SecureTree
from .v30_depth import (
    DepthMetadata,
    dequantize_depth,
    depth_metadata,
    depth_quantization_tolerance,
    is_depth_feature,
    iter_depth_codes,
)


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
_BASIC_STAT_METRICS = ("min", "max", "mean", "std", "count")
_CHUNK = re.compile(r"chunk-([0-9]{3})\Z")
_FILE = re.compile(r"file-([0-9]{3})\.parquet\Z")
_DEFAULT_CHUNKS_SIZE = 1000
_DEFAULT_DATA_FILE_SIZE_MB = 100
_DEFAULT_VIDEO_FILE_SIZE_MB = 200
_DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
_DEFAULT_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_PARQUET_BYTES = 1024 * 1024 * 1024
_MAX_VIDEO_BYTES = 16 * 1024 * 1024 * 1024

_VideoShardIdentity = tuple[str, int, int, str]
_VideoRange = tuple[float, float, int]


@dataclass(frozen=True)
class _VideoSlice:
    relative: str
    file: SecureFile
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class _Episode:
    index: int
    length: int
    task: str
    data_path: str
    data_offset: int
    videos: dict[str, _VideoSlice]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _DataFeature:
    name: str
    dtype: str
    shape: tuple[int, ...]
    arrow_type: pa.DataType
    stats_shape: tuple[int, ...] | None
    depth: DepthMetadata | None


@dataclass(frozen=True)
class _CameraFeature:
    width: int
    height: int
    depth: DepthMetadata | None


def validate_v30_release(
    root: Path,
    *,
    source_root: Path | None,
    services: ReleaseServices,
    expected_output_root: Path | None,
    deep_video_stats: bool,
) -> ReleaseReport:
    """Validate v3 directly while retaining the same safe public boundary."""
    with SecureTree(root, "release") as release_tree:
        release_tree.scan()
        info_bytes = _read_bytes(release_tree, "meta/info.json", _MAX_JSON_BYTES, "info.json")
        info = _decode_object(info_bytes, "info.json")
        if source_root is None:
            return _validate_v30_release_with_info(
                release_tree.path,
                source_root=None,
                services=services,
                expected_output_root=expected_output_root,
                deep_video_stats=deep_video_stats,
                info=info,
                release_tree=release_tree,
                source_tree=None,
                info_digest=hashlib.sha256(info_bytes).hexdigest(),
            )
        with SecureTree(source_root, "source") as source_tree:
            source_tree.scan()
            return _validate_v30_release_with_info(
                release_tree.path,
                source_root=source_tree.path,
                services=services,
                expected_output_root=expected_output_root,
                deep_video_stats=deep_video_stats,
                info=info,
                release_tree=release_tree,
                source_tree=source_tree,
                info_digest=hashlib.sha256(info_bytes).hexdigest(),
            )


def _validate_v30_release_with_info(
    root: Path,
    *,
    source_root: Path | None,
    services: ReleaseServices,
    expected_output_root: Path | None,
    deep_video_stats: bool,
    info: dict[str, Any],
    release_tree: SecureTree,
    source_tree: SecureTree | None,
    info_digest: str,
) -> ReleaseReport:
    """Validate v3 using the facade's single bounded info.json read."""
    if _string(info, "codebase_version") != "v3.0":
        raise ValueError("v3 release validator requires codebase_version 'v3.0'")
    if type(deep_video_stats) is not bool:
        raise TypeError("deep_video_stats must be a bool")

    total_episodes = _strict_int(info.get("total_episodes"), "total_episodes", 0)
    total_frames = _strict_int(info.get("total_frames"), "total_frames", 0)
    total_tasks = _strict_int(info.get("total_tasks"), "total_tasks", 0)
    chunks_size = _strict_int(
        info.get("chunks_size", _DEFAULT_CHUNKS_SIZE), "chunks_size", 1
    )
    fps = _positive_number(info.get("fps"), "fps")
    _strict_int(
        info.get("data_files_size_in_mb", _DEFAULT_DATA_FILE_SIZE_MB),
        "data_files_size_in_mb",
        1,
    )
    _strict_int(
        info.get("video_files_size_in_mb", _DEFAULT_VIDEO_FILE_SIZE_MB),
        "video_files_size_in_mb",
        1,
    )
    data_template = _template(
        info.get("data_path", _DEFAULT_DATA_PATH),
        "data_path",
        {"chunk_index", "file_index"},
    )
    video_template = _template(
        info.get("video_path", _DEFAULT_VIDEO_PATH),
        "video_path",
        {"video_key", "chunk_index", "file_index"},
    )
    episodes_template = None
    if "episodes_path" in info:
        episodes_template = _template(
            info.get("episodes_path", _DEFAULT_EPISODES_PATH),
            "episodes_path",
            {"chunk_index", "file_index"},
        )
    _validate_optional_splits(info.get("splits", {}), total_episodes)

    features = info.get("features")
    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a nonempty object")
    data_features, camera_shapes = _feature_shapes(features, fps)
    numeric_shapes = {
        feature.name: feature.stats_shape
        for feature in data_features.values()
        if feature.stats_shape is not None
    }
    cameras = list(camera_shapes)
    if not cameras:
        raise ValueError("v3 release must declare at least one video camera")
    core_digests = {"meta/info.json": info_digest}
    tasks = _read_tasks(release_tree, core_digests)
    if len(tasks) != total_tasks:
        raise ValueError("total_tasks differs from tasks.parquet")
    task_by_text = {text: index for index, text in tasks.items()}

    published_stats_bytes = _read_bytes(
        release_tree, "meta/stats.json", _MAX_JSON_BYTES, "stats.json"
    )
    published_stats = _decode_object(published_stats_bytes, "stats.json")
    core_digests["meta/stats.json"] = hashlib.sha256(published_stats_bytes).hexdigest()
    stats_profiles = _stats_profiles(
        published_stats, set(numeric_shapes) | set(cameras)
    )

    episode_shards = _read_episode_shards(release_tree, core_digests)
    if episodes_template is not None:
        for chunk_index, file_index, _, path in episode_shards:
            relative = _render(
                episodes_template,
                {"chunk_index": chunk_index, "file_index": file_index},
                "episodes_path",
                "meta",
            )
            if relative.parts[:2] != ("meta", "episodes") or relative.as_posix() != path:
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
        for metric in stats_profiles[feature]
    }
    reference_episode_schema: pa.Schema | None = None
    for _, _, table, relative in episode_shards:
        actual = set(table.column_names)
        if actual != required_episode_columns | expected_stat_columns:
            raise ValueError(
                "episode metadata schema mismatch: "
                f"missing={sorted((required_episode_columns | expected_stat_columns) - actual)}, "
                f"extra={sorted(actual - required_episode_columns - expected_stat_columns)}"
            )
        _validate_episode_arrow_schema(
            table,
            data_features,
            camera_shapes,
            stats_profiles,
            context=f"episode metadata {relative}",
        )
        if reference_episode_schema is None:
            reference_episode_schema = table.schema
        elif not table.schema.equals(reference_episode_schema, check_metadata=False):
            raise ValueError("episode metadata shards must have identical ordered schemas")

    data_tables: dict[str, pa.Table] = {}
    data_pairs: set[tuple[int, int]] = set()
    data_pair_paths: dict[tuple[int, int], str] = {}
    data_coverage: dict[str, set[int]] = defaultdict(set)
    video_files: dict[str, SecureFile] = {}
    video_probes: dict[str, Any] = {}
    video_ranges: dict[_VideoShardIdentity, list[_VideoRange]] = defaultdict(list)
    video_pairs: dict[str, set[tuple[int, int]]] = defaultdict(set)
    video_pair_paths: dict[tuple[str, int, int], str] = {}
    video_path_identities: dict[str, tuple[str, int, int]] = {}
    episodes: list[_Episode] = []
    expected_global = 0
    expected_payload: set[str] = set()
    expected_episode_shards = {
        path for _, _, _, path in episode_shards
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
        data_path = data_relative.as_posix()
        data_pairs.add((data_chunk, data_file))
        prior_data_path = data_pair_paths.setdefault(
            (data_chunk, data_file), data_path
        )
        if prior_data_path != data_path:
            raise ValueError("one data shard number resolves to multiple paths")
        expected_payload.add(data_relative.as_posix())
        if data_path not in data_tables:
            data_tables[data_path] = _read_data(
                release_tree, data_path, data_features, core_digests
            )
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
            video_file_index = _strict_int(
                row.get(f"videos/{camera}/file_index"), "file_index", 0
            )
            if video_file_index >= chunks_size:
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
                    "file_index": video_file_index,
                },
                "video_path",
                "videos",
            )
            video_path = video_relative.as_posix()
            video_pairs[camera].add((video_chunk, video_file_index))
            identity = (camera, video_chunk, video_file_index)
            prior_path = video_pair_paths.setdefault(identity, video_path)
            if prior_path != video_path:
                raise ValueError("one video shard number resolves to multiple paths")
            prior_identity = video_path_identities.setdefault(video_path, identity)
            if prior_identity != identity:
                raise ValueError("video shard identities must resolve to unique canonical locations")
            expected_payload.add(video_relative.as_posix())
            if video_path not in video_probes:
                opened_video = release_tree.open_file(
                    video_path, _MAX_VIDEO_BYTES, "video shard"
                )
                video_files[video_path] = opened_video
                try:
                    video_probes[video_path] = services.probe_video(opened_video.proc_path)
                finally:
                    opened_video.verify()
            probe = video_probes[video_path]
            camera_feature = camera_shapes[camera]
            width, height = camera_feature.width, camera_feature.height
            if (
                not video_fps_matches(probe.fps, fps)
                or probe.width != width
                or probe.height != height
            ):
                raise ValueError(f"{camera_context} video fps or shape mismatch")
            if to_timestamp > probe.frames / probe.fps + 1.0 / fps + 1e-9:
                raise ValueError(f"{camera_context} timestamp range exceeds video shard")
            video_ranges[(camera, video_chunk, video_file_index, video_path)].append(
                (from_timestamp, to_timestamp, episode_index)
            )
            episode_videos[camera] = _VideoSlice(
                video_path, video_files[video_path], from_timestamp, to_timestamp
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
    _validate_video_coverage(video_ranges, video_probes, fps, cameras)

    actual_payload = _payload_files_secure(release_tree)
    if actual_payload != expected_payload:
        raise ValueError("missing or extra v3 payload files")
    actual_episode_shards = set(
        release_tree.files_under("meta/episodes", suffix=".parquet")
    )
    if actual_episode_shards != expected_episode_shards:
        raise ValueError("missing or extra episode metadata shards")

    expected_stats_features = set(numeric_shapes) | set(cameras)
    if set(published_stats) != expected_stats_features:
        raise ValueError("aggregate stats feature coverage mismatch")
    aggregate_numeric, episode_numeric = _numeric_stats(
        data_tables,
        episodes,
        {name: data_features[name] for name in numeric_shapes},
        stats_profiles,
    )
    for feature, actual in aggregate_numeric.items():
        _compare_stats(
            published_stats[feature], actual, f"stats {feature}", tolerance=1e-6
        )
    for episode in episodes:
        for feature, actual in episode_numeric[episode.index].items():
            _compare_stats(
                _row_stats(episode.metadata, feature, stats_profiles[feature]),
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
            video_files,
            stats_profiles,
        )
        # Published v3 image statistics originate before lossy H.264 encoding.
        # A four-code-value envelope covers codec drift while still rejecting
        # semantic/statistical corruption.
        codec_tolerance = 4.0 / 255.0 + 1e-6
        for camera, actual in aggregate_video.items():
            depth = camera_shapes[camera].depth
            _compare_stats(
                published_stats[camera],
                actual,
                f"stats {camera}",
                tolerance=1e-6 if depth is not None else codec_tolerance,
                absolute_tolerance=(
                    2.0 * depth_quantization_tolerance(depth)
                    if depth is not None
                    else 0.0
                ),
            )
        for episode in episodes:
            for camera, actual in episode_video[episode.index].items():
                depth = camera_shapes[camera].depth
                _compare_stats(
                    _row_stats(episode.metadata, camera, stats_profiles[camera]),
                    actual,
                    f"episode {episode.index} stats {camera}",
                    tolerance=1e-6 if depth is not None else codec_tolerance,
                    absolute_tolerance=(
                        2.0 * depth_quantization_tolerance(depth)
                        if depth is not None
                        else 0.0
                    ),
                )
    else:
        sampled_total = sum(len(_sample_indices(episode.length)) for episode in episodes)
        for camera in cameras:
            channels = 1 if camera_shapes[camera].depth is not None else 3
            _validate_stats_shape(
                published_stats[camera],
                (channels, 1, 1),
                sampled_total,
                f"stats {camera}",
                stats_profiles[camera],
            )
        for episode in episodes:
            for camera in cameras:
                channels = 1 if camera_shapes[camera].depth is not None else 3
                _validate_stats_shape(
                    _row_stats(episode.metadata, camera, stats_profiles[camera]),
                    (channels, 1, 1),
                    len(_sample_indices(episode.length)),
                    f"episode {episode.index} stats {camera}",
                    stats_profiles[camera],
                )

    annotations = _read_secure_object(
        release_tree, "meta/lerobot_annotations.json", "lerobot_annotations.json"
    )
    template, mode, preview = _validate_annotations(
        root,
        source_root,
        expected_output_root,
        annotations,
        episodes,
        cameras,
        fps,
        services,
        release_tree,
    )

    for relative, video_file in video_files.items():
        core_digests[relative] = video_file.sha256()
    digests = {relative: core_digests[relative] for relative in sorted(expected_payload)}
    if source_tree is not None:
        release_core = _official_core_files(release_tree)
        source_core = _official_core_files(source_tree)
        if release_core != source_core:
            raise ValueError("official v3 source/release file inventory differs")
        for relative in sorted(release_core):
            release_digest = core_digests.get(relative)
            if release_digest is None:
                release_digest = _secure_digest(release_tree, relative, "official release file")
            source_digest = _secure_digest(source_tree, relative, "official source file")
            if release_digest != source_digest:
                raise ValueError(f"official v3 source byte mismatch: {relative}")
    release_tree.verify()
    aggregate_digest = hashlib.sha256(
        "".join(
            f"{name}\0{digests[name]}\n" for name in sorted(digests)
        ).encode()
    ).hexdigest()
    return ReleaseReport(
        path=release_tree.path,
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
) -> tuple[dict[str, _DataFeature], dict[str, _CameraFeature]]:
    data: dict[str, _DataFeature] = {}
    cameras: dict[str, _CameraFeature] = {}
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
            if len(shape) != 3:
                raise ValueError(f"video feature shape must have rank 3: {name}")
            video_info = value.get("info")
            if not isinstance(video_info, dict):
                raise ValueError(f"video feature info is invalid: {name}")
            video_fps = _positive_number(video_info.get("video.fps"), "video.fps")
            width = _strict_int(video_info.get("video.width"), "video.width", 1)
            height = _strict_int(video_info.get("video.height"), "video.height", 1)
            depth = depth_metadata(value, name) if is_depth_feature(value) else None
            channels = 1 if depth is not None else 3
            if not video_fps_matches(video_fps, fps) or shape not in (
                [channels, height, width],
                [height, width, channels],
            ):
                raise ValueError(f"video feature fps or shape is inconsistent: {name}")
            declared_channels = video_info.get("video.channels")
            if declared_channels is not None and declared_channels != channels:
                raise ValueError(f"video feature channel count is inconsistent: {name}")
            cameras[name] = _CameraFeature(width, height, depth)
        elif dtype == "string":
            if shape != [1]:
                raise ValueError(f"string feature shape must be [1]: {name}")
            data[name] = _DataFeature(name, dtype, (1,), pa.string(), None, None)
        elif dtype == "language":
            if name not in {"language_persistent", "language_events"} or shape != [1]:
                raise ValueError(f"unsupported language feature declaration: {name}")
            arrow_type = (
                _language_persistent_arrow_type()
                if name == "language_persistent"
                else _language_events_arrow_type()
            )
            data[name] = _DataFeature(name, dtype, (1,), arrow_type, None, None)
        elif dtype == "image":
            depth = depth_metadata(value, name) if is_depth_feature(value) else None
            channels = 1 if depth is not None else 3
            if len(shape) != 3 or (shape[0] != channels and shape[-1] != channels):
                kind = "depth" if depth is not None else "RGB"
                raise ValueError(f"{kind} image feature shape must be CHW or HWC: {name}")
            data[name] = _DataFeature(
                name,
                dtype,
                tuple(shape),
                _image_arrow_type(),
                (channels, 1, 1),
                depth,
            )
        else:
            try:
                numpy_dtype = np.dtype(dtype)
                arrow_type = pa.from_numpy_dtype(numpy_dtype)
            except (TypeError, ValueError, NotImplementedError, pa.ArrowNotImplementedError) as exc:
                raise ValueError(f"unsupported declared feature dtype: {name}") from exc
            if len(shape) > 5:
                raise ValueError(f"feature shape rank exceeds official HF support: {name}")
            data[name] = _DataFeature(
                name,
                dtype,
                tuple(shape),
                arrow_type,
                tuple(shape),
                None,
            )
    official_defaults = {
        "timestamp": ("float32", (1,)),
        "frame_index": ("int64", (1,)),
        "episode_index": ("int64", (1,)),
        "index": ("int64", (1,)),
        "task_index": ("int64", (1,)),
    }
    for name, (dtype, shape) in official_defaults.items():
        feature = data.get(name)
        if feature is None or (feature.dtype, feature.shape) != (dtype, shape):
            raise ValueError(f"official default feature declaration mismatch: {name}")
    return data, cameras


def _language_persistent_arrow_type() -> pa.ListType:
    return pa.list_(
        pa.struct(
            [
                pa.field("role", pa.string()),
                pa.field("content", pa.string()),
                pa.field("style", pa.string()),
                pa.field("timestamp", pa.float32()),
                pa.field("camera", pa.string()),
                pa.field("tool_calls", pa.list_(_json_arrow_type())),
            ]
        )
    )


def _language_events_arrow_type() -> pa.ListType:
    return pa.list_(
        pa.struct(
            [
                pa.field("role", pa.string()),
                pa.field("content", pa.string()),
                pa.field("style", pa.string()),
                pa.field("camera", pa.string()),
                pa.field("tool_calls", pa.list_(_json_arrow_type())),
            ]
        )
    )


def _json_arrow_type() -> pa.DataType:
    """Match LeRobot v0.6.1's runtime JSON feature without importing it."""
    return pa.json_() if hasattr(pa, "json_") else pa.string()


def _image_arrow_type() -> pa.StructType:
    return pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])


def _feature_arrow_type_matches(actual: pa.DataType, feature: _DataFeature) -> bool:
    storage = getattr(actual, "storage_type", actual)
    if feature.dtype in {"string", "language", "image"}:
        return storage.equals(feature.arrow_type)
    if feature.shape == (1,):
        return storage.equals(feature.arrow_type)

    def matches(value: pa.DataType, dimensions: tuple[int, ...]) -> bool:
        value = getattr(value, "storage_type", value)
        if not dimensions:
            return value.equals(feature.arrow_type)
        if pa.types.is_fixed_size_list(value):
            return value.list_size == dimensions[0] and matches(
                value.value_type, dimensions[1:]
            )
        # HF ArrayND extension storage may use variable nested lists while its
        # declared shape lives in info.json; values below are checked exactly.
        if pa.types.is_list(value) or pa.types.is_large_list(value):
            return matches(value.value_type, dimensions[1:])
        return False

    return matches(storage, feature.shape)


def _validate_feature_values(array: pa.ChunkedArray, feature: _DataFeature) -> None:
    if feature.dtype == "image":
        for value in array.to_pylist():
            if (
                not isinstance(value, dict)
                or not isinstance(value.get("bytes"), bytes)
                or not isinstance(value.get("path"), (str, type(None)))
            ):
                raise ValueError(
                    f"embedded image feature must contain bytes and a string/null path: {feature.name}"
                )
        return
    if feature.dtype == "language":
        _validate_language_values(array, feature.name)
        return
    if feature.dtype == "string" or feature.shape == (1,):
        return

    def valid_shape(value: Any, dimensions: tuple[int, ...]) -> bool:
        if not dimensions:
            return not isinstance(value, (list, tuple))
        return (
            isinstance(value, (list, tuple))
            and len(value) == dimensions[0]
            and all(valid_shape(child, dimensions[1:]) for child in value)
        )

    if any(not valid_shape(value, feature.shape) for value in array.to_pylist()):
        raise ValueError(f"data values disagree with declared feature shape: {feature.name}")


def _validate_language_values(array: pa.ChunkedArray, feature_name: str) -> None:
    persistent = feature_name == "language_persistent"
    expected_fields = {
        "role",
        "content",
        "style",
        "camera",
        "tool_calls",
        *({"timestamp"} if persistent else set()),
    }
    for frame_rows in array.to_pylist():
        if not isinstance(frame_rows, list):
            raise ValueError(f"{feature_name} must contain lists of language rows")
        for row in frame_rows:
            if not isinstance(row, dict) or set(row) != expected_fields:
                raise ValueError(f"{feature_name} has an invalid language row")
            if not isinstance(row["role"], str) or not row["role"]:
                raise ValueError(f"{feature_name} has an invalid language role")
            for field in ("content", "style", "camera"):
                if row[field] is not None and not isinstance(row[field], str):
                    raise ValueError(f"{feature_name} has an invalid {field} value")
            if persistent:
                timestamp = row["timestamp"]
                if (
                    isinstance(timestamp, bool)
                    or not isinstance(timestamp, (int, float))
                    or not math.isfinite(float(timestamp))
                ):
                    raise ValueError(f"{feature_name} has an invalid timestamp")
            tool_calls = row["tool_calls"]
            if tool_calls is None:
                continue
            if not isinstance(tool_calls, list):
                raise ValueError(f"{feature_name} has invalid tool_calls")
            for tool_call in tool_calls:
                if not isinstance(tool_call, str):
                    raise ValueError(f"{feature_name} has an invalid JSON tool call")
                try:
                    json.loads(tool_call)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{feature_name} has an invalid JSON tool call"
                    ) from exc


def _validate_episode_arrow_schema(
    table: pa.Table,
    data_features: Mapping[str, _DataFeature],
    cameras: Mapping[str, _CameraFeature],
    profiles: Mapping[str, tuple[str, ...]],
    *,
    context: str,
) -> None:
    exact = {
        "episode_index": pa.int64(),
        "tasks": pa.list_(pa.string()),
        "length": pa.int64(),
        "data/chunk_index": pa.int64(),
        "data/file_index": pa.int64(),
        "dataset_from_index": pa.int64(),
        "dataset_to_index": pa.int64(),
        "meta/episodes/chunk_index": pa.int64(),
        "meta/episodes/file_index": pa.int64(),
    }
    for camera in cameras:
        exact[f"videos/{camera}/chunk_index"] = pa.int64()
        exact[f"videos/{camera}/file_index"] = pa.int64()
        exact[f"videos/{camera}/from_timestamp"] = pa.float64()
        exact[f"videos/{camera}/to_timestamp"] = pa.float64()
    for name, expected in exact.items():
        field = table.schema.field(name)
        if not field.nullable or not field.type.equals(expected) or table[name].null_count:
            raise ValueError(f"{context} has nonofficial Arrow field {name}")
    for feature, profile in profiles.items():
        for metric in profile:
            name = f"stats/{feature}/{metric}"
            field = table.schema.field(name)
            if metric == "count":
                expected = pa.list_(pa.int64())
            else:
                if feature in cameras:
                    channels = 1 if cameras[feature].depth is not None else 3
                    stats_shape = (channels, 1, 1)
                else:
                    stats_shape = data_features[feature].stats_shape
                if stats_shape is None:
                    raise ValueError(f"{context} has statistics for nonnumeric feature {feature}")
                expected = _nested_stats_arrow_type(stats_shape)
            if not field.nullable or not field.type.equals(expected) or table[name].null_count:
                raise ValueError(f"{context} has nonofficial Arrow field {name}")


def _nested_stats_arrow_type(shape: tuple[int, ...]) -> pa.DataType:
    value: pa.DataType = pa.float64()
    for _dimension in reversed(shape):
        value = pa.list_(value)
    return value


def _read_tasks(tree: SecureTree, digests: dict[str, str]) -> dict[int, str]:
    table = _read_parquet(tree, "meta/tasks.parquet", "tasks.parquet", digests)
    expected_schema = pa.schema([pa.field("task_index", pa.int64()), pa.field("task", pa.string())])
    if not table.schema.equals(expected_schema, check_metadata=False):
        raise ValueError(
            "tasks.parquet schema must be ordered nullable task_index:int64, task:string"
        )
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
    tree: SecureTree,
    digests: dict[str, str],
) -> list[tuple[int, int, pa.Table, str]]:
    result: list[tuple[int, int, pa.Table, str]] = []
    all_under = tree.files_under("meta/episodes")
    for relative in all_under:
        parts = PurePosixPath(relative).parts
        if len(parts) != 4:
            raise ValueError(f"unexpected episode metadata entry: {relative}")
        chunk_match = _CHUNK.fullmatch(parts[2])
        file_match = _FILE.fullmatch(parts[3])
        if chunk_match is None or file_match is None:
            raise ValueError(f"unexpected episode metadata file: {relative}")
        chunk_index = int(chunk_match.group(1))
        file_index = int(file_match.group(1))
        result.append(
            (
                chunk_index,
                file_index,
                _read_parquet(tree, relative, "episode metadata", digests),
                relative,
            )
        )
    result.sort(key=lambda item: (item[0], item[1]))
    if not result:
        raise ValueError("episode metadata has no parquet shards")
    return result


def _read_parquet(
    tree: SecureTree,
    relative: str,
    context: str,
    digests: dict[str, str],
) -> pa.Table:
    data = _read_bytes(tree, relative, _MAX_PARQUET_BYTES, context)
    digests[relative] = hashlib.sha256(data).hexdigest()
    try:
        return pq.read_table(pa.BufferReader(data))
    except Exception as exc:
        raise ValueError(f"unable to read {context}: {relative}") from exc


def _read_data(
    tree: SecureTree,
    relative: str,
    features: Mapping[str, _DataFeature],
    digests: dict[str, str],
) -> pa.Table:
    table = _read_parquet(tree, relative, "data shard", digests)
    expected_names = list(features)
    if table.column_names != expected_names:
        raise ValueError(
            "data schema feature inventory mismatch: "
            f"expected ordered={expected_names}, actual={table.column_names}"
        )
    for feature, declared in features.items():
        field = table.schema.field(feature)
        if not field.nullable or table[feature].null_count:
            raise ValueError(f"data schema requires a nullable, populated field: {feature}")
        if not _feature_arrow_type_matches(field.type, declared):
            raise ValueError(f"data schema disagrees with feature declaration: {feature}")
        _validate_feature_values(table[feature], declared)
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
        if not data_timestamp_matches(timestamp, local_index, fps):
            raise ValueError(f"episode {episode_index} data timestamp mismatch")
    return offset


def _validate_video_coverage(
    ranges: Mapping[_VideoShardIdentity, list[_VideoRange]],
    probes: Mapping[str, Any],
    fps: float,
    cameras: Sequence[str],
) -> None:
    tolerance = 1.0 / fps + 1e-9
    expected_episode: dict[str, int] = defaultdict(int)
    camera_order = {camera: index for index, camera in enumerate(cameras)}
    ordered = sorted(
        ranges,
        key=lambda identity: (
            camera_order[identity[0]],
            identity[1],
            identity[2],
        ),
    )
    for camera, chunk_index, file_index, path in ordered:
        items = ranges[(camera, chunk_index, file_index, path)]
        expected = 0.0
        for start, stop, episode_index in items:
            if episode_index != expected_episode[camera]:
                raise ValueError(
                    f"video episode ranges are not in canonical physical video order for camera {camera}"
                )
            if start < expected - 1e-9:
                raise ValueError(
                    f"video slice overlap or noncanonical physical video order for camera {camera}, episode {episode_index}"
                )
            if start > expected + tolerance:
                raise ValueError(
                    f"video coverage gap or noncanonical physical video order for camera {camera}, episode {episode_index}"
                )
            expected = stop
            expected_episode[camera] += 1
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
    tables: Mapping[str, pa.Table],
    episodes: Sequence[_Episode],
    features: Mapping[str, _DataFeature],
    profiles: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, dict[str, list[float]]], dict[int, dict[str, dict[str, list[float]]]]]:
    numpy_by_episode: dict[
        int,
        dict[str, dict[str, np.ndarray[Any, Any]]],
    ] = {}
    for episode in episodes:
        table = tables[episode.data_path].slice(episode.data_offset, episode.length)
        numpy_by_episode[episode.index] = {}
        for feature, declaration in features.items():
            dtype = (
                np.dtype(np.float64 if declaration.depth is not None else np.uint8)
                if declaration.dtype == "image"
                else np.dtype(declaration.dtype)
            )
            matrix, count = _matrix(
                table[feature],
                dtype,
                declaration.stats_shape or (1,),
                sample_images=declaration.dtype == "image",
                depth=declaration.depth,
            )
            numpy_by_episode[episode.index][feature] = _stats(
                matrix,
                profiles[feature],
                count,
                image=declaration.dtype == "image" and declaration.depth is None,
                depth=declaration.depth is not None,
            )
    numpy_aggregate = {
        feature: _aggregate_stats(
            [numpy_by_episode[episode.index][feature] for episode in episodes],
            profiles[feature],
        )
        for feature in features
    }
    aggregate = {
        feature: _stats_to_lists(stats)
        for feature, stats in numpy_aggregate.items()
    }
    by_episode = {
        episode.index: {
            feature: _stats_to_lists(stats)
            for feature, stats in numpy_by_episode[episode.index].items()
        }
        for episode in episodes
    }
    return aggregate, by_episode


def _video_stats(
    episodes: Sequence[_Episode],
    cameras: Sequence[str],
    shapes: Mapping[str, _CameraFeature],
    ranges: Mapping[_VideoShardIdentity, list[_VideoRange]],
    services: ReleaseServices,
    fps: float,
    files: Mapping[str, SecureFile],
    profiles: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, dict[str, list[float]]], dict[int, dict[str, dict[str, list[float]]]]]:
    episode_values: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    camera_order = {camera: index for index, camera in enumerate(cameras)}
    ordered_ranges = sorted(
        ranges.items(),
        key=lambda item: (
            camera_order[item[0][0]],
            item[0][1],
            item[0][2],
        ),
    )
    for (camera, _chunk_index, _file_index, path), slices in ordered_ranges:
        camera_feature = shapes[camera]
        width, height = camera_feature.width, camera_feature.height
        depth = camera_feature.depth
        requested: dict[int, int] = {}
        sampled_pixels: dict[int, list[np.ndarray]] = defaultdict(list)
        for start, stop, episode_index in slices:
            left, right = round(start * fps), round(stop * fps)
            expected = next(item.length for item in episodes if item.index == episode_index)
            if right - left != expected:
                raise ValueError(f"episode {episode_index} decoded video slice length mismatch")
            for local_index in _sample_indices(expected):
                requested[left + local_index] = episode_index
        video_file = files[path]
        iterator = iter(
            iter_depth_codes(video_file.proc_path, depth)
            if depth is not None
            else services.iter_video_rgb_frames(video_file.proc_path)
        )
        decoded_count = 0
        try:
            for decoded_index, frame in enumerate(iterator):
                array = np.asarray(frame)
                if depth is not None:
                    valid = array.dtype == np.uint16 and array.shape == (height, width)
                else:
                    valid = array.dtype == np.uint8 and array.shape == (height, width, 3)
                if not valid:
                    raise ValueError(f"decoded video frame shape or dtype mismatch: {path}")
                if decoded_index in requested:
                    if depth is not None:
                        sampled = _downsample_depth(dequantize_depth(array, depth))
                        sampled_pixels[requested[decoded_index]].append(
                            sampled.reshape(-1, 1)
                        )
                    else:
                        sampled = _downsample_rgb(array).astype(np.float64)
                        sampled_pixels[requested[decoded_index]].append(
                            sampled.reshape(-1, 3)
                        )
                decoded_count += 1
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
            video_file.verify()
        probe_frames = max(round(stop * fps) for _, stop, _ in slices)
        if decoded_count != probe_frames:
            raise ValueError(f"decoded video frame count mismatch: {path}")
        for _, _, episode_index in slices:
            episode_values[episode_index][camera] = np.concatenate(
                sampled_pixels[episode_index], axis=0
            )
    numpy_by_episode = {
        episode.index: {
            camera: _stats(
                episode_values[episode.index][camera],
                profiles[camera],
                len(_sample_indices(episode.length)),
                image=shapes[camera].depth is None,
                depth=shapes[camera].depth is not None,
            )
            for camera in cameras
        }
        for episode in episodes
    }
    numpy_aggregate = {
        camera: _aggregate_stats(
            [numpy_by_episode[episode.index][camera] for episode in episodes],
            profiles[camera],
        )
        for camera in cameras
    }
    aggregate = {
        camera: _stats_to_lists(stats)
        for camera, stats in numpy_aggregate.items()
    }
    by_episode = {
        episode.index: {
            camera: _stats_to_lists(stats)
            for camera, stats in numpy_by_episode[episode.index].items()
        }
        for episode in episodes
    }
    return aggregate, by_episode


def _matrix(
    array: pa.ChunkedArray,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    *,
    sample_images: bool,
    depth: DepthMetadata | None,
) -> tuple[np.ndarray, int]:
    combined = array.combine_chunks()
    if combined.null_count:
        raise ValueError("numeric feature contains null values")
    if pa.types.is_struct(combined.type):
        pixels: list[np.ndarray] = []
        rows = combined.to_pylist()
        indices = _sample_indices(len(rows)) if sample_images else list(range(len(rows)))
        for index in indices:
            value = rows[index]
            try:
                with Image.open(io.BytesIO(value["bytes"])) as image:
                    if depth is not None:
                        decoded = np.asarray(image)
                        if decoded.ndim == 3 and decoded.shape[-1] == 1:
                            decoded = decoded[..., 0]
                        visual = _downsample_depth(decoded.astype(np.float64))
                    else:
                        visual = _downsample_rgb(np.asarray(image.convert("RGB")))
            except Exception as exc:
                raise ValueError("unable to decode embedded image feature") from exc
            pixels.append(visual.reshape(-1, 1 if depth is not None else 3))
        return np.concatenate(pixels, axis=0), len(indices)
    try:
        matrix = np.asarray(combined.to_pylist(), dtype=dtype).reshape(
            (len(combined), *shape)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "numeric feature values do not match their declared shape"
        ) from exc
    return matrix, len(combined)


def _sample_indices(length: int) -> list[int]:
    minimum = min(100, length)
    sample_count = max(minimum, min(int(length**0.75), 10_000))
    return np.round(np.linspace(0, length - 1, sample_count)).astype(int).tolist()


def _downsample_rgb(value: np.ndarray) -> np.ndarray:
    height, width = value.shape[:2]
    if max(width, height) < 300:
        return value
    factor = int(width / 150) if width > height else int(height / 150)
    return value[::factor, ::factor]


def _downsample_depth(value: np.ndarray) -> np.ndarray:
    if value.ndim != 2:
        raise ValueError("depth image must decode as one channel")
    height, width = value.shape
    if max(width, height) < 300:
        return value
    factor = int(width / 150) if width > height else int(height / 150)
    return value[::factor, ::factor]


def _stats(
    values: np.ndarray,
    profile: Sequence[str],
    count: int,
    *,
    image: bool,
    depth: bool,
) -> dict[str, np.ndarray[Any, Any]]:
    if values.ndim < 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("statistics require a nonempty finite array")
    if len(values) < 2:
        mean = np.mean(values, axis=0)
        result = {
            "min": np.min(values, axis=0),
            "max": np.max(values, axis=0),
            "mean": mean,
            "std": np.std(values, axis=0),
            "count": np.array([count]),
        }
        for metric in profile:
            if metric.startswith("q"):
                result[metric] = mean.copy()
    else:
        batch = values.astype(
            np.result_type(values.dtype, np.float32),
            copy=False,
        )
        mean = np.mean(batch, axis=0)
        mean_of_squares = np.mean(batch**2, axis=0)
        result = {
            "min": np.min(batch, axis=0),
            "max": np.max(batch, axis=0),
            "mean": mean,
            "std": np.sqrt(np.maximum(0, mean_of_squares - mean**2)),
            "count": np.array([count]),
        }
        for metric in profile:
            if metric.startswith("q"):
                quantile = int(metric[1:]) / 100.0
                flattened = batch.reshape(len(batch), -1)
                result[metric] = np.array(
                    [
                        _histogram_quantile(flattened[:, column], quantile)
                        for column in range(flattened.shape[1])
                    ]
                ).reshape(batch.shape[1:])
    if image or depth:
        result = {
            metric: value
            if metric == "count"
            else value.reshape(-1, 1, 1) / (1.0 if depth else 255.0)
            for metric, value in result.items()
        }
    return result


def _histogram_quantile(values: np.ndarray, quantile: float) -> float:
    if len(values) < 2:
        return float(values.mean())
    minimum, maximum = values.min(), values.max()
    edges = np.linspace(minimum - 1e-10, maximum + 1e-10, 5001)
    histogram, _ = np.histogram(values, bins=edges)
    cumulative = np.cumsum(histogram)
    target = quantile * len(values)
    index = int(np.searchsorted(cumulative, target))
    if index == 0:
        return float(edges[0])
    if index >= len(cumulative):
        return float(edges[-1])
    count_before = cumulative[index - 1]
    count_in_bin = cumulative[index] - count_before
    if count_in_bin == 0:
        return float(edges[index])
    fraction = (target - count_before) / count_in_bin
    return float(edges[index] + fraction * (edges[index + 1] - edges[index]))


def _aggregate_stats(
    values: Sequence[Mapping[str, np.ndarray[Any, Any]]],
    profile: Sequence[str],
) -> dict[str, np.ndarray[Any, Any]]:
    means = np.stack([item["mean"] for item in values])
    variances = np.stack([item["std"] ** 2 for item in values])
    counts = np.stack([item["count"] for item in values])
    total_count = counts.sum(axis=0)
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)
    total_mean = (means * counts).sum(axis=0) / total_count
    delta_means = means - total_mean
    total_variance = (
        ((variances + delta_means**2) * counts).sum(axis=0) / total_count
    )
    result = {
        "min": np.min(np.stack([item["min"] for item in values]), axis=0),
        "max": np.max(np.stack([item["max"] for item in values]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }
    for metric in profile:
        if metric.startswith("q"):
            quantiles = np.stack([item[metric] for item in values])
            result[metric] = (
                (quantiles * counts).sum(axis=0) / total_count
            )
    return result


def _stats_to_lists(
    stats: Mapping[str, np.ndarray[Any, Any]],
) -> dict[str, Any]:
    return {metric: value.tolist() for metric, value in stats.items()}


def _row_stats(
    row: Mapping[str, Any], feature: str, profile: Sequence[str]
) -> dict[str, Any]:
    return {
        metric: row.get(f"stats/{feature}/{metric}") for metric in profile
    }


def _compare_stats(
    published: Any,
    actual: Mapping[str, list[float]],
    context: str,
    *,
    tolerance: float,
    absolute_tolerance: float = 0.0,
) -> None:
    shape = _stats_value_shape(actual["min"])
    _validate_stats_shape(
        published,
        shape,
        round(actual["count"][0]),
        context,
        tuple(actual),
    )
    assert isinstance(published, dict)
    for metric in actual:
        left = _flatten(published[metric])
        right = _flatten(actual[metric])
        if len(left) != len(right):
            raise ValueError(f"{context} stats shape mismatch")
        differs = any(
            abs(a - b) > max(absolute_tolerance, tolerance * max(1.0, abs(b)))
            for a, b in zip(left, right, strict=True)
        )
        if differs:
            raise ValueError(f"{context} {metric} differs from payload")


def _validate_stats_shape(
    value: Any,
    expected_shape: tuple[int, ...],
    expected_count: int,
    context: str,
    profile: Sequence[str],
) -> None:
    if not isinstance(value, dict) or set(value) != set(profile):
        raise ValueError(f"{context} metric coverage mismatch")
    count = _flatten(value["count"])
    if (
        _stats_value_shape(value["count"]) != (1,)
        or len(count) != 1
        or not math.isclose(count[0], expected_count)
    ):
        raise ValueError(f"{context} count differs from episode/frame count")
    for metric in profile:
        if metric == "count":
            continue
        numbers = _flatten(value[metric])
        if (
            _stats_value_shape(value[metric]) != expected_shape
            or not all(math.isfinite(item) for item in numbers)
        ):
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
    quantiles = [metric for metric in profile if metric.startswith("q")]
    for left, right in zip(quantiles, quantiles[1:], strict=False):
        if any(
            low > high
            for low, high in zip(_flatten(value[left]), _flatten(value[right]), strict=True)
        ):
            raise ValueError(f"{context} quantile ordering is invalid")


def _stats_value_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("statistics must use nonempty declared-dimensional lists")
    child_shapes = [
        _stats_value_shape(item) if isinstance(item, list) else ()
        for item in value
    ]
    if any(shape != child_shapes[0] for shape in child_shapes[1:]):
        raise ValueError("statistics contain ragged dimensions")
    if any(isinstance(item, list) != bool(child_shapes[0]) for item in value):
        raise ValueError("statistics contain mixed dimensions")
    return (len(value), *child_shapes[0])


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
    tree: SecureTree,
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
                path=video.file.proc_path,
                from_timestamp=video.from_timestamp,
                to_timestamp=video.to_timestamp,
                fps=fps,
            )
            requested = [boundary - 1, boundary]
            try:
                samples = services.extract_frames(ref, primary, requested)
            finally:
                video.file.verify()
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
    _validate_task_info_secure(
        tree,
        len(episodes),
        instruction_names,
        [episode.length for episode in episodes],
        template,
        annotation_facts,
        augmentation_language,
    )
    return template, mode, preview


def _validate_task_info_secure(
    tree: SecureTree,
    total_episodes: int,
    instructions: list[str],
    lengths: list[int],
    template: list[Any],
    annotations: list[tuple[int, list[int]]],
    augmentation_language: str | None,
) -> None:
    files = set(tree.files_under("meta/task_info"))
    if files != {"meta/task_info/task_0.json"}:
        raise ValueError("task_info must contain exactly task_0.json")
    value = _read_secure_value(tree, "meta/task_info/task_0.json", "task_0.json")
    if not isinstance(value, list) or len(value) != total_episodes:
        raise ValueError("task_info must contain one entry per episode")
    from .augmentation import valid_augmented_text

    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {
            "episode_id", "task_id", "task_name", "label_info"
        }:
            raise ValueError("task_info episode schema mismatch")
        if (
            _integer(item, "episode_id", minimum=0) != index
            or _integer(item, "task_id", minimum=0) != 0
            or _string(item, "task_name") != instructions[index]
        ):
            raise ValueError("task_info episode/task id or task_name mismatch")
        label = item["label_info"]
        if not isinstance(label, dict) or set(label) != {"action_config"}:
            raise ValueError("task_info label_info schema mismatch")
        actions = label["action_config"]
        start_subtask, boundaries = annotations[index]
        starts, ends = [0, *boundaries], [*boundaries, lengths[index]]
        expected = template[start_subtask : start_subtask + len(starts)]
        if not isinstance(actions, list) or len(actions) != len(starts) or len(expected) != len(actions):
            raise ValueError("task_info action count mismatch")
        for action, start, end, subtask in zip(actions, starts, ends, expected, strict=True):
            if not isinstance(action, dict) or set(action) != {
                "start_frame", "end_frame", "action_text", "skill"
            }:
                raise ValueError("task_info action schema mismatch")
            action_text = _string(action, "action_text")
            text_is_valid = (
                valid_augmented_text(action_text, subtask.text, augmentation_language)
                if augmentation_language is not None
                else action_text == subtask.text
            )
            if (
                _integer(action, "start_frame", minimum=0) != start
                or _integer(action, "end_frame", minimum=1) != end
                or end <= start
                or not text_is_valid
                or _string(action, "skill") != subtask.skill
            ):
                raise ValueError("task_info action differs from annotation/template")


def _stats_profiles(
    published: Mapping[str, Any], expected_features: set[str]
) -> dict[str, tuple[str, ...]]:
    if set(published) != expected_features:
        raise ValueError("aggregate stats feature coverage mismatch")
    result: dict[str, tuple[str, ...]] = {}
    for feature, value in published.items():
        if not isinstance(value, dict):
            raise ValueError(f"stats {feature} must be an object")
        metrics = set(value)
        if not set(_BASIC_STAT_METRICS) <= metrics:
            raise ValueError(f"stats {feature} metric coverage mismatch")
        extras = metrics - set(_BASIC_STAT_METRICS)
        if any(
            len(metric) != 3 or not metric.startswith("q") or not metric[1:].isdigit()
            for metric in extras
        ):
            raise ValueError(f"stats {feature} contains an unsupported metric")
        quantiles = tuple(sorted(extras, key=lambda metric: int(metric[1:])))
        result[feature] = (*_BASIC_STAT_METRICS, *quantiles)
    return result


def _validate_optional_splits(value: Any, total_episodes: int) -> None:
    if value in (None, {}):
        return
    if not isinstance(value, dict) or not value:
        raise ValueError("splits must be an object")
    covered: set[int] = set()
    for name, interval in value.items():
        if not isinstance(name, str) or not name or not isinstance(interval, str) or interval.count(":") != 1:
            raise ValueError("invalid split range")
        left, right = interval.split(":")
        if not left.isdecimal() or not right.isdecimal():
            raise ValueError("invalid split range")
        start, stop = int(left), int(right)
        if not 0 <= start <= stop <= total_episodes:
            raise ValueError("split range is outside episode indices")
        indices = set(range(start, stop))
        if covered & indices:
            raise ValueError("split ranges overlap")
        covered |= indices
    if covered != set(range(total_episodes)):
        raise ValueError("splits do not cover every episode exactly once")


def _payload_files_secure(tree: SecureTree) -> set[str]:
    result: set[str] = set()
    for prefix, suffix in (("data", ".parquet"), ("videos", ".mp4")):
        files = tree.files_under(prefix)
        if not files:
            raise ValueError(f"missing payload directory {prefix}")
        if any(not relative.endswith(suffix) for relative in files):
            raise ValueError("unexpected payload file type")
        result.update(files)
    return result


def _read_bytes(tree: SecureTree, relative: str, limit: int, context: str) -> bytes:
    with tree.open_file(relative, limit, context) as opened:
        return opened.read_bytes()


def _decode_value(value: bytes, context: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = child
        return result

    try:
        return json.loads(
            value.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"nonfinite {item}")),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"malformed {context}") from exc


def _decode_object(value: bytes, context: str) -> dict[str, Any]:
    decoded = _decode_value(value, context)
    if not isinstance(decoded, dict):
        raise ValueError(f"{context} must contain an object")
    return decoded


def _read_secure_value(tree: SecureTree, relative: str, context: str) -> Any:
    return _decode_value(_read_bytes(tree, relative, _MAX_JSON_BYTES, context), context)


def _read_secure_object(tree: SecureTree, relative: str, context: str) -> dict[str, Any]:
    value = _read_secure_value(tree, relative, context)
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain an object")
    return value


def _secure_digest(tree: SecureTree, relative: str, context: str) -> str:
    with tree.open_file(relative, _MAX_VIDEO_BYTES, context) as opened:
        return opened.sha256()


def _official_core_files(tree: SecureTree) -> set[str]:
    result = set()
    for relative in tree.scan():
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

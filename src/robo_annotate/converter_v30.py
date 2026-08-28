"""Publication helpers for full and accepted-only LeRobot v3.0 releases."""

from __future__ import annotations

import json
import os
import secrets
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .lerobot import DatasetIndex
from .publication_metadata import (
    SelectedEpisode,
    atomic_json,
    write_public_annotations,
)
from .secure_tree import SecureFile, SecureTree
from .stats import iter_video_rgb_frames
from .v30_data_writer import (
    DataPlacement,
    V30DataWriteResult,
    _aggregate_feature_stats,
    _downsample_rgb,
    _feature_stats,
    _sample_indices,
    _stats_to_lists,
    write_v30_data_subset,
)
from .v30_depth import (
    DepthMetadata,
    dequantize_depth,
    depth_metadata,
    is_depth_feature,
    iter_depth_codes,
)
from .v30_video_writer import (
    V30VideoWriteResult,
    VideoPlacement,
    write_v30_video_subset,
)
from .writer_publication import _CleanupFailures
from .workspace import EpisodeRecord, RunManifest


_BASIC_STAT_METRICS = ("min", "max", "mean", "std", "count")
_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_VIDEO_PATH = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)
_EPISODES_PATH = (
    "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
)
_MAX_VIDEO_BYTES = 16 * 1024 * 1024 * 1024
_MAX_JSON_BYTES = 16 * 1024 * 1024


def write_full_v30_release(
    staging: Path,
    output: Path,
    manifest: RunManifest,
    records: Sequence[EpisodeRecord],
    converted_at: datetime,
    augmented_texts: Mapping[int, list[str]] | None,
) -> None:
    """Append only Robo metadata to an unchanged official v3.0 tree copy."""
    if manifest.dataset_version != "v3.0":
        raise ValueError("full v3 release writer requires a LeRobot v3.0 manifest")
    ordered = sorted(records, key=lambda record: record.episode_index)
    if [record.episode_index for record in ordered] != list(
        range(manifest.total_episodes)
    ):
        raise ValueError("full v3 release records must cover contiguous source episodes")
    selected = [
        SelectedEpisode(
            record=record,
            source_index=record.episode_index,
            output_index=record.episode_index,
            length=manifest.episode_lengths[record.episode_index],
        )
        for record in ordered
    ]
    write_public_annotations(
        staging,
        output,
        manifest,
        selected,
        converted_at,
        augmented_texts,
        extend_info=False,
    )


def rewrite_accepted_v30_release(
    staging: Path,
    source: Path,
    output: Path,
    manifest: RunManifest,
    dataset: DatasetIndex,
    records: Sequence[EpisodeRecord],
    converted_at: datetime,
    augmented_texts: Mapping[int, list[str]] | None,
    services: Any,
) -> int:
    """Rebuild every official reference for a selected v3.0 episode subset."""
    if manifest.dataset_version != "v3.0" or dataset.version != "v3.0":
        raise ValueError("accepted v3 release writer requires LeRobot v3.0 inputs")
    selected_records = sorted(records, key=lambda item: item.episode_index)
    source_indices = [record.episode_index for record in selected_records]
    if not source_indices or len(source_indices) != len(set(source_indices)):
        raise ValueError("accepted v3 release requires unique selected episodes")

    with SecureTree(source, "v3 composer source") as source_tree:
        source_tree.scan()
        source_info = _read_source_json_object(
            source_tree,
            "meta/info.json",
            "source info.json",
        )
        source_stats = _read_source_json_object(
            source_tree,
            "meta/stats.json",
            "source stats.json",
        )
        result = _compose_accepted_v30_release(
            staging,
            source,
            output,
            manifest,
            dataset,
            selected_records,
            source_indices,
            converted_at,
            augmented_texts,
            services,
            source_info,
            source_stats,
        )
        try:
            source_tree.verify()
        except (OSError, ValueError) as exc:
            raise ValueError(
                "source dataset changed during v3 composition"
            ) from exc
        return result


def _compose_accepted_v30_release(
    staging: Path,
    source: Path,
    output: Path,
    manifest: RunManifest,
    dataset: DatasetIndex,
    selected_records: Sequence[EpisodeRecord],
    source_indices: Sequence[int],
    converted_at: datetime,
    augmented_texts: Mapping[int, list[str]] | None,
    services: Any,
    source_info: Mapping[str, Any],
    source_stats: Mapping[str, Any],
) -> int:
    data_size_limit = _official_size_limit(
        source_info,
        "data_files_size_in_mb",
        100,
    )
    video_size_limit = _official_size_limit(
        source_info,
        "video_files_size_in_mb",
        200,
    )
    writer_info = source_info | {
        "data_path": _DATA_PATH,
        "video_path": _VIDEO_PATH,
        "data_files_size_in_mb": data_size_limit,
        "video_files_size_in_mb": video_size_limit,
    }
    data = write_v30_data_subset(
        source,
        staging,
        dataset,
        source_indices,
        writer_info,
    )
    videos = write_v30_video_subset(
        staging,
        dataset,
        source_indices,
        writer_info,
    )
    placements = _paired_placements(data, videos, manifest.camera_keys)
    profiles = _stats_profiles(
        source_stats,
        set(data.aggregate_stats) | set(manifest.camera_keys),
    )
    aggregate_video, episode_video = _video_stats(
        staging,
        videos,
        placements,
        writer_info,
        profiles,
        services,
    )

    aggregate_stats = data.aggregate_stats | aggregate_video
    episode_rows = _episode_rows(
        placements,
        videos,
        data.episode_stats,
        episode_video,
        profiles,
        manifest.camera_keys,
    )
    episodes_dir = _create_directory(staging / "meta", "episodes")
    chunk_dir = _create_directory(episodes_dir, "chunk-000")
    _atomic_parquet(chunk_dir / "file-000.parquet", pa.Table.from_pylist(episode_rows))
    atomic_json(staging / "meta/stats.json", aggregate_stats, sort_keys=False)

    info = dict(source_info)
    info.update(
        {
            "codebase_version": "v3.0",
            "total_episodes": len(placements),
            "total_frames": data.total_frames,
            "total_tasks": data.task_table.num_rows,
            "splits": {"train": f"0:{len(placements)}"},
            "data_path": _DATA_PATH,
            "video_path": _VIDEO_PATH,
            "episodes_path": _EPISODES_PATH,
            "data_files_size_in_mb": data_size_limit,
            "video_files_size_in_mb": video_size_limit,
        }
    )
    atomic_json(staging / "meta/info.json", info, sort_keys=False)
    selected = [
        SelectedEpisode(
            record=record,
            source_index=placement.source_index,
            output_index=placement.output_index,
            length=placement.length,
        )
        for record, placement in zip(
            selected_records,
            placements,
            strict=True,
        )
    ]
    write_public_annotations(
        staging,
        output,
        manifest,
        selected,
        converted_at,
        augmented_texts,
        extend_info=False,
    )
    return data.total_frames


def _paired_placements(
    data: V30DataWriteResult,
    videos: V30VideoWriteResult,
    cameras: Sequence[str],
) -> tuple[DataPlacement, ...]:
    placements = tuple(data.placements)
    expected_outputs = list(range(len(placements)))
    if [item.output_index for item in placements] != expected_outputs:
        raise ValueError("v3 data placements must use contiguous output indices")
    if sum(item.length for item in placements) != data.total_frames:
        raise ValueError("v3 data placements differ from the rebuilt frame count")
    if set(videos.placements) != set(cameras) or set(videos.files_by_camera) != set(
        cameras
    ):
        raise ValueError("v3 video writer camera coverage is incomplete")
    for camera in cameras:
        camera_placements = videos.placements[camera]
        if len(camera_placements) != len(placements):
            raise ValueError("v3 data and video placement counts differ")
        for data_item, video_item in zip(
            placements,
            camera_placements,
            strict=True,
        ):
            if (
                video_item.camera_key != camera
                or video_item.source_index != data_item.source_index
                or video_item.output_index != data_item.output_index
            ):
                raise ValueError("v3 data and video placements do not match")
    return placements


def _read_source_json_object(
    tree: SecureTree,
    relative: str,
    context: str,
) -> dict[str, Any]:
    with tree.open_file(relative, _MAX_JSON_BYTES, context) as opened:
        payload = opened.read_bytes()

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate key {name!r} in {context}")
            result[name] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(constant)
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {context}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _stats_profiles(
    published: Mapping[str, Any],
    expected_features: set[str],
) -> dict[str, tuple[str, ...]]:
    if set(published) != expected_features:
        raise ValueError("source stats feature coverage differs from rebuilt output")
    result = {}
    for feature, value in published.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"source stats profile is invalid for {feature}")
        metrics = set(value)
        if not set(_BASIC_STAT_METRICS) <= metrics:
            raise ValueError(f"source stats profile is incomplete for {feature}")
        extras = metrics - set(_BASIC_STAT_METRICS)
        if any(
            len(metric) != 3
            or not metric.startswith("q")
            or not metric[1:].isdigit()
            for metric in extras
        ):
            raise ValueError(f"source stats profile is invalid for {feature}")
        result[feature] = (
            *_BASIC_STAT_METRICS,
            *sorted(extras, key=lambda metric: int(metric[1:])),
        )
    return result


def _video_stats(
    staging: Path,
    videos: V30VideoWriteResult,
    data_placements: Sequence[DataPlacement],
    info: Mapping[str, Any],
    profiles: Mapping[str, tuple[str, ...]],
    services: Any,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    fps = info.get("fps")
    features = info.get("features")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or fps <= 0
        or not isinstance(features, Mapping)
    ):
        raise ValueError("v3 video statistics metadata is invalid")
    rgb_frame_iterator = _service(
        services,
        "iter_video_rgb_frames",
        iter_video_rgb_frames,
    )
    depth_frame_iterator = _service(
        services,
        "iter_video_depth_frames",
        iter_depth_codes,
    )
    if not callable(rgb_frame_iterator):
        raise TypeError("iter_video_rgb_frames service must be callable")
    if not callable(depth_frame_iterator):
        raise TypeError("iter_video_depth_frames service must be callable")
    lengths = {item.output_index: item.length for item in data_placements}
    aggregate: dict[str, Any] = {}
    by_episode: dict[int, dict[str, Any]] = {
        item.output_index: {} for item in data_placements
    }
    with SecureTree(staging, "rebuilt staging") as tree:
        tree.scan()
        for camera, files in videos.files_by_camera.items():
            declaration = features.get(camera)
            if not isinstance(declaration, Mapping):
                raise ValueError(f"video feature declaration is invalid: {camera}")
            details = declaration.get("info")
            if not isinstance(details, Mapping):
                raise ValueError(f"video feature declaration is invalid: {camera}")
            depth = depth_metadata(declaration, camera) if is_depth_feature(declaration) else None
            frame_iterator = depth_frame_iterator if depth is not None else rgb_frame_iterator
            width, height = details.get("video.width"), details.get("video.height")
            if type(width) is not int or type(height) is not int:
                raise ValueError(f"video feature dimensions are invalid: {camera}")
            placements_by_file: dict[tuple[int, int], list[VideoPlacement]] = {}
            for placement in videos.placements[camera]:
                placements_by_file.setdefault(
                    (placement.chunk_index, placement.file_index), []
                ).append(placement)
            ordered_pairs = sorted(placements_by_file)
            if len(ordered_pairs) != len(files):
                raise ValueError("v3 video file inventory differs from placements")
            numpy_episode: dict[int, dict[str, np.ndarray[Any, Any]]] = {}
            for pair, path in zip(ordered_pairs, files, strict=True):
                expected_path = _staging_video_path(
                    staging,
                    camera,
                    pair[0],
                    pair[1],
                )
                if path != expected_path:
                    raise ValueError("v3 video file path differs from its placement")
                relative = path.relative_to(staging).as_posix()
                with tree.open_file(
                    relative,
                    _MAX_VIDEO_BYTES,
                    "rebuilt video",
                ) as opened:
                    samples = _decode_video_samples(
                        opened,
                        placements_by_file[pair],
                        lengths,
                        float(fps),
                        width,
                        height,
                        frame_iterator,
                        depth,
                    )
                for output_index, values in samples.items():
                    numpy_episode[output_index] = _feature_stats(
                        values,
                        profiles[camera],
                        len(_sample_indices(lengths[output_index])),
                        image=False,
                    )
            if set(numpy_episode) != set(lengths):
                raise ValueError("v3 video statistics episode coverage is incomplete")
            aggregate_numpy = _aggregate_feature_stats(
                [numpy_episode[index] for index in range(len(lengths))],
                profiles[camera],
            )
            aggregate[camera] = _image_stats_shape(
                _stats_to_lists(aggregate_numpy)
            )
            for output_index, stats in numpy_episode.items():
                by_episode[output_index][camera] = _image_stats_shape(
                    _stats_to_lists(stats)
                )
        tree.verify()
    return aggregate, by_episode


def _staging_video_path(
    staging: Path,
    camera: str,
    chunk_index: int,
    file_index: int,
) -> Path:
    return staging / _VIDEO_PATH.format(
        video_key=camera,
        chunk_index=chunk_index,
        file_index=file_index,
    )


def _decode_video_samples(
    opened: SecureFile,
    placements: Sequence[VideoPlacement],
    lengths: Mapping[int, int],
    fps: float,
    width: int,
    height: int,
    frame_iterator: Any,
    depth: DepthMetadata | None,
) -> dict[int, np.ndarray[Any, Any]]:
    opened.verify()
    requested: dict[int, int] = {}
    pixels: dict[int, list[np.ndarray[Any, Any]]] = {
        item.output_index: [] for item in placements
    }
    expected_start = 0
    for placement in sorted(placements, key=lambda item: item.from_timestamp):
        start = round(placement.from_timestamp * fps)
        stop = round(placement.to_timestamp * fps)
        length = lengths[placement.output_index]
        if start != expected_start or stop - start != length:
            raise ValueError("v3 video placement ranges are not exact and contiguous")
        for local_index in _sample_indices(length):
            requested[start + local_index] = placement.output_index
        expected_start = stop
    iterator = iter(
        iter_depth_codes(opened.proc_path, depth)
        if depth is not None and frame_iterator is iter_depth_codes
        else frame_iterator(opened.proc_path)
    )
    decoded_count = 0
    primary: BaseException | None = None
    try:
        for frame_index, frame in enumerate(iterator):
            array = np.asarray(frame)
            if depth is not None:
                if array.dtype != np.uint16 or array.shape != (height, width):
                    raise ValueError("rebuilt depth frame shape or dtype is invalid")
            elif array.dtype != np.uint8 or array.shape != (height, width, 3):
                raise ValueError("rebuilt video frame shape or dtype is invalid")
            output_index = requested.get(frame_index)
            if output_index is not None:
                if depth is not None:
                    sampled = _downsample_depth(dequantize_depth(array, depth))
                    pixels[output_index].append(sampled.reshape(-1, 1))
                else:
                    sampled = _downsample_rgb(array).astype(np.float64) / 255.0
                    pixels[output_index].append(sampled.reshape(-1, 3))
            decoded_count += 1
    except BaseException as exc:
        primary = exc
        raise
    finally:
        failures = _CleanupFailures()
        close = getattr(iterator, "close", None)
        if callable(close):
            failures.attempt("video decoder iterator close", close)
        failures.attempt("rebuilt video verification", opened.verify)
        failures.finish(primary, "rebuilt video statistics")
    if decoded_count != expected_start:
        raise ValueError("rebuilt video decoded frame count differs from placements")
    result = {}
    for output_index, values in pixels.items():
        if len(values) != len(_sample_indices(lengths[output_index])):
            raise ValueError("rebuilt video sampling coverage is incomplete")
        result[output_index] = np.concatenate(values, axis=0)
    return result


def _downsample_depth(value: np.ndarray) -> np.ndarray:
    if value.ndim != 2:
        raise ValueError("depth frame must decode as one channel")
    height, width = value.shape
    if max(width, height) < 300:
        return value
    factor = int(width / 150) if width > height else int(height / 150)
    return value[::factor, ::factor]


def _image_stats_shape(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        metric: value
        if metric == "count"
        else [[[channel]] for channel in value]
        for metric, value in stats.items()
    }


def _episode_rows(
    data_placements: Sequence[DataPlacement],
    videos: V30VideoWriteResult,
    numeric_stats: Mapping[int, Mapping[str, Any]],
    video_stats: Mapping[int, Mapping[str, Any]],
    profiles: Mapping[str, tuple[str, ...]],
    cameras: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for placement in data_placements:
        output_index = placement.output_index
        row: dict[str, Any] = {
            "episode_index": output_index,
            "tasks": list(placement.tasks),
            "length": placement.length,
            "data/chunk_index": placement.chunk_index,
            "data/file_index": placement.file_index,
            "dataset_from_index": placement.dataset_from_index,
            "dataset_to_index": placement.dataset_to_index,
        }
        for camera in cameras:
            video = next(
                item
                for item in videos.placements[camera]
                if item.output_index == output_index
            )
            row.update(
                {
                    f"videos/{camera}/chunk_index": video.chunk_index,
                    f"videos/{camera}/file_index": video.file_index,
                    f"videos/{camera}/from_timestamp": video.from_timestamp,
                    f"videos/{camera}/to_timestamp": video.to_timestamp,
                }
            )
        feature_stats = dict(numeric_stats[output_index]) | dict(
            video_stats[output_index]
        )
        for feature, profile in profiles.items():
            for metric in profile:
                row[f"stats/{feature}/{metric}"] = feature_stats[feature][metric]
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        rows.append(row)
    return rows


def _create_directory(parent: Path, name: str) -> Path:
    if not name or "/" in name or name in (".", ".."):
        raise ValueError("unsafe staged metadata directory name")
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("staged metadata parent is unsafe")
    path = parent / name
    os.mkdir(path, 0o700)
    created = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISDIR(created.st_mode):
        raise ValueError("staged metadata directory is unsafe")
    return path


def _atomic_parquet(path: Path, table: pa.Table) -> None:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    payload = sink.getvalue().to_pybytes()
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    name = f".{path.name}.{secrets.token_hex(12)}.tmp"
    descriptor = -1
    primary: BaseException | None = None
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        write_primary: BaseException | None = None
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException as exc:
            write_primary = exc
        handle_failures = _CleanupFailures()
        handle_failures.attempt("temporary episode parquet close", handle.close)
        handle_failures.finish(write_primary, "temporary episode parquet")
        if write_primary is not None:
            raise write_primary
        os.replace(
            name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        name = ""
        os.fsync(parent_fd)
    except BaseException as exc:
        primary = exc
    finally:
        failures = _CleanupFailures()
        if descriptor >= 0:
            failures.attempt(
                "temporary episode parquet descriptor close",
                lambda: os.close(descriptor),
            )
        if name:
            failures.attempt(
                "temporary episode parquet removal",
                lambda: _unlink_if_present(parent_fd, name),
            )
        failures.attempt(
            "episode metadata parent descriptor close",
            lambda: os.close(parent_fd),
        )
        failures.finish(primary, "episode metadata publication")
    if primary is not None:
        raise primary


def _unlink_if_present(parent_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=parent_fd)
    except FileNotFoundError:
        pass


def _official_size_limit(
    info: Mapping[str, Any],
    field: str,
    default: int,
) -> int:
    value = info.get(field, default)
    if type(value) is not int or value <= 0:
        raise ValueError(f"source {field} must be a positive integer")
    return value


def _service(services: Any, name: str, default: Any) -> Any:
    if services is None:
        return default
    if isinstance(services, dict):
        return services.get(name, default)
    return getattr(services, name, default)

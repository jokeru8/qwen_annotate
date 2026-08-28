"""Repack selected LeRobot v3.0 data episodes into fresh Parquet shards."""

from __future__ import annotations

import errno
import io
import json
import math
import os
import secrets
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .lerobot import DatasetIndex, EpisodeInfo
from .secure_tree import SecureTree, rename_noreplace_at


_MAX_PARQUET_BYTES = 1024 * 1024 * 1024
_MAX_JSON_BYTES = 16 * 1024 * 1024
_OFFICIAL_INDEX_COLUMNS = ("episode_index", "frame_index", "index", "task_index")
_BASIC_STAT_METRICS = ("min", "max", "mean", "std", "count")
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_OUTPUT_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
)
_PRIVATE_DIRECTORY_PREFIX = ".v30-dir-"
StatValue = list[int | float] | list[list[list[float]]]
FeatureStats = dict[str, StatValue]


@dataclass(frozen=True)
class DataPlacement:
    """Describe one output episode's location in the rebuilt data shards."""

    source_index: int
    output_index: int
    length: int
    chunk_index: int
    file_index: int
    dataset_from_index: int
    dataset_to_index: int
    tasks: tuple[str, ...]


@dataclass(frozen=True)
class V30DataWriteResult:
    """Return rebuilt data files, placements, task metadata, and numeric stats."""

    placements: tuple[DataPlacement, ...]
    task_table: pa.Table
    parquet_files: tuple[Path, ...]
    total_frames: int
    aggregate_stats: dict[str, FeatureStats]
    episode_stats: dict[int, dict[str, FeatureStats]]


def write_v30_data_subset(
    source: Path,
    staging: Path,
    dataset: DatasetIndex,
    source_indices: Sequence[int],
    info: Mapping[str, Any],
) -> V30DataWriteResult:
    """Write selected v3 episodes in caller order without mutating the source."""
    source, staging, selected, chunks_size, size_limit = _validate_request(
        source, staging, dataset, source_indices, info
    )
    with SecureTree(source, "source") as tree:
        tree.scan()
        tasks = _read_parquet(tree, "meta/tasks.parquet", "tasks metadata")
        task_rows = _validate_tasks(tasks)
        stats_profiles = _read_stats_profiles(tree, info)
        task_by_text = {text: index for index, text in task_rows.items()}
        tables: dict[str, pa.Table] = {}
        source_schema: pa.Schema | None = None
        rewritten: list[tuple[EpisodeInfo, pa.Table, tuple[str, ...]]] = []
        task_order: list[int] = []
        seen_tasks: set[int] = set()
        output_offsets: list[int] = []
        global_offset = 0

        for episode in selected:
            relative = _source_relative(tree.path, episode.data.path, "data shard")
            table = tables.get(relative)
            if table is None:
                table = _read_parquet(tree, relative, "data shard")
                tables[relative] = table
            if source_schema is None:
                source_schema = table.schema
            elif not table.schema.equals(source_schema, check_metadata=True):
                raise ValueError("selected data shards must have one exact Arrow schema")
            sliced, source_tasks = _episode_slice(table, episode, task_by_text)
            for task_index in source_tasks:
                if task_index not in seen_tasks:
                    task_order.append(task_index)
                    seen_tasks.add(task_index)
            rewritten.append(
                (
                    episode,
                    sliced,
                    tuple(task_rows[index] for index in source_tasks),
                )
            )
            output_offsets.append(global_offset)
            global_offset += episode.length

        task_remap = {
            source_index: output_index
            for output_index, source_index in enumerate(task_order)
        }
        output_tables: list[pa.Table] = []
        for output_index, (episode, table, _) in enumerate(rewritten):
            output_from = output_offsets[output_index]
            replacements = {
                "episode_index": [output_index] * episode.length,
                "frame_index": list(range(episode.length)),
                "index": list(range(output_from, output_from + episode.length)),
                "task_index": [
                    task_remap[value] for value in table["task_index"].to_pylist()
                ],
            }
            output_tables.append(_replace_indices(table, replacements))

        packed, file_numbers = _pack_tables(output_tables, size_limit)
        aggregate_stats, episode_stats = _numeric_stats(
            output_tables,
            info,
            stats_profiles,
        )
        placements = tuple(
            DataPlacement(
                source_index=episode.episode_index,
                output_index=output_index,
                length=episode.length,
                chunk_index=file_numbers[output_index] // chunks_size,
                file_index=file_numbers[output_index] % chunks_size,
                dataset_from_index=output_offsets[output_index],
                dataset_to_index=output_offsets[output_index] + episode.length,
                tasks=rewritten[output_index][2],
            )
            for output_index, episode in enumerate(selected)
        )
        task_table = _compact_task_table(tasks, task_order)
        tree.verify()
        parquet_files = _publish(
            staging,
            packed,
            task_table,
            chunks_size,
            source_tree=tree,
        )

    return V30DataWriteResult(
        placements=placements,
        task_table=task_table,
        parquet_files=parquet_files,
        total_frames=global_offset,
        aggregate_stats=aggregate_stats,
        episode_stats=episode_stats,
    )


def _validate_request(
    source: Path,
    staging: Path,
    dataset: DatasetIndex,
    source_indices: Sequence[int],
    info: Mapping[str, Any],
) -> tuple[Path, Path, list[EpisodeInfo], int, float]:
    if not isinstance(source, Path) or not isinstance(staging, Path):
        raise TypeError("source and staging must be Path values")
    if not isinstance(dataset, DatasetIndex) or dataset.version != "v3.0":
        raise ValueError("data subset writer requires a LeRobot v3.0 dataset index")
    if not isinstance(info, Mapping) or info.get("codebase_version") != "v3.0":
        raise ValueError("info must declare LeRobot codebase_version v3.0")
    if isinstance(source_indices, (str, bytes)) or not isinstance(source_indices, Sequence):
        raise TypeError("source_indices must be a sequence of integers")
    indices = list(source_indices)
    if not indices:
        raise ValueError("source_indices must select at least one episode")
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("source_indices must contain nonnegative integers")
    if len(indices) != len(set(indices)):
        raise ValueError("source_indices must be unique")

    absolute_source = Path(os.path.abspath(source))
    absolute_staging = Path(os.path.abspath(staging))
    indexed_root = Path(os.path.abspath(dataset.root))
    if absolute_source != indexed_root:
        raise ValueError("source does not match the inspected dataset root")
    if (
        absolute_staging == absolute_source
        or absolute_source in absolute_staging.parents
        or absolute_staging in absolute_source.parents
    ):
        raise ValueError("staging and source trees must not overlap")

    by_index: dict[int, EpisodeInfo] = {}
    for episode in dataset.episodes:
        if episode.episode_index in by_index:
            raise ValueError("dataset episode indices must be unique")
        by_index[episode.episode_index] = episode
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise ValueError(f"source_indices contains unknown episode indices: {missing}")

    chunks_size = info.get("chunks_size", 1000)
    if type(chunks_size) is not int or chunks_size <= 0:
        raise ValueError("info.chunks_size must be a positive integer")
    size_mb = info.get("data_files_size_in_mb", 100)
    if (
        isinstance(size_mb, bool)
        or not isinstance(size_mb, (int, float))
        or not math.isfinite(float(size_mb))
        or size_mb <= 0
    ):
        raise ValueError("info.data_files_size_in_mb must be positive and finite")
    return (
        absolute_source,
        absolute_staging,
        [by_index[index] for index in indices],
        chunks_size,
        float(size_mb) * 1024 * 1024,
    )


def _source_relative(root: Path, path: Path, context: str) -> str:
    if not isinstance(path, Path):
        raise TypeError(f"{context} path must be a Path")
    absolute = Path(os.path.abspath(path))
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{context} path escapes the source tree") from exc
    if not relative.parts:
        raise ValueError(f"{context} path must name a file")
    return relative.as_posix()


def _read_parquet(tree: SecureTree, relative: str, context: str) -> pa.Table:
    opened = tree.open_file(relative, _MAX_PARQUET_BYTES, context)
    payload = opened.read_bytes()
    try:
        return pq.read_table(pa.BufferReader(payload))
    except Exception as exc:
        raise ValueError(f"unable to read {context}: {relative}") from exc


def _read_stats_profiles(
    tree: SecureTree,
    info: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    opened = tree.open_file("meta/stats.json", _MAX_JSON_BYTES, "aggregate stats")
    payload = opened.read_bytes()

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"aggregate stats contains duplicate key {name!r}")
            result[name] = value
        return result

    try:
        published = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("unable to read aggregate stats metadata") from exc
    if not isinstance(published, dict):
        raise ValueError("aggregate stats metadata must be an object")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("info.features must be an object")
    result: dict[str, tuple[str, ...]] = {}
    for feature, declaration in features.items():
        if not isinstance(feature, str) or not isinstance(declaration, Mapping):
            raise ValueError("info.features must contain named objects")
        if declaration.get("dtype") in {"string", "language", "video"}:
            continue
        value = published.get(feature)
        if not isinstance(value, Mapping):
            raise ValueError(f"aggregate stats is missing numeric feature {feature}")
        metrics = set(value)
        if not set(_BASIC_STAT_METRICS) <= metrics:
            raise ValueError(f"aggregate stats profile is incomplete for {feature}")
        extras = metrics - set(_BASIC_STAT_METRICS)
        if any(
            len(metric) != 3
            or not metric.startswith("q")
            or not metric[1:].isdigit()
            for metric in extras
        ):
            raise ValueError(f"aggregate stats profile is invalid for {feature}")
        result[feature] = (
            *_BASIC_STAT_METRICS,
            *sorted(extras, key=lambda metric: int(metric[1:])),
        )
    return result


def _validate_tasks(table: pa.Table) -> dict[int, str]:
    if table.column_names != ["task_index", "task"]:
        raise ValueError("tasks metadata must contain ordered task_index and task columns")
    if table["task_index"].null_count or table["task"].null_count:
        raise ValueError("tasks metadata must not contain null values")
    result: dict[int, str] = {}
    texts: set[str] = set()
    for row in table.to_pylist():
        index, text = row["task_index"], row["task"]
        if type(index) is not int or index < 0 or not isinstance(text, str) or not text:
            raise ValueError("tasks metadata contains an invalid task")
        if index in result or text in texts:
            raise ValueError("tasks metadata contains duplicate task facts")
        result[index] = text
        texts.add(text)
    if set(result) != set(range(len(result))):
        raise ValueError("task indices must be contiguous from zero")
    return result


def _episode_slice(
    table: pa.Table,
    episode: EpisodeInfo,
    task_by_text: Mapping[str, int],
) -> tuple[pa.Table, tuple[int, ...]]:
    for name in _OFFICIAL_INDEX_COLUMNS:
        if name not in table.column_names:
            raise ValueError(f"data shard is missing official index column {name}")
        values = table[name].to_pylist()
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(f"data shard contains an invalid source {name}")
    if episode.data.dataset_to_index - episode.data.dataset_from_index != episode.length:
        raise ValueError(f"episode {episode.episode_index} data range does not match its length")
    expected = list(range(episode.data.dataset_from_index, episode.data.dataset_to_index))
    values = table["index"].to_pylist()
    matches = [
        offset
        for offset in range(max(0, table.num_rows - episode.length + 1))
        if values[offset : offset + episode.length] == expected
    ]
    if len(matches) != 1:
        raise ValueError(
            f"episode {episode.episode_index} data shard must contain one exact global index range"
        )
    sliced = table.slice(matches[0], episode.length)
    if sliced["episode_index"].to_pylist() != [episode.episode_index] * episode.length:
        raise ValueError(f"episode {episode.episode_index} source episode_index mismatch")
    if sliced["frame_index"].to_pylist() != list(range(episode.length)):
        raise ValueError(f"episode {episode.episode_index} source frame_index mismatch")
    expected_task = task_by_text.get(episode.task)
    task_indices = sliced["task_index"].to_pylist()
    if expected_task is None or task_indices != [expected_task] * episode.length:
        raise ValueError(f"episode {episode.episode_index} source task facts mismatch")
    return sliced, (expected_task,)


def _replace_indices(table: pa.Table, replacements: Mapping[str, list[int]]) -> pa.Table:
    result = table
    for name in _OFFICIAL_INDEX_COLUMNS:
        position = result.schema.get_field_index(name)
        field = result.schema.field(position)
        array = pa.array(replacements[name], type=field.type)
        result = result.set_column(position, field, array)
    if not result.schema.equals(table.schema, check_metadata=True):
        raise ValueError("rewriting official indices changed the Arrow schema")
    return result


def _pack_tables(
    tables: Sequence[pa.Table],
    size_limit: float,
) -> tuple[list[pa.Table], list[int]]:
    packed: list[pa.Table] = []
    file_numbers: list[int] = []
    current: list[pa.Table] = []
    current_weight = 0
    file_number = 0
    for table in tables:
        if current and current_weight + table.nbytes > size_limit:
            packed.append(pa.concat_tables(current))
            current = []
            current_weight = 0
            file_number += 1
        current.append(table)
        current_weight += table.nbytes
        file_numbers.append(file_number)
    if current:
        packed.append(pa.concat_tables(current))
    return packed, file_numbers


def _compact_task_table(table: pa.Table, task_order: Sequence[int]) -> pa.Table:
    row_by_index = {
        task_index: row_number
        for row_number, task_index in enumerate(table["task_index"].to_pylist())
    }
    compact = table.take(
        pa.array([row_by_index[index] for index in task_order], type=pa.int64())
    )
    position = compact.schema.get_field_index("task_index")
    field = compact.schema.field(position)
    compact = compact.set_column(
        position,
        field,
        pa.array(range(len(task_order)), type=field.type),
    )
    if not compact.schema.equals(table.schema, check_metadata=True):
        raise ValueError("compacting tasks changed the Arrow schema")
    return compact


def _numeric_stats(
    tables: Sequence[pa.Table],
    info: Mapping[str, Any],
    profiles: Mapping[str, tuple[str, ...]],
) -> tuple[
    dict[str, FeatureStats],
    dict[int, dict[str, FeatureStats]],
]:
    if not tables:
        raise ValueError("numeric statistics require selected episode tables")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("info.features must be an object")
    specs: dict[str, tuple[np.dtype[Any], int, bool]] = {}
    columns = set(tables[0].column_names)
    for name, profile in profiles.items():
        declaration = features.get(name)
        if not isinstance(declaration, Mapping) or name not in columns:
            raise ValueError(f"numeric feature {name} is absent from data shards")
        dtype = declaration.get("dtype")
        shape = declaration.get("shape")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not shape or any(
            type(dimension) is not int or dimension <= 0 for dimension in shape
        ):
            raise ValueError(f"numeric feature declaration is invalid: {name}")
        if dtype == "image":
            if len(shape) != 3 or (shape[0] != 3 and shape[-1] != 3):
                raise ValueError(f"embedded image feature must declare an RGB shape: {name}")
            specs[name] = (np.dtype(np.uint8), 3, True)
            continue
        try:
            numpy_dtype = np.dtype(dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported numeric feature dtype: {name}") from exc
        if numpy_dtype.kind not in "biuf":
            raise ValueError(f"unsupported numeric feature dtype: {name}")
        specs[name] = (numpy_dtype, shape[-1], False)
        if not profile:
            raise ValueError(f"numeric stats profile is empty for {name}")

    numpy_episode_stats: dict[int, dict[str, dict[str, np.ndarray[Any, Any]]]] = {}
    for episode_index, table in enumerate(tables):
        numpy_episode_stats[episode_index] = {}
        for name, (dtype, width, is_image) in specs.items():
            matrix, count = _numeric_matrix(
                table[name],
                dtype,
                width,
                sample_images=is_image,
            )
            numpy_episode_stats[episode_index][name] = _feature_stats(
                matrix,
                profiles[name],
                count,
                image=is_image,
            )

    numpy_aggregate = {
        name: _aggregate_feature_stats(
            [numpy_episode_stats[index][name] for index in range(len(tables))],
            profiles[name],
        )
        for name in specs
    }
    aggregate = {
        name: _stats_to_lists(stats) for name, stats in numpy_aggregate.items()
    }
    episode_stats = {
        episode_index: {
            name: _stats_to_lists(stats) for name, stats in features.items()
        }
        for episode_index, features in numpy_episode_stats.items()
    }
    return aggregate, episode_stats


def _numeric_matrix(
    array: pa.ChunkedArray,
    dtype: np.dtype[Any],
    width: int,
    *,
    sample_images: bool,
) -> tuple[np.ndarray, int]:
    combined = array.combine_chunks()
    if combined.null_count:
        raise ValueError("numeric feature contains null values")
    if sample_images:
        pixels: list[np.ndarray] = []
        rows = combined.to_pylist()
        indices = _sample_indices(len(rows))
        for index in indices:
            value = rows[index]
            if not isinstance(value, dict) or not isinstance(value.get("bytes"), bytes):
                raise ValueError("embedded image feature contains an invalid value")
            try:
                with Image.open(io.BytesIO(value["bytes"])) as image:
                    rgb = _downsample_rgb(np.asarray(image.convert("RGB")))
            except Exception as exc:
                raise ValueError("unable to decode embedded image feature") from exc
            pixels.append(rgb.reshape(-1, 3))
        return np.concatenate(pixels, axis=0), len(indices)
    try:
        matrix = np.asarray(combined.to_pylist(), dtype=dtype).reshape(-1, width)
    except (TypeError, ValueError) as exc:
        raise ValueError("numeric feature values do not match their declared shape") from exc
    return matrix, len(combined)


def _sample_indices(length: int) -> list[int]:
    if length <= 0:
        raise ValueError("embedded image statistics require at least one frame")
    minimum = min(100, length)
    sample_count = max(minimum, min(int(length**0.75), 10_000))
    return np.round(np.linspace(0, length - 1, sample_count)).astype(int).tolist()


def _downsample_rgb(value: np.ndarray) -> np.ndarray:
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("embedded image feature must decode as RGB")
    height, width = value.shape[:2]
    if max(width, height) < 300:
        return value
    factor = int(width / 150) if width > height else int(height / 150)
    return value[::factor, ::factor]


def _feature_stats(
    values: np.ndarray,
    profile: Sequence[str],
    count: int,
    *,
    image: bool,
) -> dict[str, np.ndarray[Any, Any]]:
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("statistics require a nonempty finite numeric matrix")
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
        batch = values.astype(np.result_type(values.dtype, np.float32), copy=False)
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
                result[metric] = np.array(
                    [
                        _histogram_quantile(batch[:, column], quantile)
                        for column in range(batch.shape[1])
                    ]
                )
    if image:
        result = {
            metric: value
            if metric == "count"
            else value.reshape(-1, 1, 1) / 255.0
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


def _aggregate_feature_stats(
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
            quantile_values = np.stack([item[metric] for item in values])
            result[metric] = (quantile_values * counts).sum(axis=0) / total_count
    return result


def _stats_to_lists(stats: Mapping[str, np.ndarray[Any, Any]]) -> FeatureStats:
    return {metric: value.tolist() for metric, value in stats.items()}


@dataclass(frozen=True)
class _EntryIdentity:
    device: int
    inode: int
    mode: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_EntryIdentity":
        return cls(value.st_dev, value.st_ino, value.st_mode)

    @property
    def directory_key(self) -> tuple[int, int]:
        return self.device, self.inode


class _OwnershipRegistry:
    """Track active cleanup authority without allowing inode-reuse matches."""

    def __init__(self) -> None:
        self._active: set[_EntryIdentity] = set()
        self._retiring: set[_EntryIdentity] = set()

    def register(self, identity: _EntryIdentity) -> None:
        if identity in self._active or identity in self._retiring:
            raise ValueError("duplicate task-owned filesystem identity")
        self._active.add(identity)

    def is_owned(self, identity: _EntryIdentity) -> bool:
        return identity in self._active

    def begin_retirement(self, identity: _EntryIdentity) -> bool:
        if identity not in self._active:
            return False
        self._active.remove(identity)
        self._retiring.add(identity)
        return True

    def restore(self, identity: _EntryIdentity) -> None:
        if identity in self._retiring:
            self._retiring.remove(identity)
            self._active.add(identity)

    def finish_retirement(self, identity: _EntryIdentity) -> None:
        self._retiring.discard(identity)

    def release_all(self) -> None:
        self._active.clear()
        self._retiring.clear()


class _CleanupFailures:
    """Attempt every cleanup and preserve an already-active primary error."""

    def __init__(self) -> None:
        self.errors: list[tuple[str, BaseException]] = []

    def attempt(self, context: str, action: Callable[[], None]) -> None:
        try:
            action()
        except BaseException as exc:
            self.errors.append((context, exc))

    def finish(self, primary: BaseException | None, context: str) -> None:
        if not self.errors:
            return
        details = "; ".join(
            f"{label}: {type(error).__name__}: {error}"
            for label, error in self.errors
        )
        if primary is not None:
            primary.add_note(f"Secondary {context} failures: {details}")
            return
        first_error = self.errors[0][1]
        error = ValueError(f"{context} cleanup failed: {first_error}")
        error.add_note(f"Cleanup failures: {details}")
        raise error from first_error


class _AnchoredDirectoryPath:
    """Hold every component of an absolute directory path open and verified."""

    def __init__(
        self,
        path: Path,
        label: str,
        descriptors: list[int],
        names: list[str],
        identities: list[_EntryIdentity],
        created_final: bool,
    ) -> None:
        self.path = path
        self.label = label
        self._descriptors = descriptors
        self._names = names
        self._identities = identities
        self.created_final = created_final

    @classmethod
    def open(
        cls,
        path: Path,
        label: str,
        *,
        create_final: bool,
        registry: _OwnershipRegistry | None = None,
    ) -> "_AnchoredDirectoryPath":
        absolute = Path(os.path.abspath(path))
        components = absolute.parts[1:]
        if not components:
            raise ValueError(f"{label} must not be the filesystem root")
        root_fd = os.open("/", _DIRECTORY_FLAGS)
        try:
            root_identity = _directory_identity(os.fstat(root_fd), label)
        except BaseException as exc:
            failures = _CleanupFailures()
            failures.attempt("root descriptor close", lambda: os.close(root_fd))
            primary = (
                exc
                if isinstance(exc, ValueError)
                else ValueError(f"unable to anchor {label}")
            )
            failures.finish(primary, label)
            if primary is exc:
                raise
            raise primary from exc
        descriptors = [root_fd]
        names: list[str] = []
        identities = [root_identity]
        created_final_component = False
        try:
            for position, component in enumerate(components):
                parent_fd = descriptors[-1]
                try:
                    before = os.stat(
                        component,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    if not create_final or position != len(components) - 1:
                        raise ValueError(
                            f"{label} parent components must already exist"
                        ) from None
                    if registry is None:
                        raise ValueError(f"ownership registry required for {label}")
                    parent_is_attached = lambda: _anchored_chain_is_attached(
                        descriptors,
                        names,
                        identities,
                        label,
                    )
                    try:
                        child = _create_published_child_directory(
                            parent_fd,
                            component,
                            label,
                            parent_is_attached,
                            registry,
                        )
                    except (OSError, ValueError) as exc:
                        raise ValueError(f"unable to anchor {label}") from exc
                    created_final_component = True
                    descriptors.append(child.descriptor)
                    child.descriptor = -1
                    names.append(component)
                    identities.append(child.identity)
                    continue
                expected = _directory_identity(before, label)
                child_fd = -1
                try:
                    child_fd = os.open(
                        component,
                        _DIRECTORY_FLAGS,
                        dir_fd=parent_fd,
                    )
                    opened = _directory_identity(os.fstat(child_fd), label)
                    current = _directory_identity(
                        os.stat(
                            component,
                            dir_fd=parent_fd,
                            follow_symlinks=False,
                        ),
                        label,
                    )
                    if expected != opened or opened != current:
                        raise ValueError(f"{label} changed while opening")
                except BaseException as exc:
                    failures = _CleanupFailures()
                    if child_fd >= 0:
                        failures.attempt(
                            "untracked directory close",
                            lambda descriptor=child_fd: os.close(descriptor),
                        )
                    failures.finish(exc, label)
                    raise
                descriptors.append(child_fd)
                names.append(component)
                identities.append(opened)
            return cls(
                absolute,
                label,
                descriptors,
                names,
                identities,
                created_final_component,
            )
        except BaseException as exc:
            primary = (
                exc
                if not isinstance(exc, OSError)
                else ValueError(f"unable to anchor {label}")
            )
            failures = _CleanupFailures()
            for descriptor in reversed(descriptors):
                failures.attempt(
                    "anchored descriptor close",
                    lambda current=descriptor: os.close(current),
                )
            descriptors.clear()
            failures.finish(primary, label)
            if primary is exc:
                raise
            raise primary from exc

    @property
    def descriptor(self) -> int:
        return self._descriptors[-1]

    @property
    def identity(self) -> _EntryIdentity:
        return self._identities[-1]

    @property
    def ancestry(self) -> tuple[tuple[int, int], ...]:
        return tuple(identity.directory_key for identity in self._identities)

    def verify(self, context: str) -> None:
        self._verify_components(len(self._names), context)

    def is_attached(self) -> bool:
        try:
            self._verify_components(len(self._names), "cleanup")
        except (OSError, ValueError):
            return False
        return True

    def _parent_chain_is_attached(self) -> bool:
        try:
            self._verify_components(
                max(0, len(self._names) - 1),
                "cleanup",
            )
        except (OSError, ValueError):
            return False
        return True

    def _verify_components(self, count: int, context: str) -> None:
        if len(self._descriptors) != len(self._identities):
            raise ValueError(f"closed {self.label}")
        if _directory_identity(
            os.fstat(self._descriptors[0]), self.label
        ) != self._identities[0]:
            raise ValueError(f"{self.label} changed during {context}")
        for index, name in enumerate(self._names[:count]):
            expected = self._identities[index + 1]
            try:
                entry = _directory_identity(
                    os.stat(
                        name,
                        dir_fd=self._descriptors[index],
                        follow_symlinks=False,
                    ),
                    self.label,
                )
                opened = _directory_identity(
                    os.fstat(self._descriptors[index + 1]), self.label
                )
            except (OSError, ValueError) as exc:
                raise ValueError(f"{self.label} changed during {context}") from exc
            if entry != expected or opened != expected:
                raise ValueError(f"{self.label} changed during {context}")

    def remove_created_final(self, registry: _OwnershipRegistry) -> None:
        if not self.created_final:
            return
        parent_fd = self._descriptors[-2]
        name = self._names[-1]
        _remove_owned_empty_directory_at(
            parent_fd,
            name,
            self.identity,
            registry,
            self._parent_chain_is_attached,
        )

    def close(self) -> None:
        failures = _CleanupFailures()
        for descriptor in reversed(self._descriptors):
            failures.attempt(
                "anchored descriptor close",
                lambda current=descriptor: os.close(current),
            )
        self._descriptors.clear()
        failures.finish(None, self.label)


@dataclass
class _ChildDirectory:
    parent_fd: int
    name: str
    descriptor: int
    identity: _EntryIdentity
    context: str
    created: bool

    def verify(self, phase: str) -> None:
        try:
            entry = _directory_identity(
                os.stat(
                    self.name,
                    dir_fd=self.parent_fd,
                    follow_symlinks=False,
                ),
                self.context,
            )
            opened = _directory_identity(os.fstat(self.descriptor), self.context)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{self.context} changed during {phase}") from exc
        if entry != self.identity or opened != self.identity:
            raise ValueError(f"{self.context} changed during {phase}")

    def is_attached(self) -> bool:
        try:
            self.verify("cleanup")
        except (OSError, ValueError):
            return False
        return True

    def close(self) -> None:
        if self.descriptor >= 0:
            descriptor = self.descriptor
            self.descriptor = -1
            os.close(descriptor)


def _directory_identity(value: os.stat_result, context: str) -> _EntryIdentity:
    if stat.S_ISLNK(value.st_mode):
        raise ValueError(f"{context} path contains a symbolic link")
    if not stat.S_ISDIR(value.st_mode):
        raise ValueError(f"{context} path must contain only real directories")
    return _EntryIdentity.from_stat(value)


def _anchored_chain_is_attached(
    descriptors: Sequence[int],
    names: Sequence[str],
    identities: Sequence[_EntryIdentity],
    context: str,
) -> bool:
    """Prove that every held descriptor is still linked at its expected name."""
    try:
        if len(descriptors) != len(identities):
            return False
        if _directory_identity(os.fstat(descriptors[0]), context) != identities[0]:
            return False
        for index, name in enumerate(names):
            expected = identities[index + 1]
            if (
                _directory_identity(
                    os.stat(
                        name,
                        dir_fd=descriptors[index],
                        follow_symlinks=False,
                    ),
                    context,
                )
                != expected
                or _directory_identity(
                    os.fstat(descriptors[index + 1]), context
                )
                != expected
            ):
                return False
    except (OSError, ValueError):
        return False
    return True


def _open_or_create_child_directory(
    parent_fd: int,
    name: str,
    context: str,
    parent_is_attached: Callable[[], bool],
    registry: _OwnershipRegistry,
) -> _ChildDirectory:
    try:
        try:
            before = os.stat(
                name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return _create_published_child_directory(
                parent_fd,
                name,
                context,
                parent_is_attached,
                registry,
            )
        expected = _directory_identity(before, context)
        return _open_child_directory(
            parent_fd,
            name,
            expected,
            context,
            created=False,
        )
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"unable to open {context}") from exc


def _open_child_directory(
    parent_fd: int,
    name: str,
    expected: _EntryIdentity,
    context: str,
    *,
    created: bool,
) -> _ChildDirectory:
    descriptor = -1
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        opened = _directory_identity(os.fstat(descriptor), context)
        current = _directory_identity(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False),
            context,
        )
        if expected != opened or opened != current:
            raise ValueError(f"{context} changed while opening")
        return _ChildDirectory(
            parent_fd,
            name,
            descriptor,
            expected,
            context,
            created,
        )
    except BaseException as exc:
        failures = _CleanupFailures()
        if descriptor >= 0:
            failures.attempt(
                "untracked child descriptor close",
                lambda: os.close(descriptor),
            )
        failures.finish(exc, context)
        raise


def _create_private_child_directory(
    parent_fd: int,
    prefix: str,
    context: str,
    parent_is_attached: Callable[[], bool],
    registry: _OwnershipRegistry,
) -> _ChildDirectory:
    """Create an unpredictable child and establish its identity before use.

    `mkdirat` succeeds only for the freshly generated, previously absent name.
    The name has 96 random bits and is never a public dataset name. The first
    no-follow stat is therefore the ownership provenance; the held directory
    descriptor and a second no-follow stat must match it before any write or
    deterministic-name publication.
    """
    for _ in range(100):
        name = prefix + secrets.token_hex(12)
        if not parent_is_attached():
            raise ValueError(f"{context} parent changed before creation")
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        expected: _EntryIdentity | None = None
        descriptor = -1
        try:
            expected = _directory_identity(
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False), context
            )
            registry.register(expected)
            descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            opened = _directory_identity(os.fstat(descriptor), context)
            current = _directory_identity(
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False), context
            )
            if (
                not parent_is_attached()
                or expected != opened
                or opened != current
            ):
                raise ValueError(f"{context} changed while establishing ownership")
            return _ChildDirectory(
                parent_fd,
                name,
                descriptor,
                expected,
                context,
                True,
            )
        except BaseException as exc:
            failures = _CleanupFailures()
            if descriptor >= 0:
                failures.attempt(
                    "private directory descriptor close",
                    lambda current=descriptor: os.close(current),
                )
            if expected is not None:
                failures.attempt(
                    "private directory removal",
                    lambda: _remove_owned_entry_at(
                        parent_fd,
                        name,
                        expected,
                        registry,
                        parent_is_attached,
                    ),
                )
            primary = (
                exc
                if not isinstance(exc, OSError)
                else ValueError(f"unable to open {context}")
            )
            failures.finish(primary, context)
            if primary is exc:
                raise
            raise primary from exc
    raise ValueError(f"unable to create {context}")


def _publish_private_directory(
    directory: _ChildDirectory,
    destination_parent_fd: int,
    destination_name: str,
    context: str,
    parent_is_attached: Callable[[], bool],
) -> _ChildDirectory:
    if not parent_is_attached():
        raise ValueError(f"{context} parent changed before publication")
    directory.verify("private publication")
    source_name = directory.name
    rename_noreplace_at(
        directory.parent_fd,
        source_name,
        destination_parent_fd,
        destination_name,
    )
    directory.parent_fd = destination_parent_fd
    directory.name = destination_name
    directory.context = context
    if not parent_is_attached():
        raise ValueError(f"{context} changed while publishing")
    try:
        opened = _directory_identity(os.fstat(directory.descriptor), context)
        current = _directory_identity(
            os.stat(
                destination_name,
                dir_fd=destination_parent_fd,
                follow_symlinks=False,
            ),
            context,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"{context} changed while publishing") from exc
    if opened != directory.identity or current != directory.identity:
        raise ValueError(f"{context} changed while publishing")
    return directory


def _create_published_child_directory(
    parent_fd: int,
    name: str,
    context: str,
    parent_is_attached: Callable[[], bool],
    registry: _OwnershipRegistry,
) -> _ChildDirectory:
    private: _ChildDirectory | None = None
    try:
        private = _create_private_child_directory(
            parent_fd,
            _PRIVATE_DIRECTORY_PREFIX,
            context,
            parent_is_attached,
            registry,
        )
        return _publish_private_directory(
            private,
            parent_fd,
            name,
            context,
            parent_is_attached,
        )
    except BaseException as exc:
        failures = _CleanupFailures()
        if private is not None:
            failures.attempt("private directory close", private.close)
            failures.attempt(
                "private directory removal",
                lambda: _remove_owned_entry_at(
                    parent_fd,
                    private.name,
                    private.identity,
                    registry,
                    parent_is_attached,
                ),
            )
        failures.finish(exc, context)
        raise


def _entry_at(parent_fd: int, name: str) -> _EntryIdentity | None:
    try:
        return _EntryIdentity.from_stat(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
    except FileNotFoundError:
        return None


def _write_parquet_at(
    parent_fd: int,
    name: str,
    table: pa.Table,
    parent_is_attached: Callable[[], bool],
    registry: _OwnershipRegistry,
) -> _EntryIdentity:
    try:
        descriptor = os.open(name, _OUTPUT_FLAGS, 0o600, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"unable to create staged parquet {name}") from exc
    created_identity: _EntryIdentity | None = None
    manager: Any | None = None
    try:
        created = os.fstat(descriptor)
        if not stat.S_ISREG(created.st_mode):
            raise ValueError(f"staged parquet is not a regular file: {name}")
        created_identity = _EntryIdentity.from_stat(created)
        registry.register(created_identity)
        if (
            not parent_is_attached()
            or _entry_at(parent_fd, name) != created_identity
        ):
            raise ValueError(f"staged parquet changed while opening: {name}")
        manager = os.fdopen(descriptor, "wb")
        descriptor = -1
        handle = manager.__enter__()
        write_error: BaseException | None = None
        try:
            pq.write_table(table, handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException as exc:
            write_error = exc
            raise
        finally:
            failures = _CleanupFailures()
            failures.attempt(
                "parquet file close",
                lambda: manager.__exit__(
                    type(write_error) if write_error is not None else None,
                    write_error,
                    write_error.__traceback__ if write_error is not None else None,
                ),
            )
            manager = None
            failures.finish(write_error, "parquet file")
        if (
            not parent_is_attached()
            or _entry_at(parent_fd, name) != created_identity
        ):
            raise ValueError(f"staged parquet changed while writing: {name}")
        return created_identity
    except BaseException as exc:
        failures = _CleanupFailures()
        if manager is not None:
            failures.attempt(
                "parquet manager close",
                lambda: manager.__exit__(type(exc), exc, exc.__traceback__),
            )
        if descriptor >= 0:
            failures.attempt(
                "parquet descriptor close",
                lambda: os.close(descriptor),
            )
        if created_identity is not None:
            failures.attempt(
                "staged parquet removal",
                lambda: _remove_owned_entry_at(
                    parent_fd,
                    name,
                    created_identity,
                    registry,
                    parent_is_attached,
                ),
            )
        failures.finish(exc, "staged parquet")
        raise


def _retire_and_delete_owned_entry_at(
    parent_fd: int,
    name: str,
    expected: _EntryIdentity,
    registry: _OwnershipRegistry,
    still_attached: Callable[[], bool],
) -> bool:
    """Make an identity unmatchable before its deleting syscall can free it."""
    if not still_attached() or _entry_at(parent_fd, name) != expected:
        return False
    if not registry.begin_retirement(expected):
        return False
    try:
        if not still_attached() or _entry_at(parent_fd, name) != expected:
            registry.restore(expected)
            return False
        if stat.S_ISDIR(expected.mode):
            os.rmdir(name, dir_fd=parent_fd)
        else:
            os.unlink(name, dir_fd=parent_fd)
    except BaseException:
        registry.restore(expected)
        raise
    registry.finish_retirement(expected)
    return True


def _remove_owned_empty_directory_at(
    parent_fd: int,
    name: str,
    expected: _EntryIdentity,
    registry: _OwnershipRegistry,
    still_attached: Callable[[], bool],
) -> None:
    if not stat.S_ISDIR(expected.mode) or not still_attached():
        return
    if _entry_at(parent_fd, name) != expected or not still_attached():
        return
    if _entry_at(parent_fd, name) != expected:
        return
    try:
        _retire_and_delete_owned_entry_at(
            parent_fd,
            name,
            expected,
            registry,
            still_attached,
        )
    except OSError as exc:
        if exc.errno in (errno.ENOTEMPTY, errno.EEXIST):
            return
        raise


def _remove_owned_tree_at(
    parent_fd: int,
    name: str,
    expected: _EntryIdentity,
    registry: _OwnershipRegistry,
    still_attached: Callable[[], bool],
) -> None:
    """Delete only identities recorded as task-owned below an attached parent."""
    if not registry.is_owned(expected) or not still_attached():
        return
    if _entry_at(parent_fd, name) != expected:
        return
    if not stat.S_ISDIR(expected.mode):
        if not still_attached() or _entry_at(parent_fd, name) != expected:
            return
        _retire_and_delete_owned_entry_at(
            parent_fd,
            name,
            expected,
            registry,
            still_attached,
        )
        return
    _remove_owned_empty_directory_at(
        parent_fd,
        name,
        expected,
        registry,
        still_attached,
    )
    if not still_attached() or _entry_at(parent_fd, name) != expected:
        return
    descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    failures = _CleanupFailures()
    traversal_error: BaseException | None = None
    opened_valid = False
    try:
        opened_valid = not (
            _EntryIdentity.from_stat(os.fstat(descriptor)) != expected
            or not still_attached()
            or _entry_at(parent_fd, name) != expected
        )

        def directory_is_attached() -> bool:
            try:
                return (
                    still_attached()
                    and _EntryIdentity.from_stat(os.fstat(descriptor)) == expected
                    and _entry_at(parent_fd, name) == expected
                )
            except OSError:
                return False

        if opened_valid:
            for child in sorted(os.listdir(descriptor)):
                if not directory_is_attached():
                    opened_valid = False
                    break
                child_identity = _entry_at(descriptor, child)
                if (
                    child_identity is not None
                    and registry.is_owned(child_identity)
                ):
                    failures.attempt(
                        f"owned child removal {child}",
                        lambda child_name=child, identity=child_identity: (
                            _remove_owned_tree_at(
                                descriptor,
                                child_name,
                                identity,
                                registry,
                                directory_is_attached,
                            )
                        ),
                    )
    except BaseException as exc:
        traversal_error = exc
    finally:
        failures.attempt("owned tree descriptor close", lambda: os.close(descriptor))
    failures.finish(traversal_error, "owned tree")
    if traversal_error is not None:
        raise traversal_error
    if not opened_valid:
        return
    if not still_attached() or _entry_at(parent_fd, name) != expected:
        return
    _remove_owned_empty_directory_at(
        parent_fd,
        name,
        expected,
        registry,
        still_attached,
    )


def _remove_owned_entry_at(
    parent_fd: int,
    name: str,
    expected: _EntryIdentity,
    registry: _OwnershipRegistry,
    still_attached: Callable[[], bool],
) -> None:
    _remove_owned_tree_at(
        parent_fd,
        name,
        expected,
        registry,
        still_attached,
    )


def _remove_owned_identities_below(
    parent_fd: int,
    registry: _OwnershipRegistry,
    still_attached: Callable[[], bool],
    visited: set[tuple[int, int]] | None = None,
) -> None:
    """Find and remove recorded identities only below a live anchored root."""
    if visited is None:
        visited = set()
    failures = _CleanupFailures()
    for name in sorted(os.listdir(parent_fd)):
        if not still_attached():
            return
        current = _entry_at(parent_fd, name)
        if current is None:
            continue
        if registry.is_owned(current):
            failures.attempt(
                f"owned entry removal {name}",
                lambda current_name=name, current_identity=current: (
                    _remove_owned_tree_at(
                        parent_fd,
                        current_name,
                        current_identity,
                        registry,
                        still_attached,
                    )
                ),
            )
            continue
        if not stat.S_ISDIR(current.mode):
            continue
        key = current.directory_key
        if key in visited:
            continue
        descriptor = -1
        try:
            descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            opened = _EntryIdentity.from_stat(os.fstat(descriptor))
            linked = _entry_at(parent_fd, name)
            if opened != current or linked != current:
                continue
            visited.add(key)

            def directory_is_attached(
                parent_guard: Callable[[], bool] = still_attached,
                parent_descriptor: int = parent_fd,
                child_name: str = name,
                child_descriptor: int = descriptor,
                expected: _EntryIdentity = current,
            ) -> bool:
                try:
                    return (
                        parent_guard()
                        and _EntryIdentity.from_stat(
                            os.fstat(child_descriptor)
                        )
                        == expected
                        and _entry_at(parent_descriptor, child_name) == expected
                    )
                except OSError:
                    return False

            _remove_owned_identities_below(
                descriptor,
                registry,
                directory_is_attached,
                visited,
            )
        except BaseException as exc:
            failures.errors.append((f"cleanup traversal {name}", exc))
            continue
        finally:
            if descriptor >= 0:
                failures.attempt(
                    "cleanup traversal descriptor close",
                    lambda current_descriptor=descriptor: os.close(
                        current_descriptor
                    ),
                )
    failures.finish(None, "cleanup traversal")


def _publish(
    staging: Path,
    packed: Sequence[pa.Table],
    task_table: pa.Table,
    chunks_size: int,
    *,
    source_tree: SecureTree,
) -> tuple[Path, ...]:
    source_anchor: _AnchoredDirectoryPath | None = None
    staging_anchor: _AnchoredDirectoryPath | None = None
    meta: _ChildDirectory | None = None
    bundle: _ChildDirectory | None = None
    data_identity: _EntryIdentity | None = None
    tasks_identity: _EntryIdentity | None = None
    success = False
    registry = _OwnershipRegistry()
    primary_error: BaseException | None = None

    def staging_is_linked() -> bool:
        return staging_anchor is not None and staging_anchor.is_attached()

    def bundle_is_linked() -> bool:
        return staging_is_linked() and bundle is not None and bundle.is_attached()

    try:
        source_anchor = _AnchoredDirectoryPath.open(
            source_tree.path,
            "source path",
            create_final=False,
        )
        if source_anchor.identity.directory_key != source_tree.directory_identity:
            raise ValueError("source path changed before staging publication")
        staging_anchor = _AnchoredDirectoryPath.open(
            staging,
            "staging path",
            create_final=True,
            registry=registry,
        )
        if (
            source_anchor.identity.directory_key in staging_anchor.ancestry
            or staging_anchor.identity.directory_key in source_anchor.ancestry
        ):
            raise ValueError("staging and source directory identities overlap")
        if _entry_at(staging_anchor.descriptor, "data") is not None:
            raise ValueError("unsafe staging data entry already exists")
        meta = _open_or_create_child_directory(
            staging_anchor.descriptor,
            "meta",
            "staging meta",
            staging_is_linked,
            registry,
        )
        if _entry_at(meta.descriptor, "tasks.parquet") is not None:
            raise ValueError("unsafe staging tasks entry already exists")
        bundle = _create_private_child_directory(
            staging_anchor.descriptor,
            ".v30-data-",
            "staging bundle",
            staging_is_linked,
            registry,
        )
        data = _open_or_create_child_directory(
            bundle.descriptor,
            "data",
            "staging bundle data",
            bundle_is_linked,
            registry,
        )
        data_identity = data.identity

        def data_is_linked() -> bool:
            return bundle_is_linked() and data.is_attached()

        chunk_directories: dict[int, _ChildDirectory] = {}
        data_error: BaseException | None = None
        try:
            for number, table in enumerate(packed):
                chunk_index = number // chunks_size
                chunk = chunk_directories.get(chunk_index)
                if chunk is None:
                    chunk = _open_or_create_child_directory(
                        data.descriptor,
                        f"chunk-{chunk_index:03d}",
                        "staging data chunk",
                        data_is_linked,
                        registry,
                    )
                    chunk_directories[chunk_index] = chunk

                def chunk_is_linked(
                    current: _ChildDirectory = chunk,
                ) -> bool:
                    return data_is_linked() and current.is_attached()

                _write_parquet_at(
                    chunk.descriptor,
                    f"file-{number % chunks_size:03d}.parquet",
                    table,
                    chunk_is_linked,
                    registry,
                )
            tasks_identity = _write_parquet_at(
                bundle.descriptor,
                "tasks.parquet",
                task_table,
                bundle_is_linked,
                registry,
            )
            for chunk in chunk_directories.values():
                chunk.verify("publication")
            data.verify("publication")
        except BaseException as exc:
            data_error = exc
            raise
        finally:
            failures = _CleanupFailures()
            for chunk in chunk_directories.values():
                failures.attempt("data chunk close", chunk.close)
            failures.attempt("bundle data close", data.close)
            failures.finish(data_error, "staged data directories")

        staging_anchor.verify("publication")
        meta.verify("publication")
        bundle.verify("publication")
        rename_noreplace_at(
            bundle.descriptor,
            "data",
            staging_anchor.descriptor,
            "data",
        )
        if _entry_at(staging_anchor.descriptor, "data") != data_identity:
            raise ValueError("staging data changed during publication")
        staging_anchor.verify("publication")
        meta.verify("publication")
        rename_noreplace_at(
            bundle.descriptor,
            "tasks.parquet",
            meta.descriptor,
            "tasks.parquet",
        )
        if _entry_at(meta.descriptor, "tasks.parquet") != tasks_identity:
            raise ValueError("staging tasks changed during publication")
        staging_anchor.verify("publication")
        meta.verify("publication")
        source_anchor.verify("staging publication")
        try:
            source_tree.verify()
        except ValueError as exc:
            raise ValueError("source changed during staging publication") from exc
        staging_anchor.verify("publication")
        meta.verify("publication")
        if _entry_at(staging_anchor.descriptor, "data") != data_identity:
            raise ValueError("staging data changed during publication")
        if _entry_at(meta.descriptor, "tasks.parquet") != tasks_identity:
            raise ValueError("staging tasks changed during publication")
        os.fsync(meta.descriptor)
        os.fsync(staging_anchor.descriptor)
        success = True
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        failures = _CleanupFailures()
        if bundle is not None:
            failures.attempt("staging bundle close", bundle.close)

        if (
            (not success or failures.errors)
            and staging_anchor is not None
            and staging_is_linked()
        ):
            failures.attempt(
                "failed publication rollback",
                lambda: _remove_owned_identities_below(
                    staging_anchor.descriptor,
                    registry,
                    staging_is_linked,
                ),
            )
        elif bundle is not None and staging_anchor is not None:
            failures.attempt(
                "private bundle removal",
                lambda: _remove_owned_entry_at(
                    staging_anchor.descriptor,
                    bundle.name,
                    bundle.identity,
                    registry,
                    staging_is_linked,
                ),
            )
            if failures.errors and staging_is_linked():
                failures.attempt(
                    "publication rollback after bundle cleanup failure",
                    lambda: _remove_owned_identities_below(
                        staging_anchor.descriptor,
                        registry,
                        staging_is_linked,
                    ),
                )

        if meta is not None:
            failures.attempt("staging meta close", meta.close)
        if source_anchor is not None:
            failures.attempt("source anchor close", source_anchor.close)

        if (
            success
            and failures.errors
            and staging_anchor is not None
            and staging_is_linked()
        ):
            failures.attempt(
                "publication rollback after close failure",
                lambda: _remove_owned_identities_below(
                    staging_anchor.descriptor,
                    registry,
                    staging_is_linked,
                ),
            )

        if (
            (not success or failures.errors)
            and staging_anchor is not None
        ):
            failures.attempt(
                "created staging root removal",
                lambda: staging_anchor.remove_created_final(registry),
            )
        if staging_anchor is not None:
            failures.attempt("staging anchor close", staging_anchor.close)

        if primary_error is None and not failures.errors:
            registry.release_all()
        failures.finish(primary_error, "v3 staging publication")
    return tuple(
        staging
        / (
            f"data/chunk-{number // chunks_size:03d}/"
            f"file-{number % chunks_size:03d}.parquet"
        )
        for number in range(len(packed))
    )

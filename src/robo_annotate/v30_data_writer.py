"""Repack selected LeRobot v3.0 data episodes into fresh Parquet shards."""

from __future__ import annotations

import io
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .lerobot import DatasetIndex, EpisodeInfo
from .secure_tree import SecureTree


_MAX_PARQUET_BYTES = 1024 * 1024 * 1024
_MAX_JSON_BYTES = 16 * 1024 * 1024
_OFFICIAL_INDEX_COLUMNS = ("episode_index", "frame_index", "index", "task_index")
_BASIC_STAT_METRICS = ("min", "max", "mean", "std", "count")


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
    aggregate_stats: dict[str, dict[str, list[int | float]]]
    episode_stats: dict[int, dict[str, dict[str, list[int | float]]]]


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

    parquet_files = _publish(staging, packed, task_table, chunks_size)
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
    dict[str, dict[str, list[int | float]]],
    dict[int, dict[str, dict[str, list[int | float]]]],
]:
    if not tables:
        raise ValueError("numeric statistics require selected episode tables")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("info.features must be an object")
    specs: dict[str, tuple[str, int]] = {}
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
            specs[name] = (dtype, 3)
            continue
        try:
            numpy_dtype = np.dtype(dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported numeric feature dtype: {name}") from exc
        if numpy_dtype.kind not in "biuf":
            raise ValueError(f"unsupported numeric feature dtype: {name}")
        specs[name] = (dtype, shape[-1])
        if not profile:
            raise ValueError(f"numeric stats profile is empty for {name}")

    episode_stats: dict[int, dict[str, dict[str, list[int | float]]]] = {}
    for episode_index, table in enumerate(tables):
        episode_stats[episode_index] = {}
        for name, (dtype, width) in specs.items():
            matrix, count = _numeric_matrix(
                table[name],
                width,
                sample_images=dtype == "image",
            )
            episode_stats[episode_index][name] = _feature_stats(
                matrix,
                profiles[name],
                count,
            )

    aggregate = {
        name: _aggregate_feature_stats(
            [episode_stats[index][name] for index in range(len(tables))],
            profiles[name],
        )
        for name in specs
    }
    return aggregate, episode_stats


def _numeric_matrix(
    array: pa.ChunkedArray,
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
            pixels.append(rgb.astype(np.float64).reshape(-1, 3) / 255.0)
        return np.concatenate(pixels, axis=0), len(indices)
    try:
        matrix = np.asarray(combined.to_pylist(), dtype=np.float64).reshape(-1, width)
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
) -> dict[str, list[int | float]]:
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("statistics require a nonempty finite numeric matrix")
    result: dict[str, list[int | float]] = {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [count],
    }
    for metric in profile:
        if metric.startswith("q"):
            quantile = int(metric[1:]) / 100.0
            result[metric] = [
                _histogram_quantile(values[:, column], quantile)
                for column in range(values.shape[1])
            ]
    return result


def _histogram_quantile(values: np.ndarray, quantile: float) -> float:
    if len(values) < 2:
        return float(values.mean())
    minimum, maximum = float(values.min()), float(values.max())
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
    values: Sequence[Mapping[str, list[int | float]]],
    profile: Sequence[str],
) -> dict[str, list[int | float]]:
    counts = [int(item["count"][0]) for item in values]
    total = sum(counts)
    width = len(values[0]["mean"])
    means = [
        sum(
            float(item["mean"][column]) * count
            for item, count in zip(values, counts, strict=True)
        )
        / total
        for column in range(width)
    ]
    result: dict[str, list[int | float]] = {
        "min": [
            min(float(item["min"][column]) for item in values)
            for column in range(width)
        ],
        "max": [
            max(float(item["max"][column]) for item in values)
            for column in range(width)
        ],
        "mean": means,
        "std": [
            math.sqrt(
                sum(
                    (
                        float(item["std"][column]) ** 2
                        + (float(item["mean"][column]) - means[column]) ** 2
                    )
                    * count
                    for item, count in zip(values, counts, strict=True)
                )
                / total
            )
            for column in range(width)
        ],
        "count": [total],
    }
    for metric in profile:
        if metric.startswith("q"):
            result[metric] = [
                sum(
                    float(item[metric][column]) * count
                    for item, count in zip(values, counts, strict=True)
                )
                / total
                for column in range(width)
            ]
    return result


def _publish(
    staging: Path,
    packed: Sequence[pa.Table],
    task_table: pa.Table,
    chunks_size: int,
) -> tuple[Path, ...]:
    staging_preexisted = staging.exists()
    staging.mkdir(parents=True, exist_ok=True)
    data_target = staging / "data"
    meta_target = staging / "meta"
    tasks_target = meta_target / "tasks.parquet"
    if data_target.exists() or tasks_target.exists():
        raise FileExistsError("staging already contains v3 data output")
    meta_preexisted = meta_target.exists()
    published_data = False
    published_tasks = False
    try:
        with TemporaryDirectory(prefix=".v30-data-", dir=staging) as temporary:
            bundle = Path(temporary)
            temporary_data = bundle / "data"
            temporary_tasks = bundle / "tasks.parquet"
            for number, table in enumerate(packed):
                relative = (
                    f"chunk-{number // chunks_size:03d}/"
                    f"file-{number % chunks_size:03d}.parquet"
                )
                path = temporary_data / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, path)
            pq.write_table(task_table, temporary_tasks)
            meta_target.mkdir(exist_ok=True)
            os.replace(temporary_data, data_target)
            published_data = True
            os.replace(temporary_tasks, tasks_target)
            published_tasks = True
    except BaseException:
        if published_tasks:
            try:
                tasks_target.unlink()
            except OSError:
                pass
        if published_data:
            try:
                shutil.rmtree(data_target)
            except OSError:
                pass
        if not meta_preexisted:
            try:
                meta_target.rmdir()
            except OSError:
                pass
        if not staging_preexisted:
            try:
                staging.rmdir()
            except OSError:
                pass
        raise
    return tuple(
        staging / f"data/chunk-{number // chunks_size:03d}/file-{number % chunks_size:03d}.parquet"
        for number in range(len(packed))
    )

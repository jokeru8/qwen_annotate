"""Real, deterministic shared-shard LeRobot v3.0 test datasets.

The fixture deliberately writes the official v3.0 storage layout directly.  It
does not use Robo-annotate dataset adapters, so downstream adapter, converter,
and validator tests all have an independent source of truth.
"""

import hashlib
import io
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import av
import pyarrow as pa
import pyarrow.parquet as pq

from robo_annotate.config import AnnotationConfig


_FRAME_WIDTH = 32
_FRAME_HEIGHT = 24
_TASK_TEXT = "Arrange the colored blocks."
_STAT_METRICS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
_COLOR_FRAME_CAPACITY = 1 + (255 - 16) // 20
_COLOR_EPISODE_CAPACITY = 1 + (255 - 32) // 16


def make_lerobot_v30_fixture(
    tmp_path: Path,
    *,
    lengths: tuple[int, ...] = (6, 8, 5),
    fps: float = 5.0,
    cameras: tuple[str, ...] = ("observation.images.main", "observation.images.wrist"),
    video_shards_per_episode: bool = False,
    non_padded_video_template: bool = False,
) -> Path:
    """Create a compact v3.0 dataset with shared or episode-local video shards.

    Every camera video contains the concatenated frames of every episode.  RGB
    frame colors encode camera, episode, and episode-local frame indices so
    slice tests can prove that no neighbouring episode leaks into a result.
    """
    if not lengths or any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError("lengths must contain positive integer episode lengths")
    if any(length > _COLOR_FRAME_CAPACITY for length in lengths):
        raise ValueError(
            f"lengths must not exceed {_COLOR_FRAME_CAPACITY} frames for the fixture color encoding"
        )
    if len(lengths) > _COLOR_EPISODE_CAPACITY:
        raise ValueError(
            f"fixture color encoding supports at most {_COLOR_EPISODE_CAPACITY} episodes"
        )
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
        raise ValueError("fps must be positive")
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("cameras must be unique and nonempty")

    root = tmp_path / "lerobot-v30"
    root.mkdir(parents=True)
    all_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    episode_numeric_values: list[dict[str, list[tuple[float, ...]]]] = []
    global_offset = 0

    for episode_index, length in enumerate(lengths):
        numeric_values: dict[str, list[tuple[float, ...]]] = {
            name: []
            for name in (
                "observation.state",
                "action",
                "observation.matrix",
                "observation.enabled",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            )
        }
        for frame_index in range(length):
            state = (float(episode_index), float(frame_index), float(episode_index + frame_index))
            action = (float(frame_index), float(-episode_index), float(frame_index - episode_index))
            matrix = (
                (float(episode_index), float(frame_index)),
                (float(episode_index + frame_index), float(frame_index - episode_index)),
            )
            global_index = global_offset + frame_index
            enabled = frame_index % 2 == 0
            values = {
                "observation.state": state,
                "action": action,
                "observation.matrix": tuple(value for row in matrix for value in row),
                "observation.enabled": (float(enabled),),
                "timestamp": (frame_index / fps,),
                "frame_index": (float(frame_index),),
                "episode_index": (float(episode_index),),
                "index": (float(global_index),),
                "task_index": (0.0,),
            }
            for feature, feature_value in values.items():
                numeric_values[feature].append(feature_value)
            all_rows.append(
                {
                    "observation.state": state,
                    "action": action,
                    "observation.matrix": matrix,
                    "observation.enabled": enabled,
                    "note": f"episode {episode_index}, frame {frame_index}",
                    "language_persistent": (
                        [
                            {
                                "role": "system",
                                "content": "Track the active manipulation plan.",
                                "style": "plan",
                                "timestamp": frame_index / fps,
                                "camera": None,
                                "tool_calls": [
                                    json.dumps(
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": "remember",
                                                "arguments": {"episode": episode_index},
                                            },
                                        },
                                        separators=(",", ":"),
                                    )
                                ],
                            }
                        ]
                        if frame_index == 0
                        else []
                    ),
                    "language_events": (
                        [
                            {
                                "role": "user",
                                "content": "Begin the episode.",
                                "style": None,
                                "camera": None,
                                "tool_calls": None,
                            }
                        ]
                        if frame_index == 0
                        else []
                    ),
                    "timestamp": frame_index / fps,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": global_index,
                    "task_index": 0,
                }
            )

        row: dict[str, object] = {
            "episode_index": episode_index,
            "tasks": [_TASK_TEXT],
            "length": length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": global_offset,
            "dataset_to_index": global_offset + length,
        }
        for camera in cameras:
            video_index = episode_index if video_shards_per_episode else 0
            video_from = 0.0 if video_shards_per_episode else global_offset / fps
            row[f"videos/{camera}/chunk_index"] = 0
            row[f"videos/{camera}/file_index"] = video_index
            row[f"videos/{camera}/from_timestamp"] = video_from
            row[f"videos/{camera}/to_timestamp"] = video_from + length / fps
        episode_rows.append(row)
        episode_numeric_values.append(numeric_values)
        global_offset += length

    data_path = root / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    data_table = pa.table(
        {
            "observation.state": pa.array(
                [row["observation.state"] for row in all_rows], type=pa.list_(pa.float32(), 3)
            ),
            "action": pa.array([row["action"] for row in all_rows], type=pa.list_(pa.float32(), 3)),
            "observation.matrix": pa.array(
                [row["observation.matrix"] for row in all_rows],
                type=pa.list_(pa.list_(pa.float32(), 2), 2),
            ),
            "observation.enabled": pa.array(
                [row["observation.enabled"] for row in all_rows], type=pa.bool_()
            ),
            "note": pa.array([row["note"] for row in all_rows], type=pa.string()),
            "language_persistent": make_official_language_array(
                [row["language_persistent"] for row in all_rows],
                persistent=True,
            ),
            "language_events": make_official_language_array(
                [row["language_events"] for row in all_rows],
                persistent=False,
            ),
            "timestamp": pa.array([row["timestamp"] for row in all_rows], type=pa.float32()),
            "frame_index": pa.array([row["frame_index"] for row in all_rows], type=pa.int64()),
            "episode_index": pa.array([row["episode_index"] for row in all_rows], type=pa.int64()),
            "index": pa.array([row["index"] for row in all_rows], type=pa.int64()),
            "task_index": pa.array([row["task_index"] for row in all_rows], type=pa.int64()),
        }
    )
    pq.write_table(data_table, data_path)

    meta = root / "meta"
    meta.mkdir()
    pq.write_table(
        pa.table({"task_index": pa.array([0], type=pa.int64()), "task": pa.array([_TASK_TEXT])}),
        meta / "tasks.parquet",
    )

    video_values: list[dict[str, list[tuple[float, ...]]]] = [dict() for _ in lengths]
    for camera_index, camera in enumerate(cameras):
        if video_shards_per_episode:
            for episode_index, length in enumerate(lengths):
                filename = (
                    f"file-{episode_index}.mp4"
                    if non_padded_video_template
                    else f"file-{episode_index:03d}.mp4"
                )
                video_path = root / f"videos/{camera}/chunk-000/{filename}"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                colors = colors_for_episode(camera_index, episode_index, length)
                _write_video(video_path, colors, fps)
                decoded = decoded_colors(video_path)
                if len(decoded) != len(colors):
                    raise RuntimeError("fixture video encoder produced an unexpected frame count")
                video_values[episode_index][camera] = [
                    tuple(channel / 255.0 for channel in color) for color in decoded
                ]
        else:
            filename = "file-0.mp4" if non_padded_video_template else "file-000.mp4"
            video_path = root / f"videos/{camera}/chunk-000/{filename}"
            video_path.parent.mkdir(parents=True)
            colors = tuple(
                expected_color(camera_index, episode_index, frame_index)
                for episode_index, length in enumerate(lengths)
                for frame_index in range(length)
            )
            _write_video(video_path, colors, fps)
            offset = 0
            decoded = decoded_colors(video_path)
            if len(decoded) != len(colors):
                raise RuntimeError("fixture video encoder produced an unexpected frame count")
            for episode_index, length in enumerate(lengths):
                video_values[episode_index][camera] = [
                    tuple(channel / 255.0 for channel in color)
                    for color in decoded[offset : offset + length]
                ]
                offset += length

    episode_stats: list[dict[str, dict[str, list[float]]]] = []
    for episode_index in range(len(lengths)):
        values = episode_numeric_values[episode_index] | video_values[episode_index]
        stats = {
            feature: _feature_stats(feature_values, _stats_width(feature))
            for feature, feature_values in values.items()
        }
        episode_stats.append(stats)
        for feature, feature_stats in stats.items():
            for metric, value in feature_stats.items():
                episode_rows[episode_index][f"stats/{feature}/{metric}"] = (
                    value
                    if feature not in cameras or metric == "count"
                    else [[[channel]] for channel in value]
                )
        episode_rows[episode_index]["meta/episodes/chunk_index"] = 0
        episode_rows[episode_index]["meta/episodes/file_index"] = 0

    episode_path = meta / "episodes/chunk-000/file-000.parquet"
    episode_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episode_rows), episode_path)

    stats = {
        feature: _aggregate_feature_stats(
            [episode_stats[index][feature] for index in range(len(lengths))]
        )
        for feature in episode_stats[0]
    }
    for camera in cameras:
        stats[camera] = _image_stats_shape(stats[camera])
    (meta / "stats.json").write_text(json.dumps(stats, sort_keys=True), encoding="utf-8")

    info = {
        "codebase_version": "v3.0",
        "robot_type": "fixture_robot",
        "fps": fps,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [3],
                "names": ["joint_0", "joint_1", "joint_2"],
            },
            "action": {
                "dtype": "float32",
                "shape": [3],
                "names": ["action_0", "action_1", "action_2"],
            },
            "observation.matrix": {
                "dtype": "float32",
                "shape": [2, 2],
                "names": [["row_0", "row_1"], ["column_0", "column_1"]],
            },
            "observation.enabled": {
                "dtype": "bool",
                "shape": [1],
                "names": None,
            },
            "note": {"dtype": "string", "shape": [1], "names": None},
            "language_persistent": {
                "dtype": "language",
                "shape": [1],
                "names": None,
            },
            "language_events": {
                "dtype": "language",
                "shape": [1],
                "names": None,
            },
            **{
                camera: {
                    "dtype": "video",
                    "shape": [_FRAME_HEIGHT, _FRAME_WIDTH, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.codec": "h264",
                        "video.fps": fps,
                        "video.height": _FRAME_HEIGHT,
                        "video.pix_fmt": "yuv420p",
                        "video.width": _FRAME_WIDTH,
                    },
                }
                for camera in cameras
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index}.mp4"
            if non_padded_video_template
            else "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        ),
        "splits": {"train": f"0:{len(lengths)}"},
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return root


def make_v30_config(root: Path, work: Path) -> AnnotationConfig:
    """Build the deterministic annotation config used by v3.0 tests."""
    return AnnotationConfig.model_validate(
        {
            "source": root,
            "work_dir": work,
            "mode": "complete",
            "high_level_instruction": "Arrange the colored blocks.",
            "primary_camera": "observation.images.main",
            "refine_cameras": ["observation.images.main", "observation.images.wrist"],
            "subtasks": [{"skill": "arrange", "text": "Arrange the colored blocks."}],
        }
    )


def read_v30_info(root: Path) -> dict[str, object]:
    """Read fixture info metadata without involving an adapter."""
    return json.loads((root / "meta/info.json").read_text(encoding="utf-8"))


def expected_color(camera_index: int, episode_index: int, frame_index: int) -> tuple[int, int, int]:
    """Return the stable RGB code used for one episode-local fixture frame."""
    if min(camera_index, episode_index, frame_index) < 0:
        raise ValueError("color indices must be nonnegative")
    color = (32 + camera_index * 96, 32 + episode_index * 16, 16 + frame_index * 20)
    if any(channel > 255 for channel in color):
        raise ValueError("fixture color encoding supports only small test indices")
    return color


def colors_for_episode(camera_index: int, episode_index: int, length: int) -> tuple[tuple[int, int, int], ...]:
    """Return the expected decoded color sequence for one fixture episode."""
    return tuple(expected_color(camera_index, episode_index, frame_index) for frame_index in range(length))


def decoded_colors(path: Path) -> tuple[tuple[int, int, int], ...]:
    """Decode an MP4 and normalize each uniform test frame to its RGB code."""
    with av.open(str(path)) as container:
        return tuple(_normalized_rgb(frame.to_ndarray(format="rgb24")) for frame in container.decode(video=0))


def dominant_test_color(jpeg: bytes) -> tuple[int, int, int]:
    """Read a JPEG evidence image and recover its lossy fixture color code."""
    with av.open(io.BytesIO(jpeg), mode="r", format="image2") as container:
        frame = next(container.decode(video=0))
    return _normalized_rgb(frame.to_ndarray(format="rgb24"))


def official_core_file_digests(root: Path) -> dict[str, str]:
    """Return byte digests for official v3 files, excluding Robo metadata."""
    paths = [root / "meta/info.json", root / "meta/stats.json", root / "meta/tasks.parquet"]
    for directory, suffix in ((root / "meta/episodes", ".parquet"), (root / "data", ".parquet"), (root / "videos", ".mp4")):
        paths.extend(path for path in directory.rglob(f"*{suffix}") if path.is_file())
    return {str(path.relative_to(root)): _sha256(path) for path in sorted(paths)}


def source_tree_digest(root: Path) -> str:
    """Hash every regular source file with paths to detect any source mutation."""
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_video(path: Path, colors: Iterable[tuple[int, int, int]], fps: float) -> None:
    rate = Fraction(fps).limit_denominator(10_000)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = _FRAME_WIDTH
        stream.height = _FRAME_HEIGHT
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(rate.denominator, rate.numerator)
        for index, color in enumerate(colors):
            frame = av.VideoFrame(_FRAME_WIDTH, _FRAME_HEIGHT, "rgb24")
            frame.planes[0].update(bytes(color) * (_FRAME_WIDTH * _FRAME_HEIGHT))
            frame.pts = index
            frame.time_base = stream.time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _feature_stats(
    values: list[tuple[float, ...]], width: int
) -> dict[str, list[float]]:
    if not values:
        raise ValueError("statistics require at least one value")
    dimensions = len(values[0])
    if width <= 0 or dimensions % width or any(len(value) != dimensions for value in values):
        raise ValueError("statistics require equally shaped values")
    samples = [value[offset : offset + width] for value in values for offset in range(0, dimensions, width)]
    columns = tuple(tuple(value[dimension] for value in samples) for dimension in range(width))
    means = tuple(sum(column) / len(column) for column in columns)
    result = {
        "min": [min(column) for column in columns],
        "max": [max(column) for column in columns],
        "mean": list(means),
        "std": [math.sqrt(sum((value - mean) ** 2 for value in column) / len(column)) for column, mean in zip(columns, means, strict=True)],
        "count": [len(values)],
    }
    for quantile in (0.01, 0.10, 0.50, 0.90, 0.99):
        result[f"q{int(quantile * 100):02d}"] = [
            _official_histogram_quantile(column, quantile) for column in columns
        ]
    return result


def _stats_width(feature: str) -> int:
    if feature in {"observation.state", "action"} or feature.startswith("observation.images."):
        return 3
    if feature == "observation.matrix":
        return 2
    return 1


def _image_stats_shape(stats: dict[str, list[float]]) -> dict[str, list[float]]:
    return {
        metric: value if metric == "count" else [[[channel]] for channel in value]
        for metric, value in stats.items()
    }


def _aggregate_feature_stats(
    values: list[dict[str, list[float]]],
) -> dict[str, list[float]]:
    counts = [item["count"][0] for item in values]
    total = sum(counts)
    width = len(values[0]["mean"])
    means = [
        sum(item["mean"][column] * count for item, count in zip(values, counts, strict=True))
        / total
        for column in range(width)
    ]
    variances = [
        sum(
            (item["std"][column] ** 2 + (item["mean"][column] - means[column]) ** 2)
            * count
            for item, count in zip(values, counts, strict=True)
        )
        / total
        for column in range(width)
    ]
    result = {
        "min": [min(item["min"][column] for item in values) for column in range(width)],
        "max": [max(item["max"][column] for item in values) for column in range(width)],
        "mean": means,
        "std": [math.sqrt(value) for value in variances],
        "count": [total],
    }
    for metric in _STAT_METRICS[5:]:
        result[metric] = [
            sum(item[metric][column] * count for item, count in zip(values, counts, strict=True))
            / total
            for column in range(width)
        ]
    return result


def _official_histogram_quantile(values: tuple[float, ...], quantile: float) -> float:
    """Match LeRobot v0.6.1's one-batch 5,000-bin quantile estimator."""
    if len(values) < 2:
        return sum(values) / len(values)
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    minimum, maximum = float(array.min()), float(array.max())
    edges = np.linspace(minimum - 1e-10, maximum + 1e-10, 5001)
    histogram, _ = np.histogram(array, bins=edges)
    cumulative = np.cumsum(histogram)
    target = quantile * len(array)
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


def _json_arrow_type() -> pa.DataType:
    return pa.json_() if hasattr(pa, "json_") else pa.string()


def make_official_language_array(
    rows: list[object],
    *,
    persistent: bool,
) -> pa.ListArray:
    """Build the nested JSON-extension array emitted by HF ``datasets``.

    PyArrow cannot construct nested extension arrays directly from Python
    values, so the fixture assembles the official representation bottom-up.
    """
    flattened: list[dict[str, object]] = []
    offsets = [0]
    for value in rows:
        if not isinstance(value, list):
            raise TypeError("language fixture rows must be lists")
        flattened.extend(value)
        offsets.append(len(flattened))

    tool_call_values: list[str] = []
    tool_call_offsets = [0]
    tool_call_validity: list[bool] = []
    for row in flattened:
        calls = row.get("tool_calls")
        tool_call_validity.append(calls is not None)
        if calls is not None:
            tool_call_values.extend(calls)
        tool_call_offsets.append(len(tool_call_values))

    json_type = _json_arrow_type()
    json_storage = pa.array(tool_call_values, type=pa.string())
    json_values = (
        pa.ExtensionArray.from_storage(json_type, json_storage)
        if isinstance(json_type, pa.BaseExtensionType)
        else json_storage
    )
    tool_calls = pa.ListArray.from_arrays(
        pa.array(tool_call_offsets, type=pa.int32()),
        json_values,
        mask=pa.array([not valid for valid in tool_call_validity], type=pa.bool_()),
    )
    language_type = (
        _language_persistent_arrow_type()
        if persistent
        else _language_events_arrow_type()
    )
    fields = list(language_type.value_type)
    arrays = []
    for field in fields:
        if field.name == "tool_calls":
            arrays.append(tool_calls)
        else:
            arrays.append(
                pa.array([row.get(field.name) for row in flattened], type=field.type)
            )
    values = pa.StructArray.from_arrays(arrays, fields=fields)
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), values)


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


def _normalized_rgb(rgb: object) -> tuple[int, int, int]:
    """Map yuv420p/JPEG-decoded colors back to the separated fixture palette."""
    means = rgb.mean(axis=(0, 1))  # PyAV supplies the RGB ndarray.
    return (
        _nearest_palette_value(float(means[0]), 32, 96),
        _nearest_palette_value(float(means[1]), 32, 16),
        _nearest_palette_value(float(means[2]), 16, 20),
    )


def _nearest_palette_value(value: float, start: int, step: int) -> int:
    index = max(0, min(round((value - start) / step), (255 - start) // step))
    return start + index * step


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

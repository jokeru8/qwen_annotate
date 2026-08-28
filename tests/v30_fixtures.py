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
_STAT_METRICS = ("min", "max", "mean", "std", "count")


def make_lerobot_v30_fixture(
    tmp_path: Path,
    *,
    lengths: tuple[int, ...] = (6, 8, 5),
    fps: float = 5.0,
    cameras: tuple[str, ...] = ("observation.images.main", "observation.images.wrist"),
) -> Path:
    """Create a compact v3.0 dataset with one shared data and video shard.

    Every camera video contains the concatenated frames of every episode.  RGB
    frame colors encode camera, episode, and episode-local frame indices so
    slice tests can prove that no neighbouring episode leaks into a result.
    """
    if not lengths or any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError("lengths must contain positive integer episode lengths")
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
        state_values: list[tuple[float, ...]] = []
        action_values: list[tuple[float, ...]] = []
        for frame_index in range(length):
            state = (float(episode_index), float(frame_index), float(episode_index + frame_index))
            action = (float(frame_index), float(-episode_index), float(frame_index - episode_index))
            state_values.append(state)
            action_values.append(action)
            all_rows.append(
                {
                    "index": global_offset + frame_index,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": frame_index / fps,
                    "task_index": 0,
                    "observation.state": state,
                    "action": action,
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
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for camera in cameras:
            row[f"videos/{camera}/chunk_index"] = 0
            row[f"videos/{camera}/file_index"] = 0
            row[f"videos/{camera}/from_timestamp"] = global_offset / fps
            row[f"videos/{camera}/to_timestamp"] = (global_offset + length) / fps
        episode_rows.append(row)
        episode_numeric_values.append({"observation.state": state_values, "action": action_values})
        global_offset += length

    data_path = root / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    data_table = pa.table(
        {
            "index": pa.array([row["index"] for row in all_rows], type=pa.int64()),
            "episode_index": pa.array([row["episode_index"] for row in all_rows], type=pa.int64()),
            "frame_index": pa.array([row["frame_index"] for row in all_rows], type=pa.int64()),
            "timestamp": pa.array([row["timestamp"] for row in all_rows], type=pa.float64()),
            "task_index": pa.array([row["task_index"] for row in all_rows], type=pa.int64()),
            "observation.state": pa.array(
                [row["observation.state"] for row in all_rows], type=pa.list_(pa.float32(), 3)
            ),
            "action": pa.array([row["action"] for row in all_rows], type=pa.list_(pa.float32(), 3)),
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
        video_path = root / f"videos/{camera}/chunk-000/file-000.mp4"
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
                tuple(channel / 255.0 for channel in color) for color in decoded[offset : offset + length]
            ]
            offset += length

    episode_stats: list[dict[str, dict[str, list[float]]]] = []
    for episode_index in range(len(lengths)):
        values = episode_numeric_values[episode_index] | video_values[episode_index]
        stats = {feature: _feature_stats(feature_values) for feature, feature_values in values.items()}
        episode_stats.append(stats)
        for feature, feature_stats in stats.items():
            for metric, value in feature_stats.items():
                episode_rows[episode_index][f"stats/{feature}/{metric}"] = value

    episode_path = meta / "episodes/chunk-000/file-000.parquet"
    episode_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episode_rows), episode_path)

    all_feature_values: dict[str, list[tuple[float, ...]]] = {
        "observation.state": [value for values in episode_numeric_values for value in values["observation.state"]],
        "action": [value for values in episode_numeric_values for value in values["action"]],
    }
    for camera in cameras:
        all_feature_values[camera] = [value for values in video_values for value in values[camera]]
    stats = {feature: _feature_stats(values) for feature, values in all_feature_values.items()}
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
            **{
                camera: {
                    "dtype": "video",
                    "shape": [3, _FRAME_HEIGHT, _FRAME_WIDTH],
                    "names": ["channels", "height", "width"],
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
        },
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "total_tasks": 1,
        "total_videos": len(lengths) * len(cameras),
        "total_data_files": 1,
        "total_video_files": len(cameras),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "splits": {"train": f"0:{len(lengths)}"},
    }
    (meta / "info.json").write_text(json.dumps(info, sort_keys=True), encoding="utf-8")
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
    color = (32 + camera_index * 96, 32 + episode_index * 64, 16 + frame_index * 20)
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


def _feature_stats(values: list[tuple[float, ...]]) -> dict[str, list[float]]:
    if not values:
        raise ValueError("statistics require at least one value")
    dimensions = len(values[0])
    if any(len(value) != dimensions for value in values):
        raise ValueError("statistics require equally shaped values")
    columns = tuple(tuple(value[dimension] for value in values) for dimension in range(dimensions))
    means = tuple(sum(column) / len(column) for column in columns)
    return {
        "min": [min(column) for column in columns],
        "max": [max(column) for column in columns],
        "mean": list(means),
        "std": [math.sqrt(sum((value - mean) ** 2 for value in column) / len(column)) for column, mean in zip(columns, means, strict=True)],
        "count": [float(len(values))],
    }


def _normalized_rgb(rgb: object) -> tuple[int, int, int]:
    """Map yuv420p/JPEG-decoded colors back to the separated fixture palette."""
    means = rgb.mean(axis=(0, 1))  # PyAV supplies the RGB ndarray.
    return (
        _nearest_palette_value(float(means[0]), 32, 96),
        _nearest_palette_value(float(means[1]), 32, 64),
        _nearest_palette_value(float(means[2]), 16, 20),
    )


def _nearest_palette_value(value: float, start: int, step: int) -> int:
    index = max(0, min(round((value - start) / step), (255 - start) // step))
    return start + index * step


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

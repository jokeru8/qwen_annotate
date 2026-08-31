"""Version-neutral models and read-only inspection facade for LeRobot datasets."""

import json
import math
import os
import stat
import struct
from collections.abc import Callable
from pathlib import Path
from string import Formatter
from typing import Any, Literal, Self, cast

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from robo_annotate.config import AnnotationConfig


VIDEO_FPS_TOLERANCE = 0.01
DATA_TIMESTAMP_ABS_TOLERANCE = 1e-6
_MAX_INFO_JSON_BYTES = 16 * 1024 * 1024
DatasetVersion = Literal["v2.1", "v3.0"]


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


def data_timestamp_matches(measured: object, frame_index: int, fps: float) -> bool:
    """Accept the exact or IEEE-754 float32-rounded official frame timestamp."""
    if (
        isinstance(measured, bool)
        or not isinstance(measured, (int, float))
        or not math.isfinite(float(measured))
        or type(frame_index) is not int
        or frame_index < 0
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or fps <= 0
    ):
        return False
    expected = frame_index / float(fps)
    try:
        rounded_float32 = struct.unpack("!f", struct.pack("!f", expected))[0]
    except (OverflowError, struct.error):
        return False
    actual = float(measured)
    return math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=DATA_TIMESTAMP_ABS_TOLERANCE,
    ) or math.isclose(
        actual,
        rounded_float32,
        rel_tol=0.0,
        abs_tol=DATA_TIMESTAMP_ABS_TOLERANCE,
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


class _Reference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class EpisodeDataRef(_Reference):
    path: Path
    dataset_from_index: int = Field(ge=0, strict=True)
    dataset_to_index: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.dataset_to_index <= self.dataset_from_index:
            raise ValueError("dataset_to_index must exceed dataset_from_index")
        return self


class EpisodeVideoRef(_Reference):
    path: Path
    from_timestamp: float = Field(ge=0)
    to_timestamp: float = Field(gt=0)
    fps: float = Field(gt=0)


class EpisodeInfo(BaseModel):
    episode_index: int
    length: int
    task: str
    data: EpisodeDataRef
    videos: dict[str, EpisodeVideoRef]


class DatasetIndex(BaseModel):
    root: Path
    version: DatasetVersion
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
    """Validate and index a supported local LeRobot dataset without writing to it."""
    root, info = _read_dataset_info(config.source)
    safe_config = config.model_copy(update={"source": root})
    version = _dataset_version(info)
    if version == "v2.1":
        from .lerobot_v21 import inspect_v21_dataset

        return inspect_v21_dataset(safe_config, info, probe)
    from .lerobot_v30 import inspect_v30_dataset

    return inspect_v30_dataset(safe_config, info, probe)


def detect_dataset_version(root: Path) -> DatasetVersion:
    """Return the declared supported LeRobot version without guessing from layout."""
    _, info = _read_dataset_info(root)
    return _dataset_version(info)


def _dataset_version(info: dict[str, Any]) -> DatasetVersion:
    version = info.get("codebase_version")
    if version not in ("v2.1", "v3.0"):
        raise ValueError(f"Unsupported LeRobot codebase_version: {version!r}")
    return cast(DatasetVersion, version)


def _read_dataset_info(source: Path) -> tuple[Path, dict[str, Any]]:
    lexical_root = Path(os.path.abspath(source))
    root_fd = _open_root_directory(lexical_root)
    meta_fd: int | None = None
    try:
        root = lexical_root.resolve(strict=True)
        if _stat_identity(os.fstat(root_fd)) != _stat_identity(
            lexical_root.stat(follow_symlinks=False)
        ):
            raise ValueError("Malformed dataset root: identity changed during inspection")
        meta_fd = _open_directory_at(
            root_fd,
            "meta",
            "meta directory",
            root / "meta",
        )
        payload = _read_bounded_regular_at(
            meta_fd,
            "info.json",
            "info metadata",
            root / "meta" / "info.json",
            _MAX_INFO_JSON_BYTES,
        )
    finally:
        if meta_fd is not None:
            os.close(meta_fd)
        os.close(root_fd)

    path = root / "meta" / "info.json"
    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Malformed JSON in {path}: expected an object")
    return root, value


def _open_root_directory(path: Path) -> int:
    try:
        entry = path.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing dataset root: {path}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to inspect dataset root {path}: {exc}") from exc
    if stat.S_ISLNK(entry.st_mode):
        raise ValueError(f"Malformed dataset root {path}: symbolic links are not allowed")
    if not stat.S_ISDIR(entry.st_mode):
        raise ValueError(f"Malformed dataset root {path}: expected a directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Unable to open dataset root {path} without following links") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or _stat_identity(entry) != _stat_identity(opened):
            raise ValueError("Malformed dataset root: identity changed during inspection")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_directory_at(parent_fd: int, name: str, context: str, path: Path) -> int:
    try:
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required {context}: {path}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to inspect {context} {path}: {exc}") from exc
    if stat.S_ISLNK(entry.st_mode):
        raise ValueError(f"Malformed {context} {path}: symbolic links are not allowed")
    if not stat.S_ISDIR(entry.st_mode):
        raise ValueError(f"Malformed {context} {path}: expected a directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"Unable to open {context} {path} without following links") from exc
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _stat_identity(entry) != _stat_identity(opened)
            or _stat_identity(opened) != _stat_identity(current)
        ):
            raise ValueError(f"Malformed {context} {path}: identity changed during inspection")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_bounded_regular_at(
    parent_fd: int,
    name: str,
    context: str,
    path: Path,
    limit: int,
) -> bytes:
    try:
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required metadata file: {path}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to inspect {context} {path}: {exc}") from exc
    if stat.S_ISLNK(entry.st_mode):
        raise ValueError(f"Malformed {context} {path}: symbolic links are not allowed")
    if not stat.S_ISREG(entry.st_mode):
        raise ValueError(f"Malformed {context} {path}: expected a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"Unable to open {context} {path} without following links") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stat_identity(entry) != _stat_identity(opened)
        ):
            raise ValueError(f"Malformed {context} {path}: identity changed before reading")
        if opened.st_size > limit:
            raise ValueError(f"Malformed {context} {path}: exceeds {limit} bytes")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise ValueError(f"Malformed {context} {path}: file changed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _stat_identity(opened) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(current)
        ):
            raise ValueError(f"Malformed {context} {path}: identity changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


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

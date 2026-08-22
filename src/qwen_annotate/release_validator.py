"""Independent, no-follow validation for publishable LeRobot v2.1 datasets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from string import Formatter
from typing import Any, Callable, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import Subtask
from .constraints import validate_annotation
from .lerobot import VideoProbe, probe_video, video_fps_matches
from .models import FinalAnnotation
from .stats import iter_video_rgb_frames, recompute_stats, recompute_video_stats
from .video import extract_frames


_MAX_JSON = 16 * 1024 * 1024
_REQUIRED_COLUMNS = {"frame_index", "episode_index", "index", "task_index", "timestamp"}
_STAT_METRICS = {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}
_FORBIDDEN = {
    "confidence", "uncertainties", "attempts", "coarse_attempts", "refine_attempts",
    "visible_cues", "evidence", "prompt", "prompts", "api_key", "endpoint",
    "sampling", "sampling_details", "outbox", "_pipeline_transition_events", "model_response",
}


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BoundaryPreview(_Report):
    episode_index: int = Field(ge=0)
    camera_key: str = Field(min_length=1)
    frame_indices: tuple[int, int]


class ReleaseReport(_Report):
    path: Path
    valid: Literal[True] = True
    episode_count: int = Field(ge=0)
    frame_count: int = Field(ge=0)
    mode: Literal["complete", "dagger_patch"]
    subtask_template: list[Subtask] = Field(min_length=1)
    payload_files: list[str]
    payload_digests: dict[str, str]
    payload_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    preview: BoundaryPreview | None
    validated_at: datetime

    @field_validator("validated_at")
    @classmethod
    def utc_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("validated_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def payload_facts_are_canonical(self) -> "ReleaseReport":
        if self.payload_files != sorted(set(self.payload_files)):
            raise ValueError("payload_files must be sorted and unique")
        if set(self.payload_digests) != set(self.payload_files):
            raise ValueError("payload_digests must exactly cover payload_files")
        if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in self.payload_digests.values()):
            raise ValueError("payload digests must be lowercase SHA-256")
        return self


@dataclass(frozen=True)
class _Services:
    probe_video: Callable[[Path], VideoProbe]
    extract_frames: Callable[..., list[Any]]
    iter_video_rgb_frames: Callable[[Path], Any]


def validate_release(
    path: Path,
    source: Path | None = None,
    *,
    services: Any = None,
    _expected_output_root: Path | None = None,
    _expected_stats_source: Path | None = None,
    allow_legacy_sampled_image_stats: bool = False,
    deep_video_stats: bool = True,
) -> ReleaseReport:
    """Validate a converted release without consulting an annotation workspace."""
    if type(allow_legacy_sampled_image_stats) is not bool or type(deep_video_stats) is not bool:
        raise TypeError("statistics validation options must be bools")
    root = _safe_root(path, "release")
    source_root = _safe_root(source, "source") if source is not None else None
    stats_source = _safe_root(_expected_stats_source, "stats source") if _expected_stats_source is not None else source_root
    svc = _services(services)
    _walk_regular(root)

    info = _read_object(root / "meta/info.json")
    if _string(info, "codebase_version") != "v2.1":
        raise ValueError("release must use LeRobot v2.1")
    total_episodes = _integer(info, "total_episodes", minimum=0)
    total_frames = _integer(info, "total_frames", minimum=0)
    total_tasks = _integer(info, "total_tasks", minimum=0)
    if total_tasks != 1:
        raise ValueError("reference-compatible releases require exactly one task")
    chunks_size = _integer(info, "chunks_size", minimum=1)
    total_chunks = _integer(info, "total_chunks", minimum=0)
    if total_chunks != (total_episodes + chunks_size - 1) // chunks_size:
        raise ValueError("total_chunks is inconsistent")
    fps = _number(info, "fps")
    data_template = _template(info, "data_path", {"episode_index"})
    video_template = _template(info, "video_path", {"episode_index", "video_key"})
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("features must be an object")
    cameras = [key for key, value in features.items() if isinstance(key, str) and isinstance(value, dict) and value.get("dtype") == "video"]
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("release must define unique video cameras")
    camera_shapes: dict[str, tuple[int, int, int]] = {}
    for camera in cameras:
        shape = features[camera].get("shape")
        if (not isinstance(shape, list) or len(shape) != 3 or
                any(type(item) is not int or item <= 0 for item in shape) or shape[2] != 3):
            raise ValueError(f"video feature shape must be [height,width,3]: {camera}")
        camera_shapes[camera] = (shape[0], shape[1], shape[2])
    if _integer(info, "total_videos", minimum=0) != total_episodes * len(cameras):
        raise ValueError("total_videos is inconsistent")
    _validate_splits(info.get("splits"), total_episodes)

    task_rows = _read_jsonl(root / "meta/tasks.jsonl")
    if len(task_rows) != total_tasks:
        raise ValueError("task count mismatch")
    tasks: dict[int, str] = {}
    for row in task_rows:
        index = _integer(row, "task_index", minimum=0)
        task = _string(row, "task")
        if index in tasks or task in tasks.values():
            raise ValueError("duplicate task metadata")
        tasks[index] = task
    if set(tasks) != set(range(total_tasks)):
        raise ValueError("task indices are not contiguous")

    episode_rows = _read_jsonl(root / "meta/episodes.jsonl")
    if len(episode_rows) != total_episodes:
        raise ValueError("episode count mismatch")
    lengths: list[int] = []
    instructions: list[str] = []
    expected_payload: set[str] = set()
    schemas: list[pa.Schema] = []
    parquet_paths: list[Path] = []
    video_paths: dict[tuple[int, str], Path] = {}
    global_offset = 0
    for expected, row in enumerate(episode_rows):
        if _integer(row, "episode_index", minimum=0) != expected:
            raise ValueError("episode indices must be contiguous")
        length = _integer(row, "length", minimum=1)
        episode_tasks = row.get("tasks")
        if not isinstance(episode_tasks, list) or len(episode_tasks) != 1 or not isinstance(episode_tasks[0], str) or episode_tasks[0] not in tasks.values():
            raise ValueError("each episode must have exactly one known task")
        lengths.append(length)
        instructions.append(episode_tasks[0])
        expected_task_index = next(index for index, task in tasks.items() if task == episode_tasks[0])
        values = {"episode_chunk": expected // chunks_size, "episode_index": expected}
        parquet_rel = _render(data_template, values, "data_path")
        parquet = _regular_under(root, parquet_rel)
        expected_payload.add(parquet_rel.as_posix())
        schema = _validate_parquet(parquet, expected, length, global_offset, expected_task_index, fps, features)
        schemas.append(schema)
        parquet_paths.append(parquet)
        for camera in cameras:
            video_rel = _render(video_template, values | {"video_key": camera}, "video_path")
            video = _regular_under(root, video_rel)
            expected_payload.add(video_rel.as_posix())
            video_paths[(expected, camera)] = video
            measured = svc.probe_video(video)
            expected_height, expected_width, _ = camera_shapes[camera]
            if (measured.frames != length or not video_fps_matches(measured.fps, fps) or
                    measured.width != expected_width or measured.height != expected_height):
                raise ValueError(f"video metadata mismatch (including shape) for episode {expected}, camera {camera}")
        global_offset += length
    if sum(lengths) != total_frames:
        raise ValueError("frame count mismatch")
    if any(not schema.equals(schemas[0], check_metadata=False) for schema in schemas[1:]):
        raise ValueError("parquet schemas differ across episodes")

    actual_payload = _payload_files(root)
    if actual_payload != expected_payload:
        raise ValueError("missing or extra episode payload files")
    annotations = _read_object(root / "meta/lerobot_annotations.json")
    top_fields = {"source_root", "work_dir", "subtask_template", "episodes", "primary_camera", "updated_at"}
    if set(annotations) != top_fields:
        raise ValueError("annotation top-level schema mismatch")
    _aware_utc(annotations["updated_at"], "updated_at")
    for field in ("source_root", "work_dir"):
        _string(annotations, field)
    declared_source = Path(annotations["source_root"])
    if not declared_source.is_absolute():
        raise ValueError("annotation source_root must be an absolute path")
    if source_root is not None and declared_source.resolve(strict=False) != source_root:
        raise ValueError("annotation source_root does not match supplied source")
    declared_work = Path(annotations["work_dir"])
    if not declared_work.is_absolute() or declared_work.name != "meta":
        raise ValueError("annotation work_dir must be an absolute path ending in meta")
    if _expected_output_root is not None:
        if not isinstance(_expected_output_root, Path):
            raise TypeError("_expected_output_root must be a Path")
        expected_work = _expected_output_root.resolve(strict=False) / "meta"
        if declared_work.resolve(strict=False) != expected_work:
            raise ValueError("annotation work_dir does not match expected output/meta")
    _validate_optional_metadata(
        root, total_episodes, total_frames, lengths, features, cameras, stats_source,
        parquet_paths, video_paths, svc, allow_legacy_sampled_image_stats, deep_video_stats,
    )
    primary = _string(annotations, "primary_camera")
    if primary not in cameras:
        raise ValueError("primary camera is absent from features")
    template = _subtasks(annotations.get("subtask_template"))
    if annotations["subtask_template"] != info.get("subtask_template"):
        raise ValueError("subtask template differs between metadata files")
    mapping = info.get("high_level_instruction")
    if not isinstance(mapping, dict) or set(mapping) != {str(i) for i in range(total_episodes)}:
        raise ValueError("high-level instruction mapping is incomplete")
    entries = annotations.get("episodes")
    if not isinstance(entries, dict) or set(entries) != {str(i) for i in range(total_episodes)}:
        raise ValueError("annotation episodes must exactly match dataset episodes")
    has_starts = [isinstance(entry, dict) and "start_subtask_index" in entry for entry in entries.values()]
    if has_starts and all(has_starts):
        mode = "dagger_patch"
    elif not any(has_starts):
        mode = "complete"
    else:
        raise ValueError("all DAgger records must explicitly include start_subtask_index")
    _reject_forbidden(annotations)

    first_preview: BoundaryPreview | None = None
    annotation_facts: list[tuple[int, list[int]]] = []
    for index in range(total_episodes):
        entry = entries[str(index)]
        if not isinstance(entry, dict):
            raise ValueError("annotation episode must be an object")
        common = {"episode_index", "boundaries", "high_level_instruction", "saved_at"}
        allowed = common | ({"start_subtask_index"} if mode == "dagger_patch" else set())
        if set(entry) != allowed:
            raise ValueError("annotation episode schema mismatch")
        if _integer(entry, "episode_index", minimum=0) != index:
            raise ValueError("annotation episode key/index mismatch")
        instruction = _string(entry, "high_level_instruction")
        if instruction != mapping[str(index)] or instruction != instructions[index]:
            raise ValueError("instruction mismatch")
        _aware_utc(entry["saved_at"], "saved_at")
        boundaries = entry.get("boundaries")
        if not isinstance(boundaries, list) or any(type(item) is not int for item in boundaries):
            raise ValueError("boundaries must be integer list")
        if mode == "complete":
            start = entry.get("start_subtask_index", 0)
        else:
            start = _integer(entry, "start_subtask_index", minimum=0)
        annotation = FinalAnnotation(start_subtask_index=start, boundaries=boundaries)
        issues = validate_annotation(annotation, mode, len(template), lengths[index], 1)
        if issues:
            raise ValueError("invalid annotation boundaries: " + ",".join(issue.code for issue in issues))
        annotation_facts.append((start, boundaries))
        if first_preview is None and boundaries:
            boundary = boundaries[0]
            requested = [boundary - 1, boundary]
            samples = svc.extract_frames(video_paths[(index, primary)], primary, requested, fps)
            if len(samples) != 2 or [item.frame_index for item in samples] != requested or any(item.camera_key != primary for item in samples):
                raise ValueError("boundary preview labels do not match requested source frames")
            first_preview = BoundaryPreview(episode_index=index, camera_key=primary, frame_indices=(boundary - 1, boundary))

    _validate_task_info(root, total_episodes, instructions, lengths, template, annotation_facts)

    digests = {relative: _sha256(root / relative) for relative in sorted(expected_payload)}
    if source_root is not None:
        _walk_regular(source_root)
        source_payload = _payload_files(source_root)
        if source_payload != expected_payload:
            raise ValueError("source payload set differs from release")
        for relative, digest in digests.items():
            if _sha256(source_root / relative) != digest:
                raise ValueError(f"payload checksum mismatch: {relative}")
    aggregate = hashlib.sha256("".join(f"{name}\0{digests[name]}\n" for name in sorted(digests)).encode()).hexdigest()
    return ReleaseReport(
        path=root, episode_count=total_episodes, frame_count=total_frames, mode=mode,
        subtask_template=template, payload_files=sorted(expected_payload), payload_digests=digests,
        payload_checksum=aggregate, preview=first_preview, validated_at=datetime.now(UTC),
    )


def _services(value: Any) -> _Services:
    if value is None:
        return _Services(probe_video, extract_frames, iter_video_rgb_frames)
    getter = value.get if isinstance(value, dict) else lambda name, default: getattr(value, name, default)
    probe = getter("probe_video", probe_video)
    extractor = getter("extract_frames", extract_frames)
    frame_iterator = getter("iter_video_rgb_frames", iter_video_rgb_frames)
    if not callable(probe) or not callable(extractor) or not callable(frame_iterator):
        raise TypeError("release services must be callable")
    return _Services(probe, extractor, frame_iterator)


def _safe_root(path: Path | None, label: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{label} path must be a Path")
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory")
    return path.resolve(strict=True)


def _walk_regular(root: Path) -> None:
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in [*dirs, *files]:
            candidate = Path(current) / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError(f"release contains unsafe file type: {candidate.relative_to(root)}")


def _read_text(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"metadata path is a symlink: {path.name}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_JSON:
            raise ValueError("metadata must be a bounded regular file")
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            return handle.read(_MAX_JSON + 1)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"unable to read metadata {path.name}") from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _decode(text: str, context: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result
    try:
        return json.loads(text, object_pairs_hook=unique, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite {value}")))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"malformed {context}") from exc


def _read_object(path: Path) -> dict[str, Any]:
    value = _decode(_read_text(path), path.name)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = _read_text(path).splitlines()
    rows = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"blank JSONL row {number}")
        value = _decode(line, f"{path.name}:{number}")
        if not isinstance(value, dict):
            raise ValueError("JSONL rows must be objects")
        rows.append(value)
    return rows


def _validate_task_info(
    root: Path,
    total_episodes: int,
    instructions: list[str],
    lengths: list[int],
    template: list[Subtask],
    annotations: list[tuple[int, list[int]]],
) -> None:
    directory = root / "meta/task_info"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("meta/task_info must be a real directory")
    files = {path.name for path in directory.iterdir()}
    if files != {"task_0.json"}:
        raise ValueError("task_info must contain exactly task_0.json")
    value = _decode(_read_text(directory / "task_0.json"), "task_0.json")
    if not isinstance(value, list) or len(value) != total_episodes:
        raise ValueError("task_info must contain one entry per episode")
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"episode_id", "task_id", "task_name", "label_info"}:
            raise ValueError("task_info episode schema mismatch")
        if _integer(item, "episode_id", minimum=0) != index or _integer(item, "task_id", minimum=0) != 0:
            raise ValueError("task_info episode/task id mismatch")
        if _string(item, "task_name") != instructions[index]:
            raise ValueError("task_info task_name mismatch")
        label = item["label_info"]
        if not isinstance(label, dict) or set(label) != {"action_config"}:
            raise ValueError("task_info label_info schema mismatch")
        actions = label["action_config"]
        start_subtask, boundaries = annotations[index]
        expected_starts = [0, *boundaries]
        expected_ends = [*boundaries, lengths[index]]
        expected_subtasks = template[start_subtask:start_subtask + len(expected_starts)]
        if not isinstance(actions, list) or len(actions) != len(expected_starts) or len(expected_subtasks) != len(actions):
            raise ValueError("task_info action count mismatch")
        for action, start, end, subtask in zip(actions, expected_starts, expected_ends, expected_subtasks, strict=True):
            if not isinstance(action, dict) or set(action) != {"start_frame", "end_frame", "action_text", "skill"}:
                raise ValueError("task_info action schema mismatch")
            if (_integer(action, "start_frame", minimum=0) != start or
                    _integer(action, "end_frame", minimum=1) != end or end <= start or
                    _string(action, "action_text") != subtask.text or
                    _string(action, "skill") != subtask.skill):
                raise ValueError("task_info action differs from annotation/template")


def _integer(value: dict[str, Any], key: str, minimum: int) -> int:
    item = value.get(key)
    if type(item) is not int or item < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return item


def _number(value: dict[str, Any], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0:
        raise ValueError(f"{key} must be positive and finite")
    return float(item)


def _string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be a nonempty string")
    return item


def _template(info: dict[str, Any], key: str, required: set[str]) -> str:
    template = _string(info, key)
    fields = set()
    try:
        parts = list(Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"invalid {key} template") from exc
    for _, name, spec, conversion in parts:
        if name is None:
            continue
        if name not in {"episode_chunk", "episode_index", "video_key"} or "{" in spec or "}" in spec or conversion not in (None, "s", "r", "a"):
            raise ValueError(f"unsafe {key} template")
        fields.add(name)
    if not required <= fields:
        raise ValueError(f"{key} template misses required fields")
    return template


def _render(template: str, values: dict[str, Any], label: str) -> Path:
    try:
        relative = Path(template.format(**values))
    except Exception as exc:
        raise ValueError(f"invalid {label} template") from exc
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path escapes release")
    return relative


def _regular_under(root: Path, relative: Path) -> Path:
    target = root / relative
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"payload path escapes or is missing: {relative}") from exc
    if target.is_symlink() or not target.is_file() or not stat.S_ISREG(target.stat(follow_symlinks=False).st_mode):
        raise ValueError(f"payload is not a regular file: {relative}")
    return target


def _validate_parquet(
    path: Path, episode: int, length: int, global_offset: int,
    expected_task: int, fps: float, features: dict[str, Any],
) -> pa.Schema:
    parquet: pq.ParquetFile | None = None
    try:
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != length:
            raise ValueError("parquet row count mismatch")
        schema = parquet.schema_arrow
        if not _REQUIRED_COLUMNS <= set(schema.names):
            raise ValueError("parquet misses required index/task/timestamp columns")
        table = parquet.read(columns=sorted(_REQUIRED_COLUMNS))
        if table.column("episode_index").to_pylist() != [episode] * length:
            raise ValueError("parquet episode_index values mismatch")
        if table.column("frame_index").to_pylist() != list(range(length)):
            raise ValueError("parquet frame_index values mismatch")
        if table.column("index").to_pylist() != list(range(global_offset, global_offset + length)):
            raise ValueError("parquet global index values mismatch")
        if table.column("task_index").to_pylist() != [expected_task] * length:
            raise ValueError("parquet task_index values mismatch")
        timestamps = table.column("timestamp").to_pylist()
        if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in timestamps) or
                any(left >= right for left, right in zip(timestamps, timestamps[1:]))):
            raise ValueError("parquet timestamps must be finite, nonnegative, and strictly increasing")
        for name, declared in features.items():
            if isinstance(declared, dict) and declared.get("dtype") == "video":
                continue
            if name not in schema.names:
                raise ValueError(f"parquet misses declared info feature {name}")
            declared = features.get(name)
            if not isinstance(declared, dict) or not isinstance(declared.get("dtype"), str):
                raise ValueError(f"info features misses parquet column {name}")
            if not _dtype_compatible(schema.field(name).type, declared["dtype"]):
                raise ValueError(f"parquet schema disagrees with info for {name}")
            shape = declared.get("shape")
            if not isinstance(shape, list) or not shape or any(type(item) is not int or item <= 0 for item in shape):
                raise ValueError(f"info feature shape is invalid for {name}")
            field_type = schema.field(name).type
            if pa.types.is_fixed_size_list(field_type):
                declared_elements = math.prod(shape)
                if field_type.list_size != declared_elements:
                    raise ValueError(f"parquet shape disagrees with info for {name}")
            elif shape != [1]:
                raise ValueError(f"scalar parquet feature must declare shape [1] for {name}")
        return schema
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"unable to read parquet {path.name}") from exc
    finally:
        if parquet is not None:
            parquet.close()


def _dtype_compatible(dtype: pa.DataType, declared: str) -> bool:
    while pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        dtype = dtype.value_type
    return ((declared.startswith("int") and pa.types.is_integer(dtype)) or
            (declared.startswith("float") and pa.types.is_floating(dtype)) or
            (declared == "bool" and pa.types.is_boolean(dtype)) or
            (declared in ("string", "utf8") and pa.types.is_string(dtype)))


def _validate_splits(value: Any, total_episodes: int) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError("splits must be a nonempty object")
    covered: set[int] = set()
    for name, interval in value.items():
        if not isinstance(name, str) or not name or not isinstance(interval, str) or interval.count(":") != 1:
            raise ValueError("invalid split metadata")
        left, right = interval.split(":")
        if not left.isascii() or not right.isascii() or not left.isdecimal() or not right.isdecimal():
            raise ValueError("invalid split range")
        start, stop = int(left), int(right)
        if not (0 <= start <= stop <= total_episodes):
            raise ValueError("split range is outside episode indices")
        indices = set(range(start, stop))
        if covered & indices:
            raise ValueError("split ranges overlap")
        covered |= indices
    if covered != set(range(total_episodes)):
        raise ValueError("splits do not cover every episode exactly once")


def _payload_files(root: Path) -> set[str]:
    result = set()
    for directory, suffix in ((root / "data", ".parquet"), (root / "videos", ".mp4")):
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"missing payload directory {directory.name}")
        for path in directory.rglob("*"):
            if path.is_file():
                if path.suffix != suffix:
                    raise ValueError("unexpected payload file type")
                result.add(path.relative_to(root).as_posix())
    return result


def _validate_optional_metadata(
    root: Path,
    episode_count: int,
    total_frames: int,
    lengths: list[int],
    features: dict[str, Any],
    cameras: list[str],
    stats_source: Path | None,
    parquet_paths: list[Path],
    video_paths: dict[tuple[int, str], Path],
    services: _Services,
    allow_legacy_sampled_image_stats: bool,
    deep_video_stats: bool,
) -> None:
    aggregate_path = root / "meta/stats.json"
    episode_path = root / "meta/episodes_stats.jsonl"
    source_has_stats = stats_source is not None and (stats_source / "meta/stats.json").exists()
    present = (aggregate_path.exists() or aggregate_path.is_symlink(), episode_path.exists() or episode_path.is_symlink())
    if not any(present):
        raise ValueError("release stats metadata is missing")
    if not all(present):
        raise ValueError("aggregate and episode stats must be published together")

    numeric = {
        name: math.prod(value["shape"])
        for name, value in features.items()
        if isinstance(name, str) and isinstance(value, dict)
        and isinstance(value.get("dtype"), str)
        and (value["dtype"].startswith("int") or value["dtype"].startswith("float"))
        and isinstance(value.get("shape"), list)
        and value["shape"] and all(type(item) is int and item > 0 for item in value["shape"])
    }
    expected_aggregate = set(numeric) | set(cameras)
    aggregate = _read_object(aggregate_path)
    if set(aggregate) != expected_aggregate:
        raise ValueError("aggregate stats feature coverage mismatch")
    if source_has_stats:
        source_stats = _read_object(stats_source / "meta/stats.json")  # type: ignore[operator]
        if set(source_stats) != set(aggregate):
            raise ValueError("release stats keys differ from source stats")

    for feature, width in numeric.items():
        _validate_feature_stats(aggregate[feature], width, total_frames, image=False, context=f"stats {feature}")
    for camera in cameras:
        declared = features[camera]
        shape = declared.get("shape") if isinstance(declared, dict) else None
        if not isinstance(shape, list) or len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"stats camera shape metadata is invalid: {camera}")
        expected_pixels = None if allow_legacy_sampled_image_stats else total_frames * shape[0] * shape[1]
        _validate_feature_stats(
            aggregate[camera], 3, expected_pixels, image=True, context=f"stats {camera}",
        )

    rows = _read_jsonl(episode_path)
    if len(rows) != episode_count:
        raise ValueError("episode stats count mismatch")
    for index, row in enumerate(rows):
        if set(row) != {"episode_index", "stats"} or _integer(row, "episode_index", minimum=0) != index:
            raise ValueError("episode stats indices or schema are invalid")
        stats = row["stats"]
        if not isinstance(stats, dict) or set(stats) != set(numeric):
            raise ValueError("episode stats feature coverage mismatch")
        for feature, width in numeric.items():
            _validate_feature_stats(
                stats[feature], width, lengths[index], image=False,
                context=f"episode stats {index}/{feature}",
            )

    recomputed = recompute_stats(parquet_paths)
    metrics = {"min", "max", "mean", "std", "count"} if allow_legacy_sampled_image_stats else _STAT_METRICS
    for feature in numeric:
        _compare_feature_stats(
            aggregate[feature], recomputed[feature], metrics, f"stats {feature}",
            legacy_tolerance=allow_legacy_sampled_image_stats,
        )
    for index, parquet in enumerate(parquet_paths):
        actual = recompute_stats([parquet])
        for feature in numeric:
            _compare_feature_stats(
                rows[index]["stats"][feature], actual[feature], metrics, f"episode stats {index}/{feature}",
                legacy_tolerance=allow_legacy_sampled_image_stats,
            )

    if allow_legacy_sampled_image_stats:
        if stats_source is not None:
            source_rows = _read_jsonl(stats_source / "meta/episodes_stats.jsonl")
            if aggregate != _read_object(stats_source / "meta/stats.json") or rows != source_rows:
                raise ValueError("legacy release stats must exactly preserve source statistics")
    elif deep_video_stats:
        frame_iterator = services.iter_video_rgb_frames
        for camera in cameras:
            shape = features[camera]["shape"]
            paths = [video_paths[(index, camera)] for index in range(episode_count)]
            actual = recompute_video_stats(paths, lengths, shape, frame_iterator=frame_iterator)
            _compare_feature_stats(aggregate[camera], actual, _STAT_METRICS, f"stats {camera}")

    _validate_payload_sizes(root, allow_legacy_sampled_image_stats)


def _compare_feature_stats(
    published: dict[str, Any],
    actual: dict[str, Any],
    metrics: set[str],
    context: str,
    *,
    legacy_tolerance: bool = False,
) -> None:
    for metric in metrics:
        left = _flatten_numbers(published[metric])
        right = _flatten_numbers(actual[metric])
        if len(left) != len(right):
            raise ValueError(f"{context} recomputed stats shape mismatch")
        tolerance = (1e-4 if legacy_tolerance else 5e-8) if metric in {"mean", "std"} else 1e-8
        if any(abs(a - b) > tolerance * max(1.0, abs(b)) for a, b in zip(left, right, strict=True)):
            raise ValueError(f"{context} differs from recomputed stats")


def _flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_flatten_numbers(item))
        return result
    return [float(value)]


def _validate_payload_sizes(root: Path, allow_legacy: bool) -> None:
    info = _read_object(root / "meta/info.json")
    fields = ("data_files_size_in_mb", "video_files_size_in_mb")
    present = [field in info for field in fields]
    if not all(present):
        if allow_legacy and not any(present):
            return
        raise ValueError("payload size metadata is required")
    for field, directory, suffix in (
        (fields[0], root / "data", ".parquet"),
        (fields[1], root / "videos", ".mp4"),
    ):
        declared = info[field]
        if isinstance(declared, bool) or not isinstance(declared, (int, float)) or not math.isfinite(declared) or declared < 0:
            raise ValueError(f"{field} must be finite and nonnegative")
        actual = sum(path.stat(follow_symlinks=False).st_size for path in directory.rglob(f"*{suffix}")) / (2**20)
        if abs(float(declared) - actual) > max(1e-9, actual * 1e-6):
            raise ValueError(f"{field} differs from actual payload size")


def _validate_feature_stats(
    value: Any,
    width: int,
    expected_count: int | None,
    *,
    image: bool,
    context: str,
) -> int:
    if not isinstance(value, dict) or set(value) != _STAT_METRICS:
        raise ValueError(f"{context} metric coverage mismatch")
    count_value = value["count"]
    if not isinstance(count_value, list) or len(count_value) != 1 or type(count_value[0]) is not int or count_value[0] <= 0:
        raise ValueError(f"{context} count must be one positive integer")
    count = count_value[0]
    if expected_count is not None and count != expected_count:
        raise ValueError(f"{context} count differs from frame count")

    vectors: dict[str, list[float]] = {}
    for metric in _STAT_METRICS - {"count"}:
        raw = value[metric]
        if image:
            if (not isinstance(raw, list) or len(raw) != width or
                    any(not isinstance(channel, list) or len(channel) != 1 or
                        not isinstance(channel[0], list) or len(channel[0]) != 1
                        for channel in raw)):
                raise ValueError(f"{context} {metric} has wrong image stats shape")
            flat = [channel[0][0] for channel in raw]
        else:
            if not isinstance(raw, list) or len(raw) != width:
                raise ValueError(f"{context} {metric} has wrong stats shape")
            flat = raw
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in flat):
            raise ValueError(f"{context} {metric} must contain finite numbers")
        vectors[metric] = [float(item) for item in flat]

    ordered = ("min", "q01", "q10", "q50", "q90", "q99", "max")
    for component in range(width):
        values = [vectors[name][component] for name in ordered]
        tolerance = 1e-9 * max(1.0, *(abs(item) for item in values))
        if any(left > right + tolerance for left, right in zip(values, values[1:])):
            raise ValueError(f"{context} quantile ordering is invalid")
        if vectors["mean"][component] < values[0] - tolerance or vectors["mean"][component] > values[-1] + tolerance:
            raise ValueError(f"{context} mean is outside min/max")
        if vectors["std"][component] < 0:
            raise ValueError(f"{context} std must be nonnegative")
        if image and (
            any(item < -tolerance or item > 1.0 + tolerance for item in [*values, vectors["mean"][component]])
            or vectors["std"][component] > 0.5 + tolerance
        ):
            raise ValueError(f"{context} is outside normalized RGB stats range")
    return count


def _subtasks(value: Any) -> list[Subtask]:
    if not isinstance(value, list) or not value:
        raise ValueError("subtask_template must be nonempty")
    try:
        result = [Subtask.model_validate(item, strict=True) for item in value]
    except Exception as exc:
        raise ValueError("invalid subtask_template") from exc
    return result


def _aware_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field} must be UTC")
    return parsed


def _reject_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in _FORBIDDEN:
                raise ValueError(f"forbidden internal annotation field: {key}")
            _reject_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden(child)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("checksum target is not regular")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    finally:
        if fd >= 0:
            os.close(fd)
    return digest.hexdigest()

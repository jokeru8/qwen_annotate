"""Guarded, atomic publication of accepted annotation workspaces."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import stat
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import AnnotationConfig
from .constraints import validate_annotation
from .lerobot import DatasetIndex, inspect_dataset
from .release_validator import ReleaseReport, validate_release
from .stats import iter_video_rgb_frames, recompute_stats, recompute_video_stats
from .workspace import EpisodeRecord, RunManifest, WorkspaceStore, compute_run_fingerprint, compute_source_fingerprint


@dataclass(frozen=True)
class EpisodeRemap:
    source_index: int
    output_index: int
    length: int
    global_offset: int


def rewrite_episode_parquet(source: Path, destination: Path, remap: EpisodeRemap) -> None:
    """Rewrite only LeRobot index columns while preserving schema metadata and payload values."""
    if not isinstance(source, Path) or not isinstance(destination, Path) or not isinstance(remap, EpisodeRemap):
        raise TypeError("source, destination, and remap must use their declared types")
    table = pq.read_table(source)
    if table.num_rows != remap.length:
        raise ValueError("parquet row count differs from episode remap")
    replacements = {
        "episode_index": [remap.output_index] * remap.length,
        "frame_index": list(range(remap.length)),
        "index": list(range(remap.global_offset, remap.global_offset + remap.length)),
    }
    for name, values in replacements.items():
        if name not in table.schema.names:
            raise ValueError(f"parquet misses required column {name}")
        position = table.schema.get_field_index(name)
        field = table.schema.field(position)
        table = table.set_column(position, field, pa.array(values, type=field.type))
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination)


class ConversionReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    output: Path
    accepted_only: bool
    episode_count: int = Field(ge=0)
    frame_count: int = Field(ge=0)
    payload_files: list[str]
    annotation_path: str
    annotation_schema_version: str
    converted_at: datetime
    source_tree_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation: ReleaseReport

    @field_validator("converted_at")
    @classmethod
    def utc_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("converted_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def canonical_public_facts(self) -> "ConversionReport":
        if self.payload_files != sorted(set(self.payload_files)):
            raise ValueError("payload_files must be sorted and unique")
        if self.annotation_path != "meta/lerobot_annotations.json" or self.annotation_schema_version != "reference-v2.1":
            raise ValueError("unsupported public annotation schema")
        if not self.validation.valid or self.validation.episode_count != self.episode_count or self.validation.frame_count != self.frame_count:
            raise ValueError("validation report counts must match conversion")
        return self


def convert_dataset(
    work_dir: Path,
    output: Path,
    accepted_only: bool = False,
    *,
    services: Any = None,
) -> ConversionReport:
    """Copy an accepted source tree, add public annotations, validate, and publish."""
    if type(accepted_only) is not bool:
        raise TypeError("accepted_only must be a bool")
    if not isinstance(work_dir, Path) or not isinstance(output, Path):
        raise TypeError("work_dir and output must be Path objects")
    if work_dir.is_symlink():
        raise ValueError("workspace must not be a symlink")
    work = work_dir.resolve(strict=True)
    if not work.is_dir():
        raise ValueError("workspace must be a directory")
    out = output.absolute()
    _reject_existing(out)

    store = WorkspaceStore(work)
    manifest = store._load_manifest()  # secure bounded loader; conversion is a workspace peer.
    source = manifest.dataset_root.resolve(strict=True)
    if manifest.dataset_root.is_symlink() or not source.is_dir():
        raise ValueError("manifest source root is unsafe")
    if not out.parent.exists() or out.parent.is_symlink() or not out.parent.is_dir():
        raise ValueError("output parent must be an existing real directory")
    out = out.parent.resolve(strict=True) / out.name
    _reject_path_relationship(out, source, work)
    lock_path = out.parent / f".{out.name}.conversion.lock"
    lock_fd = _open_lock(lock_path)
    staging: Path | None = None
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _reject_existing(out)
        manifest, dataset, records = _guard_workspace(store, manifest, services, accepted_only)
        source_before = _tree_digest(source)
        staging = out.parent / f"{out.name}.staging-{secrets.token_hex(16)}"
        os.mkdir(staging, 0o700)
        _copy_tree(source, staging)
        converted_at = datetime.now(UTC)
        if accepted_only:
            frame_count = _rewrite_accepted_subset(
                staging, source, out, manifest, dataset, records, converted_at, services,
            )
        else:
            frame_count = manifest.total_frames
            _write_public_metadata(staging, out, manifest, records, converted_at)
        validation = validate_release(
            staging,
            source=None if accepted_only else source,
            services=services,
            _expected_output_root=out,
            _expected_stats_source=source,
        )
        if _tree_digest(source) != source_before:
            raise ValueError("source dataset changed during conversion")
        _rename_noreplace(staging, out)
        staging = None
        parent_fd = os.open(out.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        published_validation = validation.model_copy(update={"path": out.resolve()})
        return ConversionReport(
            output=out.resolve(), accepted_only=accepted_only, episode_count=len(records),
            frame_count=frame_count, payload_files=validation.payload_files,
            annotation_path="meta/lerobot_annotations.json", annotation_schema_version="reference-v2.1",
            converted_at=converted_at, source_tree_digest=source_before,
            validation=published_validation,
        )
    finally:
        primary = None
        if staging is not None:
            try:
                _remove_owned_staging(staging, out.parent, out.name)
            except Exception as cleanup_error:
                primary = cleanup_error
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
        # A cleanup error is surfaced only when there was no active primary exception.
        if primary is not None and sys.exc_info()[0] is None:
            raise primary


def _guard_workspace(
    store: WorkspaceStore, manifest: RunManifest, services: Any, accepted_only: bool,
):
    summary = store.summary()
    if summary["total"] != manifest.total_episodes:
        raise ValueError("workspace episode count differs from manifest")
    if not accepted_only and summary["counts"]["accepted"] != manifest.total_episodes:
        raise ValueError("full conversion requires every episode to be accepted")
    effective = manifest.effective_config
    primary = effective.get("primary_camera")
    refine = effective.get("refine_cameras")
    if not isinstance(primary, str) or primary not in manifest.camera_keys or not isinstance(refine, list):
        raise ValueError("manifest camera provenance is invalid")
    config_payload = json.loads(json.dumps(effective))
    model = config_payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("manifest model provenance is invalid")
    model.setdefault("api_key", "local")
    config = AnnotationConfig.model_validate(config_payload)
    if (config.source.resolve() != manifest.dataset_root.resolve() or config.work_dir.resolve() != store.root or
            config.mode != manifest.mode or config.high_level_instruction != manifest.high_level_instruction or
            config.subtasks != manifest.subtasks or config.sampling.min_segment_frames != manifest.min_segment_frames or
            config.model.name != manifest.model_repo or
            compute_run_fingerprint(config, manifest.model_revision) != manifest.run_fingerprint):
        raise ValueError("manifest run/model provenance is invalid")
    probe = _service(services, "probe_video", None)
    dataset = inspect_dataset(config) if probe is None else inspect_dataset(config, probe=probe)
    if (dataset.root != manifest.dataset_root.resolve() or dataset.version != manifest.dataset_version or
            dataset.fps != manifest.fps or dataset.camera_keys != manifest.camera_keys or
            [item.length for item in dataset.episodes] != manifest.episode_lengths or
            len(dataset.episodes) != manifest.total_episodes or
            sum(item.length for item in dataset.episodes) != manifest.total_frames):
        raise ValueError("current source facts differ from workspace manifest")
    if any(episode.task != manifest.high_level_instruction for episode in dataset.episodes):
        raise ValueError("source task instruction differs from workspace manifest")
    records = []
    for episode in dataset.episodes:
        record = store.load_episode(episode.episode_index)
        if record.run_fingerprint != manifest.run_fingerprint:
            raise ValueError("episode run provenance differs from manifest")
        if record.source_fingerprint != compute_source_fingerprint(dataset.root, episode):
            raise ValueError("episode source fingerprint is stale")
        if record.status != "accepted":
            if accepted_only:
                continue
            raise ValueError(f"episode {episode.episode_index} is not accepted")
        if record.final_annotation is None or record.decision_source not in ("human", "model"):
            raise ValueError(f"episode {episode.episode_index} is not accepted")
        issues = validate_annotation(record.final_annotation, manifest.mode, len(manifest.subtasks), episode.length, manifest.min_segment_frames)
        if issues:
            raise ValueError(f"episode {episode.episode_index} has invalid final annotation")
        if record.decision_source == "model" and (record.prompt_version != manifest.prompt_version or record.model_revision != manifest.model_revision):
            raise ValueError("model decision provenance differs from manifest")
        records.append(record)
    if not records:
        raise ValueError("accepted-only conversion requires at least one accepted episode")
    return manifest, dataset, records


def _write_public_metadata(staging: Path, output: Path, manifest: RunManifest, records: list, converted_at: datetime) -> None:
    selected = [
        (record, record.episode_index, manifest.episode_lengths[record.episode_index])
        for record in records
    ]
    _write_selection_metadata(staging, output, manifest, selected, converted_at)


def _write_selection_metadata(
    staging: Path,
    output: Path,
    manifest: RunManifest,
    selected: list[tuple[EpisodeRecord, int, int]],
    converted_at: datetime,
) -> None:
    info_path = staging / "meta/info.json"
    info = _read_source_json(info_path)
    template = [item.model_dump(mode="json") for item in manifest.subtasks]
    instruction_map = {str(index): manifest.high_level_instruction for index in range(len(selected))}
    info["subtask_template"] = template
    info["high_level_instruction"] = instruction_map
    _atomic_json(info_path, info)
    episodes = {}
    task_info = []
    for record, output_index, length in selected:
        annotation = record.final_annotation
        entry = {"episode_index": output_index}
        if manifest.mode == "dagger_patch":
            entry["start_subtask_index"] = annotation.start_subtask_index
        entry.update({
            "boundaries": list(annotation.boundaries),
            "high_level_instruction": manifest.high_level_instruction,
            "saved_at": record.updated_at.astimezone(UTC).isoformat(),
        })
        episodes[str(output_index)] = entry
        starts = [0, *annotation.boundaries]
        ends = [*annotation.boundaries, length]
        selected_subtasks = manifest.subtasks[
            annotation.start_subtask_index:annotation.start_subtask_index + len(starts)
        ]
        actions = [
            {
                "start_frame": start,
                "end_frame": end,
                "action_text": subtask.text,
                "skill": subtask.skill,
            }
            for start, end, subtask in zip(starts, ends, selected_subtasks, strict=True)
        ]
        task_info.append({
            "episode_id": output_index,
            "task_id": 0,
            "task_name": manifest.high_level_instruction,
            "label_info": {"action_config": actions},
        })
    annotations = {
        "source_root": str(manifest.dataset_root),
        "work_dir": str(output / "meta"),
        "subtask_template": template,
        "episodes": episodes,
        "primary_camera": manifest.effective_config["primary_camera"],
        "updated_at": converted_at.isoformat(),
    }
    _atomic_json(staging / "meta/lerobot_annotations.json", annotations, sort_keys=False)
    task_dir = staging / "meta/task_info"
    task_dir.mkdir(exist_ok=True)
    _atomic_json(task_dir / "task_0.json", task_info, sort_keys=False)


def _rewrite_accepted_subset(
    staging: Path,
    source: Path,
    output: Path,
    manifest: RunManifest,
    dataset: DatasetIndex,
    records: list[EpisodeRecord],
    converted_at: datetime,
    services: Any,
) -> int:
    info_path = staging / "meta/info.json"
    info = _read_source_json(info_path)
    chunks_size = info.get("chunks_size")
    if type(chunks_size) is not int or chunks_size < 1:
        raise ValueError("source chunks_size must be a positive integer")
    data_template = info.get("data_path")
    video_template = info.get("video_path")
    if not isinstance(data_template, str) or not isinstance(video_template, str):
        raise ValueError("source payload templates must be strings")

    episode_by_index = {episode.episode_index: episode for episode in dataset.episodes}
    selected_records = sorted(records, key=lambda record: record.episode_index)
    remaps: list[EpisodeRemap] = []
    offset = 0
    for output_index, record in enumerate(selected_records):
        episode = episode_by_index[record.episode_index]
        remaps.append(EpisodeRemap(record.episode_index, output_index, episode.length, offset))
        offset += episode.length

    _clear_payload_directory(staging / "data", staging)
    _clear_payload_directory(staging / "videos", staging)
    rewritten_parquets: list[Path] = []
    rewritten_videos: dict[str, list[Path]] = {camera: [] for camera in manifest.camera_keys}
    for remap in remaps:
        episode = episode_by_index[remap.source_index]
        values = {
            "episode_chunk": remap.output_index // chunks_size,
            "episode_index": remap.output_index,
        }
        parquet_relative = _render_payload_path(data_template, values, "data_path", "data")
        parquet_destination = staging / parquet_relative
        rewrite_episode_parquet(episode.parquet, parquet_destination, remap)
        rewritten_parquets.append(parquet_destination)
        for camera in manifest.camera_keys:
            video_relative = _render_payload_path(
                video_template, values | {"video_key": camera}, "video_path", "videos",
            )
            destination = staging / video_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(episode.videos[camera], destination, follow_symlinks=False)
            rewritten_videos[camera].append(destination)

    source_episode_rows = _read_source_jsonl(source / "meta/episodes.jsonl")
    rows_by_index = {row.get("episode_index"): row for row in source_episode_rows}
    output_episode_rows = []
    for remap in remaps:
        row = rows_by_index.get(remap.source_index)
        if not isinstance(row, dict):
            raise ValueError("source episodes metadata misses selected episode")
        output_episode_rows.append(row | {"episode_index": remap.output_index, "length": remap.length})
    _atomic_jsonl(staging / "meta/episodes.jsonl", output_episode_rows)

    aggregate_stats = recompute_stats(rewritten_parquets)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("source features must be an object")
    frame_iterator = _service(services, "iter_video_rgb_frames", iter_video_rgb_frames)
    if not callable(frame_iterator):
        raise TypeError("iter_video_rgb_frames service must be callable")
    expected_lengths = [remap.length for remap in remaps]
    for camera in manifest.camera_keys:
        feature = features.get(camera)
        if not isinstance(feature, dict) or feature.get("dtype") != "video" or not isinstance(feature.get("shape"), list):
            raise ValueError(f"camera feature metadata is invalid: {camera}")
        aggregate_stats[camera] = recompute_video_stats(
            rewritten_videos[camera], expected_lengths, feature["shape"], frame_iterator=frame_iterator,
        )
    source_stats_path = source / "meta/stats.json"
    if source_stats_path.exists() or source_stats_path.is_symlink():
        source_stats = _read_source_json(source_stats_path)
        if set(source_stats) != set(aggregate_stats):
            raise ValueError("source stats keys differ from declared selected feature coverage")
    _atomic_json(staging / "meta/stats.json", aggregate_stats, sort_keys=False)
    episode_stats = [
        {"episode_index": remap.output_index, "stats": recompute_stats([parquet])}
        for remap, parquet in zip(remaps, rewritten_parquets, strict=True)
    ]
    _atomic_jsonl(staging / "meta/episodes_stats.jsonl", episode_stats)

    count = len(remaps)
    info.update({
        "total_episodes": count,
        "total_frames": offset,
        "total_videos": count * len(manifest.camera_keys),
        "total_chunks": (count + chunks_size - 1) // chunks_size,
        "splits": {"train": f"0:{count}"},
        "data_files_size_in_mb": _payload_size_mb(staging / "data", ".parquet"),
        "video_files_size_in_mb": _payload_size_mb(staging / "videos", ".mp4"),
    })
    _atomic_json(info_path, info)
    selected = [
        (record, remap.output_index, remap.length)
        for record, remap in zip(selected_records, remaps, strict=True)
    ]
    _write_selection_metadata(staging, output, manifest, selected, converted_at)
    return offset


def _copy_tree(source: Path, staging: Path) -> None:
    source_stat = source.stat(follow_symlinks=False)
    for current, dirs, files in os.walk(source, topdown=True, followlinks=False):
        current_path = Path(current)
        relative = current_path.relative_to(source)
        destination = staging / relative
        for directory in dirs:
            src = current_path / directory
            before = src.stat(follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode) or src.is_symlink():
                raise ValueError(f"source contains unsafe directory: {src}")
            (destination / directory).mkdir(mode=stat.S_IMODE(before.st_mode))
        for filename in files:
            src = current_path / filename
            before = src.stat(follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode) or src.is_symlink():
                raise ValueError(f"source contains unsafe file: {src}")
            dst = destination / filename
            shutil.copy2(src, dst, follow_symlinks=False)
            after = src.stat(follow_symlinks=False)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError(f"source changed while copying: {src}")
            if not stat.S_ISREG(dst.stat(follow_symlinks=False).st_mode):
                raise ValueError("copy did not create a regular file")
        if relative != Path("."):
            shutil.copystat(current_path, destination, follow_symlinks=False)
    shutil.copystat(source, staging, follow_symlinks=False)
    if source.stat(follow_symlinks=False).st_ino != source_stat.st_ino:
        raise ValueError("source root changed while copying")


def _clear_payload_directory(path: Path, staging: Path) -> None:
    if path.parent != staging or path.name not in {"data", "videos"} or path.is_symlink() or not path.is_dir():
        raise ValueError("copied payload directory is unsafe")
    shutil.rmtree(path)
    path.mkdir()


def _render_payload_path(
    template: str,
    values: dict[str, Any],
    label: str,
    required_root: str,
) -> Path:
    try:
        relative = Path(template.format(**values))
    except Exception as exc:
        raise ValueError(f"invalid {label} template") from exc
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != required_root:
        raise ValueError(f"{label} path escapes its payload directory")
    return relative


def _payload_size_mb(directory: Path, suffix: str) -> float:
    return sum(
        path.stat(follow_symlinks=False).st_size
        for path in directory.rglob(f"*{suffix}")
        if path.is_file() and not path.is_symlink()
    ) / (2**20)


def _read_source_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("unsafe info.json")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key}")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                           parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(constant)))
    except Exception as exc:
        raise ValueError("invalid source info.json") from exc
    if not isinstance(value, dict):
        raise ValueError("info.json must be an object")
    return value


def _read_source_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"unsafe {path.name}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank {path.name} row {number}")
        try:
            value = json.loads(
                line,
                object_pairs_hook=lambda pairs: _unique_pairs(pairs, path.name),
                parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(constant)),
            )
        except Exception as exc:
            raise ValueError(f"invalid {path.name} row {number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path.name} rows must be objects")
        rows.append(value)
    return rows


def _unique_pairs(pairs: list[tuple[str, Any]], context: str) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key} in {context}")
        result[key] = value
    return result


def _atomic_json(path: Path, value: Any, *, sort_keys: bool = True) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"), allow_nan=False) + "\n").encode()
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    name = f".{path.name}.{secrets.token_hex(12)}.tmp"
    fd = None
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory_fd)
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        name = ""
        os.fsync(directory_fd)
    finally:
        if fd is not None:
            os.close(fd)
        if name:
            try: os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError: pass
        os.close(directory_fd)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    encoded = b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
        for row in rows
    )
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    name = f".{path.name}.{secrets.token_hex(12)}.tmp"
    fd = None
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory_fd)
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        name = ""
        os.fsync(directory_fd)
    finally:
        if fd is not None:
            os.close(fd)
        if name:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        metadata = path.stat(follow_symlinks=False)
        if path.is_symlink() or not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
            raise ValueError("source contains unsafe file type")
        kind = "d" if stat.S_ISDIR(metadata.st_mode) else "f"
        digest.update(f"{relative}\0{kind}\0{stat.S_IMODE(metadata.st_mode)}\0{metadata.st_size}\0{metadata.st_mtime_ns}\0".encode())
        if stat.S_ISREG(metadata.st_mode):
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        digest.update(b"\n")
    return digest.hexdigest()


def _reject_existing(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise FileExistsError(f"output already exists: {path}")


def _reject_path_relationship(output: Path, source: Path, work: Path) -> None:
    for protected, name in ((source, "source"), (work, "workspace")):
        try:
            output.relative_to(protected)
        except ValueError:
            pass
        else:
            raise ValueError(f"output must not equal or be nested in {name}")
        try:
            protected.relative_to(output)
        except ValueError:
            pass
        else:
            raise ValueError(f"output must not contain {name}")


def _open_lock(path: Path) -> int:
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError as exc:
        raise ValueError("conversion lock is unsafe") from exc
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("conversion lock must be regular")
    return fd


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory publication requires Linux renameat2")
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"output already exists: {destination}")
        raise OSError(error, os.strerror(error), destination)


def _remove_owned_staging(staging: Path, parent: Path, output_name: str) -> None:
    if staging.parent != parent or not staging.name.startswith(f"{output_name}.staging-"):
        raise ValueError("refusing to clean an unowned path")
    if staging.exists() and not staging.is_symlink():
        shutil.rmtree(staging)


def _service(services: Any, name: str, default: Any) -> Any:
    if services is None:
        return default
    return services.get(name, default) if isinstance(services, dict) else getattr(services, name, default)

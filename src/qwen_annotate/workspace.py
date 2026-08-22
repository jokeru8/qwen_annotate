"""Durable, resumable annotation workspace state.

Source dataset files are fingerprinted in place and are never copied or written.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator, model_validator

from .config import AnnotationConfig, Subtask
from .constraints import validate_annotation
from .lerobot import DatasetIndex, EpisodeInfo
from .models import CoarseResult, FinalAnnotation, RefineResult, ValidationIssue
from .prompts import PROMPT_VERSION


Status = Literal["pending", "coarse_done", "refine_done", "accepted", "needs_review", "failed"]
DecisionSource = Literal["model", "human"]
_STATUSES = ("pending", "coarse_done", "refine_done", "accepted", "needs_review", "failed")
_NON_ACCEPTED = ("pending", "coarse_done", "refine_done", "needs_review", "failed")
_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EPISODE_NAME = re.compile(r"episode_([0-9]{6})\.json\Z")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.RLock] = {}


def _utc_now() -> datetime:
    return datetime.now(UTC)


NonemptyString = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(ge=1)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EpisodeRecord(_StrictModel):
    """All persisted state and concise audit evidence for one episode."""

    episode_index: int = Field(ge=0)
    status: Status = "pending"
    coarse_attempts: list[CoarseResult] = Field(default_factory=list)
    refine_attempts: list[RefineResult] = Field(default_factory=list)
    final_annotation: FinalAnnotation | None = None
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    review_reasons: list[NonemptyString] = Field(default_factory=list)
    failure_category: NonemptyString | None = None
    decision_source: DecisionSource | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    source_fingerprint: NonemptyString
    run_fingerprint: NonemptyString
    prompt_version: NonemptyString | None = None
    model_revision: NonemptyString | None = None
    sampling_details: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("created_at", "updated_at")
    @classmethod
    def timestamp_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("sampling_details")
    @classmethod
    def sampling_values_are_finite(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if not _json_numbers_are_finite(value):
            raise ValueError("sampling_details must contain finite numbers")
        return value

    @field_validator("source_fingerprint", "run_fingerprint")
    @classmethod
    def fingerprints_are_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("fingerprint must be an exact lowercase SHA-256")
        return value

    @field_validator("model_revision")
    @classmethod
    def optional_revision_is_exact(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_revision(value)
        return value

    @model_validator(mode="after")
    def state_fields_are_consistent(self) -> "EpisodeRecord":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        if self.status == "pending":
            derived = (
                self.coarse_attempts
                or self.refine_attempts
                or self.final_annotation is not None
                or self.validation_issues
                or self.review_reasons
                or self.failure_category is not None
                or self.decision_source is not None
                or self.prompt_version is not None
                or self.model_revision is not None
                or self.sampling_details
            )
            if derived:
                raise ValueError("pending records must not contain derived results or provenance")
        if self.status == "coarse_done" and not self.coarse_attempts:
            raise ValueError("coarse_done requires at least one coarse attempt")
        if self.status == "refine_done":
            if not self.coarse_attempts:
                raise ValueError("refine_done requires coarse attempts")
            zero_transition = (
                self.final_annotation is not None
                and self.final_annotation.boundaries == []
            )
            if not self.refine_attempts and not zero_transition:
                raise ValueError(
                    "refine_done requires refine attempts unless its final annotation has no boundaries"
                )
        if self.status == "accepted":
            if self.final_annotation is None:
                raise ValueError("accepted requires a final annotation")
            if self.decision_source is None:
                raise ValueError("accepted requires decision_source")
            if self.validation_issues or self.review_reasons:
                raise ValueError("accepted records cannot retain unresolved review issues")
            if self.decision_source == "model" and (
                self.prompt_version != PROMPT_VERSION
                or self.model_revision is None
                or not self.sampling_details
            ):
                raise ValueError(
                    "model acceptance requires exact prompt_version, model_revision, and sampling provenance"
                )
        elif self.decision_source is not None:
            raise ValueError("decision_source is only valid for accepted records")
        if self.status == "needs_review" and not (self.review_reasons or self.validation_issues):
            raise ValueError("needs_review requires reasons or validation issues")
        if self.status == "failed" and self.failure_category is None:
            raise ValueError("failed requires failure_category")
        if self.status != "failed" and self.failure_category is not None:
            raise ValueError("failure_category is only valid for failed records")
        return self


class RunManifest(_StrictModel):
    dataset_root: Path
    dataset_version: Literal["v2.1"]
    fps: float = Field(gt=0)
    camera_keys: list[NonemptyString]
    total_episodes: int = Field(ge=0)
    total_frames: int = Field(ge=0)
    episode_lengths: list[PositiveInt]
    mode: Literal["complete", "dagger_patch"]
    high_level_instruction: NonemptyString
    subtasks: list[Subtask] = Field(min_length=1)
    code_version: NonemptyString
    prompt_version: NonemptyString
    model_repo: NonemptyString
    model_revision: NonemptyString
    effective_config: dict[str, JsonValue]
    min_segment_frames: int = Field(ge=1)
    run_fingerprint: NonemptyString
    created_at: datetime

    @field_validator("fps")
    @classmethod
    def fps_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fps must be finite")
        return value

    @field_validator("model_revision")
    @classmethod
    def revision_is_exact(cls, value: str) -> str:
        _validate_revision(value)
        return value

    @field_validator("run_fingerprint")
    @classmethod
    def run_hash_is_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("run_fingerprint must be lowercase SHA-256")
        return value

    @field_validator("created_at")
    @classmethod
    def creation_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def dataset_counts_and_config_are_serializable(self) -> "RunManifest":
        if len(self.episode_lengths) != self.total_episodes:
            raise ValueError("episode_lengths must match total_episodes")
        if sum(self.episode_lengths) != self.total_frames:
            raise ValueError("episode_lengths must sum to total_frames")
        if len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("camera_keys must be unique")
        try:
            json.dumps(self.effective_config, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("effective_config must be JSON serializable") from exc
        return self


def compute_source_fingerprint(dataset_root: Path, episode: EpisodeInfo) -> str:
    """Hash source metadata needed to determine whether cached evidence is stale."""
    root = _absolute_root(dataset_root)
    parquet, parquet_relative = _contained_file(root, episode.parquet, "parquet")
    parquet_stat = parquet.stat()
    videos: list[dict[str, object]] = []
    for camera, path in sorted(episode.videos.items()):
        video, relative = _contained_file(root, path, f"video {camera!r}")
        stat = video.stat()
        videos.append(
            {
                "camera": camera,
                "path": relative.as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = {
        "episode_length": episode.length,
        "parquet": {"path": parquet_relative.as_posix(), "size": parquet_stat.st_size},
        "videos": videos,
    }
    return _canonical_sha256(payload)


def compute_run_fingerprint(config: AnnotationConfig, model_revision: str) -> str:
    """Hash every immutable input that can change annotation behavior."""
    _validate_revision(model_revision)
    return _canonical_sha256(
        {
            "config_hash": config.stable_hash(),
            "prompt_version": PROMPT_VERSION,
            "model_repo": config.model.name,
            "model_revision": model_revision,
        }
    )


class WorkspaceStore:
    """Atomic persistence facade for one annotation workspace."""

    def __init__(self, root: Path, *, clock: Callable[[], datetime] = _utc_now) -> None:
        if not isinstance(root, Path):
            raise TypeError("workspace root must be a Path")
        self.root = root.resolve()
        self._clock = clock
        key = str(self.root)
        with _THREAD_LOCKS_GUARD:
            self._thread_lock = _THREAD_LOCKS.setdefault(key, threading.RLock())

    def create_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for relative in ("episodes", "previews", "previews/needs_review", "logs"):
            path = self.root / relative
            if path.is_symlink():
                raise ValueError(f"workspace layout component must not be a symlink: {path}")
            if path.exists() and not path.is_dir():
                raise ValueError(f"workspace layout path is not a directory: {path}")
            path.mkdir(exist_ok=True)
        self._assert_safe_layout()

    def initialize(
        self,
        config: AnnotationConfig,
        dataset: DatasetIndex,
        model_revision: str,
        *,
        code_version: str = "0.1.0",
    ) -> RunManifest:
        """Create or safely resume a compatible workspace without touching source data."""
        dataset_root = _absolute_root(dataset.root)
        if dataset_root != config.source.resolve():
            raise ValueError("dataset index root is incompatible with config source")
        try:
            self.root.relative_to(dataset_root)
        except ValueError:
            pass
        else:
            raise ValueError("workspace root must not equal or be nested beneath the dataset root")
        expected_indices = {episode.episode_index for episode in dataset.episodes}
        if expected_indices != set(range(len(dataset.episodes))):
            raise ValueError("dataset episode indices must be contiguous")
        source_fingerprints = {
            episode.episode_index: compute_source_fingerprint(dataset_root, episode)
            for episode in dataset.episodes
        }
        run_hash = compute_run_fingerprint(config, model_revision)
        prevalidated_manifest = self._manifest(
            config,
            dataset,
            model_revision,
            code_version,
            run_hash,
            self._now(),
        )
        self.create_layout()
        with self._locked():
            self._assert_safe_layout()
            manifest_path = self.root / "manifest.json"
            if _safe_entry_exists(manifest_path):
                manifest = self._load_manifest()
                candidate = self._manifest(config, dataset, model_revision, code_version, run_hash, manifest.created_at)
                if manifest != candidate:
                    raise ValueError("existing workspace manifest is incompatible with this run")
            else:
                if any((self.root / "episodes").iterdir()):
                    raise ValueError("workspace has episode state but no manifest")
                manifest = prevalidated_manifest
                _atomic_json_write(manifest_path, manifest.model_dump(mode="json"))

            for episode in dataset.episodes:
                expected_source = source_fingerprints[episode.episode_index]
                path = self._episode_path(episode.episode_index)
                if _safe_entry_exists(path):
                    current = self._load_episode_unlocked(episode.episode_index)
                    if current.source_fingerprint != expected_source or current.run_fingerprint != run_hash:
                        raise ValueError(
                            f"episode {episode.episode_index} cache is incompatible; explicitly invalidate it"
                        )
                    self._validate_final_annotation(current)
                else:
                    record = EpisodeRecord(
                        episode_index=episode.episode_index,
                        source_fingerprint=expected_source,
                        run_fingerprint=run_hash,
                        created_at=manifest.created_at,
                        updated_at=manifest.created_at,
                    )
                    _atomic_json_write(path, record.model_dump(mode="json"))
            self._write_summary_unlocked()
            return manifest

    def load_episode(self, index: int) -> EpisodeRecord:
        index = _validate_index(index)
        with self._locked():
            self._assert_safe_layout()
            return self._load_episode_unlocked(index)

    def save_episode(self, record: EpisodeRecord) -> None:
        """Atomically save a legal transition, then refresh the recoverable summary.

        If refreshing summary fails, the authoritative episode remains committed;
        calling :meth:`write_summary` reconstructs the derived summary.
        """
        record = EpisodeRecord.model_validate(record.model_dump())
        self.create_layout()
        with self._locked():
            self._assert_safe_layout()
            self._validate_manifest_index(record.episode_index)
            path = self._episode_path(record.episode_index)
            manifest_exists = _safe_entry_exists(self.root / "manifest.json")
            episode_exists = _safe_entry_exists(path)
            if manifest_exists and not episode_exists:
                raise ValueError(
                    f"workspace is corrupt: manifest episode {record.episode_index} is missing"
                )
            if episode_exists:
                prior = self._load_episode_unlocked(record.episode_index)
                self._validate_transition(prior, record)
                if prior == record:
                    return
            elif record.status != "pending":
                raise ValueError("a new standalone episode record must start pending")
            self._validate_final_annotation(record)
            _atomic_json_write(path, record.model_dump(mode="json"))
            self._write_summary_unlocked()

    def invalidate_episode(
        self,
        index: int,
        *,
        episode: EpisodeInfo | None = None,
        source_fingerprint: str | None = None,
        run_fingerprint: str | None = None,
    ) -> EpisodeRecord:
        """Explicitly discard all derived state and reset one record to pending."""
        index = _validate_index(index)
        with self._locked():
            self._assert_safe_layout()
            prior = self._load_episode_unlocked(index)
            if _safe_entry_exists(self.root / "manifest.json"):
                manifest = self._load_manifest()
                if index >= manifest.total_episodes:
                    raise ValueError(
                        f"episode index {index} is outside manifest range [0, {manifest.total_episodes})"
                    )
                if episode is None:
                    raise ValueError("manifest invalidation requires current EpisodeInfo")
                if source_fingerprint is not None:
                    raise ValueError("caller-provided source_fingerprint is not allowed with a manifest")
                if run_fingerprint is not None and run_fingerprint != manifest.run_fingerprint:
                    raise ValueError("run_fingerprint must equal manifest.run_fingerprint")
                if episode.episode_index != index:
                    raise ValueError("EpisodeInfo index must match the invalidated episode")
                if episode.length != manifest.episode_lengths[index]:
                    raise ValueError("EpisodeInfo length must match the manifest")
                source_hash = compute_source_fingerprint(manifest.dataset_root, episode)
                run_hash = manifest.run_fingerprint
            else:
                if source_fingerprint is None or run_fingerprint is None:
                    raise ValueError(
                        "standalone invalidation requires source_fingerprint and run_fingerprint"
                    )
                source_hash = source_fingerprint
                run_hash = run_fingerprint
            reset = EpisodeRecord(
                episode_index=index,
                source_fingerprint=source_hash,
                run_fingerprint=run_hash,
                created_at=prior.created_at,
                updated_at=self._now_strictly_after(prior.updated_at),
            )
            _atomic_json_write(self._episode_path(index), reset.model_dump(mode="json"))
            self._write_summary_unlocked()
            return reset

    @staticmethod
    def cache_is_valid(
        record: EpisodeRecord,
        *,
        source_fingerprint: str,
        run_fingerprint: str,
    ) -> bool:
        return (
            record.source_fingerprint == source_fingerprint
            and record.run_fingerprint == run_fingerprint
        )

    def summary(self) -> dict[str, object]:
        with self._locked():
            self._assert_safe_layout()
            return self._summary_unlocked()

    def write_summary(self) -> dict[str, object]:
        with self._locked():
            self._assert_safe_layout()
            return self._write_summary_unlocked()

    def _manifest(
        self,
        config: AnnotationConfig,
        dataset: DatasetIndex,
        revision: str,
        code_version: str,
        run_hash: str,
        created_at: datetime,
    ) -> RunManifest:
        effective = config.model_dump(mode="json", exclude={"model": {"api_key"}})
        effective["model"]["endpoint"] = _redacted_endpoint(str(config.model.endpoint))
        return RunManifest(
            dataset_root=dataset.root.resolve(),
            dataset_version=dataset.version,
            fps=dataset.fps,
            camera_keys=list(dataset.camera_keys),
            total_episodes=len(dataset.episodes),
            total_frames=sum(item.length for item in dataset.episodes),
            episode_lengths=[item.length for item in dataset.episodes],
            mode=config.mode,
            high_level_instruction=config.high_level_instruction,
            subtasks=list(config.subtasks),
            code_version=code_version,
            prompt_version=PROMPT_VERSION,
            model_repo=config.model.name,
            model_revision=revision,
            effective_config=effective,
            min_segment_frames=config.sampling.min_segment_frames,
            run_fingerprint=run_hash,
            created_at=created_at,
        )

    def _load_manifest(self) -> RunManifest:
        path = self.root / "manifest.json"
        _reject_symlink_path(path)
        payload = _read_json_object(path)
        return RunManifest.model_validate_json(_canonical_json(payload))

    def _load_episode_unlocked(self, index: int) -> EpisodeRecord:
        path = self._episode_path(index)
        _reject_symlink_path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing episode record: {path}")
        try:
            record = EpisodeRecord.model_validate_json(_canonical_json(_read_json_object(path)))
        except (ValidationError, ValueError) as exc:
            raise ValueError(f"Invalid episode record {path.name}: {exc}") from exc
        if record.episode_index != index:
            raise ValueError(f"Invalid episode record {path.name}: episode_index does not match filename")
        return record

    def _validate_manifest_index(self, index: int) -> None:
        manifest_path = self.root / "manifest.json"
        if _safe_entry_exists(manifest_path):
            total = self._load_manifest().total_episodes
            if index >= total:
                raise ValueError(f"episode index {index} is outside manifest range [0, {total})")

    @staticmethod
    def _validate_transition(prior: EpisodeRecord, current: EpisodeRecord) -> None:
        if prior.episode_index != current.episode_index:
            raise ValueError("episode identity cannot change")
        if (
            prior.source_fingerprint != current.source_fingerprint
            or prior.run_fingerprint != current.run_fingerprint
        ):
            raise ValueError("episode fingerprints cannot change without explicit invalidation")
        if prior.created_at != current.created_at:
            raise ValueError("episode creation time cannot change")
        if prior.status == current.status:
            if prior != current:
                raise ValueError(f"non-identical same-status update is forbidden for {prior.status}")
            return
        if current.updated_at <= prior.updated_at:
            raise ValueError("updated_at must strictly increase for every state change")
        if prior.status == "needs_review" and current.status == "accepted" and current.decision_source != "human":
            raise ValueError("needs_review may only transition to a human decision")
        allowed: dict[str, set[str]] = {
            "pending": {"coarse_done", "needs_review", "failed"},
            "coarse_done": {"refine_done", "needs_review", "failed"},
            "refine_done": {"accepted", "needs_review", "failed"},
            "needs_review": {"accepted", "failed"},
            "accepted": set(),
            "failed": set(),
        }
        if current.status not in allowed[prior.status]:
            raise ValueError(f"illegal transition {prior.status} -> {current.status}")

    def _validate_final_annotation(self, record: EpisodeRecord) -> None:
        validate_zero_transition = record.status == "refine_done" and not record.refine_attempts
        if (
            (record.status != "accepted" and not validate_zero_transition)
            or not _safe_entry_exists(self.root / "manifest.json")
        ):
            return
        manifest = self._load_manifest()
        if (
            record.status == "accepted"
            and record.decision_source == "model"
            and record.model_revision != manifest.model_revision
        ):
            raise ValueError("accepted model_revision must match the workspace manifest")
        issues = validate_annotation(
            record.final_annotation,
            manifest.mode,
            len(manifest.subtasks),
            manifest.episode_lengths[record.episode_index],
            manifest.min_segment_frames,
        )
        if issues:
            codes = ", ".join(issue.code for issue in issues)
            raise ValueError(f"invalid final annotation for episode {record.episode_index}: {codes}")

    def _summary_unlocked(self) -> dict[str, object]:
        episodes_dir = self.root / "episodes"
        if not episodes_dir.is_dir():
            raise ValueError("workspace episodes directory is missing")
        records: dict[int, EpisodeRecord] = {}
        for path in episodes_dir.iterdir():
            match = _EPISODE_NAME.fullmatch(path.name)
            if match is None or not path.is_file() or path.is_symlink():
                raise ValueError(f"invalid episode filename or file type: {path.name}")
            index = int(match.group(1))
            if index in records:
                raise ValueError(f"duplicate episode index {index}")
            records[index] = self._load_episode_unlocked(index)
        manifest_path = self.root / "manifest.json"
        if _safe_entry_exists(manifest_path):
            total = self._load_manifest().total_episodes
            expected = set(range(total))
            actual = set(records)
            if actual != expected:
                extra = sorted(actual - expected)
                missing = sorted(expected - actual)
                raise ValueError(f"episode files outside manifest range or missing: extra={extra}, missing={missing}")
        counts = {status: 0 for status in _STATUSES}
        indices = {status: [] for status in _NON_ACCEPTED}
        for index, record in sorted(records.items()):
            counts[record.status] += 1
            if record.status != "accepted":
                indices[record.status].append(index)
        return {"total": len(records), "counts": counts, "episode_indices": indices}

    def _write_summary_unlocked(self) -> dict[str, object]:
        summary = self._summary_unlocked()
        _atomic_json_write(self.root / "summary.json", summary)
        return summary

    def _episode_path(self, index: int) -> Path:
        return self.root / "episodes" / f"episode_{_validate_index(index):06d}.json"

    def _now(self, minimum: datetime | None = None) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("workspace clock must return a timezone-aware datetime")
        value = value.astimezone(UTC)
        return max(value, minimum) if minimum is not None else value

    def _now_strictly_after(self, prior: datetime) -> datetime:
        return max(self._now(), prior + timedelta(microseconds=1))

    def _assert_safe_layout(self) -> None:
        for relative in ("episodes", "previews", "previews/needs_review", "logs"):
            path = self.root / relative
            if path.is_symlink():
                raise ValueError(f"workspace layout component must not be a symlink: {path}")
            if not path.is_dir():
                raise ValueError(f"workspace layout component is not a directory: {path}")

    def _locked(self):
        return _WorkspaceLock(self.root / "logs" / "workspace.lock", self._thread_lock)


class _WorkspaceLock:
    def __init__(self, path: Path, thread_lock: threading.RLock) -> None:
        self._path = path
        self._thread_lock = thread_lock
        self._handle = None

    def __enter__(self) -> "_WorkspaceLock":
        self._thread_lock.acquire()
        try:
            if self._path.parent.is_symlink() or not self._path.parent.is_dir():
                raise ValueError("workspace lock parent is a symlink or unsafe directory")
            flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
            parent_flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                parent_fd = os.open(self._path.parent, parent_flags)
                try:
                    fd = os.open(self._path.name, flags, 0o600, dir_fd=parent_fd)
                finally:
                    os.close(parent_fd)
            except OSError as exc:
                raise ValueError("workspace lock must not be a symlink or unsafe file") from exc
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                raise ValueError("workspace lock must be a regular file, not a symlink")
            self._handle = os.fdopen(fd, "a+b")
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except BaseException:
            if self._handle is not None and not self._handle.closed:
                self._handle.close()
            self._thread_lock.release()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if self._handle is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
                self._handle.close()
        finally:
            self._thread_lock.release()


def _atomic_json_write(path: Path, value: Mapping[str, object] | dict[str, object]) -> None:
    data = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    directory_fd: int | None = None
    temporary_name: str | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            directory_fd = os.open(path.parent, flags)
        except OSError as exc:
            raise ValueError(f"JSON parent is a symlink or unsafe directory: {path.parent}") from exc
        _reject_symlink_entry(directory_fd, path.name)
        temporary_name = f".{path.name}.{secrets.token_hex(12)}.tmp"
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _reject_symlink_entry(directory_fd, path.name)
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = None
        os.fsync(directory_fd)
    finally:
        if temporary_name is not None and directory_fd is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        if directory_fd is not None:
            os.close(directory_fd)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        _reject_symlink_path(path)
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            os.close(fd)
            raise ValueError("JSON path must be a regular file")
        size = file_stat.st_size
        if size > _MAX_JSON_BYTES:
            os.close(fd)
            raise ValueError(f"JSON file exceeds {_MAX_JSON_BYTES} bytes")
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            text = handle.read(_MAX_JSON_BYTES + 1)
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(f"non-finite value {constant}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Malformed JSON in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Malformed JSON in {path.name}: expected object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _json_numbers_are_finite(value: JsonValue) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_numbers_are_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_json_numbers_are_finite(item) for item in value.values())
    return True


def _safe_entry_exists(path: Path) -> bool:
    """Return existence without ever accepting a symlink entry."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"workspace path must not be a symlink: {path}")
    return True


def _reject_symlink_path(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"workspace path must not be a symlink: {path}")


def _reject_symlink_entry(directory_fd: int, name: str) -> None:
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"JSON target must not be a symlink: {name}")


def _redacted_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    hostname = parsed.hostname
    if not parsed.scheme or hostname is None:
        raise ValueError("model endpoint must contain a scheme and host")
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host if parsed.port is None else f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _contained_file(root: Path, path: Path, label: str) -> tuple[Path, Path]:
    candidate = path.resolve(strict=False)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes dataset root: {path}") from exc
    if not candidate.exists():
        raise FileNotFoundError(f"missing {label} file: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"{label} path is not a file: {candidate}")
    return candidate, relative


def _absolute_root(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("dataset root must be a Path")
    root = path.resolve(strict=False)
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    return root


def _canonical_sha256(value: Mapping[str, object]) -> str:
    canonical = _canonical_json(value)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_revision(value: str) -> None:
    if not isinstance(value, str) or not _SHA40.fullmatch(value):
        raise ValueError("model revision must be an exact lowercase 40-character SHA")


def _validate_index(index: int) -> int:
    if type(index) is not int or index < 0 or index > 999999:
        raise ValueError("episode index must be a non-negative integer at most 999999")
    return index

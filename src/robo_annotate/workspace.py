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


class LegacyWorkspaceError(ValueError):
    """A pre-layered workspace is intentionally read-only under the current schema."""


class ConcurrentWorkspaceUpdate(ValueError):
    """Authoritative episode state changed after a caller prepared its update."""


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
            audited_zero_transition_review = _is_audited_zero_transition_review(self)
            if not self.refine_attempts and not zero_transition and not audited_zero_transition_review:
                raise ValueError(
                    "refine_done requires refine attempts unless its audited annotation has no boundaries"
                )
        if self.status == "accepted":
            if self.final_annotation is None:
                raise ValueError("accepted requires a final annotation")
            if self.decision_source is None:
                raise ValueError("accepted requires decision_source")
            if self.validation_issues or self.review_reasons:
                raise ValueError("accepted records cannot retain unresolved review issues")
            if self.decision_source == "model" and (
                self.prompt_version is None
                or self.model_revision is None
                or not self.sampling_details
            ):
                raise ValueError(
                    "model acceptance requires prompt_version, model_revision, and sampling provenance"
                )
        elif self.decision_source is not None:
            raise ValueError("decision_source is only valid for accepted records")
        if self.status == "needs_review" and not (self.review_reasons or self.validation_issues):
            raise ValueError("needs_review requires reasons or validation issues")
        if self.status == "failed" and self.failure_category is None:
            raise ValueError("failed requires failure_category")
        if self.status != "failed" and self.failure_category is not None:
            raise ValueError("failure_category is only valid for failed records")
        _validate_pipeline_transition_outbox(self)
        return self


def _is_audited_zero_transition_review(record: EpisodeRecord) -> bool:
    """Recognize the typed no-model-call refine review needed for short singleton episodes."""
    coarse_payload = record.sampling_details.get("coarse_decision")
    refine_payload = record.sampling_details.get("refine_decision")
    if not isinstance(coarse_payload, dict) or not isinstance(refine_payload, dict):
        return False
    try:
        from .coarse import CoarseDecision
        from .refine import RefineDecision

        coarse = CoarseDecision.model_validate_json(_canonical_json(coarse_payload))
        refined = RefineDecision.model_validate_json(_canonical_json(refine_payload))
    except Exception:
        return False
    candidate = refined.candidate_annotation
    persisted_attempts = [item.model_dump(mode="json") for item in record.coarse_attempts]
    audited_attempts = [item.model_dump(mode="json") for item in coarse.attempts]
    return (
        coarse.status == "coarse_done"
        and refined.status == "needs_review"
        and not refined.attempts
        and record.final_annotation is None
        and candidate is not None
        and tuple(candidate.boundaries) == ()
        and refined.annotation is None
        and refined.start_subtask_index == coarse.start_subtask_index
        and refined.observed_subtask_indices == coarse.observed_subtask_indices
        and refined.coarse_boundary_centers == coarse.boundary_centers == ()
        and persisted_attempts == audited_attempts
    )


def _validate_pipeline_transition_outbox(record: EpisodeRecord) -> None:
    key = "_pipeline_transition_events"
    if key not in record.sampling_details:
        return
    raw = record.sampling_details[key]
    if not isinstance(raw, list):
        raise ValueError("pipeline transition outbox must be a list")
    fields = {"event_id", "timestamp", "episode", "from_status", "to_status", "event", "category", "reasons"}
    statuses = {"pending", "coarse_done", "refine_done", "accepted", "needs_review", "failed"}
    event_names = {
        "coarse_review", "coarse_completed", "refine_completed", "accepted",
        "refine_review", "error", "human_takeover", "human_corrected",
    }
    categories = {None, "invalid_model_response", "model_oom", "model_call", "source_or_video", "unexpected_error", "workspace_state"}
    parsed: list[tuple[dict[str, Any], datetime]] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("pipeline transition event has invalid fields")
        if (
            not isinstance(item["event_id"], str) or not _SHA256.fullmatch(item["event_id"])
            or type(item["episode"]) is not int or item["episode"] != record.episode_index
            or not isinstance(item["from_status"], str) or item["from_status"] not in statuses
            or not isinstance(item["to_status"], str) or item["to_status"] not in statuses
            or not isinstance(item["event"], str) or item["event"] not in event_names
            or (item["category"] is not None and not isinstance(item["category"], str))
            or item["category"] not in categories
            or not isinstance(item["reasons"], list)
            or any(not isinstance(reason, str) or not reason for reason in item["reasons"])
            or len(set(item["reasons"])) != len(item["reasons"])
            or not isinstance(item["timestamp"], str)
        ):
            raise ValueError("pipeline transition event has invalid values")
        try:
            timestamp = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("pipeline transition timestamp is invalid") from None
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("pipeline transition timestamp must be timezone-aware")
        timestamp = timestamp.astimezone(UTC)
        payload = {
            "run_fingerprint": record.run_fingerprint,
            "episode": item["episode"],
            "from_status": item["from_status"],
            "to_status": item["to_status"],
            "updated_at": timestamp.isoformat(),
            "event": item["event"],
            "category": item["category"],
            "reasons": item["reasons"],
        }
        expected = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
        if item["event_id"] != expected:
            raise ValueError("pipeline transition event_id is invalid")
        parsed.append((item, timestamp))
    if len({item["event_id"] for item, _ in parsed}) != len(parsed):
        raise ValueError("pipeline transition outbox has duplicate event ids")
    for (prior, prior_time), (current, current_time) in zip(parsed, parsed[1:]):
        if prior["to_status"] != current["from_status"] or prior_time >= current_time:
            raise ValueError("pipeline transition outbox chain is invalid")
    if parsed and (parsed[0][0]["from_status"] != "pending" or parsed[-1][0]["to_status"] != record.status):
        raise ValueError("pipeline transition outbox does not match record status")


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

    def load_manifest(self) -> RunManifest:
        """Securely load the authoritative typed workspace manifest."""
        with self._locked():
            self._assert_safe_layout()
            return self._load_manifest()

    def load_manifest_with_provenance(self) -> tuple[RunManifest, AnnotationConfig]:
        """Load a manifest and verify its immutable run/prompt/config provenance.

        The run hash is recomputed from the untouched persisted lexical paths.
        Relative paths have no stable resolution base in the v1 manifest, so they
        are used only for hashing; the returned runtime config is anchored to the
        authoritative absolute manifest dataset and this workspace root.
        """
        with self._locked():
            self._assert_safe_layout()
            manifest = self._load_manifest()
            raw = json.loads(_canonical_json(manifest.effective_config))
            if not isinstance(raw, dict):
                raise ValueError("manifest effective_config must be an object")
            model = raw.get("model")
            if not isinstance(model, dict):
                raise ValueError("manifest model provenance is invalid")
            model["api_key"] = "workspace-local-redacted"
            persisted = AnnotationConfig.model_validate_json(_canonical_json(raw), strict=True)
            if manifest.prompt_version != PROMPT_VERSION:
                raise ValueError("manifest prompt version does not match the supported prompt contract")
            if (
                (persisted.source.is_absolute()
                 and persisted.source.resolve() != manifest.dataset_root.resolve())
                or (persisted.work_dir.is_absolute() and persisted.work_dir.resolve() != self.root)
                or persisted.mode != manifest.mode
                or persisted.high_level_instruction != manifest.high_level_instruction
                or persisted.subtasks != manifest.subtasks
                or persisted.sampling.min_segment_frames != manifest.min_segment_frames
                or persisted.model.name != manifest.model_repo
                or compute_run_fingerprint(persisted, manifest.model_revision) != manifest.run_fingerprint
            ):
                raise ValueError("manifest run/prompt/config provenance is invalid")

            runtime_raw = json.loads(_canonical_json(raw))
            runtime_raw["source"] = str(manifest.dataset_root)
            runtime_raw["work_dir"] = str(self.root)
            return manifest, AnnotationConfig.model_validate_json(
                _canonical_json(runtime_raw), strict=True
            )

    def validate_record_provenance(
        self,
        record: EpisodeRecord,
        *,
        manifest: RunManifest | None = None,
        episode: EpisodeInfo | None = None,
    ) -> None:
        """Fail closed when an episode belongs to another run/source/model context."""
        record = EpisodeRecord.model_validate(record.model_dump())
        with self._locked():
            self._assert_safe_layout()
            authoritative = self._load_manifest()
            if manifest is not None and manifest != authoritative:
                raise ValueError("supplied manifest is not the authoritative workspace manifest")
            manifest = authoritative
            if record.episode_index >= manifest.total_episodes:
                raise ValueError("episode identity is outside the workspace manifest")
            if record.run_fingerprint != manifest.run_fingerprint:
                raise ValueError("episode run fingerprint does not match workspace manifest")
            if record.status == "accepted" and record.decision_source == "model":
                if record.prompt_version != manifest.prompt_version:
                    raise ValueError("model prompt version does not match workspace manifest")
                if record.model_revision != manifest.model_revision:
                    raise ValueError("model revision does not match workspace manifest")
            self._validate_final_annotation(record)
            if episode is not None:
                if (
                    episode.episode_index != record.episode_index
                    or episode.length != manifest.episode_lengths[record.episode_index]
                ):
                    raise ValueError("source episode identity does not match workspace manifest")
                if record.source_fingerprint != compute_source_fingerprint(
                    manifest.dataset_root, episode
                ):
                    raise ValueError(f"episode {record.episode_index} source fingerprint is stale")

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
                    self._validate_final_annotation(record)
                    self._write_summary_unlocked()
                    return
            elif record.status != "pending":
                raise ValueError("a new standalone episode record must start pending")
            self._validate_final_annotation(record)
            _atomic_json_write(path, record.model_dump(mode="json"))
            self._write_summary_unlocked()

    def save_episode_transactional(
        self,
        record: EpisodeRecord,
        *,
        expected_prior: EpisodeRecord | None = None,
    ) -> None:
        """Save a transition and roll its authoritative file back if summary refresh fails.

        Human decisions use this stronger all-or-error contract because an operator must
        be able to safely correct the underlying failure and reapply the same decision.
        """
        record = EpisodeRecord.model_validate(record.model_dump())
        self.create_layout()
        with self._locked():
            self._assert_safe_layout()
            self._validate_manifest_index(record.episode_index)
            path = self._episode_path(record.episode_index)
            if not _safe_entry_exists(path):
                raise ValueError("transactional save requires an existing episode record")
            prior = self._load_episode_unlocked(record.episode_index)
            if expected_prior is not None and prior != expected_prior:
                raise ConcurrentWorkspaceUpdate("episode changed during transactional save")
            self._validate_transition(prior, record)
            self._validate_final_annotation(record)
            if prior == record:
                self._write_summary_unlocked()
                return

            directory_fd = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                rollback_fd = os.open(
                    self.root / "logs",
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                )
            except BaseException:
                os.close(directory_fd)
                raise
            backup = f".{path.name}.{secrets.token_hex(12)}.rollback"
            try:
                source_fd = os.open(
                    path.name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
                try:
                    backup_fd = os.open(
                        backup,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                        dir_fd=rollback_fd,
                    )
                    try:
                        while chunk := os.read(source_fd, 1024 * 1024):
                            view = memoryview(chunk)
                            while view:
                                written = os.write(backup_fd, view)
                                if written <= 0:
                                    raise OSError("rollback backup write made no progress")
                                view = view[written:]
                        os.fsync(backup_fd)
                    finally:
                        os.close(backup_fd)
                finally:
                    os.close(source_fd)
                try:
                    _atomic_json_write(path, record.model_dump(mode="json"))
                    self._write_summary_unlocked()
                except BaseException:
                    os.replace(backup, path.name, src_dir_fd=rollback_fd, dst_dir_fd=directory_fd)
                    backup = ""
                    os.fsync(directory_fd)
                    try:
                        self._write_summary_unlocked()
                    except BaseException:
                        pass
                    raise
            finally:
                if backup:
                    try:
                        os.unlink(backup, dir_fd=rollback_fd)
                    except FileNotFoundError:
                        pass
                os.close(rollback_fd)
                os.close(directory_fd)

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
        if not config.augmentation.enabled and config.augmentation.language == "English":
            effective.pop("augmentation")
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
            payload = _read_json_object(path)
            attempts = payload.get("coarse_attempts")
            if (
                payload.get("prompt_version") == "coarse-v4/refine-v1"
                and isinstance(attempts, list)
                and any(isinstance(attempt, dict) and "uncertainties" in attempt for attempt in attempts)
            ):
                raise LegacyWorkspaceError(
                    "legacy coarse-v4 record uses removed uncertainties; preserve the original "
                    "JSON for manual audit and create a new workspace"
                )
            record = EpisodeRecord.model_validate_json(_canonical_json(payload))
        except LegacyWorkspaceError:
            raise
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
            if (
                prior.status == "accepted"
                and current.updated_at > prior.updated_at
                and current.decision_source == "human"
                and WorkspaceStore._has_current_human_audit(current, prior.status)
            ):
                return
            if prior != current:
                raise ValueError(f"non-identical same-status update is forbidden for {prior.status}")
            return
        if current.updated_at <= prior.updated_at:
            raise ValueError("updated_at must strictly increase for every state change")
        if prior.status == "needs_review" and current.status == "accepted" and current.decision_source != "human":
            raise ValueError("needs_review may only transition to a human decision")
        allowed: dict[str, set[str]] = {
            "pending": {"coarse_done", "needs_review", "failed", "accepted"},
            "coarse_done": {"refine_done", "needs_review", "failed"},
            "refine_done": {"accepted", "needs_review", "failed"},
            "needs_review": {"accepted", "failed"},
            "accepted": set(),
            "failed": {"accepted"},
        }
        if current.status not in allowed[prior.status]:
            raise ValueError(f"illegal transition {prior.status} -> {current.status}")
        if prior.status in {"pending", "failed"} and current.status == "accepted":
            if current.decision_source != "human" or not WorkspaceStore._has_current_human_audit(
                current, prior.status
            ):
                raise ValueError(
                    "direct accepted transition requires a current human takeover audit"
                )

    @staticmethod
    def _has_current_human_audit(record: EpisodeRecord, prior_status: str) -> bool:
        audits = record.sampling_details.get("human_decisions")
        if not isinstance(audits, list) or not audits or record.final_annotation is None:
            return False
        latest = audits[-1]
        return bool(
            isinstance(latest, dict)
            and latest.get("prior_status") == prior_status
            and latest.get("takeover_confirmed") is True
            and latest.get("accepted_annotation") == record.final_annotation.model_dump(mode="json")
            and latest.get("timestamp") == record.updated_at.isoformat()
        )

    def _validate_final_annotation(self, record: EpisodeRecord) -> None:
        validate_zero_transition = (
            record.status == "refine_done"
            and not record.refine_attempts
            and record.final_annotation is not None
        )
        if (
            (record.status != "accepted" and not validate_zero_transition)
            or not _safe_entry_exists(self.root / "manifest.json")
        ):
            return
        manifest = self._load_manifest()
        if (
            record.status == "accepted"
            and record.decision_source == "model"
            and (
                record.prompt_version != manifest.prompt_version
                or record.model_revision != manifest.model_revision
            )
        ):
            raise ValueError(
                "accepted prompt_version and model_revision must match the workspace manifest"
            )
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
        raw_fd: int | None = None
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
                    raw_fd = os.open(self._path.name, flags, 0o600, dir_fd=parent_fd)
                finally:
                    os.close(parent_fd)
            except OSError as exc:
                raise ValueError("workspace lock must not be a symlink or unsafe file") from exc
            if not stat.S_ISREG(os.fstat(raw_fd).st_mode):
                raise ValueError("workspace lock must be a regular file, not a symlink")
            self._handle = os.fdopen(raw_fd, "a+b")
            raw_fd = None
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except BaseException:
            try:
                if raw_fd is not None:
                    os.close(raw_fd)
                if self._handle is not None and not self._handle.closed:
                    self._handle.close()
            finally:
                self._thread_lock.release()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if self._handle is not None:
                try:
                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
                finally:
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
    temporary_fd: int | None = None
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
        handle = os.fdopen(temporary_fd, "w", encoding="utf-8")
        temporary_fd = None
        with handle:
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
        try:
            if temporary_fd is not None:
                os.close(temporary_fd)
        finally:
            try:
                if temporary_name is not None and directory_fd is not None:
                    try:
                        os.unlink(temporary_name, dir_fd=directory_fd)
                    except FileNotFoundError:
                        pass
            finally:
                if directory_fd is not None:
                    os.close(directory_fd)


def _read_json_object(path: Path) -> dict[str, Any]:
    fd: int | None = None
    try:
        _reject_symlink_path(path)
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("JSON path must be a regular file")
        size = file_stat.st_size
        if size > _MAX_JSON_BYTES:
            raise ValueError(f"JSON file exceeds {_MAX_JSON_BYTES} bytes")
        handle = os.fdopen(fd, "r", encoding="utf-8")
        fd = None
        with handle:
            text = handle.read(_MAX_JSON_BYTES + 1)
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(f"non-finite value {constant}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Malformed JSON in {path.name}: {exc}") from exc
    finally:
        if fd is not None:
            os.close(fd)
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

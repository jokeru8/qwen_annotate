"""Resumable, bounded orchestration for annotation workspaces."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .coarse import CoarseDecision, run_coarse
from .config import AnnotationConfig
from .constraints import ANNOTATION_VALIDATION_ISSUE_CODES
from .lerobot import DatasetIndex, EpisodeInfo, inspect_dataset
from .model_manager import ModelInstall
from .models import CoarseResult, FinalAnnotation, RefineResult, ValidationIssue
from .prompts import PROMPT_VERSION
from .qwen_client import InvalidModelResponse, ModelCallError, ModelOutOfMemory, QwenClient
from .refine import RefineDecision, run_refine
from .workspace import EpisodeRecord, Status, WorkspaceStore


_STATUSES = ("pending", "coarse_done", "refine_done", "accepted", "needs_review", "failed")
_NONACCEPTED = ("pending", "coarse_done", "refine_done", "needs_review", "failed")
_SHA40 = frozenset("0123456789abcdef")
_OUTBOX_KEY = "_pipeline_transition_events"
_MAX_LOG_BYTES = 64 * 1024 * 1024
_MAX_LOG_LINE_BYTES = 64 * 1024
_MAX_MODEL_METADATA_BYTES = 64 * 1024


class AuditPersistenceError(RuntimeError):
    """Workspace state is durable but its derived audit log could not be synchronized."""


class _SourceOrVideoError(Exception):
    pass


class TransitionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    event_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    timestamp: datetime
    episode: int = Field(ge=0)
    from_status: Status
    to_status: Status
    event: Literal["coarse_review", "coarse_completed", "refine_completed", "accepted", "refine_review", "error"]
    category: Literal[
        "invalid_model_response", "model_oom", "model_call", "source_or_video",
        "unexpected_error", "workspace_state",
    ] | None = None
    reasons: tuple[str, ...] = ()

    @field_validator("timestamp")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        return _utc(value)

    @field_validator("reasons", mode="before")
    @classmethod
    def restore_reasons(cls, value: object, info) -> object:
        if info.mode == "json" and isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("reasons")
    @classmethod
    def strict_reasons(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or any(not item for item in value):
            raise ValueError("transition reasons must be unique nonempty strings")
        return value


class WorkspaceSummary(BaseModel):
    """Strict immutable projection of the authoritative workspace summary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    total: int = Field(ge=0)
    pending: int = Field(ge=0)
    coarse_done: int = Field(ge=0)
    refine_done: int = Field(ge=0)
    accepted: int = Field(ge=0)
    needs_review: int = Field(ge=0)
    failed: int = Field(ge=0)
    pending_episode_indices: tuple[int, ...] = ()
    coarse_done_episode_indices: tuple[int, ...] = ()
    refine_done_episode_indices: tuple[int, ...] = ()
    needs_review_episode_indices: tuple[int, ...] = ()
    failed_episode_indices: tuple[int, ...] = ()

    @model_validator(mode="after")
    def internally_consistent(self) -> "WorkspaceSummary":
        if sum(getattr(self, status) for status in _STATUSES) != self.total:
            raise ValueError("summary status counts must sum to total")
        for status in _NONACCEPTED:
            indices = getattr(self, f"{status}_episode_indices")
            if indices != tuple(sorted(indices)) or len(set(indices)) != len(indices):
                raise ValueError("summary episode lists must be sorted and unique")
            if len(indices) != getattr(self, status):
                raise ValueError("summary episode lists must match status counts")
            if any(type(index) is not int or not 0 <= index < self.total for index in indices):
                raise ValueError("summary episode index is outside range")
        all_nonaccepted = [i for status in _NONACCEPTED for i in getattr(self, f"{status}_episode_indices")]
        if len(set(all_nonaccepted)) != len(all_nonaccepted):
            raise ValueError("summary episode lists must be disjoint")
        return self

    @property
    def nonaccepted_episode_indices(self) -> tuple[int, ...]:
        return tuple(sorted(i for status in _NONACCEPTED for i in getattr(self, f"{status}_episode_indices")))

    @classmethod
    def from_store_summary(cls, value: object) -> "WorkspaceSummary":
        if not isinstance(value, dict) or set(value) != {"total", "counts", "episode_indices"}:
            raise ValueError("workspace summary has invalid top-level fields")
        counts, indices = value["counts"], value["episode_indices"]
        if not isinstance(counts, dict) or set(counts) != set(_STATUSES):
            raise ValueError("workspace summary has invalid counts")
        if not isinstance(indices, dict) or set(indices) != set(_NONACCEPTED):
            raise ValueError("workspace summary has invalid episode lists")
        payload = {"total": value["total"], **counts}
        for status in _NONACCEPTED:
            raw = indices[status]
            if not isinstance(raw, list):
                raise ValueError("workspace summary episode lists must be lists")
            payload[f"{status}_episode_indices"] = tuple(raw)
        return cls.model_validate(payload)


class _Store(Protocol):
    root: Path
    def initialize(self, config: AnnotationConfig, dataset: DatasetIndex, model_revision: str, *, code_version: str = "0.1.0") -> Any: ...
    def load_episode(self, index: int) -> EpisodeRecord: ...
    def save_episode(self, record: EpisodeRecord) -> None: ...
    def summary(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class PipelineServices:
    inspect_dataset: Callable[[AnnotationConfig], DatasetIndex] = inspect_dataset
    workspace_factory: Callable[[Path], _Store] = WorkspaceStore
    resolve_model: Callable[[AnnotationConfig], ModelInstall] = lambda config: _installed_model(config)
    client_factory: Callable[[AnnotationConfig], Any] = lambda config: QwenClient(
        endpoint=str(config.model.endpoint), api_key=config.model.api_key, model=config.model.name
    )
    run_coarse: Callable[..., Awaitable[CoarseDecision]] = run_coarse
    run_refine: Callable[..., Awaitable[RefineDecision]] = run_refine
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    close_client: bool = False


class _RunLog:
    def __init__(self, work_dir: Path, clock: Callable[[], datetime]) -> None:
        self.path = work_dir / "logs" / "run.jsonl"
        self.clock = clock
        self.lock = asyncio.Lock()

    async def sync(self, store: _Store) -> None:
        async with self.lock:
            self._sync_locked(store)

    def _sync_locked(self, store: _Store) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.parent / "run-log.lock"
        lock_fd = _open_regular(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            expected = _workspace_events(store)
            rendered = _render_log(expected)
            directory_fd = os.open(
                self.path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                current = _read_official_log(directory_fd, self.path.name)
                if current != rendered:
                    _replace_log_atomically(directory_fd, self.path.name, rendered)
            finally:
                os.close(directory_fd)
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)


async def annotate_dataset(
    config: AnnotationConfig,
    max_concurrency: int,
    episode_indices: Sequence[int] | None = None,
    *,
    services: PipelineServices | None = None,
) -> WorkspaceSummary:
    """Annotate selected resumable records while keeping at most N episodes active."""
    if not isinstance(config, AnnotationConfig):
        raise TypeError("config must be an AnnotationConfig")
    if type(max_concurrency) is not int:
        raise TypeError("max_concurrency must be an integer")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    service = services or PipelineServices()
    if not isinstance(service, PipelineServices):
        raise TypeError("services must be PipelineServices")
    dataset = service.inspect_dataset(config)
    if not isinstance(dataset, DatasetIndex):
        raise TypeError("inspect_dataset must return DatasetIndex")
    if any(episode.episode_index != position for position, episode in enumerate(dataset.episodes)):
        raise ValueError("dataset episodes must be ordered by contiguous episode indices")
    selected = _episode_selection(episode_indices, len(dataset.episodes))
    install = service.resolve_model(config)
    _validate_install(config, install)
    store = service.workspace_factory(config.work_dir)
    store.initialize(config, dataset, install.revision)
    logger = _RunLog(config.work_dir.resolve(), service.clock)
    try:
        await logger.sync(store)
    except Exception as exc:
        raise AuditPersistenceError("could not synchronize durable transition audit") from exc
    semaphore = asyncio.Semaphore(max_concurrency)
    stop_requested = asyncio.Event()

    resumable = []
    for index in selected:
        if store.load_episode(index).status in {"pending", "coarse_done", "refine_done"}:
            resumable.append(index)
    client = service.client_factory(config)

    async def process(index: int) -> BaseException | None:
        try:
            async with semaphore:
                if stop_requested.is_set():
                    return None
                await _process_episode(
                    config, dataset, dataset.episodes[index], store, install.revision,
                    client, service, logger, stop_requested,
                )
        except BaseException as exc:
            stop_requested.set()
            return exc
        return None

    iterator = iter(resumable)
    active: set[asyncio.Task[BaseException | None]] = set()
    task_indices: dict[asyncio.Task[BaseException | None], int] = {}
    interrupted: BaseException | None = None

    def launch() -> bool:
        try:
            index = next(iterator)
        except StopIteration:
            return False
        task = asyncio.create_task(process(index))
        active.add(task)
        task_indices[task] = index
        return True

    primary: BaseException | None = None
    result: WorkspaceSummary | None = None
    try:
        for _ in range(min(max_concurrency, len(resumable))):
            launch()
        while active:
            done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            for task in sorted(done, key=lambda item: task_indices[item]):
                active.remove(task)
                outcome = task.result()
                if outcome is not None and interrupted is None:
                    interrupted = outcome
                    stop_requested.set()
            if interrupted is None:
                while len(active) < max_concurrency and launch():
                    pass
        if interrupted is not None:
            raise interrupted
        result = WorkspaceSummary.from_store_summary(store.summary())
    except BaseException as exc:
        primary = exc
        stop_requested.set()
        if active:
            await asyncio.shield(asyncio.gather(*active, return_exceptions=True))
    finally:
        if services is None or service.close_client:
            close = getattr(client, "aclose", None)
            if callable(close):
                try:
                    await close()
                except BaseException:
                    if primary is None:
                        raise
    if primary is not None:
        raise primary
    assert result is not None
    return result


async def _process_episode(config, dataset, episode, store, revision, client, services, logger, stop_requested) -> None:
    record = store.load_episode(episode.episode_index)
    coarse: CoarseDecision | None = None
    try:
        if record.status == "pending":
            try:
                coarse = await services.run_coarse(
                    config, episode, client=client, source_fps=float(dataset.fps),
                    expected_source_fingerprint=record.source_fingerprint,
                )
            except (FileNotFoundError, OSError, ValueError) as exc:
                raise _SourceOrVideoError from exc
            if not isinstance(coarse, CoarseDecision):
                raise TypeError("run_coarse must return CoarseDecision")
            try:
                _validate_coarse_context(coarse, config, episode)
            except ValueError:
                raise InvalidModelResponse("coarse decision context mismatch", attempt_count=1) from None
            details = dict(record.sampling_details)
            details["coarse_decision"] = coarse.model_dump(mode="json")
            if coarse.status == "needs_review":
                updated = _replace(record, services.clock, status="needs_review", coarse_attempts=_coarse_attempts(coarse),
                    review_reasons=list(coarse.reasons), prompt_version=PROMPT_VERSION, model_revision=revision, sampling_details=details)
                await _save_transition(store, logger, record, updated, "coarse_review", reasons=coarse.reasons)
                return
            updated = _replace(record, services.clock, status="coarse_done", coarse_attempts=_coarse_attempts(coarse),
                prompt_version=PROMPT_VERSION, model_revision=revision, sampling_details=details)
            record = await _save_transition(store, logger, record, updated, "coarse_completed")
            if stop_requested.is_set():
                return
        if record.status == "coarse_done":
            if coarse is None:
                try:
                    coarse = CoarseDecision.model_validate_json(
                        json.dumps(record.sampling_details["coarse_decision"], allow_nan=False)
                    )
                    _validate_coarse_context(coarse, config, episode)
                except Exception:
                    await _fail(store, logger, record, services.clock, "workspace_state")
                    return
            try:
                refined = await services.run_refine(
                    config, episode, coarse, client=client, source_fps=float(dataset.fps),
                    expected_source_fingerprint=record.source_fingerprint,
                )
            except (FileNotFoundError, OSError, ValueError) as exc:
                raise _SourceOrVideoError from exc
            if not isinstance(refined, RefineDecision):
                raise TypeError("run_refine must return RefineDecision")
            try:
                _validate_refine_context(refined, coarse, config, dataset, episode)
            except ValueError:
                raise InvalidModelResponse("refine decision context mismatch", attempt_count=1) from None
            details = dict(record.sampling_details)
            details["refine_decision"] = refined.model_dump(mode="json")
            if refined.status == "failed":
                await _fail(store, logger, record, services.clock, refined.failure_category or "model_oom", details=details)
                return
            updated = _replace(record, services.clock, status="refine_done", refine_attempts=_refine_attempts(refined),
                final_annotation=_annotation(refined.annotation), sampling_details=details)
            record = await _save_transition(store, logger, record, updated, "refine_completed")
            if stop_requested.is_set():
                return
        if record.status == "refine_done":
            try:
                coarse = CoarseDecision.model_validate_json(
                    json.dumps(record.sampling_details["coarse_decision"], allow_nan=False)
                )
                _validate_coarse_context(coarse, config, episode)
                refined = RefineDecision.model_validate_json(
                    json.dumps(record.sampling_details["refine_decision"], allow_nan=False)
                )
                _validate_refine_context(refined, coarse, config, dataset, episode)
            except Exception:
                await _fail(store, logger, record, services.clock, "workspace_state")
                return
            if refined.status == "accepted":
                updated = _replace(record, services.clock, status="accepted", final_annotation=_annotation(refined.annotation),
                    validation_issues=[], review_reasons=[], decision_source="model")
                await _save_transition(store, logger, record, updated, "accepted")
            elif refined.status == "needs_review":
                issues = [
                    ValidationIssue(code=reason, message=f"deterministic review reason: {reason}")
                    for reason in refined.reasons if reason in ANNOTATION_VALIDATION_ISSUE_CODES
                ]
                updated = _replace(record, services.clock, status="needs_review", validation_issues=issues,
                    review_reasons=list(refined.reasons), decision_source=None)
                await _save_transition(store, logger, record, updated, "refine_review", reasons=refined.reasons)
            else:
                await _fail(store, logger, record, services.clock, refined.failure_category or "model_oom")
    except AuditPersistenceError:
        raise
    except InvalidModelResponse:
        current = store.load_episode(episode.episode_index)
        updated = _replace(current, services.clock, status="needs_review", review_reasons=["invalid_model_response"])
        await _save_transition(store, logger, current, updated, "error", category="invalid_model_response", reasons=("invalid_model_response",))
    except ModelOutOfMemory:
        await _fail(store, logger, store.load_episode(episode.episode_index), services.clock, "model_oom")
    except ModelCallError:
        await _fail(store, logger, store.load_episode(episode.episode_index), services.clock, "model_call")
    except _SourceOrVideoError:
        await _fail(store, logger, store.load_episode(episode.episode_index), services.clock, "source_or_video")
    except ValueError:
        raise
    except (KeyboardInterrupt, asyncio.CancelledError):
        raise
    except Exception:
        await _fail(store, logger, store.load_episode(episode.episode_index), services.clock, "unexpected_error")


async def _save_transition(store, logger, old, new, event, *, category=None, reasons=()) -> EpisodeRecord:
    transition = _transition_event(old, new, event, category, reasons)
    details = dict(new.sampling_details)
    prior_events = _record_events(old)
    details[_OUTBOX_KEY] = [
        *[item.model_dump(mode="json") for item in prior_events],
        transition.model_dump(mode="json"),
    ]
    persisted = EpisodeRecord.model_validate(new.model_dump() | {"sampling_details": details})
    try:
        store.save_episode(persisted)
    except (OSError, ValueError) as exc:
        raise AuditPersistenceError("could not atomically persist transition state") from exc
    try:
        await logger.sync(store)
    except Exception as exc:
        raise AuditPersistenceError("state saved but transition log synchronization failed") from exc
    return persisted


async def _fail(store, logger, record, clock, category, *, details=None) -> None:
    updated = _replace(record, clock, status="failed", failure_category=category,
        sampling_details=details if details is not None else record.sampling_details)
    await _save_transition(store, logger, record, updated, "error", category=category)


def _replace(record: EpisodeRecord, clock, **changes) -> EpisodeRecord:
    now = _utc(clock())
    changes["updated_at"] = max(now, record.updated_at + timedelta(microseconds=1))
    return record.model_copy(update=changes)


def _coarse_attempts(decision: CoarseDecision) -> list[CoarseResult]:
    return [CoarseResult.model_validate(item.model_dump(mode="json")) for item in decision.attempts]


def _refine_attempts(decision: RefineDecision) -> list[RefineResult]:
    return [RefineResult.model_validate(item.model_dump(mode="json")) for item in decision.attempts]


def _annotation(value: Any) -> FinalAnnotation | None:
    return None if value is None else FinalAnnotation.model_validate(value.model_dump(mode="json"))


def _validate_coarse_context(decision: CoarseDecision, config: AnnotationConfig, episode: EpisodeInfo) -> None:
    if (
        decision.mode != config.mode
        or decision.subtask_count != len(config.subtasks)
        or decision.frame_count != episode.length
    ):
        raise ValueError("coarse decision does not match immutable run context")


def _validate_refine_context(
    decision: RefineDecision,
    coarse: CoarseDecision,
    config: AnnotationConfig,
    dataset: DatasetIndex,
    episode: EpisodeInfo,
) -> None:
    cameras = tuple(dict.fromkeys([config.primary_camera, *config.refine_cameras]))
    expected = (
        decision.mode == config.mode,
        decision.subtask_count == len(config.subtasks),
        decision.frame_count == episode.length,
        decision.min_segment_frames == config.sampling.min_segment_frames,
        decision.agreement_tolerance_frames == config.sampling.agreement_tolerance_frames,
        decision.source_fps == float(dataset.fps),
        decision.refine_window_seconds == float(config.sampling.refine_window_seconds),
        decision.refine_fps == float(config.sampling.refine_fps),
        decision.dense_radius_seconds == float(config.sampling.dense_radius_seconds),
        decision.camera_order == cameras,
        decision.start_subtask_index == coarse.start_subtask_index,
        decision.observed_subtask_indices == coarse.observed_subtask_indices,
        decision.coarse_boundary_centers == coarse.boundary_centers,
    )
    if not all(expected):
        raise ValueError("refine decision does not match immutable run context")


def _transition_event(
    old: EpisodeRecord,
    new: EpisodeRecord,
    event: str,
    category: str | None,
    reasons: Sequence[str],
) -> TransitionEvent:
    payload = {
        "run_fingerprint": new.run_fingerprint,
        "episode": new.episode_index,
        "from_status": old.status,
        "to_status": new.status,
        "updated_at": new.updated_at.isoformat(),
        "event": event,
        "category": category,
        "reasons": list(reasons),
    }
    event_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    ).hexdigest()
    return TransitionEvent(
        event_id=event_id,
        timestamp=new.updated_at,
        episode=new.episode_index,
        from_status=old.status,
        to_status=new.status,
        event=event,
        category=category,
        reasons=tuple(reasons),
    )


def _record_events(record: EpisodeRecord) -> tuple[TransitionEvent, ...]:
    raw = record.sampling_details.get(_OUTBOX_KEY, [])
    if not isinstance(raw, list):
        raise ValueError("transition outbox must be a list")
    events: list[TransitionEvent] = []
    for item in raw:
        event = TransitionEvent.model_validate_json(
            json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
        if event.episode != record.episode_index:
            raise ValueError("transition outbox episode does not match record")
        expected = _transition_event_from_values(record.run_fingerprint, event)
        if event.event_id != expected:
            raise ValueError("transition outbox event_id is invalid")
        events.append(event)
    if len({item.event_id for item in events}) != len(events):
        raise ValueError("transition outbox contains duplicate event ids")
    for prior, current in zip(events, events[1:]):
        if prior.to_status != current.from_status or prior.timestamp >= current.timestamp:
            raise ValueError("transition outbox chain is invalid")
    if events:
        if events[0].from_status != "pending" or events[-1].to_status != record.status:
            raise ValueError("transition outbox does not cover record status")
    elif record.status != "pending":
        raise ValueError("non-pending record is missing its transition outbox")
    return tuple(events)


def _transition_event_from_values(run_fingerprint: str, event: TransitionEvent) -> str:
    payload = {
        "run_fingerprint": run_fingerprint,
        "episode": event.episode,
        "from_status": event.from_status,
        "to_status": event.to_status,
        "updated_at": event.timestamp.isoformat(),
        "event": event.event,
        "category": event.category,
        "reasons": list(event.reasons),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    ).hexdigest()


def _workspace_events(store: _Store) -> list[TransitionEvent]:
    summary = store.summary()
    total = summary.get("total")
    if type(total) is not int or total < 0:
        raise ValueError("workspace summary total is invalid")
    events = [event for index in range(total) for event in _record_events(store.load_episode(index))]
    return sorted(events, key=lambda item: (item.timestamp, item.episode, item.event_id))


def _open_regular(path: Path, flags: int, mode: int) -> int:
    if path.is_symlink():
        raise ValueError("audit path must not be a symlink")
    safe_flags = flags | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, safe_flags, mode)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("audit path must be a regular file")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _render_log(events: Sequence[TransitionEvent]) -> bytes:
    lines = []
    for event in events:
        line = json.dumps(
            event.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode() + b"\n"
        if len(line) > _MAX_LOG_LINE_BYTES:
            raise ValueError("rendered run log line exceeds the bounded size limit")
        lines.append(line)
    rendered = b"".join(lines)
    if len(rendered) > _MAX_LOG_BYTES:
        raise ValueError("rendered run log exceeds the bounded size limit")
    return rendered


def _read_official_log(directory_fd: int, name: str) -> bytes | None:
    try:
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(entry.st_mode):
        raise ValueError("official run log must be a regular non-symlink file")
    descriptor = os.open(
        name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("official run log must be a regular file")
        if opened.st_size > _MAX_LOG_BYTES:
            return None
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _replace_log_atomically(directory_fd: int, destination: str, rendered: bytes) -> None:
    temporary = ""
    descriptor = -1
    for _ in range(16):
        candidate = f".{destination}.{secrets.token_hex(12)}.tmp"
        try:
            descriptor = os.open(
                candidate,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            continue
        temporary = candidate
        break
    if descriptor < 0:
        raise OSError("could not allocate a unique audit log temporary file")
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("audit log temporary must be a regular file")
        _write_all(descriptor, rendered)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _assert_destination_safe(directory_fd, destination)
        os.replace(
            temporary, destination,
            src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
        )
        temporary = ""
        os.fsync(directory_fd)
    finally:
        try:
            if descriptor >= 0:
                os.close(descriptor)
        finally:
            if temporary:
                try:
                    os.unlink(temporary, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass


def _assert_destination_safe(directory_fd: int, destination: str) -> None:
    try:
        entry = os.stat(destination, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(entry.st_mode):
        raise ValueError("official run log must be a regular non-symlink file")


def _strict_json_object(value: bytes | str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    parsed = json.loads(
        value,
        object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonstandard JSON")),
    )
    if not isinstance(parsed, dict):
        raise ValueError("JSON value must be an object")
    return parsed


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("audit log write made no progress")
        view = view[written:]


def _episode_selection(values: Sequence[int] | None, total: int) -> list[int]:
    if values is None:
        return list(range(total))
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("episode_indices must be a sequence of integers")
    result = list(values)
    if any(type(value) is not int for value in result):
        raise TypeError("episode indices must be integers")
    if len(set(result)) != len(result):
        raise ValueError("episode indices must be unique")
    if any(value < 0 or value >= total for value in result):
        raise ValueError("episode index is outside dataset range")
    return sorted(result)


def _installed_model(config: AnnotationConfig) -> ModelInstall:
    local = config.model.local_path.resolve()
    metadata = local / "model-install.json"
    try:
        raw = _strict_json_object(
            _read_bounded_regular(metadata, _MAX_MODEL_METADATA_BYTES).decode("utf-8")
        )
        install = ModelInstall.from_dict(raw)
    except Exception:
        raise ValueError("model-install.json is invalid") from None
    _validate_install(config, install)
    return install


def _read_bounded_regular(path: Path, limit: int) -> bytes:
    if path.is_symlink():
        raise ValueError("file must not be a symlink")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("file must be a bounded regular file")
        chunks: list[bytes] = []
        length = 0
        while length <= limit:
            chunk = os.read(descriptor, min(64 * 1024, limit + 1 - length))
            if not chunk:
                break
            chunks.append(chunk)
            length += len(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        def identity(value):
            return (
                value.st_dev, value.st_ino, value.st_mode, value.st_size,
                value.st_mtime_ns, value.st_ctime_ns,
            )
        if length > limit or length != after.st_size or identity(before) != identity(after) or identity(after) != identity(current):
            raise ValueError("file identity changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validate_install(config: AnnotationConfig, install: object) -> None:
    if not isinstance(install, ModelInstall):
        raise TypeError("model resolver must return ModelInstall")
    local = config.model.local_path.resolve()
    if install.repo != config.model.name or install.local_path.resolve() != local:
        raise ValueError("model install does not match configured repository and local path")
    if len(install.revision) != 40 or any(char not in _SHA40 for char in install.revision):
        raise ValueError("model revision must be an exact lowercase 40-character SHA")
    if config.model.revision is not None and config.model.revision != install.revision:
        raise ValueError("model install revision does not match configured revision")


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC)

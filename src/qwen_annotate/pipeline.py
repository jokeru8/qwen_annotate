"""Resumable, bounded orchestration for annotation workspaces."""

from __future__ import annotations

import asyncio
import json
import os
import stat
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .coarse import CoarseDecision, run_coarse
from .config import AnnotationConfig
from .lerobot import DatasetIndex, EpisodeInfo, inspect_dataset
from .model_manager import ModelInstall
from .models import CoarseResult, FinalAnnotation, RefineResult, ValidationIssue
from .prompts import PROMPT_VERSION
from .qwen_client import InvalidModelResponse, ModelCallError, ModelOutOfMemory, QwenClient
from .refine import RefineDecision, run_refine
from .workspace import EpisodeRecord, WorkspaceStore


_STATUSES = ("pending", "coarse_done", "refine_done", "accepted", "needs_review", "failed")
_NONACCEPTED = ("pending", "coarse_done", "refine_done", "needs_review", "failed")
_SHA40 = frozenset("0123456789abcdef")
_VALIDATION_ISSUE_CODES = {
    "start_subtask_range", "complete_start_index", "complete_boundary_count",
    "dagger_suffix_length", "boundary_order", "boundary_range", "segment_too_short",
}


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


class _RunLog:
    def __init__(self, work_dir: Path, clock: Callable[[], datetime]) -> None:
        self.path = work_dir / "logs" / "run.jsonl"
        self.clock = clock
        self.lock = asyncio.Lock()

    def prepare(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("run log must not be a symlink")
        if self.path.exists() and not self.path.is_file():
            raise ValueError("run log must be a regular file")

    async def transition(self, episode: int, old: str, new: str, event: str, *, category: str | None = None, reasons: Sequence[str] = ()) -> None:
        payload: dict[str, object] = {
            "timestamp": _utc(self.clock()).isoformat().replace("+00:00", "Z"),
            "episode": episode, "from_status": old, "to_status": new, "event": event,
        }
        if category is not None:
            payload["category"] = category
        if reasons:
            payload["reasons"] = list(reasons)
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"
        async with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_symlink():
                raise ValueError("run log must not be a symlink")
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self.path, flags, 0o600)
            try:
                mode = os.fstat(descriptor).st_mode
                if not stat.S_ISREG(mode):
                    raise ValueError("run log must be a regular file")
                os.write(descriptor, line.encode("utf-8"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


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
    logger.prepare()
    client = service.client_factory(config)
    semaphore = asyncio.Semaphore(max_concurrency)
    stop_requested = asyncio.Event()

    resumable = []
    for index in selected:
        if store.load_episode(index).status in {"pending", "coarse_done", "refine_done"}:
            resumable.append(index)

    async def process(index: int) -> BaseException | None:
        try:
            async with semaphore:
                if stop_requested.is_set():
                    return None
                await _process_episode(
                    config, dataset, dataset.episodes[index], store, install.revision,
                    client, service, logger, stop_requested,
                )
        except (KeyboardInterrupt, asyncio.CancelledError) as exc:
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

    for _ in range(min(max_concurrency, len(resumable))):
        launch()
    try:
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
    except asyncio.CancelledError:
        stop_requested.set()
        if active:
            await asyncio.shield(asyncio.gather(*active, return_exceptions=True))
        raise
    return WorkspaceSummary.from_store_summary(store.summary())


async def _process_episode(config, dataset, episode, store, revision, client, services, logger, stop_requested) -> None:
    record = store.load_episode(episode.episode_index)
    coarse: CoarseDecision | None = None
    try:
        if record.status == "pending":
            coarse = await services.run_coarse(
                config, episode, client=client, source_fps=float(dataset.fps),
                expected_source_fingerprint=record.source_fingerprint,
            )
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
            await _save_transition(store, logger, record, updated, "coarse_completed")
            record = updated
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
            refined = await services.run_refine(
                config, episode, coarse, client=client, source_fps=float(dataset.fps),
                expected_source_fingerprint=record.source_fingerprint,
            )
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
            await _save_transition(store, logger, record, updated, "refine_completed")
            record = updated
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
                    for reason in refined.reasons if reason in _VALIDATION_ISSUE_CODES
                ]
                updated = _replace(record, services.clock, status="needs_review", validation_issues=issues,
                    review_reasons=list(refined.reasons), decision_source=None)
                await _save_transition(store, logger, record, updated, "refine_review", reasons=refined.reasons)
            else:
                await _fail(store, logger, record, services.clock, refined.failure_category or "model_oom")
    except InvalidModelResponse:
        current = store.load_episode(episode.episode_index)
        updated = _replace(current, services.clock, status="needs_review", review_reasons=["invalid_model_response"])
        await _save_transition(store, logger, current, updated, "error", category="invalid_model_response", reasons=("invalid_model_response",))
    except ModelOutOfMemory:
        await _fail(store, logger, store.load_episode(episode.episode_index), services.clock, "model_oom")
    except ModelCallError:
        await _fail(store, logger, store.load_episode(episode.episode_index), services.clock, "model_call")
    except (FileNotFoundError, OSError, ValueError):
        await _fail(store, logger, store.load_episode(episode.episode_index), services.clock, "source_or_video")
    except (KeyboardInterrupt, asyncio.CancelledError):
        raise
    except Exception:
        await _fail(store, logger, store.load_episode(episode.episode_index), services.clock, "unexpected_error")


async def _save_transition(store, logger, old, new, event, *, category=None, reasons=()) -> None:
    store.save_episode(new)
    await logger.transition(old.episode_index, old.status, new.status, event, category=category, reasons=reasons)


async def _fail(store, logger, record, clock, category, *, details=None) -> None:
    updated = _replace(record, clock, status="failed", failure_category=category,
        sampling_details=details if details is not None else record.sampling_details)
    await _save_transition(store, logger, record, updated, "error", category=category)


def _replace(record: EpisodeRecord, clock, **changes) -> EpisodeRecord:
    now = _utc(clock())
    changes["updated_at"] = max(now, record.updated_at + timedelta(microseconds=1))
    return EpisodeRecord.model_validate(record.model_dump() | changes)


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
    if metadata.is_symlink() or not metadata.is_file():
        raise ValueError("verified model-install.json is required")
    try:
        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate JSON key")
                value[key] = item
            return value

        raw = json.loads(
            metadata.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonstandard JSON")),
        )
        install = ModelInstall.from_dict(raw)
    except Exception:
        raise ValueError("model-install.json is invalid") from None
    _validate_install(config, install)
    return install


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

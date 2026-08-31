"""Adaptive, auditable refinement of coarse subtask boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SkipValidation,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .coarse import CoarseDecision
from .config import AnnotationConfig
from .constraints import coarse_sequence_is_legal, validate_annotation
from .lerobot import EpisodeInfo, EpisodeVideoRef
from .models import FinalAnnotation, RefineResult
from .prompts import build_refine_prompt
from .qwen_client import ModelOutOfMemory, QwenClient
from .video import FrameSample, extract_frames, window_indices


ReviewReason = Literal[
    "refine_boundary_disagreement",
    "refine_transition_mismatch",
    "camera_evidence_conflict",
    "start_subtask_range",
    "complete_start_index",
    "complete_boundary_count",
    "dagger_suffix_length",
    "boundary_order",
    "boundary_range",
    "segment_too_short",
    "model_oom",
]
_REASON_ORDER: tuple[ReviewReason, ...] = (
    "refine_boundary_disagreement",
    "refine_transition_mismatch",
    "camera_evidence_conflict",
    "start_subtask_range",
    "complete_start_index",
    "complete_boundary_count",
    "dagger_suffix_length",
    "boundary_order",
    "boundary_range",
    "segment_too_short",
    "model_oom",
)


class _Completer(Protocol):
    async def complete(
        self, prompt: str, frames: list[FrameSample], response_type: type[RefineResult]
    ) -> RefineResult: ...


Sampler = Callable[[EpisodeVideoRef, str, list[int]], list[FrameSample]]


class _ImmutableRefineResult(RefineResult):
    visible_cues: tuple[str, ...]


class _ImmutableFinalAnnotation(FinalAnnotation):
    boundaries: tuple[int, ...]


class CameraSampling(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    camera_key: str = Field(min_length=1)
    frame_indices: tuple[int, ...] = Field(min_length=1)

    @field_validator("frame_indices", mode="before")
    @classmethod
    def restore_json_tuple(cls, value: object, info: ValidationInfo) -> object:
        if info.mode == "json" and isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("frame_indices")
    @classmethod
    def valid_grid(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(type(item) is not int or item < 0 for item in value):
            raise ValueError("frame_indices must be non-negative strict integers")
        if any(left >= right for left, right in zip(value, value[1:])):
            raise ValueError("frame_indices must be ordered and unique")
        return value


class SamplingProvenance(BaseModel):
    """The exact evidence request associated with one model call."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    boundary_index: int = Field(ge=0)
    from_subtask_index: int = Field(ge=0)
    to_subtask_index: int = Field(ge=0)
    stage: Literal["broad", "broad_retry", "dense"]
    pass_id: int = Field(ge=0)
    request_center: int = Field(ge=0)
    radius_frames: int = Field(ge=0)
    stride: int = Field(ge=1)
    cameras: tuple[str, ...] = Field(min_length=1)
    samples: tuple[CameraSampling, ...] = Field(min_length=1)
    outcome: Literal["completed", "model_oom"]

    @field_validator("cameras", "samples", mode="before")
    @classmethod
    def restore_json_tuples(cls, value: object, info: ValidationInfo) -> object:
        if info.mode == "json" and isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def internally_consistent(self) -> "SamplingProvenance":
        if self.to_subtask_index != self.from_subtask_index + 1:
            raise ValueError("provenance transition must be consecutive")
        if tuple(item.camera_key for item in self.samples) != self.cameras:
            raise ValueError("sample camera order must equal cameras")
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError("provenance cameras must be unique")
        return self


class RefineDecision(BaseModel):
    """Deeply immutable final outcome and complete typed refinement audit."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    mode: Literal["complete", "dagger_patch"]
    subtask_count: int = Field(ge=1)
    frame_count: int = Field(ge=1)
    min_segment_frames: int = Field(ge=1)
    agreement_tolerance_frames: int = Field(ge=0)
    start_subtask_index: int = Field(ge=0)
    observed_subtask_indices: tuple[int, ...] = Field(min_length=1)
    coarse_boundary_centers: tuple[int, ...]
    source_fps: float = Field(gt=0)
    refine_window_seconds: float = Field(gt=0)
    refine_fps: float = Field(gt=0)
    dense_radius_seconds: float = Field(gt=0)
    camera_order: tuple[str, ...] = Field(min_length=1)
    broad_radius_frames: int = Field(ge=0)
    base_broad_stride: int = Field(ge=1)
    dense_radius_frames: int = Field(ge=0)
    status: Literal["accepted", "needs_review", "failed"]
    attempts: tuple[SkipValidation[_ImmutableRefineResult], ...]
    provenance: tuple[SamplingProvenance, ...]
    reasons: tuple[ReviewReason, ...]
    annotation: SkipValidation[_ImmutableFinalAnnotation] | None = None
    candidate_annotation: SkipValidation[_ImmutableFinalAnnotation] | None = None
    failure_category: Literal["model_oom"] | None = None

    @field_validator(
        "observed_subtask_indices", "coarse_boundary_centers", "camera_order",
        "provenance", "reasons", mode="before"
    )
    @classmethod
    def restore_json_sequences(cls, value: object, info: ValidationInfo) -> object:
        if info.mode == "json" and isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("attempts", mode="before")
    @classmethod
    def freeze_attempts(cls, value: object, info: ValidationInfo) -> object:
        if info.mode == "json" and isinstance(value, list):
            value = tuple(value)
        if not isinstance(value, tuple):
            return value
        return tuple(_freeze_result(item) for item in value)

    @field_validator("annotation", "candidate_annotation", mode="before")
    @classmethod
    def freeze_annotations(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, FinalAnnotation):
            payload = value.model_dump()
        elif isinstance(value, dict) and set(value) == {"start_subtask_index", "boundaries"}:
            payload = value
        else:
            raise ValueError("annotations must be FinalAnnotation-compatible")
        start = payload["start_subtask_index"]
        boundaries = payload["boundaries"]
        if type(start) is not int or start < 0:
            raise ValueError("annotation start must be a non-negative strict integer")
        if not isinstance(boundaries, (list, tuple)) or any(
            type(boundary) is not int or boundary < 0 for boundary in boundaries
        ):
            raise ValueError("annotation boundaries must be non-negative strict integers")
        return _ImmutableFinalAnnotation.model_construct(
            start_subtask_index=start,
            boundaries=tuple(boundaries),
        )

    @model_validator(mode="after")
    def status_context_is_consistent(self) -> "RefineDecision":
        if self.start_subtask_index != self.observed_subtask_indices[0]:
            raise ValueError("start_subtask_index must match the first observed subtask")
        if not coarse_sequence_is_legal(
            list(self.observed_subtask_indices), self.mode, self.subtask_count
        ):
            raise ValueError("observed_subtask_indices must be a legal coarse sequence")
        for value, name in (
            (self.source_fps, "source_fps"),
            (self.refine_window_seconds, "refine_window_seconds"),
            (self.refine_fps, "refine_fps"),
            (self.dense_radius_seconds, "dense_radius_seconds"),
        ):
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a strict finite float")
        if self.broad_radius_frames != round(self.refine_window_seconds * self.source_fps):
            raise ValueError("broad radius must be recomputable from source_fps and window seconds")
        if self.base_broad_stride != max(1, round(self.source_fps / self.refine_fps)):
            raise ValueError("base broad stride must be recomputable from source/refine fps")
        if self.dense_radius_frames != round(self.dense_radius_seconds * self.source_fps):
            raise ValueError("dense radius must be recomputable from source_fps and radius seconds")
        if len(set(self.camera_order)) != len(self.camera_order) or any(
            not isinstance(camera, str) or not camera for camera in self.camera_order
        ):
            raise ValueError("camera_order must contain unique nonempty camera names")
        if len(self.coarse_boundary_centers) != len(self.observed_subtask_indices) - 1:
            raise ValueError("coarse centers must correspond to every observed transition")
        if any(
            type(center) is not int or not 1 <= center < self.frame_count
            for center in self.coarse_boundary_centers
        ) or any(
            left >= right
            for left, right in zip(self.coarse_boundary_centers, self.coarse_boundary_centers[1:])
        ):
            raise ValueError("coarse centers must be ordered valid transition frames")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("reasons must be unique")
        if self.reasons != tuple(reason for reason in _REASON_ORDER if reason in self.reasons):
            raise ValueError("reasons must use stable deterministic order")
        completed = sum(item.outcome == "completed" for item in self.provenance)
        if completed != len(self.attempts):
            raise ValueError("each completed evidence call must have exactly one typed attempt")
        for result in self.attempts:
            _validate_result(result)
        audit_reasons, selected = _semantic_audit(self)
        if self.status == "accepted":
            if self.reasons or self.failure_category is not None or self.annotation is None or self.candidate_annotation is not None:
                raise ValueError("accepted requires only a final annotation and no reasons")
            if validate_annotation(
                _mutable_annotation(self.annotation), self.mode, self.subtask_count,
                self.frame_count, self.min_segment_frames,
            ):
                raise ValueError("accepted annotation must satisfy mode/range/order constraints")
            if self.annotation.start_subtask_index != self.start_subtask_index:
                raise ValueError("accepted annotation start must match the coarse sequence")
            if len(self.annotation.boundaries) != len(self.observed_subtask_indices) - 1:
                raise ValueError("accepted annotation must cover every observed transition")
            if tuple(self.annotation.boundaries) != selected:
                raise ValueError("accepted boundaries must exactly equal audited agreed results")
        elif self.status == "needs_review":
            if not self.reasons or self.failure_category is not None or self.annotation is not None:
                raise ValueError("needs_review requires reasons, no failure, and no accepted annotation")
            validation_codes: set[str] = set()
            if self.candidate_annotation is not None:
                if self.candidate_annotation.start_subtask_index != self.start_subtask_index:
                    raise ValueError("review candidate start must match the coarse sequence")
                if len(self.candidate_annotation.boundaries) != len(self.observed_subtask_indices) - 1:
                    raise ValueError("review candidate must cover every observed transition")
                validation_codes = {
                    issue.code for issue in validate_annotation(
                        _mutable_annotation(self.candidate_annotation), self.mode,
                        self.subtask_count, self.frame_count, self.min_segment_frames,
                    )
                }
                if tuple(self.candidate_annotation.boundaries) != selected:
                    raise ValueError("review candidate must contain every audited agreed boundary")
            expected = _ordered(audit_reasons | validation_codes)
            if self.reasons != expected:
                raise ValueError("needs_review reasons must exactly match the auditable outcome")
        else:
            if self.failure_category != "model_oom" or "model_oom" not in self.reasons:
                raise ValueError("failed decisions must identify model_oom")
            if self.reasons != _ordered(audit_reasons):
                raise ValueError("failed reasons must preserve prior audit reasons plus model_oom")
            if not self.provenance or self.provenance[-1].outcome != "model_oom":
                raise ValueError("failed model_oom decisions require terminal OOM provenance")
            if self.annotation is not None or self.candidate_annotation is not None:
                raise ValueError("failed decisions cannot expose an annotation")
        return self


def choose_agreed_boundary(results: list[RefineResult], tolerance: int) -> int | None:
    """Return the positive half-up median for exactly two results within tolerance."""
    if type(results) is not list:
        raise TypeError("results must be a list")
    if len(results) != 2:
        raise ValueError("results must contain exactly two RefineResult values")
    if type(tolerance) is not int:
        raise TypeError("tolerance must be an integer")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if any(not isinstance(item, RefineResult) for item in results):
        raise TypeError("results must contain exactly two RefineResult values")
    if any(type(item.boundary_frame) is not int for item in results):
        raise TypeError("RefineResult boundary_frame values must be strict integers")
    values = sorted(item.boundary_frame for item in results)
    if values[-1] - values[0] > tolerance:
        return None
    return (values[0] + values[1] + 1) // 2


@dataclass(frozen=True)
class _FileState:
    entry: Path
    resolved: Path
    identity: tuple[int, ...]


@dataclass(frozen=True)
class _SourceState:
    root: Path
    info: _FileState
    info_digest: str
    videos: tuple[tuple[str, _FileState], ...]
    fps: float


async def run_refine(
    config: AnnotationConfig,
    episode: EpisodeInfo,
    coarse: CoarseDecision,
    sampler: Sampler = extract_frames,
    client: _Completer | None = None,
    *,
    source_fps: float | None = None,
    expected_source_fingerprint: str | None = None,
) -> RefineDecision:
    """Refine every coarse transition using broad then dense visual evidence.

    Frame radii and strides use Python's deterministic nearest-even ``round``.
    Window endpoints and the center are always retained.

    Source integrity is guarded by contained-path, metadata digest, and file
    identity checks around evidence calls; it does not claim protection from a
    hostile process that retains and mutates an already-open file descriptor.
    """
    cameras = _validate_context(config, episode, coarse)
    source = _prepare_source(config, episode, cameras, source_fps, expected_source_fingerprint)
    common = {
        "mode": config.mode,
        "subtask_count": len(config.subtasks),
        "frame_count": episode.length,
        "min_segment_frames": config.sampling.min_segment_frames,
        "agreement_tolerance_frames": config.sampling.agreement_tolerance_frames,
        "start_subtask_index": coarse.start_subtask_index,
        "observed_subtask_indices": tuple(coarse.observed_subtask_indices),
        "coarse_boundary_centers": tuple(coarse.boundary_centers),
        "source_fps": source.fps,
        "refine_window_seconds": float(config.sampling.refine_window_seconds),
        "refine_fps": float(config.sampling.refine_fps),
        "dense_radius_seconds": float(config.sampling.dense_radius_seconds),
        "camera_order": cameras,
        "broad_radius_frames": round(config.sampling.refine_window_seconds * source.fps),
        "base_broad_stride": max(1, round(source.fps / config.sampling.refine_fps)),
        "dense_radius_frames": round(config.sampling.dense_radius_seconds * source.fps),
    }
    if not coarse.boundary_centers:
        annotation = FinalAnnotation(
            start_subtask_index=coarse.start_subtask_index,
            boundaries=[],
        )
        issues = validate_annotation(
            annotation, config.mode, len(config.subtasks), episode.length,
            config.sampling.min_segment_frames,
        )
        reasons = _ordered({item.code for item in issues})
        if reasons:
            return RefineDecision(
                status="needs_review", attempts=(), provenance=(), reasons=reasons,
                annotation=None, candidate_annotation=annotation, failure_category=None, **common,
            )
        return RefineDecision(
            status="accepted", attempts=(), provenance=(), reasons=(),
            annotation=annotation, candidate_annotation=None, failure_category=None, **common,
        )

    completer: _Completer = client if client is not None else QwenClient(
        endpoint=str(config.model.endpoint), api_key=config.model.api_key, model=config.model.name
    )
    attempts: list[RefineResult] = []
    provenance: list[SamplingProvenance] = []
    agreed: list[int] = []
    found: set[ReviewReason] = set()
    degraded = False

    broad_radius = common["broad_radius_frames"]
    base_stride = common["base_broad_stride"]
    dense_radius = common["dense_radius_frames"]

    pairs = list(zip(coarse.observed_subtask_indices, coarse.observed_subtask_indices[1:]))
    for boundary_index, ((left, right), center) in enumerate(zip(pairs, coarse.boundary_centers)):
        broad_pass = boundary_index * 3
        active_cameras = (config.primary_camera,) if degraded else cameras
        active_stride = base_stride * 2 if degraded else base_stride
        broad_indices = _evidence_indices(center, broad_radius, active_stride, episode.length)
        broad_samples = await _sample_stage(
            source, active_cameras, broad_indices, sampler, episode,
            expected_source_fingerprint,
        )
        broad_prompt = build_refine_prompt(
            config, episode.episode_index, episode.length, left, right, center, broad_pass
        )
        try:
            broad = await completer.complete(broad_prompt, broad_samples, RefineResult)
            provenance.append(_provenance(
                boundary_index, left, right, "broad", broad_pass, center,
                broad_radius, active_stride, active_cameras, broad_indices, "completed",
            ))
        except ModelOutOfMemory:
            _assert_source_unchanged(source, episode, expected_source_fingerprint)
            provenance.append(_provenance(
                boundary_index, left, right, "broad", broad_pass, center,
                broad_radius, active_stride, active_cameras, broad_indices, "model_oom",
            ))
            if degraded or len(active_cameras) == 1:
                return _oom_failure(common, attempts, provenance, found)
            degraded = True
            active_cameras = (config.primary_camera,)
            retry_stride = base_stride * 2
            retry_indices = _evidence_indices(center, broad_radius, retry_stride, episode.length)
            retry_samples = await _sample_stage(
                source, active_cameras, retry_indices, sampler, episode,
                expected_source_fingerprint,
            )
            retry_pass = broad_pass + 1
            retry_prompt = build_refine_prompt(
                config, episode.episode_index, episode.length, left, right, center, retry_pass
            )
            try:
                broad = await completer.complete(retry_prompt, retry_samples, RefineResult)
                provenance.append(_provenance(
                    boundary_index, left, right, "broad_retry", retry_pass, center,
                    broad_radius, retry_stride, active_cameras, retry_indices, "completed",
                ))
            except ModelOutOfMemory:
                _assert_source_unchanged(source, episode, expected_source_fingerprint)
                provenance.append(_provenance(
                    boundary_index, left, right, "broad_retry", retry_pass, center,
                    broad_radius, retry_stride, active_cameras, retry_indices, "model_oom",
                ))
                return _oom_failure(common, attempts, provenance, found)

        if not isinstance(broad, RefineResult):
            raise TypeError("client.complete must return a RefineResult")
        attempts.append(broad)
        _assert_source_unchanged(source, episode, expected_source_fingerprint)

        dense_center = broad.boundary_frame
        if not 1 <= dense_center < episode.length:
            found.add("refine_transition_mismatch")
            break
        dense_indices = _evidence_indices(dense_center, dense_radius, 1, episode.length)
        dense_samples = await _sample_stage(
            source, active_cameras, dense_indices, sampler, episode,
            expected_source_fingerprint,
        )
        dense_pass = broad_pass + 2
        dense_prompt = build_refine_prompt(
            config, episode.episode_index, episode.length, left, right, dense_center, dense_pass
        )
        try:
            dense = await completer.complete(dense_prompt, dense_samples, RefineResult)
            provenance.append(_provenance(
                boundary_index, left, right, "dense", dense_pass, dense_center,
                dense_radius, 1, active_cameras, dense_indices, "completed",
            ))
        except ModelOutOfMemory:
            _assert_source_unchanged(source, episode, expected_source_fingerprint)
            provenance.append(_provenance(
                boundary_index, left, right, "dense", dense_pass, dense_center,
                dense_radius, 1, active_cameras, dense_indices, "model_oom",
            ))
            return _oom_failure(common, attempts, provenance, found)
        if not isinstance(dense, RefineResult):
            raise TypeError("client.complete must return a RefineResult")
        attempts.append(dense)
        _assert_source_unchanged(source, episode, expected_source_fingerprint)

        if not (_matches(broad, left, right, episode.length) and _matches(dense, left, right, episode.length)):
            found.add("refine_transition_mismatch")
            continue
        selected = choose_agreed_boundary([broad, dense], config.sampling.agreement_tolerance_frames)
        if selected is None:
            found.add("refine_boundary_disagreement")
            continue
        agreed.append(selected)

    _assert_source_unchanged(source, episode, expected_source_fingerprint)
    candidate: FinalAnnotation | None = None
    if len(agreed) == len(coarse.boundary_centers):
        candidate = FinalAnnotation(
            start_subtask_index=coarse.start_subtask_index,
            boundaries=agreed,
        )
        issues = validate_annotation(
            candidate, config.mode, len(config.subtasks), episode.length,
            config.sampling.min_segment_frames,
        )
        found.update(item.code for item in issues)
    reasons = _ordered(found)
    if reasons:
        return RefineDecision(
            status="needs_review", attempts=tuple(attempts), provenance=tuple(provenance),
            reasons=reasons, annotation=None, candidate_annotation=candidate,
            failure_category=None, **common,
        )
    assert candidate is not None
    return RefineDecision(
        status="accepted", attempts=tuple(attempts), provenance=tuple(provenance),
        reasons=(), annotation=candidate, candidate_annotation=None,
        failure_category=None, **common,
    )


def _validate_context(config: AnnotationConfig, episode: EpisodeInfo, coarse: CoarseDecision) -> tuple[str, ...]:
    if not isinstance(config, AnnotationConfig):
        raise TypeError("config must be an AnnotationConfig")
    if not isinstance(episode, EpisodeInfo):
        raise TypeError("episode must be an EpisodeInfo")
    if not isinstance(coarse, CoarseDecision) or coarse.status != "coarse_done":
        raise ValueError("refine requires a successful coarse_done CoarseDecision")
    if coarse.mode != config.mode:
        raise ValueError("coarse mode does not match config mode")
    if coarse.subtask_count != len(config.subtasks):
        raise ValueError("coarse subtask_count does not match config")
    if coarse.frame_count != episode.length:
        raise ValueError("coarse frame_count does not match episode length")
    try:
        CoarseDecision.model_validate_json(coarse.model_dump_json())
    except Exception:
        raise ValueError("refine requires a structurally valid coarse_done CoarseDecision") from None
    if len(coarse.boundary_centers) != len(coarse.observed_subtask_indices) - 1:
        raise ValueError("coarse boundary count does not match observed sequence")
    numeric = (
        (config.sampling.refine_window_seconds, "refine_window_seconds"),
        (config.sampling.refine_fps, "refine_fps"),
        (config.sampling.dense_radius_seconds, "dense_radius_seconds"),
    )
    for value, name in numeric:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    if type(config.sampling.agreement_tolerance_frames) is not int or config.sampling.agreement_tolerance_frames < 0:
        raise ValueError("agreement_tolerance_frames must be a non-negative integer")
    if type(config.sampling.min_segment_frames) is not int or config.sampling.min_segment_frames < 1:
        raise ValueError("min_segment_frames must be a positive integer")
    cameras = tuple(dict.fromkeys([config.primary_camera, *config.refine_cameras]))
    for camera in cameras:
        if camera not in episode.videos:
            raise ValueError(f"configured refine camera {camera!r} is missing from episode")
    return cameras


def _prepare_source(config, episode, cameras, source_fps, expected_fingerprint):
    try:
        root = config.source.resolve(strict=True)
    except OSError:
        raise ValueError("configured source root must exist") from None
    if not root.is_dir():
        raise ValueError("configured source root must be a directory")
    info_entry, info_resolved = _contained_regular(root, root / "meta" / "info.json", "source info.json")
    info_identity, digest = _capture(info_entry, True)
    metadata_fps = _read_fps(info_entry)
    after_identity, after_digest = _capture(info_entry, True)
    if (after_identity, after_digest) != (info_identity, digest):
        raise ValueError("source evidence changed during refine annotation")
    if source_fps is not None:
        if isinstance(source_fps, bool) or not isinstance(source_fps, (int, float)):
            raise ValueError("authoritative source_fps must be a positive finite number")
        effective = float(source_fps)
        if not math.isfinite(effective) or effective <= 0 or not math.isclose(effective, metadata_fps, rel_tol=0, abs_tol=1e-9):
            raise ValueError("authoritative source_fps does not match current source metadata")
    else:
        effective = metadata_fps
    videos = []
    for camera in cameras:
        entry, resolved = _contained_regular(
            root,
            episode.videos[camera].path,
            f"camera {camera!r} video",
        )
        identity, _ = _capture(entry, False)
        videos.append((camera, _FileState(entry, resolved, identity)))
    if expected_fingerprint is not None:
        _validate_fingerprint(expected_fingerprint)
        if _fingerprint(root, episode) != expected_fingerprint:
            raise ValueError("expected source fingerprint does not match current episode source")
    return _SourceState(
        root, _FileState(info_entry, info_resolved, info_identity), digest, tuple(videos), effective
    )


def _contained_regular(root: Path, path: Path, label: str):
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        raise ValueError(f"{label} must be inside source root") from None
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} must not use a symlink")
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
        file_stat = os.stat(lexical, follow_symlinks=False)
    except (OSError, ValueError):
        raise ValueError(f"{label} must resolve inside source root") from None
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return lexical, resolved


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _capture(path: Path, digest: bool):
    if path.is_symlink():
        raise ValueError("source evidence changed during refine annotation")
    try:
        before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("source evidence changed during refine annotation")
        data = path.read_bytes() if digest else b""
        after = os.stat(path, follow_symlinks=False)
    except OSError:
        raise ValueError("source evidence changed during refine annotation") from None
    if _identity(before) != _identity(after):
        raise ValueError("source evidence changed during refine annotation")
    return _identity(after), hashlib.sha256(data).hexdigest() if digest else ""


def _read_fps(path: Path) -> float:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonstandard JSON")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError("source info.json must contain valid standard JSON with unique keys") from None
    value = payload.get("fps") if isinstance(payload, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError("source metadata fps must be a positive finite number")
    return float(value)


def _assert_source_unchanged(source, episode, expected_fingerprint):
    identity, digest = _capture(source.info.entry, True)
    if identity != source.info.identity or digest != source.info_digest or source.info.entry.resolve(strict=True) != source.info.resolved:
        raise ValueError("source evidence changed during refine annotation")
    for _, item in source.videos:
        identity, _ = _capture(item.entry, False)
        if identity != item.identity or item.entry.resolve(strict=True) != item.resolved:
            raise ValueError("source evidence changed during refine annotation")
    if expected_fingerprint is not None and _fingerprint(source.root, episode) != expected_fingerprint:
        raise ValueError("source fingerprint changed during refine annotation")


def _fingerprint(root, episode):
    from .workspace import compute_source_fingerprint
    return compute_source_fingerprint(root, episode)


def _validate_fingerprint(value):
    if not isinstance(value, str) or len(value) != 64 or any(x not in "0123456789abcdef" for x in value):
        raise ValueError("expected source fingerprint must be lowercase SHA-256")


def _evidence_indices(center, radius, stride, frame_count):
    values = window_indices(center, radius, stride, frame_count)
    lower, upper = max(0, center - radius), min(frame_count - 1, center + radius)
    return sorted({*values, lower, center, upper})


async def _sample_stage(source, cameras, indices, sampler, episode, expected_fingerprint):
    source_videos = dict(source.videos)
    batches = await asyncio.gather(*(
        asyncio.to_thread(
            sampler,
            episode.videos[camera].model_copy(
                update={"path": source_videos[camera].resolved}
            ),
            camera,
            indices,
        )
        for camera in cameras
    ))
    combined = []
    for camera, batch in zip(cameras, batches):
        _validate_samples(batch, indices, camera, episode.videos[camera].fps)
        combined.extend(batch)
    order = {camera: index for index, camera in enumerate(cameras)}
    combined.sort(key=lambda item: (item.frame_index, order[item.camera_key]))
    if len({(item.camera_key, item.frame_index) for item in combined}) != len(combined):
        raise ValueError("sampler evidence contains duplicate camera/frame pairs")
    _assert_source_unchanged(source, episode, expected_fingerprint)
    return combined


def _validate_samples(samples, indices, camera, fps):
    if not isinstance(samples, list) or len(samples) != len(indices):
        raise ValueError("sampler evidence must contain exactly the requested frames")
    actual = []
    for item in samples:
        if not isinstance(item, FrameSample) or item.camera_key != camera:
            raise ValueError("sampler evidence camera must match the requested camera")
        if not math.isclose(item.timestamp_seconds, item.frame_index / fps, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError("sampler timestamp must match frame_index/source_fps")
        actual.append(item.frame_index)
    if actual != indices:
        raise ValueError("sampler evidence indices must match requested order without duplicates")


def _provenance(index, left, right, stage, pass_id, center, radius, stride, cameras, indices, outcome):
    return SamplingProvenance(
        boundary_index=index, from_subtask_index=left, to_subtask_index=right,
        stage=stage, pass_id=pass_id, request_center=center, radius_frames=radius,
        stride=stride, cameras=cameras,
        samples=tuple(CameraSampling(camera_key=camera, frame_indices=tuple(indices)) for camera in cameras),
        outcome=outcome,
    )


def _matches(result, left, right, length):
    return (
        result.from_subtask_index == left and result.to_subtask_index == right
        and 1 <= result.boundary_frame < length
        and result.first_frame_after == result.boundary_frame
        and result.last_frame_before == result.boundary_frame - 1
    )


def _freeze_result(value):
    fields = {
        "from_subtask_index", "to_subtask_index", "last_frame_before",
        "first_frame_after", "boundary_frame", "confidence", "visible_cues",
    }
    if isinstance(value, RefineResult):
        payload = value.model_dump()
    elif isinstance(value, dict) and set(value) == fields:
        payload = value
    else:
        raise ValueError("attempts must contain RefineResult-compatible values")
    _validate_attempt_payload(payload)
    return _ImmutableRefineResult.model_construct(
        **{field: payload[field] for field in fields - {"visible_cues"}},
        visible_cues=tuple(payload["visible_cues"]),
    )


def _validate_attempt_payload(payload):
    for field in (
        "from_subtask_index", "to_subtask_index", "last_frame_before",
        "first_frame_after", "boundary_frame",
    ):
        if type(payload.get(field)) is not int:
            raise ValueError(f"attempt {field} must be a strict integer")
    confidence = payload.get("confidence")
    if type(confidence) is not float or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("attempt confidence must be a finite strict float in [0, 1]")
    cues = payload.get("visible_cues")
    if not isinstance(cues, (list, tuple)) or not cues or any(
        not isinstance(cue, str) or not cue for cue in cues
    ):
        raise ValueError("attempt visible_cues must contain nonempty strings")


def _validate_result(value):
    if not isinstance(value, RefineResult):
        raise ValueError("attempts must contain RefineResult values")
    _validate_attempt_payload(value.model_dump())


def _mutable_annotation(value):
    return FinalAnnotation(
        start_subtask_index=value.start_subtask_index, boundaries=list(value.boundaries)
    )


def _semantic_audit(decision: RefineDecision) -> tuple[set[ReviewReason], tuple[int, ...]]:
    """Recompute semantic review state solely from frozen attempts/provenance."""
    found: set[ReviewReason] = set()
    expected_pairs = tuple(
        zip(decision.observed_subtask_indices, decision.observed_subtask_indices[1:])
    )
    provenance_index = 0
    attempt_index = 0
    degraded = False
    terminal: Literal["review", "failed"] | None = None
    selected: list[int] = []
    for boundary_index, pair in enumerate(expected_pairs):
        if provenance_index >= len(decision.provenance):
            raise ValueError("provenance must cover every expected boundary in execution order")
        broad = decision.provenance[provenance_index]
        broad_cameras = (decision.camera_order[0],) if degraded else decision.camera_order
        broad_stride = decision.base_broad_stride * 2 if degraded else decision.base_broad_stride
        _validate_provenance_entry(
            decision, broad, boundary_index, pair, "broad", boundary_index * 3,
            decision.coarse_boundary_centers[boundary_index], decision.broad_radius_frames,
            broad_stride, broad_cameras,
        )
        provenance_index += 1

        if broad.outcome == "model_oom":
            if degraded or len(decision.camera_order) == 1:
                terminal = "failed"
                break
            degraded = True
            if provenance_index >= len(decision.provenance):
                raise ValueError("multi-camera broad OOM must be followed by one retry")
            retry = decision.provenance[provenance_index]
            _validate_provenance_entry(
                decision, retry, boundary_index, pair, "broad_retry", boundary_index * 3 + 1,
                decision.coarse_boundary_centers[boundary_index], decision.broad_radius_frames,
                decision.base_broad_stride * 2, (decision.camera_order[0],),
            )
            provenance_index += 1
            if retry.outcome == "model_oom":
                terminal = "failed"
                break
            broad_result = decision.attempts[attempt_index]
            attempt_index += 1
        else:
            broad_result = decision.attempts[attempt_index]
            attempt_index += 1

        if not 1 <= broad_result.boundary_frame < decision.frame_count:
            found.add("refine_transition_mismatch")
            terminal = "review"
            break

        if provenance_index >= len(decision.provenance):
            raise ValueError("completed broad evidence must be followed by dense evidence")
        dense = decision.provenance[provenance_index]
        dense_cameras = (decision.camera_order[0],) if degraded else decision.camera_order
        _validate_provenance_entry(
            decision, dense, boundary_index, pair, "dense", boundary_index * 3 + 2,
            broad_result.boundary_frame, decision.dense_radius_frames, 1, dense_cameras,
        )
        provenance_index += 1
        if dense.outcome == "model_oom":
            terminal = "failed"
            break
        dense_result = decision.attempts[attempt_index]
        attempt_index += 1
        if not (
            _matches(broad_result, *pair, decision.frame_count)
            and _matches(dense_result, *pair, decision.frame_count)
        ):
            found.add("refine_transition_mismatch")
            continue
        agreed = choose_agreed_boundary(
            [broad_result, dense_result], decision.agreement_tolerance_frames
        )
        if agreed is None:
            found.add("refine_boundary_disagreement")
            continue
        selected.append(agreed)

    if provenance_index != len(decision.provenance) or attempt_index != len(decision.attempts):
        raise ValueError("provenance and attempts must end at the semantic terminal stage")
    if terminal == "failed":
        if decision.status != "failed":
            raise ValueError("terminal OOM provenance requires failed status")
        found.add("model_oom")
    elif decision.status == "failed":
        raise ValueError("failed status requires one terminal OOM stage")
    if terminal == "review" and decision.status != "needs_review":
        raise ValueError("invalid broad result requires needs_review status")
    if terminal is None and len(expected_pairs) and provenance_index == 0:
        raise ValueError("provenance must cover every expected boundary")
    if decision.status == "accepted" and len(selected) != len(expected_pairs):
        raise ValueError("accepted refinement must agree on every observed transition")
    return found, tuple(selected)


def _validate_provenance_entry(
    decision: RefineDecision,
    item: SamplingProvenance,
    boundary_index: int,
    pair: tuple[int, int],
    stage: str,
    pass_id: int,
    center: int,
    radius: int,
    stride: int,
    cameras: tuple[str, ...],
) -> None:
    if item.boundary_index != boundary_index:
        raise ValueError("provenance boundary order must be chronological and contiguous")
    if (item.from_subtask_index, item.to_subtask_index) != pair:
        raise ValueError("provenance transition must match the exact observed pair")
    if item.stage != stage or item.pass_id != pass_id:
        raise ValueError("provenance stages and pass IDs must match execution order")
    if item.request_center != center:
        raise ValueError("provenance request center does not match its semantic stage")
    if item.radius_frames != radius:
        raise ValueError("provenance radius does not match decision sampling context")
    if item.stride != stride:
        raise ValueError("provenance stride does not match degradation state")
    if item.cameras != cameras:
        raise ValueError("provenance cameras do not match degradation state")
    expected_grid = tuple(_evidence_indices(center, radius, stride, decision.frame_count))
    if tuple(sample.camera_key for sample in item.samples) != cameras or any(
        sample.frame_indices != expected_grid for sample in item.samples
    ):
        raise ValueError("provenance camera sampling grid is not recomputable")


def _ordered(found):
    return tuple(reason for reason in _REASON_ORDER if reason in found)


def _oom_failure(common, attempts, provenance, found):
    reasons = _ordered(set(found) | {"model_oom"})
    return RefineDecision(
        status="failed", attempts=tuple(attempts), provenance=tuple(provenance),
        reasons=reasons, annotation=None, candidate_annotation=None,
        failure_category="model_oom", **common,
    )

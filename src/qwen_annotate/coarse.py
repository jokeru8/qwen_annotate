"""Whole-episode coarse inference with deterministic two-pass agreement."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import AnnotationConfig
from .constraints import coarse_sequence_is_legal
from .lerobot import EpisodeInfo
from .models import CoarseResult
from .prompts import build_coarse_prompt
from .qwen_client import QwenClient
from .video import FrameSample, extract_frames, uniform_indices


ReviewReason = Literal[
    "coarse_sequence_disagreement",
    "illegal_coarse_sequence",
    "coarse_boundary_count",
    "coarse_boundary_order",
    "coarse_uncertain",
]

_REASON_ORDER: tuple[ReviewReason, ...] = (
    "coarse_sequence_disagreement",
    "illegal_coarse_sequence",
    "coarse_boundary_count",
    "coarse_boundary_order",
    "coarse_uncertain",
)


class _Completer(Protocol):
    async def complete(
        self,
        prompt: str,
        frames: list[FrameSample],
        response_type: type[CoarseResult],
    ) -> CoarseResult: ...


Sampler = Callable[[Path, str, list[int], float], list[FrameSample]]


class CoarseDecision(BaseModel):
    """Auditable outcome of two independent whole-episode model passes."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["coarse_done", "needs_review"]
    attempts: tuple[CoarseResult, CoarseResult]
    reasons: tuple[ReviewReason, ...]
    start_subtask_index: int | None = Field(default=None, ge=0)
    observed_subtask_indices: tuple[int, ...] = ()
    boundary_centers: tuple[int, ...] = ()
    sampled_frame_indices: tuple[tuple[int, ...], tuple[int, ...]]

    @model_validator(mode="after")
    def status_fields_are_consistent(self) -> "CoarseDecision":
        if self.reasons != tuple(reason for reason in _REASON_ORDER if reason in self.reasons):
            raise ValueError("reasons must be unique and in stable reason-code order")
        if any(not grid for grid in self.sampled_frame_indices):
            raise ValueError("each coarse pass must preserve at least one sampled frame")
        if self.status == "coarse_done":
            if self.reasons:
                raise ValueError("coarse_done cannot contain review reasons")
            if self.start_subtask_index is None or not self.observed_subtask_indices:
                raise ValueError("coarse_done requires an agreed start and observed sequence")
            if self.start_subtask_index != self.observed_subtask_indices[0]:
                raise ValueError("agreed start must equal the first observed subtask")
            if len(self.boundary_centers) != len(self.observed_subtask_indices) - 1:
                raise ValueError("boundary centers must match the agreed sequence")
        else:
            if not self.reasons:
                raise ValueError("needs_review requires at least one stable reason")
            if (
                self.start_subtask_index is not None
                or self.observed_subtask_indices
                or self.boundary_centers
            ):
                raise ValueError("needs_review must not expose an agreed success candidate")
        return self


def coarse_pass_indices(
    frame_count: int,
    source_fps: float,
    target_fps: float,
    max_frames: int,
    pass_id: int,
) -> list[int]:
    """Return one of two endpoint-preserving deterministic sparse grids.

    Pass zero is the canonical uniform grid. Pass one moves one interior point
    by one original frame whenever an alternative sorted grid exists. If every
    frame is selected (or there are only endpoint samples), the grids coincide.
    """
    if type(pass_id) is not int or pass_id not in (0, 1):
        raise ValueError("pass_id must be 0 or 1")
    base = uniform_indices(frame_count, source_fps, target_fps, max_frames)
    if pass_id == 0 or len(base) <= 2 or len(base) == frame_count:
        return base
    shifted = list(base)
    for index in range(1, len(shifted) - 1):
        if shifted[index] + 1 < shifted[index + 1]:
            shifted[index] += 1
            return shifted
        if shifted[index] - 1 > shifted[index - 1]:
            shifted[index] -= 1
            return shifted
    return shifted


async def run_coarse(
    config: AnnotationConfig,
    episode: EpisodeInfo,
    sampler: Sampler = extract_frames,
    client: _Completer | None = None,
) -> CoarseDecision:
    """Run two semantic passes and accept only deterministic agreement."""
    _validate_inputs(config, episode)
    source_fps = _read_source_fps(config.source)
    completer: _Completer = client if client is not None else QwenClient(
        endpoint=str(config.model.endpoint),
        api_key=config.model.api_key,
        model=config.model.name,
    )
    camera = config.primary_camera
    video = episode.videos[camera]
    attempts: list[CoarseResult] = []
    sampled_grids: list[tuple[int, ...]] = []

    for pass_id in (0, 1):
        indices = coarse_pass_indices(
            episode.length,
            source_fps,
            config.sampling.coarse_fps,
            config.sampling.coarse_max_frames,
            pass_id,
        )
        samples = sampler(video, camera, indices, source_fps)
        _validate_samples(samples, indices, camera)
        prompt = build_coarse_prompt(
            config,
            episode_index=episode.episode_index,
            frame_count=episode.length,
            pass_id=pass_id,
        )
        attempt = await completer.complete(prompt, samples, CoarseResult)
        if not isinstance(attempt, CoarseResult):
            raise TypeError("client.complete must return a CoarseResult")
        attempts.append(attempt)
        sampled_grids.append(tuple(indices))

    first, second = attempts
    found = _decision_reasons(first, second, config.mode, len(config.subtasks), episode.length)
    reasons = tuple(reason for reason in _REASON_ORDER if reason in found)
    common = {
        "attempts": (first, second),
        "reasons": reasons,
        "sampled_frame_indices": (sampled_grids[0], sampled_grids[1]),
    }
    if reasons:
        return _make_decision_preserving_attempts(
            status="needs_review",
            start_subtask_index=None,
            observed_subtask_indices=(),
            boundary_centers=(),
            **common,
        )

    centers = tuple(
        _positive_half_up(left.estimated_frame, right.estimated_frame)
        for left, right in zip(first.coarse_boundaries, second.coarse_boundaries)
    )
    return _make_decision_preserving_attempts(
        status="coarse_done",
        start_subtask_index=first.start_subtask_index,
        observed_subtask_indices=tuple(first.observed_subtask_indices),
        boundary_centers=centers,
        **common,
    )


def _make_decision_preserving_attempts(**fields: object) -> CoarseDecision:
    """Build without revalidating deliberately bypassed model responses.

    A caller can receive a ``CoarseResult.model_construct`` instance from an
    injected or future client. Re-running its Pydantic validators here would
    discard the audit result instead of quarantining it. All decision fields
    are produced locally, and the decision's own cross-field invariants are
    still executed explicitly.
    """
    decision = CoarseDecision.model_construct(**fields)
    decision.status_fields_are_consistent()
    return decision


def _validate_inputs(config: AnnotationConfig, episode: EpisodeInfo) -> None:
    if not isinstance(config, AnnotationConfig):
        raise TypeError("config must be an AnnotationConfig")
    if not isinstance(episode, EpisodeInfo):
        raise TypeError("episode must be an EpisodeInfo")
    if type(episode.episode_index) is not int or episode.episode_index < 0:
        raise ValueError("episode_index must be a non-negative integer")
    if type(episode.length) is not int or episode.length < 1:
        raise ValueError("episode length must be a positive integer")
    if config.primary_camera not in episode.videos:
        raise ValueError(f"configured primary camera {config.primary_camera!r} is missing from episode")
    video = episode.videos[config.primary_camera]
    if not isinstance(video, Path):
        raise TypeError("primary camera video path must be a Path")
    target_fps = config.sampling.coarse_fps
    if isinstance(target_fps, bool) or not isinstance(target_fps, (int, float)):
        raise ValueError("coarse_fps must be a positive finite number")
    if not math.isfinite(float(target_fps)) or target_fps <= 0:
        raise ValueError("coarse_fps must be a positive finite number")
    if type(config.sampling.coarse_max_frames) is not int or config.sampling.coarse_max_frames < 2:
        raise ValueError("coarse_max_frames must be an integer of at least 2")


def _read_source_fps(source: Path) -> float:
    info_path = source / "meta" / "info.json"
    try:
        payload = json.loads(
            info_path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid constant {value}")),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"unable to read source fps from {info_path}: {exc}") from exc
    value = payload.get("fps") if isinstance(payload, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("source metadata fps must be a positive finite number")
    fps = float(value)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("source metadata fps must be a positive finite number")
    return fps


def _validate_samples(samples: object, requested: list[int], camera: str) -> None:
    if not isinstance(samples, list) or len(samples) != len(requested):
        raise ValueError("sampler evidence must contain exactly the requested frames")
    actual: list[int] = []
    for sample in samples:
        if not isinstance(sample, FrameSample):
            raise ValueError("sampler evidence must contain FrameSample values")
        if sample.camera_key != camera:
            raise ValueError("sampler evidence camera must match the requested camera")
        actual.append(sample.frame_index)
    if actual != requested:
        raise ValueError("sampler evidence indices must match requested order without duplicates")


def _decision_reasons(
    first: CoarseResult,
    second: CoarseResult,
    mode: Literal["complete", "dagger_patch"],
    subtask_count: int,
    episode_length: int,
) -> set[ReviewReason]:
    reasons: set[ReviewReason] = set()
    first_sequence = _strict_int_list(getattr(first, "observed_subtask_indices", None))
    second_sequence = _strict_int_list(getattr(second, "observed_subtask_indices", None))
    first_start = getattr(first, "start_subtask_index", None)
    second_start = getattr(second, "start_subtask_index", None)

    if first_start != second_start or first_sequence != second_sequence:
        reasons.add("coarse_sequence_disagreement")
    for attempt, sequence in ((first, first_sequence), (second, second_sequence)):
        if (
            sequence is None
            or type(getattr(attempt, "start_subtask_index", None)) is not int
            or not sequence
            or attempt.start_subtask_index != sequence[0]
            or not coarse_sequence_is_legal(sequence, mode, subtask_count)
        ):
            reasons.add("illegal_coarse_sequence")
        _validate_attempt_boundaries(attempt, sequence, episode_length, reasons)
        uncertainties = getattr(attempt, "uncertainties", None)
        if not isinstance(uncertainties, list) or bool(uncertainties):
            reasons.add("coarse_uncertain")
    return reasons


def _validate_attempt_boundaries(
    attempt: CoarseResult,
    sequence: list[int] | None,
    episode_length: int,
    reasons: set[ReviewReason],
) -> None:
    boundaries = getattr(attempt, "coarse_boundaries", None)
    expected_count = len(sequence) - 1 if sequence else 0
    if not isinstance(boundaries, list) or len(boundaries) != expected_count:
        reasons.add("coarse_boundary_count")
    if not isinstance(boundaries, list):
        return
    previous_frame = 0
    for index, item in enumerate(boundaries):
        left = getattr(item, "from_subtask_index", None)
        right = getattr(item, "to_subtask_index", None)
        frame = getattr(item, "estimated_frame", None)
        expected_left = sequence[index] if sequence is not None and index < len(sequence) else None
        expected_right = sequence[index + 1] if sequence is not None and index + 1 < len(sequence) else None
        if (
            type(left) is not int
            or type(right) is not int
            or right != left + 1
            or left != expected_left
            or right != expected_right
        ):
            reasons.add("coarse_boundary_order")
        if type(frame) is not int or not (1 <= frame < episode_length) or frame <= previous_frame:
            reasons.add("coarse_boundary_order")
        if type(frame) is int:
            previous_frame = frame


def _strict_int_list(value: object) -> list[int] | None:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        return None
    return value


def _positive_half_up(left: int, right: int) -> int:
    """Round the mean of positive integer frames to nearest, ties upward."""
    return (left + right + 1) // 2

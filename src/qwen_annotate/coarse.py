"""Whole-episode coarse inference with deterministic two-pass agreement."""

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

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, field_validator, model_validator

from .config import AnnotationConfig
from .constraints import coarse_sequence_is_legal
from .lerobot import EpisodeInfo
from .models import CoarseBoundary, CoarseResult
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


@dataclass(frozen=True)
class _SourceState:
    root: Path
    info_entry: Path
    video_entry: Path
    video_resolved: Path
    fps: float
    info_identity: tuple[int, ...]
    video_identity: tuple[int, ...]
    info_sha256: str


class _ImmutableCoarseResult(CoarseResult):
    """A CoarseResult-compatible audit snapshot with tuple-backed sequences."""

    observed_subtask_indices: tuple[int, ...]
    coarse_boundaries: tuple[CoarseBoundary, ...]
    uncertainties: tuple[str, ...] = ()


class CoarseDecision(BaseModel):
    """Auditable outcome of two independent whole-episode model passes."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    mode: Literal["complete", "dagger_patch"]
    subtask_count: int = Field(ge=1, strict=True)
    frame_count: int = Field(ge=1, strict=True)
    status: Literal["coarse_done", "needs_review"]
    attempts: tuple[
        SkipValidation[_ImmutableCoarseResult],
        SkipValidation[_ImmutableCoarseResult],
    ]
    reasons: tuple[ReviewReason, ...]
    start_subtask_index: int | None = Field(default=None, ge=0)
    observed_subtask_indices: tuple[int, ...] = ()
    boundary_centers: tuple[int, ...] = ()
    sampled_frame_indices: tuple[tuple[int, ...], tuple[int, ...]]

    @field_validator("attempts", mode="before")
    @classmethod
    def preserve_typed_bypassed_attempts(cls, value: object) -> object:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return value
        parsed: list[_ImmutableCoarseResult] = []
        for attempt in value:
            if isinstance(attempt, CoarseResult):
                _validate_attempt_shape(attempt)
                parsed.append(_freeze_attempt(attempt))
            elif isinstance(attempt, dict):
                restored = _attempt_from_mapping(attempt)
                _validate_attempt_shape(restored)
                parsed.append(_freeze_attempt(restored))
            else:
                raise ValueError("attempts must contain exactly two CoarseResult values")
        return tuple(parsed)

    @model_validator(mode="after")
    def status_fields_are_consistent(self) -> "CoarseDecision":
        for attempt in self.attempts:
            _validate_attempt_shape(attempt)
        for grid in self.sampled_frame_indices:
            _validate_sampled_grid(grid, self.frame_count)
        _validate_grid_independence(self.sampled_frame_indices, self.frame_count)

        recomputed_set = _decision_reasons(
            self.attempts[0],
            self.attempts[1],
            self.mode,
            self.subtask_count,
            self.frame_count,
        )
        recomputed = tuple(reason for reason in _REASON_ORDER if reason in recomputed_set)
        if self.status == "coarse_done":
            if recomputed or self.reasons:
                raise ValueError("coarse_done requires two legal, certain, sequence-agreeing attempts")
            if self.start_subtask_index is None or not self.observed_subtask_indices:
                raise ValueError("coarse_done requires an agreed start and observed sequence")
            first, second = self.attempts
            agreed_sequence = tuple(first.observed_subtask_indices)
            if (
                self.start_subtask_index != first.start_subtask_index
                or self.start_subtask_index != second.start_subtask_index
                or self.observed_subtask_indices != agreed_sequence
                or self.observed_subtask_indices != tuple(second.observed_subtask_indices)
                or not coarse_sequence_is_legal(list(agreed_sequence), self.mode, self.subtask_count)
            ):
                raise ValueError("exposed coarse success sequence must exactly match both attempts")
            expected_centers = tuple(
                _positive_half_up(left.estimated_frame, right.estimated_frame)
                for left, right in zip(first.coarse_boundaries, second.coarse_boundaries)
            )
            if self.boundary_centers != expected_centers:
                raise ValueError("boundary centers must be exact pairwise half-up averages")
            if any(
                type(center) is not int or not (1 <= center < self.frame_count)
                for center in self.boundary_centers
            ) or any(
                left >= right
                for left, right in zip(self.boundary_centers, self.boundary_centers[1:])
            ):
                raise ValueError("boundary centers must be ordered valid transition frames")
        else:
            if not recomputed or self.reasons != recomputed:
                raise ValueError("needs_review reasons must exactly equal recomputed audit reasons")
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

    Pass zero is the canonical uniform grid. Pass one samples approximately a
    half base interval later across the interior. If every frame is selected
    (or there are only endpoint samples), the grids coincide.
    """
    if type(pass_id) is not int or pass_id not in (0, 1):
        raise ValueError("pass_id must be 0 or 1")
    base = uniform_indices(frame_count, source_fps, target_fps, max_frames)
    if pass_id == 0 or len(base) <= 2 or len(base) == frame_count:
        return base
    span = frame_count - 1
    sample_count = len(base)
    shifted = [0]
    for index in range(1, sample_count - 1):
        ideal = round((index + 0.5) * span / (sample_count - 1))
        lower = shifted[-1] + 1
        upper = span - (sample_count - 1 - index)
        shifted.append(min(upper, max(lower, ideal)))
    shifted.append(span)
    required = _minimum_distinct_interior(frame_count, sample_count)
    while len(set(shifted[1:-1]) - set(base[1:-1])) < required:
        changed = False
        base_values = set(base[1:-1])
        for index in range(1, sample_count - 1):
            lower = shifted[index - 1] + 1
            upper = shifted[index + 1] - 1
            for candidate in (shifted[index] - 1, shifted[index] + 1, lower, upper):
                if lower <= candidate <= upper and candidate not in base_values:
                    shifted[index] = candidate
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return shifted


async def run_coarse(
    config: AnnotationConfig,
    episode: EpisodeInfo,
    sampler: Sampler = extract_frames,
    client: _Completer | None = None,
    *,
    source_fps: float | None = None,
    expected_source_fingerprint: str | None = None,
) -> CoarseDecision:
    """Run two semantic passes and accept only deterministic agreement."""
    _validate_inputs(config, episode)
    source = _prepare_source(
        config,
        episode,
        authoritative_fps=source_fps,
        expected_fingerprint=expected_source_fingerprint,
    )
    effective_fps = source.fps
    completer: _Completer = client if client is not None else QwenClient(
        endpoint=str(config.model.endpoint),
        api_key=config.model.api_key,
        model=config.model.name,
    )
    camera = config.primary_camera
    attempts: list[CoarseResult] = []
    grids = [
        coarse_pass_indices(
            episode.length,
            effective_fps,
            config.sampling.coarse_fps,
            config.sampling.coarse_max_frames,
            pass_id,
        )
        for pass_id in (0, 1)
    ]
    union = sorted(set(grids[0]) | set(grids[1]))
    union_samples = await asyncio.to_thread(
        sampler,
        source.video_resolved,
        camera,
        union,
        effective_fps,
    )
    _validate_samples(union_samples, union, camera, effective_fps)
    _assert_source_unchanged(source, episode, expected_source_fingerprint)
    by_index = {sample.frame_index: sample for sample in union_samples}

    for pass_id, indices in enumerate(grids):
        samples = [by_index[index] for index in indices]
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
        _assert_source_unchanged(source, episode, expected_source_fingerprint)

    first, second = attempts
    found = _decision_reasons(first, second, config.mode, len(config.subtasks), episode.length)
    reasons = tuple(reason for reason in _REASON_ORDER if reason in found)
    common = {
        "mode": config.mode,
        "subtask_count": len(config.subtasks),
        "frame_count": episode.length,
        "attempts": (first, second),
        "reasons": reasons,
        "sampled_frame_indices": (tuple(grids[0]), tuple(grids[1])),
    }
    if reasons:
        return CoarseDecision(
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
    return CoarseDecision(
        status="coarse_done",
        start_subtask_index=first.start_subtask_index,
        observed_subtask_indices=tuple(first.observed_subtask_indices),
        boundary_centers=centers,
        **common,
    )


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


def _prepare_source(
    config: AnnotationConfig,
    episode: EpisodeInfo,
    *,
    authoritative_fps: float | None,
    expected_fingerprint: str | None,
) -> _SourceState:
    try:
        root = config.source.resolve(strict=True)
    except OSError:
        raise ValueError("configured source root must exist") from None
    if not root.is_dir():
        raise ValueError("configured source root must be a directory")
    info_entry = root / "meta" / "info.json"
    video_entry, video_resolved = _contained_regular_file(
        root,
        episode.videos[config.primary_camera],
        "primary video",
    )
    info_entry, _ = _contained_regular_file(root, info_entry, "source info.json")
    info_identity, info_sha256 = _capture_file(info_entry, with_digest=True)
    metadata_fps = _read_source_fps(info_entry)
    info_identity_after_parse, info_sha256_after_parse = _capture_file(
        info_entry, with_digest=True
    )
    if (
        info_identity_after_parse != info_identity
        or info_sha256_after_parse != info_sha256
    ):
        raise ValueError("source evidence changed during coarse annotation")
    if authoritative_fps is not None:
        if isinstance(authoritative_fps, bool) or not isinstance(authoritative_fps, (int, float)):
            raise ValueError("authoritative source_fps must be a positive finite number")
        effective = float(authoritative_fps)
        if not math.isfinite(effective) or effective <= 0:
            raise ValueError("authoritative source_fps must be a positive finite number")
        if not math.isclose(effective, metadata_fps, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("authoritative source_fps does not match current source metadata")
    else:
        effective = metadata_fps
    video_identity, _ = _capture_file(video_entry, with_digest=False)
    if expected_fingerprint is not None:
        _validate_expected_fingerprint(expected_fingerprint)
        if _source_fingerprint(root, episode) != expected_fingerprint:
            raise ValueError("expected source fingerprint does not match current episode source")
    return _SourceState(
        root=root,
        info_entry=info_entry,
        video_entry=video_entry,
        video_resolved=video_resolved,
        fps=effective,
        info_identity=info_identity,
        video_identity=video_identity,
        info_sha256=info_sha256,
    )


def _contained_regular_file(root: Path, path: Path, label: str) -> tuple[Path, Path]:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        raise ValueError(f"{label} must be inside source root") from None
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} must not use a symlink")
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        raise ValueError(f"{label} must resolve inside source root") from None
    try:
        file_stat = os.stat(lexical, follow_symlinks=False)
    except OSError:
        raise ValueError(f"{label} must exist") from None
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return lexical, resolved


def _identity(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _capture_file(path: Path, *, with_digest: bool) -> tuple[tuple[int, ...], str]:
    if path.is_symlink():
        raise ValueError("source evidence changed during coarse annotation")
    try:
        before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("source evidence changed during coarse annotation")
        data = path.read_bytes() if with_digest else b""
        after = os.stat(path, follow_symlinks=False)
    except OSError:
        raise ValueError("source evidence changed during coarse annotation") from None
    if _identity(before) != _identity(after):
        raise ValueError("source evidence changed during coarse annotation")
    digest = hashlib.sha256(data).hexdigest() if with_digest else ""
    return _identity(after), digest


def _assert_source_unchanged(
    source: _SourceState,
    episode: EpisodeInfo,
    expected_fingerprint: str | None,
) -> None:
    try:
        current_video_resolved = source.video_entry.resolve(strict=True)
    except OSError:
        raise ValueError("source evidence changed during coarse annotation") from None
    if current_video_resolved != source.video_resolved:
        raise ValueError("source evidence changed during coarse annotation")
    info_identity, info_sha256 = _capture_file(source.info_entry, with_digest=True)
    video_identity, _ = _capture_file(source.video_entry, with_digest=False)
    if (
        info_identity != source.info_identity
        or info_sha256 != source.info_sha256
        or video_identity != source.video_identity
    ):
        raise ValueError("source evidence changed during coarse annotation")
    if expected_fingerprint is not None and _source_fingerprint(source.root, episode) != expected_fingerprint:
        raise ValueError("source fingerprint changed during coarse annotation")


def _validate_expected_fingerprint(value: object) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("expected source fingerprint must be lowercase SHA-256")


def _source_fingerprint(root: Path, episode: EpisodeInfo) -> str:
    from .workspace import compute_source_fingerprint

    return compute_source_fingerprint(root, episode)


def _read_source_fps(info_path: Path) -> float:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    try:
        payload = json.loads(
            info_path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError("non-standard JSON constant")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError("source info.json must contain valid standard JSON with unique keys") from None
    if not isinstance(payload, dict):
        raise ValueError("source info.json must contain a JSON object")
    value = payload.get("fps")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("source metadata fps must be a positive finite number")
    fps = float(value)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("source metadata fps must be a positive finite number")
    return fps


def _validate_samples(
    samples: object,
    requested: list[int],
    camera: str,
    source_fps: float,
) -> None:
    if not isinstance(samples, list) or len(samples) != len(requested):
        raise ValueError("sampler evidence must contain exactly the requested frames")
    actual: list[int] = []
    for sample in samples:
        if not isinstance(sample, FrameSample):
            raise ValueError("sampler evidence must contain FrameSample values")
        if sample.camera_key != camera:
            raise ValueError("sampler evidence camera must match the requested camera")
        expected_timestamp = sample.frame_index / source_fps
        if not math.isclose(
            sample.timestamp_seconds,
            expected_timestamp,
            rel_tol=1e-9,
            abs_tol=1e-7,
        ):
            raise ValueError(
                "sampler evidence timestamp must match frame_index/source_fps within 1e-7 seconds"
            )
        actual.append(sample.frame_index)
    if actual != requested:
        raise ValueError("sampler evidence indices must match requested order without duplicates")


def _validate_sampled_grid(grid: tuple[int, ...], frame_count: int) -> None:
    if not grid:
        raise ValueError("each sampled frame grid must be nonempty")
    if any(type(frame) is not int for frame in grid):
        raise ValueError("sampled frame indices must be strict integers")
    if grid[0] != 0 or grid[-1] != frame_count - 1:
        raise ValueError("sampled frame grids must preserve both episode endpoints")
    if any(frame < 0 or frame >= frame_count for frame in grid):
        raise ValueError("sampled frame indices must be within the episode")
    if any(left >= right for left, right in zip(grid, grid[1:])):
        raise ValueError("sampled frame indices must be sorted and unique")


def _minimum_distinct_interior(frame_count: int, sample_count: int) -> int:
    interiors = max(0, sample_count - 2)
    available_outside_first_grid = max(0, frame_count - sample_count)
    possible_new = min(interiors, available_outside_first_grid)
    return (possible_new + 1) // 2


def _validate_grid_independence(
    grids: tuple[tuple[int, ...], tuple[int, ...]], frame_count: int
) -> None:
    first, second = grids
    if len(first) != len(second):
        raise ValueError("sampled grids must have the same capped sample count")
    required = _minimum_distinct_interior(frame_count, len(first))
    distinct = len(set(second[1:-1]) - set(first[1:-1]))
    if distinct < required:
        raise ValueError(
            "sampled grids are not sufficiently independent for the available sparse frames"
        )


def _freeze_attempt(attempt: CoarseResult) -> _ImmutableCoarseResult:
    boundaries = tuple(
        CoarseBoundary.model_construct(
            from_subtask_index=getattr(boundary, "from_subtask_index", None),
            to_subtask_index=getattr(boundary, "to_subtask_index", None),
            estimated_frame=getattr(boundary, "estimated_frame", None),
            evidence=getattr(boundary, "evidence", None),
        )
        if isinstance(boundary, CoarseBoundary)
        else boundary
        for boundary in getattr(attempt, "coarse_boundaries", [])
    )
    return _ImmutableCoarseResult.model_construct(
        start_subtask_index=getattr(attempt, "start_subtask_index", None),
        observed_subtask_indices=tuple(getattr(attempt, "observed_subtask_indices", [])),
        coarse_boundaries=boundaries,
        confidence=getattr(attempt, "confidence", None),
        uncertainties=tuple(getattr(attempt, "uncertainties", [])),
    )


def _attempt_from_mapping(value: dict[object, object]) -> CoarseResult:
    expected = {
        "start_subtask_index",
        "observed_subtask_indices",
        "coarse_boundaries",
        "confidence",
        "uncertainties",
    }
    if set(value) != expected:
        raise ValueError("serialized coarse attempt has missing or extra fields")
    raw_boundaries = value["coarse_boundaries"]
    if not isinstance(raw_boundaries, list):
        raise ValueError("serialized coarse boundaries must be a list")
    boundaries: list[CoarseBoundary] = []
    boundary_fields = {
        "from_subtask_index",
        "to_subtask_index",
        "estimated_frame",
        "evidence",
    }
    for raw in raw_boundaries:
        if not isinstance(raw, dict) or set(raw) != boundary_fields:
            raise ValueError("serialized coarse boundary has missing or extra fields")
        boundaries.append(CoarseBoundary.model_construct(**raw))
    return CoarseResult.model_construct(
        start_subtask_index=value["start_subtask_index"],
        observed_subtask_indices=value["observed_subtask_indices"],
        coarse_boundaries=boundaries,
        confidence=value["confidence"],
        uncertainties=value["uncertainties"],
    )


def _validate_attempt_shape(attempt: object) -> None:
    """Reject malformed audit containers while leaving semantic issues reviewable."""
    if not isinstance(attempt, CoarseResult):
        raise ValueError("attempts must contain CoarseResult values")
    if type(getattr(attempt, "start_subtask_index", None)) is not int:
        raise ValueError("attempt start_subtask_index must be a strict integer")
    observed = getattr(attempt, "observed_subtask_indices", None)
    if not isinstance(observed, (list, tuple)) or any(type(index) is not int for index in observed):
        raise ValueError("attempt observed_subtask_indices must be a sequence of strict integers")
    boundaries = getattr(attempt, "coarse_boundaries", None)
    if not isinstance(boundaries, (list, tuple)):
        raise ValueError("attempt coarse_boundaries must be a sequence")
    for boundary in boundaries:
        if not isinstance(boundary, CoarseBoundary):
            raise ValueError("attempt boundaries must be CoarseBoundary values")
        if any(
            type(getattr(boundary, field, None)) is not int
            for field in ("from_subtask_index", "to_subtask_index", "estimated_frame")
        ):
            raise ValueError("attempt boundary indices and frame must be strict integers")
        evidence = getattr(boundary, "evidence", None)
        if not isinstance(evidence, str) or not evidence:
            raise ValueError("attempt boundary evidence must be a nonempty string")
    confidence = getattr(attempt, "confidence", None)
    if type(confidence) is not float or not math.isfinite(confidence) or not (0 <= confidence <= 1):
        raise ValueError("attempt confidence must be a finite float in [0, 1]")
    uncertainties = getattr(attempt, "uncertainties", None)
    if not isinstance(uncertainties, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in uncertainties
    ):
        raise ValueError("attempt uncertainties must be a sequence of nonempty strings")


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
        if not isinstance(uncertainties, (list, tuple)) or bool(uncertainties):
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
    if not isinstance(boundaries, (list, tuple)) or len(boundaries) != expected_count:
        reasons.add("coarse_boundary_count")
    if not isinstance(boundaries, (list, tuple)):
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
    if not isinstance(value, (list, tuple)) or any(type(item) is not int for item in value):
        return None
    return list(value)


def _positive_half_up(left: int, right: int) -> int:
    """Round the mean of positive integer frames to nearest, ties upward."""
    return (left + right + 1) // 2

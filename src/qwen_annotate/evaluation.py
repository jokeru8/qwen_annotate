"""Offline golden-set and non-destructive synthetic DAgger evaluation."""

from __future__ import annotations

import json
import math
import os
import secrets
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import AnnotationConfig
from .constraints import DETERMINISTIC_REJECTION_REASONS, validate_annotation
from .lerobot import DatasetIndex, inspect_dataset, probe_video
from .models import FinalAnnotation
from .workspace import (
    EpisodeRecord,
    RunManifest,
    WorkspaceStore,
    compute_run_fingerprint,
    compute_source_fingerprint,
)


EvaluationStatus = Literal["pending", "coarse_done", "refine_done", "accepted", "needs_review", "failed"]

class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EvaluationMetrics(_FrozenModel):
    """Metrics with explicit counts for every reported rate and empty case."""

    episode_count: int = Field(ge=0)
    predicted_episode_count: int = Field(ge=0)
    missing_prediction_count: int = Field(ge=0)
    extra_prediction_count: int = Field(ge=0)
    boundary_count: int = Field(ge=0)
    aligned_boundary_count: int = Field(ge=0)
    transition_mismatch_count: int = Field(ge=0)
    start_index_evaluated_count: int = Field(ge=0)
    start_index_correct_count: int = Field(ge=0)
    start_subtask_index_accuracy: float | None
    median_absolute_error_frames: float | None
    p90_absolute_error_frames: float | None
    median_absolute_error_seconds: float | None
    p90_absolute_error_seconds: float | None
    accepted_count: int = Field(ge=0)
    needs_review_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    status_denominator: int = Field(ge=0)
    accepted_coverage: float = Field(ge=0, le=1)
    needs_review_rate: float = Field(ge=0, le=1)
    failed_rate: float = Field(ge=0, le=1)
    constraint_violation_count: int = Field(ge=0)
    constraint_violation_blocked_count: int = Field(ge=0)
    constraint_blocking_rate: float = Field(ge=0, le=1)
    false_accept_count: int = Field(ge=0)
    obvious_error_threshold_seconds: float = Field(gt=0)
    fps: float = Field(gt=0)

    @field_validator(
        "start_subtask_index_accuracy", "median_absolute_error_frames",
        "p90_absolute_error_frames", "median_absolute_error_seconds",
        "p90_absolute_error_seconds", "obvious_error_threshold_seconds", "fps",
    )
    @classmethod
    def finite_metrics(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("evaluation metrics must be finite")
        return value


class DaggerView(_FrozenModel):
    """A virtual left-closed/right-open slice; it never materializes video."""

    source_episode: int = Field(ge=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    expected_start_subtask_index: int = Field(ge=0)
    expected_boundaries_relative: list[int]
    kind: Literal["suffix", "singleton"]


class DaggerPrediction(_FrozenModel):
    start_subtask_index: int = Field(ge=0)
    boundaries: list[int]
    status: EvaluationStatus
    constraint_violated: bool = False


def evaluate_boundaries(
    predicted: Mapping[int, Sequence[int]],
    golden: Mapping[int, Sequence[int]],
    statuses: Mapping[int, EvaluationStatus | str],
    fps: float,
    *,
    predicted_start_indices: Mapping[int, int] | None = None,
    golden_start_indices: Mapping[int, int] | None = None,
    constraint_violations: Mapping[int, bool] | None = None,
    obvious_error_threshold_seconds: float = 1.0,
) -> EvaluationMetrics:
    """Align by episode and transition, then compute deterministic exact metrics.

    An episode with a different transition count is excluded from boundary-error
    aggregation rather than being partially ``zip``-aligned. It is simultaneously
    counted as a constraint violation and, if accepted, as a false accept.
    """
    fps_value = _positive_finite_number(fps, "fps")
    threshold = _positive_finite_number(obvious_error_threshold_seconds, "obvious_error_threshold_seconds")
    predicted_values = _boundary_map(predicted, "predicted")
    golden_values = _boundary_map(golden, "golden")
    status_values = _status_map(statuses)
    golden_keys = set(golden_values)
    if set(status_values) != golden_keys:
        raise ValueError("statuses must contain exactly the golden episode indices")

    predicted_starts = _index_map(predicted_start_indices or {}, "predicted_start_indices")
    golden_starts = _index_map(golden_start_indices or {}, "golden_start_indices")
    if predicted_start_indices is not None and not set(predicted_starts).issubset(set(predicted_values)):
        raise ValueError("predicted start indices require a prediction for the same episode")
    if golden_start_indices is not None and set(golden_starts) != golden_keys:
        raise ValueError("golden start indices must contain exactly the golden episode indices")
    violation_values = dict(constraint_violations or {})
    if any(type(key) is not int or key < 0 or type(value) is not bool for key, value in violation_values.items()):
        raise ValueError("constraint_violations must map non-negative integer episodes to booleans")

    common = sorted(golden_keys & set(predicted_values))
    errors: list[int] = []
    mismatches: set[int] = set()
    episode_errors: dict[int, list[int]] = {}
    for episode in common:
        expected = golden_values[episode]
        actual = predicted_values[episode]
        if len(expected) != len(actual):
            mismatches.add(episode)
            continue
        current = [abs(left - right) for left, right in zip(actual, expected)]
        errors.extend(current)
        episode_errors[episode] = current

    start_evaluated = sorted(set(predicted_starts) & golden_keys) if golden_start_indices is not None else []
    start_correct = sum(predicted_starts[index] == golden_starts[index] for index in start_evaluated)
    accepted = sum(value == "accepted" for value in status_values.values())
    review = sum(value == "needs_review" for value in status_values.values())
    failed = sum(value == "failed" for value in status_values.values())
    denominator = len(golden_values)

    violation_episodes = {index for index, value in violation_values.items() if value and index in golden_keys}
    violation_episodes.update(mismatches)
    blocked = sum(status_values[index] != "accepted" for index in violation_episodes)
    false_accepts = 0
    for episode, status in status_values.items():
        if status != "accepted":
            continue
        obvious = (
            episode not in predicted_values
            or episode in violation_episodes
            or (
                golden_start_indices is not None
                and (episode not in predicted_starts or predicted_starts[episode] != golden_starts[episode])
            )
            or any(error / fps_value > threshold for error in episode_errors.get(episode, []))
        )
        false_accepts += int(obvious)

    median_frames = _median(errors)
    p90_frames = _nearest_rank(errors, 0.90)
    return EvaluationMetrics(
        episode_count=denominator,
        predicted_episode_count=len(golden_keys & set(predicted_values)),
        missing_prediction_count=len(golden_keys - set(predicted_values)),
        extra_prediction_count=len(set(predicted_values) - golden_keys),
        boundary_count=sum(len(value) for value in golden_values.values()),
        aligned_boundary_count=len(errors),
        transition_mismatch_count=len(mismatches),
        start_index_evaluated_count=len(start_evaluated),
        start_index_correct_count=start_correct,
        start_subtask_index_accuracy=(start_correct / len(start_evaluated) if start_evaluated else None),
        median_absolute_error_frames=median_frames,
        p90_absolute_error_frames=p90_frames,
        median_absolute_error_seconds=(median_frames / fps_value if median_frames is not None else None),
        p90_absolute_error_seconds=(p90_frames / fps_value if p90_frames is not None else None),
        accepted_count=accepted,
        needs_review_count=review,
        failed_count=failed,
        status_denominator=denominator,
        accepted_coverage=(accepted / denominator if denominator else 0.0),
        needs_review_rate=(review / denominator if denominator else 0.0),
        failed_rate=(failed / denominator if denominator else 0.0),
        constraint_violation_count=len(violation_episodes),
        constraint_violation_blocked_count=blocked,
        constraint_blocking_rate=(blocked / len(violation_episodes) if violation_episodes else 1.0),
        false_accept_count=false_accepts,
        obvious_error_threshold_seconds=threshold,
        fps=fps_value,
    )


def make_dagger_views(
    golden_boundaries: Mapping[int, Sequence[int]],
    episode_lengths: Mapping[int, int],
    min_segment_frames: int,
) -> list[DaggerView]:
    """Build deterministic virtual suffix and early-ending singleton views."""
    boundaries = _boundary_map(golden_boundaries, "golden_boundaries")
    lengths = _length_map(episode_lengths)
    if set(boundaries) != set(lengths):
        raise ValueError("episode_lengths must contain exactly the golden episode indices")
    minimum = _positive_integer(min_segment_frames, "min_segment_frames")
    result: list[DaggerView] = []
    for episode in sorted(boundaries):
        length = lengths[episode]
        points = [0, *boundaries[episode], length]
        if any(right <= left for left, right in zip(points, points[1:])):
            raise ValueError(f"golden boundaries are out of range for episode {episode}")
        for subtask_index, (left, right) in enumerate(zip(points, points[1:])):
            start = left + (right - left) // 2
            relative = [value - start for value in boundaries[episode] if value > start]
            suffix_segments = [0, *relative, length - start]
            if start < length and all(
                next_point - point >= minimum for point, next_point in zip(suffix_segments, suffix_segments[1:])
            ):
                result.append(DaggerView(
                    source_episode=episode,
                    start_frame=start,
                    end_frame=length,
                    expected_start_subtask_index=subtask_index,
                    expected_boundaries_relative=relative,
                    kind="suffix",
                ))
            singleton_end = right - minimum
            has_next_subtask = subtask_index < len(boundaries[episode])
            if has_next_subtask and singleton_end - start >= minimum:
                result.append(DaggerView(
                    source_episode=episode,
                    start_frame=start,
                    end_frame=singleton_end,
                    expected_start_subtask_index=subtask_index,
                    expected_boundaries_relative=[],
                    kind="singleton",
                ))
    return result


def evaluate_dagger(
    views: Sequence[DaggerView],
    *,
    sampler: Callable[[int, int, int], object],
    inference: Callable[[object], DaggerPrediction],
    fps: float,
    min_segment_frames: int = 1,
    obvious_error_threshold_seconds: float = 1.0,
) -> EvaluationMetrics:
    """Evaluate virtual views while keeping frame-range sampling explicit."""
    predicted: dict[int, list[int]] = {}
    golden: dict[int, list[int]] = {}
    statuses: dict[int, str] = {}
    predicted_starts: dict[int, int] = {}
    golden_starts: dict[int, int] = {}
    violations: dict[int, bool] = {}
    minimum = _positive_integer(min_segment_frames, "min_segment_frames")
    for index, raw_view in enumerate(views):
        view = DaggerView.model_validate(raw_view)
        if view.end_frame <= view.start_frame:
            raise ValueError("DAgger view end_frame must exceed start_frame")
        evidence = sampler(view.source_episode, view.start_frame, view.end_frame)
        prediction = DaggerPrediction.model_validate(inference(evidence))
        golden[index] = list(view.expected_boundaries_relative)
        predicted[index] = list(prediction.boundaries)
        statuses[index] = prediction.status
        golden_starts[index] = view.expected_start_subtask_index
        predicted_starts[index] = prediction.start_subtask_index
        violations[index] = prediction.constraint_violated or _dagger_prediction_violates(
            view, prediction, minimum
        )
    return evaluate_boundaries(
        predicted, golden, statuses, fps,
        predicted_start_indices=predicted_starts,
        golden_start_indices=golden_starts,
        constraint_violations=violations,
        obvious_error_threshold_seconds=obvious_error_threshold_seconds,
    )


def evaluate_complete(
    work_dir: Path,
    golden_dataset: Path,
    *,
    obvious_error_threshold_seconds: float = 1.0,
) -> EvaluationMetrics:
    """Load a workspace and a reference-shaped golden dataset, then evaluate."""
    work = Path(work_dir).resolve()
    golden_root = Path(golden_dataset).resolve()
    store = WorkspaceStore(work)
    manifest = store._load_manifest()  # secure workspace peer loader
    summary = store.summary()
    if summary["total"] != manifest.total_episodes:
        raise ValueError("workspace episode count does not match manifest")
    if manifest.mode != "complete":
        raise ValueError("evaluate_complete requires a complete workspace manifest")
    annotations = _load_golden(golden_root, manifest)
    predicted: dict[int, list[int]] = {}
    predicted_starts: dict[int, int] = {}
    statuses: dict[int, str] = {}
    violations: dict[int, bool] = {}
    records: dict[int, EpisodeRecord] = {}
    expected_indices = set(range(manifest.total_episodes))
    if set(annotations.boundaries) != expected_indices:
        raise ValueError("golden annotation episode indices do not match the workspace manifest")
    for index in range(manifest.total_episodes):
        record = store.load_episode(index)
        if record.episode_index != index:
            raise ValueError("workspace episode index does not match its filename")
        if record.run_fingerprint != manifest.run_fingerprint:
            raise ValueError("episode run fingerprint does not match workspace manifest")
        records[index] = record
        statuses[index] = record.status
        if record.final_annotation is None:
            continue
        annotation = record.final_annotation
        predicted[index] = list(annotation.boundaries)
        predicted_starts[index] = annotation.start_subtask_index
        violations[index] = bool(validate_annotation(
            annotation, "complete", len(manifest.subtasks), manifest.episode_lengths[index],
            manifest.min_segment_frames,
        )) or _record_has_deterministic_rejection(record)
    for index in expected_indices - set(violations):
        violations[index] = _record_has_deterministic_rejection(records[index])
    _validate_source_provenance(work, manifest, records)
    return evaluate_boundaries(
        predicted,
        annotations.boundaries,
        statuses,
        annotations.fps,
        predicted_start_indices=predicted_starts,
        golden_start_indices=annotations.start_indices,
        constraint_violations=violations,
        obvious_error_threshold_seconds=obvious_error_threshold_seconds,
    )


class _Golden:
    def __init__(self, boundaries: dict[int, list[int]], start_indices: dict[int, int], fps: float) -> None:
        self.boundaries = boundaries
        self.start_indices = start_indices
        self.fps = fps


def _load_golden(root: Path, manifest: RunManifest) -> _Golden:
    annotations = _read_object(root / "meta" / "lerobot_annotations.json")
    info = _read_object(root / "meta" / "info.json")
    fps = _positive_finite_number(info.get("fps"), "golden info.fps")
    if not math.isclose(fps, manifest.fps, rel_tol=0, abs_tol=1e-9):
        raise ValueError("golden fps does not match workspace fps")
    if annotations.get("subtask_template") != [item.model_dump(mode="json") for item in manifest.subtasks]:
        raise ValueError("golden subtask template does not match workspace manifest")
    raw_episodes = annotations.get("episodes")
    if not isinstance(raw_episodes, dict):
        raise ValueError("golden annotations.episodes must be an object keyed by episode index")
    lengths: dict[int, int] = {}
    episodes_path = root / "meta" / "episodes.jsonl"
    try:
        lines = episodes_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"unable to read golden episodes metadata: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid golden episodes.jsonl row {line_number}") from exc
        if not isinstance(row, dict) or type(row.get("episode_index")) is not int:
            raise ValueError(f"invalid golden episodes.jsonl row {line_number}")
        length = row.get("length")
        if type(length) is not int or length <= 0 or row["episode_index"] in lengths:
            raise ValueError(f"invalid golden episode length at row {line_number}")
        lengths[row["episode_index"]] = length
    boundaries: dict[int, list[int]] = {}
    starts: dict[int, int] = {}
    for key, raw in raw_episodes.items():
        if not isinstance(key, str) or not key.isascii() or not key.isdecimal() or str(int(key)) != key:
            raise ValueError("golden annotation keys must be canonical decimal episode indices")
        index = int(key)
        if not isinstance(raw, dict) or raw.get("episode_index") != index:
            raise ValueError(f"golden annotation episode {key} has an invalid identity")
        if "start_subtask_index" in raw:
            raise ValueError("complete golden annotations must not carry start_subtask_index")
        if raw.get("high_level_instruction") != manifest.high_level_instruction:
            raise ValueError(f"golden annotation episode {key} has a mismatched high-level instruction")
        current = _boundary_sequence(raw.get("boundaries"), f"golden episode {key} boundaries")
        start = 0
        if index not in lengths:
            raise ValueError(f"golden annotation episode {key} has no episode metadata")
        issues = validate_annotation(
            FinalAnnotation(start_subtask_index=start, boundaries=current),
            "complete", len(manifest.subtasks), lengths[index], 1,
        )
        if issues:
            raise ValueError(f"golden annotation episode {key} is invalid: {issues[0].code}")
        boundaries[index] = current
        starts[index] = start
    if set(boundaries) != set(lengths):
        raise ValueError("golden annotations and episodes metadata are misaligned")
    for index, length in lengths.items():
        if index >= len(manifest.episode_lengths) or manifest.episode_lengths[index] != length:
            raise ValueError("golden episode lengths do not match workspace manifest")
    return _Golden(boundaries, starts, fps)


def _dagger_prediction_violates(
    view: DaggerView,
    prediction: DaggerPrediction,
    min_segment_frames: int,
) -> bool:
    """Independently enforce virtual-view sequence and frame constraints."""
    boundaries = prediction.boundaries
    length = view.end_frame - view.start_frame
    wrong_identity = prediction.start_subtask_index != view.expected_start_subtask_index
    wrong_count = len(boundaries) != len(view.expected_boundaries_relative)
    singleton_transition = view.kind == "singleton" and bool(boundaries)
    wrong_order = any(left >= right for left, right in zip(boundaries, boundaries[1:]))
    wrong_range = any(boundary <= 0 or boundary >= length for boundary in boundaries)
    points = [0, *boundaries, length]
    short_segment = any(
        right - left < min_segment_frames for left, right in zip(points, points[1:])
    )
    return any((
        wrong_identity, wrong_count, singleton_transition, wrong_order, wrong_range, short_segment,
    ))


def _record_has_deterministic_rejection(record: EpisodeRecord) -> bool:
    return bool(
        record.validation_issues
        or DETERMINISTIC_REJECTION_REASONS.intersection(record.review_reasons)
        or record.failure_category == "invalid_model_response"
    )


def _validate_source_provenance(
    work: Path,
    manifest: RunManifest,
    records: Mapping[int, EpisodeRecord],
) -> DatasetIndex:
    """Re-inspect the authoritative source and verify cached episode provenance."""
    raw = json.loads(json.dumps(manifest.effective_config, allow_nan=False))
    if not isinstance(raw, dict):
        raise ValueError("manifest effective_config must be an object")
    model = raw.get("model")
    if not isinstance(model, dict):
        raise ValueError("manifest model provenance is invalid")
    model["api_key"] = "evaluation-local-redacted"
    config = AnnotationConfig.model_validate_json(json.dumps(raw), strict=True)
    if (
        config.source.resolve() != manifest.dataset_root.resolve()
        or config.work_dir.resolve() != work
        or config.mode != manifest.mode
        or config.high_level_instruction != manifest.high_level_instruction
        or config.subtasks != manifest.subtasks
        or config.sampling.min_segment_frames != manifest.min_segment_frames
        or config.model.name != manifest.model_repo
        or compute_run_fingerprint(config, manifest.model_revision) != manifest.run_fingerprint
    ):
        raise ValueError("manifest run/model provenance is invalid")
    dataset = inspect_dataset(config, probe=probe_video)
    if (
        dataset.root.resolve() != manifest.dataset_root.resolve()
        or dataset.version != manifest.dataset_version
        or not math.isclose(dataset.fps, manifest.fps, rel_tol=0, abs_tol=1e-9)
        or list(dataset.camera_keys) != list(manifest.camera_keys)
        or [episode.length for episode in dataset.episodes] != list(manifest.episode_lengths)
        or len(dataset.episodes) != manifest.total_episodes
        or sum(episode.length for episode in dataset.episodes) != manifest.total_frames
    ):
        raise ValueError("current source dataset does not match workspace manifest")
    if any(episode.task != manifest.high_level_instruction for episode in dataset.episodes):
        raise ValueError("current source task instruction does not match workspace manifest")
    for episode in dataset.episodes:
        record = records[episode.episode_index]
        if record.source_fingerprint != compute_source_fingerprint(dataset.root, episode):
            raise ValueError(f"episode {episode.episode_index} source fingerprint is stale")
    return dataset


def launch_gate_report(metrics: EvaluationMetrics) -> dict[str, Any]:
    """Return stable threshold inputs, observed values, and gate decisions."""
    definitions = {
        "median_boundary_error_seconds": (metrics.median_absolute_error_seconds, 0.5, "<="),
        "p90_boundary_error_seconds": (metrics.p90_absolute_error_seconds, 1.0, "<="),
        "accepted_coverage": (metrics.accepted_coverage, 0.85, ">="),
        "constraint_blocking_rate": (metrics.constraint_blocking_rate, 1.0, ">="),
        "false_accept_count": (metrics.false_accept_count, 0, "=="),
    }
    gates: dict[str, Any] = {}
    for name, (observed, threshold, operator) in definitions.items():
        passed = observed is not None and (
            observed <= threshold if operator == "<=" else
            observed >= threshold if operator == ">=" else observed == threshold
        )
        gates[name] = {"observed": observed, "operator": operator, "threshold": threshold, "passed": passed}
    gates["all_passed"] = all(item["passed"] for item in gates.values())
    return gates


def evaluation_report(metrics: EvaluationMetrics) -> dict[str, Any]:
    return {"metrics": metrics.model_dump(mode="json"), "launch_gates": launch_gate_report(metrics)}


def write_evaluation_report(path: Path, metrics: EvaluationMetrics) -> dict[str, Any]:
    """Atomically create a deterministic report without replacing any path."""
    target = Path(path)
    parent = target.parent.resolve()
    if not parent.is_dir():
        raise ValueError("evaluation output parent directory does not exist")
    if target.name in ("", ".", "..") or target.parent.resolve() != parent:
        raise ValueError("invalid evaluation output path")
    report = evaluation_report(metrics)
    payload = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = parent / f".{target.name}.{secrets.token_hex(12)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, parent / target.name, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"evaluation output already exists: {target}") from None
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return report


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read valid JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _positive_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _boundary_sequence(value: object, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{name} must be a sequence of non-negative integer frames")
    return list(value)


def _boundary_map(value: Mapping[int, Sequence[int]], name: str) -> dict[int, list[int]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result: dict[int, list[int]] = {}
    for key, boundaries in value.items():
        if type(key) is not int or key < 0:
            raise ValueError(f"{name} episode indices must be non-negative integers")
        result[key] = _boundary_sequence(boundaries, f"{name}[{key}]")
    return result


def _index_map(value: Mapping[int, int], name: str) -> dict[int, int]:
    if not isinstance(value, Mapping) or any(
        type(key) is not int or key < 0 or type(index) is not int or index < 0
        for key, index in value.items()
    ):
        raise ValueError(f"{name} must map non-negative integer episodes to non-negative indices")
    return dict(value)


def _status_map(value: Mapping[int, str]) -> dict[int, EvaluationStatus]:
    allowed = {"pending", "coarse_done", "refine_done", "accepted", "needs_review", "failed"}
    if not isinstance(value, Mapping):
        raise ValueError("statuses must be a mapping")
    result: dict[int, EvaluationStatus] = {}
    for key, status in value.items():
        if type(key) is not int or key < 0 or status not in allowed:
            raise ValueError("statuses contain an invalid episode index or status")
        result[key] = status  # type: ignore[assignment]
    return result


def _length_map(value: Mapping[int, int]) -> dict[int, int]:
    if not isinstance(value, Mapping) or any(
        type(key) is not int or key < 0 or type(length) is not int or length <= 0
        for key, length in value.items()
    ):
        raise ValueError("episode_lengths must map non-negative episode indices to positive lengths")
    return dict(value)


def _median(values: Sequence[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _nearest_rank(values: Sequence[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[math.ceil(quantile * len(ordered)) - 1])

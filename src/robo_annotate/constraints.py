"""Pure validation rules for coarse and final annotations."""

from typing import Literal, get_args

from .models import FinalAnnotation, IssueCode, ValidationIssue

Mode = Literal["complete", "dagger_patch"]

# Canonical classification shared by orchestration and offline evaluation.
# Keep schema/parse failures here because the pipeline persists them as a
# deterministic rejection before a FinalAnnotation can exist.
ANNOTATION_VALIDATION_ISSUE_CODES = frozenset(get_args(IssueCode))
DETERMINISTIC_REJECTION_REASONS = ANNOTATION_VALIDATION_ISSUE_CODES | frozenset({
    "invalid_model_response",
    "illegal_coarse_sequence",
    "coarse_boundary_count",
    "coarse_boundary_order",
    "refine_transition_mismatch",
})


def _valid_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def coarse_sequence_is_legal(observed: list[int], mode: Mode, subtask_count: int) -> bool:
    """Return whether a coarse model's observed sequence matches the run mode."""
    if mode not in ("complete", "dagger_patch"):
        raise ValueError(f"unsupported mode: {mode!r}")
    if not _valid_positive_int(subtask_count):
        raise ValueError("subtask_count must be >= 1")
    if not isinstance(observed, list) or not all(
        isinstance(index, int) and not isinstance(index, bool) for index in observed
    ):
        return False
    if mode == "complete":
        return observed == list(range(subtask_count))
    if not observed:
        return False
    if len(observed) == 1:
        return 0 <= observed[0] < subtask_count
    start = observed[0]
    return 0 <= start < subtask_count and observed == list(range(start, subtask_count))


def validate_annotation(
    annotation: FinalAnnotation,
    mode: Mode,
    subtask_count: int,
    episode_length: int,
    min_segment_frames: int,
) -> list[ValidationIssue]:
    """Accumulate all deterministic validity issues in a final annotation."""
    if mode not in ("complete", "dagger_patch"):
        raise ValueError(f"unsupported mode: {mode!r}")
    if not _valid_positive_int(subtask_count):
        raise ValueError("subtask_count must be >= 1")
    if not _valid_positive_int(episode_length):
        raise ValueError("episode_length must be >= 1")
    if not _valid_positive_int(min_segment_frames):
        raise ValueError("min_segment_frames must be >= 1")

    issues: list[ValidationIssue] = []
    start = annotation.start_subtask_index
    boundaries = annotation.boundaries

    if start >= subtask_count:
        issues.append(ValidationIssue(code="start_subtask_range", message=(
            f"start_subtask_index {start} must be less than subtask_count {subtask_count}"
        )))
    if mode == "complete":
        if start != 0:
            issues.append(ValidationIssue(code="complete_start_index", message=(
                f"complete mode requires start_subtask_index 0, got {start}"
            )))
        if len(boundaries) != subtask_count - 1:
            issues.append(ValidationIssue(code="complete_boundary_count", message=(
                f"complete mode requires {subtask_count - 1} boundaries, got {len(boundaries)}"
            )))
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        issues.append(ValidationIssue(code="boundary_order", message="boundaries must be strictly increasing"))
    if any(boundary <= 0 or boundary >= episode_length for boundary in boundaries):
        issues.append(ValidationIssue(code="boundary_range", message=(
            f"each boundary must satisfy 0 < boundary < episode_length ({episode_length})"
        )))

    points = [0, *boundaries, episode_length]
    if any(right - left < min_segment_frames for left, right in zip(points, points[1:])):
        issues.append(ValidationIssue(code="segment_too_short", message=(
            f"each consecutive segment must be at least {min_segment_frames} frames"
        )))
    if mode == "dagger_patch" and start < subtask_count and len(boundaries) not in (0, subtask_count - start - 1):
        issues.append(ValidationIssue(code="dagger_suffix_length", message=(
            f"dagger_patch requires 0 or {subtask_count - start - 1} boundaries, got {len(boundaries)}"
        )))
    return issues

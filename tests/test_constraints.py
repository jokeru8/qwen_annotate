import pytest
from pydantic import ValidationError

from qwen_annotate.constraints import coarse_sequence_is_legal, validate_annotation
from qwen_annotate.models import (
    CoarseBoundary,
    CoarseResult,
    FinalAnnotation,
    RefineResult,
)


def test_complete_wrong_boundary_count_is_only_issue() -> None:
    annotation = FinalAnnotation(start_subtask_index=0, boundaries=[100, 220])
    issues = validate_annotation(annotation, "complete", 4, 400, 8)
    assert [issue.code for issue in issues] == ["complete_boundary_count"]


def test_valid_dagger_single_current_subtask() -> None:
    annotation = FinalAnnotation(start_subtask_index=2, boundaries=[])
    assert validate_annotation(annotation, "dagger_patch", 4, 150, 8) == []


def test_valid_dagger_full_suffix_boundary_count() -> None:
    annotation = FinalAnnotation(start_subtask_index=1, boundaries=[40, 90])
    assert validate_annotation(annotation, "dagger_patch", 4, 150, 8) == []


def test_combined_invalid_dagger_annotation_accumulates_unique_codes() -> None:
    annotation = FinalAnnotation(start_subtask_index=1, boundaries=[5, 4, 301])
    issues = validate_annotation(annotation, "dagger_patch", 4, 300, 8)
    codes = [issue.code for issue in issues]
    assert codes == ["boundary_order", "boundary_range", "segment_too_short", "dagger_suffix_length"]
    assert len(codes) == len(set(codes))


@pytest.mark.parametrize(
    ("observed", "mode", "count", "expected"),
    [
        ([0, 1, 2], "complete", 3, True),
        ([0], "complete", 3, False),
        ([1], "dagger_patch", 3, True),
        ([1, 2], "dagger_patch", 3, True),
        ([0, 1, 2], "dagger_patch", 3, True),
        ([0, 2], "dagger_patch", 3, False),
        ([], "dagger_patch", 3, False),
        ([3], "dagger_patch", 3, False),
    ],
)
def test_coarse_sequence_legality(observed, mode, count, expected) -> None:
    assert coarse_sequence_is_legal(observed, mode, count) is expected


def test_complete_requires_start_zero_and_exact_count() -> None:
    annotation = FinalAnnotation(start_subtask_index=1, boundaries=[10, 20])
    assert {i.code for i in validate_annotation(annotation, "complete", 3, 100, 8)} == {
        "complete_start_index"
    }


def test_models_are_frozen_and_forbid_extra_fields() -> None:
    result = CoarseResult(
        start_subtask_index=0,
        observed_subtask_indices=[0],
        coarse_boundaries=[],
        confidence=0.5,
    )
    with pytest.raises(ValidationError):
        CoarseBoundary(from_subtask_index=0, to_subtask_index=1, estimated_frame=2, evidence="x", extra=1)
    with pytest.raises(ValidationError):
        RefineResult(
            from_subtask_index=0,
            to_subtask_index=1,
            last_frame_before=1,
            first_frame_after=2,
            boundary_frame=1,
            confidence=0.5,
            visible_cues=[],
            extra=1,
        )
    with pytest.raises(ValidationError):
        result.start_subtask_index = 1


@pytest.mark.parametrize("kwargs", [{"subtask_count": 0}, {"episode_length": 0}, {"min_segment_frames": 0}])
def test_invalid_caller_arguments_raise_value_error(kwargs) -> None:
    annotation = FinalAnnotation(start_subtask_index=0, boundaries=[])
    args = {"mode": "complete", "subtask_count": 1, "episode_length": 10, "min_segment_frames": 1}
    args.update(kwargs)
    with pytest.raises(ValueError):
        validate_annotation(annotation, **args)


@pytest.mark.parametrize("name", ["subtask_count", "episode_length", "min_segment_frames"])
@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_invalid_caller_argument_types_raise_value_error(name, value) -> None:
    annotation = FinalAnnotation(start_subtask_index=0, boundaries=[])
    args = {"mode": "complete", "subtask_count": 1, "episode_length": 10, "min_segment_frames": 1}
    args[name] = value
    with pytest.raises(ValueError):
        validate_annotation(annotation, **args)


@pytest.mark.parametrize("observed", [[0, True], [0, 1.5], [0, "1"]])
def test_coarse_sequence_rejects_non_integer_observed_indices(observed) -> None:
    assert coarse_sequence_is_legal(observed, "complete", 2) is False


def test_dagger_out_of_range_start_has_no_negative_suffix_issue() -> None:
    annotation = FinalAnnotation(start_subtask_index=4, boundaries=[])
    issues = validate_annotation(annotation, "dagger_patch", 4, 100, 8)
    assert [issue.code for issue in issues] == ["start_subtask_range"]
    assert "-1" not in issues[0].message

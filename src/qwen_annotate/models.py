"""Immutable, strict data models exchanged by annotation stages."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class CoarseBoundary(FrozenModel):
    from_subtask_index: int = Field(ge=0)
    to_subtask_index: int = Field(ge=0)
    estimated_frame: int = Field(ge=0)
    evidence: str = Field(min_length=1)

    @model_validator(mode="after")
    def adjacent_subtasks(self) -> "CoarseBoundary":
        if self.to_subtask_index != self.from_subtask_index + 1:
            raise ValueError("to_subtask_index must equal from_subtask_index + 1")
        return self


SemanticUncertaintyCode = Literal[
    "subtask_order_unclear",
    "start_subtask_unclear",
    "transition_neighborhood_unclear",
]


class CoarseResult(FrozenModel):
    start_subtask_index: int = Field(ge=0)
    observed_subtask_indices: list[int] = Field(min_length=1)
    coarse_boundaries: list[CoarseBoundary]
    confidence: float = Field(ge=0, le=1)
    semantic_uncertainty_codes: list[SemanticUncertaintyCode] = Field(default_factory=list)
    boundary_precision_notes: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_sequence(self) -> "CoarseResult":
        if self.start_subtask_index != self.observed_subtask_indices[0]:
            raise ValueError("start_subtask_index must equal observed_subtask_indices[0]")
        expected_count = len(self.observed_subtask_indices) - 1
        if len(self.coarse_boundaries) != expected_count:
            raise ValueError("coarse_boundaries count must equal observed sequence length minus one")
        for index, boundary in enumerate(self.coarse_boundaries):
            if boundary.from_subtask_index != self.observed_subtask_indices[index] or boundary.to_subtask_index != self.observed_subtask_indices[index + 1]:
                raise ValueError("coarse boundary subtask indices must match adjacent observed indices")
        if any(left.estimated_frame >= right.estimated_frame for left, right in zip(self.coarse_boundaries, self.coarse_boundaries[1:])):
            raise ValueError("estimated_frame values must be strictly increasing")
        return self


class RefineResult(FrozenModel):
    from_subtask_index: int = Field(ge=0)
    to_subtask_index: int = Field(ge=0)
    last_frame_before: int = Field(ge=0)
    first_frame_after: int = Field(ge=0)
    boundary_frame: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    visible_cues: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_transition(self) -> "RefineResult":
        if self.to_subtask_index != self.from_subtask_index + 1:
            raise ValueError("to_subtask_index must equal from_subtask_index + 1")
        if not (self.last_frame_before + 1 == self.first_frame_after == self.boundary_frame):
            raise ValueError("last_frame_before + 1 must equal first_frame_after and boundary_frame")
        return self


class FinalAnnotation(FrozenModel):
    start_subtask_index: int = Field(ge=0)
    boundaries: list[int]


IssueCode = Literal[
    "start_subtask_range",
    "complete_start_index",
    "complete_boundary_count",
    "dagger_suffix_length",
    "boundary_order",
    "boundary_range",
    "segment_too_short",
]


class ValidationIssue(FrozenModel):
    code: IssueCode
    message: str

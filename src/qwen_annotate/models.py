"""Immutable, strict data models exchanged by annotation stages."""

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CoarseBoundary(FrozenModel):
    from_subtask_index: int = Field(ge=0)
    to_subtask_index: int = Field(ge=0)
    estimated_frame: int = Field(ge=0)
    evidence: str


class CoarseResult(FrozenModel):
    start_subtask_index: int = Field(ge=0)
    observed_subtask_indices: list[int] = Field(min_length=1)
    coarse_boundaries: list[CoarseBoundary]
    confidence: float = Field(ge=0, le=1)
    uncertainties: list[str] = Field(default_factory=list)


class RefineResult(FrozenModel):
    from_subtask_index: int = Field(ge=0)
    to_subtask_index: int = Field(ge=0)
    last_frame_before: int = Field(ge=0)
    first_frame_after: int = Field(ge=0)
    boundary_frame: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    visible_cues: list[str]


class FinalAnnotation(FrozenModel):
    start_subtask_index: int = Field(ge=0)
    boundaries: list[int]


class ValidationIssue(FrozenModel):
    code: str
    message: str

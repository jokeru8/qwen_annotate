"""Strict, serializable configuration for annotation runs."""

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Subtask(StrictModel):
    skill: str = Field(min_length=1)
    text: str = Field(min_length=1)


class ModelConfig(StrictModel):
    name: str = "Qwen/Qwen3.8-27B"
    local_path: Path = Path("/mnt/data/user/zhoukr/models/Qwen3.8-27B")
    endpoint: HttpUrl = HttpUrl("http://127.0.0.1:8000/v1")
    api_key: str = "local"
    revision: str | None = None


class SamplingConfig(StrictModel):
    coarse_fps: float = Field(default=1.0, gt=0)
    coarse_max_frames: int = Field(default=64, ge=8)
    refine_window_seconds: float = Field(default=2.5, gt=0)
    refine_fps: float = Field(default=8.0, gt=0)
    dense_radius_seconds: float = Field(default=0.5, gt=0)
    agreement_tolerance_frames: int = Field(default=12, ge=0)
    min_segment_frames: int = Field(default=8, ge=1)


class AnnotationConfig(StrictModel):
    source: Path
    work_dir: Path
    mode: Literal["complete", "dagger_patch"]
    high_level_instruction: str = Field(min_length=1)
    primary_camera: str = Field(min_length=1)
    refine_cameras: list[str] = Field(min_length=1)
    subtasks: list[Subtask] = Field(min_length=1)
    model: ModelConfig = Field(default_factory=ModelConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)

    def stable_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"model": {"api_key"}})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(path: Path) -> AnnotationConfig:
    """Load and validate a YAML annotation configuration."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AnnotationConfig.model_validate(raw)

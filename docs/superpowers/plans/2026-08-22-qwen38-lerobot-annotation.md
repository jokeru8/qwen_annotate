# Qwen3.8 LeRobot Annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, auditable Qwen3.8-27B pipeline that annotates complete and DAgger-patch LeRobot v2.1 datasets with ordered subtask boundaries, quarantines uncertain episodes, and converts approved results into ModelScope-compatible datasets.

**Architecture:** A typed Python CLI reads LeRobot datasets without mutation, samples frame-indexed multi-camera evidence, calls a vLLM OpenAI-compatible endpoint in coarse and refine stages, validates results with deterministic sequence constraints, and atomically persists per-episode state. Review, conversion, validation, and golden-set evaluation are separate consumers of the workspace so inference never writes into source data.

**Tech Stack:** Python 3.12, Pydantic 2, Typer, PyYAML, PyArrow, NumPy, PyAV, Pillow, OpenAI Python client, HTTPX, Jinja2, Hugging Face CLI, vLLM 0.27.1, pytest, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-08-22-qwen38-lerobot-annotation-design.md`

## Global Constraints

- Support LeRobot codebase version exactly `v2.1` in the first release.
- Use only user-supplied subtask template entries; model-created labels are invalid.
- `complete` episodes contain every template item exactly once and in order.
- `dagger_patch` episodes contain either one item `[k]` or the suffix `[k, k+1, ..., N-1]`.
- Boundaries are integer frame indices with left-closed/right-open semantics and identify the first frame of the next subtask.
- Never mutate the source dataset during inspect, annotate, review, evaluation, or conversion.
- Only `accepted` or human-confirmed episodes may enter release conversion.
- Download official `Qwen/Qwen3.8-27B` to `/mnt/data/user/zhoukr/models/Qwen3.8-27B` at an immutable commit revision and verify it after download.
- Use `/mnt/data/user/zhoukr/datasets/jokeru/arrange_orange_juice_and_green_tea_2_annotated` as the golden complete-dataset reference.
- Keep model responses and confidence in the workspace; do not write them into the published dataset.
- Require Python `>=3.12,<3.13`; pin dependency ranges in `pyproject.toml` and commit `uv.lock`.

---

## File Map

```text
pyproject.toml                         package metadata, dependencies, CLI entry point
uv.lock                                reproducible dependency lock
README.md                              install and operator workflow
examples/complete.yaml                 complete-mode example
examples/dagger_patch.yaml             patch-mode example
src/qwen_annotate/config.py            YAML schema and stable configuration hash
src/qwen_annotate/models.py            domain and workspace Pydantic models
src/qwen_annotate/constraints.py       deterministic annotation validation
src/qwen_annotate/lerobot.py           v2.1 metadata indexing and dataset validation
src/qwen_annotate/video.py             exact frame extraction and visual labeling
src/qwen_annotate/prompts.py           versioned coarse/refine prompts and schemas
src/qwen_annotate/qwen_client.py       async structured-output model client
src/qwen_annotate/model_manager.py     immutable model download and verification
src/qwen_annotate/workspace.py         fingerprints, atomic state, resume logic
src/qwen_annotate/coarse.py            whole-episode inference and agreement
src/qwen_annotate/refine.py            adaptive boundary refinement and agreement
src/qwen_annotate/pipeline.py          per-episode and batch orchestration
src/qwen_annotate/review.py            static review page and human decisions
src/qwen_annotate/converter.py         full and accepted-only dataset conversion
src/qwen_annotate/release_validator.py independent release consistency checks
src/qwen_annotate/evaluation.py        golden and synthetic-DAgger metrics
src/qwen_annotate/cli.py               Typer commands and exit behavior
tests/                                 unit and integration tests mirroring modules
```

### Task 1: Package Skeleton and Typed Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/qwen_annotate/__init__.py`
- Create: `src/qwen_annotate/config.py`
- Create: `examples/complete.yaml`
- Create: `examples/dagger_patch.yaml`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: no project interfaces.
- Produces: `Subtask`, `ModelConfig`, `SamplingConfig`, `AnnotationConfig`, `load_config(path: Path) -> AnnotationConfig`, and `AnnotationConfig.stable_hash() -> str`.

- [ ] **Step 1: Add the packaging test and configuration tests**

```python
# tests/test_config.py
from pathlib import Path
import pytest
from pydantic import ValidationError
from qwen_annotate.config import AnnotationConfig, load_config

def test_complete_config_loads_and_hash_is_stable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    work = tmp_path / "work"
    source.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"source: {source}\n"
        f"work_dir: {work}\n"
        "mode: complete\n"
        "high_level_instruction: arrange objects\n"
        "primary_camera: observation.images.right_eye\n"
        "refine_cameras: [observation.images.right_eye]\n"
        "subtasks:\n  - {skill: pick, text: Pick the object}\n"
        "model:\n  endpoint: http://127.0.0.1:8000/v1\n",
        encoding="utf-8",
    )
    first = load_config(config_path)
    second = load_config(config_path)
    assert first.mode == "complete"
    assert first.stable_hash() == second.stable_hash()

def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        AnnotationConfig.model_validate({
            "source": str(tmp_path), "work_dir": str(tmp_path / "work"),
            "mode": "complete", "high_level_instruction": "x",
            "primary_camera": "cam", "refine_cameras": ["cam"],
            "subtasks": [{"skill": "pick", "text": "pick"}],
            "model": {}, "unexpected": True,
        })
```

- [ ] **Step 2: Run the tests and confirm the package is absent**

Run: `uv run pytest tests/test_config.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'qwen_annotate'`.

- [ ] **Step 3: Create `pyproject.toml` with the application and test dependencies**

```toml
[project]
name = "qwen-lerobot-annotate"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "av>=15,<17", "httpx>=0.28,<1", "jinja2>=3.1,<4",
  "numpy>=2.2,<3", "openai>=2,<3", "pillow>=11,<13", "pyarrow>=22,<23",
  "pydantic>=2.11,<3", "pyyaml>=6,<7", "typer>=0.16,<1",
]

[project.optional-dependencies]
dev = ["pytest>=8.4,<9", "pytest-asyncio>=1,<2"]

[project.scripts]
qwen-annotate = "qwen_annotate.cli:app"

[build-system]
requires = ["hatchling>=1.27,<2"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/qwen_annotate"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 4: Implement the strict configuration models and canonical hash**

```python
# src/qwen_annotate/config.py
import hashlib, json
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
    model: ModelConfig = ModelConfig()
    sampling: SamplingConfig = SamplingConfig()

    def stable_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"model": {"api_key"}})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def load_config(path: Path) -> AnnotationConfig:
    return AnnotationConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
```

- [ ] **Step 5: Add the two runnable YAML examples and concise README setup instructions**

Copy the four exact reference subtasks into `examples/complete.yaml`; copy the same file to `examples/dagger_patch.yaml` and change only `mode` and `work_dir`. Document `uv sync --extra dev`, the source-read-only guarantee, and the inspect → annotate → review → convert → validate workflow.

- [ ] **Step 6: Lock dependencies and run the tests**

Run: `uv lock && uv sync --extra dev && uv run pytest tests/test_config.py -v`

Expected: both tests pass and `uv.lock` is created.

- [ ] **Step 7: Commit the package foundation**

```bash
git add pyproject.toml uv.lock README.md examples src/qwen_annotate/__init__.py src/qwen_annotate/config.py tests/test_config.py
git commit -m "feat: add typed annotation configuration"
```

### Task 2: Annotation Domain Models and Hard Constraints

**Files:**
- Create: `src/qwen_annotate/models.py`
- Create: `src/qwen_annotate/constraints.py`
- Create: `tests/test_constraints.py`

**Interfaces:**
- Consumes: `Subtask` from `qwen_annotate.config`.
- Produces: `CoarseBoundary`, `CoarseResult`, `RefineResult`, `FinalAnnotation`, `ValidationIssue`, `validate_annotation(...) -> list[ValidationIssue]`, and `coarse_sequence_is_legal(...) -> bool`.

- [ ] **Step 1: Write failing complete and DAgger constraint tests**

```python
# tests/test_constraints.py
from qwen_annotate.constraints import validate_annotation
from qwen_annotate.models import FinalAnnotation

def test_complete_requires_all_segments() -> None:
    ann = FinalAnnotation(start_subtask_index=0, boundaries=[100, 220])
    issues = validate_annotation(ann, "complete", 4, 400, 8)
    assert {x.code for x in issues} == {"complete_boundary_count"}

def test_dagger_single_subtask_is_valid() -> None:
    ann = FinalAnnotation(start_subtask_index=2, boundaries=[])
    assert validate_annotation(ann, "dagger_patch", 4, 150, 8) == []

def test_boundary_range_order_and_minimum_segment_are_checked() -> None:
    ann = FinalAnnotation(start_subtask_index=1, boundaries=[5, 4, 301])
    codes = {x.code for x in validate_annotation(ann, "dagger_patch", 4, 300, 8)}
    assert codes == {"boundary_order", "boundary_range", "segment_too_short", "dagger_suffix_length"}
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `uv run pytest tests/test_constraints.py -v`

Expected: import fails because `qwen_annotate.constraints` does not exist.

- [ ] **Step 3: Implement immutable typed inference results**

```python
# src/qwen_annotate/models.py
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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
```

- [ ] **Step 4: Implement deterministic sequence and boundary validation**

`coarse_sequence_is_legal` must require `list(range(subtask_count))` for complete mode and either `[k]` or `list(range(k, subtask_count))` for DAgger. `validate_annotation` must report all applicable issues without stopping at the first one, including template range, expected boundary count, strictly increasing values, `0 < boundary < episode_length`, and minimum segment length over `[0, *boundaries, episode_length]`.

- [ ] **Step 5: Run the domain tests**

Run: `uv run pytest tests/test_constraints.py -v`

Expected: all three tests pass.

- [ ] **Step 6: Commit the domain layer**

```bash
git add src/qwen_annotate/models.py src/qwen_annotate/constraints.py tests/test_constraints.py
git commit -m "feat: validate ordered subtask annotations"
```

### Task 3: Read-Only LeRobot v2.1 Inspection

**Files:**
- Create: `src/qwen_annotate/lerobot.py`
- Create: `tests/__init__.py`
- Create: `tests/fixtures.py`
- Create: `tests/test_lerobot.py`

**Interfaces:**
- Consumes: `AnnotationConfig`.
- Produces: `EpisodeInfo`, `DatasetIndex`, `VideoProbe`, `probe_video(path: Path) -> VideoProbe`, and `inspect_dataset(config: AnnotationConfig, probe=probe_video) -> DatasetIndex`.

- [ ] **Step 1: Build a minimal v2.1 fixture and write inspection tests**

```python
# tests/test_lerobot.py
from qwen_annotate.lerobot import VideoProbe, inspect_dataset
from tests.fixtures import make_lerobot_fixture, make_config

def test_inspect_builds_episode_video_index(tmp_path) -> None:
    root = make_lerobot_fixture(tmp_path, lengths=[12, 12], fps=5, cameras=["cam.eye"])
    cfg = make_config(root, tmp_path / "work", primary="cam.eye")
    index = inspect_dataset(cfg, probe=lambda _: VideoProbe(frames=12, fps=5, width=16, height=16))
    assert index.version == "v2.1"
    assert index.episodes[0].length == 12
    assert index.episodes[0].videos["cam.eye"].name == "episode_000000.mp4"

def test_inspect_rejects_missing_refine_camera(tmp_path) -> None:
    root = make_lerobot_fixture(tmp_path, lengths=[12], fps=5, cameras=["cam.eye"])
    cfg = make_config(root, tmp_path / "work", primary="cam.eye", refine=["cam.wrist"])
    try:
        inspect_dataset(cfg, probe=lambda _: VideoProbe(frames=12, fps=5, width=16, height=16))
    except ValueError as exc:
        assert "cam.wrist" in str(exc)
    else:
        raise AssertionError("missing camera was accepted")
```

- [ ] **Step 2: Run the tests and verify the reader is missing**

Run: `uv run pytest tests/test_lerobot.py -v`

Expected: import fails because `qwen_annotate.lerobot` does not exist.

- [ ] **Step 3: Implement typed metadata indexing and validation**

```python
# src/qwen_annotate/lerobot.py
class VideoProbe(BaseModel):
    frames: int
    fps: float
    width: int
    height: int

class EpisodeInfo(BaseModel):
    episode_index: int
    length: int
    task: str
    parquet: Path
    videos: dict[str, Path]

class DatasetIndex(BaseModel):
    root: Path
    version: str
    fps: float
    camera_keys: list[str]
    episodes: list[EpisodeInfo]
```

Read `meta/info.json`, `meta/episodes.jsonl`, and `meta/tasks.jsonl`; resolve paths through the `data_path` and `video_path` templates rather than hard-coded `chunk-000`. Verify version, contiguous episode indices, parquet row counts, configured cameras, video existence, video frame count, and FPS tolerance `abs(video_fps - info_fps) <= 0.01`. Return all discovered facts without writing under `source`.

- [ ] **Step 4: Add PyAV video probing and parquet row-count checks**

Use `av.open(path)` and the first video stream. Prefer `stream.frames`; if zero, decode and count. Use `pyarrow.parquet.ParquetFile(path).metadata.num_rows` and require it to equal `episodes.jsonl.length`.

- [ ] **Step 5: Run reader tests and inspect the real golden source**

Run: `uv run pytest tests/test_lerobot.py -v`

Run: `uv run python -c "from pathlib import Path; from qwen_annotate.config import load_config; from qwen_annotate.lerobot import inspect_dataset; print(len(inspect_dataset(load_config(Path('examples/complete.yaml'))).episodes))"`

Expected: unit tests pass; `examples/complete.yaml` uses the confirmed local reference source and the command prints `47`.

- [ ] **Step 6: Commit the dataset reader**

```bash
git add src/qwen_annotate/lerobot.py tests/__init__.py tests/fixtures.py tests/test_lerobot.py
git commit -m "feat: inspect LeRobot v2.1 datasets"
```

### Task 4: Exact Frame Sampling and Camera-Labeled Evidence

**Files:**
- Create: `src/qwen_annotate/video.py`
- Create: `tests/test_video.py`

**Interfaces:**
- Consumes: `EpisodeInfo` and `SamplingConfig`.
- Produces: `FrameSample`, `uniform_indices(...)`, `window_indices(...)`, `extract_frames(...)`, and `as_data_url(sample: FrameSample) -> str`.

- [ ] **Step 1: Write index-selection and exact-frame tests**

```python
# tests/test_video.py
from qwen_annotate.video import uniform_indices, window_indices

def test_uniform_indices_cover_full_episode_and_respect_cap() -> None:
    values = uniform_indices(frame_count=1000, source_fps=25, target_fps=1, max_frames=10)
    assert values[0] == 0
    assert values[-1] == 999
    assert len(values) == 10
    assert values == sorted(set(values))

def test_window_indices_are_clipped_and_include_center() -> None:
    values = window_indices(center=3, radius_frames=10, stride=2, frame_count=20)
    assert min(values) == 0
    assert max(values) <= 13
    assert 3 in values
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `uv run pytest tests/test_video.py -v`

Expected: import fails because `qwen_annotate.video` does not exist.

- [ ] **Step 3: Implement deterministic sample-index functions**

Use rounded `numpy.linspace` semantics without adding NumPy as a direct dependency: calculate evenly spaced indices with integer arithmetic, always include 0 and `frame_count - 1`, remove duplicates, and enforce `max_frames`. Window indices must include the center even when it is off the stride grid.

- [ ] **Step 4: Implement exact PyAV decoding and overlays**

```python
class FrameSample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    camera_key: str
    frame_index: int
    timestamp_seconds: float
    jpeg: bytes
```

Decode sequentially once per requested video, retain only requested indices, convert frames to Pillow images, draw `camera_key`, `frame_index`, and timestamp on an opaque label, and encode JPEG at quality 90. Raise an error listing every missing requested frame. `as_data_url` returns `data:image/jpeg;base64,<payload>`.

- [ ] **Step 5: Add a generated-video integration test**

Create a 12-frame 16×16 H.264 or MPEG-4 fixture with PyAV where frame `i` has red channel `i * 10`. Extract `[0, 6, 11]`; assert returned `frame_index` values are exact and JPEG payloads are non-empty. Do not depend on system `ffmpeg` in the unit test.

- [ ] **Step 6: Run sampler tests and commit**

Run: `uv run pytest tests/test_video.py -v`

Expected: all sampler tests pass.

```bash
git add src/qwen_annotate/video.py tests/test_video.py
git commit -m "feat: sample frame-indexed video evidence"
```

### Task 5: Versioned Prompts and Strict Response Schemas

**Files:**
- Create: `src/qwen_annotate/prompts.py`
- Create: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `AnnotationConfig`, `CoarseResult`, `FrameSample` metadata.
- Produces: `PROMPT_VERSION`, `build_coarse_prompt(...) -> str`, `build_refine_prompt(...) -> str`, `coarse_json_schema() -> dict`, and `refine_json_schema() -> dict`.

- [ ] **Step 1: Write prompt-invariant tests**

```python
# tests/test_prompts.py
from qwen_annotate.prompts import build_coarse_prompt
from tests.fixtures import make_config

def test_dagger_prompt_contains_template_and_legal_sequences(tmp_path) -> None:
    cfg = make_config(tmp_path, tmp_path / "work", mode="dagger_patch")
    text = build_coarse_prompt(cfg, episode_index=7, frame_count=500, pass_id=1)
    assert "Do not invent or rewrite labels" in text
    assert "[k] or [k, k+1, ..., N-1]" in text
    assert '0: {"skill": "pick", "text": "pick"}' in text
    assert "first frame of the next subtask" in text
```

- [ ] **Step 2: Run the prompt test and verify failure**

Run: `uv run pytest tests/test_prompts.py -v`

Expected: import fails because `qwen_annotate.prompts` does not exist.

- [ ] **Step 3: Implement prompts as pure functions**

Set `PROMPT_VERSION = "coarse-v1/refine-v1"`. Include high-level instruction, ordered JSON-rendered template, mode rules, episode/frame metadata, boundary definition, uncertainty behavior, and pass identifier. The refine prompt must name exactly one `from_subtask_index`/`to_subtask_index` pair and require visible cues rather than hidden reasoning.

- [ ] **Step 4: Export Pydantic-derived JSON schemas with no free-form labels**

Generate schemas from `CoarseResult.model_json_schema()` and `RefineResult.model_json_schema()`. Prompt output fields reference only integer template indices. Add a test that neither schema contains a field named `label` or `subtask_text`.

- [ ] **Step 5: Run tests and commit**

Run: `uv run pytest tests/test_prompts.py -v`

Expected: prompt and schema tests pass.

```bash
git add src/qwen_annotate/prompts.py tests/test_prompts.py
git commit -m "feat: constrain Qwen annotation prompts"
```

### Task 6: Async Qwen Client with Structured Output and Bounded Retries

**Files:**
- Create: `src/qwen_annotate/qwen_client.py`
- Create: `tests/test_qwen_client.py`

**Interfaces:**
- Consumes: endpoint settings, prompt string, `FrameSample` list, response Pydantic type.
- Produces: `QwenClient.complete(prompt, frames, response_type) -> T`, `ModelCallError`, `ModelOutOfMemory`, and `InvalidModelResponse`.

- [ ] **Step 1: Write async retry and parse tests using an injected transport**

```python
# tests/test_qwen_client.py
import pytest
from qwen_annotate.models import FinalAnnotation
from qwen_annotate.qwen_client import QwenClient

@pytest.mark.asyncio
async def test_client_retries_transient_error_then_parses() -> None:
    calls = 0
    async def send(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("slow")
        return '{"start_subtask_index": 1, "boundaries": [20]}'
    client = QwenClient(send=send, max_attempts=2, retry_seconds=0)
    result = await client.complete("prompt", [], FinalAnnotation)
    assert result.boundaries == [20]
    assert calls == 2
```

- [ ] **Step 2: Run the client test and verify failure**

Run: `uv run pytest tests/test_qwen_client.py -v`

Expected: import fails because `qwen_annotate.qwen_client` does not exist.

- [ ] **Step 3: Implement the OpenAI-compatible request adapter**

Build message content as one text item followed by `image_url` items from `as_data_url`. Use `AsyncOpenAI(base_url=..., api_key=..., timeout=...)`. Send a JSON-schema response format and also pass vLLM guided JSON in `extra_body` when enabled. Extract only `choices[0].message.content`, parse JSON, and validate it with `response_type.model_validate`.

- [ ] **Step 4: Implement finite retry classification**

Retry `APITimeoutError`, `APIConnectionError`, HTTP 429, and HTTP 5xx with delays `[1, 2, 4]` capped by `max_attempts`. Detect CUDA/worker OOM responses and raise `ModelOutOfMemory` so the refine stage can degrade visual input once. Retry invalid JSON once with a text-only format-repair request containing the invalid response and schema; never treat format repair as semantic acceptance. Raise typed errors with attempt count and safe response excerpts.

- [ ] **Step 5: Run tests and commit**

Run: `uv run pytest tests/test_qwen_client.py -v`

Expected: transient retry, invalid JSON, and exhausted retry tests pass.

```bash
git add src/qwen_annotate/qwen_client.py tests/test_qwen_client.py
git commit -m "feat: call Qwen with structured multimodal output"
```

### Task 7: Immutable Qwen3.8 Model Download and Verification

**Files:**
- Create: `src/qwen_annotate/model_manager.py`
- Create: `tests/test_model_manager.py`

**Interfaces:**
- Consumes: model repo, optional revision, local path, an injectable command runner and HTTP client.
- Produces: `ModelInstall`, `resolve_revision(...) -> str`, `download_model(...) -> ModelInstall`, and `verify_model(...) -> None`.

- [ ] **Step 1: Write tests for SHA resolution, proxy isolation, and command construction**

```python
# tests/test_model_manager.py
from pathlib import Path
from qwen_annotate.model_manager import download_model

def test_download_pins_sha_and_clears_proxy_only_for_child(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks5://proxy:1080")
    seen = []
    def runner(args, env):
        seen.append((args, env))
    result = download_model(
        "Qwen/Qwen3.8-27B", tmp_path / "model",
        revision="a" * 40, runner=runner,
    )
    assert result.revision == "a" * 40
    assert "--revision" in seen[0][0]
    assert "ALL_PROXY" not in seen[0][1]
    assert seen[1][0][:3] == ["hf", "cache", "verify"]
```

- [ ] **Step 2: Run the model-manager test and verify failure**

Run: `uv run pytest tests/test_model_manager.py -v`

Expected: import fails because `qwen_annotate.model_manager` does not exist.

- [ ] **Step 3: Implement immutable revision resolution**

Use `httpx.Client(trust_env=False)` against `https://huggingface.co/api/models/{repo}/revision/{revision}` and require the returned `sha` to match `[0-9a-f]{40}`. Default requested revision is `main`, but always pass the resolved SHA to download and record it.

- [ ] **Step 4: Implement child-process download and verification**

Invoke:

```bash
hf download Qwen/Qwen3.8-27B --revision <40-char-sha> --local-dir /mnt/data/user/zhoukr/models/Qwen3.8-27B --max-workers 8
hf cache verify Qwen/Qwen3.8-27B --revision <40-char-sha> --local-dir /mnt/data/user/zhoukr/models/Qwen3.8-27B --fail-on-missing-files
```

Build the subprocess environment from `os.environ.copy()` and remove uppercase and lowercase HTTP, HTTPS, and ALL proxy keys only from that copy. Write `model-install.json` containing repo, SHA, verified timestamp, and local path after verification succeeds.

- [ ] **Step 5: Run tests and commit without downloading weights**

Run: `uv run pytest tests/test_model_manager.py -v`

Expected: all tests pass and no network request occurs in tests.

```bash
git add src/qwen_annotate/model_manager.py tests/test_model_manager.py
git commit -m "feat: download immutable Qwen3.8 weights"
```

### Task 8: Atomic Workspace, Fingerprints, and Resume State

**Files:**
- Create: `src/qwen_annotate/workspace.py`
- Create: `tests/test_workspace.py`

**Interfaces:**
- Consumes: `AnnotationConfig`, `DatasetIndex`, `CoarseResult`, `RefineResult`, `FinalAnnotation`.
- Produces: `EpisodeRecord`, `RunManifest`, `WorkspaceStore.initialize(...)`, `.load_episode(index)`, `.save_episode(record)`, `.cache_is_valid(...)`, `.summary()`, and `.write_summary()`.

- [ ] **Step 1: Write atomic persistence and cache invalidation tests**

```python
# tests/test_workspace.py
from qwen_annotate.workspace import EpisodeRecord, WorkspaceStore

def test_save_is_atomic_and_round_trips(tmp_path) -> None:
    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    record = EpisodeRecord(episode_index=3, status="pending", source_fingerprint="abc")
    store.save_episode(record)
    assert store.load_episode(3) == record
    assert list((tmp_path / "work" / "episodes").glob("*.tmp")) == []

def test_cache_invalidates_on_config_or_source_change(tmp_path) -> None:
    store = WorkspaceStore(tmp_path / "work")
    record = EpisodeRecord(episode_index=0, status="accepted", source_fingerprint="old", run_fingerprint="cfg1")
    assert not store.cache_is_valid(record, source_fingerprint="new", run_fingerprint="cfg1")
    assert not store.cache_is_valid(record, source_fingerprint="old", run_fingerprint="cfg2")
```

- [ ] **Step 2: Run the workspace tests and verify failure**

Run: `uv run pytest tests/test_workspace.py -v`

Expected: import fails because `qwen_annotate.workspace` does not exist.

- [ ] **Step 3: Implement state models and legal transitions**

Statuses are `pending`, `coarse_done`, `refine_done`, `accepted`, `needs_review`, and `failed`. Store structured coarse/refine attempts, final annotation, validation issues, failure category, decision source, timestamps, source fingerprint, and run fingerprint. Reject backward transitions except an explicit cache invalidation back to `pending`.

- [ ] **Step 4: Implement fingerprints and atomic JSON writes**

Source fingerprint is SHA-256 over relative video path, video size, video mtime-ns, parquet path, parquet size, and episode length. Run fingerprint is SHA-256 over config stable hash, `PROMPT_VERSION`, model repo, and immutable model revision. Write with `NamedTemporaryFile(dir=target.parent, delete=False)`, flush, `os.fsync`, then `os.replace`.

- [ ] **Step 5: Implement manifest and summary**

Manifest records dataset root, dataset facts, template, mode, code version, prompt version, model repo/revision, full effective config, and creation time. Summary returns exact counts per status and lists episode indices for non-accepted states; `write_summary()` atomically updates `summary.json` after every episode transition.

- [ ] **Step 6: Run tests and commit**

Run: `uv run pytest tests/test_workspace.py -v`

Expected: transition, atomicity, summary, and invalidation tests pass.

```bash
git add src/qwen_annotate/workspace.py tests/test_workspace.py
git commit -m "feat: persist resumable annotation state"
```

### Task 9: Coarse Whole-Episode Annotation

**Files:**
- Create: `src/qwen_annotate/coarse.py`
- Create: `tests/test_coarse.py`

**Interfaces:**
- Consumes: `AnnotationConfig`, `EpisodeInfo`, frame sampler, `QwenClient`, coarse prompt/schema, sequence constraints.
- Produces: `CoarseDecision` and `run_coarse(config, episode, sampler, client) -> CoarseDecision`.

- [ ] **Step 1: Write agreement and quarantine tests with fake clients**

```python
# tests/test_coarse.py
import pytest
from qwen_annotate.coarse import run_coarse
from qwen_annotate.models import CoarseBoundary, CoarseResult

def result(start, observed, boundary):
    return CoarseResult(
        start_subtask_index=start, observed_subtask_indices=observed,
        coarse_boundaries=[CoarseBoundary(from_subtask_index=0, to_subtask_index=1,
            estimated_frame=boundary, evidence="handoff")],
        confidence=0.9, uncertainties=[],
    )

@pytest.mark.asyncio
async def test_disagreeing_sequences_require_review(config, episode, sampler) -> None:
    client = FakeClient([result(0, [0, 1], 50), result(1, [1], 0)])
    decision = await run_coarse(config, episode, sampler, client)
    assert decision.status == "needs_review"
    assert "coarse_sequence_disagreement" in decision.reasons
```

- [ ] **Step 2: Run the coarse tests and verify failure**

Run: `uv run pytest tests/test_coarse.py -v`

Expected: import fails because `qwen_annotate.coarse` does not exist.

- [ ] **Step 3: Implement two independent coarse passes**

Both passes use full-span uniform sampling but different deterministic interior offsets while preserving first/last frames. Complete mode still asks the model for boundary evidence, while its observed sequence is validated against all template indices. DAgger requires both passes to agree on start and observed sequence. Reject any boundary whose from/to indices are not consecutive.

- [ ] **Step 4: Implement coarse decision reasons**

Return `coarse_sequence_disagreement`, `illegal_coarse_sequence`, `coarse_boundary_count`, `coarse_boundary_order`, or `coarse_uncertain` as stable machine-readable reason codes. Preserve both raw typed attempts for review. A successful decision exposes averaged coarse boundary centers rounded to integers.

- [ ] **Step 5: Run tests and commit**

Run: `uv run pytest tests/test_coarse.py -v`

Expected: success, complete violation, DAgger violation, and disagreement tests pass.

```bash
git add src/qwen_annotate/coarse.py tests/test_coarse.py
git commit -m "feat: infer coarse ordered subtasks"
```

### Task 10: Adaptive Boundary Refinement and Acceptance

**Files:**
- Create: `src/qwen_annotate/refine.py`
- Create: `tests/test_refine.py`

**Interfaces:**
- Consumes: successful `CoarseDecision`, cameras, sampler, `QwenClient`, `SamplingConfig`, `validate_annotation`.
- Produces: `RefineDecision` and `run_refine(config, episode, coarse, sampler, client) -> RefineDecision`.

- [ ] **Step 1: Write boundary agreement and final validation tests**

```python
# tests/test_refine.py
import pytest
from qwen_annotate.refine import choose_agreed_boundary
from qwen_annotate.models import RefineResult

def refined(frame: int) -> RefineResult:
    return RefineResult(from_subtask_index=1, to_subtask_index=2,
        last_frame_before=frame-1, first_frame_after=frame,
        boundary_frame=frame, confidence=0.9, visible_cues=["release complete"])

def test_agreement_uses_median_integer_boundary() -> None:
    assert choose_agreed_boundary([refined(100), refined(106)], tolerance=12) == 103

def test_disagreement_has_no_boundary() -> None:
    assert choose_agreed_boundary([refined(100), refined(140)], tolerance=12) is None
```

- [ ] **Step 2: Run the refine tests and verify failure**

Run: `uv run pytest tests/test_refine.py -v`

Expected: import fails because `qwen_annotate.refine` does not exist.

- [ ] **Step 3: Implement adaptive broad and dense evidence sampling**

For each coarse center, sample every `round(source_fps / refine_fps)` frames over ±`refine_window_seconds` on configured cameras. Use the first response to choose a candidate; then sample every frame over ±`dense_radius_seconds` around that candidate for a second independent response. Clip all indices to the episode range and deduplicate frames by `(camera_key, frame_index)`.

- [ ] **Step 4: Implement agreement, final annotation, and review reasons**

Require both responses to retain the requested consecutive from/to indices. Accept their rounded median only when distance is within `agreement_tolerance_frames`. Build `FinalAnnotation(start_subtask_index=coarse.start, boundaries=...)`, run `validate_annotation`, and return `needs_review` for `refine_boundary_disagreement`, `refine_transition_mismatch`, `camera_evidence_conflict`, or any deterministic validation issue.

- [ ] **Step 5: Add an OOM degradation test**

Simulate a typed model OOM on the multi-camera broad request. Verify one retry keeps the primary camera and halves broad sample density; a second OOM returns `failed` with `model_oom` rather than accepting partial output.

- [ ] **Step 6: Run tests and commit**

Run: `uv run pytest tests/test_refine.py -v`

Expected: agreement, mismatch, validation, and OOM tests pass.

```bash
git add src/qwen_annotate/refine.py tests/test_refine.py
git commit -m "feat: refine and validate subtask boundaries"
```

### Task 11: Resumable Batch Pipeline and Operational CLI

**Files:**
- Create: `src/qwen_annotate/pipeline.py`
- Create: `src/qwen_annotate/cli.py`
- Create: `tests/test_pipeline.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: config, dataset index, workspace, coarse/refine functions, Qwen client, model manager.
- Produces: `annotate_dataset(config, max_concurrency, episode_indices=None) -> WorkspaceSummary` and Typer commands `model download`, `inspect`, `annotate`, and `status`.

- [ ] **Step 1: Write an interruption/resume integration test**

```python
# tests/test_pipeline.py
import pytest
from qwen_annotate.pipeline import annotate_dataset

@pytest.mark.asyncio
async def test_resume_skips_accepted_episode(tmp_path, pipeline_fixture) -> None:
    fixture = pipeline_fixture(tmp_path, episode_count=2, interrupt_after=1)
    with pytest.raises(KeyboardInterrupt):
        await annotate_dataset(fixture.config, services=fixture.services, max_concurrency=1)
    fixture.services.interrupt_after = None
    summary = await annotate_dataset(fixture.config, services=fixture.services, max_concurrency=1)
    assert summary.accepted == 2
    assert fixture.services.coarse_calls_by_episode[0] == 2
```

- [ ] **Step 2: Run pipeline tests and verify failure**

Run: `uv run pytest tests/test_pipeline.py tests/test_cli.py -v`

Expected: imports fail because pipeline and CLI modules do not exist.

- [ ] **Step 3: Implement episode orchestration and bounded concurrency**

Initialize/validate the manifest, build pending episode IDs, apply an optional explicit episode-index filter, and use `asyncio.Semaphore(max_concurrency)` around model work. Persist after coarse and refine stages. On Ctrl-C, stop scheduling new episodes, allow in-flight atomic saves, then re-raise. Classify source/video errors as failed and model uncertainty as needs_review. Append one structured JSON line per transition/error to `logs/run.jsonl`; never convert exceptions into accepted results.

- [ ] **Step 4: Implement inspect, annotate, and status commands**

`inspect CONFIG` prints version, FPS, cameras, episode/frame counts, and validation success. `annotate CONFIG --max-concurrency N --episodes 0,3,7` runs async orchestration on all or the explicit subset and exits nonzero if failed episodes exist. `status WORK_DIR --json` emits exact status counts and episode lists. `model download` delegates to Task 7 and prints the immutable SHA.

- [ ] **Step 5: Run CLI tests**

Use Typer `CliRunner` to verify invalid config exit code 2, failed inspection exit code 1, JSON status output, and that `model download` receives the expected repo/local directory without a real download.

Run: `uv run pytest tests/test_pipeline.py tests/test_cli.py -v`

Expected: all orchestration and command tests pass.

- [ ] **Step 6: Commit the runnable annotation pipeline**

```bash
git add src/qwen_annotate/pipeline.py src/qwen_annotate/cli.py tests/test_pipeline.py tests/test_cli.py
git commit -m "feat: run resumable annotation batches"
```

### Task 12: Static Review UI and Validated Human Decisions

**Files:**
- Create: `src/qwen_annotate/review.py`
- Create: `src/qwen_annotate/templates/review.html.j2`
- Create: `src/qwen_annotate/static/review.js`
- Create: `tests/test_review.py`
- Modify: `src/qwen_annotate/cli.py`

**Interfaces:**
- Consumes: workspace records, source videos, sampler, annotation constraints.
- Produces: `render_review_site(work_dir) -> Path`, `apply_human_decision(work_dir, episode_index, annotation) -> EpisodeRecord`, and CLI `review`.

- [ ] **Step 1: Write review rendering and rejection tests**

```python
# tests/test_review.py
def test_review_page_contains_reason_and_candidate_frames(review_workspace) -> None:
    page = render_review_site(review_workspace.root)
    html = page.read_text(encoding="utf-8")
    assert "coarse_sequence_disagreement" in html
    assert "episode_000003" in html
    assert "boundary-184-before.jpg" in html

def test_invalid_human_boundary_is_not_accepted(review_workspace) -> None:
    with pytest.raises(ValueError, match="boundary_order"):
        apply_human_decision(review_workspace.root, 3,
            FinalAnnotation(start_subtask_index=1, boundaries=[200, 100]))
```

- [ ] **Step 2: Run review tests and verify failure**

Run: `uv run pytest tests/test_review.py -v`

Expected: import fails because `qwen_annotate.review` does not exist.

- [ ] **Step 3: Implement self-contained static review output**

Render one index page and per-episode JSON under `previews/needs_review`. Generate JPEGs for global coarse frames and ±configured context around each candidate. The page displays reason codes, both model attempts, camera/frame labels, and editable integer fields. JavaScript exports a decision JSON file locally; it does not run a network server.

- [ ] **Step 4: Implement decision import and validation**

`qwen-annotate review WORK_DIR --apply decision.json` validates episode identity, start index, boundaries, mode, segment lengths, and source fingerprint. On success set status `accepted`, `decision_source="human"`, store the previous candidates, and atomically save. On failure leave the record unchanged and exit nonzero.

- [ ] **Step 5: Run tests and commit**

Run: `uv run pytest tests/test_review.py tests/test_cli.py -v`

Expected: rendering, invalid decision, valid decision, and CLI tests pass.

```bash
git add src/qwen_annotate/review.py src/qwen_annotate/templates src/qwen_annotate/static tests/test_review.py src/qwen_annotate/cli.py
git commit -m "feat: review uncertain episode annotations"
```

### Task 13: Full Dataset Conversion and Independent Release Validation

**Files:**
- Create: `src/qwen_annotate/converter.py`
- Create: `src/qwen_annotate/release_validator.py`
- Create: `tests/test_converter.py`
- Create: `tests/test_release_validator.py`
- Modify: `src/qwen_annotate/cli.py`

**Interfaces:**
- Consumes: source dataset, workspace manifest and accepted records.
- Produces: `convert_dataset(work_dir, output, accepted_only=False) -> ConversionReport`, `validate_release(path, source=None) -> ReleaseReport`, and CLI `convert`/`validate`.

- [ ] **Step 1: Write a full-conversion compatibility test**

```python
# tests/test_converter.py
def test_full_conversion_preserves_payload_and_writes_reference_schema(accepted_workspace, tmp_path) -> None:
    output = tmp_path / "annotated"
    report = convert_dataset(accepted_workspace.root, output)
    annotations = json.loads((output / "meta/lerobot_annotations.json").read_text())
    info = json.loads((output / "meta/info.json").read_text())
    assert report.episode_count == 2
    assert annotations["episodes"]["0"]["boundaries"] == [10]
    assert annotations["episodes"]["1"]["start_subtask_index"] == 1
    assert info["subtask_template"] == annotations["subtask_template"]
    assert sha256(output / report.payload_files[0]) == sha256(accepted_workspace.source / report.payload_files[0])
```

- [ ] **Step 2: Run conversion tests and verify failure**

Run: `uv run pytest tests/test_converter.py tests/test_release_validator.py -v`

Expected: imports fail because converter modules do not exist.

- [ ] **Step 3: Implement guarded full conversion**

Refuse unresolved records and existing output. Create `<output>.staging-<uuid>` beside output, copy the source tree with `shutil.copy2`, update `info.json`, create `lerobot_annotations.json` with source/work paths, template, episodes, primary camera, and UTC timestamp, then validate. Rename staging to output only after validation succeeds. Remove only the exact staging directory created by this run after a failed conversion.

- [ ] **Step 4: Implement independent release validation**

Re-read `info.json`, JSONL metadata, every parquet schema/row count, every video probe, and annotations. Verify counts, one annotation per episode, template equality, high-level mappings, mode constraints, and boundaries. When `source` is supplied, stream SHA-256 for each copied parquet/video pair and require equality.

- [ ] **Step 5: Add conversion refusal tests**

Cover existing output, pending record, needs-review record, failed record, invalid boundary, source changed after annotation, and copied payload checksum mismatch. Assert no final output directory exists after each failure.

- [ ] **Step 6: Wire CLI commands and commit**

`convert WORK_DIR --output PATH` prints the report and returns nonzero on refusal. `validate PATH --source PATH` can run without a workspace.

Run: `uv run pytest tests/test_converter.py tests/test_release_validator.py tests/test_cli.py -v`

Expected: all tests pass.

```bash
git add src/qwen_annotate/converter.py src/qwen_annotate/release_validator.py src/qwen_annotate/cli.py tests/test_converter.py tests/test_release_validator.py
git commit -m "feat: convert and validate annotated datasets"
```

### Task 14: Accepted-Only Reindexing and Statistics Integrity

**Files:**
- Modify: `src/qwen_annotate/converter.py`
- Create: `src/qwen_annotate/stats.py`
- Create: `tests/test_accepted_only.py`
- Create: `tests/test_stats.py`

**Interfaces:**
- Consumes: selected accepted episode records and LeRobot data features.
- Produces: `EpisodeRemap`, `rewrite_episode_parquet(...)`, `recompute_stats(...)`, and accepted-only conversion through the existing `convert_dataset` interface.

- [ ] **Step 1: Write a three-to-two episode reindexing test**

```python
# tests/test_accepted_only.py
def test_accepted_only_reindexes_every_reference(mixed_workspace, tmp_path) -> None:
    output = tmp_path / "accepted"
    convert_dataset(mixed_workspace.root, output, accepted_only=True)
    episodes = [json.loads(x) for x in (output / "meta/episodes.jsonl").read_text().splitlines()]
    assert [x["episode_index"] for x in episodes] == [0, 1]
    assert [x["length"] for x in episodes] == [12, 9]
    second = pq.read_table(output / "data/chunk-000/episode_000001.parquet")
    assert set(second["episode_index"].to_pylist()) == {1}
    assert second["index"].to_pylist() == list(range(12, 21))
```

- [ ] **Step 2: Run accepted-only tests and verify failure**

Run: `uv run pytest tests/test_accepted_only.py tests/test_stats.py -v`

Expected: accepted-only conversion fails because reindexing is not implemented.

- [ ] **Step 3: Implement explicit episode and global index remapping**

Sort selected source episodes, map them to contiguous output indices, calculate cumulative global offsets, rewrite parquet columns `episode_index`, `index`, and `frame_index`, preserve all action/state values and schema metadata, and name files through output `data_path`. Copy and rename each camera video through `video_path`; video bytes remain unchanged even though paths change.

- [ ] **Step 4: Rewrite all episode-level metadata**

Filter and remap `episodes.jsonl` and `episodes_stats.jsonl`; update annotation episode keys and embedded indices; rebuild `splits.train` as `0:<count>`; update `total_episodes`, `total_frames`, `total_videos`, `total_chunks`, `data_files_size_in_mb`, and `video_files_size_in_mb`. Preserve task indices and `tasks.jsonl` because accepted-only selection does not change task identities.

- [ ] **Step 5: Recompute aggregate feature statistics**

Implement streaming count, mean, variance, min, max, q01, q10, q50, q90, and q99 using PyArrow/NumPy-compatible arrays already supplied by PyArrow. Match the keys, list shapes, and JSON number representation of the source `stats.json`. Unit-test scalar and fixed-size-list features against hand-calculated values and require finite output.

- [ ] **Step 6: Run accepted-only conversion and validation tests**

Run: `uv run pytest tests/test_accepted_only.py tests/test_stats.py tests/test_release_validator.py -v`

Expected: reindexing and statistics tests pass, and the independent validator accepts the output.

- [ ] **Step 7: Commit accepted-only support**

```bash
git add src/qwen_annotate/converter.py src/qwen_annotate/stats.py tests/test_accepted_only.py tests/test_stats.py
git commit -m "feat: convert accepted episode subsets"
```

### Task 15: Golden-Set and Synthetic DAgger Evaluation

**Files:**
- Create: `src/qwen_annotate/evaluation.py`
- Create: `tests/test_evaluation.py`
- Modify: `src/qwen_annotate/cli.py`

**Interfaces:**
- Consumes: workspace results, golden `lerobot_annotations.json`, source episode lengths and FPS.
- Produces: `EvaluationMetrics`, `evaluate_boundaries(...)`, `evaluate_complete(...)`, `make_dagger_views(...)`, `evaluate_dagger(...)`, and CLI `evaluate`.

- [ ] **Step 1: Write exact metric tests**

```python
# tests/test_evaluation.py
def test_boundary_metrics_and_coverage_are_exact() -> None:
    predicted = {0: [102, 198], 1: [120, 240]}
    golden = {0: [100, 200], 1: [110, 250]}
    statuses = {0: "accepted", 1: "needs_review"}
    metrics = evaluate_boundaries(predicted, golden, statuses, fps=20)
    assert metrics.accepted_coverage == 0.5
    assert metrics.median_absolute_error_frames == 6.0
    assert metrics.p90_absolute_error_frames == 10.0
```

- [ ] **Step 2: Run evaluation tests and verify failure**

Run: `uv run pytest tests/test_evaluation.py -v`

Expected: import fails because `qwen_annotate.evaluation` does not exist.

- [ ] **Step 3: Implement complete-dataset metrics**

Align predictions and golden boundaries by episode and transition index. Report start-index accuracy, median and nearest-rank P90 in frames and seconds, accepted coverage, needs-review rate, failed rate, deterministic violation count, and false-accept count under a configurable obvious-error threshold of one second.

- [ ] **Step 4: Implement non-destructive synthetic DAgger views**

Represent a view as `(source_episode, start_frame, end_frame, expected_start_subtask_index, expected_boundaries_relative)`. Generate deterministic suffix views beginning halfway through a selected subtask and single-subtask views ending at least `min_segment_frames` before its next boundary. Do not create or rewrite video files; pass frame ranges into the sampler.

- [ ] **Step 5: Add the evaluate CLI and launch-gate report**

`evaluate WORK_DIR --golden DATASET --output metrics.json` writes metrics plus pass/fail for median ≤0.5 s, P90 ≤1 s, accepted coverage ≥0.85, constraint blocking =100%, and false accepts =0. Exit 0 only when every launch gate passes.

- [ ] **Step 6: Run tests and commit**

Run: `uv run pytest tests/test_evaluation.py tests/test_cli.py -v`

Expected: complete and synthetic-DAgger metric tests pass.

```bash
git add src/qwen_annotate/evaluation.py src/qwen_annotate/cli.py tests/test_evaluation.py
git commit -m "feat: evaluate annotation quality"
```

### Task 16: End-to-End Documentation, Real Model Smoke Test, and Golden Benchmark

**Files:**
- Modify: `README.md`
- Create: `docs/operations.md`
- Create: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: every public CLI and the approved design.
- Produces: operator documentation, a fake-model end-to-end regression test, a verified Qwen3.8 installation, and a recorded golden benchmark report.

- [ ] **Step 1: Add a fake-model end-to-end regression test**

Create a two-episode, two-camera fixture; run inspect, annotate with deterministic coarse/refine fake responses, status, full convert, and validate through Typer `CliRunner`. Assert the source tree hash is unchanged and the release contains compatible `info.json` and `lerobot_annotations.json`.

- [ ] **Step 2: Run the complete automated suite**

Run: `uv run pytest -q`

Expected: all tests pass with no warnings promoted by project code.

- [ ] **Step 3: Expand operator documentation with exact commands**

Document environment setup, immutable model download, the vLLM single-GPU smoke command, example configs, each CLI stage, resume semantics, reason codes, review import, full conversion, accepted-only conversion, release validation, multi-worker scaling, and recovery from timeout/OOM/corrupt video. Include this initial server command:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/user/zhoukr/envs/vllm/bin/vllm serve \
  /mnt/data/user/zhoukr/models/Qwen3.8-27B \
  --served-model-name Qwen/Qwen3.8-27B \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --port 8000
```

- [ ] **Step 4: Download and verify the official model**

Run: `uv run qwen-annotate model download --repo Qwen/Qwen3.8-27B --local-dir /mnt/data/user/zhoukr/models/Qwen3.8-27B`

Expected: command prints a 40-character revision SHA, `model-install.json` records it, and verification succeeds. If direct access is unavailable, retain the resumable partial directory and report the network error; do not mark this step complete.

- [ ] **Step 5: Run one real multimodal smoke episode**

Start vLLM with the documented command. Create a config targeting only reference episode 0 through the CLI episode filter, run annotation, and confirm both coarse responses and every refine response validate against their JSON schemas. Record peak GPU memory, request latency, visual frame count, and server version in `docs/operations.md`.

- [ ] **Step 6: Run and tune the 47-episode golden benchmark**

Annotate the full reference source, then run:

```bash
uv run qwen-annotate evaluate \
  /mnt/data/user/zhoukr/annotations/arrange_orange_juice_and_green_tea_2 \
  --golden /mnt/data/user/zhoukr/datasets/jokeru/arrange_orange_juice_and_green_tea_2_annotated \
  --output /mnt/data/user/zhoukr/annotations/arrange_orange_juice_and_green_tea_2/metrics.json
```

Expected: all launch gates pass. If a gate fails, change only prompt text, sampling density/window, or acceptance thresholds; increment `PROMPT_VERSION`; invalidate the affected workspace cache; rerun the benchmark; and record before/after metrics in `docs/operations.md`.

- [ ] **Step 7: Test multi-worker throughput after correctness passes**

Run the same fixed 8-episode subset at concurrency 1, 2, and 4 against available independent vLLM workers. Record episodes/hour, GPU memory, decode CPU load, and error rate. Set the documented default concurrency to the fastest configuration with zero OOM and zero additional needs-review results; do not hard-code eight workers.

- [ ] **Step 8: Run final verification and commit documentation**

Run: `uv run pytest -q && uv run qwen-annotate inspect examples/complete.yaml`

Expected: full suite passes and the real reference dataset inspection reports 47 valid episodes.

```bash
git add README.md docs/operations.md tests/test_end_to_end.py
git commit -m "docs: complete annotation operator workflow"
```

## Completion Checklist

- [ ] `git status --short` shows no unintended files.
- [ ] `uv run pytest -q` passes.
- [ ] The model installation is pinned and verified.
- [ ] The real single-episode multimodal smoke test succeeds.
- [ ] The 47-episode launch gates pass and metrics are saved.
- [ ] Source dataset hashes are unchanged.
- [ ] A converted release passes independent validation.
- [ ] `README.md` and `docs/operations.md` reproduce the successful commands.

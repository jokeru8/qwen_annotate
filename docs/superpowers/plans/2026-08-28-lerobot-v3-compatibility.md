# LeRobot v3.0 Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-release-complete LeRobot Dataset v3.0 input, annotation, review, full conversion, accepted-only conversion, and release validation while preserving v2.1 as the default and fully compatible path.

**Architecture:** Keep `robo_annotate.lerobot` as the public version-detecting facade, move storage-specific inspection behind v2.1/v3.0 adapters, and expose episode-local data/video slice references to the existing pipeline. Add isolated v3.0 publication and validation modules so ordinary conversion preserves shared payload bytes while accepted-only conversion rebuilds shared Parquet/MP4 shards and all relational metadata.

**Tech Stack:** Python 3.12, Pydantic 2, PyArrow 22, PyAV 15–16, FastAPI, vanilla JavaScript, pytest, optional `lerobot[dataset]==0.6.1` oracle.

**Spec:** `docs/superpowers/specs/2026-08-28-lerobot-v3-compatibility-design.md`

## Global Constraints

- Input auto-detects only exact `codebase_version` values `v2.1` and `v3.0`; unknown or missing versions fail closed.
- Output always preserves the source dataset version; do not add an output-version option or cross-version conversion.
- v2.1 remains the README/example/default path and its current public output remains byte/schema compatible except for the additive `dataset_version` report field.
- Runtime v3.0 support uses existing `pyarrow` and `av`; official LeRobot is optional and pinned to `lerobot[dataset]==0.6.1` under `v3-validation`.
- All annotation boundaries and model-facing frame indices remain episode-local half-open coordinates `[0, length)`.
- Ordinary v3.0 conversion preserves official core metadata and payload bytes; accepted-only v3.0 conversion re-encodes selected video slices and reports that fact.
- Source datasets and workspaces are read-only during conversion; staging validation must pass before no-replace atomic publication.
- User-facing project documentation is Chinese. Code identifiers, schema fields, errors, configuration snippets, default annotations, and generated action text are English.

## File Structure

- `src/robo_annotate/lerobot.py`: common public models, video probing, strict version detection, and adapter dispatch.
- `src/robo_annotate/lerobot_v21.py`: existing v2.1 metadata/path inspection behind the common interface.
- `src/robo_annotate/lerobot_v30.py`: v3.0 tasks/episode Parquet parsing, shared-slice resolution, and structural input checks.
- `src/robo_annotate/video.py`: episode-slice-aware frame extraction returning local indices.
- `src/robo_annotate/workspace.py`: dual-version manifest and slice-aware fingerprints.
- `src/robo_annotate/review_server.py`, `src/robo_annotate/review_web/app.js`: shared MP4 serving plus local/media time translation.
- `src/robo_annotate/publication_metadata.py`: version-neutral Robo annotation and task-info writing with optional v2.1 `info.json` extension.
- `src/robo_annotate/v30_data_writer.py`: accepted-only v3.0 task compaction and Parquet repacking.
- `src/robo_annotate/v30_video_writer.py`: selected video slice decoding, repacking, and new timestamp placement.
- `src/robo_annotate/converter_v30.py`: v3.0 full/accepted-only staging orchestration and official metadata/stat rebuilding.
- `src/robo_annotate/converter.py`: guarded publication facade and version dispatch.
- `src/robo_annotate/release_validator_v30.py`: independent v3.0 release validation.
- `src/robo_annotate/release_validator.py`: common report/services and strict version dispatch while retaining v2.1 checks.
- `tests/v30_fixtures.py`: real shared-Parquet/shared-MP4 v3.0 fixture builder.
- `tests/test_lerobot_v30.py`, `tests/test_video_v30.py`, `tests/test_converter_v30.py`, `tests/test_release_validator_v30.py`, `tests/test_lerobot_v30_oracle.py`: focused v3.0 coverage.
- Existing v2.1 tests remain authoritative regression coverage and are updated only for the common reference types/report field.

---

### Task 1: Build a Real Shared-Shard v3.0 Test Fixture

**Files:**
- Create: `tests/v30_fixtures.py`
- Create: `tests/test_v30_fixtures.py`

**Interfaces:**
- Produces: `make_lerobot_v30_fixture(tmp_path: Path, *, lengths: tuple[int, ...] = (6, 8, 5), fps: float = 5.0, cameras: tuple[str, ...] = ("observation.images.main", "observation.images.wrist")) -> Path`.
- Produces: `make_v30_config(root: Path, work: Path) -> AnnotationConfig` configured with `observation.images.main` as primary and both fixture cameras for refinement.
- Produces test helpers: `read_v30_info(root)`, `expected_color(camera_index, episode_index, frame_index)`, `colors_for_episode(camera_index, episode_index, length)`, `decoded_colors(path)`, `dominant_test_color(jpeg)`, `official_core_file_digests(root)`, and `source_tree_digest(root)`.
- Produces: deterministic RGB frames whose pixel color encodes `(camera_index, episode_index, local_frame_index)` for later slice assertions.
- Depends on: PyArrow and PyAV only; it must not import production adapters.

- [ ] **Step 1: Write the failing fixture contract test**

```python
def test_v30_fixture_uses_shared_real_payloads(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    info = json.loads((root / "meta/info.json").read_text())
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet")

    assert info["codebase_version"] == "v3.0"
    assert info["data_path"] == "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    assert episodes.num_rows == 3
    assert len(list((root / "data").glob("**/*.parquet"))) == 1
    assert len(list((root / "videos/observation.images.main").glob("**/*.mp4"))) == 1
    with av.open(str(next((root / "videos/observation.images.main").glob("**/*.mp4")))) as container:
        assert sum(1 for _ in container.decode(video=0)) == 19
```

- [ ] **Step 2: Run the fixture test and confirm the missing module failure**

Run: `uv run pytest tests/test_v30_fixtures.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'tests.v30_fixtures'`.

- [ ] **Step 3: Implement the fixture with official v3.0 columns and actual MP4 frames**

Create one data Parquet containing all episodes, one MP4 per camera containing all episodes, `meta/tasks.parquet`, and one episode metadata Parquet. Use these exact relational columns:

```python
row = {
    "episode_index": episode_index,
    "tasks": [task_text],
    "length": length,
    "data/chunk_index": 0,
    "data/file_index": 0,
    "dataset_from_index": global_offset,
    "dataset_to_index": global_offset + length,
    "meta/episodes/chunk_index": 0,
    "meta/episodes/file_index": 0,
}
for camera in cameras:
    row[f"videos/{camera}/chunk_index"] = 0
    row[f"videos/{camera}/file_index"] = 0
    row[f"videos/{camera}/from_timestamp"] = global_offset / fps
    row[f"videos/{camera}/to_timestamp"] = (global_offset + length) / fps
episode_rows.append(row)
```

Data rows must include `index`, `episode_index`, `frame_index`, `timestamp`, `task_index`, `observation.state`, and `action`. `info.json` must declare the official version, path templates, fps, `chunks_size`, splits, total counts, and full numeric/video feature schemas including dtype, shape, and names. Compute `meta/stats.json` from fixture payload values and add flattened `stats/{feature}/{metric}` episode metadata columns for numeric and video features. Encode CFR `yuv420p` MP4 with width and height divisible by two.

- [ ] **Step 4: Run fixture tests**

Run: `uv run pytest tests/test_v30_fixtures.py -v`

Expected: PASS; the decoded frame count is `sum((6, 8, 5)) == 19` for each camera.

- [ ] **Step 5: Commit the fixture**

```bash
git add tests/v30_fixtures.py tests/test_v30_fixtures.py
git commit -m "test: add shared-shard LeRobot v3 fixture"
```

### Task 2: Introduce Version-Neutral Episode References and Preserve v2.1

**Files:**
- Create: `src/robo_annotate/lerobot_v21.py`
- Modify: `src/robo_annotate/lerobot.py:1-230`
- Modify mechanically for reference access: `src/robo_annotate/coarse.py`, `src/robo_annotate/refine.py`, `src/robo_annotate/review.py`, `src/robo_annotate/review_server.py`, `src/robo_annotate/converter.py`, `src/robo_annotate/release_validator.py`, `src/robo_annotate/workspace.py`
- Modify: `tests/fixtures.py` to provide a common `make_episode_info(...)` builder for unit tests
- Modify: `tests/test_lerobot.py`
- Modify mechanically: `tests/test_accepted_only.py`, `tests/test_cli.py`, `tests/test_coarse.py`, `tests/test_converter.py`, `tests/test_evaluation.py`, `tests/test_pipeline.py`, `tests/test_refine.py`, `tests/test_review.py`, `tests/test_workspace.py`

**Interfaces:**
- Produces: `DatasetVersion = Literal["v2.1", "v3.0"]`.
- Produces: `EpisodeDataRef(path: Path, dataset_from_index: int, dataset_to_index: int)`.
- Produces: `EpisodeVideoRef(path: Path, from_timestamp: float, to_timestamp: float, fps: float)`.
- Produces: `detect_dataset_version(root: Path) -> DatasetVersion`.
- Produces: `inspect_dataset(config: AnnotationConfig, probe: Callable[[Path], VideoProbe] = probe_video) -> DatasetIndex` as the stable facade.
- Produces: `inspect_v21_dataset(config, info, probe) -> DatasetIndex` in `lerobot_v21.py`.

- [ ] **Step 1: Write failing common-reference and detection tests**

```python
def test_v21_is_exposed_through_local_slice_references(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12], 5.0, ["cam.eye"])
    dataset = inspect_dataset(make_config(root, tmp_path / "work"), fixed_probe)
    episode = dataset.episodes[0]

    assert dataset.version == "v2.1"
    assert episode.data == EpisodeDataRef(
        path=root / "data/chunk-000/episode_000000.parquet",
        dataset_from_index=0,
        dataset_to_index=12,
    )
    assert episode.videos["cam.eye"] == EpisodeVideoRef(
        path=root / "videos/chunk-000/cam.eye/episode_000000.mp4",
        from_timestamp=0.0,
        to_timestamp=12 / 5.0,
        fps=5.0,
    )

def test_version_detector_rejects_missing_and_unknown_versions(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12], 5.0, ["cam.eye"])
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    for value in (None, "v4.0"):
        mutated = {k: v for k, v in info.items() if k != "codebase_version"}
        if value is not None:
            mutated["codebase_version"] = value
        info_path.write_text(json.dumps(mutated))
        with pytest.raises(ValueError, match="Unsupported LeRobot codebase_version"):
            detect_dataset_version(root)
```

- [ ] **Step 2: Run the focused tests and verify missing type/function failures**

Run: `uv run pytest tests/test_lerobot.py -k "local_slice or version_detector" -v`

Expected: FAIL because `EpisodeDataRef`, `EpisodeVideoRef`, and `detect_dataset_version` do not exist.

- [ ] **Step 3: Add strict common models and move the current v2.1 inspector behind the facade**

Use frozen, extra-forbidden Pydantic models with finite timestamps and these invariants:

```python
class EpisodeDataRef(_Reference):
    path: Path
    dataset_from_index: int = Field(ge=0, strict=True)
    dataset_to_index: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.dataset_to_index <= self.dataset_from_index:
            raise ValueError("dataset_to_index must exceed dataset_from_index")
        return self

class EpisodeVideoRef(_Reference):
    path: Path
    from_timestamp: float = Field(ge=0)
    to_timestamp: float = Field(gt=0)
    fps: float = Field(gt=0)

class EpisodeInfo(BaseModel):
    episode_index: int
    length: int
    task: str
    data: EpisodeDataRef
    videos: dict[str, EpisodeVideoRef]
```

The v2.1 adapter must preserve every current validation and construct per-episode zero-origin refs. Keep `probe_video`, safe JSON/path helpers, and facade imports free of circular dependencies. In the same change, migrate all production path access to `episode.data.path` and `episode.videos[camera].path`; until Task 5, pass the latter path and `ref.fps` into the existing extractor. Update direct test constructors through `make_episode_info(...)` so the repository never contains an intermediate commit where `EpisodeInfo` has two competing representations or the full suite cannot collect.

- [ ] **Step 4: Run the complete v2.1 regression suite after the reference migration**

Run: `uv run pytest -q`

Expected: PASS with v2.1 behavior unchanged except assertions now use reference objects internally.

- [ ] **Step 5: Commit the adapter foundation**

```bash
git add src/robo_annotate/lerobot.py src/robo_annotate/lerobot_v21.py src/robo_annotate/coarse.py src/robo_annotate/refine.py src/robo_annotate/review.py src/robo_annotate/review_server.py src/robo_annotate/converter.py src/robo_annotate/release_validator.py src/robo_annotate/workspace.py tests/fixtures.py tests/test_lerobot.py tests/test_accepted_only.py tests/test_cli.py tests/test_coarse.py tests/test_converter.py tests/test_evaluation.py tests/test_pipeline.py tests/test_refine.py tests/test_review.py tests/test_workspace.py
git commit -m "refactor: add versioned LeRobot dataset references"
```

### Task 3: Implement Strict v3.0 Dataset Inspection

**Files:**
- Create: `src/robo_annotate/lerobot_v30.py`
- Modify: `src/robo_annotate/lerobot.py`
- Create: `tests/test_lerobot_v30.py`
- Modify: `src/robo_annotate/cli.py:45-67`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `EpisodeDataRef`, `EpisodeVideoRef`, `DatasetIndex`, `VideoProbe`, and safe root/version facts from Task 2.
- Produces: `inspect_v30_dataset(config: AnnotationConfig, info: dict[str, Any], probe: Callable[[Path], VideoProbe]) -> DatasetIndex`.
- Produces: `read_v30_tasks(root: Path) -> dict[int, str]` and `read_v30_episode_table(root: Path) -> pa.Table` for read-only inspection only; publishers must not use these functions as validation proof.

- [ ] **Step 1: Write failing happy-path and malformed-range tests**

```python
def test_inspects_shared_v30_slices(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))

    assert dataset.version == "v3.0"
    assert [item.length for item in dataset.episodes] == [6, 8, 5]
    assert dataset.episodes[0].data.path == dataset.episodes[2].data.path
    assert dataset.episodes[1].data.dataset_from_index == 6
    main = dataset.episodes[1].videos["observation.images.main"]
    assert main.from_timestamp == pytest.approx(6 / 5)
    assert main.to_timestamp == pytest.approx(14 / 5)
    assert main.path == dataset.episodes[0].videos["observation.images.main"].path

def test_rejects_v30_data_range_length_mismatch(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    position = table.schema.get_field_index("dataset_to_index")
    field = table.schema.field(position)
    table = table.set_column(position, field, pa.array([5, 14, 19], type=field.type))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match=r"episode 0.*dataset range"):
        inspect_dataset(make_v30_config(root, tmp_path / "work"))
```

- [ ] **Step 2: Run the v3.0 inspector tests**

Run: `uv run pytest tests/test_lerobot_v30.py -v`

Expected: FAIL because `v3.0` dispatch and metadata parsing are not implemented.

- [ ] **Step 3: Implement bounded Parquet metadata loading and shared-reference validation**

Read only files matching `meta/episodes/chunk-[0-9][0-9][0-9]/file-[0-9][0-9][0-9].parquet`, sort by chunk/file, and reject extra/missing required columns. Resolve templates with exactly these fields:

```python
DATA_FIELDS = {"chunk_index", "file_index"}
VIDEO_FIELDS = {"video_key", "chunk_index", "file_index"}
REQUIRED_EPISODE_COLUMNS = {
    "episode_index", "tasks", "length",
    "data/chunk_index", "data/file_index",
    "dataset_from_index", "dataset_to_index",
    "meta/episodes/chunk_index", "meta/episodes/file_index",
}
```

For each camera, require four `videos/{camera}/...` columns. Validate contiguous episode/task indices, one nonempty known task per episode, contained regular payloads, row slice values, fps/shape, timestamp order, and `round((to_timestamp - from_timestamp) * fps) == length` within one-frame time-base tolerance. Shared resolved paths are allowed; conflicting or overlapping slices in one camera shard are not.

- [ ] **Step 4: Add CLI version output and run inspection/security coverage**

Make `inspect` include `dataset_version` in both JSON and human output. Run:

`uv run pytest tests/test_lerobot_v30.py tests/test_lerobot.py tests/test_cli.py -v`

Expected: PASS, including mutations for missing shard, unknown template field, path traversal, symlink, noncontiguous episode index, missing camera columns, overlapping timestamp ranges, and mismatched data rows.

- [ ] **Step 5: Commit v3.0 inspection**

```bash
git add src/robo_annotate/lerobot.py src/robo_annotate/lerobot_v30.py src/robo_annotate/cli.py tests/test_lerobot_v30.py tests/test_cli.py
git commit -m "feat: inspect LeRobot v3 shared shards"
```

### Task 4: Make Workspace Provenance Slice- and Version-Aware

**Files:**
- Modify: `src/robo_annotate/workspace.py:266-370,730-770`
- Modify: `tests/test_workspace.py`

**Interfaces:**
- Consumes: `DatasetVersion`, `EpisodeDataRef`, and `EpisodeVideoRef` from Task 2.
- Produces: `RunManifest.dataset_version: Literal["v2.1", "v3.0"]`.
- Produces: `compute_source_fingerprint(dataset_root: Path, episode: EpisodeInfo) -> str` that hashes normalized slice metadata and every referenced file identity/digest.

- [ ] **Step 1: Write failing manifest and fingerprint tests**

```python
def test_v30_manifest_round_trips_and_slices_change_fingerprint(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    config = make_v30_config(root, tmp_path / "work")
    dataset = inspect_dataset(config)
    store = WorkspaceStore(config.work_dir)
    manifest = store.initialize(config, dataset, model_revision="a" * 40)

    assert manifest.dataset_version == "v3.0"
    first = compute_source_fingerprint(root, dataset.episodes[0])
    shifted = dataset.episodes[0].model_copy(update={
        "videos": dataset.episodes[1].videos,
    })
    assert compute_source_fingerprint(root, shifted) != first
```

Also add a test that mutates a shared Parquet byte and confirms fingerprints for all episodes referencing it change.

- [ ] **Step 2: Run focused workspace tests**

Run: `uv run pytest tests/test_workspace.py -k "v30 or fingerprint" -v`

Expected: FAIL because manifest version is restricted to `v2.1` and fingerprints expect path-only videos.

- [ ] **Step 3: Hash canonical refs and referenced shared files**

Construct the canonical fingerprint payload in stable camera order:

```python
payload = {
    "episode_index": episode.episode_index,
    "length": episode.length,
    "task": episode.task,
    "data": {
        "path": relative(episode.data.path),
        "from": episode.data.dataset_from_index,
        "to": episode.data.dataset_to_index,
        "sha256": sha256_file(episode.data.path),
    },
    "videos": {
        camera: {
            "path": relative(ref.path),
            "from_timestamp": ref.from_timestamp,
            "to_timestamp": ref.to_timestamp,
            "fps": ref.fps,
            "identity": regular_file_identity(ref.path),
        }
        for camera, ref in sorted(episode.videos.items())
    },
}
```

Retain current no-follow/containment checks and manifest invalidation behavior.

- [ ] **Step 4: Run all workspace and pipeline initialization regressions**

Run: `uv run pytest tests/test_workspace.py tests/test_pipeline.py -v`

Expected: PASS for both manifest versions and existing v2.1 resume/fail-closed behavior.

- [ ] **Step 5: Commit workspace provenance**

```bash
git add src/robo_annotate/workspace.py tests/test_workspace.py tests/test_pipeline.py
git commit -m "feat: fingerprint versioned episode slices"
```

### Task 5: Extract Frames from Episode Video Slices

**Files:**
- Modify: `src/robo_annotate/video.py:1-130`
- Modify: `src/robo_annotate/coarse.py:220-370`
- Modify: `src/robo_annotate/refine.py:338-640`
- Modify: `src/robo_annotate/review.py:120-470`
- Modify: `src/robo_annotate/release_validator.py` sampler service typing/calls
- Modify: `tests/test_video.py`, `tests/test_coarse.py`, `tests/test_refine.py`, `tests/test_review.py`
- Create: `tests/test_video_v30.py`

**Interfaces:**
- Consumes: `EpisodeVideoRef`.
- Produces: `extract_frames(video: EpisodeVideoRef, camera_key: str, indices: list[int]) -> list[FrameSample]`.
- Preserves: `FrameSample.frame_index` and `timestamp_seconds` are episode-local.

- [ ] **Step 1: Write failing slice extraction tests using color-coded fixture frames**

```python
def test_extracts_local_frames_from_middle_shared_video_slice(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    ref = dataset.episodes[1].videos["observation.images.main"]

    samples = extract_frames(ref, "observation.images.main", [0, 3, 7])

    assert [sample.frame_index for sample in samples] == [0, 3, 7]
    assert [sample.timestamp_seconds for sample in samples] == [0.0, 0.6, 1.4]
    assert [dominant_test_color(sample.jpeg) for sample in samples] == [
        expected_color(0, 1, 0), expected_color(0, 1, 3), expected_color(0, 1, 7)
    ]
```

Add rejection tests for index `length`, insufficient decoded frames, PTS before/after the slice, and mismatched ref fps.

- [ ] **Step 2: Run slice tests**

Run: `uv run pytest tests/test_video_v30.py -v`

Expected: FAIL because `extract_frames` still accepts a path and absolute file indices.

- [ ] **Step 3: Implement seek/decode mapping and update evidence call sites**

Use stream PTS when available and decode forward from a keyframe at or before `from_timestamp`. Map decoded media time to a local frame and only accept requested local indices:

```python
media_time = float(frame.pts * stream.time_base)
local_index = round((media_time - video.from_timestamp) * video.fps)
if 0 <= local_index < episode_frame_count and local_index in requested:
    found[local_index] = _make_sample(frame, camera_key, local_index, video.fps)
```

Derive `episode_frame_count = round((to_timestamp - from_timestamp) * fps)`, seek slightly before the start, reject duplicate/missing local frames, and stop at `to_timestamp` plus one frame tolerance. Update coarse/refine/review source guards to inspect `ref.path`, preserve the ref offsets after resolving the path, and call samplers with `(ref, camera, indices)`.

- [ ] **Step 4: Run frame/pipeline sampling regressions**

Run: `uv run pytest tests/test_video.py tests/test_video_v30.py tests/test_coarse.py tests/test_refine.py tests/test_review.py -v`

Expected: PASS; mock samplers receive `EpisodeVideoRef` and all evidence labels remain local.

- [ ] **Step 5: Commit slice-aware sampling**

```bash
git add src/robo_annotate/video.py src/robo_annotate/coarse.py src/robo_annotate/refine.py src/robo_annotate/review.py src/robo_annotate/release_validator.py tests/test_video.py tests/test_video_v30.py tests/test_coarse.py tests/test_refine.py tests/test_review.py
git commit -m "feat: sample episode-local frames from shared videos"
```

### Task 6: Make Browser Review Offset-Aware

**Files:**
- Modify: `src/robo_annotate/review_server.py:75-135`
- Modify: `src/robo_annotate/review_web/app.js`
- Modify: `tests/test_review_server.py`
- Modify: `tests/test_review_web.py`

**Interfaces:**
- Produces episode payload field `videos: dict[str, {url: str, from_timestamp: float, to_timestamp: float}]`.
- Produces JavaScript helpers `mediaTimeForFrame(frame, fps, source)`, `localFrameForMediaTime(time, fps, source)`, and `clampMediaTime(time, source)` through the existing test export object.
- Produces test-local `v30_review_client(tmp_path: Path, episode_index: int) -> TestClient` by initializing a v3.0 workspace with `make_lerobot_v30_fixture` and `make_v30_config`.

- [ ] **Step 1: Write failing server payload and JavaScript mapping tests**

```python
def test_v30_episode_payload_exposes_shared_video_offsets(tmp_path: Path) -> None:
    client = v30_review_client(tmp_path, episode_index=1)
    source = client.get("/api/episodes/1").json()["videos"]["observation.images.main"]
    assert source["url"].endswith("observation.images.main")
    assert source["from_timestamp"] == pytest.approx(1.2)
    assert source["to_timestamp"] == pytest.approx(2.8)
```

Add Node assertions that local frame `0` maps to media time `1.2`, media time `1.8` maps to local frame `3`, and values outside `[1.2, 2.8)` clamp to that range.

- [ ] **Step 2: Run server/web tests**

Run: `uv run pytest tests/test_review_server.py tests/test_review_web.py -v`

Expected: FAIL because the payload only exposes `video_urls` and JavaScript treats `currentTime` as episode-local.

- [ ] **Step 3: Implement local/media time translation**

Change the payload shape to:

```python
"videos": {
    camera: {
        "url": f"/api/episodes/{index}/videos/{quote(camera, safe='')}",
        "from_timestamp": ref.from_timestamp,
        "to_timestamp": ref.to_timestamp,
    }
    for camera, ref in episode.videos.items()
}
```

The HTTP endpoint continues serving the verified shared file descriptor. In JavaScript, set each element's `currentTime` to `from_timestamp + local_frame / fps`, subtract `from_timestamp` when updating the shared slider, synchronize cameras by local time, and pause/clamp before `to_timestamp`.

- [ ] **Step 4: Run review tests and the existing static UI behavior suite**

Run: `uv run pytest tests/test_review_server.py tests/test_review_web.py tests/test_review.py -v`

Expected: PASS for v2.1 zero offsets, v3.0 nonzero offsets, Range responses, path controls, frame stepping, and multi-camera synchronization.

- [ ] **Step 5: Commit review support**

```bash
git add src/robo_annotate/review_server.py src/robo_annotate/review_web/app.js tests/test_review_server.py tests/test_review_web.py tests/test_review.py
git commit -m "feat: review shared v3 video slices"
```

### Task 7: Extract Version-Neutral Public Annotation Writing

**Files:**
- Create: `src/robo_annotate/publication_metadata.py`
- Modify: `src/robo_annotate/converter.py:300-395`
- Modify: `tests/test_converter.py`
- Create: `tests/test_converter_v30.py`

**Interfaces:**
- Produces: `SelectedEpisode(record: EpisodeRecord, source_index: int, output_index: int, length: int)`.
- Produces: `write_public_annotations(staging: Path, output: Path, manifest: RunManifest, selected: Sequence[SelectedEpisode], converted_at: datetime, augmented_texts: Mapping[int, list[str]] | None, *, extend_info: bool) -> None`.
- Preserves: v2.1 calls with `extend_info=True`; v3.0 calls with `extend_info=False`.
- Produces test-local `prepared_v30_publication(tmp_path)` that returns a copied fixture staging tree, output path, initialized manifest, and one accepted `SelectedEpisode`.

- [ ] **Step 1: Write failing v3 metadata isolation and v2 regression tests**

```python
def test_v30_public_annotations_do_not_modify_info(tmp_path: Path) -> None:
    staging, output, manifest, selected = prepared_v30_publication(tmp_path)
    before = (staging / "meta/info.json").read_bytes()
    write_public_annotations(
        staging, output, manifest, selected, datetime(2026, 8, 28, tzinfo=UTC),
        None, extend_info=False,
    )
    assert (staging / "meta/info.json").read_bytes() == before
    assert json.loads((staging / "meta/lerobot_annotations.json").read_text())["episodes"]["0"]["boundaries"]
```

Retain the existing v2.1 assertion that `info.json` receives `subtask_template` and `high_level_instruction`.

- [ ] **Step 2: Run focused converter metadata tests**

Run: `uv run pytest tests/test_converter.py tests/test_converter_v30.py -k "public_annotations or reference_schema" -v`

Expected: FAIL because the writer is private and always mutates `info.json`.

- [ ] **Step 3: Move the writer into a focused module and parameterize only the version difference**

Use `SelectedEpisode.output_index` for public keys and `SelectedEpisode.source_index` to find per-source augmented text. The only `extend_info` branch is:

```python
if extend_info:
    info = read_bounded_json(staging / "meta/info.json")
    info["subtask_template"] = template
    info["high_level_instruction"] = instruction_map
    atomic_json(staging / "meta/info.json", info)
```

Always write the two namespaced Robo files with the existing exact schema and English action text.

- [ ] **Step 4: Run all current converter tests plus the new metadata tests**

Run: `uv run pytest tests/test_converter.py tests/test_accepted_only.py tests/test_converter_v30.py -v`

Expected: PASS; no v2.1 public metadata or augmentation behavior changes.

- [ ] **Step 5: Commit the metadata boundary**

```bash
git add src/robo_annotate/publication_metadata.py src/robo_annotate/converter.py tests/test_converter.py tests/test_converter_v30.py
git commit -m "refactor: isolate public annotation metadata"
```

### Task 8: Add Independent v3.0 Release Validation and Full Conversion

**Files:**
- Create: `src/robo_annotate/release_validator_v30.py`
- Modify: `src/robo_annotate/release_validator.py:1-370`
- Create: `src/robo_annotate/converter_v30.py`
- Modify: `src/robo_annotate/converter.py:50-190`
- Create: `tests/test_release_validator_v30.py`
- Modify: `tests/test_converter_v30.py`
- Modify: `tests/test_release_validator.py`

**Interfaces:**
- Produces: `validate_v30_release(root: Path, *, source_root: Path | None, services: ReleaseServices, expected_output_root: Path | None, deep_video_stats: bool) -> ReleaseReport`.
- Produces: `write_full_v30_release(staging, output, manifest, records, converted_at, augmented_texts) -> None`.
- Extends: `ReleaseReport.dataset_version: Literal["v2.1", "v3.0"]` and `ConversionReport.dataset_version`.
- Produces test-local helpers `accepted_v30_workspace(tmp_path)`, `converted_v30_release(tmp_path)`, and `mutate_episode_metadata(root, episode_index, field, value)` using the Task 1 fixture and existing workspace acceptance helpers.

- [ ] **Step 1: Write failing full-conversion and independent-corruption tests**

```python
def test_full_v30_conversion_preserves_core_payload_and_validates(tmp_path: Path) -> None:
    work, source, services = accepted_v30_workspace(tmp_path)
    source_files = official_core_file_digests(source)
    report = convert_dataset(work, tmp_path / "out", services=services)

    assert report.dataset_version == "v3.0"
    assert report.annotation_schema_version == "reference-v3.0"
    assert official_core_file_digests(report.output) == source_files
    assert validate_release(report.output, source=source, services=services).dataset_version == "v3.0"

def test_v30_validator_rejects_episode_slice_corruption(tmp_path: Path) -> None:
    output = converted_v30_release(tmp_path)
    mutate_episode_metadata(output, 1, "dataset_to_index", 999)
    with pytest.raises(ValueError, match=r"episode 1.*dataset_to_index"):
        validate_release(output)
```

- [ ] **Step 2: Run full v3 conversion/validation tests**

Run: `uv run pytest tests/test_converter_v30.py tests/test_release_validator_v30.py -v`

Expected: FAIL because reports and release validation only accept v2.1.

- [ ] **Step 3: Implement validator dispatch and deep v3.0 checks**

Have `validate_release` read bounded `meta/info.json` once and dispatch by exact version. The v3 validator independently parses tasks/episode Parquets and checks required schemas, contiguous indices, data rows, task references, shared video ranges, shapes/fps, annotations/task-info, stats, payload inventory, digests, and source byte equality when `source` is supplied. The custom high-level instruction must agree between Robo annotation files, but it is not required to equal the unchanged official task text in `tasks.parquet`.

Boundary preview must use the common ref API:

```python
ref = EpisodeVideoRef(
    path=video_path,
    from_timestamp=from_timestamp,
    to_timestamp=to_timestamp,
    fps=fps,
)
samples = services.extract_frames(ref, primary_camera, [boundary - 1, boundary])
```

Keep the existing v2 validator logic independent and set `dataset_version="v2.1"` in its report.

- [ ] **Step 4: Implement ordinary v3.0 converter dispatch and run regressions**

For full v3.0 conversion, use the existing safe tree copy, skip `_write_payload_sizes`, call `write_public_annotations(..., extend_info=False)`, validate staging, recheck the source digest, and publish atomically. Run:

`uv run pytest tests/test_converter_v30.py tests/test_release_validator_v30.py tests/test_converter.py tests/test_release_validator.py -v`

Expected: PASS; v3 official files remain byte-identical and both report versions validate.

- [ ] **Step 5: Commit full v3 publication and validation**

```bash
git add src/robo_annotate/release_validator.py src/robo_annotate/release_validator_v30.py src/robo_annotate/converter.py src/robo_annotate/converter_v30.py tests/test_converter_v30.py tests/test_release_validator_v30.py tests/test_release_validator.py
git commit -m "feat: publish and validate full LeRobot v3 datasets"
```

### Task 9: Repack Selected v3.0 Parquet and Tasks

**Files:**
- Create: `src/robo_annotate/v30_data_writer.py`
- Create: `tests/test_v30_data_writer.py`

**Interfaces:**
- Consumes: selected source `EpisodeInfo` values from Task 3 and source `info.json`.
- Produces: `DataPlacement(source_index, output_index, length, chunk_index, file_index, dataset_from_index, dataset_to_index, tasks)`.
- Produces: `V30DataWriteResult(placements, task_table, parquet_files, total_frames, aggregate_stats, episode_stats)`.
- Produces: `write_v30_data_subset(source: Path, staging: Path, dataset: DatasetIndex, source_indices: Sequence[int], info: Mapping[str, Any]) -> V30DataWriteResult`.

- [ ] **Step 1: Write failing selection/reindex/task-compaction test**

```python
def test_rewrites_selected_shared_parquet_and_compacts_tasks(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    result = write_v30_data_subset(root, staging, dataset, [0, 2], read_v30_info(root))
    table = pq.read_table(result.parquet_files[0])

    assert result.total_frames == 11
    assert table["episode_index"].to_pylist() == [0] * 6 + [1] * 5
    assert table["frame_index"].to_pylist() == list(range(6)) + list(range(5))
    assert table["index"].to_pylist() == list(range(11))
    assert [p.dataset_from_index for p in result.placements] == [0, 6]
    assert [p.dataset_to_index for p in result.placements] == [6, 11]
    assert result.task_table["task_index"].to_pylist() == list(range(result.task_table.num_rows))
```

- [ ] **Step 2: Run the data-writer test**

Run: `uv run pytest tests/test_v30_data_writer.py -v`

Expected: FAIL because `v30_data_writer` does not exist.

- [ ] **Step 3: Implement per-episode slicing, reindexing, task mapping, and deterministic packing**

For every selected episode, read its referenced shard and locate the unique contiguous row span whose global `index` values equal `range(dataset_from_index, dataset_to_index)`. Do not treat the global dataset index as a file-local row offset. Slice that located span, verify the source index columns, and replace only official index columns using their original Arrow types:

```python
replacements = {
    "episode_index": [output_index] * length,
    "frame_index": list(range(length)),
    "index": list(range(global_offset, global_offset + length)),
    "task_index": [task_remap[value] for value in source_task_indices],
}
```

Pack whole episode tables in source order, using each Arrow table's `nbytes` as the deterministic packing weight. Flush before the next episode when the accumulated weight would exceed `data_files_size_in_mb * 1024 * 1024`; allow an oversized episode alone. Map sequential file number to `chunk_index = file_number // chunks_size` and `file_index = file_number % chunks_size`. Write a fresh tasks Parquet and recompute numeric aggregate/episode stats from output tables.

- [ ] **Step 4: Run data writer edge cases**

Run: `uv run pytest tests/test_v30_data_writer.py -v`

Expected: PASS for shared input, two output files under a tiny threshold, a single oversized episode, mixed task use, schema metadata preservation, and source immutability.

- [ ] **Step 5: Commit data repacking**

```bash
git add src/robo_annotate/v30_data_writer.py tests/test_v30_data_writer.py
git commit -m "feat: repack selected v3 parquet shards"
```

### Task 10: Re-encode and Repack Selected v3.0 Video Slices

**Files:**
- Create: `src/robo_annotate/v30_video_writer.py`
- Create: `tests/test_v30_video_writer.py`

**Interfaces:**
- Consumes: `EpisodeVideoRef` and ordered selected episodes.
- Produces: `VideoPlacement(source_index, output_index, camera_key, chunk_index, file_index, from_timestamp, to_timestamp)`.
- Produces: `V30VideoWriteResult(placements, files_by_camera)`.
- Produces: `write_v30_video_subset(staging: Path, dataset: DatasetIndex, source_indices: Sequence[int], info: Mapping[str, Any]) -> V30VideoWriteResult`.

- [ ] **Step 1: Write failing middle-episode-removal video test**

```python
def test_reencodes_selected_video_slices_without_middle_episode(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    result = write_v30_video_subset(tmp_path / "staging", dataset, [0, 2], read_v30_info(root))
    placements = result.placements["observation.images.main"]

    assert [(p.from_timestamp, p.to_timestamp) for p in placements] == [
        (pytest.approx(0.0), pytest.approx(1.2)),
        (pytest.approx(1.2), pytest.approx(2.2)),
    ]
    assert decoded_colors(result.files_by_camera["observation.images.main"][0]) == (
        colors_for_episode(0, 0, 6) + colors_for_episode(0, 2, 5)
    )
```

- [ ] **Step 2: Run video writer tests**

Run: `uv run pytest tests/test_v30_video_writer.py -v`

Expected: FAIL because `v30_video_writer` does not exist.

- [ ] **Step 3: Implement exact slice decoding and CFR output packing**

Decode exactly `round((to_timestamp - from_timestamp) * fps)` frames per selected ref, convert to RGB, and feed a CFR output stream using the declared video feature size/fps. Do not copy packets because slice starts need not be keyframes. Record placement from actual written frame counts:

```python
from_timestamp = frames_already_written / fps
for frame in decode_episode_frames(ref, expected_length=episode.length):
    for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
        output.mux(packet)
frames_already_written += episode.length
to_timestamp = frames_already_written / fps
```

Flush the encoder on file close and keep whole episodes together. Use `width * height * 3 * length` as a deterministic conservative packing weight, flush before the next episode when the accumulated weight would exceed `video_files_size_in_mb * 1024 * 1024`, and allow an oversized episode alone. Derive chunk/file numbers with v3.0 `chunks_size` file semantics.

- [ ] **Step 4: Run video writer coverage**

Run: `uv run pytest tests/test_v30_video_writer.py -v`

Expected: PASS for two cameras, shared input, missing-middle selection, tiny-size repacking, exact decoded frame counts, shape/fps preservation, and one-frame timestamp tolerance.

- [ ] **Step 5: Commit video repacking**

```bash
git add src/robo_annotate/v30_video_writer.py tests/test_v30_video_writer.py
git commit -m "feat: repack selected v3 video slices"
```

### Task 11: Integrate Complete v3.0 accepted-only Publication

**Files:**
- Modify: `src/robo_annotate/converter_v30.py`
- Modify: `src/robo_annotate/converter.py:100-190`
- Modify: `src/robo_annotate/release_validator_v30.py`
- Modify: `tests/test_converter_v30.py`
- Modify: `tests/test_release_validator_v30.py`

**Interfaces:**
- Consumes: `V30DataWriteResult` from Task 9 and `V30VideoWriteResult` from Task 10.
- Produces: `rewrite_accepted_v30_release(staging: Path, source: Path, output: Path, manifest: RunManifest, dataset: DatasetIndex, records: Sequence[EpisodeRecord], converted_at: datetime, augmented_texts: Mapping[int, list[str]] | None, services: Any) -> int` returning output frame count.
- Produces test-local `selectively_accepted_v30_workspace(tmp_path, accepted)` by initializing all fixture episodes and marking only the requested source indices accepted.

- [ ] **Step 1: Write the failing end-to-end accepted-only test**

```python
def test_v30_accepted_only_removes_middle_episode_and_rebuilds_every_reference(tmp_path: Path) -> None:
    work, source, services = selectively_accepted_v30_workspace(tmp_path, accepted=(0, 2))
    source_digest_before = source_tree_digest(source)
    report = convert_dataset(work, tmp_path / "accepted", accepted_only=True, services=services)
    output = inspect_dataset(make_v30_config(report.output, tmp_path / "inspect-work"))

    assert report.dataset_version == "v3.0"
    assert report.episode_count == 2
    assert report.frame_count == 11
    assert [episode.episode_index for episode in output.episodes] == [0, 1]
    assert [episode.length for episode in output.episodes] == [6, 5]
    assert validate_release(report.output, services=services).valid
    assert source_tree_digest(source) == source_digest_before
```

Also assert augmented action text exists for both selected output episodes and no annotation/task-info entry references source episode `2` after remapping.

- [ ] **Step 2: Run accepted-only integration tests**

Run: `uv run pytest tests/test_converter_v30.py -k accepted_only -v`

Expected: FAIL because converter dispatch still uses the v2.1 accepted-only writer.

- [ ] **Step 3: Compose output metadata and stats from writer results**

Write `meta/tasks.parquet`, build one episode metadata row per paired data/video placement, and flatten episode stats using official slash-separated column names. Update `info.json` official facts only:

```python
info.update({
    "codebase_version": "v3.0",
    "total_episodes": len(selected),
    "total_frames": data.total_frames,
    "total_tasks": data.task_table.num_rows,
    "splits": {"train": f"0:{len(selected)}"},
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "episodes_path": "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
})
```

Derive all remaining official counts and size facts from staged files. Recompute numeric stats from the rewritten Arrow tables. Recompute aggregate video stats by decoding every output MP4 frame exactly once, and compute episode video stats by decoding each new `VideoPlacement` half-open range; do not pass shared files to the v2.1 one-file-per-episode stats helper. Flatten those episode stats into the official slash-separated metadata columns, write public annotations with `extend_info=False`, and return `data.total_frames`.

- [ ] **Step 4: Validate atomicity, corruption rejection, and v2.1 regressions**

Run: `uv run pytest tests/test_converter_v30.py tests/test_release_validator_v30.py tests/test_converter.py tests/test_accepted_only.py tests/test_release_validator.py -v`

Expected: PASS, including source change during rebuild, malformed slice, encoder failure, validator failure, cleanup of only owned staging, no final output on failure, and concurrent no-replace publication.

- [ ] **Step 5: Commit accepted-only v3.0 publication**

```bash
git add src/robo_annotate/converter_v30.py src/robo_annotate/converter.py src/robo_annotate/release_validator_v30.py tests/test_converter_v30.py tests/test_release_validator_v30.py
git commit -m "feat: publish accepted-only LeRobot v3 subsets"
```

### Task 12: Add the Optional Official LeRobot Loader Oracle

**Files:**
- Modify: `pyproject.toml`
- Modify mechanically: `uv.lock`
- Create: `tests/test_lerobot_v30_oracle.py`

**Interfaces:**
- Produces optional extra: `v3-validation = ["lerobot[dataset]==0.6.1"]`.
- Consumes the Task 1 fixture and Task 8/11 conversion helpers.
- Produces test-local `build_or_convert_artifact(tmp_path, kind) -> tuple[Path, int]` for `source`, `full`, and `accepted_only` oracle cases.

- [ ] **Step 1: Write the oracle test with an import skip**

```python
lerobot = pytest.importorskip("lerobot")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

@pytest.mark.parametrize("kind", ["source", "full", "accepted_only"])
def test_official_lerobot_loads_v30_artifact(tmp_path: Path, kind: str) -> None:
    root, expected_length = build_or_convert_artifact(tmp_path, kind)
    dataset = LeRobotDataset(repo_id=f"local/{kind}", root=root)
    assert len(dataset) == expected_length
    for index in sorted({0, expected_length // 2, expected_length - 1}):
        sample = dataset[index]
        assert int(sample["index"]) == index
        assert "observation.images.main" in sample
```

- [ ] **Step 2: Run without the optional dependency and confirm a clean skip**

Run: `uv run pytest tests/test_lerobot_v30_oracle.py -v`

Expected: SKIPPED with the import-skip reason; default installation remains unaffected.

- [ ] **Step 3: Add the optional extra and refresh the lock file**

Add exactly:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.4,<9", "pytest-asyncio>=1,<2"]
v3-validation = ["lerobot[dataset]==0.6.1"]
```

Run: `uv lock`

Expected: exit 0 and `uv.lock` records LeRobot 0.6.1 without changing the project Python range.

- [ ] **Step 4: Install and run the official oracle**

Run: `uv sync --extra dev --extra v3-validation && uv run pytest tests/test_lerobot_v30_oracle.py -v`

Expected: 3 PASS; official LeRobot loads the shared fixture, byte-preserving full output, and rebuilt accepted-only output and decodes sample frames.

- [ ] **Step 5: Commit the optional oracle**

```bash
git add pyproject.toml uv.lock tests/test_lerobot_v30_oracle.py
git commit -m "test: validate v3 outputs with official LeRobot"
```

### Task 13: Document Compatibility and Run the Complete Verification Matrix

**Files:**
- Modify: `README.md`
- Modify: `docs/operations.md`
- Modify: `examples/complete.yaml` only if comments mention v2.1 exclusivity
- Modify: `examples/dagger_patch.yaml` only if comments mention v2.1 exclusivity
- Modify: any test files needed solely to assert documentation/branding consistency

**Interfaces:**
- Documents: automatic version detection, v2.1 default positioning, v3.0 shared shards, version-preserving output, accepted-only video re-encoding, and optional oracle commands.
- Preserves: Chinese prose and English code/config/annotation examples.

- [ ] **Step 1: Write the failing documentation/CLI assertions**

Extend the branding/docs test to require these exact concepts:

```python
readme = Path("README.md").read_text(encoding="utf-8")
operations = Path("docs/operations.md").read_text(encoding="utf-8")
assert "LeRobot v2.1" in readme
assert "LeRobot v3.0" in readme
assert "自动识别" in readme
assert "--accepted-only" in operations
assert "重新编码" in operations
assert "v3-validation" in operations
```

- [ ] **Step 2: Run docs/branding tests**

Run: `uv run pytest tests/test_branding.py tests/test_cli.py -v`

Expected: FAIL because v3.0 behavior is not documented.

- [ ] **Step 3: Update Chinese documentation with English examples**

Put the v2.1 example first, state that no version flag is needed, show v3.0 inspect/convert commands using English paths and annotations, warn that accepted-only re-encodes shared MP4 slices, and document:

```bash
uv sync --extra dev --extra v3-validation
uv run pytest tests/test_lerobot_v30_oracle.py -v
```

- [ ] **Step 4: Run the complete default and optional verification matrix**

Run:

```bash
uv run pytest -q
uv run pytest tests/test_lerobot_v30.py tests/test_video_v30.py tests/test_converter_v30.py tests/test_release_validator_v30.py -v
uv sync --extra dev --extra v3-validation
uv run pytest tests/test_lerobot_v30_oracle.py -v
git diff --check
```

Expected: all default tests PASS, all focused v3.0 tests PASS, all 3 oracle cases PASS, and `git diff --check` produces no output.

- [ ] **Step 5: Commit documentation and final compatibility assertions**

```bash
git add README.md docs/operations.md examples/complete.yaml examples/dagger_patch.yaml tests/test_branding.py tests/test_cli.py
git commit -m "docs: explain LeRobot v3 compatibility"
```

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from robo_annotate.coarse import CoarseDecision
from robo_annotate.config import AnnotationConfig, Subtask
from robo_annotate.lerobot import DatasetIndex, EpisodeInfo
from robo_annotate.model_manager import ModelInstall
from robo_annotate.models import CoarseBoundary, CoarseResult, FinalAnnotation
from robo_annotate.pipeline import AuditPersistenceError, PipelineServices, WorkspaceSummary, _installed_model, annotate_dataset
from robo_annotate.refine import CameraSampling, RefineDecision, SamplingProvenance
from robo_annotate.qwen_client import InvalidModelResponse, ModelCallError, ModelOutOfMemory
from robo_annotate.workspace import EpisodeRecord, WorkspaceStore
from tests.fixtures import make_config


SHA = "a" * 40


def _sync_log_in_process(work_dir: str, results) -> None:
    import asyncio
    from pathlib import Path
    from robo_annotate.pipeline import _RunLog
    from robo_annotate.workspace import WorkspaceStore

    try:
        asyncio.run(_RunLog(Path(work_dir), lambda: datetime.now(UTC)).sync(WorkspaceStore(Path(work_dir))))
    except BaseException as exc:
        results.put(type(exc).__name__)
    else:
        results.put("ok")


def _dataset(tmp_path: Path, count: int = 2) -> DatasetIndex:
    root = tmp_path / "source"
    root.mkdir(parents=True)
    episodes = []
    for index in range(count):
        parquet = root / f"episode-{index}.parquet"
        video = root / f"episode-{index}.mp4"
        parquet.write_bytes(b"parquet")
        video.write_bytes(b"video")
        episodes.append(EpisodeInfo(episode_index=index, length=10, task="task", parquet=parquet, videos={"cam.eye": video}))
    return DatasetIndex(root=root.resolve(), version="v2.1", fps=5.0, camera_keys=["cam.eye"], episodes=episodes)


def _config(tmp_path: Path, dataset: DatasetIndex) -> AnnotationConfig:
    config = make_config(dataset.root, tmp_path / "work", mode="complete")
    return config.model_copy(update={"subtasks": [Subtask(skill="place", text="Place item")]})


def _coarse() -> CoarseDecision:
    attempt = CoarseResult(
        start_subtask_index=0, observed_subtask_indices=[0], coarse_boundaries=[],
        confidence=0.9, semantic_uncertainty_codes=[], boundary_precision_notes=[],
    )
    return CoarseDecision(
        mode="complete", subtask_count=1, frame_count=10, status="coarse_done",
        attempts=(attempt, attempt.model_copy(deep=True)), reasons=(), start_subtask_index=0,
        observed_subtask_indices=(0,), boundary_centers=(), sampled_frame_indices=((0, 9), (0, 9)),
    )


def _refine() -> RefineDecision:
    return RefineDecision(
        mode="complete", subtask_count=1, frame_count=10, min_segment_frames=8,
        agreement_tolerance_frames=12, start_subtask_index=0, observed_subtask_indices=(0,),
        coarse_boundary_centers=(), source_fps=5.0, refine_window_seconds=2.5,
        refine_fps=8.0, dense_radius_seconds=0.5, camera_order=("cam.eye",),
        broad_radius_frames=12, base_broad_stride=1, dense_radius_frames=2,
        status="accepted", attempts=(), provenance=(), reasons=(),
        annotation=FinalAnnotation(start_subtask_index=0, boundaries=[]),
    )


def _coarse_review() -> CoarseDecision:
    attempt = CoarseResult(
        start_subtask_index=0, observed_subtask_indices=[0], coarse_boundaries=[],
        confidence=0.2,
        semantic_uncertainty_codes=["transition_neighborhood_unclear"],
        boundary_precision_notes=[],
    )
    return CoarseDecision(
        mode="complete", subtask_count=1, frame_count=10, status="needs_review",
        attempts=(attempt, attempt.model_copy(deep=True)), reasons=("coarse_uncertain",),
        sampled_frame_indices=((0, 9), (0, 9)),
    )


def _refine_review(min_segment_frames: int) -> RefineDecision:
    return RefineDecision(
        mode="complete", subtask_count=1, frame_count=10, min_segment_frames=min_segment_frames,
        agreement_tolerance_frames=12, start_subtask_index=0, observed_subtask_indices=(0,),
        coarse_boundary_centers=(), source_fps=5.0, refine_window_seconds=2.5,
        refine_fps=8.0, dense_radius_seconds=0.5, camera_order=("cam.eye",),
        broad_radius_frames=12, base_broad_stride=1, dense_radius_frames=2,
        status="needs_review", attempts=(), provenance=(), reasons=("segment_too_short",),
        candidate_annotation=FinalAnnotation(start_subtask_index=0, boundaries=[]),
    )


def _coarse_two() -> CoarseDecision:
    attempt = CoarseResult(
        start_subtask_index=0, observed_subtask_indices=[0, 1],
        coarse_boundaries=[CoarseBoundary(from_subtask_index=0, to_subtask_index=1, estimated_frame=5, evidence="transition")],
        confidence=0.9,
        semantic_uncertainty_codes=[],
        boundary_precision_notes=[],
    )
    return CoarseDecision(
        mode="complete", subtask_count=2, frame_count=10, status="coarse_done",
        attempts=(attempt, attempt.model_copy(deep=True)), reasons=(), start_subtask_index=0,
        observed_subtask_indices=(0, 1), boundary_centers=(5,), sampled_frame_indices=((0, 9), (0, 9)),
    )


def _refine_failed() -> RefineDecision:
    provenance = SamplingProvenance(
        boundary_index=0, from_subtask_index=0, to_subtask_index=1, stage="broad",
        pass_id=0, request_center=5, radius_frames=12, stride=1, cameras=("cam.eye",),
        samples=(CameraSampling(camera_key="cam.eye", frame_indices=tuple(range(10))),), outcome="model_oom",
    )
    return RefineDecision(
        mode="complete", subtask_count=2, frame_count=10, min_segment_frames=8,
        agreement_tolerance_frames=12, start_subtask_index=0, observed_subtask_indices=(0, 1),
        coarse_boundary_centers=(5,), source_fps=5.0, refine_window_seconds=2.5,
        refine_fps=8.0, dense_radius_seconds=0.5, camera_order=("cam.eye",),
        broad_radius_frames=12, base_broad_stride=1, dense_radius_frames=2,
        status="failed", attempts=(), provenance=(provenance,), reasons=("model_oom",),
        failure_category="model_oom",
    )


@dataclass
class FakeRuntime:
    dataset: DatasetIndex
    coarse_calls: list[int] = field(default_factory=list)
    refine_calls: list[int] = field(default_factory=list)

    def services(self) -> PipelineServices:
        async def coarse(config, episode, *, client, source_fps, expected_source_fingerprint):
            self.coarse_calls.append(episode.episode_index)
            return _coarse()

        async def refine(config, episode, coarse, *, client, source_fps, expected_source_fingerprint):
            self.refine_calls.append(episode.episode_index)
            return _refine()

        return PipelineServices(
            inspect_dataset=lambda config: self.dataset,
            workspace_factory=WorkspaceStore,
            resolve_model=lambda config: ModelInstall(config.model.name, SHA, config.model.local_path.resolve(), datetime(2026, 8, 22, tzinfo=UTC)),
            client_factory=lambda config: object(),
            run_coarse=coarse,
            run_refine=refine,
        )


def test_workspace_summary_is_strict_and_validates_integrity() -> None:
    raw = {"total": 3, "counts": {"pending": 1, "coarse_done": 0, "refine_done": 0, "accepted": 1, "needs_review": 1, "failed": 0}, "episode_indices": {"pending": [2], "coarse_done": [], "refine_done": [], "needs_review": [1], "failed": []}}
    summary = WorkspaceSummary.from_store_summary(raw)
    assert (summary.total, summary.pending, summary.accepted, summary.needs_review) == (3, 1, 1, 1)
    assert summary.nonaccepted_episode_indices == (1, 2)
    with pytest.raises((ValueError, ValidationError)):
        WorkspaceSummary.from_store_summary(raw | {"total": 4})


@pytest.mark.asyncio
async def test_pending_episodes_flow_through_durable_stages_and_terminal_resume_skips_calls(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    config = _config(tmp_path, dataset)
    runtime = FakeRuntime(dataset)
    summary = await annotate_dataset(config, 1, services=runtime.services())
    assert summary.accepted == 2
    assert runtime.coarse_calls == [0, 1]
    assert runtime.refine_calls == [0, 1]
    resumed = await annotate_dataset(config, 1, services=runtime.services())
    assert resumed.accepted == 2
    assert runtime.coarse_calls == [0, 1]
    lines = [__import__("json").loads(line) for line in (config.work_dir / "logs" / "run.jsonl").read_text().splitlines()]
    assert len(lines) == 6
    for episode in (0, 1):
        chain = [(line["from_status"], line["to_status"]) for line in lines if line["episode"] == episode]
        assert chain == [("pending", "coarse_done"), ("coarse_done", "refine_done"), ("refine_done", "accepted")]
        assert len({line["event_id"] for line in lines if line["episode"] == episode}) == 3


@pytest.mark.asyncio
async def test_episode_filter_and_strict_argument_validation(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 3)
    config = _config(tmp_path, dataset)
    runtime = FakeRuntime(dataset)
    summary = await annotate_dataset(config, 2, [2, 0], services=runtime.services())
    assert summary.pending == 1 and runtime.coarse_calls == [0, 2]
    for invalid in (0, True, 1.5):
        with pytest.raises((TypeError, ValueError)):
            await annotate_dataset(config, invalid, services=runtime.services())
    for invalid in ([0, 0], [-1], [3], [True]):
        with pytest.raises((TypeError, ValueError)):
            await annotate_dataset(config, 1, invalid, services=runtime.services())


@pytest.mark.asyncio
async def test_resume_after_refine_interruption_does_not_repeat_coarse(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    runtime = FakeRuntime(dataset)
    services = runtime.services()
    original_refine = services.run_refine
    interrupted = False

    async def refine(*args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return await original_refine(*args, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        await annotate_dataset(config, 1, services=PipelineServices(
            inspect_dataset=services.inspect_dataset, workspace_factory=services.workspace_factory,
            resolve_model=services.resolve_model, client_factory=services.client_factory,
            run_coarse=services.run_coarse, run_refine=refine,
        ))
    assert WorkspaceStore(config.work_dir).load_episode(0).status == "coarse_done"
    summary = await annotate_dataset(config, 1, services=services)
    assert summary.accepted == 1
    assert runtime.coarse_calls == [0]
    assert runtime.refine_calls == [0]


@pytest.mark.asyncio
async def test_interruption_after_first_acceptance_resumes_only_remaining_episode(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 2)
    config = _config(tmp_path, dataset)
    runtime = FakeRuntime(dataset)
    base = runtime.services()
    interrupted = False

    async def coarse(config, episode, **kwargs):
        nonlocal interrupted
        runtime.coarse_calls.append(episode.episode_index)
        if episode.episode_index == 1 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return _coarse()

    services = PipelineServices(
        inspect_dataset=base.inspect_dataset, workspace_factory=base.workspace_factory,
        resolve_model=base.resolve_model, client_factory=base.client_factory,
        run_coarse=coarse, run_refine=base.run_refine,
    )
    with pytest.raises(KeyboardInterrupt):
        await annotate_dataset(config, 1, services=services)
    assert WorkspaceStore(config.work_dir).load_episode(0).status == "accepted"
    assert WorkspaceStore(config.work_dir).load_episode(1).status == "pending"
    assert (await annotate_dataset(config, 1, services=services)).accepted == 2
    assert runtime.coarse_calls == [0, 1, 1]


@pytest.mark.asyncio
async def test_concurrency_is_exactly_bounded_and_shared_client_is_reused(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 5)
    config = _config(tmp_path, dataset)
    active = peak = 0
    client = object()
    seen_clients = []

    async def coarse(config, episode, *, client, **kwargs):
        nonlocal active, peak
        seen_clients.append(client)
        active += 1
        peak = max(peak, active)
        await __import__("asyncio").sleep(0.01)
        active -= 1
        return _coarse()

    async def refine(config, episode, coarse, *, client, **kwargs):
        seen_clients.append(client)
        return _refine()

    services = FakeRuntime(dataset).services()
    services = PipelineServices(
        inspect_dataset=services.inspect_dataset, workspace_factory=services.workspace_factory,
        resolve_model=services.resolve_model, client_factory=lambda config: client,
        run_coarse=coarse, run_refine=refine,
    )
    assert (await annotate_dataset(config, 2, services=services)).accepted == 5
    assert peak == 2 and set(map(id, seen_clients)) == {id(client)}
    lines = [__import__("json").loads(line) for line in (config.work_dir / "logs" / "run.jsonl").read_text().splitlines()]
    assert len(lines) == 15 and all(isinstance(item, dict) for item in lines)
    assert {item["episode"] for item in lines} == set(range(5))


@pytest.mark.parametrize(
    ("error", "status", "category"),
    [
        (InvalidModelResponse("bad", attempt_count=1, excerpt="SECRET raw output"), "needs_review", "invalid_model_response"),
        (ModelOutOfMemory("oom", attempt_count=1), "failed", "model_oom"),
        (ModelCallError("down", attempt_count=2), "failed", "model_call"),
        (OSError("SECRET source path"), "failed", "source_or_video"),
        (ValueError("SECRET corrupt video"), "failed", "source_or_video"),
        (RuntimeError("SECRET unexpected"), "failed", "unexpected_error"),
    ],
)
@pytest.mark.asyncio
async def test_errors_are_classified_without_leaking_exception_content(tmp_path: Path, error, status: str, category: str) -> None:
    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    base = FakeRuntime(dataset).services()

    async def fail(*args, **kwargs):
        raise error

    services = PipelineServices(
        inspect_dataset=base.inspect_dataset, workspace_factory=base.workspace_factory,
        resolve_model=base.resolve_model, client_factory=base.client_factory,
        run_coarse=fail, run_refine=base.run_refine,
    )
    summary = await annotate_dataset(config, 1, services=services)
    assert getattr(summary, status) == 1
    record = WorkspaceStore(config.work_dir).load_episode(0)
    assert (record.failure_category if status == "failed" else record.review_reasons[0]) == category
    log = (config.work_dir / "logs" / "run.jsonl").read_text()
    assert category in log and "SECRET" not in log


@pytest.mark.asyncio
async def test_model_install_must_match_config_and_exact_revision(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    base = FakeRuntime(dataset).services()
    wrong = ModelInstall("Other/Model", SHA, config.model.local_path.resolve(), datetime(2026, 8, 22, tzinfo=UTC))
    services = PipelineServices(
        inspect_dataset=base.inspect_dataset, resolve_model=lambda config: wrong,
        workspace_factory=base.workspace_factory, client_factory=base.client_factory,
        run_coarse=base.run_coarse, run_refine=base.run_refine,
    )
    with pytest.raises(ValueError, match="does not match"):
        await annotate_dataset(config, 1, services=services)


@pytest.mark.asyncio
async def test_symlink_run_log_is_rejected_before_any_model_call(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    runtime = FakeRuntime(dataset)
    WorkspaceStore(config.work_dir).initialize(config, dataset, SHA)
    target = tmp_path / "outside.log"
    target.write_text("")
    (config.work_dir / "logs" / "run.jsonl").symlink_to(target)
    with pytest.raises(AuditPersistenceError):
        await annotate_dataset(config, 1, services=runtime.services())
    assert runtime.coarse_calls == [] and target.read_text() == ""


@pytest.mark.asyncio
async def test_default_model_resolver_reads_matching_verified_install_metadata(tmp_path: Path) -> None:
    import json

    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    local = (tmp_path / "model").resolve()
    local.mkdir()
    model = config.model.model_copy(update={"local_path": local, "revision": SHA})
    config = config.model_copy(update={"model": model})
    install = ModelInstall(config.model.name, SHA, local, datetime(2026, 8, 22, tzinfo=UTC))
    (local / "model-install.json").write_text(json.dumps(install.to_dict()))
    runtime = FakeRuntime(dataset)
    base = runtime.services()
    services = PipelineServices(
        inspect_dataset=base.inspect_dataset, workspace_factory=base.workspace_factory,
        client_factory=base.client_factory, run_coarse=base.run_coarse, run_refine=base.run_refine,
    )
    assert (await annotate_dataset(config, 1, services=services)).accepted == 1

    metadata = install.to_dict()
    duplicate = json.dumps(metadata).replace('{"repo":', '{"repo":"Evil/Model","repo":', 1)
    (local / "model-install.json").write_text(duplicate)
    duplicate_config = config.model_copy(update={"work_dir": tmp_path / "duplicate-work"})
    with pytest.raises(ValueError, match="invalid"):
        await annotate_dataset(duplicate_config, 1, services=services)
    (local / "model-install.json").write_text(json.dumps(metadata))

    bad_config = config.model_copy(update={"work_dir": tmp_path / "other-work", "model": model.model_copy(update={"revision": "b" * 40})})
    with pytest.raises(ValueError, match="revision"):
        await annotate_dataset(bad_config, 1, services=services)


def test_model_install_reader_rejects_symlink_oversize_and_path_swap(monkeypatch, tmp_path: Path) -> None:
    import json
    import robo_annotate.pipeline as pipeline_module

    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    local = (tmp_path / "model").resolve()
    local.mkdir()
    config = config.model_copy(update={"model": config.model.model_copy(update={"local_path": local})})
    install = ModelInstall(config.model.name, SHA, local, datetime(2026, 8, 22, tzinfo=UTC))
    metadata = local / "model-install.json"
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(install.to_dict()))
    metadata.symlink_to(outside)
    with pytest.raises(ValueError, match="invalid|required"):
        _installed_model(config)

    metadata.unlink()
    metadata.write_bytes(b" " * (70 * 1024))
    with pytest.raises(ValueError, match="invalid"):
        _installed_model(config)

    valid = json.dumps(install.to_dict()).encode()
    metadata.write_bytes(valid)
    original_read = pipeline_module.os.read
    swapped = False

    def swapping_read(fd, size):
        nonlocal swapped
        if not swapped:
            swapped = True
            metadata.unlink()
            metadata.write_bytes(valid)
        return original_read(fd, size)

    monkeypatch.setattr(pipeline_module.os, "read", swapping_read)
    with pytest.raises(ValueError, match="invalid"):
        _installed_model(config)


@pytest.mark.asyncio
async def test_injected_dataset_episode_order_must_match_indices(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 2)
    config = _config(tmp_path, dataset)
    reordered = dataset.model_copy(update={"episodes": list(reversed(dataset.episodes))})
    runtime = FakeRuntime(reordered)
    with pytest.raises(ValueError, match="ordered|indices"):
        await annotate_dataset(config, 1, services=runtime.services())


@pytest.mark.asyncio
async def test_tampered_but_internally_valid_coarse_audit_fails_workspace_state(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    runtime = FakeRuntime(dataset)
    base = runtime.services()

    async def interrupt_refine(*args, **kwargs):
        raise KeyboardInterrupt()

    interrupted_services = PipelineServices(
        inspect_dataset=base.inspect_dataset, workspace_factory=base.workspace_factory,
        resolve_model=base.resolve_model, client_factory=base.client_factory,
        run_coarse=base.run_coarse, run_refine=interrupt_refine,
    )
    with pytest.raises(KeyboardInterrupt):
        await annotate_dataset(config, 1, services=interrupted_services)
    path = config.work_dir / "episodes" / "episode_000000.json"
    payload = __import__("json").loads(path.read_text())
    payload["sampling_details"]["coarse_decision"]["frame_count"] = 11
    payload["sampling_details"]["coarse_decision"]["sampled_frame_indices"] = [[0, 10], [0, 10]]
    path.write_text(__import__("json").dumps(payload))
    summary = await annotate_dataset(config, 1, services=base)
    assert summary.failed == 1
    assert WorkspaceStore(config.work_dir).load_episode(0).failure_category == "workspace_state"
    assert runtime.refine_calls == []


@pytest.mark.asyncio
async def test_interruption_stops_new_scheduling_but_drains_other_inflight_episode(tmp_path: Path) -> None:
    import asyncio

    dataset = _dataset(tmp_path, 3)
    config = _config(tmp_path, dataset)
    base = FakeRuntime(dataset).services()
    episode_one_started = asyncio.Event()

    async def coarse(config, episode, **kwargs):
        if episode.episode_index == 1:
            episode_one_started.set()
            await asyncio.sleep(0.02)
        return _coarse()

    async def refine(config, episode, coarse, **kwargs):
        if episode.episode_index == 0:
            await episode_one_started.wait()
            raise KeyboardInterrupt()
        return _refine()

    services = PipelineServices(
        inspect_dataset=base.inspect_dataset, workspace_factory=base.workspace_factory,
        resolve_model=base.resolve_model, client_factory=base.client_factory,
        run_coarse=coarse, run_refine=refine,
    )
    with pytest.raises(KeyboardInterrupt):
        await annotate_dataset(config, 2, services=services)
    store = WorkspaceStore(config.work_dir)
    assert store.load_episode(0).status == "coarse_done"
    assert store.load_episode(1).status == "coarse_done"
    assert store.load_episode(2).status == "pending"


@pytest.mark.asyncio
async def test_outer_cancellation_finishes_current_atomic_transition_only(monkeypatch, tmp_path: Path) -> None:
    import asyncio
    import robo_annotate.pipeline as pipeline_module

    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    runtime = FakeRuntime(dataset)
    entered_log = asyncio.Event()
    release_log = asyncio.Event()
    original = pipeline_module._RunLog.sync
    sync_calls = 0

    async def blocked_log(self, *args, **kwargs):
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 1:
            return await original(self, *args, **kwargs)
        entered_log.set()
        await release_log.wait()
        await original(self, *args, **kwargs)

    monkeypatch.setattr(pipeline_module._RunLog, "sync", blocked_log)
    task = asyncio.create_task(annotate_dataset(config, 1, services=runtime.services()))
    await entered_log.wait()
    task.cancel()
    release_log.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert WorkspaceStore(config.work_dir).load_episode(0).status == "coarse_done"
    assert runtime.refine_calls == []


@pytest.mark.asyncio
async def test_coarse_and_refine_review_outcomes_are_quarantined_with_audit(tmp_path: Path) -> None:
    first_dataset = _dataset(tmp_path / "first", 1)
    first_config = _config(tmp_path / "first", first_dataset)
    first = FakeRuntime(first_dataset).services()

    async def coarse_review(*args, **kwargs):
        return _coarse_review()

    services = PipelineServices(
        inspect_dataset=first.inspect_dataset, workspace_factory=first.workspace_factory,
        resolve_model=first.resolve_model, client_factory=first.client_factory,
        run_coarse=coarse_review, run_refine=first.run_refine,
    )
    assert (await annotate_dataset(first_config, 1, services=services)).needs_review == 1
    coarse_record = WorkspaceStore(first_config.work_dir).load_episode(0)
    assert coarse_record.review_reasons == ["coarse_uncertain"]
    assert "coarse_decision" in coarse_record.sampling_details

    second_dataset = _dataset(tmp_path / "second", 1)
    second_config = _config(tmp_path / "second", second_dataset)
    second_config = second_config.model_copy(update={
        "sampling": second_config.sampling.model_copy(update={"min_segment_frames": 11})
    })
    second = FakeRuntime(second_dataset).services()

    async def refine_review(*args, **kwargs):
        return _refine_review(11)

    services = PipelineServices(
        inspect_dataset=second.inspect_dataset, workspace_factory=second.workspace_factory,
        resolve_model=second.resolve_model, client_factory=second.client_factory,
        run_coarse=second.run_coarse, run_refine=refine_review,
    )
    assert (await annotate_dataset(second_config, 1, services=services)).needs_review == 1
    refine_record = WorkspaceStore(second_config.work_dir).load_episode(0)
    assert refine_record.validation_issues[0].code == "segment_too_short"
    assert "refine_decision" in refine_record.sampling_details


@pytest.mark.asyncio
async def test_failed_refine_decision_transitions_directly_to_model_oom(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset).model_copy(update={
        "subtasks": [Subtask(skill="first", text="First"), Subtask(skill="second", text="Second")]
    })
    base = FakeRuntime(dataset).services()

    async def coarse(*args, **kwargs):
        return _coarse_two()

    async def refine(*args, **kwargs):
        return _refine_failed()

    services = PipelineServices(
        inspect_dataset=base.inspect_dataset, workspace_factory=base.workspace_factory,
        resolve_model=base.resolve_model, client_factory=base.client_factory,
        run_coarse=coarse, run_refine=refine,
    )
    assert (await annotate_dataset(config, 1, services=services)).failed == 1
    record = WorkspaceStore(config.work_dir).load_episode(0)
    assert record.failure_category == "model_oom"
    assert record.sampling_details["refine_decision"]["status"] == "failed"


@pytest.mark.asyncio
async def test_resume_from_durably_saved_refine_done_only_finalizes(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    runtime = FakeRuntime(dataset)
    base = runtime.services()

    class InterruptAfterRefineSave:
        def __init__(self, root):
            self.inner = WorkspaceStore(root)
            self.root = self.inner.root
            self.interrupted = False

        def initialize(self, *args, **kwargs):
            return self.inner.initialize(*args, **kwargs)

        def load_episode(self, index):
            return self.inner.load_episode(index)

        def summary(self):
            return self.inner.summary()

        def save_episode(self, record):
            self.inner.save_episode(record)
            if record.status == "refine_done" and not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt()

    wrapped = InterruptAfterRefineSave(config.work_dir)
    services = PipelineServices(
        inspect_dataset=base.inspect_dataset, workspace_factory=lambda root: wrapped,
        resolve_model=base.resolve_model, client_factory=base.client_factory,
        run_coarse=base.run_coarse, run_refine=base.run_refine,
    )
    with pytest.raises(KeyboardInterrupt):
        await annotate_dataset(config, 1, services=services)
    assert WorkspaceStore(config.work_dir).load_episode(0).status == "refine_done"
    summary = await annotate_dataset(config, 1, services=base)
    assert summary.accepted == 1
    assert runtime.coarse_calls == [0] and runtime.refine_calls == [0]
    lines = [__import__("json").loads(line) for line in (config.work_dir / "logs" / "run.jsonl").read_text().splitlines()]
    assert [(line["from_status"], line["to_status"]) for line in lines] == [
        ("pending", "coarse_done"), ("coarse_done", "refine_done"), ("refine_done", "accepted")
    ]


@pytest.mark.asyncio
async def test_deleted_log_is_rebuilt_from_outbox_without_duplicates(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    runtime = FakeRuntime(dataset)
    services = runtime.services()
    await annotate_dataset(config, 1, services=services)
    log = config.work_dir / "logs" / "run.jsonl"
    expected = log.read_text()
    log.unlink()
    await annotate_dataset(config, 1, services=services)
    assert log.read_text() == expected
    record = WorkspaceStore(config.work_dir).load_episode(0)
    payload = record.model_dump()
    events = payload["sampling_details"]["_pipeline_transition_events"]
    events[0]["category"] = "SECRET=api-key"
    with pytest.raises(ValidationError, match="transition"):
        EpisodeRecord.model_validate(payload)
    await annotate_dataset(config, 1, services=services)
    assert log.read_text() == expected


@pytest.mark.asyncio
async def test_trailing_partial_legacy_log_is_rebuilt_from_authoritative_outbox(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    services = FakeRuntime(dataset).services()
    await annotate_dataset(config, 1, services=services)
    log = config.work_dir / "logs" / "run.jsonl"
    expected = log.read_bytes()
    log.write_bytes(expected + b'{"partial":')
    await annotate_dataset(config, 1, services=services)
    assert log.read_bytes() == expected and expected.endswith(b"\n")


@pytest.mark.parametrize("failure", [OSError("short write"), KeyboardInterrupt()])
@pytest.mark.asyncio
async def test_partial_temp_write_never_corrupts_official_log_and_resume_repairs(
    monkeypatch, tmp_path: Path, failure: BaseException,
) -> None:
    import robo_annotate.pipeline as pipeline_module

    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    runtime = FakeRuntime(dataset)
    base = runtime.services()

    async def interrupt_refine(*args, **kwargs):
        raise KeyboardInterrupt()

    interrupted = PipelineServices(
        inspect_dataset=base.inspect_dataset, workspace_factory=base.workspace_factory,
        resolve_model=base.resolve_model, client_factory=base.client_factory,
        run_coarse=base.run_coarse, run_refine=interrupt_refine,
    )
    with pytest.raises(KeyboardInterrupt):
        await annotate_dataset(config, 1, services=interrupted)
    log = config.work_dir / "logs" / "run.jsonl"
    prior = log.read_bytes()
    original_write = pipeline_module.os.write
    write_calls = 0

    def partial_then_fail(fd, payload):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return original_write(fd, bytes(payload[: max(1, len(payload) // 2)]))
        raise failure

    monkeypatch.setattr(pipeline_module.os, "write", partial_then_fail)
    expected_error = AuditPersistenceError if isinstance(failure, OSError) else KeyboardInterrupt
    with pytest.raises(expected_error):
        await annotate_dataset(config, 1, services=base)
    assert WorkspaceStore(config.work_dir).load_episode(0).status == "refine_done"
    assert log.read_bytes() == prior and prior.endswith(b"\n")
    assert not list((config.work_dir / "logs").glob(".run.jsonl.*.tmp"))

    monkeypatch.setattr(pipeline_module.os, "write", original_write)
    assert (await annotate_dataset(config, 1, services=base)).accepted == 1
    lines = log.read_text().splitlines()
    assert len(lines) == 3 and all(line.endswith("}") for line in lines)


@pytest.mark.asyncio
async def test_atomic_replace_failure_keeps_prior_log_and_cleans_temp(monkeypatch, tmp_path: Path) -> None:
    import robo_annotate.pipeline as pipeline_module

    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    runtime = FakeRuntime(dataset)
    base = runtime.services()

    async def interrupt_refine(*args, **kwargs):
        raise KeyboardInterrupt()

    services = PipelineServices(
        inspect_dataset=base.inspect_dataset, workspace_factory=base.workspace_factory,
        resolve_model=base.resolve_model, client_factory=base.client_factory,
        run_coarse=base.run_coarse, run_refine=interrupt_refine,
    )
    with pytest.raises(KeyboardInterrupt):
        await annotate_dataset(config, 1, services=services)
    log = config.work_dir / "logs" / "run.jsonl"
    prior = log.read_bytes()
    original_replace = pipeline_module.os.replace

    def fail_log_replace(source, destination, **kwargs):
        if destination == "run.jsonl":
            raise OSError("replace failed")
        return original_replace(source, destination, **kwargs)

    monkeypatch.setattr(pipeline_module.os, "replace", fail_log_replace)
    with pytest.raises(AuditPersistenceError):
        await annotate_dataset(config, 1, services=base)
    assert log.read_bytes() == prior
    assert not list((config.work_dir / "logs").glob(".run.jsonl.*.tmp"))


@pytest.mark.asyncio
async def test_cross_process_log_synchronizers_produce_one_canonical_file(tmp_path: Path) -> None:
    import multiprocessing

    dataset = _dataset(tmp_path, 2)
    config = _config(tmp_path, dataset)
    await annotate_dataset(config, 2, services=FakeRuntime(dataset).services())
    log = config.work_dir / "logs" / "run.jsonl"
    expected = log.read_bytes()
    log.write_bytes(b"legacy partial")
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [context.Process(target=_sync_log_in_process, args=(str(config.work_dir), results)) for _ in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    assert [results.get(timeout=2) for _ in processes] == ["ok"] * 4
    assert log.read_bytes() == expected
    assert not list((config.work_dir / "logs").glob(".run.jsonl.*.tmp"))


@pytest.mark.asyncio
async def test_temp_fsync_failure_preserves_official_log_and_cleans_temp(monkeypatch, tmp_path: Path) -> None:
    import robo_annotate.pipeline as pipeline_module

    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    runtime = FakeRuntime(dataset)
    base = runtime.services()

    async def interrupt_refine(*args, **kwargs):
        raise KeyboardInterrupt()

    services = PipelineServices(
        inspect_dataset=base.inspect_dataset, workspace_factory=base.workspace_factory,
        resolve_model=base.resolve_model, client_factory=base.client_factory,
        run_coarse=base.run_coarse, run_refine=interrupt_refine,
    )
    with pytest.raises(KeyboardInterrupt):
        await annotate_dataset(config, 1, services=services)
    log = config.work_dir / "logs" / "run.jsonl"
    prior = log.read_bytes()
    original_fsync = pipeline_module.os.fsync

    def fail_temp_fsync(fd):
        target = Path(f"/proc/self/fd/{fd}").resolve()
        if target.name.startswith(".run.jsonl."):
            raise OSError("temp fsync failed")
        return original_fsync(fd)

    monkeypatch.setattr(pipeline_module.os, "fsync", fail_temp_fsync)
    with pytest.raises(AuditPersistenceError):
        await annotate_dataset(config, 1, services=base)
    assert WorkspaceStore(config.work_dir).load_episode(0).status == "refine_done"
    assert log.read_bytes() == prior
    assert not list((config.work_dir / "logs").glob(".run.jsonl.*.tmp"))


@pytest.mark.asyncio
async def test_log_failure_after_state_save_aborts_and_resume_repairs(monkeypatch, tmp_path: Path) -> None:
    import robo_annotate.pipeline as pipeline_module

    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    runtime = FakeRuntime(dataset)
    original = pipeline_module._RunLog.sync
    calls = 0

    async def fail_once(self, store):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("audit disk unavailable SECRET")
        return await original(self, store)

    monkeypatch.setattr(pipeline_module._RunLog, "sync", fail_once)
    with pytest.raises(AuditPersistenceError):
        await annotate_dataset(config, 1, services=runtime.services())
    record = WorkspaceStore(config.work_dir).load_episode(0)
    assert record.status == "coarse_done" and record.failure_category is None
    monkeypatch.setattr(pipeline_module._RunLog, "sync", original)
    assert (await annotate_dataset(config, 1, services=runtime.services())).accepted == 1
    lines = [__import__("json").loads(line) for line in (config.work_dir / "logs" / "run.jsonl").read_text().splitlines()]
    assert len(lines) == 3 and len({line["event_id"] for line in lines}) == 3
    assert "SECRET" not in (config.work_dir / "logs" / "run.jsonl").read_text()


@pytest.mark.asyncio
async def test_custom_base_exception_stops_scheduling_and_drains_inflight(tmp_path: Path) -> None:
    import asyncio

    class StopNow(BaseException):
        pass

    dataset = _dataset(tmp_path, 3)
    config = _config(tmp_path, dataset)
    base = FakeRuntime(dataset).services()
    second_started = asyncio.Event()
    second_drained = asyncio.Event()
    calls = []

    async def coarse(config, episode, **kwargs):
        calls.append(episode.episode_index)
        if episode.episode_index == 1:
            second_started.set()
            await asyncio.sleep(0.02)
            second_drained.set()
            return _coarse()
        await second_started.wait()
        raise StopNow()

    services = PipelineServices(
        inspect_dataset=base.inspect_dataset, workspace_factory=base.workspace_factory,
        resolve_model=base.resolve_model, client_factory=base.client_factory,
        run_coarse=coarse, run_refine=base.run_refine,
    )
    with pytest.raises(StopNow):
        await annotate_dataset(config, 2, services=services)
    assert second_drained.is_set() and calls == [0, 1]
    assert WorkspaceStore(config.work_dir).load_episode(2).status == "pending"


@pytest.mark.asyncio
async def test_pipeline_closes_opted_in_shared_client_without_masking_primary_error(tmp_path: Path) -> None:
    class StopNow(BaseException):
        pass

    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    base = FakeRuntime(dataset).services()
    close_calls = 0

    class Client:
        async def aclose(self):
            nonlocal close_calls
            close_calls += 1
            raise RuntimeError("close failure")

    async def stop(*args, **kwargs):
        raise StopNow()

    services = PipelineServices(
        inspect_dataset=base.inspect_dataset, workspace_factory=base.workspace_factory,
        resolve_model=base.resolve_model, client_factory=lambda config: Client(),
        run_coarse=stop, run_refine=base.run_refine, close_client=True,
    )
    with pytest.raises(StopNow):
        await annotate_dataset(config, 1, services=services)
    assert close_calls == 1


@pytest.mark.asyncio
async def test_internal_clock_value_error_aborts_without_mislabeling_episode(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 1)
    config = _config(tmp_path, dataset)
    base = FakeRuntime(dataset).services()
    services = PipelineServices(
        inspect_dataset=base.inspect_dataset, workspace_factory=base.workspace_factory,
        resolve_model=base.resolve_model, client_factory=base.client_factory,
        run_coarse=base.run_coarse, run_refine=base.run_refine,
        clock=lambda: datetime(2026, 8, 22),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        await annotate_dataset(config, 1, services=services)
    record = WorkspaceStore(config.work_dir).load_episode(0)
    assert record.status == "pending" and record.failure_category is None

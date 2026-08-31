import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from robo_annotate.coarse import CoarseDecision
from robo_annotate.config import AnnotationConfig
from robo_annotate.lerobot import EpisodeInfo
from robo_annotate.models import CoarseBoundary, CoarseResult, RefineResult
from robo_annotate.qwen_client import ModelCallError, ModelOutOfMemory
from robo_annotate.refine import (
    CameraSampling,
    RefineDecision,
    SamplingProvenance,
    choose_agreed_boundary,
    run_refine,
)
from robo_annotate.video import FrameSample
from robo_annotate.workspace import compute_source_fingerprint
from tests.fixtures import make_episode_info


def make_config(
    tmp_path: Path,
    *,
    mode: str = "complete",
    subtasks: int = 3,
    cameras: tuple[str, ...] = ("cam.eye", "cam.wrist"),
    min_segment_frames: int = 2,
) -> AnnotationConfig:
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True, exist_ok=True)
    (source / "meta" / "info.json").write_text(json.dumps({"fps": 10.0}), encoding="utf-8")
    return AnnotationConfig.model_validate(
        {
            "source": source,
            "work_dir": tmp_path / "work",
            "mode": mode,
            "high_level_instruction": "arrange objects",
            "primary_camera": cameras[0],
            "refine_cameras": [cameras[0], *cameras[1:], cameras[0]],
            "subtasks": [
                {"skill": "move", "text": f"step {index}"} for index in range(subtasks)
            ],
            "sampling": {
                "refine_window_seconds": 1.0,
                "refine_fps": 2.0,
                "dense_radius_seconds": 0.2,
                "agreement_tolerance_frames": 2,
                "min_segment_frames": min_segment_frames,
            },
        }
    )


def make_episode(tmp_path: Path, *, length: int = 41, cameras=("cam.eye", "cam.wrist")) -> EpisodeInfo:
    source = tmp_path / "source"
    parquet = source / "data" / "episode.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.touch()
    videos = {}
    for camera in cameras:
        path = source / "videos" / f"{camera}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        videos[camera] = path
    return make_episode_info(
        episode_index=7,
        length=length,
        task="arrange",
        parquet=parquet,
        videos=videos,
        fps=10.0,
    )


def coarse_result(observed, centers):
    return CoarseResult(
        start_subtask_index=observed[0],
        observed_subtask_indices=observed,
        coarse_boundaries=[
            CoarseBoundary(
                from_subtask_index=left,
                to_subtask_index=right,
                estimated_frame=center,
                evidence="transition",
            )
            for left, right, center in zip(observed, observed[1:], centers)
        ],
        confidence=0.9,
        semantic_uncertainty_codes=[],
        boundary_precision_notes=[],
    )


def coarse_decision(mode="complete", observed=(0, 1, 2), centers=(10, 30), frame_count=41, subtask_count=3):
    attempt = coarse_result(list(observed), list(centers))
    return CoarseDecision(
        mode=mode,
        subtask_count=subtask_count,
        frame_count=frame_count,
        status="coarse_done",
        attempts=(attempt, attempt),
        reasons=(),
        start_subtask_index=observed[0],
        observed_subtask_indices=observed,
        boundary_centers=centers,
        sampled_frame_indices=((0, frame_count - 1), (0, frame_count - 1)),
    )


def refined(frame: int, left: int = 0) -> RefineResult:
    return RefineResult(
        from_subtask_index=left,
        to_subtask_index=left + 1,
        last_frame_before=frame - 1,
        first_frame_after=frame,
        boundary_frame=frame,
        confidence=0.9,
        visible_cues=["release complete"],
    )


class RecordingSampler:
    def __init__(self, mutate=None):
        self.calls = []
        self.mutate = mutate

    def __call__(self, video, camera, indices):
        self.calls.append((video, camera, list(indices)))
        if self.mutate:
            self.mutate(len(self.calls))
        return [
            FrameSample(
                camera_key=camera,
                frame_index=index,
                timestamp_seconds=index / video.fps,
                jpeg=b"jpeg",
            )
            for index in indices
        ]


class RecordingClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def complete(self, prompt, frames, response_type):
        self.calls.append((prompt, list(frames), response_type))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


def test_choose_agreed_boundary_is_strict_and_half_up() -> None:
    assert choose_agreed_boundary([refined(100), refined(105)], 5) == 103
    assert choose_agreed_boundary([refined(100), refined(106)], 5) is None
    with pytest.raises(ValueError, match="exactly two"):
        choose_agreed_boundary([], 2)
    with pytest.raises(TypeError):
        choose_agreed_boundary([refined(2), refined(3)], True)
    with pytest.raises(TypeError):
        choose_agreed_boundary([refined(2), object()], 2)
    bypassed = RefineResult.model_construct(
        from_subtask_index=0, to_subtask_index=1, last_frame_before=0,
        first_frame_after=1, boundary_frame=True, confidence=0.9,
        visible_cues=["cue"],
    )
    with pytest.raises(TypeError, match="boundary_frame"):
        choose_agreed_boundary([bypassed, refined(2)], 2)


@pytest.mark.asyncio
async def test_two_stage_sampling_order_prompts_and_multiple_boundary_acceptance(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    episode = make_episode(tmp_path)
    sampler = RecordingSampler()
    client = RecordingClient([refined(11, 0), refined(12, 0), refined(29, 1), refined(30, 1)])

    decision = await run_refine(config, episode, coarse_decision(), sampler, client)

    assert decision.status == "accepted"
    assert decision.annotation.start_subtask_index == 0
    assert decision.annotation.boundaries == (12, 30)
    assert decision.reasons == ()
    # broad radius=round(1*10)=10, stride=round(10/2)=5, endpoints and center included
    assert [call[2] for call in sampler.calls[:2]] == [[0, 5, 10, 15, 20], [0, 5, 10, 15, 20]]
    # Dense is centered on broad result 11, ±round(.2*10)=2, every original frame.
    assert [call[2] for call in sampler.calls[2:4]] == [[9, 10, 11, 12, 13], [9, 10, 11, 12, 13]]
    first_frames = client.calls[0][1]
    assert [(x.frame_index, x.camera_key) for x in first_frames] == [
        (index, camera)
        for index in [0, 5, 10, 15, 20]
        for camera in ["cam.eye", "cam.wrist"]
    ]
    contexts = [
        json.loads(call[0].split("BEGIN_UNTRUSTED_CONTEXT_JSON\n", 1)[1].split("\nEND_UNTRUSTED_CONTEXT_JSON", 1)[0])
        for call in client.calls
    ]
    assert [x["pass_id"] for x in contexts] == [0, 2, 3, 5]
    assert all(call[2] is RefineResult for call in client.calls)
    assert "from_subtask_index=0, to_subtask_index=1" in client.calls[0][0]
    assert "coarse center frame is 11" in client.calls[1][0]
    restored = RefineDecision.model_validate_json(decision.model_dump_json())
    assert restored == decision
    assert RefineDecision.model_validate(decision.model_dump()) == decision
    assert isinstance(restored.observed_subtask_indices, tuple)
    assert isinstance(restored.provenance[0].cameras, tuple)
    assert isinstance(restored.provenance[0].samples, tuple)
    assert isinstance(restored.provenance[0].samples[0].frame_indices, tuple)
    dumped = restored.model_dump()
    dumped["provenance"][0]["samples"][0]["frame_indices"] += (999,)
    assert 999 not in restored.provenance[0].samples[0].frame_indices
    with pytest.raises(ValidationError):
        restored.provenance[0].samples[0].frame_indices += (999,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "observed", "start"),
    [("complete", (0,), 0), ("dagger_patch", (1,), 1), ("dagger_patch", (2,), 2)],
)
async def test_zero_boundary_short_circuit(mode, observed, start, tmp_path: Path) -> None:
    config = make_config(tmp_path, mode=mode, subtasks=1 if mode == "complete" else 3)
    episode = make_episode(tmp_path)
    if mode == "complete":
        coarse = coarse_decision(mode, observed, (), 41, 1)
    else:
        coarse = coarse_decision(mode, observed, (), 41)
    sampler = RecordingSampler()
    client = RecordingClient([])

    decision = await run_refine(config, episode, coarse, sampler, client)

    assert decision.status == "accepted"
    assert decision.annotation.start_subtask_index == start
    assert decision.annotation.boundaries == ()
    assert sampler.calls == [] and client.calls == []


@pytest.mark.asyncio
async def test_disagreement_and_multicamera_conflict_preserve_audit(tmp_path: Path) -> None:
    decision = await run_refine(
        make_config(tmp_path), make_episode(tmp_path), coarse_decision(), RecordingSampler(),
        RecordingClient([refined(8, 0), refined(14, 0), refined(29, 1), refined(30, 1)]),
    )
    assert decision.status == "needs_review"
    assert decision.reasons == ("refine_boundary_disagreement",)
    assert decision.annotation is None and decision.candidate_annotation is None
    assert len(decision.attempts) == 4
    assert RefineDecision.model_validate_json(decision.model_dump_json()) == decision
    assert RefineDecision.model_validate(decision.model_dump()) == decision


@pytest.mark.asyncio
async def test_transition_mismatch_is_review_not_schema_bypass(tmp_path: Path) -> None:
    client = RecordingClient([refined(10, 1), refined(11, 0), refined(29, 1), refined(30, 1)])
    decision = await run_refine(make_config(tmp_path), make_episode(tmp_path), coarse_decision(), RecordingSampler(), client)
    assert decision.status == "needs_review"
    assert decision.reasons == ("refine_transition_mismatch",)
    assert decision.candidate_annotation is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responses",
    [
        [refined(41, 0)],
        [refined(10, 0), refined(41, 0), refined(30, 1), refined(30, 1)],
        [
            RefineResult.model_construct(
                from_subtask_index=2, to_subtask_index=0, last_frame_before=-1,
                first_frame_after=0, boundary_frame=0, confidence=0.5,
                visible_cues=["defensive malformed output"],
            )
        ],
    ],
)
async def test_dynamic_out_of_range_or_defensive_attempt_is_quarantined(
    responses, tmp_path: Path
) -> None:
    decision = await run_refine(
        make_config(tmp_path), make_episode(tmp_path), coarse_decision(),
        RecordingSampler(), RecordingClient(responses),
    )
    assert decision.status == "needs_review"
    assert decision.reasons == ("refine_transition_mismatch",)
    assert decision.annotation is None and decision.candidate_annotation is None
    assert RefineDecision.model_validate_json(decision.model_dump_json()) == decision
    assert RefineDecision.model_validate(decision.model_dump()) == decision


@pytest.mark.asyncio
async def test_final_validation_issues_are_stable_and_candidate_is_complete(tmp_path: Path) -> None:
    config = make_config(tmp_path, min_segment_frames=15)
    decision = await run_refine(
        config, make_episode(tmp_path), coarse_decision(), RecordingSampler(),
        RecordingClient([refined(10, 0), refined(11, 0), refined(29, 1), refined(30, 1)]),
    )
    assert decision.status == "needs_review"
    assert decision.reasons == ("segment_too_short",)
    assert decision.candidate_annotation.boundaries == (11, 30)
    assert decision.annotation is None


def oom() -> ModelOutOfMemory:
    return ModelOutOfMemory("oom", attempt_count=1)


@pytest.mark.asyncio
async def test_first_multicamera_oom_degrades_once_and_keeps_primary(tmp_path: Path) -> None:
    sampler = RecordingSampler()
    client = RecordingClient([oom(), refined(10, 0), refined(11, 0), refined(30, 1), refined(30, 1)])
    decision = await run_refine(make_config(tmp_path), make_episode(tmp_path), coarse_decision(), sampler, client)
    assert decision.status == "accepted"
    assert [call[1] for call in sampler.calls[:3]] == ["cam.eye", "cam.wrist", "cam.eye"]
    assert sampler.calls[2][2] == [0, 10, 20]
    assert all(sample.camera_key == "cam.eye" for call in client.calls[1:] for sample in call[1])
    assert len(client.calls) == 5
    contexts = [json.loads(x[0].split("BEGIN_UNTRUSTED_CONTEXT_JSON\n", 1)[1].split("\nEND_UNTRUSTED_CONTEXT_JSON", 1)[0]) for x in client.calls]
    assert [x["pass_id"] for x in contexts] == [0, 1, 2, 3, 5]


@pytest.mark.asyncio
async def test_second_oom_fails_without_partial_annotation(tmp_path: Path) -> None:
    decision = await run_refine(
        make_config(tmp_path), make_episode(tmp_path), coarse_decision(), RecordingSampler(),
        RecordingClient([oom(), oom()]),
    )
    assert decision.status == "failed"
    assert decision.failure_category == "model_oom"
    assert decision.reasons == ("model_oom",)
    assert decision.annotation is None and decision.candidate_annotation is None
    assert [item.outcome for item in decision.provenance] == ["model_oom", "model_oom"]
    assert RefineDecision.model_validate_json(decision.model_dump_json()) == decision
    assert RefineDecision.model_validate(decision.model_dump()) == decision


@pytest.mark.asyncio
async def test_later_dense_oom_fails_immediately_without_degradation_retry(tmp_path: Path) -> None:
    client = RecordingClient([refined(10, 0), oom()])
    decision = await run_refine(
        make_config(tmp_path), make_episode(tmp_path), coarse_decision(), RecordingSampler(), client,
    )
    assert decision.status == "failed"
    assert len(client.calls) == 2
    assert [item.stage for item in decision.provenance] == ["broad", "dense"]
    assert [item.outcome for item in decision.provenance] == ["completed", "model_oom"]
    assert len(decision.attempts) == 1


@pytest.mark.asyncio
async def test_unrelated_model_errors_propagate(tmp_path: Path) -> None:
    with pytest.raises(ModelCallError):
        await run_refine(
            make_config(tmp_path), make_episode(tmp_path), coarse_decision(), RecordingSampler(),
            RecordingClient([ModelCallError("bad", attempt_count=1)]),
        )


@pytest.mark.asyncio
async def test_sampler_runs_off_event_loop(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    loop = asyncio.get_running_loop()

    class BlockingSampler(RecordingSampler):
        def __call__(self, *args):
            loop.call_soon_threadsafe(started.set)
            import time
            time.sleep(0.05)
            return super().__call__(*args)

    task = asyncio.create_task(run_refine(
        make_config(tmp_path), make_episode(tmp_path), coarse_decision(), BlockingSampler(),
        RecordingClient([refined(10, 0), refined(10, 0), refined(30, 1), refined(30, 1)]),
    ))
    await asyncio.wait_for(started.wait(), 0.5)
    await asyncio.wait_for(asyncio.sleep(0), 0.1)
    await task


def test_refine_decision_is_deeply_immutable_and_roundtrips(tmp_path: Path) -> None:
    payload = {
        "mode": "complete", "subtask_count": 1, "frame_count": 20, "min_segment_frames": 2,
        "agreement_tolerance_frames": 2,
        "start_subtask_index": 0, "observed_subtask_indices": (0,),
        "coarse_boundary_centers": (), "source_fps": 10.0,
        "refine_window_seconds": 1.0, "refine_fps": 2.0, "dense_radius_seconds": 0.2,
        "camera_order": ("cam.eye", "cam.wrist"), "broad_radius_frames": 10,
        "base_broad_stride": 5, "dense_radius_frames": 2,
        "status": "accepted", "attempts": (), "provenance": (), "reasons": (),
        "annotation": {"start_subtask_index": 0, "boundaries": []},
        "candidate_annotation": None, "failure_category": None,
    }
    decision = RefineDecision.model_validate(payload)
    payload["annotation"]["boundaries"].append(2)
    assert decision.annotation.boundaries == ()
    with pytest.raises((AttributeError, TypeError, ValidationError)):
        decision.annotation.boundaries += (2,)
    restored = RefineDecision.model_validate_json(decision.model_dump_json())
    assert restored == decision
    with pytest.raises(ValidationError):
        RefineDecision.model_validate({**payload, "status": "failed", "annotation": None, "failure_category": None, "reasons": []})


def test_refine_decision_freezes_attempts_and_rejects_inconsistent_audit() -> None:
    attempt = refined(10)
    provenance = {
        "boundary_index": 0, "from_subtask_index": 0, "to_subtask_index": 1,
        "stage": "broad", "pass_id": 0, "request_center": 10,
        "radius_frames": 2, "stride": 1, "cameras": ("cam.eye",),
        "samples": ({"camera_key": "cam.eye", "frame_indices": (8, 9, 10, 11, 12)},),
        "outcome": "completed",
    }
    payload = {
        "mode": "complete", "subtask_count": 2, "frame_count": 20,
        "min_segment_frames": 2, "agreement_tolerance_frames": 2,
        "start_subtask_index": 0, "observed_subtask_indices": (0, 1),
        "coarse_boundary_centers": (10,), "source_fps": 10.0,
        "refine_window_seconds": 0.2, "refine_fps": 10.0, "dense_radius_seconds": 0.2,
        "camera_order": ("cam.eye",), "broad_radius_frames": 2,
        "base_broad_stride": 1, "dense_radius_frames": 2,
        "status": "needs_review", "attempts": (attempt, refined(15)),
        "provenance": (provenance, {**provenance, "stage": "dense", "pass_id": 2}),
        "reasons": ("refine_boundary_disagreement",),
        "annotation": None, "candidate_annotation": None, "failure_category": None,
    }
    decision = RefineDecision.model_validate(payload)
    attempt.visible_cues.append("caller mutation")
    dumped = decision.model_dump()
    dumped["attempts"][0]["visible_cues"] += ("dump mutation",)
    assert decision.attempts[0].visible_cues == ("release complete",)
    with pytest.raises(ValidationError, match="completed evidence"):
        RefineDecision.model_validate({**payload, "attempts": ()})
    with pytest.raises(ValidationError, match="expected boundary"):
        RefineDecision.model_validate({
            **payload, "status": "accepted", "attempts": (), "provenance": (),
            "reasons": (), "annotation": {"start_subtask_index": 0, "boundaries": [1]},
            "candidate_annotation": None, "min_segment_frames": 2,
        })


def test_python_lists_are_rejected_for_strict_audit_tuples() -> None:
    base = {
        "mode": "complete", "subtask_count": 1, "frame_count": 20,
        "min_segment_frames": 2, "agreement_tolerance_frames": 2,
        "start_subtask_index": 0, "observed_subtask_indices": (0,),
        "coarse_boundary_centers": (), "source_fps": 10.0,
        "refine_window_seconds": 0.2, "refine_fps": 10.0, "dense_radius_seconds": 0.1,
        "camera_order": ("cam.eye",), "broad_radius_frames": 2,
        "base_broad_stride": 1, "dense_radius_frames": 1,
        "status": "accepted", "attempts": (), "provenance": (), "reasons": (),
        "annotation": {"start_subtask_index": 0, "boundaries": []},
        "candidate_annotation": None, "failure_category": None,
    }
    for field in (
        "observed_subtask_indices", "coarse_boundary_centers", "camera_order",
        "attempts", "provenance", "reasons",
    ):
        with pytest.raises(ValidationError):
            RefineDecision.model_validate({**base, field: list(base[field])})
    with pytest.raises(ValidationError):
        CameraSampling(camera_key="cam.eye", frame_indices=[1, 2])
    camera = CameraSampling.model_validate_json('{"camera_key":"cam.eye","frame_indices":[1,2]}')
    assert camera.frame_indices == (1, 2)
    provenance = {
        "boundary_index": 0, "from_subtask_index": 0, "to_subtask_index": 1,
        "stage": "broad", "pass_id": 0, "request_center": 2,
        "radius_frames": 1, "stride": 1, "cameras": ["cam.eye"],
        "samples": [{"camera_key": "cam.eye", "frame_indices": [1, 2]}],
        "outcome": "completed",
    }
    with pytest.raises(ValidationError):
        SamplingProvenance.model_validate(provenance)
    restored = SamplingProvenance.model_validate_json(json.dumps(provenance))
    assert restored.cameras == ("cam.eye",) and restored.samples == (camera,)


def test_decision_rejects_incomplete_or_masquerading_coarse_mapping() -> None:
    base = {
        "mode": "dagger_patch", "subtask_count": 4, "frame_count": 40,
        "min_segment_frames": 2, "agreement_tolerance_frames": 2,
        "start_subtask_index": 1, "status": "accepted", "attempts": (),
        "coarse_boundary_centers": (), "source_fps": 10.0,
        "refine_window_seconds": 0.2, "refine_fps": 10.0, "dense_radius_seconds": 0.1,
        "camera_order": ("cam.eye",), "broad_radius_frames": 2,
        "base_broad_stride": 1, "dense_radius_frames": 1,
        "provenance": (), "reasons": (), "annotation": {"start_subtask_index": 1, "boundaries": []},
        "candidate_annotation": None, "failure_category": None,
    }
    # [1] is a legal explicit singleton, while [1,2,3] requires both refinements.
    singleton = RefineDecision.model_validate({**base, "observed_subtask_indices": (1,)})
    assert singleton.annotation.boundaries == ()
    with pytest.raises(ValidationError, match="expected boundary"):
        RefineDecision.model_validate({
            **base, "observed_subtask_indices": (1, 2, 3),
            "coarse_boundary_centers": (10, 30),
        })
    with pytest.raises(ValidationError, match="legal coarse sequence"):
        RefineDecision.model_validate({**base, "observed_subtask_indices": (1, 2)})
    with pytest.raises(ValidationError, match="start"):
        RefineDecision.model_validate({**base, "start_subtask_index": 2, "observed_subtask_indices": (1,)})
    with pytest.raises(ValidationError, match="terminal OOM"):
        RefineDecision.model_validate({
            **base, "observed_subtask_indices": (1,), "status": "failed",
            "annotation": None, "reasons": ("model_oom",), "failure_category": "model_oom",
        })


@pytest.mark.asyncio
async def test_dagger_suffix_records_and_replays_full_observed_mapping(tmp_path: Path) -> None:
    config = make_config(tmp_path, mode="dagger_patch", subtasks=4)
    coarse = coarse_decision(
        "dagger_patch", observed=(1, 2, 3), centers=(10, 30), subtask_count=4
    )
    decision = await run_refine(
        config, make_episode(tmp_path), coarse, RecordingSampler(),
        RecordingClient([refined(10, 1), refined(10, 1), refined(30, 2), refined(30, 2)]),
    )
    assert decision.status == "accepted"
    assert decision.start_subtask_index == 1
    assert decision.observed_subtask_indices == (1, 2, 3)
    assert decision.annotation.boundaries == (10, 30)
    assert RefineDecision.model_validate_json(decision.model_dump_json()) == decision


@pytest.mark.asyncio
async def test_oom_degradation_stride_persists_across_later_boundaries(tmp_path: Path) -> None:
    sampler = RecordingSampler()
    client = RecordingClient([oom(), refined(10, 0), refined(10, 0), refined(30, 1), refined(30, 1)])
    decision = await run_refine(make_config(tmp_path), make_episode(tmp_path), coarse_decision(), sampler, client)
    assert decision.status == "accepted"
    # Boundary 1 broad is still primary-only and uses doubled stride (10), not base stride (5).
    later_broad = [call for call in sampler.calls if call[2] == [20, 30, 40]]
    assert [(call[1], call[2]) for call in later_broad] == [("cam.eye", [20, 30, 40])]


@pytest.mark.asyncio
async def test_any_oom_after_degradation_fails_without_another_retry(tmp_path: Path) -> None:
    client = RecordingClient([oom(), refined(10, 0), refined(10, 0), oom()])
    decision = await run_refine(make_config(tmp_path), make_episode(tmp_path), coarse_decision(), RecordingSampler(), client)
    assert decision.status == "failed"
    assert len(client.calls) == 4
    assert [item.stage for item in decision.provenance] == ["broad", "broad_retry", "dense", "broad"]


@pytest.mark.asyncio
async def test_failed_oom_preserves_prior_review_reasons(tmp_path: Path) -> None:
    client = RecordingClient([refined(5, 0), refined(10, 0), oom(), oom()])
    decision = await run_refine(
        make_config(tmp_path), make_episode(tmp_path), coarse_decision(), RecordingSampler(), client,
    )
    assert decision.status == "failed"
    assert decision.reasons == ("refine_boundary_disagreement", "model_oom")
    assert decision.annotation is None and decision.candidate_annotation is None
    assert RefineDecision.model_validate_json(decision.model_dump_json()) == decision


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda data: data["provenance"][0].__setitem__("request_center", 9), "center"),
        (lambda data: data.__setitem__("coarse_boundary_centers", [9, 30]), "center"),
        (lambda data: data.__setitem__("source_fps", 11.0), "recomputable"),
        (lambda data: data["provenance"][0].__setitem__("radius_frames", 9), "radius"),
        (lambda data: data["provenance"][0].__setitem__("stride", 9), "stride"),
        (lambda data: data["provenance"][0].__setitem__("cameras", ["cam.eye"]), "cameras"),
        (lambda data: data["provenance"][0]["samples"][0].__setitem__("frame_indices", [0, 10, 20]), "grid"),
        (lambda data: data["provenance"].__setitem__(slice(0, 2), list(reversed(data["provenance"][:2]))), "order|stages"),
        (lambda data: data["provenance"].__delitem__(slice(2, None)), "completed evidence|every expected boundary"),
    ],
)
async def test_provenance_tampering_is_rejected_for_accepted_decision(
    mutate, match: str, tmp_path: Path
) -> None:
    decision = await run_refine(
        make_config(tmp_path), make_episode(tmp_path), coarse_decision(), RecordingSampler(),
        RecordingClient([refined(10, 0), refined(10, 0), refined(30, 1), refined(30, 1)]),
    )
    payload = json.loads(decision.model_dump_json())
    mutate(payload)
    with pytest.raises(ValidationError, match=match):
        RefineDecision.model_validate_json(json.dumps(payload))


@pytest.mark.asyncio
async def test_provenance_tampering_is_rejected_for_review_and_failed_decisions(tmp_path: Path) -> None:
    review = await run_refine(
        make_config(tmp_path), make_episode(tmp_path), coarse_decision(), RecordingSampler(),
        RecordingClient([refined(5, 0), refined(10, 0), refined(30, 1), refined(30, 1)]),
    )
    failed = await run_refine(
        make_config(tmp_path / "failed"), make_episode(tmp_path / "failed"), coarse_decision(),
        RecordingSampler(), RecordingClient([oom(), oom()]),
    )
    for decision in (review, failed):
        payload = json.loads(decision.model_dump_json())
        payload["provenance"][0]["request_center"] += 1
        with pytest.raises(ValidationError, match="center"):
            RefineDecision.model_validate_json(json.dumps(payload))


@pytest.mark.asyncio
async def test_rejects_quarantined_or_mismatched_coarse_context(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    episode = make_episode(tmp_path)
    coarse = coarse_decision()
    with pytest.raises(ValueError, match="coarse_done"):
        await run_refine(config, episode, coarse.model_copy(update={"status": "needs_review"}), RecordingSampler(), RecordingClient([]))
    with pytest.raises(ValueError, match="frame_count"):
        await run_refine(config, episode, coarse.model_copy(update={"frame_count": 40}), RecordingSampler(), RecordingClient([]))


@pytest.mark.asyncio
async def test_source_fps_camera_and_change_integrity(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    episode = make_episode(tmp_path)
    with pytest.raises(ValueError, match="authoritative"):
        await run_refine(config, episode, coarse_decision(), RecordingSampler(), RecordingClient([]), source_fps=11.0)
    missing = episode.model_copy(update={"videos": {"cam.eye": episode.videos["cam.eye"]}})
    with pytest.raises(ValueError, match="cam.wrist"):
        await run_refine(config, missing, coarse_decision(), RecordingSampler(), RecordingClient([]))

    def mutate(call_count):
        if call_count == 1:
            (config.source / "meta" / "info.json").write_text(json.dumps({"fps": 11.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        await run_refine(config, episode, coarse_decision(), RecordingSampler(mutate), RecordingClient([]))


@pytest.mark.asyncio
async def test_expected_fingerprint_duplicate_metadata_and_symlink_are_rejected(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    episode = make_episode(tmp_path)
    fingerprint = compute_source_fingerprint(config.source.resolve(), episode)
    with pytest.raises(ValueError, match="fingerprint"):
        await run_refine(
            config, episode, coarse_decision(), RecordingSampler(), RecordingClient([]),
            expected_source_fingerprint="0" * 64,
        )
    accepted = await run_refine(
        make_config(tmp_path, subtasks=1), episode,
        coarse_decision(observed=(0,), centers=(), subtask_count=1),
        RecordingSampler(), RecordingClient([]), expected_source_fingerprint=fingerprint,
    )
    assert accepted.status == "accepted"

    (config.source / "meta" / "info.json").write_text('{"fps":10,"fps":10}', encoding="utf-8")
    with pytest.raises(ValueError, match="unique keys"):
        await run_refine(config, episode, coarse_decision(), RecordingSampler(), RecordingClient([]))

    (config.source / "meta" / "info.json").write_text('{"fps":10}', encoding="utf-8")
    target = episode.videos["cam.wrist"].path
    target.unlink()
    real = target.with_suffix(".real.mp4")
    real.touch()
    target.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        await run_refine(config, episode, coarse_decision(), RecordingSampler(), RecordingClient([]))


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["order", "camera", "timestamp", "duplicate"])
async def test_invalid_sampler_evidence_is_rejected(corruption: str, tmp_path: Path) -> None:
    class BadSampler(RecordingSampler):
        def __call__(self, video, camera, indices):
            samples = super().__call__(video, camera, indices)
            if corruption == "order":
                return list(reversed(samples))
            if corruption == "camera":
                return [samples[0].model_copy(update={"camera_key": "wrong"}), *samples[1:]]
            if corruption == "timestamp":
                return [samples[0].model_copy(update={"timestamp_seconds": 99.0}), *samples[1:]]
            return [samples[0], samples[0], *samples[2:]]

    with pytest.raises(ValueError, match="sampler"):
        await run_refine(
            make_config(tmp_path), make_episode(tmp_path), coarse_decision(), BadSampler(), RecordingClient([]),
        )


@pytest.mark.asyncio
async def test_window_clipping_keeps_endpoint_center_and_dense_original_frames(tmp_path: Path) -> None:
    sampler = RecordingSampler()
    coarse = coarse_decision(centers=(2, 38))
    await run_refine(
        make_config(tmp_path), make_episode(tmp_path), coarse, sampler,
        RecordingClient([refined(2, 0), refined(2, 0), refined(38, 1), refined(38, 1)]),
    )
    assert sampler.calls[0][2] == [0, 2, 5, 10, 12]
    assert sampler.calls[2][2] == [0, 1, 2, 3, 4]
    assert sampler.calls[4][2] == [28, 33, 38, 40]
    assert sampler.calls[6][2] == [36, 37, 38, 39, 40]


@pytest.mark.asyncio
async def test_source_tree_is_never_written(tmp_path: Path) -> None:
    config = make_config(tmp_path, subtasks=1)
    episode = make_episode(tmp_path)
    before = sorted((p.relative_to(config.source), p.read_bytes()) for p in config.source.rglob("*") if p.is_file())
    await run_refine(config, episode, coarse_decision(observed=(0,), centers=(), subtask_count=1), RecordingSampler(), RecordingClient([]))
    after = sorted((p.relative_to(config.source), p.read_bytes()) for p in config.source.rglob("*") if p.is_file())
    assert after == before

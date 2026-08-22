import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from qwen_annotate.coarse import CoarseDecision, coarse_pass_indices, run_coarse
from qwen_annotate.config import AnnotationConfig
from qwen_annotate.lerobot import EpisodeInfo
from qwen_annotate.models import CoarseBoundary, CoarseResult
from qwen_annotate.video import FrameSample


def make_config(tmp_path: Path, *, mode: str = "complete", subtask_count: int = 3) -> AnnotationConfig:
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True, exist_ok=True)
    (source / "meta" / "info.json").write_text(json.dumps({"fps": 20.0}), encoding="utf-8")
    return AnnotationConfig.model_validate(
        {
            "source": source,
            "work_dir": tmp_path / "work",
            "mode": mode,
            "high_level_instruction": "arrange objects",
            "primary_camera": "cam.eye",
            "refine_cameras": ["cam.eye"],
            "subtasks": [
                {"skill": "move", "text": f"perform step {index}"}
                for index in range(subtask_count)
            ],
            "sampling": {"coarse_fps": 2.0, "coarse_max_frames": 8},
        }
    )


def make_episode(tmp_path: Path, *, length: int = 101, camera: str = "cam.eye") -> EpisodeInfo:
    return EpisodeInfo(
        episode_index=7,
        length=length,
        task="arrange objects",
        parquet=tmp_path / "episode.parquet",
        videos={camera: tmp_path / f"{camera}.mp4"},
    )


def boundary(left: int, frame: int) -> CoarseBoundary:
    return CoarseBoundary(
        from_subtask_index=left,
        to_subtask_index=left + 1,
        estimated_frame=frame,
        evidence="visible transition",
    )


def result(
    observed: list[int],
    frames: list[int],
    *,
    start: int | None = None,
    uncertainties: list[str] | None = None,
) -> CoarseResult:
    return CoarseResult(
        start_subtask_index=observed[0] if start is None else start,
        observed_subtask_indices=observed,
        coarse_boundaries=[boundary(left, frame) for left, frame in zip(observed, frames)],
        confidence=0.9,
        uncertainties=[] if uncertainties is None else uncertainties,
    )


class RecordingSampler:
    def __init__(self, *, corrupt=None):
        self.calls = []
        self.corrupt = corrupt

    def __call__(self, video_path, camera_key, indices, fps):
        self.calls.append((video_path, camera_key, list(indices), fps))
        samples = [
            FrameSample(
                camera_key=camera_key,
                frame_index=index,
                timestamp_seconds=index / fps,
                jpeg=b"jpeg",
            )
            for index in indices
        ]
        return samples if self.corrupt is None else self.corrupt(samples)


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


@pytest.mark.asyncio
async def test_complete_agreement_returns_half_up_averaged_boundaries(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    episode = make_episode(tmp_path)
    sampler = RecordingSampler()
    client = RecordingClient([result([0, 1, 2], [20, 60]), result([0, 1, 2], [21, 63])])

    decision = await run_coarse(config, episode, sampler, client)

    assert decision.status == "coarse_done"
    assert decision.start_subtask_index == 0
    assert decision.observed_subtask_indices == (0, 1, 2)
    assert decision.boundary_centers == (21, 62)
    assert decision.reasons == ()
    assert decision.attempts[0].coarse_boundaries[0].estimated_frame == 20
    assert len(client.calls) == 2
    contexts = [
        json.loads(call[0].split("BEGIN_UNTRUSTED_CONTEXT_JSON\n", 1)[1].split("\nEND_UNTRUSTED_CONTEXT_JSON", 1)[0])
        for call in client.calls
    ]
    assert [context["pass_id"] for context in contexts] == [0, 1]
    assert all(context["episode_index"] == 7 and context["frame_count"] == 101 for context in contexts)
    assert all(call[2] is CoarseResult for call in client.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("observed", "frames", "expected_start", "expected_centers"),
    [
        ([1, 2, 3], [30, 70], 1, (31, 71)),
        ([2], [], 2, ()),
        ([3], [], 3, ()),
    ],
)
async def test_dagger_legal_suffix_and_singletons_succeed(
    tmp_path: Path,
    observed: list[int],
    frames: list[int],
    expected_start: int,
    expected_centers: tuple[int, ...],
) -> None:
    config = make_config(tmp_path, mode="dagger_patch", subtask_count=4)
    attempts = [result(observed, frames), result(observed, [frame + 1 for frame in frames])]

    decision = await run_coarse(config, make_episode(tmp_path), RecordingSampler(), RecordingClient(attempts))

    assert decision.status == "coarse_done"
    assert decision.start_subtask_index == expected_start
    assert decision.boundary_centers == expected_centers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "observed"),
    [
        ("complete", [0, 1]),
        ("dagger_patch", [0, 2]),
        ("dagger_patch", [1, 2]),
    ],
)
async def test_illegal_mode_sequence_requires_review(tmp_path: Path, mode: str, observed: list[int]) -> None:
    config = make_config(tmp_path, mode=mode, subtask_count=4 if mode == "dagger_patch" else 3)
    frames = list(range(20, 20 + 20 * (len(observed) - 1), 20))
    if observed == [0, 2]:
        attempt = bypass_result(
            observed=observed,
            boundaries=[
                CoarseBoundary.model_construct(
                    from_subtask_index=0,
                    to_subtask_index=2,
                    estimated_frame=20,
                    evidence="visible transition",
                )
            ],
        )
    else:
        attempt = result(observed, frames)

    decision = await run_coarse(config, make_episode(tmp_path), RecordingSampler(), RecordingClient([attempt, attempt]))

    assert decision.status == "needs_review"
    assert "illegal_coarse_sequence" in decision.reasons
    assert decision.start_subtask_index is None
    assert decision.observed_subtask_indices == ()
    assert decision.boundary_centers == ()


@pytest.mark.asyncio
async def test_sequence_or_start_disagreement_has_stable_reason(tmp_path: Path) -> None:
    config = make_config(tmp_path, mode="dagger_patch", subtask_count=4)
    first = result([1, 2, 3], [30, 70])
    second = result([2, 3], [60])

    decision = await run_coarse(config, make_episode(tmp_path), RecordingSampler(), RecordingClient([first, second]))

    assert decision.reasons == ("coarse_sequence_disagreement",)


def bypass_result(*, start=0, observed=None, boundaries=None, uncertainties=None):
    return CoarseResult.model_construct(
        start_subtask_index=start,
        observed_subtask_indices=[0, 1, 2] if observed is None else observed,
        coarse_boundaries=[boundary(0, 20), boundary(1, 60)] if boundaries is None else boundaries,
        confidence=0.9,
        uncertainties=[] if uncertainties is None else uncertainties,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (bypass_result(boundaries=[boundary(0, 20)]), "coarse_boundary_count"),
        (
            bypass_result(
                boundaries=[
                    CoarseBoundary.model_construct(from_subtask_index=0, to_subtask_index=2, estimated_frame=20, evidence="x"),
                    boundary(1, 60),
                ]
            ),
            "coarse_boundary_order",
        ),
        (bypass_result(boundaries=[boundary(0, 60), boundary(1, 40)]), "coarse_boundary_order"),
        (bypass_result(boundaries=[boundary(0, 0), boundary(1, 60)]), "coarse_boundary_order"),
        (bypass_result(boundaries=[boundary(0, 20), boundary(1, 101)]), "coarse_boundary_order"),
        (bypass_result(start=1), "illegal_coarse_sequence"),
    ],
)
async def test_bypassed_model_validation_cannot_evade_deterministic_checks(
    tmp_path: Path, bad: CoarseResult, reason: str
) -> None:
    good = result([0, 1, 2], [20, 60])

    decision = await run_coarse(make_config(tmp_path), make_episode(tmp_path), RecordingSampler(), RecordingClient([bad, good]))

    assert reason in decision.reasons
    assert decision.status == "needs_review"


@pytest.mark.asyncio
async def test_uncertainty_in_either_attempt_requires_review_without_confidence_threshold(tmp_path: Path) -> None:
    uncertain = result([0, 1, 2], [20, 60], uncertainties=["handoff is occluded"])
    confident = result([0, 1, 2], [21, 61])

    decision = await run_coarse(make_config(tmp_path), make_episode(tmp_path), RecordingSampler(), RecordingClient([uncertain, confident]))

    assert decision.reasons == ("coarse_uncertain",)


def test_two_sampling_grids_preserve_endpoints_cap_and_differ_when_possible() -> None:
    first = coarse_pass_indices(101, 20.0, 2.0, 8, 0)
    second = coarse_pass_indices(101, 20.0, 2.0, 8, 1)
    assert first[0] == second[0] == 0
    assert first[-1] == second[-1] == 100
    assert len(first) == len(second) == 8
    assert first == sorted(set(first)) and second == sorted(set(second))
    assert first != second
    assert coarse_pass_indices(4, 20.0, 20.0, 8, 0) == [0, 1, 2, 3]
    assert coarse_pass_indices(4, 20.0, 20.0, 8, 1) == [0, 1, 2, 3]
    assert coarse_pass_indices(1, 20.0, 2.0, 8, 1) == [0]


@pytest.mark.asyncio
async def test_sampler_receives_exact_video_camera_grid_and_source_fps(tmp_path: Path) -> None:
    sampler = RecordingSampler()
    client = RecordingClient([result([0, 1, 2], [20, 60]), result([0, 1, 2], [21, 61])])
    episode = make_episode(tmp_path)

    decision = await run_coarse(make_config(tmp_path), episode, sampler, client)

    assert [call[0] for call in sampler.calls] == [episode.videos["cam.eye"], episode.videos["cam.eye"]]
    assert [call[1] for call in sampler.calls] == ["cam.eye", "cam.eye"]
    assert [call[2] for call in sampler.calls] == [list(grid) for grid in decision.sampled_frame_indices]
    assert [call[3] for call in sampler.calls] == [20.0, 20.0]
    assert decision.sampled_frame_indices[0] != decision.sampled_frame_indices[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corrupt",
    [
        lambda samples: samples[:-1],
        lambda samples: list(reversed(samples)),
        lambda samples: samples + [samples[-1]],
        lambda samples: [samples[0].model_copy(update={"camera_key": "cam.other"}), *samples[1:]],
    ],
)
async def test_sampler_evidence_must_exactly_match_request(tmp_path: Path, corrupt) -> None:
    client = RecordingClient([result([0, 1, 2], [20, 60]), result([0, 1, 2], [21, 61])])

    with pytest.raises(ValueError, match="sampler evidence"):
        await run_coarse(make_config(tmp_path), make_episode(tmp_path), RecordingSampler(corrupt=corrupt), client)
    assert client.calls == []


@pytest.mark.asyncio
async def test_missing_primary_camera_and_invalid_episode_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="primary camera"):
        await run_coarse(
            make_config(tmp_path),
            make_episode(tmp_path, camera="cam.other"),
            RecordingSampler(),
            RecordingClient([]),
        )
    bad_episode = make_episode(tmp_path).model_construct(
        episode_index=7,
        length=0,
        task="arrange",
        parquet=tmp_path / "episode.parquet",
        videos={"cam.eye": tmp_path / "cam.eye.mp4"},
    )
    with pytest.raises(ValueError, match="length"):
        await run_coarse(make_config(tmp_path), bad_episode, RecordingSampler(), RecordingClient([]))


@pytest.mark.asyncio
async def test_client_exception_propagates_after_first_pass(tmp_path: Path) -> None:
    failure = RuntimeError("service failed")
    client = RecordingClient([failure])
    with pytest.raises(RuntimeError, match="service failed"):
        await run_coarse(make_config(tmp_path), make_episode(tmp_path), RecordingSampler(), client)


def test_coarse_decision_is_strict_immutable_and_enforces_status_invariants() -> None:
    attempts = (result([0], []), result([0], []))
    decision = CoarseDecision(
        status="coarse_done",
        attempts=attempts,
        reasons=(),
        start_subtask_index=0,
        observed_subtask_indices=(0,),
        boundary_centers=(),
        sampled_frame_indices=((0,), (0,)),
    )
    with pytest.raises(ValidationError):
        decision.status = "needs_review"
    with pytest.raises(ValidationError):
        CoarseDecision(
            status="needs_review",
            attempts=attempts,
            reasons=(),
            start_subtask_index=None,
            observed_subtask_indices=(),
            boundary_centers=(),
            sampled_frame_indices=((0,), (0,)),
        )
    with pytest.raises(ValidationError):
        CoarseDecision(
            status="coarse_done",
            attempts=attempts,
            reasons=(),
            start_subtask_index="0",
            observed_subtask_indices=(0,),
            boundary_centers=(),
            sampled_frame_indices=((0,), (0,)),
        )


@pytest.mark.asyncio
async def test_source_files_are_not_modified(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    video = make_episode(tmp_path).videos["cam.eye"]
    video.write_bytes(b"source video")
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (config.source / "meta" / "info.json", video)
    }

    await run_coarse(
        config,
        make_episode(tmp_path),
        RecordingSampler(),
        RecordingClient([result([0, 1, 2], [20, 60]), result([0, 1, 2], [21, 61])]),
    )

    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in before
    } == before

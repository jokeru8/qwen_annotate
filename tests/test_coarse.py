import json
import asyncio
import hashlib
import math
import os
import stat
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from robo_annotate.coarse import CoarseDecision, coarse_pass_indices, run_coarse
from robo_annotate.config import AnnotationConfig
from robo_annotate.lerobot import EpisodeInfo
from robo_annotate.models import CoarseBoundary, CoarseResult
from robo_annotate.video import FrameSample
from robo_annotate.workspace import compute_source_fingerprint
from tests.fixtures import make_episode_info


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
    source = tmp_path / "source"
    parquet = source / "data" / "episode.parquet"
    video = source / "videos" / f"{camera}.mp4"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    video.parent.mkdir(parents=True, exist_ok=True)
    if not parquet.exists():
        parquet.touch()
    if not video.exists():
        video.touch()
    return make_episode_info(
        episode_index=7,
        length=length,
        task="arrange objects",
        parquet=parquet,
        videos={camera: video},
        fps=20.0,
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
    semantic_codes: list[str] | None = None,
    precision_notes: list[str] | None = None,
) -> CoarseResult:
    return CoarseResult(
        start_subtask_index=observed[0] if start is None else start,
        observed_subtask_indices=observed,
        coarse_boundaries=[boundary(left, frame) for left, frame in zip(observed, frames)],
        confidence=0.9,
        semantic_uncertainty_codes=[] if semantic_codes is None else semantic_codes,
        boundary_precision_notes=[] if precision_notes is None else precision_notes,
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
        ("dagger_patch", [3, 4]),
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


def bypass_result(*, start=0, observed=None, boundaries=None, semantic_codes=None, precision_notes=None):
    return CoarseResult.model_construct(
        start_subtask_index=start,
        observed_subtask_indices=[0, 1, 2] if observed is None else observed,
        coarse_boundaries=[boundary(0, 20), boundary(1, 60)] if boundaries is None else boundaries,
        confidence=0.9,
        semantic_uncertainty_codes=[] if semantic_codes is None else semantic_codes,
        boundary_precision_notes=[] if precision_notes is None else precision_notes,
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
    uncertain = result([0, 1, 2], [20, 60], semantic_codes=["transition_neighborhood_unclear"])
    confident = result([0, 1, 2], [21, 61])

    decision = await run_coarse(make_config(tmp_path), make_episode(tmp_path), RecordingSampler(), RecordingClient([uncertain, confident]))

    assert decision.reasons == ("coarse_uncertain",)


@pytest.mark.asyncio
async def test_precision_notes_are_audited_without_blocking_coarse(tmp_path: Path) -> None:
    """Catches precision-only notes being confused with semantic uncertainty."""
    first = CoarseResult(
        start_subtask_index=0,
        observed_subtask_indices=[0, 1, 2],
        coarse_boundaries=[boundary(0, 20), boundary(1, 60)],
        confidence=0.9,
        semantic_uncertainty_codes=[],
        boundary_precision_notes=["Exact first boundary may be a few frames later."],
    )
    second = first.model_copy(update={
        "coarse_boundaries": [boundary(0, 21), boundary(1, 61)],
        "boundary_precision_notes": ["Sparse evidence only localizes approximate centers."],
    })

    decision = await run_coarse(
        make_config(tmp_path), make_episode(tmp_path), RecordingSampler(),
        RecordingClient([first, second]),
    )

    assert decision.status == "coarse_done" and decision.reasons == ()
    assert decision.attempts[0].semantic_uncertainty_codes == ()
    assert decision.attempts[0].boundary_precision_notes == (
        "Exact first boundary may be a few frames later.",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [
    "subtask_order_unclear",
    "start_subtask_unclear",
    "transition_neighborhood_unclear",
])
async def test_each_semantic_uncertainty_code_blocks_coarse(tmp_path: Path, code: str) -> None:
    uncertain = CoarseResult(
        start_subtask_index=0,
        observed_subtask_indices=[0, 1, 2],
        coarse_boundaries=[boundary(0, 20), boundary(1, 60)],
        confidence=0.9,
        semantic_uncertainty_codes=[code],
        boundary_precision_notes=[],
    )
    clear = uncertain.model_copy(update={
        "semantic_uncertainty_codes": [],
        "coarse_boundaries": [boundary(0, 21), boundary(1, 61)],
    })

    decision = await run_coarse(
        make_config(tmp_path), make_episode(tmp_path), RecordingSampler(),
        RecordingClient([uncertain, clear]),
    )

    assert decision.status == "needs_review"
    assert decision.reasons == ("coarse_uncertain",)


def test_layered_uncertainty_schema_rejects_unknown_codes_and_legacy_field() -> None:
    schema = CoarseResult.model_json_schema()
    assert "semantic_uncertainty_codes" in schema["properties"]
    assert "boundary_precision_notes" in schema["properties"]
    assert "uncertainties" not in schema["properties"]
    assert {"semantic_uncertainty_codes", "boundary_precision_notes"} <= set(schema["required"])
    payload = {
        "start_subtask_index": 0,
        "observed_subtask_indices": [0],
        "coarse_boundaries": [],
        "confidence": 0.9,
        "semantic_uncertainty_codes": ["invented_code"],
        "boundary_precision_notes": [],
    }
    with pytest.raises(ValidationError):
        CoarseResult.model_validate(payload)
    payload["semantic_uncertainty_codes"] = []
    payload["uncertainties"] = ["legacy ambiguity"]
    with pytest.raises(ValidationError):
        CoarseResult.model_validate(payload)


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


def test_sampling_grids_have_complementary_phase_over_most_available_interior_frames() -> None:
    for frame_count in range(1, 80):
        for source_fps in (5.0, 20.0, 29.97):
            for target_fps in (0.25, 1.0, 8.0, 40.0):
                for cap in (2, 3, 8, 17):
                    first = coarse_pass_indices(frame_count, source_fps, target_fps, cap, 0)
                    second = coarse_pass_indices(frame_count, source_fps, target_fps, cap, 1)
                    for grid in (first, second):
                        assert grid == sorted(set(grid))
                        assert grid[0] == 0 and grid[-1] == frame_count - 1
                        assert len(grid) <= cap and all(0 <= value < frame_count for value in grid)
                    assert len(first) == len(second)
                    interiors = len(first) - 2
                    slack = frame_count - len(first)
                    if interiors > 0 and slack > 0:
                        possible_new = min(interiors, slack)
                        new_second = len(set(second[1:-1]) - set(first[1:-1]))
                        assert new_second >= math.ceil(possible_new / 2)


@pytest.mark.asyncio
async def test_sampler_receives_exact_video_camera_grid_and_source_fps(tmp_path: Path) -> None:
    sampler = RecordingSampler()
    client = RecordingClient([result([0, 1, 2], [20, 60]), result([0, 1, 2], [21, 61])])
    episode = make_episode(tmp_path)

    decision = await run_coarse(make_config(tmp_path), episode, sampler, client)

    assert len(sampler.calls) == 1
    assert sampler.calls[0][0] == episode.videos["cam.eye"].path.resolve()
    assert sampler.calls[0][1] == "cam.eye"
    assert sampler.calls[0][2] == sorted(set().union(*decision.sampled_frame_indices))
    assert sampler.calls[0][3] == 20.0
    assert decision.sampled_frame_indices[0] != decision.sampled_frame_indices[1]
    assert [[sample.frame_index for sample in call[1]] for call in client.calls] == [
        list(grid) for grid in decision.sampled_frame_indices
    ]


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
async def test_sampler_timestamp_must_match_exact_source_frame_time(tmp_path: Path) -> None:
    def wrong_timestamp(samples):
        return [samples[0].model_copy(update={"timestamp_seconds": 99.0}), *samples[1:]]

    with pytest.raises(ValueError, match="timestamp"):
        await run_coarse(
            make_config(tmp_path),
            make_episode(tmp_path),
            RecordingSampler(corrupt=wrong_timestamp),
            RecordingClient([]),
        )


@pytest.mark.asyncio
async def test_blocking_sampler_runs_once_without_blocking_event_loop(tmp_path: Path) -> None:
    events = []
    base = RecordingSampler()

    def blocking_sampler(*args):
        events.append("sampler_start")
        time.sleep(0.08)
        events.append("sampler_end")
        return base(*args)

    async def heartbeat():
        await asyncio.sleep(0.01)
        events.append("heartbeat")

    client = RecordingClient([result([0, 1, 2], [20, 60]), result([0, 1, 2], [21, 61])])
    annotation, _ = await asyncio.gather(
        run_coarse(make_config(tmp_path), make_episode(tmp_path), blocking_sampler, client),
        heartbeat(),
    )
    assert annotation.status == "coarse_done"
    assert events.index("heartbeat") < events.index("sampler_end")
    assert len(base.calls) == 1


@pytest.mark.asyncio
async def test_sampler_failure_from_background_decode_propagates(tmp_path: Path) -> None:
    def broken_sampler(*args):
        raise RuntimeError("decode failed")

    with pytest.raises(RuntimeError, match="decode failed"):
        await run_coarse(make_config(tmp_path), make_episode(tmp_path), broken_sampler, RecordingClient([]))


@pytest.mark.asyncio
async def test_missing_primary_camera_and_invalid_episode_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="primary camera"):
        await run_coarse(
            make_config(tmp_path),
            make_episode(tmp_path, camera="cam.other"),
            RecordingSampler(),
            RecordingClient([]),
        )
    bad_episode = make_episode(tmp_path).model_copy(update={"length": 0})
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
        mode="complete",
        subtask_count=1,
        frame_count=1,
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
            mode="complete",
            subtask_count=1,
            frame_count=1,
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
            mode="complete",
            subtask_count=1,
            frame_count=1,
            status="coarse_done",
            attempts=attempts,
            reasons=(),
            start_subtask_index="0",
            observed_subtask_indices=(0,),
            boundary_centers=(),
            sampled_frame_indices=((0,), (0,)),
        )


def decision_fields(
    *,
    attempts=None,
    mode="complete",
    subtask_count=3,
    frame_count=101,
    status="coarse_done",
    reasons=(),
    start=0,
    observed=(0, 1, 2),
    centers=(21, 61),
    grids=((0, 20, 40, 60, 80, 100), (0, 30, 50, 70, 90, 100)),
):
    return {
        "mode": mode,
        "subtask_count": subtask_count,
        "frame_count": frame_count,
        "status": status,
        "attempts": attempts or (result([0, 1, 2], [20, 60]), result([0, 1, 2], [21, 61])),
        "reasons": reasons,
        "start_subtask_index": start,
        "observed_subtask_indices": observed,
        "boundary_centers": centers,
        "sampled_frame_indices": grids,
    }


@pytest.mark.parametrize(
    "grids",
    [
        ((-1, 50, 100), (0, 50, 100)),
        ((0, 50, 50, 100), (0, 50, 100)),
        ((0, 80, 20, 100), (0, 50, 100)),
        ((1, 50, 100), (0, 50, 100)),
        ((0, 50, 99), (0, 50, 100)),
        ((0, 50, 101), (0, 50, 100)),
        ((0, True, 100), (0, 50, 100)),
    ],
)
def test_decision_rejects_malformed_sample_provenance(grids) -> None:
    with pytest.raises(ValidationError, match="sampled"):
        CoarseDecision(**decision_fields(grids=grids))


@pytest.mark.parametrize(
    "overrides",
    [
        {"start": 1},
        {"observed": (0, 2)},
        {"centers": (21,)},
        {"centers": (22, 61)},
        {"centers": (-1, 61)},
        {"centers": (61, 21)},
        {"centers": (21, 101)},
    ],
)
def test_success_fields_must_be_exactly_derived_from_attempts(overrides) -> None:
    with pytest.raises(ValidationError):
        CoarseDecision(**decision_fields(**overrides))


def test_success_rejects_illegal_or_uncertain_audit_attempts() -> None:
    illegal = bypass_result(observed=[0, 2], boundaries=[
        CoarseBoundary.model_construct(
            from_subtask_index=0,
            to_subtask_index=2,
            estimated_frame=20,
            evidence="x",
        )
    ])
    with pytest.raises(ValidationError):
        CoarseDecision(**decision_fields(attempts=(illegal, illegal)))
    uncertain = result([0, 1, 2], [20, 60], semantic_codes=["transition_neighborhood_unclear"])
    with pytest.raises(ValidationError):
        CoarseDecision(**decision_fields(attempts=(uncertain, uncertain), centers=(20, 60)))


def test_needs_review_reasons_must_exactly_match_recomputed_audit_truth() -> None:
    illegal = bypass_result(start=1)
    base = decision_fields(
        attempts=(illegal, result([0, 1, 2], [20, 60])),
        status="needs_review",
        reasons=("illegal_coarse_sequence",),
        start=None,
        observed=(),
        centers=(),
    )
    with pytest.raises(ValidationError, match="recomputed"):
        CoarseDecision(**base)
    actual = (
        "coarse_sequence_disagreement",
        "illegal_coarse_sequence",
    )
    accepted = CoarseDecision(**(base | {"reasons": actual}))
    assert accepted.reasons == actual
    with pytest.raises(ValidationError):
        CoarseDecision(**(base | {"reasons": tuple(reversed(actual))}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "other"),
        ("subtask_count", True),
        ("subtask_count", "3"),
        ("subtask_count", 0),
        ("frame_count", True),
        ("frame_count", "101"),
        ("frame_count", 0),
    ],
)
def test_decision_audit_context_is_strict_and_positive(field, value) -> None:
    data = decision_fields()
    data[field] = value
    with pytest.raises(ValidationError):
        CoarseDecision(**data)


def test_valid_complete_dagger_and_singleton_decisions_roundtrip_json() -> None:
    complete = CoarseDecision(**decision_fields())
    dagger_attempts = (result([1, 2], [40]), result([1, 2], [41]))
    dagger = CoarseDecision(
        **decision_fields(
            attempts=dagger_attempts,
            mode="dagger_patch",
            subtask_count=3,
            start=1,
            observed=(1, 2),
            centers=(41,),
        )
    )
    singleton_attempts = (result([1], []), result([1], []))
    singleton = CoarseDecision(
        **decision_fields(
            attempts=singleton_attempts,
            mode="dagger_patch",
            subtask_count=3,
            start=1,
            observed=(1,),
            centers=(),
        )
    )
    for original in (complete, dagger, singleton):
        restored = CoarseDecision.model_validate_json(original.model_dump_json())
        assert restored == original


def test_needs_review_with_bypassed_semantic_attempt_roundtrips_for_audit() -> None:
    illegal = bypass_result(start=1)
    decision = CoarseDecision(
        **decision_fields(
            attempts=(illegal, result([0, 1, 2], [20, 60])),
            status="needs_review",
            reasons=("coarse_sequence_disagreement", "illegal_coarse_sequence"),
            start=None,
            observed=(),
            centers=(),
        )
    )

    restored = CoarseDecision.model_validate_json(decision.model_dump_json())

    assert restored == decision
    assert restored.attempts[0].start_subtask_index == 1


def test_model_validate_json_rejects_tampered_sample_grid() -> None:
    payload = CoarseDecision(**decision_fields()).model_dump(mode="json")
    payload["sampled_frame_indices"][0] = [0, 50, 50, 100]
    with pytest.raises(ValidationError, match="sampled"):
        CoarseDecision.model_validate_json(json.dumps(payload))


def test_decision_rejects_two_insufficiently_independent_sparse_grids() -> None:
    with pytest.raises(ValidationError, match="independent"):
        CoarseDecision(
            **decision_fields(
                grids=((0, 20, 40, 60, 80, 100), (0, 21, 40, 60, 80, 100))
            )
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.append(9),
        lambda value: value.extend([9]),
        lambda value: value.insert(0, 9),
        lambda value: value.pop(),
        lambda value: value.remove(value[0]),
        lambda value: value.clear(),
        lambda value: value.reverse(),
        lambda value: value.sort(),
        lambda value: value.__setitem__(0, value[0]),
        lambda value: value.__delitem__(0),
        lambda value: value.__iadd__([9]),
        lambda value: value.__imul__(2),
    ],
)
@pytest.mark.parametrize("field", [
    "observed_subtask_indices", "coarse_boundaries",
    "semantic_uncertainty_codes", "boundary_precision_notes",
])
def test_attempt_audit_lists_block_every_mutation_api(field, mutate) -> None:
    first = result(
        [0, 1, 2], [20, 60],
        semantic_codes=["transition_neighborhood_unclear"], precision_notes=["review note"],
    )
    second = result(
        [0, 1, 2], [21, 61],
        semantic_codes=["transition_neighborhood_unclear"], precision_notes=["review note"],
    )
    decision = CoarseDecision(
        **decision_fields(
            attempts=(first, second),
            status="needs_review",
            reasons=("coarse_uncertain",),
            start=None,
            observed=(),
            centers=(),
        )
    )
    before = decision.model_dump(mode="json")

    with pytest.raises((AttributeError, TypeError)):
        mutate(getattr(decision.attempts[0], field))

    assert decision.model_dump(mode="json") == before


@pytest.mark.parametrize(
    ("operation", "field"),
    [
        (lambda value: list.append(value, 9), "observed_subtask_indices"),
        (lambda value: list.extend(value, [9]), "observed_subtask_indices"),
        (lambda value: list.insert(value, 0, 9), "coarse_boundaries"),
        (lambda value: list.pop(value), "coarse_boundaries"),
        (lambda value: list.clear(value), "semantic_uncertainty_codes"),
        (lambda value: list.clear(value), "boundary_precision_notes"),
        (lambda value: list.__setitem__(value, 0, 9), "observed_subtask_indices"),
        (lambda value: list.__delitem__(value, 0), "coarse_boundaries"),
    ],
)
def test_unbound_list_mutators_cannot_bypass_attempt_immutability(operation, field) -> None:
    first = result(
        [0, 1, 2], [20, 60],
        semantic_codes=["transition_neighborhood_unclear"], precision_notes=["review note"],
    )
    second = result(
        [0, 1, 2], [21, 61],
        semantic_codes=["transition_neighborhood_unclear"], precision_notes=["review note"],
    )
    decision = CoarseDecision(
        **decision_fields(
            attempts=(first, second),
            status="needs_review",
            reasons=("coarse_uncertain",),
            start=None,
            observed=(),
            centers=(),
        )
    )
    snapshot = decision.model_dump(mode="json")
    audit_field = getattr(decision.attempts[0], field)

    assert isinstance(decision.attempts[0], CoarseResult)
    assert isinstance(audit_field, tuple)
    with pytest.raises(TypeError):
        operation(audit_field)
    assert decision.model_dump(mode="json") == snapshot


def test_decision_deep_copies_caller_owned_attempt_lists() -> None:
    first = result([0, 1, 2], [20, 60])
    second = result([0, 1, 2], [21, 61])
    decision = CoarseDecision(**decision_fields(attempts=(first, second)))
    assert decision.attempts[0] is not first

    first.observed_subtask_indices.append(99)
    first.coarse_boundaries.clear()
    first.semantic_uncertainty_codes.append("transition_neighborhood_unclear")
    first.boundary_precision_notes.append("changed later")

    assert decision.status == "coarse_done"
    assert decision.attempts[0].observed_subtask_indices == (0, 1, 2)
    assert len(decision.attempts[0].coarse_boundaries) == 2
    assert decision.attempts[0].semantic_uncertainty_codes == ()
    assert decision.attempts[0].boundary_precision_notes == ()


def test_mutating_dumped_payload_cannot_change_decision_audit_truth() -> None:
    decision = CoarseDecision(**decision_fields())
    python_payload = decision.model_dump()
    assert isinstance(python_payload["attempts"][0]["observed_subtask_indices"], tuple)
    assert isinstance(python_payload["attempts"][0]["coarse_boundaries"], tuple)
    assert isinstance(python_payload["attempts"][0]["semantic_uncertainty_codes"], tuple)
    assert isinstance(python_payload["attempts"][0]["boundary_precision_notes"], tuple)
    python_payload["attempts"][0]["coarse_boundaries"][0]["evidence"] = "tampered"
    payload = decision.model_dump(mode="json")
    assert isinstance(payload["attempts"][0]["observed_subtask_indices"], list)
    payload["attempts"][0]["observed_subtask_indices"].append(99)
    payload["attempts"][0]["coarse_boundaries"].clear()
    payload["attempts"][0]["semantic_uncertainty_codes"].append("subtask_order_unclear")
    payload["attempts"][0]["boundary_precision_notes"].append("tampered")

    assert decision.status == "coarse_done"
    assert decision.attempts[0].observed_subtask_indices == (0, 1, 2)
    assert len(decision.attempts[0].coarse_boundaries) == 2
    assert decision.attempts[0].coarse_boundaries[0].evidence == "visible transition"
    assert decision.attempts[0].semantic_uncertainty_codes == ()
    assert decision.attempts[0].boundary_precision_notes == ()


def test_malformed_model_construct_attempt_is_rejected_as_validation_error() -> None:
    malformed = CoarseResult.model_construct(
        start_subtask_index=0,
        observed_subtask_indices=None,
        coarse_boundaries=None,
        confidence=0.9,
        semantic_uncertainty_codes=None,
        boundary_precision_notes=None,
    )
    with pytest.raises(ValidationError, match="attempt"):
        CoarseDecision(
            **decision_fields(
                attempts=(malformed, malformed),
                status="needs_review",
                reasons=("illegal_coarse_sequence",),
                start=None,
                observed=(),
                centers=(),
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        '{"fps":20.0,"fps":21.0}',
        '{"fps":NaN}',
        '[{"fps":20.0}]',
        '{"fps":20.0 trailing secret-source-content}',
    ],
)
async def test_source_info_requires_unique_standard_json_object_without_echoing_content(
    tmp_path: Path, raw: str
) -> None:
    config = make_config(tmp_path)
    episode = make_episode(tmp_path)
    (config.source / "meta" / "info.json").write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        await run_coarse(config, episode, RecordingSampler(), RecordingClient([]))

    assert "secret-source-content" not in str(caught.value)


@pytest.mark.asyncio
async def test_authoritative_source_fps_must_match_current_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="authoritative"):
        await run_coarse(
            make_config(tmp_path),
            make_episode(tmp_path),
            RecordingSampler(),
            RecordingClient([]),
            source_fps=19.0,
        )


@pytest.mark.asyncio
async def test_primary_video_must_be_regular_contained_source_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    contained = make_episode(tmp_path)
    outside = tmp_path / "outside.mp4"
    outside.touch()
    external = contained.model_copy(
        update={
            "videos": {
                "cam.eye": contained.videos["cam.eye"].model_copy(update={"path": outside})
            }
        }
    )
    with pytest.raises(ValueError, match="inside source"):
        await run_coarse(config, external, RecordingSampler(), RecordingClient([]))

    link = contained.videos["cam.eye"].path
    link.unlink()
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        await run_coarse(config, contained, RecordingSampler(), RecordingClient([]))


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["metadata", "video"])
async def test_source_change_during_sampling_is_detected(tmp_path: Path, changed: str) -> None:
    config = make_config(tmp_path)
    episode = make_episode(tmp_path)
    base = RecordingSampler()

    def changing_sampler(*args):
        samples = base(*args)
        if changed == "metadata":
            (config.source / "meta" / "info.json").write_text('{"fps":20.0,"changed":true}', encoding="utf-8")
        else:
            episode.videos["cam.eye"].path.write_bytes(b"changed")
        return samples

    with pytest.raises(ValueError, match="changed during coarse"):
        await run_coarse(config, episode, changing_sampler, RecordingClient([]))


@pytest.mark.asyncio
async def test_primary_video_symlink_swap_during_sampling_is_detected(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    episode = make_episode(tmp_path)
    outside = tmp_path / "replacement.mp4"
    outside.touch()
    base = RecordingSampler()

    def swapping_sampler(*args):
        samples = base(*args)
        episode.videos["cam.eye"].path.unlink()
        episode.videos["cam.eye"].path.symlink_to(outside)
        return samples

    with pytest.raises(ValueError, match="changed during coarse"):
        await run_coarse(config, episode, swapping_sampler, RecordingClient([]))


@pytest.mark.asyncio
async def test_expected_source_fingerprint_is_checked_before_sampling(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    episode = make_episode(tmp_path)
    actual = compute_source_fingerprint(config.source, episode)
    client = RecordingClient([result([0, 1, 2], [20, 60]), result([0, 1, 2], [21, 61])])
    accepted = await run_coarse(
        config,
        episode,
        RecordingSampler(),
        client,
        source_fps=20.0,
        expected_source_fingerprint=actual,
    )
    assert accepted.status == "coarse_done"

    with pytest.raises(ValueError, match="fingerprint"):
        await run_coarse(
            config,
            episode,
            RecordingSampler(),
            RecordingClient([]),
            expected_source_fingerprint="0" * 64,
        )


@pytest.mark.asyncio
async def test_source_files_are_not_modified(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    video = make_episode(tmp_path).videos["cam.eye"].path
    video.write_bytes(b"source video")
    before = recursive_source_snapshot(config.source)

    await run_coarse(
        config,
        make_episode(tmp_path),
        RecordingSampler(),
        RecordingClient([result([0, 1, 2], [20, 60]), result([0, 1, 2], [21, 61])]),
    )

    assert recursive_source_snapshot(config.source) == before


def recursive_source_snapshot(root: Path):
    snapshot = {}
    for path in [root, *sorted(root.rglob("*"))]:
        file_stat = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        common = (
            file_stat.st_dev,
            file_stat.st_ino,
            stat.S_IFMT(file_stat.st_mode),
            stat.S_IMODE(file_stat.st_mode),
            file_stat.st_size,
            file_stat.st_mtime_ns,
        )
        if stat.S_ISREG(file_stat.st_mode):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        elif stat.S_ISLNK(file_stat.st_mode):
            digest = os.readlink(path)
        else:
            digest = None
        snapshot[relative] = (common, digest)
    return snapshot

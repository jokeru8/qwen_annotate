import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, Thread

import pytest
from pydantic import ValidationError

from qwen_annotate.coarse import CoarseDecision
from qwen_annotate.config import Subtask
from qwen_annotate.lerobot import DatasetIndex, EpisodeInfo
from qwen_annotate.models import CoarseBoundary, CoarseResult, FinalAnnotation, RefineResult, ValidationIssue
from qwen_annotate.prompts import PROMPT_VERSION
from qwen_annotate.refine import RefineDecision
from qwen_annotate.workspace import (
    EpisodeRecord,
    RunManifest,
    WorkspaceStore,
    compute_run_fingerprint,
    compute_source_fingerprint,
)
from tests.fixtures import make_config


SHA = "a" * 40
FP = "b" * 64
NOW = datetime(2026, 8, 22, tzinfo=UTC)
MODEL_AUDIT = {
    "prompt_version": PROMPT_VERSION,
    "model_revision": SHA,
    "sampling_details": {"coarse": {"frames": [0, 5, 9]}},
}


def make_index(tmp_path: Path, lengths: list[int] = [10, 12]) -> DatasetIndex:
    root = tmp_path / "source"
    root.mkdir()
    episodes = []
    for index, length in enumerate(lengths):
        parquet = root / f"data/episode_{index:06d}.parquet"
        video = root / f"videos/cam.eye/episode_{index:06d}.mp4"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        video.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(bytes([index + 1]) * 7)
        video.write_bytes(bytes([index + 2]) * 11)
        episodes.append(EpisodeInfo(episode_index=index, length=length, task="arrange", parquet=parquet, videos={"cam.eye": video}))
    return DatasetIndex(root=root.resolve(), version="v2.1", fps=5.0, camera_keys=["cam.eye"], episodes=episodes)


def pending(index: int = 0, source: str = FP, run: str = "c" * 64) -> EpisodeRecord:
    return EpisodeRecord(episode_index=index, source_fingerprint=source, run_fingerprint=run, created_at=NOW, updated_at=NOW)


def coarse_result() -> CoarseResult:
    return CoarseResult(
        start_subtask_index=0,
        observed_subtask_indices=[0, 1],
        coarse_boundaries=[CoarseBoundary(from_subtask_index=0, to_subtask_index=1, estimated_frame=5, evidence="placed")],
        confidence=0.8,
    )


def refine_result() -> RefineResult:
    return RefineResult(
        from_subtask_index=0,
        to_subtask_index=1,
        last_frame_before=4,
        first_frame_after=5,
        boundary_frame=5,
        confidence=0.9,
        visible_cues=["release"],
    )


def zero_transition_review_audit() -> tuple[CoarseDecision, RefineDecision]:
    attempt = CoarseResult(
        start_subtask_index=0,
        observed_subtask_indices=[0],
        coarse_boundaries=[],
        confidence=0.9,
    )
    coarse = CoarseDecision(
        mode="complete",
        subtask_count=1,
        frame_count=10,
        status="coarse_done",
        attempts=(attempt, attempt.model_copy(deep=True)),
        reasons=(),
        start_subtask_index=0,
        observed_subtask_indices=(0,),
        boundary_centers=(),
        sampled_frame_indices=((0, 9), (0, 9)),
    )
    refined = RefineDecision(
        mode="complete",
        subtask_count=1,
        frame_count=10,
        min_segment_frames=11,
        agreement_tolerance_frames=12,
        start_subtask_index=0,
        observed_subtask_indices=(0,),
        coarse_boundary_centers=(),
        source_fps=5.0,
        refine_window_seconds=2.5,
        refine_fps=8.0,
        dense_radius_seconds=0.5,
        camera_order=("cam.eye",),
        broad_radius_frames=12,
        base_broad_stride=1,
        dense_radius_frames=2,
        status="needs_review",
        attempts=(),
        provenance=(),
        reasons=("segment_too_short",),
        candidate_annotation=FinalAnnotation(start_subtask_index=0, boundaries=[]),
    )
    return coarse, refined


def transition(record: EpisodeRecord, status: str, **updates) -> EpisodeRecord:
    return record.model_copy(update={"status": status, "updated_at": record.updated_at + timedelta(seconds=1), **updates})


def test_models_are_strict_and_status_fields_are_consistent() -> None:
    with pytest.raises(ValidationError):
        EpisodeRecord(episode_index="0", source_fingerprint=FP, run_fingerprint=FP)
    with pytest.raises(ValidationError, match="accepted"):
        pending().model_copy(update={"status": "accepted"}).__class__.model_validate(
            pending().model_dump() | {"status": "accepted"}
        )
    with pytest.raises(ValidationError, match="decision_source"):
        EpisodeRecord.model_validate(pending().model_dump() | {"status": "accepted", "final_annotation": {"start_subtask_index": 0, "boundaries": []}})
    with pytest.raises(ValidationError, match="needs_review"):
        EpisodeRecord.model_validate(pending().model_dump() | {"status": "needs_review"})
    with pytest.raises(ValidationError, match="failure_category"):
        EpisodeRecord.model_validate(pending().model_dump() | {"status": "failed"})
    with pytest.raises(ValidationError, match="pending"):
        EpisodeRecord.model_validate(pending().model_dump() | {"coarse_attempts": [coarse_result().model_dump()]})
    with pytest.raises(ValidationError, match="SHA-256"):
        EpisodeRecord(episode_index=0, source_fingerprint="not-a-hash", run_fingerprint=FP)
    with pytest.raises(ValidationError, match="40"):
        EpisodeRecord(
            episode_index=0, source_fingerprint=FP, run_fingerprint=FP,
            status="failed", failure_category="x", model_revision="main",
        )
    audit = EpisodeRecord(
        episode_index=0, source_fingerprint=FP, run_fingerprint=FP,
        status="failed", failure_category="x",
        sampling_details={"coarse": {"frames": [0, 4, 8], "camera": "eye"}},
    )
    assert audit.sampling_details["coarse"] == {"frames": [0, 4, 8], "camera": "eye"}


@pytest.mark.parametrize(
    "audit",
    [
        {},
        {"prompt_version": "historical", "sampling_details": {"fps": 1}},
        {"prompt_version": PROMPT_VERSION, "model_revision": SHA, "sampling_details": {}},
    ],
)
def test_model_acceptance_requires_exact_reproducibility_provenance(audit: dict) -> None:
    data = pending().model_dump() | {
        "status": "accepted",
        "final_annotation": {"start_subtask_index": 0, "boundaries": []},
        "decision_source": "model",
    } | audit
    with pytest.raises(ValidationError, match="provenance|prompt_version|sampling"):
        EpisodeRecord.model_validate(data)


def test_historical_prompt_model_record_can_deserialize_and_be_invalidated(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    base = pending()
    store.save_episode(base)
    coarse = transition(base, "coarse_done", coarse_attempts=[coarse_result()])
    store.save_episode(coarse)
    refined = transition(coarse, "refine_done", refine_attempts=[refine_result()])
    store.save_episode(refined)
    historical = transition(
        refined,
        "accepted",
        final_annotation=FinalAnnotation(start_subtask_index=0, boundaries=[5]),
        decision_source="model",
        prompt_version="historical-coarse-v0/refine-v0",
        model_revision=SHA,
        sampling_details={"frames": [0, 5, 9]},
    )
    store.save_episode(historical)
    assert store.load_episode(0).prompt_version == "historical-coarse-v0/refine-v0"
    assert store.invalidate_episode(
        0, source_fingerprint="d" * 64, run_fingerprint="e" * 64
    ).status == "pending"


def test_human_acceptance_does_not_need_model_provenance() -> None:
    record = EpisodeRecord.model_validate(
        pending().model_dump()
        | {
            "status": "accepted",
            "final_annotation": {"start_subtask_index": 0, "boundaries": []},
            "decision_source": "human",
        }
    )
    assert record.prompt_version is None and record.sampling_details == {}


def test_all_legal_transitions_and_human_review_acceptance(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "work", clock=lambda: NOW)
    store.create_layout()
    first = pending()
    store.save_episode(first)
    coarse = transition(first, "coarse_done", coarse_attempts=[coarse_result()])
    store.save_episode(coarse)
    refined = transition(coarse, "refine_done", refine_attempts=[refine_result()])
    store.save_episode(refined)
    review = transition(refined, "needs_review", validation_issues=[ValidationIssue(code="segment_too_short", message="short")], review_reasons=["low agreement"])
    store.save_episode(review)
    accepted = transition(review, "accepted", final_annotation=FinalAnnotation(start_subtask_index=0, boundaries=[5]), validation_issues=[], review_reasons=[], decision_source="human")
    store.save_episode(accepted)
    assert store.load_episode(0) == accepted


@pytest.mark.parametrize("from_status", ["pending", "coarse_done"])
def test_coarse_uncertainty_can_enter_review(from_status: str, tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    record = pending()
    store.save_episode(record)
    if from_status == "coarse_done":
        record = transition(record, "coarse_done", coarse_attempts=[coarse_result()])
        store.save_episode(record)
    review = transition(record, "needs_review", review_reasons=["coarse evidence is ambiguous"])
    store.save_episode(review)
    assert store.load_episode(0).status == "needs_review"


@pytest.mark.parametrize("mode", ["complete", "dagger_patch"])
def test_zero_transition_episode_flows_through_refine_done_to_accepted(
    mode: str, tmp_path: Path
) -> None:
    index = make_index(tmp_path, [10])
    config = make_config(index.root, tmp_path / "work", mode=mode)
    if mode == "dagger_patch":
        config = config.model_copy(
            update={"subtasks": [Subtask(skill="first", text="First"), Subtask(skill="second", text="Second")]}
        )
    store = WorkspaceStore(config.work_dir, clock=lambda: NOW)
    store.initialize(config, index, SHA)
    base = store.load_episode(0)
    one_subtask_coarse = CoarseResult(
        start_subtask_index=0,
        observed_subtask_indices=[0],
        coarse_boundaries=[],
        confidence=0.8,
    )
    coarse = transition(base, "coarse_done", coarse_attempts=[one_subtask_coarse])
    store.save_episode(coarse)
    refined = transition(
        coarse,
        "refine_done",
        final_annotation=FinalAnnotation(start_subtask_index=0, boundaries=[]),
    )
    store.save_episode(refined)
    accepted = transition(refined, "accepted", decision_source="model", **MODEL_AUDIT)
    store.save_episode(accepted)
    assert store.load_episode(0).status == "accepted"


def test_multi_subtask_complete_cannot_claim_zero_transition_refine_done(tmp_path: Path) -> None:
    index = make_index(tmp_path, [10])
    config = make_config(index.root, tmp_path / "work")
    config = config.model_copy(
        update={"subtasks": [Subtask(skill="first", text="First"), Subtask(skill="second", text="Second")]}
    )
    store = WorkspaceStore(config.work_dir, clock=lambda: NOW)
    store.initialize(config, index, SHA)
    base = store.load_episode(0)
    coarse = transition(base, "coarse_done", coarse_attempts=[coarse_result()])
    store.save_episode(coarse)
    invalid = transition(
        coarse,
        "refine_done",
        final_annotation=FinalAnnotation(start_subtask_index=0, boundaries=[]),
    )
    with pytest.raises(ValueError, match="final annotation"):
        store.save_episode(invalid)


@pytest.mark.parametrize(
    "final_annotation",
    [None, FinalAnnotation(start_subtask_index=0, boundaries=[5])],
)
def test_refine_done_without_refine_attempts_requires_zero_transition_final(
    final_annotation: FinalAnnotation | None,
) -> None:
    data = pending().model_dump() | {
        "status": "refine_done",
        "coarse_attempts": [coarse_result().model_dump()],
        "final_annotation": final_annotation.model_dump() if final_annotation else None,
    }
    with pytest.raises(ValidationError, match="refine_done"):
        EpisodeRecord.model_validate(data)


def test_audited_zero_transition_review_forbids_unrelated_final_annotation_on_model_save_and_replay(
    tmp_path: Path,
) -> None:
    coarse_decision, refine_decision = zero_transition_review_audit()
    base = pending()
    coarse = transition(base, "coarse_done", coarse_attempts=[
        CoarseResult.model_validate(item.model_dump(mode="json"))
        for item in coarse_decision.attempts
    ])
    details = {
        "coarse_decision": coarse_decision.model_dump(mode="json"),
        "refine_decision": refine_decision.model_dump(mode="json"),
    }
    valid = transition(coarse, "refine_done", sampling_details=details)
    assert valid.final_annotation is None

    adversarial_payload = valid.model_dump() | {
        "final_annotation": {"start_subtask_index": 0, "boundaries": [5]}
    }
    with pytest.raises(ValidationError, match="refine_done|annotation"):
        EpisodeRecord.model_validate(adversarial_payload)

    save_store = WorkspaceStore(tmp_path / "save")
    save_store.create_layout()
    save_store.save_episode(base)
    save_store.save_episode(coarse)
    bypassed = valid.model_copy(
        update={"final_annotation": FinalAnnotation(start_subtask_index=0, boundaries=[5])}
    )
    with pytest.raises(ValidationError, match="refine_done|annotation"):
        save_store.save_episode(bypassed)

    replay_store = WorkspaceStore(tmp_path / "replay")
    replay_store.create_layout()
    replay_store.save_episode(base)
    replay_store.save_episode(coarse)
    episode_path = replay_store.root / "episodes" / "episode_000000.json"
    adversarial_json = valid.model_dump(mode="json") | {
        "final_annotation": {"start_subtask_index": 0, "boundaries": [5]}
    }
    episode_path.write_text(json.dumps(adversarial_json))
    with pytest.raises(ValueError, match="Invalid episode record"):
        replay_store.load_episode(0)

    valid_store = WorkspaceStore(tmp_path / "valid")
    valid_store.create_layout()
    valid_store.save_episode(base)
    valid_store.save_episode(coarse)
    valid_store.save_episode(valid)
    assert valid_store.load_episode(0) == valid


def test_needs_review_can_only_be_accepted_by_a_human(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    base = pending()
    store.save_episode(base)
    review = transition(base, "needs_review", review_reasons=["check"])
    store.save_episode(review)
    model_accept = transition(
        review, "accepted", final_annotation=FinalAnnotation(start_subtask_index=0, boundaries=[]),
        review_reasons=[], decision_source="model", **MODEL_AUDIT,
    )
    with pytest.raises(ValueError, match="human"):
        store.save_episode(model_accept)


def test_state_changes_require_strictly_increasing_updated_at(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    record = pending()
    store.save_episode(record)
    changed = record.model_copy(
        update={"status": "coarse_done", "coarse_attempts": [coarse_result()]}
    )
    with pytest.raises(ValueError, match="updated_at"):
        store.save_episode(changed)


@pytest.mark.parametrize("status", ["pending", "coarse_done", "refine_done", "needs_review"])
def test_nonidentical_same_status_updates_are_rejected(status: str, tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    base = pending()
    if status == "coarse_done":
        base = transition(base, status, coarse_attempts=[coarse_result()])
    elif status == "refine_done":
        coarse = transition(base, "coarse_done", coarse_attempts=[coarse_result()])
        base = transition(
            coarse,
            status,
            refine_attempts=[refine_result()],
        )
    elif status == "needs_review":
        base = transition(base, status, review_reasons=["ambiguous"])
    initial = pending()
    store.save_episode(initial)
    if status != "pending":
        if status == "refine_done":
            coarse = transition(initial, "coarse_done", coarse_attempts=[coarse_result()])
            store.save_episode(coarse)
        store.save_episode(base)
    mutation = base.model_copy(
        update={"updated_at": base.updated_at + timedelta(seconds=1)}
    )
    with pytest.raises(ValueError, match="same-status"):
        store.save_episode(mutation)
    store.save_episode(base)  # exact retry remains idempotent


def test_manifest_missing_episode_cannot_be_recreated_by_save(tmp_path: Path) -> None:
    index = make_index(tmp_path, [10])
    config = make_config(index.root, tmp_path / "work")
    store = WorkspaceStore(config.work_dir, clock=lambda: NOW)
    store.initialize(config, index, SHA)
    base = store.load_episode(0)
    (store.root / "episodes/episode_000000.json").unlink()
    accepted = transition(
        base,
        "accepted",
        final_annotation=FinalAnnotation(start_subtask_index=0, boundaries=[]),
        decision_source="model",
        **MODEL_AUDIT,
    )
    with pytest.raises(ValueError, match="missing|corrupt"):
        store.save_episode(accepted)
    assert not (store.root / "episodes/episode_000000.json").exists()


def test_standalone_store_only_creates_new_pending_records(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    accepted = EpisodeRecord.model_validate(
        pending().model_dump()
        | {
            "status": "accepted",
            "final_annotation": {"start_subtask_index": 0, "boundaries": []},
            "decision_source": "model",
        }
        | MODEL_AUDIT
    )
    with pytest.raises(ValueError, match="new.*pending"):
        store.save_episode(accepted)


@pytest.mark.parametrize(
    ("old", "new"),
    [("pending", "refine_done"), ("pending", "accepted"), ("coarse_done", "pending"), ("coarse_done", "accepted"), ("refine_done", "coarse_done"), ("accepted", "needs_review"), ("failed", "pending")],
)
def test_rejects_skipped_backward_and_terminal_transitions(tmp_path: Path, old: str, new: str) -> None:
    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    base = pending()
    records = {
        "pending": base,
        "coarse_done": transition(base, "coarse_done", coarse_attempts=[coarse_result()]),
        "refine_done": transition(transition(base, "coarse_done", coarse_attempts=[coarse_result()]), "refine_done", refine_attempts=[refine_result()]),
        "accepted": EpisodeRecord.model_validate(base.model_dump() | {"status": "accepted", "final_annotation": {"start_subtask_index": 0, "boundaries": []}, "decision_source": "model"} | MODEL_AUDIT),
        "failed": EpisodeRecord.model_validate(base.model_dump() | {"status": "failed", "failure_category": "model_unavailable"}),
    }
    store.save_episode(base)
    if old == "coarse_done":
        store.save_episode(records[old])
    elif old == "refine_done":
        store.save_episode(records["coarse_done"])
        store.save_episode(records[old])
    elif old == "accepted":
        store.save_episode(records["coarse_done"])
        store.save_episode(records["refine_done"])
        records[old] = transition(
            records["refine_done"], "accepted",
            final_annotation=FinalAnnotation(start_subtask_index=0, boundaries=[]),
            decision_source="model", **MODEL_AUDIT,
        )
        store.save_episode(records[old])
    elif old == "failed":
        records[old] = transition(base, "failed", failure_category="model_unavailable")
        store.save_episode(records[old])
    candidate_data = records[old].model_dump() | {
        "status": new, "updated_at": records[old].updated_at + timedelta(seconds=1)
    }
    if new == "pending":
        candidate_data |= {
            "coarse_attempts": [], "refine_attempts": [], "final_annotation": None,
            "validation_issues": [], "review_reasons": [], "failure_category": None,
            "decision_source": None, "prompt_version": None, "model_revision": None,
            "sampling_details": {},
        }
    if new == "refine_done":
        candidate_data |= {"coarse_attempts": [coarse_result().model_dump()], "refine_attempts": [refine_result().model_dump()]}
    if new == "accepted":
        candidate_data |= {"final_annotation": {"start_subtask_index": 0, "boundaries": []}, "decision_source": "model"} | MODEL_AUDIT
    if new == "needs_review":
        candidate_data |= {"review_reasons": ["manual check"], "decision_source": None}
    with pytest.raises(ValueError, match="transition"):
        store.save_episode(EpisodeRecord.model_validate(candidate_data))


@pytest.mark.parametrize("status", ["pending", "coarse_done", "refine_done", "needs_review"])
def test_active_stages_can_fail(tmp_path: Path, status: str) -> None:
    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    base = pending()
    if status == "coarse_done":
        base = transition(base, status, coarse_attempts=[coarse_result()])
    elif status == "refine_done":
        base = transition(transition(base, "coarse_done", coarse_attempts=[coarse_result()]), status, refine_attempts=[refine_result()])
    elif status == "needs_review":
        base = transition(base, status, review_reasons=["check"])
    initial = pending()
    store.save_episode(initial)
    if status != "pending":
        if status == "refine_done":
            coarse = transition(initial, "coarse_done", coarse_attempts=[coarse_result()])
            store.save_episode(coarse)
        store.save_episode(base)
    failed = transition(base, "failed", failure_category="video_corrupt", review_reasons=[])
    store.save_episode(failed)
    assert store.load_episode(0).status == "failed"


def test_explicit_invalidation_clears_all_derived_state(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    base = pending()
    store.save_episode(base)
    coarse = transition(base, "coarse_done", coarse_attempts=[coarse_result()])
    store.save_episode(coarse)
    refined = transition(coarse, "refine_done", refine_attempts=[refine_result()])
    store.save_episode(refined)
    accepted = transition(
        refined, "accepted", final_annotation=FinalAnnotation(start_subtask_index=0, boundaries=[5]),
        decision_source="model", **MODEL_AUDIT,
    )
    store.save_episode(accepted)
    reset = store.invalidate_episode(0, source_fingerprint="d" * 64, run_fingerprint="e" * 64)
    assert reset.status == "pending"
    assert reset.source_fingerprint == "d" * 64 and reset.run_fingerprint == "e" * 64
    assert reset.coarse_attempts == [] and reset.refine_attempts == []
    assert reset.final_annotation is None and reset.validation_issues == []
    assert reset.failure_category is None and reset.decision_source is None
    assert reset.prompt_version is None and reset.model_revision is None and reset.sampling_details == {}
    assert reset.updated_at > accepted.updated_at


def test_manifest_invalidation_recomputes_source_and_enforces_run_integrity(tmp_path: Path) -> None:
    index = make_index(tmp_path, [10])
    config = make_config(index.root, tmp_path / "work")
    store = WorkspaceStore(config.work_dir, clock=lambda: NOW)
    manifest = store.initialize(config, index, SHA)
    episode = index.episodes[0]
    video = next(iter(episode.videos.values()))
    video.write_bytes(b"source changed to a different size")
    reset = store.invalidate_episode(0, episode=episode)
    assert reset.source_fingerprint == compute_source_fingerprint(index.root, episode)
    assert reset.run_fingerprint == manifest.run_fingerprint
    assert reset.updated_at > NOW
    with pytest.raises(ValueError, match="run_fingerprint"):
        store.invalidate_episode(0, episode=episode, run_fingerprint="e" * 64)
    with pytest.raises(ValueError, match="caller-provided source"):
        store.invalidate_episode(0, episode=episode, source_fingerprint="f" * 64)
    with pytest.raises(ValueError, match="index"):
        store.invalidate_episode(0, episode=episode.model_copy(update={"episode_index": 1}))
    with pytest.raises(ValueError, match="length"):
        store.invalidate_episode(0, episode=episode.model_copy(update={"length": 11}))


def test_manifest_invalidation_rejects_out_of_range_rogue_record(tmp_path: Path) -> None:
    index = make_index(tmp_path, [10])
    config = make_config(index.root, tmp_path / "work")
    store = WorkspaceStore(config.work_dir, clock=lambda: NOW)
    store.initialize(config, index, SHA)
    payload = store.load_episode(0).model_dump(mode="json") | {"episode_index": 1}
    (store.root / "episodes/episode_000001.json").write_text(json.dumps(payload))
    rogue_episode = index.episodes[0].model_copy(update={"episode_index": 1})
    with pytest.raises(ValueError, match="range"):
        store.invalidate_episode(1, episode=rogue_episode)


def test_source_fingerprint_is_stable_and_tracks_required_dimensions(tmp_path: Path) -> None:
    index = make_index(tmp_path, [10])
    episode = index.episodes[0]
    first = compute_source_fingerprint(index.root, episode)
    assert first == compute_source_fingerprint(index.root, episode)
    video = next(iter(episode.videos.values()))
    stat = video.stat()
    os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    assert compute_source_fingerprint(index.root, episode) != first
    os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    video.write_bytes(b"different-size")
    assert compute_source_fingerprint(index.root, episode) != first
    video.write_bytes(bytes([2]) * 11)
    episode.parquet.write_bytes(b"different")
    assert compute_source_fingerprint(index.root, episode) != first
    changed_length = episode.model_copy(update={"length": 11})
    assert compute_source_fingerprint(index.root, changed_length) != compute_source_fingerprint(index.root, episode)


@pytest.mark.parametrize("kind", ["video", "parquet"])
def test_source_fingerprint_rejects_escape_missing_and_nonfile(tmp_path: Path, kind: str) -> None:
    index = make_index(tmp_path, [10])
    episode = index.episodes[0]
    outside = tmp_path / "outside"
    outside.write_bytes(b"x")
    bad = episode.model_copy(update={"videos": {"cam.eye": outside}}) if kind == "video" else episode.model_copy(update={"parquet": outside})
    with pytest.raises(ValueError, match="dataset root"):
        compute_source_fingerprint(index.root, bad)
    outside.unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        compute_source_fingerprint(index.root, bad)


def test_run_fingerprint_dimensions_and_revision_validation(tmp_path: Path) -> None:
    config = make_config(tmp_path / "source", tmp_path / "work")
    value = compute_run_fingerprint(config, SHA)
    assert value == compute_run_fingerprint(config, SHA)
    assert value != compute_run_fingerprint(config.model_copy(update={"high_level_instruction": "other"}), SHA)
    assert value != compute_run_fingerprint(config, "b" * 40)
    with pytest.raises(ValueError, match="40"):
        compute_run_fingerprint(config, "main")


def test_initialize_layout_manifest_redaction_roundtrip_and_exact_records(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    config = make_config(index.root, tmp_path / "work")
    config = config.model_copy(update={"model": config.model.model_copy(update={"api_key": "TOP-SECRET"})})
    store = WorkspaceStore(config.work_dir, clock=lambda: NOW)
    manifest = store.initialize(config, index, SHA, code_version="0.1-test")
    assert {p.relative_to(store.root).as_posix() for p in store.root.iterdir()} >= {"manifest.json", "episodes", "previews", "logs", "summary.json"}
    assert (store.root / "previews/needs_review").is_dir()
    raw = (store.root / "manifest.json").read_text()
    assert "TOP-SECRET" not in raw and "api_key" not in raw
    assert RunManifest.model_validate_json(raw) == manifest
    assert manifest.dataset_root == index.root.resolve()
    assert manifest.dataset_version == "v2.1" and manifest.total_episodes == 2 and manifest.total_frames == 22
    assert manifest.prompt_version == PROMPT_VERSION and manifest.model_revision == SHA
    for episode in index.episodes:
        record = store.load_episode(episode.episode_index)
        assert record.status == "pending" and record.run_fingerprint == manifest.run_fingerprint
        assert record.source_fingerprint == compute_source_fingerprint(index.root, episode)


def test_manifest_strips_all_endpoint_credentials_query_and_fragment(tmp_path: Path) -> None:
    index = make_index(tmp_path, [10])
    config = make_config(index.root, tmp_path / "work")
    secret_endpoint = "http://alice:password@[::1]:8123/v1/chat?token=SECRET&api_key=SECRET2#fragment"
    secret_model = config.model.__class__.model_validate(
        config.model.model_dump() | {"endpoint": secret_endpoint, "api_key": "SECRET3"}
    )
    config = config.model_copy(
        update={"model": secret_model}
    )
    store = WorkspaceStore(config.work_dir, clock=lambda: NOW)
    manifest = store.initialize(config, index, SHA)
    raw = (store.root / "manifest.json").read_text()
    assert all(secret not in raw for secret in ["alice", "password", "SECRET", "SECRET2", "SECRET3", "token", "api_key", "fragment"])
    assert manifest.effective_config["model"]["endpoint"] == "http://[::1]:8123/v1/chat"


def test_initialize_resumes_compatible_and_rejects_incompatible_state(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    config = make_config(index.root, tmp_path / "work")
    store = WorkspaceStore(config.work_dir, clock=lambda: NOW)
    original = store.initialize(config, index, SHA)
    assert store.initialize(config, index, SHA) == original
    changed = config.model_copy(update={"high_level_instruction": "changed"})
    with pytest.raises(ValueError, match="incompatible"):
        store.initialize(changed, index, SHA)
    first_path = store.root / "episodes/episode_000000.json"
    payload = json.loads(first_path.read_text())
    payload["source_fingerprint"] = "f" * 64
    first_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        store.initialize(config, index, SHA)


def test_initialized_store_rejects_invalid_final_annotation(tmp_path: Path) -> None:
    index = make_index(tmp_path, [10])
    config = make_config(index.root, tmp_path / "work")
    store = WorkspaceStore(config.work_dir, clock=lambda: NOW)
    store.initialize(config, index, SHA)
    base = store.load_episode(0)
    coarse = transition(base, "coarse_done", coarse_attempts=[coarse_result()])
    store.save_episode(coarse)
    refined = transition(coarse, "refine_done", refine_attempts=[refine_result()])
    store.save_episode(refined)
    invalid = transition(
        refined, "accepted", final_annotation=FinalAnnotation(start_subtask_index=0, boundaries=[5]),
        decision_source="model", **MODEL_AUDIT,
    )
    with pytest.raises(ValueError, match="final annotation"):
        store.save_episode(invalid)


def test_initialized_store_requires_model_revision_to_match_manifest(tmp_path: Path) -> None:
    index = make_index(tmp_path, [10])
    config = make_config(index.root, tmp_path / "work")
    store = WorkspaceStore(config.work_dir, clock=lambda: NOW)
    store.initialize(config, index, SHA)
    base = store.load_episode(0)
    coarse = transition(
        base,
        "coarse_done",
        coarse_attempts=[CoarseResult(
            start_subtask_index=0,
            observed_subtask_indices=[0],
            coarse_boundaries=[],
            confidence=0.8,
        )],
    )
    store.save_episode(coarse)
    refined = transition(
        coarse,
        "refine_done",
        final_annotation=FinalAnnotation(start_subtask_index=0, boundaries=[]),
    )
    store.save_episode(refined)
    accepted = transition(
        refined,
        "accepted",
        decision_source="model",
        prompt_version=PROMPT_VERSION,
        model_revision="b" * 40,
        sampling_details={"frames": [0, 9]},
    )
    with pytest.raises(ValueError, match="model_revision"):
        store.save_episode(accepted)


def test_initialized_store_requires_model_prompt_to_match_manifest(tmp_path: Path) -> None:
    index = make_index(tmp_path, [10])
    config = make_config(index.root, tmp_path / "work")
    store = WorkspaceStore(config.work_dir, clock=lambda: NOW)
    store.initialize(config, index, SHA)
    base = store.load_episode(0)
    coarse = transition(
        base,
        "coarse_done",
        coarse_attempts=[CoarseResult(
            start_subtask_index=0,
            observed_subtask_indices=[0],
            coarse_boundaries=[],
            confidence=0.8,
        )],
    )
    store.save_episode(coarse)
    refined = transition(
        coarse,
        "refine_done",
        final_annotation=FinalAnnotation(start_subtask_index=0, boundaries=[]),
    )
    store.save_episode(refined)
    accepted = transition(
        refined,
        "accepted",
        decision_source="model",
        prompt_version="historical-coarse-v0/refine-v0",
        model_revision=SHA,
        sampling_details={"frames": [0, 9]},
    )
    with pytest.raises(ValueError, match="prompt_version"):
        store.save_episode(accepted)


def test_resume_revalidates_semantically_accepted_records(tmp_path: Path) -> None:
    index = make_index(tmp_path, [10])
    config = make_config(index.root, tmp_path / "work")
    store = WorkspaceStore(config.work_dir, clock=lambda: NOW)
    store.initialize(config, index, SHA)
    path = store.root / "episodes/episode_000000.json"
    payload = json.loads(path.read_text())
    payload |= {
        "status": "accepted", "final_annotation": {"start_subtask_index": 0, "boundaries": [5]},
        "decision_source": "model", **MODEL_AUDIT,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="final annotation"):
        store.initialize(config, index, SHA)


def test_save_is_atomic_cleans_temp_and_fsyncs_file_and_directory(tmp_path: Path, monkeypatch) -> None:
    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    calls = []
    real_fsync = os.fsync
    monkeypatch.setattr("qwen_annotate.workspace.os.fsync", lambda fd: (calls.append(fd), real_fsync(fd))[1])
    store.save_episode(pending())
    assert len(calls) >= 4
    assert not list(store.root.rglob("*.tmp"))
    loaded = store.load_episode(0)
    assert loaded.episode_index == 0

    def fail_replace(source, target, **kwargs):
        raise OSError("replace failed")
    monkeypatch.setattr("qwen_annotate.workspace.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.save_episode(transition(loaded, "failed", failure_category="test"))
    assert not list(store.root.rglob("*.tmp"))
    assert store.load_episode(0) == loaded


@pytest.mark.parametrize(
    "component",
    ["episodes", "previews", "previews/needs_review", "logs"],
)
def test_layout_rejects_symlink_components_without_touching_target(
    component: str, tmp_path: Path
) -> None:
    root = tmp_path / "work"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("unchanged")
    link = root / component
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        WorkspaceStore(root).create_layout()
    assert sentinel.read_text() == "unchanged"
    assert list(outside.iterdir()) == [sentinel]


@pytest.mark.parametrize("target", ["lock", "manifest", "summary", "episode"])
def test_workspace_rejects_symlink_file_targets_without_touching_outside(
    target: str, tmp_path: Path
) -> None:
    root = tmp_path / "work"
    store = WorkspaceStore(root)
    store.create_layout()
    outside = tmp_path / "outside.json"
    outside.write_text('{"sentinel":"unchanged"}')
    if target == "lock":
        link = root / "logs/workspace.lock"
        link.symlink_to(outside)
        operation = store.summary
    elif target == "manifest":
        index = make_index(tmp_path, [10])
        config = make_config(index.root, root)
        link = root / "manifest.json"
        link.symlink_to(outside)
        operation = lambda: store.initialize(config, index, SHA)
    elif target == "summary":
        link = root / "summary.json"
        link.symlink_to(outside)
        operation = lambda: store.save_episode(pending())
    else:
        link = root / "episodes/episode_000000.json"
        link.symlink_to(outside)
        operation = lambda: store.save_episode(pending())
    with pytest.raises((ValueError, OSError), match="symlink|follow"):
        operation()
    assert outside.read_text() == '{"sentinel":"unchanged"}'


def test_episode_commit_survives_summary_failure_and_summary_is_recoverable(tmp_path: Path, monkeypatch) -> None:
    import qwen_annotate.workspace as workspace

    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    original = workspace._atomic_json_write

    def fail_only_summary(path, value):
        if path.name == "summary.json":
            raise OSError("summary disk error")
        return original(path, value)

    monkeypatch.setattr(workspace, "_atomic_json_write", fail_only_summary)
    with pytest.raises(OSError, match="summary disk error"):
        store.save_episode(pending())
    committed = store.load_episode(0)
    assert committed.status == "pending"
    monkeypatch.setattr(workspace, "_atomic_json_write", original)
    store.save_episode(committed)
    assert json.loads((store.root / "summary.json").read_text()) == store.summary()


def _fd_count() -> int:
    fd_root = Path("/proc/self/fd")
    if not fd_root.is_dir():
        pytest.skip("fd accounting requires procfs")
    return len(list(fd_root.iterdir()))


def test_lock_fdopen_failure_closes_fd_and_releases_thread_lock(tmp_path: Path, monkeypatch) -> None:
    import qwen_annotate.workspace as workspace

    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    before = _fd_count()
    with monkeypatch.context() as patcher:
        patcher.setattr(workspace.os, "fdopen", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fdopen failed")))
        with pytest.raises(RuntimeError, match="fdopen failed"):
            store.summary()
    assert _fd_count() == before
    assert store.summary()["total"] == 0


def test_atomic_fdopen_failure_closes_raw_fd_and_cleans_temp(tmp_path: Path, monkeypatch) -> None:
    import qwen_annotate.workspace as workspace

    target = tmp_path / "value.json"
    before = _fd_count()
    with monkeypatch.context() as patcher:
        patcher.setattr(workspace.os, "fdopen", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fdopen failed")))
        with pytest.raises(RuntimeError, match="fdopen failed"):
            workspace._atomic_json_write(target, {"value": 1})
    assert _fd_count() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_json_fstat_failure_closes_raw_fd(tmp_path: Path, monkeypatch) -> None:
    import qwen_annotate.workspace as workspace

    path = tmp_path / "value.json"
    path.write_text('{"value":1}')
    before = _fd_count()
    with monkeypatch.context() as patcher:
        patcher.setattr(workspace.os, "fstat", lambda fd: (_ for _ in ()).throw(OSError("fstat failed")))
        with pytest.raises(ValueError, match="fstat failed"):
            workspace._read_json_object(path)
    assert _fd_count() == before


def test_unlock_failure_closes_handle_and_releases_thread_lock(tmp_path: Path, monkeypatch) -> None:
    import qwen_annotate.workspace as workspace

    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    before = _fd_count()
    original = workspace.fcntl.flock

    def fail_unlock(fd, operation):
        if operation == workspace.fcntl.LOCK_UN:
            raise OSError("unlock failed")
        return original(fd, operation)

    with monkeypatch.context() as patcher:
        patcher.setattr(workspace.fcntl, "flock", fail_unlock)
        with pytest.raises(OSError, match="unlock failed"):
            store.summary()
    assert _fd_count() == before
    assert store.summary()["total"] == 0


def test_summary_has_all_zero_counts_sorted_indices_and_updates_on_save(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    store.save_episode(pending(2))
    store.save_episode(pending(0))
    summary = store.summary()
    assert summary["counts"] == {"pending": 2, "coarse_done": 0, "refine_done": 0, "accepted": 0, "needs_review": 0, "failed": 0}
    assert summary["episode_indices"]["pending"] == [0, 2]
    assert "accepted" not in summary["episode_indices"]
    failed = transition(store.load_episode(2), "failed", failure_category="model")
    store.save_episode(failed)
    assert json.loads((store.root / "summary.json").read_text()) == store.summary()
    assert store.summary()["episode_indices"]["failed"] == [2]


@pytest.mark.parametrize("contents", ["{", "[]", '{"episode_index":0,"episode_index":1}', '{"episode_index":NaN}'])
def test_local_json_rejects_malformed_nonobject_duplicate_and_nonfinite(tmp_path: Path, contents: str) -> None:
    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    (store.root / "episodes/episode_000000.json").write_text(contents)
    with pytest.raises(ValueError, match="episode_000000"):
        store.load_episode(0)


def test_summary_rejects_bad_filenames_identity_and_manifest_range(tmp_path: Path) -> None:
    index = make_index(tmp_path, [10])
    config = make_config(index.root, tmp_path / "work")
    store = WorkspaceStore(config.work_dir, clock=lambda: NOW)
    store.initialize(config, index, SHA)
    (store.root / "episodes/bad.json").write_text("{}")
    with pytest.raises(ValueError, match="filename"):
        store.summary()
    (store.root / "episodes/bad.json").unlink()
    extra = pending(1, run=store.load_episode(0).run_fingerprint)
    (store.root / "episodes/episode_000001.json").write_text(extra.model_dump_json())
    with pytest.raises(ValueError, match="range"):
        store.summary()


def test_index_validation_cache_validity_and_fingerprint_identity(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path / "work")
    store.create_layout()
    with pytest.raises((ValueError, TypeError)):
        store.load_episode(-1)
    record = pending()
    assert store.cache_is_valid(record, source_fingerprint=FP, run_fingerprint="c" * 64)
    assert not store.cache_is_valid(record, source_fingerprint="d" * 64, run_fingerprint="c" * 64)
    store.save_episode(record)
    changed = transition(record, "failed", failure_category="x", source_fingerprint="d" * 64)
    with pytest.raises(ValueError, match="fingerprint"):
        store.save_episode(changed)


def test_concurrent_writers_do_not_lose_records_or_summary(tmp_path: Path) -> None:
    root = tmp_path / "work"
    WorkspaceStore(root).create_layout()
    barrier = Barrier(3)
    errors = []
    def worker(index: int) -> None:
        try:
            barrier.wait()
            WorkspaceStore(root).save_episode(pending(index))
        except Exception as exc:
            errors.append(exc)
    threads = [Thread(target=worker, args=(index,)) for index in (0, 1)]
    for thread in threads: thread.start()
    barrier.wait()
    for thread in threads: thread.join()
    assert errors == []
    assert WorkspaceStore(root).summary()["counts"]["pending"] == 2


def test_initialize_does_not_write_source_dataset(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    before = {p.relative_to(index.root): (p.read_bytes(), p.stat().st_mtime_ns) for p in index.root.rglob("*") if p.is_file()}
    config = make_config(index.root, tmp_path / "work")
    WorkspaceStore(config.work_dir, clock=lambda: NOW).initialize(config, index, SHA)
    after = {p.relative_to(index.root): (p.read_bytes(), p.stat().st_mtime_ns) for p in index.root.rglob("*") if p.is_file()}
    assert after == before


@pytest.mark.parametrize("workspace_kind", ["equal", "nested"])
def test_initialize_rejects_workspace_inside_source_before_any_write(
    workspace_kind: str, tmp_path: Path
) -> None:
    index = make_index(tmp_path)
    work = index.root if workspace_kind == "equal" else index.root / "annotations/work"
    config = make_config(index.root, work)
    before = {
        path.relative_to(index.root).as_posix(): (
            "dir" if path.is_dir() else path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in index.root.rglob("*")
    }
    with pytest.raises(ValueError, match="workspace.*dataset"):
        WorkspaceStore(work).initialize(config, index, SHA)
    after = {
        path.relative_to(index.root).as_posix(): (
            "dir" if path.is_dir() else path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in index.root.rglob("*")
    }
    assert after == before


def test_dataset_config_mismatch_is_rejected_before_workspace_write(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    work = tmp_path / "work"
    config = make_config(tmp_path / "different-source", work)
    with pytest.raises(ValueError, match="incompatible"):
        WorkspaceStore(work).initialize(config, index, SHA)
    assert not work.exists()


def test_manifest_inputs_are_validated_before_workspace_write(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    work = tmp_path / "work"
    config = make_config(index.root, work)
    with pytest.raises(ValidationError, match="code_version"):
        WorkspaceStore(work).initialize(config, index, SHA, code_version="")
    assert not work.exists()

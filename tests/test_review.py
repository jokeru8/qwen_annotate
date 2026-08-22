import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from qwen_annotate.config import AnnotationConfig
from qwen_annotate.lerobot import DatasetIndex, EpisodeInfo
from qwen_annotate.models import CoarseBoundary, CoarseResult, FinalAnnotation, RefineResult, ValidationIssue
from qwen_annotate.review import HumanDecision, ReviewServices, apply_human_decision, render_review_site
from qwen_annotate.video import FrameSample
from qwen_annotate.workspace import EpisodeRecord, WorkspaceStore


NOW = datetime(2026, 8, 22, tzinfo=UTC)


def _workspace(tmp_path: Path, *, mode: str = "dagger_patch", reasons=None, issues=None):
    source = tmp_path / "source"
    source.mkdir()
    parquet = source / "episode.parquet"
    parquet.write_bytes(b"parquet")
    videos = {}
    for camera in ("cam.eye", "cam/wrist"):
        path = source / f"{camera.replace('/', '_')}.mp4"
        path.write_bytes(b"video")
        videos[camera] = path
    episode = EpisodeInfo(episode_index=0, length=240, task="task", parquet=parquet, videos=videos)
    dataset = DatasetIndex(root=source, version="v2.1", fps=20.0, camera_keys=list(videos), episodes=[episode])
    config = AnnotationConfig.model_validate({
        "source": source, "work_dir": tmp_path / "work", "mode": mode,
        "high_level_instruction": "Arrange <script>alert(1)</script>",
        "primary_camera": "cam.eye", "refine_cameras": ["cam/wrist", "cam.eye"],
        "subtasks": [
            {"skill": "pick", "text": "Pick <img src=x onerror=alert(2)>"},
            {"skill": "place", "text": "Place"},
            {"skill": "finish", "text": "Finish"},
        ],
        "sampling": {"coarse_fps": 1, "coarse_max_frames": 8, "refine_window_seconds": 1,
                     "refine_fps": 5, "dense_radius_seconds": .5,
                     "agreement_tolerance_frames": 3, "min_segment_frames": 8},
        "model": {"api_key": "TOP-SECRET"},
    })
    store = WorkspaceStore(config.work_dir, clock=lambda: NOW)
    store.initialize(config, dataset, "a" * 40)
    prior = store.load_episode(0)
    review_time = NOW.replace(microsecond=1)
    event_payload = {"run_fingerprint": prior.run_fingerprint, "episode": 0,
                     "from_status": "pending", "to_status": "needs_review",
                     "updated_at": review_time.isoformat(), "event": "coarse_review",
                     "category": None, "reasons": ["coarse_sequence_disagreement"]}
    event = {"event_id": hashlib.sha256(json.dumps(event_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
             "timestamp": review_time.isoformat(), "episode": 0, "from_status": "pending",
             "to_status": "needs_review", "event": "coarse_review", "category": None,
             "reasons": ["coarse_sequence_disagreement"]}
    coarse = CoarseResult(
        start_subtask_index=1, observed_subtask_indices=[1, 2], confidence=.4,
        coarse_boundaries=[CoarseBoundary(from_subtask_index=1, to_subtask_index=2,
                                          estimated_frame=184, evidence="cue <b>unsafe</b>")],
        uncertainties=["occluded"],
    )
    refine = RefineResult(from_subtask_index=1, to_subtask_index=2, last_frame_before=182,
                          first_frame_after=183, boundary_frame=183, confidence=.5,
                          visible_cues=["hand releases object"])
    record = EpisodeRecord(
        episode_index=0, status="needs_review", coarse_attempts=[coarse, coarse],
        refine_attempts=[refine], validation_issues=issues or [],
        review_reasons=reasons or ["coarse_sequence_disagreement"],
        source_fingerprint=prior.source_fingerprint, run_fingerprint=prior.run_fingerprint,
        prompt_version="qwen-lerobot-annotation-v1", model_revision="a" * 40,
        sampling_details={"coarse_decision": {"sampled_frame_indices": [[0, 100, 239], [0, 120, 239]],
                                                  "api_key": "NESTED-SECRET"},
                          "refine_decision": {"candidate_annotation": {"start_subtask_index": 1,
                                                                        "boundaries": [184]}},
                          "_pipeline_transition_events": [event]},
        created_at=prior.created_at, updated_at=review_time,
    )
    store.save_episode(record)
    calls = []

    def sample(path, camera, indices, fps):
        calls.append((path, camera, list(indices), fps))
        return [FrameSample(camera_key=camera, frame_index=i, timestamp_seconds=i / fps,
                            jpeg=f"jpeg:{camera}:{i}".encode()) for i in indices]

    services = ReviewServices(inspect_dataset=lambda cfg: dataset, sampler=sample, clock=lambda: NOW.replace(microsecond=2))
    return config.work_dir, source, store, record, services, calls


def test_human_decision_is_strict_and_rejects_extra_or_coercion() -> None:
    good = {"episode_index": 0, "source_fingerprint": "a" * 64,
            "start_subtask_index": 1, "boundaries": [100]}
    assert HumanDecision.model_validate(good).episode_index == 0
    with pytest.raises(ValidationError):
        HumanDecision.model_validate(good | {"episode_index": "0"})
    with pytest.raises(ValidationError):
        HumanDecision.model_validate(good | {"extra": 1})


def test_review_renders_safe_evidence_json_and_exact_aliases(tmp_path: Path) -> None:
    work, _, _, record, services, calls = _workspace(tmp_path)
    page = render_review_site(work, services=services)
    html = page.read_text(encoding="utf-8")
    assert "episode_000000" in html and "coarse_sequence_disagreement" in html
    assert "cue &lt;b&gt;unsafe&lt;/b&gt;" in html and "hand releases object" in html
    assert "0.4" in html and "0.5" in html
    assert "boundary-184-before.jpg" in html and "boundary-184-after.jpg" in html
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert "<b>unsafe</b>" not in html and "TOP-SECRET" not in html and "NESTED-SECRET" not in html
    payload = json.loads((page.parent / "episode_000000.json").read_text())
    assert "NESTED-SECRET" not in json.dumps(payload)
    assert payload["source_fingerprint"] == record.source_fingerprint
    assert len(payload["coarse_attempts"]) == 2 and len(payload["refine_attempts"]) == 1
    assert payload["candidates"] == [183, 184]
    assert (page.parent / "episode_000000" / "boundary-184-before.jpg").read_bytes() == b"jpeg:cam.eye:183"
    assert (page.parent / "episode_000000" / "boundary-184-after.jpg").read_bytes() == b"jpeg:cam.eye:184"
    assert any(camera == "cam/wrist" and indices[0] == 164 and indices[-1] == 204 for _, camera, indices, _ in calls)


def test_empty_review_set_is_clear_and_does_not_sample(tmp_path: Path) -> None:
    work, _, store, record, services, calls = _workspace(tmp_path)
    apply_human_decision(work, 0, FinalAnnotation(start_subtask_index=1, boundaries=[184]),
                         source_fingerprint=record.source_fingerprint, services=services)
    page = render_review_site(work, services=services)
    assert "No episodes need review" in page.read_text()
    assert calls == []


@pytest.mark.parametrize("annotation,code", [
    (FinalAnnotation(start_subtask_index=1, boundaries=[200, 100]), "boundary_order"),
    (FinalAnnotation(start_subtask_index=1, boundaries=[5]), "segment_too_short"),
    (FinalAnnotation(start_subtask_index=3, boundaries=[]), "start_subtask_range"),
])
def test_invalid_human_decision_preserves_authoritative_bytes(tmp_path: Path, annotation, code) -> None:
    work, _, _, record, services, _ = _workspace(tmp_path)
    path = work / "episodes/episode_000000.json"
    before = path.read_bytes()
    with pytest.raises(ValueError, match=code):
        apply_human_decision(work, 0, annotation, source_fingerprint=record.source_fingerprint, services=services)
    assert path.read_bytes() == before


def test_valid_human_decision_accepts_and_audits_without_losing_attempts(tmp_path: Path) -> None:
    work, _, _, record, services, _ = _workspace(tmp_path)
    decision = HumanDecision(episode_index=0, source_fingerprint=record.source_fingerprint,
                             start_subtask_index=1, boundaries=[184])
    accepted = apply_human_decision(work, 0, decision, services=services)
    assert accepted.status == "accepted" and accepted.decision_source == "human"
    assert accepted.final_annotation == FinalAnnotation(start_subtask_index=1, boundaries=[184])
    assert accepted.coarse_attempts == record.coarse_attempts and accepted.refine_attempts == record.refine_attempts
    assert accepted.review_reasons == [] and accepted.validation_issues == []
    audit = accepted.sampling_details["human_decisions"][-1]
    assert audit["prior_reasons"] == ["coarse_sequence_disagreement"]
    assert audit["prior_candidate"] == {"start_subtask_index": 1, "boundaries": [184]}
    assert audit["accepted_annotation"] == {"start_subtask_index": 1, "boundaries": [184]}
    assert accepted.updated_at > record.updated_at
    assert [item["to_status"] for item in accepted.sampling_details["_pipeline_transition_events"]] == ["needs_review", "accepted"]
    with pytest.raises(ValueError, match="needs_review"):
        apply_human_decision(work, 0, decision, services=services)


def test_stale_source_or_symlink_output_aborts_without_writes(tmp_path: Path) -> None:
    work, source, _, record, services, _ = _workspace(tmp_path)
    source.joinpath("cam.eye.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        render_review_site(work, services=services)
    destination = work / "previews/needs_review"
    assert list(destination.iterdir()) == []

    source.joinpath("cam.eye.mp4").write_bytes(b"video")
    destination.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    destination.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        render_review_site(work, services=services)
    assert list(outside.iterdir()) == []


def test_complete_mode_human_decision_uses_same_constraints(tmp_path: Path) -> None:
    work, _, _, record, services, _ = _workspace(tmp_path, mode="complete")
    before = (work / "episodes/episode_000000.json").read_bytes()
    with pytest.raises(ValueError, match="complete_start_index.*complete_boundary_count"):
        apply_human_decision(work, 0, FinalAnnotation(start_subtask_index=1, boundaries=[100]),
                             source_fingerprint=record.source_fingerprint, services=services)
    assert (work / "episodes/episode_000000.json").read_bytes() == before


def test_repeat_render_is_deterministic_and_failed_refresh_keeps_prior_bundle(tmp_path: Path) -> None:
    work, _, _, _, services, _ = _workspace(tmp_path)
    page = render_review_site(work, services=services)
    first = page.read_bytes()
    assert render_review_site(work, services=services).read_bytes() == first

    broken = ReviewServices(inspect_dataset=services.inspect_dataset,
                            sampler=lambda *args: (_ for _ in ()).throw(OSError("decode failed")),
                            clock=services.clock)
    with pytest.raises(OSError, match="decode failed"):
        render_review_site(work, services=broken)
    assert page.read_bytes() == first
    assert not list((work / "previews").glob(".needs_review.staging-*"))


def test_source_change_during_sampling_never_publishes_new_bundle(tmp_path: Path) -> None:
    work, source, _, _, services, _ = _workspace(tmp_path)
    destination = work / "previews/needs_review"
    calls = 0

    def mutating_sampler(path, camera, indices, fps):
        nonlocal calls
        calls += 1
        result = services.sampler(path, camera, indices, fps)
        if calls == 1:
            source.joinpath("cam.eye.mp4").write_bytes(b"mutated-during-sample")
        return result

    changed = ReviewServices(inspect_dataset=services.inspect_dataset, sampler=mutating_sampler,
                             clock=services.clock)
    with pytest.raises(ValueError, match="fingerprint"):
        render_review_site(work, services=changed)
    assert list(destination.iterdir()) == []


def test_human_save_failure_restores_authoritative_episode_bytes(tmp_path: Path, monkeypatch) -> None:
    work, _, _, record, services, _ = _workspace(tmp_path)
    episode_path = work / "episodes/episode_000000.json"
    before = episode_path.read_bytes()
    import qwen_annotate.workspace as workspace_module
    real_write = workspace_module._atomic_json_write
    failed = False

    def fail_summary_once(path, value):
        nonlocal failed
        if path.name == "summary.json" and not failed:
            failed = True
            raise OSError("summary disk failure")
        return real_write(path, value)

    monkeypatch.setattr(workspace_module, "_atomic_json_write", fail_summary_once)
    with pytest.raises(OSError, match="summary disk failure"):
        apply_human_decision(work, 0, FinalAnnotation(start_subtask_index=1, boundaries=[184]),
                             source_fingerprint=record.source_fingerprint, services=services)
    assert episode_path.read_bytes() == before

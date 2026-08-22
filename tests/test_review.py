import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from qwen_annotate.cli import app
from qwen_annotate.config import AnnotationConfig
from qwen_annotate.lerobot import DatasetIndex, EpisodeInfo
from qwen_annotate.models import CoarseBoundary, CoarseResult, FinalAnnotation, RefineResult, ValidationIssue
from qwen_annotate.prompts import PROMPT_VERSION
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
        prompt_version=PROMPT_VERSION, model_revision="a" * 40,
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
            "run_fingerprint": "b" * 64, "mode": "dagger_patch",
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
    assert payload["run_fingerprint"] == record.run_fingerprint and payload["mode"] == "dagger_patch"
    assert f'data-run-fingerprint="{record.run_fingerprint}"' in html
    assert 'data-mode="dagger_patch"' in html
    assert "Content-Security-Policy" in html and "default-src 'none'" in html
    assert '<script src="review.js"></script>' in html
    javascript = (page.parent / "review.js").read_text()
    assert "run_fingerprint" in javascript and "mode:" in javascript
    assert "fetch(" not in javascript and "eval(" not in javascript and "innerHTML" not in javascript
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
                             run_fingerprint=record.run_fingerprint, mode="dagger_patch",
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
    assert audit["run_fingerprint"] == record.run_fingerprint and audit["mode"] == "dagger_patch"
    assert accepted.updated_at > record.updated_at
    assert [item["to_status"] for item in accepted.sampling_details["_pipeline_transition_events"]] == ["needs_review", "accepted"]
    with pytest.raises(ValueError, match="needs_review"):
        apply_human_decision(work, 0, decision, services=services)


@pytest.mark.parametrize("field,value,message", [
    ("mode", "complete", "mode"),
    ("run_fingerprint", "f" * 64, "run fingerprint"),
])
def test_human_decision_replay_context_mismatch_preserves_bytes(tmp_path: Path, field, value, message) -> None:
    work, _, _, record, services, _ = _workspace(tmp_path)
    payload = {"episode_index": 0, "source_fingerprint": record.source_fingerprint,
               "run_fingerprint": record.run_fingerprint, "mode": "dagger_patch",
               "start_subtask_index": 1, "boundaries": [184]}
    payload[field] = value
    decision = HumanDecision.model_validate(payload)
    path = work / "episodes/episode_000000.json"
    before = path.read_bytes()
    with pytest.raises(ValueError, match=message):
        apply_human_decision(work, 0, decision, services=services)
    assert path.read_bytes() == before


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


def test_publish_recovers_owned_backup_left_between_renames(tmp_path: Path) -> None:
    work, _, _, _, services, _ = _workspace(tmp_path)
    page = render_review_site(work, services=services)
    destination = page.parent
    backup = destination.parent / ".needs_review.backup"
    destination.replace(backup)
    assert not destination.exists() and (backup / ".qwen-annotate-review-v1").is_file()
    recovered = render_review_site(work, services=services)
    assert recovered.is_file() and not backup.exists()


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_publish_never_touches_unowned_or_symlink_backup(tmp_path: Path, kind: str) -> None:
    work, _, _, _, services, _ = _workspace(tmp_path)
    backup = work / "previews/.needs_review.backup"
    outside = tmp_path / "outside"
    outside.mkdir()
    if kind == "directory":
        backup.mkdir()
        (backup / "user.txt").write_text("keep")
    else:
        backup.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="backup"):
        render_review_site(work, services=services)
    assert (backup / "user.txt").read_text() == "keep" if kind == "directory" else backup.is_symlink()


def test_publish_staging_rename_failure_rolls_back_live_bundle(tmp_path: Path, monkeypatch) -> None:
    work, _, _, _, services, _ = _workspace(tmp_path)
    page = render_review_site(work, services=services)
    before = page.read_bytes()
    import qwen_annotate.review as review_module
    real_replace = review_module.os.replace

    def fail_staging(source, destination, *args, **kwargs):
        if Path(source).name.startswith(".needs_review.staging-") and Path(destination).name == "needs_review":
            raise OSError("staging rename failed")
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(review_module.os, "replace", fail_staging)
    with pytest.raises(OSError, match="staging rename failed"):
        render_review_site(work, services=services)
    assert page.read_bytes() == before
    assert not (work / "previews/.needs_review.backup").exists()


def test_postcommit_backup_cleanup_failure_still_succeeds_then_recovers(tmp_path: Path, monkeypatch) -> None:
    work, _, _, _, services, _ = _workspace(tmp_path)
    render_review_site(work, services=services)
    import qwen_annotate.review as review_module
    real_rmtree = review_module.shutil.rmtree
    failed = False

    def fail_backup_once(path, *args, **kwargs):
        nonlocal failed
        if Path(path).name == ".needs_review.backup" and not failed:
            failed = True
            raise OSError("cleanup failed")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(review_module.shutil, "rmtree", fail_backup_once)
    page = render_review_site(work, services=services)
    assert page.is_file() and (work / "previews/.needs_review.backup").is_dir()
    monkeypatch.setattr(review_module.shutil, "rmtree", real_rmtree)
    assert render_review_site(work, services=services).is_file()
    assert not (work / "previews/.needs_review.backup").exists()


def test_two_concurrent_renderers_are_serialized(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    work, _, _, _, services, _ = _workspace(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pages = list(pool.map(lambda _: render_review_site(work, services=services), range(2)))
    assert pages[0] == pages[1] and all(page.is_file() for page in pages)
    assert not list((work / "previews").glob(".needs_review.staging-*"))
    assert not (work / "previews/.needs_review.backup").exists()


def test_parent_directory_fsync_failure_rolls_back_existing_live_bundle(tmp_path: Path, monkeypatch) -> None:
    work, _, _, _, services, _ = _workspace(tmp_path)
    page = render_review_site(work, services=services)
    before = page.read_bytes()
    previews = work / "previews"
    import qwen_annotate.review as review_module
    real_fsync_directory = review_module._fsync_directory
    failed = False

    def fail_publish_fsync(path):
        nonlocal failed
        if Path(path) == previews and not failed:
            failed = True
            raise OSError("parent fsync failed")
        return real_fsync_directory(path)

    monkeypatch.setattr(review_module, "_fsync_directory", fail_publish_fsync)
    with pytest.raises(OSError, match="parent fsync failed"):
        render_review_site(work, services=services)
    assert page.read_bytes() == before
    assert not (previews / ".needs_review.backup").exists()


def test_publish_lock_symlink_and_stale_stage_safety(tmp_path: Path) -> None:
    work, _, _, _, services, _ = _workspace(tmp_path)
    previews = work / "previews"
    outside = tmp_path / "outside-lock"
    outside.write_text("keep")
    lock = previews / ".review-publish.lock"
    lock.symlink_to(outside)
    with pytest.raises(ValueError, match="lock"):
        render_review_site(work, services=services)
    assert outside.read_text() == "keep"
    lock.unlink()

    owned = previews / ".needs_review.staging-owned"
    owned.mkdir()
    (owned / ".qwen-annotate-review-v1").write_text("owned\n")
    unowned = previews / ".needs_review.staging-user"
    unowned.mkdir()
    (unowned / "keep.txt").write_text("keep")
    assert render_review_site(work, services=services).is_file()
    assert not owned.exists() and (unowned / "keep.txt").read_text() == "keep"


def _tamper_record_context(work: Path, **updates) -> bytes:
    path = work / "episodes/episode_000000.json"
    payload = json.loads(path.read_text())
    payload.update(updates)
    payload["sampling_details"].pop("_pipeline_transition_events", None)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return path.read_bytes()


def test_render_stale_record_context_does_not_sample_or_replace_live(tmp_path: Path) -> None:
    work, _, _, record, services, calls = _workspace(tmp_path)
    page = render_review_site(work, services=services)
    live_before = page.read_bytes()
    call_count = len(calls)
    _tamper_record_context(work, run_fingerprint="f" * 64)
    with pytest.raises(ValueError, match="run fingerprint"):
        render_review_site(work, services=services)
    assert page.read_bytes() == live_before and len(calls) == call_count


def test_edited_decision_cannot_bypass_stale_record_run_context(tmp_path: Path) -> None:
    work, _, _, record, services, _ = _workspace(tmp_path)
    before = _tamper_record_context(work, run_fingerprint="f" * 64)
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({
        "episode_index": 0, "source_fingerprint": record.source_fingerprint,
        "run_fingerprint": record.run_fingerprint, "mode": "dagger_patch",
        "start_subtask_index": 1, "boundaries": [184],
    }))
    result = CliRunner().invoke(app, ["review", str(work), "--apply", str(decision)])
    assert result.exit_code == 1 and "Traceback" not in result.output
    assert (work / "episodes/episode_000000.json").read_bytes() == before


def test_direct_annotation_rejects_stale_record_context_without_write(tmp_path: Path) -> None:
    work, _, _, record, services, _ = _workspace(tmp_path)
    before = _tamper_record_context(work, run_fingerprint="f" * 64)
    with pytest.raises(ValueError, match="record run fingerprint"):
        apply_human_decision(work, 0, FinalAnnotation(start_subtask_index=1, boundaries=[184]),
                             source_fingerprint=record.source_fingerprint, services=services)
    assert (work / "episodes/episode_000000.json").read_bytes() == before


@pytest.mark.parametrize("updates,message", [
    ({"prompt_version": "old-prompt"}, "prompt_version"),
    ({"model_revision": "b" * 40}, "model_revision"),
])
def test_present_model_provenance_must_match_manifest(tmp_path: Path, updates, message) -> None:
    work, _, _, _, services, calls = _workspace(tmp_path)
    _tamper_record_context(work, **updates)
    with pytest.raises(ValueError, match=message):
        render_review_site(work, services=services)
    assert calls == []


def test_model_derived_review_requires_complete_manifest_provenance(tmp_path: Path) -> None:
    work, _, _, _, services, calls = _workspace(tmp_path)
    _tamper_record_context(work, prompt_version=None, model_revision=None)
    with pytest.raises(ValueError, match="model-derived.*provenance"):
        render_review_site(work, services=services)
    assert calls == []


def test_manual_review_without_model_provenance_remains_renderable(tmp_path: Path) -> None:
    work, _, _, _, services, _ = _workspace(tmp_path)
    path = work / "episodes/episode_000000.json"
    payload = json.loads(path.read_text())
    payload.update({"coarse_attempts": [], "refine_attempts": [],
                    "prompt_version": None, "model_revision": None})
    payload["sampling_details"] = {}
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    assert render_review_site(work, services=services).is_file()

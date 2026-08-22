from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qwen_annotate.cli import app
from qwen_annotate.config import AnnotationConfig
from qwen_annotate.constraints import DETERMINISTIC_REJECTION_REASONS
from qwen_annotate.evaluation import (
    DaggerPrediction,
    DaggerView,
    evaluate_boundaries,
    evaluate_dagger,
    make_dagger_views,
)
from qwen_annotate.lerobot import EpisodeInfo, VideoProbe
from qwen_annotate.workspace import compute_run_fingerprint, compute_source_fingerprint
from tests.fixtures import make_lerobot_fixture


@pytest.fixture(autouse=True)
def _probe_fixture_videos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qwen_annotate.evaluation.probe_video",
        lambda _: VideoProbe(frames=300, fps=20, width=16, height=16),
        raising=False,
    )


def test_boundary_metrics_use_transition_alignment_and_nearest_rank_p90() -> None:
    predicted = {0: [102, 198], 1: [120, 240]}
    golden = {0: [100, 200], 1: [110, 250]}
    statuses = {0: "accepted", 1: "needs_review"}

    metrics = evaluate_boundaries(predicted, golden, statuses, fps=20)

    assert metrics.accepted_coverage == 0.5
    assert metrics.needs_review_rate == 0.5
    assert metrics.failed_rate == 0.0
    assert metrics.boundary_count == 4
    assert metrics.median_absolute_error_frames == 6.0
    assert metrics.p90_absolute_error_frames == 10.0
    assert metrics.median_absolute_error_seconds == 0.3
    assert metrics.p90_absolute_error_seconds == 0.5


def test_start_accuracy_and_false_accepts_have_episode_denominators() -> None:
    metrics = evaluate_boundaries(
        {0: [100], 1: [240]},
        {0: [100], 1: [200], 2: [300]},
        {0: "accepted", 1: "accepted", 2: "failed"},
        fps=20,
        predicted_start_indices={0: 0, 1: 2},
        golden_start_indices={0: 0, 1: 1, 2: 1},
    )

    assert metrics.episode_count == 3
    assert metrics.predicted_episode_count == 2
    assert metrics.start_index_evaluated_count == 2
    assert metrics.start_subtask_index_accuracy == 0.5
    assert metrics.false_accept_count == 1
    assert metrics.missing_prediction_count == 1


def test_misaligned_transition_counts_are_violations_and_never_silently_zipped() -> None:
    metrics = evaluate_boundaries(
        {0: [10]}, {0: [10, 20]}, {0: "needs_review"}, fps=10,
    )
    assert metrics.aligned_boundary_count == 0
    assert metrics.transition_mismatch_count == 1
    assert metrics.constraint_violation_count == 1
    assert metrics.constraint_violation_blocked_count == 1
    assert metrics.constraint_blocking_rate == 1.0
    assert metrics.median_absolute_error_frames is None


@pytest.mark.parametrize("fps", [0, -1, float("inf"), float("nan"), True])
def test_invalid_fps_is_rejected(fps: object) -> None:
    with pytest.raises(ValueError, match="fps"):
        evaluate_boundaries({}, {}, {}, fps=fps)  # type: ignore[arg-type]


def test_empty_evaluation_has_explicit_vacuous_rates() -> None:
    metrics = evaluate_boundaries({}, {}, {}, fps=20)
    assert metrics.episode_count == 0
    assert metrics.accepted_coverage == 0.0
    assert metrics.start_subtask_index_accuracy is None
    assert metrics.median_absolute_error_frames is None
    assert metrics.constraint_blocking_rate == 1.0


def test_obvious_error_threshold_is_seconds_and_configurable() -> None:
    default = evaluate_boundaries({0: [121]}, {0: [100]}, {0: "accepted"}, fps=20)
    relaxed = evaluate_boundaries(
        {0: [121]}, {0: [100]}, {0: "accepted"}, fps=20,
        obvious_error_threshold_seconds=2.0,
    )
    assert default.false_accept_count == 1
    assert relaxed.false_accept_count == 0


def test_synthetic_dagger_views_are_deterministic_bounded_and_relative() -> None:
    kwargs = dict(
        golden_boundaries={0: [100, 220, 360]},
        episode_lengths={0: 500},
        min_segment_frames=20,
    )
    views = make_dagger_views(**kwargs)

    assert views == make_dagger_views(**kwargs)
    assert DaggerView(
        source_episode=0,
        start_frame=50,
        end_frame=500,
        expected_start_subtask_index=0,
        expected_boundaries_relative=[50, 170, 310],
        kind="suffix",
    ) in views
    assert DaggerView(
        source_episode=0,
        start_frame=50,
        end_frame=80,
        expected_start_subtask_index=0,
        expected_boundaries_relative=[],
        kind="singleton",
    ) in views
    for view in views:
        assert 0 <= view.start_frame < view.end_frame <= 500
        assert view.end_frame - view.start_frame >= 20
        assert all(0 < boundary < view.end_frame - view.start_frame for boundary in view.expected_boundaries_relative)
    assert not any(
        view.kind == "singleton" and view.expected_start_subtask_index == 3
        for view in views
    )


def test_short_segments_are_skipped_without_invalid_dagger_views() -> None:
    assert make_dagger_views(
        {0: [10]}, {0: 20}, min_segment_frames=8,
    ) == []


def test_evaluate_dagger_passes_only_frame_ranges_to_sampler_and_inference() -> None:
    view = DaggerView(
        source_episode=4, start_frame=30, end_frame=90,
        expected_start_subtask_index=2, expected_boundaries_relative=[], kind="singleton",
    )
    seen: list[object] = []

    def sampler(episode: int, start: int, end: int) -> object:
        seen.append((episode, start, end))
        return {"frames": [start, end - 1]}

    def inference(evidence: object) -> DaggerPrediction:
        seen.append(evidence)
        return DaggerPrediction(start_subtask_index=2, boundaries=[], status="accepted")

    metrics = evaluate_dagger([view], sampler=sampler, inference=inference, fps=30)
    assert seen == [(4, 30, 90), {"frames": [30, 89]}]
    assert metrics.start_subtask_index_accuracy == 1.0
    assert metrics.accepted_coverage == 1.0


@pytest.mark.parametrize(
    "start,boundaries",
    [
        (2, [60, 30]),  # reversed
        (2, [30, 100]),  # out of the virtual range
        (2, [30]),  # wrong transition count
        (1, [30, 60]),  # wrong starting subtask
    ],
)
def test_dagger_evaluation_independently_blocks_structurally_invalid_predictions(
    start: int, boundaries: list[int],
) -> None:
    view = DaggerView(
        source_episode=0, start_frame=50, end_frame=150,
        expected_start_subtask_index=2, expected_boundaries_relative=[30, 60], kind="suffix",
    )

    metrics = evaluate_dagger(
        [view], sampler=lambda *_: object(),
        inference=lambda _: DaggerPrediction(
            start_subtask_index=start, boundaries=boundaries, status="accepted",
            constraint_violated=False,
        ),
        fps=20, min_segment_frames=8, obvious_error_threshold_seconds=100.0,
    )

    assert metrics.constraint_violation_count == 1
    assert metrics.constraint_violation_blocked_count == 0
    assert metrics.constraint_blocking_rate == 0.0
    assert metrics.false_accept_count == 1


def test_dagger_evaluation_derives_minimum_segment_violation() -> None:
    view = DaggerView(
        source_episode=0, start_frame=10, end_frame=110,
        expected_start_subtask_index=1, expected_boundaries_relative=[30, 60], kind="suffix",
    )
    metrics = evaluate_dagger(
        [view], sampler=lambda *_: object(),
        inference=lambda _: DaggerPrediction(
            start_subtask_index=1, boundaries=[4, 60], status="needs_review",
        ),
        fps=20, min_segment_frames=8,
    )
    assert metrics.constraint_violation_count == 1
    assert metrics.constraint_violation_blocked_count == 1


def _write_evaluation_fixture(root: Path, *, boundary: int, status: str = "accepted") -> tuple[Path, Path]:
    work = root / "work"
    golden = root / "golden"
    source = make_lerobot_fixture(
        root, lengths=[300], fps=20, cameras=["observation.images.cam"]
    )
    (source / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "task"}) + "\n", encoding="utf-8"
    )
    (source / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["task"], "length": 300}) + "\n",
        encoding="utf-8",
    )
    (work / "episodes").mkdir(parents=True)
    (work / "logs").mkdir()
    (work / "previews" / "needs_review").mkdir(parents=True)
    (golden / "meta").mkdir(parents=True)
    config = AnnotationConfig.model_validate({
        "source": source, "work_dir": work, "mode": "complete",
        "high_level_instruction": "task", "primary_camera": "observation.images.cam",
        "refine_cameras": ["observation.images.cam"],
        "subtasks": [{"skill": "a", "text": "A"}, {"skill": "b", "text": "B"}],
    })
    revision = "b" * 40
    run_sha = compute_run_fingerprint(config, revision)
    episode_info = EpisodeInfo(
        episode_index=0, length=300, task="task",
        parquet=source / "data/chunk-000/episode_000000.parquet",
        videos={"observation.images.cam": source / "videos/chunk-000/observation.images.cam/episode_000000.mp4"},
    )
    source_sha = compute_source_fingerprint(source, episode_info)
    manifest = {
        "dataset_root": str(source), "dataset_version": "v2.1", "fps": 20.0,
        "camera_keys": ["observation.images.cam"], "total_episodes": 1, "total_frames": 300,
        "episode_lengths": [300], "mode": "complete", "high_level_instruction": "task",
        "subtasks": [{"skill": "a", "text": "A"}, {"skill": "b", "text": "B"}],
        "code_version": "0.1.0", "prompt_version": "v1", "model_repo": config.model.name,
        "model_revision": revision,
        "effective_config": config.model_dump(mode="json", exclude={"model": {"api_key"}}),
        "min_segment_frames": 8, "run_fingerprint": run_sha,
        "created_at": "2026-08-22T00:00:00Z",
    }
    (work / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    episode = {
        "episode_index": 0, "status": status, "coarse_attempts": [], "refine_attempts": [],
        "final_annotation": {"start_subtask_index": 0, "boundaries": [boundary]},
        "validation_issues": [], "review_reasons": [], "failure_category": None,
        "decision_source": "human" if status == "accepted" else None,
        "created_at": "2026-08-22T00:00:00Z", "updated_at": "2026-08-22T00:00:01Z",
        "source_fingerprint": source_sha, "run_fingerprint": run_sha, "prompt_version": None,
        "model_revision": None, "sampling_details": {},
    }
    if status == "needs_review":
        episode["review_reasons"] = ["uncertain"]
    (work / "episodes" / "episode_000000.json").write_text(json.dumps(episode), encoding="utf-8")
    annotations = {
        "source_root": "/old/source", "work_dir": "/old/meta",
        "subtask_template": [{"skill": "a", "text": "A"}, {"skill": "b", "text": "B"}],
        "episodes": {"0": {"episode_index": 0, "boundaries": [100],
                              "high_level_instruction": "task", "saved_at": "2026-01-01T00:00:00Z"}},
        "primary_camera": "observation.images.cam", "updated_at": "2026-01-01T00:00:00Z",
    }
    (golden / "meta" / "lerobot_annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
    (golden / "meta" / "info.json").write_text(json.dumps({"fps": 20}), encoding="utf-8")
    (golden / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["task"], "length": 300}) + "\n", encoding="utf-8",
    )
    return work, golden


def test_cli_evaluate_writes_deterministic_launch_gate_report_without_overwrite(tmp_path: Path) -> None:
    work, golden = _write_evaluation_fixture(tmp_path, boundary=100)
    output = tmp_path / "metrics.json"
    runner = CliRunner()

    result = runner.invoke(app, ["evaluate", str(work), "--golden", str(golden), "--output", str(output)])

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["launch_gates"]["all_passed"] is True
    first_bytes = output.read_bytes()
    again = runner.invoke(app, ["evaluate", str(work), "--golden", str(golden), "--output", str(output)])
    assert again.exit_code == 1
    assert output.read_bytes() == first_bytes


def test_cli_evaluate_exits_failure_when_quality_gate_fails(tmp_path: Path) -> None:
    work, golden = _write_evaluation_fixture(tmp_path, boundary=140)
    output = tmp_path / "metrics.json"
    result = CliRunner().invoke(
        app, ["evaluate", str(work), "--golden", str(golden), "--output", str(output)],
    )
    assert result.exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["launch_gates"]["median_boundary_error_seconds"]["passed"] is False


def test_cli_rejects_misaligned_golden_episode_metadata(tmp_path: Path) -> None:
    work, golden = _write_evaluation_fixture(tmp_path, boundary=100)
    (golden / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["task"], "length": 50}) + "\n", encoding="utf-8",
    )
    output = tmp_path / "metrics.json"
    result = CliRunner().invoke(
        app, ["evaluate", str(work), "--golden", str(golden), "--output", str(output)],
    )
    assert result.exit_code == 1
    assert not output.exists()


@pytest.mark.parametrize("reason", sorted(DETERMINISTIC_REJECTION_REASONS))
def test_complete_evaluation_counts_recorded_pre_final_constraint_failures(
    tmp_path: Path, reason: str,
) -> None:
    work, golden = _write_evaluation_fixture(tmp_path, boundary=100, status="needs_review")
    path = work / "episodes" / "episode_000000.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["final_annotation"] = None
    record["review_reasons"] = [reason]
    path.write_text(json.dumps(record), encoding="utf-8")

    from qwen_annotate.evaluation import evaluate_complete

    metrics = evaluate_complete(work, golden)
    assert metrics.constraint_violation_count == 1
    assert metrics.constraint_violation_blocked_count == 1
    assert metrics.constraint_blocking_rate == 1.0


def test_complete_evaluation_rejects_cross_run_episode_and_cli_writes_no_report(tmp_path: Path) -> None:
    work, golden = _write_evaluation_fixture(tmp_path, boundary=100)
    path = work / "episodes" / "episode_000000.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["run_fingerprint"] = "c" * 64
    path.write_text(json.dumps(record), encoding="utf-8")

    from qwen_annotate.evaluation import evaluate_complete

    with pytest.raises(ValueError, match="run fingerprint"):
        evaluate_complete(work, golden)
    output = tmp_path / "metrics.json"
    result = CliRunner().invoke(
        app, ["evaluate", str(work), "--golden", str(golden), "--output", str(output)],
    )
    assert result.exit_code == 1
    assert not output.exists()


def test_complete_evaluation_rejects_stale_source_episode(tmp_path: Path) -> None:
    work, golden = _write_evaluation_fixture(tmp_path, boundary=100)
    source_video = (
        tmp_path / "source" / "videos/chunk-000/observation.images.cam/episode_000000.mp4"
    )
    source_video.write_bytes(b"changed after annotation")

    from qwen_annotate.evaluation import evaluate_complete

    with pytest.raises(ValueError, match="source fingerprint"):
        evaluate_complete(work, golden)
    output = tmp_path / "stale-source-metrics.json"
    result = CliRunner().invoke(
        app, ["evaluate", str(work), "--golden", str(golden), "--output", str(output)],
    )
    assert result.exit_code == 1
    assert not output.exists()


def test_source_provenance_preserves_hashed_lexical_config_paths(tmp_path: Path) -> None:
    work, golden = _write_evaluation_fixture(tmp_path, boundary=100)
    manifest_path = work / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["effective_config"]["source"] = str(tmp_path / "source" / ".." / "source")
    manifest["effective_config"]["work_dir"] = str(work / ".." / "work")
    config_payload = json.loads(json.dumps(manifest["effective_config"]))
    config_payload["model"]["api_key"] = "local"
    lexical_config = AnnotationConfig.model_validate(config_payload)
    run_sha = compute_run_fingerprint(lexical_config, manifest["model_revision"])
    manifest["run_fingerprint"] = run_sha
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    episode_path = work / "episodes" / "episode_000000.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["run_fingerprint"] = run_sha
    episode_path.write_text(json.dumps(episode), encoding="utf-8")

    from qwen_annotate.evaluation import evaluate_complete

    assert evaluate_complete(work, golden).false_accept_count == 0


def test_complete_evaluation_rejects_dagger_workspace(tmp_path: Path) -> None:
    work, golden = _write_evaluation_fixture(tmp_path, boundary=100)
    path = work / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["mode"] = "dagger_patch"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    from qwen_annotate.evaluation import evaluate_complete

    with pytest.raises(ValueError, match="complete workspace"):
        evaluate_complete(work, golden)


def test_complete_evaluation_rejects_golden_dagger_extension(tmp_path: Path) -> None:
    work, golden = _write_evaluation_fixture(tmp_path, boundary=100)
    path = golden / "meta" / "lerobot_annotations.json"
    annotations = json.loads(path.read_text(encoding="utf-8"))
    annotations["episodes"]["0"]["start_subtask_index"] = 0
    path.write_text(json.dumps(annotations), encoding="utf-8")

    from qwen_annotate.evaluation import evaluate_complete

    with pytest.raises(ValueError, match="complete golden"):
        evaluate_complete(work, golden)

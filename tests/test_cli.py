import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qwen_annotate.cli import app, parse_episode_indices
from qwen_annotate.model_manager import ModelInstall
from datetime import UTC, datetime
from qwen_annotate.workspace import EpisodeRecord, WorkspaceStore
from qwen_annotate.pipeline import WorkspaceSummary
from qwen_annotate.lerobot import DatasetIndex, EpisodeInfo


runner = CliRunner()


@pytest.mark.parametrize("value", ["", "0,", ",0", "0,,1", "-1", "0,0", "true", "0, 1"])
def test_episode_parser_rejects_ambiguous_or_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_episode_indices(value)


def test_help_loads_without_network_or_gpu_work() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "annotate" in result.stdout and "inspect" in result.stdout and "status" in result.stdout


def test_invalid_config_is_usage_error_without_traceback(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("unknown: value\n")
    result = runner.invoke(app, ["inspect", str(config)])
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_model_download_delegates_and_prints_immutable_install(monkeypatch, tmp_path: Path) -> None:
    calls = []
    local = (tmp_path / "model").resolve()

    def download(repo, local_path, revision, *, max_workers):
        calls.append((repo, local_path, revision, max_workers))
        return ModelInstall(repo, "b" * 40, local_path.resolve(), datetime(2026, 8, 22, tzinfo=UTC))

    monkeypatch.setattr("qwen_annotate.cli.download_model", download)
    result = runner.invoke(app, ["model", "download", "--repo", "Qwen/Test", "--local-dir", str(local), "--revision", "release", "--max-workers", "3"])
    assert result.exit_code == 0
    assert calls == [("Qwen/Test", local, "release", 3)]
    assert "b" * 40 in result.stdout and str(local) in result.stdout


def test_status_json_preserves_workspace_counts_and_lists_shape(tmp_path: Path) -> None:
    work = tmp_path / "work"
    store = WorkspaceStore(work)
    store.create_layout()
    store.save_episode(EpisodeRecord(
        episode_index=0, source_fingerprint="a" * 64, run_fingerprint="b" * 64,
        created_at=datetime(2026, 8, 22, tzinfo=UTC), updated_at=datetime(2026, 8, 22, tzinfo=UTC),
    ))
    result = runner.invoke(app, ["status", str(work), "--json"])
    assert result.exit_code == 0
    assert __import__("json").loads(result.stdout) == store.summary()


def test_malformed_yaml_is_invalid_config_exit_two(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("subtasks: [\n")
    result = runner.invoke(app, ["inspect", str(config)])
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_inspect_prints_dataset_metadata(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    episode = EpisodeInfo(episode_index=0, length=12, task="task", parquet=source / "x", videos={"cam.eye": source / "v"})
    dataset = DatasetIndex(root=source, version="v2.1", fps=28.0, camera_keys=["cam.eye"], episodes=[episode])
    monkeypatch.setattr("qwen_annotate.cli._config", lambda path: object())
    monkeypatch.setattr("qwen_annotate.cli.inspect_dataset", lambda config: dataset)
    result = runner.invoke(app, ["inspect", str(tmp_path / "config.yaml")])
    assert result.exit_code == 0
    for expected in ("v2.1", "28.0", "cam.eye", "episodes: 1", "frames: 12", "OK"):
        assert expected in result.stdout


def test_annotate_prints_summary_and_failed_episode_causes_exit_one(monkeypatch, tmp_path: Path) -> None:
    raw = {"total": 1, "counts": {"pending": 0, "coarse_done": 0, "refine_done": 0, "accepted": 0, "needs_review": 0, "failed": 1}, "episode_indices": {"pending": [], "coarse_done": [], "refine_done": [], "needs_review": [], "failed": [0]}}
    monkeypatch.setattr("qwen_annotate.cli._config", lambda path: object())

    async def annotate(config, max_concurrency, episodes):
        assert max_concurrency == 2 and episodes == [0, 3]
        return WorkspaceSummary.from_store_summary(raw)

    monkeypatch.setattr("qwen_annotate.cli.annotate_dataset", annotate)
    result = runner.invoke(app, ["annotate", str(tmp_path / "config.yaml"), "--max-concurrency", "2", "--episodes", "0,3"])
    assert result.exit_code == 1
    assert "failed=1" in result.stdout and "Traceback" not in result.output


def test_malformed_workspace_is_operational_exit_one(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "episodes").mkdir(parents=True)
    result = runner.invoke(app, ["status", str(work), "--json"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output


def test_review_render_delegates_and_prints_page(monkeypatch, tmp_path: Path) -> None:
    page = tmp_path / "work/previews/needs_review/index.html"
    monkeypatch.setattr("qwen_annotate.cli.render_review_site", lambda work: page)
    result = runner.invoke(app, ["review", str(tmp_path / "work")])
    assert result.exit_code == 0
    assert str(page) in result.stdout


def test_review_apply_strictly_loads_and_delegates(monkeypatch, tmp_path: Path) -> None:
    fingerprint = "a" * 64
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps({"episode_index": 3, "source_fingerprint": fingerprint,
                                         "run_fingerprint": "b" * 64, "mode": "dagger_patch",
                                         "start_subtask_index": 1, "boundaries": [50]}))
    calls = []

    def apply(work, episode, decision):
        calls.append((work, episode, decision))
        return type("Accepted", (), {"episode_index": 3})()

    monkeypatch.setattr("qwen_annotate.cli.apply_human_decision", apply)
    result = runner.invoke(app, ["review", str(tmp_path / "work"), "--apply", str(decision_path)])
    assert result.exit_code == 0
    assert calls[0][0] == tmp_path / "work" and calls[0][1] == 3
    assert calls[0][2].source_fingerprint == fingerprint
    assert "accepted episode 3" in result.stdout


@pytest.mark.parametrize("payload", [
    '{"episode_index":0,"episode_index":1,"source_fingerprint":"' + "a" * 64 + '","run_fingerprint":"' + "b" * 64 + '","mode":"dagger_patch","start_subtask_index":0,"boundaries":[]}',
    '{"episode_index":0,"source_fingerprint":"' + "a" * 64 + '","run_fingerprint":"' + "b" * 64 + '","mode":"dagger_patch","start_subtask_index":0,"boundaries":[NaN]}',
    '{"episode_index":"0","source_fingerprint":"' + "a" * 64 + '","run_fingerprint":"' + "b" * 64 + '","mode":"dagger_patch","start_subtask_index":0,"boundaries":[]}',
    '{"episode_index":0,"source_fingerprint":"' + "a" * 64 + '","run_fingerprint":"' + "b" * 64 + '","start_subtask_index":0,"boundaries":[]}',
])
def test_review_apply_invalid_decision_is_usage_exit_two(tmp_path: Path, payload: str) -> None:
    decision = tmp_path / "decision.json"
    decision.write_text(payload)
    result = runner.invoke(app, ["review", str(tmp_path / "work"), "--apply", str(decision)])
    assert result.exit_code == 2 and "Traceback" not in result.output


def test_review_apply_operational_rejection_is_exit_one(monkeypatch, tmp_path: Path) -> None:
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"episode_index": 0, "source_fingerprint": "a" * 64,
                                    "run_fingerprint": "b" * 64, "mode": "dagger_patch",
                                    "start_subtask_index": 0, "boundaries": []}))
    monkeypatch.setattr("qwen_annotate.cli.apply_human_decision",
                        lambda *args: (_ for _ in ()).throw(ValueError("stale fingerprint")))
    result = runner.invoke(app, ["review", str(tmp_path / "work"), "--apply", str(decision)])
    assert result.exit_code == 1 and "Traceback" not in result.output

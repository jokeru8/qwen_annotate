"""Fast command-line regression for the complete annotation workflow."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from qwen_annotate.cli import app
from qwen_annotate.lerobot import inspect_dataset
from qwen_annotate.model_manager import ModelInstall
from qwen_annotate.models import CoarseBoundary, CoarseResult, RefineResult
from qwen_annotate.pipeline import PipelineServices, annotate_dataset
from qwen_annotate.stats import recompute_stats, recompute_video_stats


def _write_video(path: Path, camera_value: int, episode_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=10)
    stream.width, stream.height, stream.pix_fmt = 16, 12, "yuv420p"
    try:
        for frame_index in range(16):
            value = camera_value + episode_index * 4 + (120 if frame_index >= 8 else 0)
            array = np.full((12, 16, 3), value, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _make_source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    (root / "meta").mkdir(parents=True)
    cameras = ("observation.images.right_eye", "observation.images.left_wrist")
    info = {
        "codebase_version": "v2.1",
        "total_episodes": 2,
        "total_frames": 32,
        "total_tasks": 1,
        "chunks_size": 1000,
        "total_chunks": 1,
        "fps": 10,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            **{
                camera: {
                    "dtype": "video",
                    "shape": [12, 16, 3],
                    "info": {"video.fps": 10.0},
                }
                for camera in cameras
            },
        },
        "total_videos": 4,
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "Arrange both drinks."}) + "\n",
        encoding="utf-8",
    )
    episode_rows: list[str] = []
    parquets: list[Path] = []
    video_paths = {camera: [] for camera in cameras}
    for episode_index in range(2):
        episode_rows.append(json.dumps({
            "episode_index": episode_index,
            "tasks": ["Arrange both drinks."],
            "length": 16,
        }))
        parquet = root / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({
            "frame_index": pa.array(range(16), type=pa.int64()),
            "episode_index": pa.array([episode_index] * 16, type=pa.int64()),
            "index": pa.array(range(episode_index * 16, (episode_index + 1) * 16), type=pa.int64()),
            "task_index": pa.array([0] * 16, type=pa.int64()),
            "timestamp": pa.array([frame / 10 for frame in range(16)], type=pa.float32()),
        }), parquet)
        parquets.append(parquet)
        for camera_index, camera in enumerate(cameras):
            video = root / f"videos/chunk-000/{camera}/episode_{episode_index:06d}.mp4"
            _write_video(video, 20 + camera_index * 20, episode_index)
            video_paths[camera].append(video)
    (root / "meta/episodes.jsonl").write_text("\n".join(episode_rows) + "\n", encoding="utf-8")
    stats = recompute_stats(parquets)
    for camera in cameras:
        stats[camera] = recompute_video_stats(video_paths[camera], [16, 16], [12, 16, 3])
    (root / "meta/stats.json").write_text(json.dumps(stats), encoding="utf-8")
    (root / "meta/episodes_stats.jsonl").write_text(
        "\n".join(json.dumps({"episode_index": index, "stats": recompute_stats([parquet])})
                  for index, parquet in enumerate(parquets)) + "\n",
        encoding="utf-8",
    )
    return root


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


class _DeterministicModel:
    def __init__(self) -> None:
        self.coarse_calls = 0
        self.refine_calls = 0
        self.refine_camera_sets: list[set[str]] = []

    async def complete(self, prompt, frames, response_type):
        assert frames and "BEGIN_UNTRUSTED_CONTEXT_JSON" in prompt
        if response_type is CoarseResult:
            self.coarse_calls += 1
            return CoarseResult(
                start_subtask_index=0,
                observed_subtask_indices=[0, 1],
                coarse_boundaries=[CoarseBoundary(
                    from_subtask_index=0,
                    to_subtask_index=1,
                    estimated_frame=8,
                    evidence="The first action ends before the second begins.",
                )],
                confidence=0.99,
            )
        assert response_type is RefineResult
        self.refine_calls += 1
        self.refine_camera_sets.append({frame.camera_key for frame in frames})
        return RefineResult(
            from_subtask_index=0,
            to_subtask_index=1,
            last_frame_before=7,
            first_frame_after=8,
            boundary_frame=8,
            confidence=0.99,
            visible_cues=["The hand begins the second action at frame 8."],
        )


def test_two_episode_two_camera_cli_workflow_is_non_destructive(monkeypatch, tmp_path: Path) -> None:
    """Catches a broken public stage handoff or a stage mutating source payloads."""
    source = _make_source(tmp_path)
    work, output = tmp_path / "work", tmp_path / "annotated"
    config_path = tmp_path / "complete.yaml"
    config_path.write_text(yaml.safe_dump({
        "source": str(source),
        "work_dir": str(work),
        "mode": "complete",
        "high_level_instruction": "Arrange both drinks.",
        "primary_camera": "observation.images.right_eye",
        "refine_cameras": ["observation.images.right_eye", "observation.images.left_wrist"],
        "subtasks": [
            {"skill": "pick", "text": "Pick up the first drink."},
            {"skill": "place", "text": "Place the second drink."},
        ],
        "sampling": {
            "coarse_fps": 2.0,
            "coarse_max_frames": 8,
            "refine_window_seconds": 0.4,
            "refine_fps": 5.0,
            "dense_radius_seconds": 0.2,
            "agreement_tolerance_frames": 1,
            "min_segment_frames": 4,
        },
    }, sort_keys=False), encoding="utf-8")
    fake = _DeterministicModel()
    original_annotate = annotate_dataset

    async def annotate_with_fake(config, max_concurrency, episode_indices):
        services = PipelineServices(
            inspect_dataset=inspect_dataset,
            resolve_model=lambda cfg: ModelInstall(
                cfg.model.name, "a" * 40, cfg.model.local_path.resolve(),
                datetime(2026, 8, 22, tzinfo=UTC),
            ),
            client_factory=lambda cfg: fake,
        )
        return await original_annotate(
            config, max_concurrency, episode_indices, services=services,
        )

    monkeypatch.setattr("qwen_annotate.cli.annotate_dataset", annotate_with_fake)
    runner = CliRunner()
    before = _tree_hash(source)

    inspected = runner.invoke(app, ["inspect", str(config_path)])
    assert inspected.exit_code == 0 and "episodes: 2" in inspected.stdout
    assert "observation.images.right_eye" in inspected.stdout
    annotated = runner.invoke(app, ["annotate", str(config_path), "--max-concurrency", "2"])
    assert annotated.exit_code == 0 and "accepted=2" in annotated.stdout
    status = runner.invoke(app, ["status", str(work), "--json"])
    assert status.exit_code == 0
    assert json.loads(status.stdout)["counts"]["accepted"] == 2
    converted = runner.invoke(app, ["convert", str(work), "--output", str(output)])
    assert converted.exit_code == 0 and "episodes=2" in converted.stdout
    validated = runner.invoke(app, ["validate", str(output), "--source", str(source)])
    assert validated.exit_code == 0 and "validation_level=strict_deep" in validated.stdout

    assert fake.coarse_calls == 4 and fake.refine_calls == 4
    assert fake.refine_camera_sets == [
        {"observation.images.right_eye", "observation.images.left_wrist"}
    ] * 4
    assert _tree_hash(source) == before
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    annotations = json.loads(
        (output / "meta/lerobot_annotations.json").read_text(encoding="utf-8")
    )
    assert info["subtask_template"] == [
        {"skill": "pick", "text": "Pick up the first drink."},
        {"skill": "place", "text": "Place the second drink."},
    ]
    assert set(annotations) == {
        "source_root", "work_dir", "subtask_template", "episodes", "primary_camera", "updated_at",
    }
    assert annotations["episodes"]["0"]["boundaries"] == [8]
    assert set(annotations["episodes"]["0"]) == {
        "episode_index", "boundaries", "high_level_instruction", "saved_at",
    }
    assert (output / "meta/task_info/task_0.json").is_file()

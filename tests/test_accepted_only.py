import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qwen_annotate.config import AnnotationConfig
from qwen_annotate.converter import convert_dataset
from qwen_annotate.lerobot import DatasetIndex, EpisodeInfo, VideoProbe
from qwen_annotate.models import FinalAnnotation
from qwen_annotate.release_validator import validate_release
from qwen_annotate.workspace import EpisodeRecord, WorkspaceStore


NOW = datetime(2026, 8, 22, 12, tzinfo=UTC)


def _mixed_workspace(tmp_path: Path) -> tuple[Path, Path, dict]:
    source, work = tmp_path / "source", tmp_path / "work"
    meta = source / "meta"
    meta.mkdir(parents=True)
    lengths = [12, 7, 9]
    cameras = ["cam.eye", "cam.wrist"]
    info = {
        "codebase_version": "v2.1", "total_episodes": 3, "total_frames": 28,
        "total_tasks": 1, "chunks_size": 1, "total_chunks": 3, "fps": 10,
        "splits": {"train": "0:3"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "action": {"dtype": "float32", "shape": [2]},
            **{camera: {"dtype": "video", "shape": [4, 6, 3]} for camera in cameras},
        },
        "total_videos": 6,
    }
    (meta / "info.json").write_text(json.dumps(info))
    tasks_text = json.dumps({"task_index": 0, "task": "Arrange."}) + "\n"
    (meta / "tasks.jsonl").write_text(tasks_text)
    episode_rows = []
    episodes = []
    offset = 0
    schema_metadata = {b"fixture": b"preserve-me"}
    vector_type = pa.list_(pa.float32(), 2)
    for source_index, length in enumerate(lengths):
        episode_rows.append(json.dumps({"episode_index": source_index, "tasks": ["Arrange."], "length": length}))
        parquet = source / f"data/chunk-{source_index:03d}/episode_{source_index:06d}.parquet"
        parquet.parent.mkdir(parents=True)
        action = [[float(source_index), float(frame)] for frame in range(length)]
        table = pa.table({
            "frame_index": pa.array(range(length), type=pa.int64()),
            "episode_index": pa.array([source_index] * length, type=pa.int64()),
            "index": pa.array(range(offset, offset + length), type=pa.int64()),
            "task_index": pa.array([0] * length, type=pa.int64()),
            "timestamp": pa.array([frame / 10 for frame in range(length)], type=pa.float32()),
            "action": pa.array(action, type=vector_type),
        }).replace_schema_metadata(schema_metadata)
        pq.write_table(table, parquet)
        video_paths = {}
        for camera in cameras:
            video = source / f"videos/chunk-{source_index:03d}/{camera}/episode_{source_index:06d}.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(f"{camera}-source-{source_index}".encode())
            video_paths[camera] = video
        episodes.append(EpisodeInfo(
            episode_index=source_index, length=length, task="Arrange.",
            parquet=parquet, videos=video_paths,
        ))
        offset += length
    (meta / "episodes.jsonl").write_text("\n".join(episode_rows) + "\n")
    # Deliberately stale source stats: accepted-only output must derive data, not trust these.
    (meta / "stats.json").write_text(json.dumps({"action": {"mean": [999, 999]}}))
    (meta / "episodes_stats.jsonl").write_text("\n".join(
        json.dumps({"episode_index": i, "stats": {"action": {"mean": [999, 999]}}}) for i in range(3)
    ) + "\n")

    cfg = AnnotationConfig.model_validate({
        "source": source, "work_dir": work, "mode": "complete",
        "high_level_instruction": "Arrange.", "primary_camera": "cam.eye",
        "refine_cameras": cameras,
        "subtasks": [{"skill": "pick", "text": "Pick."}, {"skill": "place", "text": "Place."}],
        "sampling": {"min_segment_frames": 2},
    })
    dataset = DatasetIndex(
        root=source.resolve(), version="v2.1", fps=10, camera_keys=cameras, episodes=episodes,
    )
    store = WorkspaceStore(work, clock=lambda: NOW)
    store.initialize(cfg, dataset, "a" * 40)
    for source_index in (0, 2):
        pending = store.load_episode(source_index)
        accepted = EpisodeRecord.model_validate(pending.model_dump() | {
            "status": "accepted", "updated_at": NOW,
            "final_annotation": FinalAnnotation(
                start_subtask_index=0, boundaries=[lengths[source_index] // 2],
            ).model_dump(),
            "decision_source": "human",
        })
        (work / f"episodes/episode_{source_index:06d}.json").write_text(accepted.model_dump_json())
    pending = store.load_episode(1)
    review = EpisodeRecord.model_validate(pending.model_dump() | {
        "status": "needs_review", "updated_at": NOW, "review_reasons": ["uncertain"],
    })
    (work / "episodes/episode_000001.json").write_text(review.model_dump_json())

    selected_lengths = {0: 12, 1: 9}
    def probe(path: Path) -> VideoProbe:
        index = int(path.stem.split("_")[-1])
        length = lengths[index] if source in path.parents else selected_lengths[index]
        return VideoProbe(frames=length, fps=10, width=6, height=4)
    services = {
        "probe_video": probe,
        "extract_frames": lambda path, camera, indices, fps: [
            type("S", (), {"frame_index": n, "camera_key": camera})() for n in indices
        ],
    }
    return source, work, services


def test_accepted_only_reindexes_every_reference_and_validates(tmp_path: Path) -> None:
    """Catches gaps/stale indices and payload collisions when a middle episode is rejected."""
    source, work, services = _mixed_workspace(tmp_path)
    source_before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "accepted"

    report = convert_dataset(work, output, accepted_only=True, services=services)

    episodes = [json.loads(line) for line in (output / "meta/episodes.jsonl").read_text().splitlines()]
    assert [row["episode_index"] for row in episodes] == [0, 1]
    assert [row["length"] for row in episodes] == [12, 9]
    second_path = output / "data/chunk-001/episode_000001.parquet"
    second = pq.read_table(second_path)
    assert set(second["episode_index"].to_pylist()) == {1}
    assert second["frame_index"].to_pylist() == list(range(9))
    assert second["index"].to_pylist() == list(range(12, 21))
    assert second["action"].to_pylist() == [[2.0, float(i)] for i in range(9)]
    assert second.schema.metadata == {b"fixture": b"preserve-me"}
    for camera in ("cam.eye", "cam.wrist"):
        assert (output / f"videos/chunk-001/{camera}/episode_000001.mp4").read_bytes() == f"{camera}-source-2".encode()

    info = json.loads((output / "meta/info.json").read_text())
    assert info["splits"] == {"train": "0:2"}
    assert (info["total_episodes"], info["total_frames"], info["total_videos"], info["total_chunks"]) == (2, 21, 4, 2)
    assert info["data_files_size_in_mb"] == pytest.approx(sum(p.stat().st_size for p in (output / "data").rglob("*.parquet")) / 2**20)
    assert info["video_files_size_in_mb"] == pytest.approx(sum(p.stat().st_size for p in (output / "videos").rglob("*.mp4")) / 2**20)
    assert (output / "meta/tasks.jsonl").read_text() == (source / "meta/tasks.jsonl").read_text()

    annotations = json.loads((output / "meta/lerobot_annotations.json").read_text())
    assert set(annotations["episodes"]) == {"0", "1"}
    assert annotations["episodes"]["1"]["episode_index"] == 1
    task_info = json.loads((output / "meta/task_info/task_0.json").read_text())
    assert [row["episode_id"] for row in task_info] == [0, 1]
    episode_stats = [json.loads(line) for line in (output / "meta/episodes_stats.jsonl").read_text().splitlines()]
    assert [row["episode_index"] for row in episode_stats] == [0, 1]
    aggregate = json.loads((output / "meta/stats.json").read_text())
    assert aggregate["episode_index"]["min"] == [0]
    assert aggregate["episode_index"]["max"] == [1]
    assert aggregate["index"]["max"] == [20]
    assert aggregate["action"]["mean"] == pytest.approx([18 / 21, 102 / 21])
    assert all(math.isfinite(value) for feature in aggregate.values() for values in feature.values() for value in values)
    assert report.accepted_only and report.episode_count == 2 and report.frame_count == 21
    assert validate_release(output, services=services).valid
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == source_before


def test_accepted_only_requires_at_least_one_approved_episode(tmp_path: Path) -> None:
    """Catches producing a structurally unusable empty release."""
    _, work, services = _mixed_workspace(tmp_path)
    for index in (0, 2):
        path = work / f"episodes/episode_{index:06d}.json"
        payload = json.loads(path.read_text())
        payload.update(status="needs_review", final_annotation=None, decision_source=None, review_reasons=["uncertain"])
        path.write_text(json.dumps(payload))
    output = tmp_path / "empty"
    with pytest.raises(ValueError, match="accepted"):
        convert_dataset(work, output, accepted_only=True, services=services)
    assert not output.exists()

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qwen_annotate.config import AnnotationConfig
from qwen_annotate.converter import ConversionReport, convert_dataset
from qwen_annotate.lerobot import DatasetIndex, EpisodeInfo, VideoProbe
from qwen_annotate.models import FinalAnnotation
from qwen_annotate.workspace import EpisodeRecord, WorkspaceStore


NOW = datetime(2026, 8, 22, 12, tzinfo=UTC)


def _fixture(tmp_path: Path, *, mode: str = "complete") -> tuple[Path, Path, dict]:
    source, work = tmp_path / "source", tmp_path / "work"
    (source / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1", "total_episodes": 2, "total_frames": 40,
        "total_tasks": 1, "chunks_size": 1000, "total_chunks": 1, "fps": 10,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "cam.eye": {"dtype": "video", "shape": [4, 6, 3], "info": {"video.fps": 10}},
        },
        "total_videos": 2,
        "custom_key": {"preserved": True},
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    (source / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "Arrange."}) + "\n")
    rows = []
    episodes = []
    for i in range(2):
        length = 20
        rows.append(json.dumps({"episode_index": i, "tasks": ["Arrange."], "length": length}))
        parquet = source / f"data/chunk-000/episode_{i:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({
            "frame_index": list(range(length)), "episode_index": [i] * length,
            "index": list(range(i * length, (i + 1) * length)), "task_index": [0] * length,
            "timestamp": [j / 10 for j in range(length)],
        }), parquet)
        video = source / f"videos/chunk-000/cam.eye/episode_{i:06d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes((f"video-{i}").encode())
        episodes.append(EpisodeInfo(episode_index=i, length=length, task="Arrange.", parquet=parquet, videos={"cam.eye": video}))
    (source / "meta/episodes.jsonl").write_text("\n".join(rows) + "\n")
    cfg = AnnotationConfig.model_validate({
        "source": source, "work_dir": work, "mode": mode,
        "high_level_instruction": "Arrange.", "primary_camera": "cam.eye", "refine_cameras": ["cam.eye"],
        "subtasks": [{"skill": "pick", "text": "Pick."}, {"skill": "place", "text": "Place."}],
        "sampling": {"min_segment_frames": 2},
    })
    dataset = DatasetIndex(root=source.resolve(), version="v2.1", fps=10, camera_keys=["cam.eye"], episodes=episodes)
    store = WorkspaceStore(work, clock=lambda: NOW)
    manifest = store.initialize(cfg, dataset, "a" * 40)
    for i in range(2):
        pending = store.load_episode(i)
        annotation = FinalAnnotation(start_subtask_index=(0 if mode == "complete" else i), boundaries=([10] if i == 0 or mode == "complete" else []))
        accepted = EpisodeRecord.model_validate(pending.model_dump() | {
            "status": "accepted", "updated_at": NOW.replace(second=i + 1),
            "final_annotation": annotation.model_dump(), "decision_source": "human",
        })
        (work / f"episodes/episode_{i:06d}.json").write_text(accepted.model_dump_json())
    services = {
        "probe_video": lambda path: VideoProbe(frames=20, fps=10, width=6, height=4),
        "extract_frames": lambda path, camera, indices, fps: [type("S", (), {"frame_index": n, "camera_key": camera})() for n in indices],
    }
    return source, work, services


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_conversion_preserves_payload_and_writes_reference_schema(tmp_path: Path) -> None:
    source, work, services = _fixture(tmp_path)
    output = tmp_path / "annotated"
    report = convert_dataset(work, output, services=services)
    annotations = json.loads((output / "meta/lerobot_annotations.json").read_text())
    info = json.loads((output / "meta/info.json").read_text())
    assert report.episode_count == 2 and report.frame_count == 40
    assert annotations["episodes"]["0"]["boundaries"] == [10]
    assert "start_subtask_index" not in annotations["episodes"]["0"]
    assert annotations["schema_version"] == "1.0" and annotations["mode"] == "complete"
    assert annotations["min_segment_frames"] == 2
    assert info["subtask_template"] == annotations["subtask_template"]
    assert info["custom_key"] == {"preserved": True}
    assert report.payload_files == sorted(report.payload_files)
    for relative in report.payload_files:
        assert _sha(output / relative) == _sha(source / relative)
    assert ConversionReport.model_validate_json(report.model_dump_json()) == report


def test_dagger_serializes_explicit_start_including_singleton(tmp_path: Path) -> None:
    _, work, services = _fixture(tmp_path, mode="dagger_patch")
    output = tmp_path / "annotated"
    convert_dataset(work, output, services=services)
    episodes = json.loads((output / "meta/lerobot_annotations.json").read_text())["episodes"]
    assert episodes["0"]["start_subtask_index"] == 0
    assert episodes["1"]["start_subtask_index"] == 1 and episodes["1"]["boundaries"] == []


@pytest.mark.parametrize("status", ["pending", "coarse_done", "refine_done", "needs_review", "failed"])
def test_conversion_refuses_every_nonaccepted_status_without_output(tmp_path: Path, status: str) -> None:
    _, work, services = _fixture(tmp_path)
    record_path = work / "episodes/episode_000000.json"
    payload = json.loads(record_path.read_text())
    payload["status"] = status
    record_path.write_text(json.dumps(payload))
    output = tmp_path / "out"
    with pytest.raises(Exception):
        convert_dataset(work, output, services=services)
    assert not output.exists()


def test_conversion_refuses_existing_nested_and_accepted_only(tmp_path: Path) -> None:
    source, work, services = _fixture(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        convert_dataset(work, existing, services=services)
    with pytest.raises(ValueError):
        convert_dataset(work, source / "nested", services=services)
    with pytest.raises(NotImplementedError):
        convert_dataset(work, tmp_path / "partial", accepted_only=True, services=services)


def test_source_change_and_copy_validation_failure_leave_no_final_output(tmp_path: Path) -> None:
    source, work, services = _fixture(tmp_path)
    video = source / "videos/chunk-000/cam.eye/episode_000000.mp4"
    video.write_bytes(b"changed-size")
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        convert_dataset(work, output, services=services)
    assert not output.exists()


def test_conversion_refuses_tampered_manifest_provenance(tmp_path: Path) -> None:
    _, work, services = _fixture(tmp_path)
    manifest = work / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["model_repo"] = "attacker/other-model"
    manifest.write_text(json.dumps(payload))
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="provenance"):
        convert_dataset(work, output, services=services)
    assert not output.exists()


def test_concurrent_converters_never_replace_output(tmp_path: Path) -> None:
    _, work, services = _fixture(tmp_path)
    output = tmp_path / "out"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(convert_dataset, work, output, services=services) for _ in range(2)]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except Exception as exc:
            outcomes.append(exc)
    assert sum(isinstance(item, ConversionReport) for item in outcomes) == 1
    assert sum(isinstance(item, FileExistsError) for item in outcomes) == 1


def test_unsafe_source_and_validation_failure_clean_only_staging(tmp_path: Path) -> None:
    source, work, services = _fixture(tmp_path)
    fifo = source / "unsafe.fifo"
    os.mkfifo(fifo)
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        convert_dataset(work, output, services=services)
    assert not output.exists() and not list(tmp_path.glob("out.staging-*"))
    fifo.unlink()
    bad_services = services | {"extract_frames": lambda *args: []}
    with pytest.raises(ValueError, match="preview"):
        convert_dataset(work, output, services=bad_services)
    assert not output.exists() and not list(tmp_path.glob("out.staging-*"))


def test_source_is_byte_and_metadata_unchanged(tmp_path: Path) -> None:
    source, work, services = _fixture(tmp_path)
    before = {path.relative_to(source): (path.read_bytes(), path.stat().st_mtime_ns) for path in source.rglob("*") if path.is_file()}
    convert_dataset(work, tmp_path / "out", services=services)
    after = {path.relative_to(source): (path.read_bytes(), path.stat().st_mtime_ns) for path in source.rglob("*") if path.is_file()}
    assert after == before

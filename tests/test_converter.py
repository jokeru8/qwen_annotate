import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pytest

from robo_annotate.config import AnnotationConfig
from robo_annotate.converter import ConversionReport, convert_dataset
from robo_annotate.lerobot import DatasetIndex, VideoProbe
from robo_annotate.models import FinalAnnotation
from robo_annotate.release_validator import validate_release
from robo_annotate.workspace import EpisodeRecord, WorkspaceStore
from tests.fixtures import make_episode_info


NOW = datetime(2026, 8, 22, 12, tzinfo=UTC)


def _fixture(
    tmp_path: Path, *, mode: str = "complete", legacy_stats: bool = False,
    augmentation: bool = False,
) -> tuple[Path, Path, dict]:
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
        episodes.append(make_episode_info(
            episode_index=i,
            length=length,
            task="Arrange.",
            parquet=parquet,
            videos={"cam.eye": video},
            fps=10.0,
        ))
    (source / "meta/episodes.jsonl").write_text("\n".join(rows) + "\n")
    from robo_annotate.stats import recompute_stats
    aggregate_stats = recompute_stats([episode.data.path for episode in episodes])
    image_values = (
        {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.1,
         "q01": 0.0, "q10": 0.1, "q50": 0.5, "q90": 0.9, "q99": 1.0}
        if legacy_stats else
        {"min": 0.0, "max": 50 / 255, "mean": 25 / 255, "std": 25 / 255,
         "q01": 0.0, "q10": 0.0, "q50": 25 / 255, "q90": 50 / 255, "q99": 50 / 255}
    )
    aggregate_stats["cam.eye"] = {
        metric: [[[value]], [[value]], [[value]]] for metric, value in image_values.items()
    } | {"count": [24 if legacy_stats else 40 * 4 * 6]}
    (source / "meta/stats.json").write_text(json.dumps(aggregate_stats))
    episode_stats = [
        {"episode_index": i, "stats": recompute_stats([episode.data.path])}
        for i, episode in enumerate(episodes)
    ]
    (source / "meta/episodes_stats.jsonl").write_text(
        "\n".join(json.dumps(row) for row in episode_stats) + "\n"
    )
    cfg = AnnotationConfig.model_validate({
        "source": source, "work_dir": work, "mode": mode,
        "high_level_instruction": "Arrange.", "primary_camera": "cam.eye", "refine_cameras": ["cam.eye"],
        "subtasks": [{"skill": "pick", "text": "Pick."}, {"skill": "place", "text": "Place."}],
        "sampling": {"min_segment_frames": 2},
        "augmentation": {"enabled": augmentation},
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
        "iter_video_rgb_frames": lambda path: iter([
            np.full((4, 6, 3), int(path.read_bytes().decode().split("-")[-1]) * 50, dtype=np.uint8)
            for _ in range(20)
        ]),
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
    assert list(annotations) == ["source_root", "work_dir", "subtask_template", "episodes", "primary_camera", "updated_at"]
    assert set(annotations["episodes"]["0"]) == {"episode_index", "boundaries", "high_level_instruction", "saved_at"}
    assert annotations["work_dir"] == str(output.resolve() / "meta")
    assert info["subtask_template"] == annotations["subtask_template"]
    assert info["custom_key"] == {"preserved": True}
    assert report.payload_files == sorted(report.payload_files)
    for relative in report.payload_files:
        assert _sha(output / relative) == _sha(source / relative)
    assert ConversionReport.model_validate_json(report.model_dump_json()) == report
    task_info = json.loads((output / "meta/task_info/task_0.json").read_text())
    assert task_info == [
        {"episode_id": 0, "task_id": 0, "task_name": "Arrange.", "label_info": {"action_config": [
            {"start_frame": 0, "end_frame": 10, "action_text": "Pick.", "skill": "pick"},
            {"start_frame": 10, "end_frame": 20, "action_text": "Place.", "skill": "place"},
        ]}},
        {"episode_id": 1, "task_id": 0, "task_name": "Arrange.", "label_info": {"action_config": [
            {"start_frame": 0, "end_frame": 10, "action_text": "Pick.", "skill": "pick"},
            {"start_frame": 10, "end_frame": 20, "action_text": "Place.", "skill": "place"},
        ]}},
    ]


def test_enabled_augmentation_rewrites_each_selected_episode_subtask(tmp_path: Path) -> None:
    _, work, services = _fixture(tmp_path, augmentation=True)
    calls = []

    def augment(config, episodes):
        calls.append((config, episodes))
        return {
            0: ["Lift the item.", "Set the item down."],
            1: ["Raise the object.", "Put the object in place."],
        }

    services["augment_episodes"] = augment
    output = tmp_path / "augmented"

    convert_dataset(work, output, services=services)

    task_info = json.loads((output / "meta/task_info/task_0.json").read_text())
    assert [
        [action["action_text"] for action in episode["label_info"]["action_config"]]
        for episode in task_info
    ] == [
        ["Lift the item.", "Set the item down."],
        ["Raise the object.", "Put the object in place."],
    ]
    annotations = json.loads((output / "meta/lerobot_annotations.json").read_text())
    assert annotations["augmentation"] == {
        "enabled": True,
        "language": "English",
        "model_repo": "Qwen/Qwen3.8-27B",
        "model_revision": "a" * 40,
        "prompt_version": "subtask-paraphrase-v1",
    }
    assert len(calls) == 1
    requested = calls[0][1]
    assert [item.episode_index for item in requested] == [0, 1]
    assert [[subtask.text for subtask in item.subtasks] for item in requested] == [
        ["Pick.", "Place."],
        ["Pick.", "Place."],
    ]


def test_enabled_augmentation_uses_recorded_qwen_for_each_episode(
    monkeypatch, tmp_path: Path
) -> None:
    _, work, services = _fixture(tmp_path, augmentation=True)
    requests = []
    responses = iter([
        {
            "episode_index": 0,
            "subtasks": [
                {"subtask_index": 0, "text": "Lift the item."},
                {"subtask_index": 1, "text": "Set the item down."},
            ],
        },
        {
            "episode_index": 1,
            "subtasks": [
                {"subtask_index": 0, "text": "Raise the object."},
                {"subtask_index": 1, "text": "Put the object in place."},
            ],
        },
    ])

    class FakeQwenClient:
        def __init__(self, **kwargs):
            assert kwargs == {
                "endpoint": "http://127.0.0.1:8000/v1",
                "api_key": "local",
                "model": "Qwen/Qwen3.8-27B",
            }

        async def complete(self, prompt, frames, response_type):
            requests.append((prompt, frames))
            return response_type.model_validate(next(responses))

        async def aclose(self):
            return None

    monkeypatch.setattr(
        "robo_annotate.augmentation.QwenClient", FakeQwenClient, raising=False
    )
    output = tmp_path / "augmented"

    convert_dataset(work, output, services=services)

    task_info = json.loads((output / "meta/task_info/task_0.json").read_text())
    assert [
        [action["action_text"] for action in episode["label_info"]["action_config"]]
        for episode in task_info
    ] == [
        ["Lift the item.", "Set the item down."],
        ["Raise the object.", "Put the object in place."],
    ]
    assert len(requests) == 2
    assert all(frames == [] for _, frames in requests)
    assert all('"target_language":"English"' in prompt for prompt, _ in requests)


def test_english_augmentation_rejects_non_english_text_without_output(
    tmp_path: Path,
) -> None:
    _, work, services = _fixture(tmp_path, augmentation=True)
    services["augment_episodes"] = lambda config, episodes: {
        0: ["拿起物品。", "放下物品。"],
        1: ["抬起物体。", "把物体放好。"],
    }
    output = tmp_path / "invalid-augmentation"

    with pytest.raises(ValueError, match="augmentation"):
        convert_dataset(work, output, services=services)

    assert not output.exists()


def test_source_change_during_augmentation_is_rejected(tmp_path: Path) -> None:
    source, work, services = _fixture(tmp_path, augmentation=True)

    def augment(config, episodes):
        (source / "videos/chunk-000/cam.eye/episode_000000.mp4").write_bytes(b"video-1")
        return {
            0: ["Lift the item.", "Set the item down."],
            1: ["Raise the object.", "Put the object in place."],
        }

    services["augment_episodes"] = augment
    output = tmp_path / "changed-source"

    with pytest.raises(ValueError, match="source dataset changed"):
        convert_dataset(work, output, services=services)

    assert not output.exists()


def test_model_augmentation_rejects_mismatched_episode_identity(
    monkeypatch, tmp_path: Path
) -> None:
    _, work, services = _fixture(tmp_path, augmentation=True)

    class FakeQwenClient:
        def __init__(self, **kwargs):
            pass

        async def complete(self, prompt, frames, response_type):
            return response_type.model_validate({
                "episode_index": 99,
                "subtasks": [
                    {"subtask_index": 0, "text": "Lift the item."},
                    {"subtask_index": 1, "text": "Set the item down."},
                ],
            })

        async def aclose(self):
            return None

    monkeypatch.setattr("robo_annotate.augmentation.QwenClient", FakeQwenClient)
    output = tmp_path / "mismatched-augmentation"

    with pytest.raises(ValueError, match="indices"):
        convert_dataset(work, output, services=services)

    assert not output.exists()


def test_full_conversion_preserves_legacy_stats_without_pixel_decode(tmp_path: Path) -> None:
    source, work, services = _fixture(tmp_path, legacy_stats=True)
    source_stats = (source / "meta/stats.json").read_bytes()
    services["iter_video_rgb_frames"] = lambda path: (_ for _ in ()).throw(AssertionError("decoded pixels"))
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    assert (output / "meta/stats.json").read_bytes() == source_stats
    with pytest.raises(ValueError, match="stats"):
        validate_release(output, services=services)
    assert validate_release(
        output, source=source, services=services,
        allow_legacy_sampled_image_stats=True, deep_video_stats=False,
    ).validation_level == "source_backed_legacy"


def test_dagger_serializes_explicit_start_including_singleton(tmp_path: Path) -> None:
    _, work, services = _fixture(tmp_path, mode="dagger_patch")
    output = tmp_path / "annotated"
    convert_dataset(work, output, services=services)
    episodes = json.loads((output / "meta/lerobot_annotations.json").read_text())["episodes"]
    assert episodes["0"]["start_subtask_index"] == 0
    assert episodes["1"]["start_subtask_index"] == 1 and episodes["1"]["boundaries"] == []
    task_info = json.loads((output / "meta/task_info/task_0.json").read_text())
    assert task_info[0]["label_info"]["action_config"][0]["action_text"] == "Pick."
    assert task_info[1]["label_info"]["action_config"] == [
        {"start_frame": 0, "end_frame": 20, "action_text": "Place.", "skill": "place"}
    ]


def test_dagger_augmentation_only_requests_each_episode_observed_suffix(
    tmp_path: Path,
) -> None:
    _, work, services = _fixture(
        tmp_path, mode="dagger_patch", augmentation=True
    )

    def augment(config, episodes):
        assert [episode.start_subtask_index for episode in episodes] == [0, 1]
        assert [[subtask.text for subtask in episode.subtasks] for episode in episodes] == [
            ["Pick.", "Place."],
            ["Place."],
        ]
        return {
            0: ["Lift the item.", "Set the item down."],
            1: ["Position the item."],
        }

    services["augment_episodes"] = augment
    output = tmp_path / "dagger-augmented"

    convert_dataset(work, output, services=services)

    task_info = json.loads((output / "meta/task_info/task_0.json").read_text())
    assert task_info[1]["label_info"]["action_config"] == [{
        "start_frame": 0,
        "end_frame": 20,
        "action_text": "Position the item.",
        "skill": "place",
    }]


def test_accepted_only_augmentation_excludes_unselected_episodes(tmp_path: Path) -> None:
    _, work, services = _fixture(tmp_path, augmentation=True)
    record_path = work / "episodes/episode_000000.json"
    record = json.loads(record_path.read_text())
    record.update({
        "status": "needs_review",
        "final_annotation": None,
        "decision_source": None,
        "review_reasons": ["manual_review"],
    })
    record_path.write_text(json.dumps(record))

    def augment(config, episodes):
        assert [episode.episode_index for episode in episodes] == [1]
        return {1: ["Raise the object.", "Put the object in place."]}

    services["augment_episodes"] = augment
    output = tmp_path / "selected-augmented"

    convert_dataset(work, output, accepted_only=True, services=services)

    task_info = json.loads((output / "meta/task_info/task_0.json").read_text())
    assert [action["action_text"] for action in task_info[0]["label_info"]["action_config"]] == [
        "Raise the object.",
        "Put the object in place.",
    ]


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


def test_conversion_refuses_existing_and_nested_outputs(tmp_path: Path) -> None:
    source, work, services = _fixture(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        convert_dataset(work, existing, services=services)
    with pytest.raises(ValueError):
        convert_dataset(work, source / "nested", services=services)
    partial = convert_dataset(work, tmp_path / "partial", accepted_only=True, services=services)
    assert partial.accepted_only and partial.episode_count == 2


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

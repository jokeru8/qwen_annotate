import json
import sys
import types
from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from robo_annotate.lerobot import DatasetIndex, VideoProbe, inspect_dataset, probe_video
from tests.fixtures import make_config, make_lerobot_fixture


def fixed_probe(path: Path) -> VideoProbe:
    return VideoProbe(frames=12, fps=5.0, width=320, height=240)


def metadata(root: Path) -> dict[str, object]:
    return json.loads((root / "meta/info.json").read_text(encoding="utf-8"))


def write_metadata(root: Path, value: dict[str, object]) -> None:
    (root / "meta/info.json").write_text(json.dumps(value), encoding="utf-8")


def test_inspects_valid_v21_dataset_with_resolved_paths(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12, 12], 5.0, ["cam.eye"])

    index = inspect_dataset(make_config(root, tmp_path / "work"), probe=fixed_probe)

    assert index.version == "v2.1"
    assert index.fps == 5.0
    assert index.camera_keys == ["cam.eye"]
    assert [episode.length for episode in index.episodes] == [12, 12]
    assert index.episodes[1].parquet == root / "data/chunk-000/episode_000001.parquet"
    assert index.episodes[0].videos == {
        "cam.eye": root / "videos/chunk-000/cam.eye/episode_000000.mp4"
    }


def test_rejects_missing_refine_camera_with_key(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12], 5.0, ["cam.eye"])

    with pytest.raises(ValueError, match="cam.wrist"):
        inspect_dataset(make_config(root, tmp_path / "work", refine=["cam.wrist"]), fixed_probe)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda root: metadata(root) | {"codebase_version": "v2.0"}, "v2.1"),
        (lambda root: metadata(root) | {"total_episodes": 3}, "total_episodes"),
        (lambda root: metadata(root) | {"total_frames": 99}, "total_frames"),
        (lambda root: metadata(root) | {"total_videos": 99}, "total_videos"),
        (lambda root: metadata(root) | {"chunks_size": 0}, "chunks_size"),
    ],
)
def test_rejects_invalid_info_counts(tmp_path: Path, mutate, message: str) -> None:
    root = make_lerobot_fixture(tmp_path, [12, 12], 5.0, ["cam.eye"])
    write_metadata(root, mutate(root))

    with pytest.raises(ValueError, match=message):
        inspect_dataset(make_config(root, tmp_path / "work"), fixed_probe)


def test_rejects_noncontiguous_episode_indices(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12, 12], 5.0, ["cam.eye"])
    episodes = root / "meta/episodes.jsonl"
    rows = [json.loads(line) for line in episodes.read_text().splitlines()]
    rows[1]["episode_index"] = 2
    episodes.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="contiguous"):
        inspect_dataset(make_config(root, tmp_path / "work"), fixed_probe)


def test_rejects_parquet_row_count_mismatch(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12], 5.0, ["cam.eye"])
    parquet = root / "data/chunk-000/episode_000000.parquet"
    pq.write_table(pa.table({"frame": list(range(11))}), parquet)

    with pytest.raises(ValueError, match="row count"):
        inspect_dataset(make_config(root, tmp_path / "work"), fixed_probe)


def test_rejects_video_frame_mismatch(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12], 5.0, ["cam.eye"])

    with pytest.raises(ValueError, match="frame count"):
        inspect_dataset(
            make_config(root, tmp_path / "work"),
            lambda path: VideoProbe(frames=11, fps=5.0, width=320, height=240),
        )


def test_rejects_video_fps_mismatch(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12], 5.0, ["cam.eye"])

    with pytest.raises(ValueError, match="fps"):
        inspect_dataset(
            make_config(root, tmp_path / "work"),
            lambda path: VideoProbe(frames=12, fps=5.1, width=320, height=240),
        )


def test_rejects_missing_task_reference(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12], 5.0, ["cam.eye"])
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "other"}) + "\n")

    with pytest.raises(ValueError, match="task"):
        inspect_dataset(make_config(root, tmp_path / "work"), fixed_probe)


@pytest.mark.parametrize(
    ("field", "template"),
    [
        ("data_path", "data/constant.parquet"),
        ("video_path", "videos/{episode_index:06d}.mp4"),
        ("data_path", "../outside/episode_{episode_index:06d}.parquet"),
        ("data_path", "/tmp/episode_{episode_index:06d}.parquet"),
        ("data_path", "data/{episode_index.real}.parquet"),
        ("video_path", "videos/{unknown}/episode_{episode_index:06d}.mp4"),
    ],
)
def test_rejects_unsafe_or_incomplete_path_templates(
    tmp_path: Path, field: str, template: str
) -> None:
    root = make_lerobot_fixture(tmp_path, [12], 5.0, ["cam.eye"])
    write_metadata(root, metadata(root) | {field: template})

    with pytest.raises(ValueError, match=field):
        inspect_dataset(make_config(root, tmp_path / "work"), fixed_probe)


def test_rejects_symlink_template_escape(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12], 5.0, ["cam.eye"])
    (root / "escape").symlink_to(tmp_path / "outside", target_is_directory=True)
    write_metadata(root, metadata(root) | {"data_path": "escape/episode_{episode_index:06d}.parquet"})

    with pytest.raises(ValueError, match="data_path"):
        inspect_dataset(make_config(root, tmp_path / "work"), fixed_probe)


def test_rejects_duplicate_resolved_parquet_paths(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12] * 11, 5.0, ["cam.eye"])
    for index in range(10):
        original = root / f"data/chunk-{index // 2:03d}/episode_{index:06d}.parquet"
        alias = root / f"data/chunk-000/episode_{str(index)[:1]}.parquet"
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_bytes(original.read_bytes())
        video = root / f"videos/chunk-000/cam.eye/episode_{index:06d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.touch()
    write_metadata(
        root,
        metadata(root)
        | {
            "chunks_size": 100,
            "total_chunks": 1,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index!s:.1}.parquet",
        },
    )

    with pytest.raises(ValueError, match="duplicate parquet"):
        inspect_dataset(make_config(root, tmp_path / "work"), fixed_probe)


def test_rejects_duplicate_resolved_video_paths(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12], 5.0, ["cam.eye", "cam.wrist"])
    alias = root / "videos/chunk-000/c/episode_000000.mp4"
    alias.parent.mkdir(parents=True)
    alias.touch()
    write_metadata(
        root,
        metadata(root)
        | {"video_path": "videos/chunk-{episode_chunk:03d}/{video_key!s:.1}/episode_{episode_index:06d}.mp4"},
    )

    with pytest.raises(ValueError, match="duplicate video"):
        inspect_dataset(
            make_config(root, tmp_path / "work", refine=["cam.wrist"]), fixed_probe
        )


@pytest.mark.parametrize("bad_fps", [float("nan"), float("inf")])
def test_rejects_nonfinite_info_fps(tmp_path: Path, bad_fps: float) -> None:
    root = make_lerobot_fixture(tmp_path, [12], 5.0, ["cam.eye"])
    write_metadata(root, metadata(root) | {"fps": bad_fps})

    with pytest.raises(ValueError, match="non-finite"):
        inspect_dataset(make_config(root, tmp_path / "work"), fixed_probe)


@pytest.mark.parametrize("bad_fps", [float("nan"), float("inf")])
def test_video_probe_and_dataset_index_reject_nonfinite_fps(bad_fps: float) -> None:
    with pytest.raises(ValidationError):
        VideoProbe(frames=12, fps=bad_fps, width=320, height=240)
    with pytest.raises(ValidationError):
        DatasetIndex(root=Path("source"), version="v2.1", fps=bad_fps, camera_keys=[], episodes=[])


def test_rejects_duplicate_task_indices(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12], 5.0, ["cam.eye"])
    (root / "meta/tasks.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"task_index": 0, "task": "arrange items"}),
                json.dumps({"task_index": 0, "task": "another task"}),
            ]
        )
        + "\n"
    )
    write_metadata(root, metadata(root) | {"total_tasks": 2})

    with pytest.raises(ValueError, match="task_index"):
        inspect_dataset(make_config(root, tmp_path / "work"), fixed_probe)


def test_rejects_noncontiguous_task_indices(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12], 5.0, ["cam.eye"])
    (root / "meta/tasks.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"task_index": 0, "task": "arrange items"}),
                json.dumps({"task_index": 2, "task": "another task"}),
            ]
        )
        + "\n"
    )
    write_metadata(root, metadata(root) | {"total_tasks": 2})

    with pytest.raises(ValueError, match="contiguous"):
        inspect_dataset(make_config(root, tmp_path / "work"), fixed_probe)


def test_rejects_unknown_secondary_episode_task(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12], 5.0, ["cam.eye"])
    episodes = root / "meta/episodes.jsonl"
    row = json.loads(episodes.read_text().strip())
    row["tasks"].append("unknown task")
    episodes.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="unknown task"):
        inspect_dataset(make_config(root, tmp_path / "work"), fixed_probe)


def test_rejects_valid_multi_task_episode_for_singular_interface(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12], 5.0, ["cam.eye"])
    (root / "meta/tasks.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"task_index": 0, "task": "arrange items"}),
                json.dumps({"task_index": 1, "task": "another valid task"}),
            ]
        )
        + "\n"
    )
    episodes = root / "meta/episodes.jsonl"
    row = json.loads(episodes.read_text().strip())
    row["tasks"].append("another valid task")
    episodes.write_text(json.dumps(row) + "\n")
    write_metadata(root, metadata(root) | {"total_tasks": 2})

    with pytest.raises(ValueError, match="exactly one task per episode"):
        inspect_dataset(make_config(root, tmp_path / "work"), fixed_probe)


def test_rejects_duplicate_task_texts_at_different_indices(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12], 5.0, ["cam.eye"])
    (root / "meta/tasks.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"task_index": 0, "task": "arrange items"}),
                json.dumps({"task_index": 1, "task": "arrange items"}),
            ]
        )
        + "\n"
    )
    write_metadata(root, metadata(root) | {"total_tasks": 2})

    with pytest.raises(ValueError, match="duplicate task text"):
        inspect_dataset(make_config(root, tmp_path / "work"), fixed_probe)


@pytest.mark.parametrize("field", ["total_tasks", "total_chunks"])
def test_rejects_task_and_chunk_count_mismatches(tmp_path: Path, field: str) -> None:
    root = make_lerobot_fixture(tmp_path, [12, 12, 12], 5.0, ["cam.eye"])
    write_metadata(root, metadata(root) | {field: 99})

    with pytest.raises(ValueError, match=field):
        inspect_dataset(make_config(root, tmp_path / "work"), fixed_probe)


def test_inspection_does_not_write_to_source(tmp_path: Path) -> None:
    root = make_lerobot_fixture(tmp_path, [12, 12], 5.0, ["cam.eye"])
    before = {path.relative_to(root): path.stat().st_mtime_ns for path in root.rglob("*")}

    inspect_dataset(make_config(root, tmp_path / "work"), fixed_probe)

    after = {path.relative_to(root): path.stat().st_mtime_ns for path in root.rglob("*")}
    assert after == before


def test_probe_video_uses_stream_metadata_and_closes_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = types.SimpleNamespace(
        type="video", average_rate=Fraction(5, 1), frames=12, width=320, height=240
    )
    container = types.SimpleNamespace(streams=[stream], close=lambda: None)
    closed = False

    def close() -> None:
        nonlocal closed
        closed = True

    container.close = close
    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(open=lambda _: container))

    result = probe_video(tmp_path / "video.mp4")

    assert result == VideoProbe(frames=12, fps=5.0, width=320, height=240)
    assert closed


def test_probe_video_decodes_when_stream_frame_count_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = types.SimpleNamespace(
        type="video", average_rate=Fraction(5, 1), frames=0, width=320, height=240
    )
    closed = False

    def close() -> None:
        nonlocal closed
        closed = True

    container = types.SimpleNamespace(streams=[stream], close=close, decode=lambda _: iter(range(12)))
    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(open=lambda _: container))

    result = probe_video(tmp_path / "video.mp4")

    assert result.frames == 12
    assert closed


@pytest.mark.parametrize("mode", ["no_stream", "decode_error"])
def test_probe_video_closes_container_on_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    stream = types.SimpleNamespace(
        type="video", average_rate=Fraction(5, 1), frames=0, width=320, height=240
    )
    closed = False

    def close() -> None:
        nonlocal closed
        closed = True

    container = types.SimpleNamespace(
        streams=[] if mode == "no_stream" else [stream],
        close=close,
        decode=lambda _: (_ for _ in ()).throw(RuntimeError("decode failed")),
    )
    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(open=lambda _: container))

    with pytest.raises((ValueError, RuntimeError)):
        probe_video(tmp_path / "video.mp4")

    assert closed


@pytest.mark.parametrize("bad_fps", [float("nan"), float("inf")])
def test_probe_video_rejects_nonfinite_stream_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_fps: float
) -> None:
    stream = types.SimpleNamespace(
        type="video", average_rate=bad_fps, frames=12, width=320, height=240
    )
    container = types.SimpleNamespace(streams=[stream], close=lambda: None)
    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(open=lambda _: container))

    with pytest.raises(ValueError, match="invalid fps"):
        probe_video(tmp_path / "video.mp4")

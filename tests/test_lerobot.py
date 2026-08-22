import json
import sys
import types
from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qwen_annotate.lerobot import VideoProbe, inspect_dataset, probe_video
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
    container = types.SimpleNamespace(
        streams=[stream], close=lambda: None, decode=lambda _: iter(range(12))
    )
    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(open=lambda _: container))

    result = probe_video(tmp_path / "video.mp4")

    assert result.frames == 12

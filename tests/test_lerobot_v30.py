import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robo_annotate.lerobot import detect_dataset_version, inspect_dataset
from robo_annotate.lerobot_v30 import read_v30_episode_table, read_v30_tasks
from tests.v30_fixtures import make_lerobot_v30_fixture, make_v30_config


def _read_info(root: Path) -> dict[str, object]:
    return json.loads((root / "meta/info.json").read_text(encoding="utf-8"))


def _write_info(root: Path, info: dict[str, object]) -> None:
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")


def _episode_path(root: Path) -> Path:
    return root / "meta/episodes/chunk-000/file-000.parquet"


def _replace_column(table: pa.Table, name: str, values: list[object]) -> pa.Table:
    position = table.schema.get_field_index(name)
    field = table.schema.field(position)
    return table.set_column(position, field, pa.array(values, type=field.type))


def _track_info_target_reads(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> list[Path]:
    reads: list[Path] = []
    original = Path.read_text
    resolved_target = target.resolve()

    def tracked(path: Path, *args, **kwargs) -> str:
        if path.resolve() == resolved_target:
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked)
    return reads


def test_inspects_shared_v30_slices(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))

    assert dataset.version == "v3.0"
    assert [item.length for item in dataset.episodes] == [6, 8, 5]
    assert dataset.episodes[0].data.path == dataset.episodes[2].data.path
    assert dataset.episodes[1].data.dataset_from_index == 6
    main = dataset.episodes[1].videos["observation.images.main"]
    assert main.from_timestamp == pytest.approx(6 / 5)
    assert main.to_timestamp == pytest.approx(14 / 5)
    assert main.path == dataset.episodes[0].videos["observation.images.main"].path


def test_rejects_v30_data_range_length_mismatch(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    position = table.schema.get_field_index("dataset_to_index")
    field = table.schema.field(position)
    table = table.set_column(position, field, pa.array([5, 14, 19], type=field.type))
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"episode 0.*dataset range"):
        inspect_dataset(make_v30_config(root, tmp_path / "work"))


def test_reads_bounded_v30_task_and_episode_metadata(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    ignored = root / "meta/episodes/chunk-invalid/file-000.parquet"
    ignored.parent.mkdir()
    ignored.write_bytes(b"not parquet")

    assert read_v30_tasks(root) == {0: "Arrange the colored blocks."}
    assert read_v30_episode_table(root).num_rows == 3


def test_locates_global_data_indices_inside_each_referenced_shard(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    original_path = root / "data/chunk-000/file-000.parquet"
    original = pq.read_table(original_path)
    pq.write_table(original.slice(0, 6), original_path)
    second_path = root / "data/chunk-000/file-001.parquet"
    pq.write_table(original.slice(6), second_path)
    episode_path = _episode_path(root)
    episodes = pq.read_table(episode_path)
    episodes = _replace_column(episodes, "data/file_index", [0, 1, 1])
    pq.write_table(episodes, episode_path)
    _write_info(root, _read_info(root) | {"total_data_files": 2})

    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))

    assert dataset.episodes[1].data.path == second_path
    assert dataset.episodes[1].data.dataset_from_index == 6
    assert dataset.episodes[2].data.path == second_path
    assert dataset.episodes[2].data.dataset_from_index == 14


def test_rejects_missing_v30_data_shard(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    (root / "data/chunk-000/file-000.parquet").unlink()

    with pytest.raises(FileNotFoundError, match=r"episode 0.*data shard"):
        inspect_dataset(make_v30_config(root, tmp_path / "work"))


@pytest.mark.parametrize(
    ("field", "template"),
    [
        (
            "data_path",
            "data/chunk-{chunk_index:03d}/file-{unknown:03d}-{file_index:03d}.parquet",
        ),
        (
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{episode_index:03d}-{file_index:03d}.mp4",
        ),
        (
            "data_path",
            "../outside/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        ),
        (
            "video_path",
            "other/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        ),
    ],
)
def test_rejects_v30_unsafe_or_unknown_template_fields(
    tmp_path: Path, field: str, template: str
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    _write_info(root, _read_info(root) | {field: template})

    with pytest.raises(ValueError, match=field):
        inspect_dataset(make_v30_config(root, tmp_path / "work"))


def test_rejects_v30_symlink_payload(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    data_path = root / "data/chunk-000/file-000.parquet"
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(data_path.read_bytes())
    data_path.unlink()
    data_path.symlink_to(outside)

    with pytest.raises(ValueError, match=r"data shard.*symbolic link"):
        inspect_dataset(make_v30_config(root, tmp_path / "work"))


def test_rejects_v30_symlink_dataset_root_before_reading_info_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_root = make_lerobot_v30_fixture(tmp_path)
    target = real_root / "meta/info.json"
    alias = tmp_path / "dataset-alias"
    alias.symlink_to(real_root, target_is_directory=True)
    reads = _track_info_target_reads(monkeypatch, target)

    with pytest.raises(ValueError, match=r"dataset root.*symbolic link"):
        inspect_dataset(make_v30_config(alias, tmp_path / "work"))
    with pytest.raises(ValueError, match=r"dataset root.*symbolic link"):
        detect_dataset_version(alias)

    assert reads == []


def test_rejects_v30_symlink_info_before_reading_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    info_path = root / "meta/info.json"
    target = tmp_path / "outside-info.json"
    target.write_bytes(info_path.read_bytes())
    info_path.unlink()
    info_path.symlink_to(target)
    reads = _track_info_target_reads(monkeypatch, target)

    with pytest.raises(ValueError, match=r"info metadata.*symbolic link"):
        inspect_dataset(make_v30_config(root, tmp_path / "work"))
    with pytest.raises(ValueError, match=r"info metadata.*symbolic link"):
        detect_dataset_version(root)

    assert reads == []


def test_rejects_noncontiguous_v30_episode_indices(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    path = _episode_path(root)
    table = _replace_column(pq.read_table(path), "episode_index", [0, 2, 3])
    pq.write_table(table, path)

    with pytest.raises(ValueError, match="Episode indices must be contiguous"):
        inspect_dataset(make_v30_config(root, tmp_path / "work"))


def test_rejects_missing_v30_camera_column(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    path = _episode_path(root)
    table = pq.read_table(path).drop(
        ["videos/observation.images.main/to_timestamp"]
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"missing required.*to_timestamp"):
        inspect_dataset(make_v30_config(root, tmp_path / "work"))


def test_rejects_overlapping_v30_video_ranges(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    path = _episode_path(root)
    table = pq.read_table(path)
    table = _replace_column(
        table,
        "videos/observation.images.main/from_timestamp",
        [0.0, 1.0, 2.8],
    )
    table = _replace_column(
        table,
        "videos/observation.images.main/to_timestamp",
        [1.2, 2.6, 3.8],
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"camera 'observation.images.main'.*overlap"):
        inspect_dataset(make_v30_config(root, tmp_path / "work"))


def test_rejects_mismatched_v30_data_rows(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    frame_indices = table.column("frame_index").to_pylist()
    frame_indices[6] = 7
    table = _replace_column(table, "frame_index", frame_indices)
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"episode 1.*frame_index"):
        inspect_dataset(make_v30_config(root, tmp_path / "work"))


def test_rejects_noninteger_v30_global_data_indices(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    position = table.schema.get_field_index("index")
    table = table.set_column(
        position,
        pa.field("index", pa.float64()),
        pa.array(table.column("index").to_pylist(), type=pa.float64()),
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"episode 0.*index.*integer"):
        inspect_dataset(make_v30_config(root, tmp_path / "work"))


def test_rejects_noncontiguous_v30_task_indices(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    pq.write_table(
        pa.table({"task_index": pa.array([1], type=pa.int64()), "task": ["Arrange the colored blocks."]}),
        root / "meta/tasks.parquet",
    )

    with pytest.raises(ValueError, match=r"task_index.*contiguous"):
        inspect_dataset(make_v30_config(root, tmp_path / "work"))


def test_rejects_v30_video_feature_shape_mismatch(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    info = _read_info(root)
    features = info["features"]
    assert isinstance(features, dict)
    main = features["observation.images.main"]
    assert isinstance(main, dict)
    main["shape"] = [3, 99, 32]
    _write_info(root, info)

    with pytest.raises(ValueError, match=r"observation.images.main.*shape"):
        inspect_dataset(make_v30_config(root, tmp_path / "work"))


def test_rejects_v30_video_range_past_shard_duration(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    path = _episode_path(root)
    table = pq.read_table(path)
    table = _replace_column(
        table,
        "videos/observation.images.main/from_timestamp",
        [0.0, 1.2, 3.2],
    )
    table = _replace_column(
        table,
        "videos/observation.images.main/to_timestamp",
        [1.2, 2.8, 4.2],
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"episode 2.*observation.images.main.*duration"):
        inspect_dataset(make_v30_config(root, tmp_path / "work"))


def test_rejects_unknown_v30_episode_metadata_column(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    path = _episode_path(root)
    table = pq.read_table(path).append_column("unexpected", pa.array([1, 2, 3]))
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"Unexpected episode metadata column"):
        inspect_dataset(make_v30_config(root, tmp_path / "work"))

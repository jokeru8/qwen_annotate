import errno
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from robo_annotate.lerobot import inspect_dataset
from robo_annotate import secure_tree, v30_data_writer, writer_publication
from robo_annotate.v30_data_writer import write_v30_data_subset
from tests.v30_fixtures import (
    make_lerobot_v30_fixture,
    make_v30_config,
    read_v30_info,
    source_tree_digest,
)


_SECOND_TASK = "Stack the colored blocks."


def test_rewrites_selected_shared_parquet_and_compacts_tasks(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"

    result = write_v30_data_subset(
        root,
        staging,
        dataset,
        [0, 2],
        read_v30_info(root),
    )

    table = pq.read_table(result.parquet_files[0])
    assert result.total_frames == 11
    assert table["episode_index"].to_pylist() == [0] * 6 + [1] * 5
    assert table["frame_index"].to_pylist() == list(range(6)) + list(range(5))
    assert table["index"].to_pylist() == list(range(11))
    assert [placement.dataset_from_index for placement in result.placements] == [0, 6]
    assert [placement.dataset_to_index for placement in result.placements] == [6, 11]
    assert result.task_table["task_index"].to_pylist() == list(
        range(result.task_table.num_rows)
    )


def test_locates_global_episode_range_at_local_zero_in_a_later_shard(
    tmp_path: Path,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    _move_final_episode_to_second_data_shard(root)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))

    result = write_v30_data_subset(
        root,
        tmp_path / "staging",
        dataset,
        [2],
        read_v30_info(root),
    )

    output = pq.read_table(result.parquet_files[0])
    assert output["note"].to_pylist() == [
        f"episode 2, frame {index}" for index in range(5)
    ]
    assert output["index"].to_pylist() == list(range(5))


def test_preserves_caller_order_and_compacts_mixed_tasks_by_first_use(
    tmp_path: Path,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    _add_second_task(root, episode_index=2, task=_SECOND_TASK, reverse_task_rows=True)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))

    result = write_v30_data_subset(
        root,
        tmp_path / "staging",
        dataset,
        [2, 0],
        read_v30_info(root),
    )

    output = pq.read_table(result.parquet_files[0])
    assert [placement.source_index for placement in result.placements] == [2, 0]
    assert [placement.tasks for placement in result.placements] == [
        (_SECOND_TASK,),
        ("Arrange the colored blocks.",),
    ]
    assert output["note"].to_pylist() == [
        *(f"episode 2, frame {index}" for index in range(5)),
        *(f"episode 0, frame {index}" for index in range(6)),
    ]
    assert output["task_index"].to_pylist() == [0] * 5 + [1] * 6
    assert result.task_table.to_pylist() == [
        {"task_index": 0, "task": _SECOND_TASK},
        {"task_index": 1, "task": "Arrange the colored blocks."},
    ]


def test_packs_whole_episodes_across_numeric_chunks(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    info = read_v30_info(root)
    source_table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    info["data_files_size_in_mb"] = source_table.slice(0, 6).nbytes / (1024 * 1024)
    info["chunks_size"] = 1

    result = write_v30_data_subset(
        root,
        tmp_path / "staging",
        dataset,
        [0, 2],
        info,
    )

    relative_paths = [
        path.relative_to(tmp_path / "staging").as_posix()
        for path in result.parquet_files
    ]
    assert relative_paths == [
        "data/chunk-000/file-000.parquet",
        "data/chunk-001/file-000.parquet",
    ]
    assert [(item.chunk_index, item.file_index) for item in result.placements] == [
        (0, 0),
        (1, 0),
    ]
    assert [pq.read_table(path).num_rows for path in result.parquet_files] == [6, 5]


def test_allows_one_oversized_episode_without_splitting_it(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    info = read_v30_info(root)
    info["data_files_size_in_mb"] = 1 / (1024 * 1024)

    result = write_v30_data_subset(
        root,
        tmp_path / "staging",
        dataset,
        [1],
        info,
    )

    assert len(result.parquet_files) == 1
    assert pq.read_table(result.parquet_files[0]).num_rows == 8
    assert result.placements[0].length == 8


def test_preserves_ordered_arrow_schema_nullability_and_metadata(
    tmp_path: Path,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    data_path = root / "data/chunk-000/file-000.parquet"
    data = pq.read_table(data_path)
    data_schema = _schema_with_metadata(data.schema, b"data-schema", "episode_index")
    pq.write_table(pa.Table.from_arrays(data.columns, schema=data_schema), data_path)
    tasks_path = root / "meta/tasks.parquet"
    tasks = pq.read_table(tasks_path)
    task_schema = _schema_with_metadata(tasks.schema, b"task-schema", "task_index")
    pq.write_table(pa.Table.from_arrays(tasks.columns, schema=task_schema), tasks_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))

    result = write_v30_data_subset(
        root,
        tmp_path / "staging",
        dataset,
        [0, 2],
        read_v30_info(root),
    )

    assert pq.read_table(result.parquet_files[0]).schema.equals(
        data_schema,
        check_metadata=True,
    )
    assert result.task_table.schema.equals(task_schema, check_metadata=True)
    assert pq.read_table(tmp_path / "staging/meta/tasks.parquet").schema.equals(
        task_schema,
        check_metadata=True,
    )


def test_recomputes_selected_numeric_episode_and_aggregate_stats(
    tmp_path: Path,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))

    result = write_v30_data_subset(
        root,
        tmp_path / "staging",
        dataset,
        [2, 0],
        read_v30_info(root),
    )

    assert result.episode_stats[0]["episode_index"] == {
        "min": [0],
        "max": [0],
        "mean": [0.0],
        "std": [0.0],
        "count": [5],
        "q01": pytest.approx([4e-16], abs=1e-18),
        "q10": pytest.approx([4e-15], abs=1e-18),
        "q50": pytest.approx([2e-14], abs=1e-18),
        "q90": pytest.approx([3.6e-14], abs=1e-18),
        "q99": pytest.approx([3.96e-14], abs=1e-18),
    }
    assert result.episode_stats[1]["episode_index"]["mean"] == [1.0]
    assert result.aggregate_stats["index"]["min"] == [0]
    assert result.aggregate_stats["index"]["max"] == [10]
    assert result.aggregate_stats["index"]["mean"] == pytest.approx([5.0])
    assert result.aggregate_stats["index"]["std"] == pytest.approx([math.sqrt(10.0)])
    assert result.aggregate_stats["index"]["count"] == [11]
    assert tuple(result.aggregate_stats["action"]) == (
        "min",
        "max",
        "mean",
        "std",
        "count",
        "q01",
        "q10",
        "q50",
        "q90",
        "q99",
    )
    assert "note" not in result.aggregate_stats
    assert "language_events" not in result.aggregate_stats


def test_retains_basic_only_stats_profile_declared_by_source(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    stats_path = root / "meta/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for feature_stats in stats.values():
        for metric in list(feature_stats):
            if metric.startswith("q"):
                del feature_stats[metric]
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))

    result = write_v30_data_subset(
        root,
        tmp_path / "staging",
        dataset,
        [0, 2],
        read_v30_info(root),
    )

    assert tuple(result.aggregate_stats["action"]) == (
        "min",
        "max",
        "mean",
        "std",
        "count",
    )
    assert tuple(result.episode_stats[0]["task_index"]) == (
        "min",
        "max",
        "mean",
        "std",
        "count",
    )


def test_all_selected_numeric_stats_match_pinned_fixture_metadata(
    tmp_path: Path,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))

    result = write_v30_data_subset(
        root,
        tmp_path / "staging",
        dataset,
        [0, 1, 2],
        read_v30_info(root),
    )

    expected_aggregate = json.loads(
        (root / "meta/stats.json").read_text(encoding="utf-8")
    )
    expected_episodes = pq.read_table(
        root / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pylist()
    for feature, actual in result.aggregate_stats.items():
        for metric, values in actual.items():
            assert values == pytest.approx(
                expected_aggregate[feature][metric],
                rel=1e-6,
                abs=1e-6,
            )
        for episode_index, episode in result.episode_stats.items():
            for metric, values in episode[feature].items():
                expected = expected_episodes[episode_index][
                    f"stats/{feature}/{metric}"
                ]
                assert values == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_preserves_embedded_image_feature_and_recomputes_rgb_stats(
    tmp_path: Path,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    name = "observation.image.embedded"
    encoded = io.BytesIO()
    Image.new("RGB", (2, 2), (64, 128, 192)).save(encoded, format="PNG")
    data_path = root / "data/chunk-000/file-000.parquet"
    data = pq.read_table(data_path)
    image_type = pa.struct(
        [pa.field("bytes", pa.binary()), pa.field("path", pa.string())]
    )
    data = data.add_column(
        data.schema.get_field_index("timestamp"),
        name,
        pa.array(
            [{"bytes": encoded.getvalue(), "path": "embedded.png"}] * data.num_rows,
            type=image_type,
        ),
    )
    pq.write_table(data, data_path)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    rebuilt_features = {}
    for feature, declaration in info["features"].items():
        if feature == "timestamp":
            rebuilt_features[name] = {
                "dtype": "image",
                "shape": [2, 2, 3],
                "names": None,
            }
        rebuilt_features[feature] = declaration
    info["features"] = rebuilt_features
    info_path.write_text(json.dumps(info), encoding="utf-8")
    stats_path = root / "meta/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats[name] = {
        metric: ([19] if metric == "count" else [[[0.0]], [[0.0]], [[0.0]]])
        for metric in stats["action"]
    }
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))

    result = write_v30_data_subset(
        root,
        tmp_path / "staging",
        dataset,
        [2],
        read_v30_info(root),
    )

    output = pq.read_table(result.parquet_files[0])
    assert output.schema.field(name).type.equals(image_type)
    assert output[name].to_pylist()[0]["bytes"] == encoded.getvalue()
    expected_mean = [
        [[64 / 255]],
        [[128 / 255]],
        [[192 / 255]],
    ]
    np.testing.assert_allclose(result.aggregate_stats[name]["mean"], expected_mean)
    np.testing.assert_allclose(result.episode_stats[0][name]["mean"], expected_mean)
    assert result.aggregate_stats[name]["count"] == [5]
    image_stats_type = pa.list_(pa.list_(pa.list_(pa.float64())))
    episode_stat_column = pa.array(
        [result.episode_stats[0][name]["mean"]],
        type=image_stats_type,
    )
    assert episode_stat_column.type.equals(image_stats_type)


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        (
            "float32",
            [10000, 10001, 9999, 10002, 10003, 9998],
            {
                "min": 9998.0,
                "max": 10003.0,
                "mean": 10000.5,
                "std": 2.8284270763397217,
                "q50": 10000.0009765625,
            },
        ),
        (
            "int16",
            [30000, 30001, 29999, 30002, 30003, 29998],
            {
                "min": 29998.0,
                "max": 30003.0,
                "mean": 30000.5,
                "std": 0.0,
                "q50": 30000.001953125,
            },
        ),
    ],
)
def test_matches_pinned_v061_dtype_promotion_and_cancellation_order(
    tmp_path: Path,
    dtype: str,
    values: list[int],
    expected: dict[str, float],
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    name = "observation.cancellation"
    _add_numeric_feature(root, name, dtype, values)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))

    result = write_v30_data_subset(
        root,
        tmp_path / "staging",
        dataset,
        [0],
        read_v30_info(root),
    )

    stats = result.episode_stats[0][name]
    assert stats["count"] == [6]
    for metric, value in expected.items():
        assert stats[metric] == pytest.approx([value], rel=0.0, abs=1e-9)


def test_preserves_source_bytes_and_reads_parquet_only_through_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    before = source_tree_digest(root)
    actual_read_table = pq.read_table

    def descriptor_only(source: object, *args: object, **kwargs: object) -> pa.Table:
        if isinstance(source, (str, Path)):
            raise AssertionError("source Parquet was reopened by pathname")
        return actual_read_table(source, *args, **kwargs)

    monkeypatch.setattr(v30_data_writer.pq, "read_table", descriptor_only)
    result = write_v30_data_subset(
        root,
        tmp_path / "staging",
        dataset,
        [0, 2],
        read_v30_info(root),
    )

    assert result.total_frames == 11
    assert source_tree_digest(root) == before


@pytest.mark.parametrize("column", ["frame_index", "episode_index", "task_index"])
def test_rejects_changed_source_episode_frame_or_task_facts(
    tmp_path: Path,
    column: str,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    values = table[column].to_pylist()
    values[14] = 99
    position = table.schema.get_field_index(column)
    table = table.set_column(
        position,
        table.schema.field(position),
        pa.array(values, type=table.schema.field(position).type),
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=rf"source {column}|source task facts"):
        write_v30_data_subset(
            root,
            tmp_path / "staging",
            dataset,
            [2],
            read_v30_info(root),
        )


def test_rejects_noninteger_official_source_index_values(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    position = table.schema.get_field_index("frame_index")
    table = table.set_column(
        position,
        pa.field("frame_index", pa.float64()),
        pa.array([float(value) for value in table["frame_index"].to_pylist()]),
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match="source frame_index"):
        write_v30_data_subset(
            root,
            tmp_path / "staging",
            dataset,
            [0],
            read_v30_info(root),
        )


def test_rejects_duplicate_global_index_range_in_shared_shard(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    values = table["index"].to_pylist()
    values[6:12] = list(range(6))
    position = table.schema.get_field_index("index")
    table = table.set_column(
        position,
        table.schema.field(position),
        pa.array(values, type=table.schema.field(position).type),
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match="one exact global index range"):
        write_v30_data_subset(
            root,
            tmp_path / "staging",
            dataset,
            [0],
            read_v30_info(root),
        )


def test_write_failure_leaves_no_published_data_or_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    actual_write_table = pq.write_table
    writes = 0

    def fail_tasks_write(*args: object, **kwargs: object) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise RuntimeError("injected write failure")
        actual_write_table(*args, **kwargs)

    monkeypatch.setattr(v30_data_writer.pq, "write_table", fail_tasks_write)
    with pytest.raises(RuntimeError, match="injected write failure"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert not (staging / "data").exists()
    assert not (staging / "meta/tasks.parquet").exists()


@pytest.mark.parametrize("target_source", [False, True])
def test_rejects_symlinked_staging_root_without_writing_target(
    tmp_path: Path,
    target_source: bool,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    sink = root if target_source else tmp_path / "outside"
    if not target_source:
        sink.mkdir()
        (sink / "marker.txt").write_text("unchanged", encoding="utf-8")
    staging = tmp_path / "staging-link"
    staging.symlink_to(sink, target_is_directory=True)
    before_source = source_tree_digest(root)
    before_sink = _tree_digest(sink)

    with pytest.raises(ValueError, match="symbolic link|real directory"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert source_tree_digest(root) == before_source
    assert _tree_digest(sink) == before_sink


def test_rejects_symlinked_intermediate_staging_component_without_writing_target(
    tmp_path: Path,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    sink = tmp_path / "outside"
    sink.mkdir()
    (sink / "marker.txt").write_text("unchanged", encoding="utf-8")
    alias = tmp_path / "staging-parent-link"
    alias.symlink_to(sink, target_is_directory=True)
    before_source = source_tree_digest(root)
    before_sink = _tree_digest(sink)

    with pytest.raises(ValueError, match="symbolic link|real directory"):
        write_v30_data_subset(
            root,
            alias / "staging",
            dataset,
            [0],
            read_v30_info(root),
        )

    assert source_tree_digest(root) == before_source
    assert _tree_digest(sink) == before_sink


@pytest.mark.parametrize("component", ["meta", "data"])
def test_rejects_symlinked_internal_staging_component_without_partial_output(
    tmp_path: Path,
    component: str,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    staging.mkdir()
    sink = tmp_path / "outside"
    sink.mkdir()
    (sink / "marker.txt").write_text("unchanged", encoding="utf-8")
    (staging / component).symlink_to(sink, target_is_directory=True)
    before_source = source_tree_digest(root)
    before_sink = _tree_digest(sink)

    with pytest.raises(ValueError, match="symbolic link|unsafe staging"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert source_tree_digest(root) == before_source
    assert _tree_digest(sink) == before_sink
    if component == "data":
        assert (staging / "data").is_symlink()
    else:
        assert not (staging / "data").exists()


def test_component_replacement_during_publication_is_detected_and_rolled_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    (staging / "meta").mkdir(parents=True)
    sink = tmp_path / "outside"
    sink.mkdir()
    (sink / "marker.txt").write_text("unchanged", encoding="utf-8")
    before_source = source_tree_digest(root)
    before_sink = _tree_digest(sink)
    actual_publish = writer_publication.rename_noreplace_at
    publications = 0

    def publish_then_swap_meta(
        source_fd: int,
        source_name: str,
        destination_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal publications
        actual_publish(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )
        publications += 1
        if publications == 1:
            os.replace(staging / "meta", staging / "displaced-meta")
            (staging / "meta").symlink_to(sink, target_is_directory=True)

    monkeypatch.setattr(
        writer_publication,
        "rename_noreplace_at",
        publish_then_swap_meta,
    )
    with pytest.raises(ValueError, match="changed during publication"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert source_tree_digest(root) == before_source
    assert _tree_digest(sink) == before_sink
    assert not (staging / "data").exists()
    assert not (staging / "displaced-meta/tasks.parquet").exists()


def test_source_change_after_staging_publication_is_detected_and_rolled_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    info_path = root / "meta/info.json"
    original_info = info_path.read_bytes()
    actual_publish = writer_publication.rename_noreplace_at
    publications = 0

    def publish_then_change_source(
        source_fd: int,
        source_name: str,
        destination_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal publications
        actual_publish(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )
        publications += 1
        if publications == 2:
            info_path.write_bytes(original_info + b" ")

    monkeypatch.setattr(
        writer_publication,
        "rename_noreplace_at",
        publish_then_change_source,
    )
    try:
        with pytest.raises(ValueError, match="source.*changed"):
            write_v30_data_subset(
                root,
                staging,
                dataset,
                [0],
                read_v30_info(root),
            )
        assert not (staging / "data").exists()
        assert not (staging / "meta/tasks.parquet").exists()
    finally:
        info_path.write_bytes(original_info)


@pytest.mark.parametrize("destination", ["data", "tasks.parquet"])
def test_publication_does_not_clobber_destination_created_at_publish_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination: str,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    (staging / "meta").mkdir(parents=True)
    before_source = source_tree_digest(root)
    original_verify = v30_data_writer._ChildDirectory.verify
    meta_verifications = 0
    injected = False
    data_identity: tuple[int, int] | None = None

    def verify_then_create_destination(
        child: object,
        phase: str,
    ) -> None:
        nonlocal meta_verifications, injected, data_identity
        original_verify(child, phase)
        context = getattr(child, "context")
        if context == "staging meta":
            meta_verifications += 1
        if destination == "data" and context == "staging bundle" and not injected:
            (staging / "data").mkdir()
            value = (staging / "data").stat()
            data_identity = value.st_dev, value.st_ino
            injected = True
        if (
            destination == "tasks.parquet"
            and context == "staging meta"
            and meta_verifications == 2
            and not injected
        ):
            (staging / "meta/tasks.parquet").write_bytes(b"competing tasks")
            injected = True

    monkeypatch.setattr(
        v30_data_writer._ChildDirectory,
        "verify",
        verify_then_create_destination,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert injected
    assert source_tree_digest(root) == before_source
    if destination == "data":
        value = (staging / "data").stat()
        assert (value.st_dev, value.st_ino) == data_identity
        assert not (staging / "meta/tasks.parquet").exists()
    else:
        assert (staging / "meta/tasks.parquet").read_bytes() == b"competing tasks"
        assert not (staging / "data").exists()
    assert list(staging.glob(".v30-data-*")) == []


@pytest.mark.parametrize("replacement", ["directory", "symlink"])
def test_replaced_published_data_preserves_non_owned_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    sink = tmp_path / "outside"
    sink.mkdir()
    (sink / "marker.txt").write_text("unchanged", encoding="utf-8")
    before_source = source_tree_digest(root)
    before_sink = _tree_digest(sink)
    actual_entry_at = v30_data_writer._entry_at
    injected = False

    def entry_at_then_replace(parent_fd: int, name: str):
        nonlocal injected
        identity = actual_entry_at(parent_fd, name)
        try:
            parent_path = Path(os.readlink(f"/proc/self/fd/{parent_fd}"))
        except OSError:
            parent_path = Path()
        if (
            not injected
            and name == "data"
            and identity is not None
            and parent_path == staging
        ):
            os.replace(staging / "data", staging / "displaced-data")
            if replacement == "directory":
                (staging / "data").mkdir()
                (staging / "data/intruder.txt").write_text(
                    "preserve competitor",
                    encoding="utf-8",
                )
            else:
                (staging / "data").symlink_to(sink, target_is_directory=True)
            injected = True
        return identity

    monkeypatch.setattr(v30_data_writer, "_entry_at", entry_at_then_replace)

    with pytest.raises(ValueError, match="staging data changed during publication"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert injected
    assert source_tree_digest(root) == before_source
    assert _tree_digest(sink) == before_sink
    if replacement == "directory":
        assert (staging / "data/intruder.txt").read_text(encoding="utf-8") == (
            "preserve competitor"
        )
    else:
        assert (staging / "data").is_symlink()
    assert not (staging / "displaced-data").exists()
    assert not (staging / "meta/tasks.parquet").exists()
    assert list(staging.glob(".v30-data-*")) == []


def test_relocated_staging_is_never_cleaned_through_displaced_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "marker.txt").write_text("unchanged", encoding="utf-8")
    relocated = outside / "relocated-staging"
    before_source = source_tree_digest(root)
    before_fds = len(os.listdir("/proc/self/fd"))
    actual_publish = writer_publication.rename_noreplace_at
    outside_after_injection: str | None = None

    def publish_then_relocate_staging(
        source_fd: int,
        source_name: str,
        destination_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal outside_after_injection
        actual_publish(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )
        if destination_name != "data" or source_name != "data":
            return
        os.replace(staging, relocated)
        staging.mkdir()
        (staging / "replacement-marker.txt").write_text(
            "preserve replacement root",
            encoding="utf-8",
        )
        os.replace(relocated / "data", relocated / "displaced-data")
        (relocated / "data").mkdir()
        (relocated / "data/competitor.txt").write_text(
            "preserve external competitor",
            encoding="utf-8",
        )
        outside_after_injection = _tree_digest(outside)

    monkeypatch.setattr(
        writer_publication,
        "rename_noreplace_at",
        publish_then_relocate_staging,
    )

    with pytest.raises(ValueError, match="staging data changed during publication"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert outside_after_injection is not None
    assert source_tree_digest(root) == before_source
    assert _tree_digest(outside) == outside_after_injection
    assert (relocated / "data/competitor.txt").read_text(encoding="utf-8") == (
        "preserve external competitor"
    )
    assert (relocated / "displaced-data").is_dir()
    assert sorted(path.name for path in staging.iterdir()) == [
        "replacement-marker.txt"
    ]
    assert len(os.listdir("/proc/self/fd")) == before_fds


def test_relocated_meta_is_never_cleaned_through_displaced_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "marker.txt").write_text("unchanged", encoding="utf-8")
    relocated = outside / "relocated-meta"
    before_source = source_tree_digest(root)
    before_fds = len(os.listdir("/proc/self/fd"))
    actual_publish = writer_publication.rename_noreplace_at
    outside_after_injection: str | None = None

    def publish_then_relocate_meta(
        source_fd: int,
        source_name: str,
        destination_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal outside_after_injection
        actual_publish(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )
        if destination_name != "tasks.parquet":
            return
        os.replace(staging / "meta", relocated)
        (staging / "meta").mkdir()
        (staging / "meta/replacement-marker.txt").write_text(
            "preserve replacement meta",
            encoding="utf-8",
        )
        os.replace(
            relocated / "tasks.parquet",
            relocated / "displaced-tasks.parquet",
        )
        (relocated / "tasks.parquet").write_text(
            "preserve external competitor",
            encoding="utf-8",
        )
        outside_after_injection = _tree_digest(outside)

    monkeypatch.setattr(
        writer_publication,
        "rename_noreplace_at",
        publish_then_relocate_meta,
    )

    with pytest.raises(ValueError, match="staging tasks changed during publication"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert outside_after_injection is not None
    assert source_tree_digest(root) == before_source
    assert _tree_digest(outside) == outside_after_injection
    assert (relocated / "tasks.parquet").read_text(encoding="utf-8") == (
        "preserve external competitor"
    )
    assert (relocated / "displaced-tasks.parquet").is_file()
    assert not (staging / "data").exists()
    assert sorted(path.name for path in (staging / "meta").iterdir()) == [
        "replacement-marker.txt"
    ]
    assert list(staging.glob(".v30-data-*")) == []
    assert len(os.listdir("/proc/self/fd")) == before_fds


def test_task_owned_inode_is_removed_after_meta_moves_within_attached_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    displaced_meta = staging / "displaced-meta"
    before_source = source_tree_digest(root)
    actual_publish = writer_publication.rename_noreplace_at

    def publish_then_move_meta_within_staging(
        source_fd: int,
        source_name: str,
        destination_fd: int,
        destination_name: str,
    ) -> None:
        actual_publish(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )
        if destination_name != "tasks.parquet":
            return
        os.replace(staging / "meta", displaced_meta)
        (staging / "meta").mkdir()
        (staging / "meta/replacement-marker.txt").write_text(
            "preserve replacement meta",
            encoding="utf-8",
        )
        os.replace(
            displaced_meta / "tasks.parquet",
            displaced_meta / "task-owned.parquet",
        )
        (displaced_meta / "tasks.parquet").write_text(
            "preserve attached competitor",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        writer_publication,
        "rename_noreplace_at",
        publish_then_move_meta_within_staging,
    )

    with pytest.raises(ValueError, match="staging tasks changed during publication"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert source_tree_digest(root) == before_source
    assert not (staging / "data").exists()
    assert not (displaced_meta / "task-owned.parquet").exists()
    assert (displaced_meta / "tasks.parquet").read_text(encoding="utf-8") == (
        "preserve attached competitor"
    )
    assert (staging / "meta/replacement-marker.txt").read_text(
        encoding="utf-8"
    ) == "preserve replacement meta"
    assert list(staging.glob(".v30-data-*")) == []


def test_renamed_tasks_cleanup_preserves_attached_competitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    before_source = source_tree_digest(root)
    actual_entry_at = v30_data_writer._entry_at
    injected = False

    def entry_at_then_replace_tasks(parent_fd: int, name: str):
        nonlocal injected
        identity = actual_entry_at(parent_fd, name)
        try:
            parent_path = Path(os.readlink(f"/proc/self/fd/{parent_fd}"))
        except OSError:
            parent_path = Path()
        if (
            not injected
            and name == "tasks.parquet"
            and identity is not None
            and parent_path == staging / "meta"
        ):
            os.replace(
                staging / "meta/tasks.parquet",
                staging / "meta/displaced-tasks.parquet",
            )
            (staging / "meta/tasks.parquet").write_text(
                "preserve attached competitor",
                encoding="utf-8",
            )
            injected = True
        return identity

    monkeypatch.setattr(
        v30_data_writer,
        "_entry_at",
        entry_at_then_replace_tasks,
    )

    with pytest.raises(ValueError, match="staging tasks changed during publication"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert injected
    assert source_tree_digest(root) == before_source
    assert not (staging / "data").exists()
    assert (staging / "meta/tasks.parquet").read_text(encoding="utf-8") == (
        "preserve attached competitor"
    )
    assert not (staging / "meta/displaced-tasks.parquet").exists()
    assert list(staging.glob(".v30-data-*")) == []


def test_bundle_open_failure_removes_owned_bundle_and_new_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    before_source = source_tree_digest(root)
    before_fds = len(os.listdir("/proc/self/fd"))
    actual_open = os.open

    def fail_private_bundle_open(path: object, *args: object, **kwargs: object) -> int:
        if isinstance(path, str) and path.startswith(".v30-data-"):
            raise OSError(errno.EMFILE, "injected bundle open failure")
        return actual_open(path, *args, **kwargs)

    monkeypatch.setattr(v30_data_writer.os, "open", fail_private_bundle_open)

    with pytest.raises(ValueError, match="unable to open staging bundle"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert source_tree_digest(root) == before_source
    assert not staging.exists()
    assert len(os.listdir("/proc/self/fd")) == before_fds


def test_staging_open_failure_preserves_replacement_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "marker.txt").write_text("unchanged", encoding="utf-8")
    relocated = outside / "relocated-owned-staging"
    before_source = source_tree_digest(root)
    before_fds = len(os.listdir("/proc/self/fd"))
    actual_publish = writer_publication.rename_noreplace_at
    replacement_identity: tuple[int, int] | None = None

    def replace_published_staging(
        source_fd: int,
        source_name: str,
        destination_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal replacement_identity
        actual_publish(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )
        if destination_name == "staging":
            os.replace(staging, relocated)
            staging.mkdir()
            value = staging.stat()
            replacement_identity = value.st_dev, value.st_ino

    monkeypatch.setattr(
        writer_publication,
        "rename_noreplace_at",
        replace_published_staging,
    )

    with pytest.raises(ValueError, match="unable to anchor staging path"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert replacement_identity is not None
    assert source_tree_digest(root) == before_source
    assert relocated.is_dir()
    value = staging.stat()
    assert (value.st_dev, value.st_ino) == replacement_identity
    assert (outside / "marker.txt").read_text(encoding="utf-8") == "unchanged"
    assert len(os.listdir("/proc/self/fd")) == before_fds


def test_meta_open_failure_preserves_non_owned_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "marker.txt").write_text("unchanged", encoding="utf-8")
    relocated = outside / "relocated-owned-meta"
    before_source = source_tree_digest(root)
    before_fds = len(os.listdir("/proc/self/fd"))
    actual_open = os.open
    injected = False
    replacement_name: str | None = None

    def fail_meta_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal injected, replacement_name
        parent_fd = kwargs.get("dir_fd")
        try:
            parent_path = Path(os.readlink(f"/proc/self/fd/{parent_fd}"))
        except (OSError, TypeError):
            parent_path = Path()
        if (
            isinstance(path, str)
            and path.startswith(".writer-dir-")
            and parent_path == staging
            and not injected
        ):
            replacement_name = path
            os.replace(staging / path, relocated)
            (staging / path).mkdir()
            (staging / path / "competitor.txt").write_bytes(b"preserve competitor")
            injected = True
            raise OSError(errno.EMFILE, "injected meta open failure")
        return actual_open(path, *args, **kwargs)

    monkeypatch.setattr(v30_data_writer.os, "open", fail_meta_open)

    with pytest.raises(ValueError, match="unable to open staging meta"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert injected
    assert replacement_name is not None
    assert source_tree_digest(root) == before_source
    assert relocated.is_dir()
    assert (staging / replacement_name / "competitor.txt").read_bytes() == (
        b"preserve competitor"
    )
    assert (outside / "marker.txt").read_text(encoding="utf-8") == "unchanged"
    assert len(os.listdir("/proc/self/fd")) == before_fds


def test_private_bundle_open_failure_preserves_non_owned_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "marker.txt").write_text("unchanged", encoding="utf-8")
    relocated = outside / "relocated-owned-bundle"
    before_source = source_tree_digest(root)
    before_fds = len(os.listdir("/proc/self/fd"))
    actual_open = os.open
    bundle_name: str | None = None
    injected = False

    def fail_bundle_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal bundle_name, injected
        if (
            isinstance(path, str)
            and path.startswith(".v30-data-")
            and not injected
        ):
            bundle_name = path
            os.replace(staging / path, relocated)
            (staging / path).mkdir()
            (staging / path / "competitor.txt").write_bytes(b"preserve competitor")
            injected = True
            raise OSError(errno.EMFILE, "injected bundle open failure")
        return actual_open(path, *args, **kwargs)

    monkeypatch.setattr(v30_data_writer.os, "open", fail_bundle_open)

    with pytest.raises(ValueError, match="unable to open staging bundle"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert bundle_name is not None
    assert source_tree_digest(root) == before_source
    assert relocated.is_dir()
    assert (staging / bundle_name / "competitor.txt").read_bytes() == (
        b"preserve competitor"
    )
    assert (outside / "marker.txt").read_text(encoding="utf-8") == "unchanged"
    assert len(os.listdir("/proc/self/fd")) == before_fds


@pytest.mark.parametrize("failure_phase", ["write", "close"])
def test_parquet_failure_preserves_replacement_output_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "marker.txt").write_text("unchanged", encoding="utf-8")
    before_source = source_tree_digest(root)
    before_fds = len(os.listdir("/proc/self/fd"))
    replacement: Path | None = None
    displaced: Path | None = None

    def replace_output() -> None:
        nonlocal replacement, displaced
        candidates = list(
            staging.glob(".v30-data-*/data/chunk-000/file-000.parquet")
        )
        assert len(candidates) == 1
        replacement = candidates[0]
        displaced = replacement.with_name("owned-file-000.parquet")
        os.replace(replacement, displaced)
        replacement.write_bytes(b"preserve competitor")

    if failure_phase == "write":
        def replace_output_then_fail(*args: object, **kwargs: object) -> None:
            replace_output()
            raise RuntimeError("injected parquet write failure")

        monkeypatch.setattr(
            v30_data_writer.pq,
            "write_table",
            replace_output_then_fail,
        )
        expected_error = "injected parquet write failure"
    else:
        actual_fdopen = os.fdopen

        class FailingClose:
            def __init__(self, descriptor: int, mode: str) -> None:
                self.handle = actual_fdopen(descriptor, mode)

            def __enter__(self) -> object:
                return self.handle

            def __exit__(self, *args: object) -> None:
                self.handle.close()
                replace_output()
                raise OSError(errno.EIO, "injected parquet close failure")

        monkeypatch.setattr(v30_data_writer.os, "fdopen", FailingClose)
        expected_error = "injected parquet close failure"

    with pytest.raises((RuntimeError, OSError, ValueError), match=expected_error):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert replacement is not None
    assert displaced is not None
    assert source_tree_digest(root) == before_source
    assert replacement.read_bytes() == b"preserve competitor"
    assert not displaced.exists()
    assert (outside / "marker.txt").read_text(encoding="utf-8") == "unchanged"
    assert not (staging / "data").exists()
    assert not (staging / "meta/tasks.parquet").exists()
    assert len(os.listdir("/proc/self/fd")) == before_fds


def test_deterministic_directories_are_only_published_from_private_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    actual_mkdir = os.mkdir
    created_names: list[str] = []

    def record_directory_creation(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if isinstance(path, str):
            created_names.append(path)
        actual_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(v30_data_writer.os, "mkdir", record_directory_creation)

    write_v30_data_subset(
        root,
        staging,
        dataset,
        [0],
        read_v30_info(root),
    )

    deterministic = {"staging", "meta", "data", "chunk-000"}
    assert deterministic.isdisjoint(created_names)
    assert created_names


def test_private_directory_publication_rejects_replacement_without_owning_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "marker.txt").write_bytes(b"unchanged")
    relocated = outside / "relocated-owned-meta"
    before_source = source_tree_digest(root)
    actual_publish = writer_publication.rename_noreplace_at
    injected = False

    def publish_then_replace_directory(
        source_fd: int,
        source_name: str,
        destination_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal injected
        actual_publish(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )
        if (
            not injected
            and source_name.startswith(".writer-dir-")
            and destination_name == "meta"
        ):
            destination_parent = Path(
                os.readlink(f"/proc/self/fd/{destination_fd}")
            )
            os.replace(destination_parent / destination_name, relocated)
            (destination_parent / destination_name).mkdir()
            (destination_parent / destination_name / "competitor.txt").write_bytes(
                b"preserve competitor"
            )
            injected = True

    monkeypatch.setattr(
        writer_publication,
        "rename_noreplace_at",
        publish_then_replace_directory,
    )

    with pytest.raises(ValueError, match="staging meta changed while publishing"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert injected
    assert source_tree_digest(root) == before_source
    assert (staging / "meta/competitor.txt").read_bytes() == b"preserve competitor"
    assert relocated.is_dir()
    assert (outside / "marker.txt").read_bytes() == b"unchanged"


def test_private_directory_birth_rejects_replacement_before_first_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    outside = tmp_path / "outside"
    outside.mkdir()
    relocated = outside / "relocated-newborn-bundle"
    before_source = source_tree_digest(root)
    actual_mkdir = writer_publication.os.mkdir
    actual_register = writer_publication._OwnershipRegistry.register
    registered: list[writer_publication._EntryIdentity] = []
    bundle_name: str | None = None
    competitor_identity: writer_publication._EntryIdentity | None = None

    def replace_before_first_open(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal bundle_name, competitor_identity
        actual_mkdir(path, *args, **kwargs)
        if (
            isinstance(path, str)
            and path.startswith(".v30-data-")
            and bundle_name is None
        ):
            parent_fd = kwargs.get("dir_fd")
            assert isinstance(parent_fd, int)
            parent = Path(os.readlink(f"/proc/self/fd/{parent_fd}"))
            bundle_name = path
            os.replace(parent / path, relocated)
            actual_mkdir(path, 0o700, dir_fd=parent_fd)
            (parent / path / "competitor.txt").write_bytes(
                b"preserve newborn competitor"
            )
            competitor_identity = writer_publication._EntryIdentity.from_stat(
                os.stat(path, dir_fd=parent_fd, follow_symlinks=False)
            )

    def record_registration(
        registry: writer_publication._OwnershipRegistry,
        identity: writer_publication._EntryIdentity,
    ) -> None:
        registered.append(identity)
        actual_register(registry, identity)

    monkeypatch.setattr(
        writer_publication.os,
        "mkdir",
        replace_before_first_open,
    )
    monkeypatch.setattr(
        writer_publication._OwnershipRegistry,
        "register",
        record_registration,
    )

    with pytest.raises(ValueError, match="changed while establishing ownership"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert bundle_name is not None
    assert competitor_identity is not None
    assert competitor_identity not in registered
    assert relocated.is_dir()
    assert list(relocated.iterdir()) == []
    assert (staging / bundle_name / "competitor.txt").read_bytes() == (
        b"preserve newborn competitor"
    )
    assert list((staging / bundle_name).iterdir()) == [
        staging / bundle_name / "competitor.txt"
    ]
    assert not (staging / "data").exists()
    assert not (staging / "meta/tasks.parquet").exists()
    assert source_tree_digest(root) == before_source


@pytest.mark.parametrize(
    "replacement_kind",
    ["file", "empty_directory", "nonempty_directory"],
)
def test_retirement_quarantines_before_deleting_and_preserves_a_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    target = parent / "owned"
    if replacement_kind == "file":
        target.write_bytes(b"owned bytes")
    else:
        target.mkdir()
    displaced = parent / "displaced-owned"
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    identity = writer_publication._EntryIdentity.from_stat(
        os.stat("owned", dir_fd=parent_fd, follow_symlinks=False)
    )
    registry = writer_publication._OwnershipRegistry()
    registry.register(identity)
    actual_rename = writer_publication.rename_noreplace_at
    injected = False

    def swap_at_quarantine_move(
        source_fd: int,
        source_name: str,
        destination_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal injected
        if (
            not injected
            and source_fd == parent_fd
            and source_name == "owned"
            and destination_fd == parent_fd
            and destination_name.startswith(".writer-retire-")
        ):
            os.replace(target, displaced)
            if replacement_kind == "file":
                target.write_bytes(b"preserve replacement bytes")
            else:
                target.mkdir()
                if replacement_kind == "nonempty_directory":
                    (target / "competitor.txt").write_bytes(
                        b"preserve replacement directory"
                    )
            injected = True
        actual_rename(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )

    monkeypatch.setattr(
        writer_publication,
        "rename_noreplace_at",
        swap_at_quarantine_move,
    )
    try:
        with pytest.raises(ValueError, match="changed during retirement"):
            writer_publication._retire_and_delete_owned_entry_at(
                parent_fd,
                "owned",
                identity,
                registry,
                lambda: True,
            )
    finally:
        os.close(parent_fd)

    assert injected
    assert displaced.exists()
    assert target.exists()
    if replacement_kind == "file":
        assert target.read_bytes() == b"preserve replacement bytes"
    elif replacement_kind == "nonempty_directory":
        assert (target / "competitor.txt").read_bytes() == (
            b"preserve replacement directory"
        )
    assert registry.is_owned(identity)
    assert list(parent.glob(".writer-retire-*")) == []


@pytest.mark.parametrize("entry_kind", ["file", "directory"])
def test_retiring_owned_identity_prevents_reentrant_inode_reuse_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    target = parent / "owned"
    if entry_kind == "file":
        target.write_bytes(b"owned")
    else:
        target.mkdir()
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    identity = v30_data_writer._EntryIdentity.from_stat(
        os.stat("owned", dir_fd=parent_fd, follow_symlinks=False)
    )
    registry = v30_data_writer._OwnershipRegistry()
    registry.register(identity)
    actual_entry_at = v30_data_writer._entry_at
    actual_unlink = os.unlink
    actual_rmdir = os.rmdir
    replacement_installed = False

    def reused_identity(parent_descriptor: int, name: str):
        current = actual_entry_at(parent_descriptor, name)
        if replacement_installed and current is not None:
            return identity
        return current

    def install_file_replacement(
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal replacement_installed
        actual_unlink(name, *args, **kwargs)
        target.write_bytes(b"replacement")
        replacement_installed = True
        v30_data_writer._remove_owned_entry_at(
            parent_fd,
            "owned",
            identity,
            registry,
            lambda: True,
        )

    def install_directory_replacement(
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal replacement_installed
        actual_rmdir(name, *args, **kwargs)
        target.mkdir()
        replacement_installed = True
        v30_data_writer._remove_owned_entry_at(
            parent_fd,
            "owned",
            identity,
            registry,
            lambda: True,
        )

    monkeypatch.setattr(writer_publication, "_entry_at", reused_identity)
    if entry_kind == "file":
        monkeypatch.setattr(v30_data_writer.os, "unlink", install_file_replacement)
    else:
        monkeypatch.setattr(v30_data_writer.os, "rmdir", install_directory_replacement)
    try:
        v30_data_writer._remove_owned_entry_at(
            parent_fd,
            "owned",
            identity,
            registry,
            lambda: True,
        )
    finally:
        os.close(parent_fd)

    assert replacement_installed
    assert target.exists()
    if entry_kind == "file":
        assert target.read_bytes() == b"replacement"
    assert not registry.is_owned(identity)


def test_write_and_close_failure_preserves_primary_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    actual_fdopen = os.fdopen

    class SecondaryCloseFailure:
        def __init__(self, descriptor: int, mode: str) -> None:
            self.handle = actual_fdopen(descriptor, mode)

        def __enter__(self) -> object:
            return self.handle

        def __exit__(self, *args: object) -> None:
            self.handle.close()
            raise OSError(errno.EIO, "secondary file close failure")

    def fail_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("primary parquet write failure")

    monkeypatch.setattr(v30_data_writer.os, "fdopen", SecondaryCloseFailure)
    monkeypatch.setattr(v30_data_writer.pq, "write_table", fail_write)

    with pytest.raises(RuntimeError, match="primary parquet write failure") as raised:
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert any(
        "secondary file close failure" in note
        for note in getattr(raised.value, "__notes__", ())
    )
    assert not (staging / "data").exists()
    assert not (staging / "meta/tasks.parquet").exists()


def test_first_directory_close_failure_does_not_mask_body_or_skip_later_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    before_fds = len(os.listdir("/proc/self/fd"))
    actual_close = v30_data_writer._ChildDirectory.close
    close_contexts: list[str] = []
    injected = False

    def close_then_fail_first(child: object) -> None:
        nonlocal injected
        context = str(getattr(child, "context"))
        close_contexts.append(context)
        actual_close(child)
        if not injected:
            injected = True
            raise OSError(errno.EIO, "secondary directory close failure")

    def fail_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("primary parquet write failure")

    monkeypatch.setattr(
        v30_data_writer._ChildDirectory,
        "close",
        close_then_fail_first,
    )
    monkeypatch.setattr(v30_data_writer.pq, "write_table", fail_write)

    with pytest.raises(RuntimeError, match="primary parquet write failure") as raised:
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert "staging data chunk" in close_contexts
    assert "staging bundle data" in close_contexts
    assert any(
        "secondary directory close failure" in note
        for note in getattr(raised.value, "__notes__", ())
    )
    assert len(os.listdir("/proc/self/fd")) == before_fds


def test_successful_publication_close_failure_rolls_back_and_closes_everything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    before_source = source_tree_digest(root)
    before_fds = len(os.listdir("/proc/self/fd"))
    actual_close = v30_data_writer._ChildDirectory.close
    close_contexts: list[str] = []
    injected = False

    def close_bundle_then_fail(child: object) -> None:
        nonlocal injected
        context = str(getattr(child, "context"))
        close_contexts.append(context)
        actual_close(child)
        if context == "staging bundle" and not injected:
            injected = True
            raise OSError(errno.EIO, "injected bundle close failure")

    monkeypatch.setattr(
        v30_data_writer._ChildDirectory,
        "close",
        close_bundle_then_fail,
    )

    with pytest.raises(ValueError, match="cleanup failed") as raised:
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert injected
    assert "staging meta" in close_contexts
    assert "injected bundle close failure" in str(raised.value.__cause__)
    assert source_tree_digest(root) == before_source
    assert not (staging / "data").exists()
    assert not (staging / "meta/tasks.parquet").exists()
    assert len(os.listdir("/proc/self/fd")) == before_fds


def test_source_close_only_failure_rolls_back_and_attempts_every_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    before_source = source_tree_digest(root)
    before_fds = len(os.listdir("/proc/self/fd"))
    actual_fsync = os.fsync
    actual_close = secure_tree.SecureFile.close
    close_phase = False
    injected = False
    attempted: list[str] = []

    def mark_final_publication(descriptor: int) -> None:
        nonlocal close_phase
        actual_fsync(descriptor)
        try:
            target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            return
        if target == staging:
            close_phase = True

    def close_then_fail_first_source(item: secure_tree.SecureFile) -> None:
        nonlocal injected
        was_open = item._fd >= 0 or item._parent_fd >= 0
        if close_phase and was_open:
            attempted.append(item.relative)
        actual_close(item)
        if close_phase and was_open and not injected:
            injected = True
            raise OSError(errno.EIO, "injected secure source close failure")

    monkeypatch.setattr(v30_data_writer.os, "fsync", mark_final_publication)
    monkeypatch.setattr(
        secure_tree.SecureFile,
        "close",
        close_then_fail_first_source,
    )

    with pytest.raises(
        ValueError,
        match="injected secure source close failure",
    ):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    expected_data = dataset.episodes[0].data.path.relative_to(root).as_posix()
    assert injected
    assert {"meta/tasks.parquet", "meta/stats.json", expected_data} <= set(
        attempted
    )
    assert source_tree_digest(root) == before_source
    assert not (staging / "data").exists()
    assert not (staging / "meta/tasks.parquet").exists()
    assert len(os.listdir("/proc/self/fd")) == before_fds


def test_source_close_failure_is_secondary_to_primary_publication_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    actual_publish = writer_publication.WriterPublication.publish
    actual_close = secure_tree.SecureFile.close
    close_phase = False
    source_close_failures = 0
    attempted: list[str] = []

    def publish_then_fail(
        publication: writer_publication.WriterPublication,
        source_parent_fd: int,
        source_name: str,
        expected: writer_publication._EntryIdentity,
        destination_parent_fd: int,
        destination_name: str,
        destination_is_attached,
        context: str,
    ) -> None:
        nonlocal close_phase
        actual_publish(
            publication,
            source_parent_fd,
            source_name,
            expected,
            destination_parent_fd,
            destination_name,
            destination_is_attached,
            context,
        )
        if destination_name == "tasks.parquet":
            close_phase = True
            raise RuntimeError("primary publication failure")

    def close_then_fail_first_source(item: secure_tree.SecureFile) -> None:
        nonlocal source_close_failures
        was_open = item._fd >= 0 or item._parent_fd >= 0
        if close_phase and was_open:
            attempted.append(item.relative)
        actual_close(item)
        if close_phase and was_open and source_close_failures < 2:
            source_close_failures += 1
            raise OSError(
                errno.EIO,
                f"secondary secure source close failure {source_close_failures}",
            )

    monkeypatch.setattr(
        writer_publication.WriterPublication,
        "publish",
        publish_then_fail,
    )
    monkeypatch.setattr(
        secure_tree.SecureFile,
        "close",
        close_then_fail_first_source,
    )

    with pytest.raises(RuntimeError, match="primary publication failure") as raised:
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    expected_data = dataset.episodes[0].data.path.relative_to(root).as_posix()
    assert source_close_failures == 2
    assert {"meta/tasks.parquet", "meta/stats.json", expected_data} <= set(
        attempted
    )
    assert any(
        "secondary secure source close failure 1" in note
        and "secondary secure source close failure 2" in note
        for note in getattr(raised.value, "__notes__", ())
    )
    assert not (staging / "data").exists()
    assert not (staging / "meta/tasks.parquet").exists()


def test_private_directory_birth_uses_open_descriptor_as_ownership_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    outside = tmp_path / "outside"
    outside.mkdir()
    relocated = outside / "relocated-private-directory"
    before_source = source_tree_digest(root)
    actual_stat = os.stat
    injected = False

    def replace_private_directory_before_first_path_observation(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        nonlocal injected
        directory_fd = kwargs.get("dir_fd")
        if (
            not injected
            and isinstance(path, str)
            and path.startswith(".writer-dir-")
            and isinstance(directory_fd, int)
        ):
            parent = Path(os.readlink(f"/proc/self/fd/{directory_fd}"))
            os.replace(parent / path, relocated)
            (parent / path).mkdir()
            (parent / path / "competitor.txt").write_bytes(b"preserve competitor")
            injected = True
        return actual_stat(path, *args, **kwargs)

    monkeypatch.setattr(
        v30_data_writer.os,
        "stat",
        replace_private_directory_before_first_path_observation,
    )

    with pytest.raises(ValueError, match="unable to anchor staging path") as raised:
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert injected
    assert isinstance(raised.value.__cause__, ValueError)
    assert "changed while establishing ownership" in str(raised.value.__cause__)
    assert relocated.is_dir()
    competitors = list(tmp_path.glob(".writer-dir-*/competitor.txt"))
    assert len(competitors) == 1
    assert competitors[0].read_bytes() == b"preserve competitor"
    assert source_tree_digest(root) == before_source
    assert not (staging / "data").exists()
    assert not (staging / "meta/tasks.parquet").exists()


@pytest.mark.parametrize("reserved_entry", ["data", "chunk-000"])
def test_bundle_directories_reject_preexisting_entries_instead_of_adopting_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reserved_entry: str,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    actual_create = v30_data_writer._create_private_child_directory
    actual_publish = writer_publication._publish_private_directory
    injected = False

    def create_then_preoccupy_data(*args: object, **kwargs: object):
        nonlocal injected
        directory = actual_create(*args, **kwargs)
        context = str(args[2])
        if reserved_entry == "data" and context == "staging bundle":
            bundle = Path(os.readlink(f"/proc/self/fd/{directory.descriptor}"))
            (bundle / "data").mkdir()
            (bundle / "data/competitor.txt").write_bytes(b"preserve competitor")
            injected = True
        return directory

    def publish_then_preoccupy_chunk(*args: object, **kwargs: object):
        nonlocal injected
        directory = actual_publish(*args, **kwargs)
        destination_name = str(args[2])
        if reserved_entry == "chunk-000" and destination_name == "data":
            data = Path(os.readlink(f"/proc/self/fd/{directory.descriptor}"))
            (data / "chunk-000").mkdir()
            (data / "chunk-000/competitor.txt").write_bytes(
                b"preserve competitor"
            )
            injected = True
        return directory

    monkeypatch.setattr(
        writer_publication,
        "_create_private_child_directory",
        create_then_preoccupy_data,
    )
    monkeypatch.setattr(
        writer_publication,
        "_publish_private_directory",
        publish_then_preoccupy_chunk,
    )

    with pytest.raises((FileExistsError, ValueError), match="already exists"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert injected
    competitors = list(staging.glob(".v30-data-*/**/competitor.txt"))
    assert len(competitors) == 1
    assert competitors[0].read_bytes() == b"preserve competitor"
    assert not (staging / "data").exists()
    assert not (staging / "meta/tasks.parquet").exists()


def test_incomplete_private_bundle_retirement_rolls_back_published_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    before_source = source_tree_digest(root)
    actual_rmdir = os.rmdir
    injected = False

    def fail_private_bundle_retirement(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal injected
        if (
            not injected
            and isinstance(path, str)
            and path.startswith(".writer-retire-")
        ):
            injected = True
            return
        actual_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(v30_data_writer.os, "rmdir", fail_private_bundle_retirement)

    with pytest.raises(ValueError, match="cleanup failed"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert injected
    assert source_tree_digest(root) == before_source
    assert not (staging / "data").exists()
    assert not (staging / "meta/tasks.parquet").exists()


def test_final_staging_anchor_close_failure_rolls_back_published_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    before_source = source_tree_digest(root)
    before_fds = len(os.listdir("/proc/self/fd"))
    actual_close = v30_data_writer._AnchoredDirectoryPath.close
    injected = False

    def close_then_fail_final_staging(anchor: object) -> None:
        nonlocal injected
        actual_close(anchor)
        if getattr(anchor, "label") == "staging path" and not injected:
            injected = True
            raise OSError(errno.EIO, "injected final staging anchor close failure")

    monkeypatch.setattr(
        v30_data_writer._AnchoredDirectoryPath,
        "close",
        close_then_fail_final_staging,
    )

    with pytest.raises(ValueError, match="cleanup failed"):
        write_v30_data_subset(
            root,
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert injected
    assert source_tree_digest(root) == before_source
    assert not (staging / "data").exists()
    assert not (staging / "meta/tasks.parquet").exists()
    assert len(os.listdir("/proc/self/fd")) == before_fds


def test_attachment_loss_preserves_cleanup_errors_already_accumulated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    owned = parent / "a-owned"
    owned.write_bytes(b"owned")
    (parent / "z-competitor").write_bytes(b"competitor")
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    identity = v30_data_writer._EntryIdentity.from_stat(
        os.stat("a-owned", dir_fd=parent_fd, follow_symlinks=False)
    )
    registry = v30_data_writer._OwnershipRegistry()
    registry.register(identity)
    actual_unlink = os.unlink
    removal_failed = False

    def fail_owned_removal(path: object, *args: object, **kwargs: object) -> None:
        nonlocal removal_failed
        if isinstance(path, str) and path.startswith(".writer-retire-"):
            removal_failed = True
            raise OSError(errno.EIO, "injected owned cleanup failure")
        actual_unlink(path, *args, **kwargs)

    monkeypatch.setattr(v30_data_writer.os, "unlink", fail_owned_removal)
    try:
        with pytest.raises(ValueError, match="injected owned cleanup failure"):
            v30_data_writer._remove_owned_identities_below(
                parent_fd,
                registry,
                lambda: not removal_failed,
            )
    finally:
        os.close(parent_fd)

    quarantined = list(parent.glob(".writer-retire-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"owned"
    assert not owned.exists()
    assert (parent / "z-competitor").read_bytes() == b"competitor"


@pytest.mark.parametrize("opening", ["anchored path", "child directory"])
def test_identity_failure_closes_untracked_directory_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opening: str,
) -> None:
    actual_open = os.open
    actual_fstat = os.fstat
    fail_descriptors: set[int] = set()

    def capture_open(path: object, *args: object, **kwargs: object) -> int:
        descriptor = actual_open(path, *args, **kwargs)
        if isinstance(path, str) and path.startswith(".writer-dir-"):
            fail_descriptors.add(descriptor)
        return descriptor

    def fail_identity(descriptor: int) -> os.stat_result:
        if descriptor in fail_descriptors:
            raise OSError(errno.EMFILE, "injected identity failure")
        return actual_fstat(descriptor)

    monkeypatch.setattr(v30_data_writer.os, "open", capture_open)
    monkeypatch.setattr(v30_data_writer.os, "fstat", fail_identity)
    before_fds = len(os.listdir("/proc/self/fd"))

    if opening == "anchored path":
        target = tmp_path / "tracked"
        for _ in range(5):
            with pytest.raises(ValueError, match="unable to anchor tracked path"):
                    v30_data_writer._AnchoredDirectoryPath.open(
                        target,
                        "tracked path",
                        create_final=True,
                        registry=v30_data_writer._OwnershipRegistry(),
                )
            assert not target.exists()
    else:
        parent = tmp_path / "parent"
        parent.mkdir()
        parent_fd = actual_open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for _ in range(5):
                with pytest.raises(ValueError, match="unable to open tracked child"):
                    v30_data_writer._open_or_create_child_directory(
                        parent_fd,
                        "tracked",
                        "tracked child",
                        lambda: True,
                        v30_data_writer._OwnershipRegistry(),
                    )
                assert not (parent / "tracked").exists()
        finally:
            os.close(parent_fd)

    assert len(os.listdir("/proc/self/fd")) == before_fds


def _add_second_task(
    root: Path,
    *,
    episode_index: int,
    task: str,
    reverse_task_rows: bool,
) -> None:
    tasks_path = root / "meta/tasks.parquet"
    tasks = pq.read_table(tasks_path)
    source_rows = [(0, "Arrange the colored blocks."), (1, task)]
    if reverse_task_rows:
        source_rows.reverse()
    task_table = pa.Table.from_arrays(
        [
            pa.array([row[0] for row in source_rows], type=tasks.schema.field("task_index").type),
            pa.array([row[1] for row in source_rows], type=tasks.schema.field("task").type),
        ],
        schema=tasks.schema,
    )
    pq.write_table(task_table, tasks_path)

    data_path = root / "data/chunk-000/file-000.parquet"
    data = pq.read_table(data_path)
    task_values = data["task_index"].to_pylist()
    episode_values = data["episode_index"].to_pylist()
    task_values = [
        1 if value == episode_index else task_value
        for value, task_value in zip(episode_values, task_values, strict=True)
    ]
    position = data.schema.get_field_index("task_index")
    data = data.set_column(
        position,
        data.schema.field(position),
        pa.array(task_values, type=data.schema.field(position).type),
    )
    pq.write_table(data, data_path)

    episode_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(episode_path)
    values = episodes["tasks"].to_pylist()
    values[episode_index] = [task]
    position = episodes.schema.get_field_index("tasks")
    episodes = episodes.set_column(
        position,
        episodes.schema.field(position),
        pa.array(values, type=episodes.schema.field(position).type),
    )
    pq.write_table(episodes, episode_path)

    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_tasks"] = 2
    info_path.write_text(json.dumps(info), encoding="utf-8")


def _move_final_episode_to_second_data_shard(root: Path) -> None:
    first_path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(first_path)
    pq.write_table(table.slice(0, 14), first_path)
    second_path = root / "data/chunk-000/file-001.parquet"
    pq.write_table(table.slice(14, 5), second_path)

    episode_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(episode_path)
    values = episodes["data/file_index"].to_pylist()
    values[2] = 1
    position = episodes.schema.get_field_index("data/file_index")
    episodes = episodes.set_column(
        position,
        episodes.schema.field(position),
        pa.array(values, type=episodes.schema.field(position).type),
    )
    pq.write_table(episodes, episode_path)


def _add_numeric_feature(
    root: Path,
    name: str,
    dtype: str,
    first_episode_values: list[int],
) -> None:
    data_path = root / "data/chunk-000/file-000.parquet"
    data = pq.read_table(data_path)
    values = [
        first_episode_values[index % len(first_episode_values)]
        for index in range(data.num_rows)
    ]
    arrow_type = {
        "float32": pa.float32(),
        "int16": pa.int16(),
    }[dtype]
    data = data.add_column(
        data.schema.get_field_index("timestamp"),
        name,
        pa.array(values, type=arrow_type),
    )
    pq.write_table(data, data_path)

    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = {}
    for feature, declaration in info["features"].items():
        if feature == "timestamp":
            features[name] = {"dtype": dtype, "shape": [1], "names": None}
        features[feature] = declaration
    info["features"] = features
    info_path.write_text(json.dumps(info), encoding="utf-8")

    stats_path = root / "meta/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats[name] = {
        metric: ([19] if metric == "count" else [0.0])
        for metric in stats["action"]
    }
    stats_path.write_text(json.dumps(stats), encoding="utf-8")


def _tree_digest(root: Path) -> str:
    return source_tree_digest(root)


def _schema_with_metadata(
    schema: pa.Schema,
    schema_value: bytes,
    field_name: str,
) -> pa.Schema:
    fields = [
        field.with_nullable(False).with_metadata({b"field-purpose": b"identity"})
        if field.name == field_name
        else field
        for field in schema
    ]
    return pa.schema(fields, metadata={b"fixture-schema": schema_value})

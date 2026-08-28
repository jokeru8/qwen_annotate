import io
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from robo_annotate.lerobot import inspect_dataset
from robo_annotate import v30_data_writer
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
            assert values == pytest.approx(expected_aggregate[feature][metric])
        for episode_index, episode in result.episode_stats.items():
            for metric, values in episode[feature].items():
                expected = expected_episodes[episode_index][
                    f"stats/{feature}/{metric}"
                ]
                assert values == pytest.approx(expected)


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
    assert result.aggregate_stats[name]["mean"] == pytest.approx(
        [64 / 255, 128 / 255, 192 / 255]
    )
    assert result.aggregate_stats[name]["count"] == [5]


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

import json
import io
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from robo_annotate.converter import convert_dataset
from robo_annotate.lerobot import EpisodeVideoRef, inspect_dataset, probe_video
from robo_annotate.models import FinalAnnotation
from robo_annotate.release_validator import validate_release
from robo_annotate.workspace import EpisodeRecord, WorkspaceStore
from tests.v30_fixtures import (
    make_lerobot_v30_fixture,
    make_official_language_array,
    make_v30_config,
)


RECORDED_AT = datetime(2026, 8, 28, 12, tzinfo=UTC)


def test_v30_stats_sampling_matches_pinned_v061_profile() -> None:
    from robo_annotate.release_validator_v30 import _sample_indices

    assert len(_sample_indices(19)) == 19
    assert len(_sample_indices(1_000)) == 177
    assert _sample_indices(1_000)[0] == 0
    assert _sample_indices(1_000)[-1] == 999


def accepted_v30_workspace(
    tmp_path: Path,
    *,
    lengths: tuple[int, ...] = (6, 8, 5),
    video_shards_per_episode: bool = False,
    non_padded_video_template: bool = False,
) -> tuple[Path, Path, dict]:
    source = make_lerobot_v30_fixture(
        tmp_path,
        lengths=lengths,
        video_shards_per_episode=video_shards_per_episode,
        non_padded_video_template=non_padded_video_template,
    )
    work = tmp_path / "work"
    base = make_v30_config(source, work)
    payload = base.model_dump(mode="python")
    payload["subtasks"] = [
        {"skill": "pick", "text": "Pick up the colored block."},
        {"skill": "place", "text": "Place the colored block."},
    ]
    payload["sampling"]["min_segment_frames"] = 1
    config = type(base).model_validate(payload)
    dataset = inspect_dataset(config)
    store = WorkspaceStore(work, clock=lambda: RECORDED_AT)
    store.initialize(config, dataset, model_revision="a" * 40)
    for episode in dataset.episodes:
        pending = store.load_episode(episode.episode_index)
        accepted = EpisodeRecord.model_validate(
            pending.model_dump()
            | {
                "status": "accepted",
                "updated_at": RECORDED_AT
                + timedelta(seconds=episode.episode_index + 1),
                "final_annotation": FinalAnnotation(
                    start_subtask_index=0,
                    boundaries=[episode.length // 2],
                ).model_dump(),
                "decision_source": "human",
            }
        )
        (work / f"episodes/episode_{episode.episode_index:06d}.json").write_text(
            accepted.model_dump_json(),
            encoding="utf-8",
        )
    return work, source, {}


def converted_v30_release(
    tmp_path: Path,
    *,
    lengths: tuple[int, ...] = (6, 8, 5),
    video_shards_per_episode: bool = False,
    non_padded_video_template: bool = False,
) -> Path:
    work, _, services = accepted_v30_workspace(
        tmp_path,
        lengths=lengths,
        video_shards_per_episode=video_shards_per_episode,
        non_padded_video_template=non_padded_video_template,
    )
    return convert_dataset(work, tmp_path / "converted", services=services).output


def test_v30_validator_accepts_pinned_v061_feature_and_stats_schema(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))

    assert "total_videos" not in info
    assert info["features"]["observation.images.main"]["shape"] == [24, 32, 3]
    assert info["features"]["observation.enabled"]["dtype"] == "bool"
    assert info["features"]["language_persistent"]["dtype"] == "language"
    assert "q50" in json.loads((output / "meta/stats.json").read_text())["action"]
    assert validate_release(output, deep_video_stats=False).valid


def test_v30_validator_accepts_positive_rebuilt_subset_shard_size_limits(
    tmp_path: Path,
) -> None:
    from tests.test_converter_v30 import selectively_accepted_v30_workspace

    work, _, services = selectively_accepted_v30_workspace(
        tmp_path,
        accepted=(0, 2),
    )
    output = convert_dataset(
        work,
        tmp_path / "accepted",
        accepted_only=True,
        services=services,
    ).output
    info_path = output / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["data_files_size_in_mb"] = 17
    info["video_files_size_in_mb"] = 23
    info_path.write_text(json.dumps(info), encoding="utf-8")

    assert validate_release(output, services=services).valid


def test_v30_validator_accepts_hf_features_json_extension_language_columns(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    table = pq.read_table(output / "data/chunk-000/file-000.parquet")
    for name in ("language_persistent", "language_events"):
        row_type = table.schema.field(name).type.value_type
        json_type = row_type.field("tool_calls").type.value_type
        assert isinstance(json_type, pa.BaseExtensionType)
        assert json_type.extension_name == "arrow.json"
    assert table["language_persistent"].to_pylist()[0][0]["tool_calls"]

    assert validate_release(output, deep_video_stats=False).valid


def test_v30_validator_rejects_string_tool_calls_when_json_extension_is_available(
    tmp_path: Path,
) -> None:
    if not hasattr(pa, "json_"):
        pytest.skip("the pinned fallback represents JSON tool calls as strings")
    output = converted_v30_release(tmp_path)
    path = output / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    name = "language_persistent"
    string_type = pa.list_(
        pa.struct(
            [
                pa.field("role", pa.string()),
                pa.field("content", pa.string()),
                pa.field("style", pa.string()),
                pa.field("timestamp", pa.float32()),
                pa.field("camera", pa.string()),
                pa.field("tool_calls", pa.list_(pa.string())),
            ]
        )
    )
    position = table.schema.get_field_index(name)
    table = table.set_column(
        position,
        name,
        pa.array(table[name].to_pylist(), type=string_type),
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"feature declaration: language_persistent"):
        validate_release(output, deep_video_stats=False)


def test_v30_validator_rejects_malformed_json_language_tool_call(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    path = output / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    name = "language_persistent"
    values = table[name].to_pylist()
    values[0][0]["tool_calls"] = ["not-json"]
    position = table.schema.get_field_index(name)
    table = table.set_column(
        position,
        name,
        make_official_language_array(values, persistent=True),
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"language_persistent.*invalid JSON tool call"):
        validate_release(output, deep_video_stats=False)


@pytest.mark.parametrize(
    "dtype",
    ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64"],
)
def test_v30_validator_accepts_official_numpy_value_dtypes(
    tmp_path: Path,
    dtype: str,
) -> None:
    output = converted_v30_release(tmp_path)
    info_path = output / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.enabled"]["dtype"] = dtype
    info_path.write_text(json.dumps(info), encoding="utf-8")
    data_path = output / "data/chunk-000/file-000.parquet"
    table = pq.read_table(data_path)
    position = table.schema.get_field_index("observation.enabled")
    arrow_type = pa.from_numpy_dtype(dtype)
    values = table["observation.enabled"].to_pylist()
    if dtype.startswith(("int", "uint")):
        values = [int(value) for value in values]
    elif dtype.startswith("float"):
        values = [float(value) for value in values]
    table = table.set_column(
        position,
        "observation.enabled",
        pa.array(values, type=arrow_type),
    )
    pq.write_table(table, data_path)

    assert validate_release(output, deep_video_stats=False).valid


@pytest.mark.parametrize(
    "shape",
    [(2,), (1, 2), (1, 1, 2), (1, 1, 1, 2), (1, 1, 1, 1, 2)],
)
def test_v30_validator_accepts_official_hf_array_shape_ranks(
    tmp_path: Path,
    shape: tuple[int, ...],
) -> None:
    output = converted_v30_release(tmp_path)
    name = "observation.tensor"
    info_path = output / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    rebuilt = {}
    for feature, declaration in info["features"].items():
        if feature == "timestamp":
            rebuilt[name] = {"dtype": "float32", "shape": list(shape), "names": None}
        rebuilt[feature] = declaration
    info["features"] = rebuilt
    info_path.write_text(json.dumps(info), encoding="utf-8")

    def nested(dimensions: tuple[int, ...]):
        if len(dimensions) == 1:
            return [0.0] * dimensions[0]
        return [nested(dimensions[1:]) for _ in range(dimensions[0])]

    def nested_type(dimensions: tuple[int, ...]) -> pa.DataType:
        value: pa.DataType = pa.float32()
        for dimension in reversed(dimensions):
            value = pa.list_(value, dimension)
        return value

    data_path = output / "data/chunk-000/file-000.parquet"
    data = pq.read_table(data_path)
    timestamp_position = data.schema.get_field_index("timestamp")
    data = data.add_column(
        timestamp_position,
        name,
        pa.array([nested(shape)] * data.num_rows, type=nested_type(shape)),
    )
    pq.write_table(data, data_path)

    stats_path = output / "meta/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    profile = {metric: ([19] if metric == "count" else [0.0, 0.0]) for metric in stats["action"]}
    stats[name] = profile
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    episode_path = output / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(episode_path)
    insertion = episodes.schema.get_field_index("meta/episodes/chunk_index")
    for metric in profile:
        values = [
            [length] if metric == "count" else [0.0, 0.0]
            for length in episodes["length"].to_pylist()
        ]
        value_type = pa.list_(pa.int64()) if metric == "count" else pa.list_(pa.float64())
        episodes = episodes.add_column(
            insertion,
            f"stats/{name}/{metric}",
            pa.array(values, type=value_type),
        )
        insertion += 1
    pq.write_table(episodes, episode_path)

    assert validate_release(output, deep_video_stats=False).valid


def test_v30_validator_accepts_official_embedded_image_feature(tmp_path: Path) -> None:
    output = converted_v30_release(tmp_path)
    name = "observation.image.embedded"
    info_path = output / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    rebuilt = {}
    for feature, declaration in info["features"].items():
        if feature == "timestamp":
            rebuilt[name] = {"dtype": "image", "shape": [2, 2, 3], "names": None}
        rebuilt[feature] = declaration
    info["features"] = rebuilt
    info_path.write_text(json.dumps(info), encoding="utf-8")

    encoded = io.BytesIO()
    Image.new("RGB", (2, 2), (0, 0, 0)).save(encoded, format="PNG")
    data_path = output / "data/chunk-000/file-000.parquet"
    data = pq.read_table(data_path)
    data = data.add_column(
        data.schema.get_field_index("timestamp"),
        name,
        pa.array(
            [{"bytes": encoded.getvalue(), "path": "embedded.png"}] * data.num_rows,
            type=pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]),
        ),
    )
    pq.write_table(data, data_path)

    stats_path = output / "meta/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats[name] = {
        metric: ([19] if metric == "count" else [[[0.0]], [[0.0]], [[0.0]]])
        for metric in stats["action"]
    }
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    episode_path = output / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(episode_path)
    insertion = episodes.schema.get_field_index("meta/episodes/chunk_index")
    for metric in stats[name]:
        if metric == "count":
            values = [[length] for length in episodes["length"].to_pylist()]
            value_type = pa.list_(pa.int64())
        else:
            values = [[[[0.0]], [[0.0]], [[0.0]]]] * episodes.num_rows
            value_type = pa.list_(pa.list_(pa.list_(pa.float64())))
        episodes = episodes.add_column(
            insertion,
            f"stats/{name}/{metric}",
            pa.array(values, type=value_type),
        )
        insertion += 1
    pq.write_table(episodes, episode_path)

    assert validate_release(output, deep_video_stats=False).valid


def test_v30_validator_applies_official_dataset_info_defaults(tmp_path: Path) -> None:
    output = converted_v30_release(tmp_path)
    path = output / "meta/info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    for field in (
        "chunks_size",
        "data_files_size_in_mb",
        "video_files_size_in_mb",
        "data_path",
        "video_path",
        "robot_type",
        "splits",
    ):
        info.pop(field, None)
    path.write_text(json.dumps(info), encoding="utf-8")

    assert validate_release(output, deep_video_stats=False).valid


def test_v30_validator_accepts_official_basic_stats_profile(tmp_path: Path) -> None:
    output = converted_v30_release(tmp_path)
    stats_path = output / "meta/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for feature_stats in stats.values():
        for metric in list(feature_stats):
            if metric.startswith("q"):
                del feature_stats[metric]
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    episode_path = output / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(episode_path)
    episodes = episodes.select(
        [
            name
            for name in episodes.column_names
            if not (name.startswith("stats/") and "/q" in name)
        ]
    )
    pq.write_table(episodes, episode_path)

    assert validate_release(output, deep_video_stats=False).valid


def test_v30_validator_rejects_symlinked_release_root_without_reading_target(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    link = tmp_path / "release-link"
    link.symlink_to(output, target_is_directory=True)

    with pytest.raises(ValueError, match=r"release must be a real directory"):
        validate_release(link, deep_video_stats=False)


def test_v30_validator_rejects_symlink_in_release_root_path(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    alias = tmp_path / "aliased-parent"
    alias.symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(ValueError, match=r"release must be a real directory"):
        validate_release(alias / output.name, deep_video_stats=False)


@pytest.mark.parametrize(
    "relative",
    [
        "data",
        "data/chunk-000/file-000.parquet",
        "videos/observation.images.main/chunk-000/file-000.mp4",
    ],
)
def test_v30_validator_rejects_symlinked_tree_components_without_reading_outside(
    tmp_path: Path,
    relative: str,
) -> None:
    output = converted_v30_release(tmp_path)
    target = output / relative
    outside = tmp_path / "outside"
    if target.is_dir():
        outside.mkdir()
        os.rename(target, output / "held-data")
        target.symlink_to(outside, target_is_directory=True)
    else:
        outside.write_bytes(b"outside bytes must never be parsed or decoded")
        target.unlink()
        target.symlink_to(outside)

    with pytest.raises(ValueError, match=r"symbolic link"):
        validate_release(output, deep_video_stats=False)


def test_v30_validator_parses_a_bounded_parquet_snapshot_during_replacement_race(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import robo_annotate.release_validator_v30 as validator

    output = converted_v30_release(tmp_path)
    data_path = output / "data/chunk-000/file-000.parquet"
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"outside bytes must never reach PyArrow")
    original_read = validator.pq.read_table
    calls = 0

    def racing_read(source, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            data_path.unlink()
            data_path.symlink_to(outside)
        return original_read(source, *args, **kwargs)

    monkeypatch.setattr(validator.pq, "read_table", racing_read)

    with pytest.raises(ValueError, match=r"changed during validation"):
        validate_release(output, deep_video_stats=False)


def test_v30_validator_decodes_stable_video_descriptor_during_replacement_race(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    video_path = output / "videos/observation.images.main/chunk-000/file-000.mp4"
    held = video_path.with_name("held.mp4")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside bytes must never reach PyAV")
    calls: list[Path] = []

    def racing_probe(path: Path):
        calls.append(path)
        video_path.rename(held)
        video_path.symlink_to(outside)
        return probe_video(path)

    with pytest.raises(ValueError, match=r"changed during validation"):
        validate_release(
            output,
            services={"probe_video": racing_probe},
            deep_video_stats=False,
        )

    assert calls and str(calls[0]).startswith("/proc/self/fd/")


def test_v30_validator_rejects_root_replacement_after_descriptor_anchor(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    held = tmp_path / "held-release"
    outside = tmp_path / "outside-root"
    outside.mkdir()
    (outside / "marker").write_bytes(b"outside bytes must never be read")
    replaced = False

    def racing_probe(path: Path):
        nonlocal replaced
        if not replaced:
            output.rename(held)
            output.symlink_to(outside, target_is_directory=True)
            replaced = True
        return probe_video(path)

    with pytest.raises(ValueError, match=r"release root changed"):
        validate_release(
            output,
            services={"probe_video": racing_probe},
            deep_video_stats=False,
        )


def test_v30_validator_rejects_oversize_parquet_before_pyarrow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import robo_annotate.release_validator_v30 as validator

    output = converted_v30_release(tmp_path)
    monkeypatch.setattr(validator, "_MAX_PARQUET_BYTES", 64)
    monkeypatch.setattr(
        validator.pq,
        "read_table",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("oversize Parquet reached PyArrow")
        ),
    )

    with pytest.raises(ValueError, match=r"parquet.*exceeds|tasks.parquet.*exceeds"):
        validate_release(output, deep_video_stats=False)


def mutate_episode_metadata(
    root: Path,
    episode_index: int,
    field: str,
    value: object,
) -> None:
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    values = table[field].to_pylist()
    values[episode_index] = value
    position = table.schema.get_field_index(field)
    schema_field = table.schema.field(position)
    table = table.set_column(
        position,
        schema_field,
        pa.array(values, type=schema_field.type),
    )
    pq.write_table(table, path)


def test_v30_validator_rejects_episode_slice_corruption(tmp_path: Path) -> None:
    output = converted_v30_release(tmp_path)
    mutate_episode_metadata(output, 1, "dataset_to_index", 999)

    with pytest.raises(ValueError, match=r"episode 1.*dataset_to_index"):
        validate_release(output)


def test_v30_validator_rejects_noninteger_data_index_schema(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    path = output / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    position = table.schema.get_field_index("episode_index")
    table = table.set_column(
        position,
        "episode_index",
        pa.array(table["episode_index"].to_pylist(), type=pa.float64()),
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"data schema.*episode_index"):
        validate_release(output)


def test_v30_validator_rejects_nonofficial_task_index_arrow_type(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    path = output / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    position = table.schema.get_field_index("task_index")
    table = table.set_column(
        position,
        "task_index",
        pa.array(table["task_index"].to_pylist(), type=pa.int32()),
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"data schema.*task_index"):
        validate_release(output, deep_video_stats=False)


@pytest.mark.parametrize("mutation", ["dtype", "order"])
def test_v30_validator_enforces_official_tasks_arrow_schema(
    tmp_path: Path,
    mutation: str,
) -> None:
    output = converted_v30_release(tmp_path)
    path = output / "meta/tasks.parquet"
    table = pq.read_table(path)
    if mutation == "dtype":
        table = table.set_column(
            0,
            "task_index",
            pa.array(table["task_index"].to_pylist(), type=pa.int32()),
        )
    else:
        table = table.select(["task", "task_index"])
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"tasks.parquet.*schema"):
        validate_release(output, deep_video_stats=False)


def test_v30_validator_enforces_official_episode_arrow_types(tmp_path: Path) -> None:
    output = converted_v30_release(tmp_path)
    path = output / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    position = table.schema.get_field_index("length")
    table = table.set_column(
        position,
        "length",
        pa.array(table["length"].to_pylist(), type=pa.int32()),
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"episode metadata.*length"):
        validate_release(output, deep_video_stats=False)


def test_v30_validator_requires_identical_ordered_episode_shard_schemas(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    first_path = output / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(first_path)
    meta_file = "meta/episodes/file_index"
    position = table.schema.get_field_index(meta_file)
    field = table.schema.field(position)
    first = table.slice(0, 2)
    second = table.slice(2, 1).set_column(position, field, pa.array([1], type=field.type))
    second = second.select([*second.column_names[1:], second.column_names[0]])
    pq.write_table(first, first_path)
    pq.write_table(second, first_path.with_name("file-001.parquet"))

    with pytest.raises(ValueError, match=r"episode metadata.*schema"):
        validate_release(output, deep_video_stats=False)


def test_v30_validator_cross_checks_declared_feature_dtype(tmp_path: Path) -> None:
    output = converted_v30_release(tmp_path)
    path = output / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    position = table.schema.get_field_index("observation.state")
    table = table.set_column(
        position,
        "observation.state",
        pa.array(
            table["observation.state"].to_pylist(),
            type=pa.list_(pa.float64(), 3),
        ),
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"schema.*observation.state"):
        validate_release(output)


def test_v30_validator_rejects_undeclared_data_feature(tmp_path: Path) -> None:
    output = converted_v30_release(tmp_path)
    path = output / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path).append_column(
        "invented.feature",
        pa.array([0.0] * 19, type=pa.float32()),
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"data schema.*invented.feature"):
        validate_release(output)


def test_v30_validator_rejects_out_of_order_global_data_rows(tmp_path: Path) -> None:
    output = converted_v30_release(tmp_path)
    path = output / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    reordered = pa.concat_tables(
        [table.slice(6, 8), table.slice(0, 6), table.slice(14, 5)]
    )
    pq.write_table(reordered, path)

    with pytest.raises(ValueError, match=r"global index.*physical shard order"):
        validate_release(output, deep_video_stats=False)


@pytest.mark.parametrize(
    "field",
    ["data_files_size_in_mb", "video_files_size_in_mb"],
)
def test_v30_validator_rejects_invalid_shard_size_limit(
    tmp_path: Path,
    field: str,
) -> None:
    output = converted_v30_release(tmp_path)
    path = output / "meta/info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info[field] = -1
    path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        validate_release(output)


@pytest.mark.parametrize(
    "field",
    ["data_files_size_in_mb", "video_files_size_in_mb"],
)
def test_v30_validator_rejects_noninteger_official_shard_size_limit(
    tmp_path: Path,
    field: str,
) -> None:
    output = converted_v30_release(tmp_path)
    path = output / "meta/info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info[field] = 1.5
    path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        validate_release(output)


def test_v30_structural_validation_rejects_impossible_video_stats(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    path = output / "meta/stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    camera = "observation.images.main"
    stats[camera]["min"] = [1.0, 1.0, 1.0]
    stats[camera]["max"] = [0.0, 0.0, 0.0]
    path.write_text(json.dumps(stats), encoding="utf-8")

    with pytest.raises(ValueError, match=r"stats.*ordering"):
        validate_release(output, deep_video_stats=False)


def test_v30_validator_rejects_noncanonical_data_shard_numbering(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    source_path = output / "data/chunk-000/file-000.parquet"
    destination = output / "data/chunk-001/file-000.parquet"
    destination.parent.mkdir()
    source_path.rename(destination)
    metadata = output / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(metadata)
    position = table.schema.get_field_index("data/chunk_index")
    field = table.schema.field(position)
    table = table.set_column(position, field, pa.array([1, 1, 1], type=field.type))
    pq.write_table(table, metadata)

    with pytest.raises(ValueError, match=r"data shard numbering"):
        validate_release(output)


def test_v30_validator_checks_optional_episode_path_template(tmp_path: Path) -> None:
    output = converted_v30_release(tmp_path)
    path = output / "meta/info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["episodes_path"] = "../outside/{chunk_index}/{file_index}.parquet"
    path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match=r"episodes_path"):
        validate_release(output, deep_video_stats=False)


def test_v30_custom_instruction_need_not_equal_official_task_text(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    annotations_path = output / "meta/lerobot_annotations.json"
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    custom = "Perform the annotated manipulation workflow."
    for entry in annotations["episodes"].values():
        entry["high_level_instruction"] = custom
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
    task_info_path = output / "meta/task_info/task_0.json"
    task_info = json.loads(task_info_path.read_text(encoding="utf-8"))
    for entry in task_info:
        entry["task_name"] = custom
    task_info_path.write_text(json.dumps(task_info), encoding="utf-8")

    official_tasks = pq.read_table(output / "meta/tasks.parquet")["task"].to_pylist()
    assert official_tasks == ["Arrange the colored blocks."]
    assert validate_release(output, deep_video_stats=False).valid


def test_v30_validator_rejects_namespaced_instruction_disagreement(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    annotations_path = output / "meta/lerobot_annotations.json"
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    annotations["episodes"]["1"]["high_level_instruction"] = "Different instruction."
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")

    with pytest.raises(ValueError, match=r"task_info.*task_name"):
        validate_release(output, deep_video_stats=False)


def test_v30_validator_rejects_shared_video_slice_overlap(tmp_path: Path) -> None:
    output = converted_v30_release(tmp_path)
    camera = "observation.images.main"
    path = output / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    from_field_name = f"videos/{camera}/from_timestamp"
    to_field_name = f"videos/{camera}/to_timestamp"
    from_values = table[from_field_name].to_pylist()
    to_values = table[to_field_name].to_pylist()
    from_values[1] -= 0.1
    to_values[1] -= 0.1
    for field_name, values in (
        (from_field_name, from_values),
        (to_field_name, to_values),
    ):
        position = table.schema.get_field_index(field_name)
        field = table.schema.field(position)
        table = table.set_column(position, field, pa.array(values, type=field.type))
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"video.*overlap"):
        validate_release(output, deep_video_stats=False)


def test_v30_validator_rejects_equal_length_video_range_swap(tmp_path: Path) -> None:
    output = converted_v30_release(tmp_path, lengths=(6, 6, 5))
    camera = "observation.images.main"
    path = output / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    for suffix in ("from_timestamp", "to_timestamp"):
        field_name = f"videos/{camera}/{suffix}"
        values = table[field_name].to_pylist()
        values[0], values[1] = values[1], values[0]
        position = table.schema.get_field_index(field_name)
        field = table.schema.field(position)
        table = table.set_column(position, field, pa.array(values, type=field.type))
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"canonical physical video order"):
        validate_release(output, deep_video_stats=False)


def test_v30_validator_orders_nonpadded_video_shards_by_numeric_identity(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(
        tmp_path,
        lengths=(2,) * 11,
        video_shards_per_episode=True,
        non_padded_video_template=True,
    )

    assert (output / "videos/observation.images.main/chunk-000/file-10.mp4").is_file()
    assert validate_release(output, deep_video_stats=False).valid


def test_v30_validator_rejects_nonpadded_video_numeric_identity_swap(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(
        tmp_path,
        lengths=(2,) * 11,
        video_shards_per_episode=True,
        non_padded_video_template=True,
    )
    camera = "observation.images.main"
    path = output / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    name = f"videos/{camera}/file_index"
    values = table[name].to_pylist()
    values[2], values[10] = values[10], values[2]
    position = table.schema.get_field_index(name)
    table = table.set_column(
        position,
        table.schema.field(position),
        pa.array(values, pa.int64()),
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"canonical physical video order"):
        validate_release(output, deep_video_stats=False)


def test_v30_validator_rejects_video_identities_with_same_rendered_location(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    info_path = output / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["video_path"] = (
        "videos/{video_key:.0}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    info_path.write_text(json.dumps(info), encoding="utf-8")
    source = output / "videos/observation.images.main/chunk-000/file-000.mp4"
    shared = output / "videos/chunk-000/file-000.mp4"
    shared.parent.mkdir()
    source.rename(shared)

    with pytest.raises(ValueError, match=r"unique canonical locations"):
        validate_release(output, deep_video_stats=False)


def test_v30_boundary_preview_uses_local_indices_and_shared_slice_ref(
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    annotations_path = output / "meta/lerobot_annotations.json"
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    annotations["episodes"]["0"]["start_subtask_index"] = 1
    annotations["episodes"]["0"]["boundaries"] = []
    for index in (1, 2):
        annotations["episodes"][str(index)]["start_subtask_index"] = 0
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
    task_info_path = output / "meta/task_info/task_0.json"
    task_info = json.loads(task_info_path.read_text(encoding="utf-8"))
    task_info[0]["label_info"]["action_config"] = [
        {
            "start_frame": 0,
            "end_frame": 6,
            "action_text": "Place the colored block.",
            "skill": "place",
        }
    ]
    task_info_path.write_text(json.dumps(task_info), encoding="utf-8")
    calls = []

    def record_extract(ref, camera, indices):
        calls.append((ref, camera, indices))
        return [
            type("Sample", (), {"frame_index": index, "camera_key": camera})()
            for index in indices
        ]

    report = validate_release(
        output,
        services={"extract_frames": record_extract},
        deep_video_stats=False,
    )

    assert report.preview is not None and report.preview.episode_index == 1
    assert len(calls) == 1
    ref, camera, indices = calls[0]
    assert isinstance(ref, EpisodeVideoRef)
    assert str(ref.path).startswith("/proc/self/fd/")
    assert ref.from_timestamp == pytest.approx(6 / 5)
    assert ref.to_timestamp == pytest.approx(14 / 5)
    assert camera == "observation.images.main"
    assert indices == [3, 4]


def test_v30_release_dispatch_does_not_use_import_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = converted_v30_release(tmp_path)
    monkeypatch.setattr(
        "robo_annotate.lerobot.inspect_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("adapter used")),
    )

    assert validate_release(output, deep_video_stats=False).valid


def test_v30_release_dispatch_does_not_import_or_call_any_v30_adapter_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import builtins
    import importlib
    import robo_annotate.lerobot_v30 as adapter

    output = converted_v30_release(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("v3 adapter entry point used")

    for name in ("inspect_v30_dataset", "read_v30_tasks", "read_v30_episode_table"):
        monkeypatch.setattr(adapter, name, forbidden)
    original_import = builtins.__import__
    original_import_module = importlib.import_module

    def guarded_import(name, *args, **kwargs):
        if name == "robo_annotate.lerobot_v30" or name.endswith(".lerobot_v30"):
            raise AssertionError("v3 adapter imported")
        return original_import(name, *args, **kwargs)

    def guarded_import_module(name, *args, **kwargs):
        if name == "robo_annotate.lerobot_v30" or name.endswith(".lerobot_v30"):
            raise AssertionError("v3 adapter imported")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib, "import_module", guarded_import_module)

    assert validate_release(output, deep_video_stats=False).valid


def test_v30_release_dispatch_reads_info_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from robo_annotate.secure_tree import SecureFile

    output = converted_v30_release(tmp_path)
    original = SecureFile.read_bytes
    reads = []

    def counting_reader(opened: SecureFile):
        if opened.relative == "meta/info.json":
            reads.append(opened.relative)
        return original(opened)

    monkeypatch.setattr(SecureFile, "read_bytes", counting_reader)
    assert validate_release(output, deep_video_stats=False).valid
    assert reads == ["meta/info.json"]


def test_v30_validator_rejects_source_core_byte_mismatch(tmp_path: Path) -> None:
    work, source, services = accepted_v30_workspace(tmp_path)
    output = convert_dataset(work, tmp_path / "converted", services=services).output
    path = output / "meta/info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["robot_type"] = "changed_robot"
    path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match=r"source byte mismatch.*info.json"):
        validate_release(
            output,
            source=source,
            services=services,
            deep_video_stats=False,
        )


def test_v30_validator_rejects_extra_payload_file(tmp_path: Path) -> None:
    output = converted_v30_release(tmp_path)
    extra = output / "data/chunk-000/file-001.parquet"
    extra.write_bytes((output / "data/chunk-000/file-000.parquet").read_bytes())

    with pytest.raises(ValueError, match=r"extra v3 payload"):
        validate_release(output, deep_video_stats=False)


@pytest.mark.parametrize("kind", ["aggregate", "episode"])
def test_v30_validator_rejects_published_stats_corruption(
    tmp_path: Path,
    kind: str,
) -> None:
    output = converted_v30_release(tmp_path)
    if kind == "aggregate":
        path = output / "meta/stats.json"
        stats = json.loads(path.read_text(encoding="utf-8"))
        stats["observation.state"]["mean"][0] += 10
        path.write_text(json.dumps(stats), encoding="utf-8")
    else:
        path = output / "meta/episodes/chunk-000/file-000.parquet"
        table = pq.read_table(path)
        field_name = "stats/observation.state/mean"
        values = table[field_name].to_pylist()
        values[1][0] += 10
        position = table.schema.get_field_index(field_name)
        field = table.schema.field(position)
        table = table.set_column(position, field, pa.array(values, type=field.type))
        pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"stats observation.state"):
        validate_release(output, deep_video_stats=False)


def test_v30_validator_rejects_published_quantile_corruption(tmp_path: Path) -> None:
    output = converted_v30_release(tmp_path)
    path = output / "meta/stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    stats["action"]["q50"][0] += 10
    path.write_text(json.dumps(stats), encoding="utf-8")

    with pytest.raises(ValueError, match=r"stats action"):
        validate_release(output, deep_video_stats=False)

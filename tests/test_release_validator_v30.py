import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robo_annotate.converter import convert_dataset
from robo_annotate.lerobot import EpisodeVideoRef, inspect_dataset
from robo_annotate.models import FinalAnnotation
from robo_annotate.release_validator import validate_release
from robo_annotate.workspace import EpisodeRecord, WorkspaceStore
from tests.v30_fixtures import make_lerobot_v30_fixture, make_v30_config


RECORDED_AT = datetime(2026, 8, 28, 12, tzinfo=UTC)


def accepted_v30_workspace(tmp_path: Path) -> tuple[Path, Path, dict]:
    source = make_lerobot_v30_fixture(tmp_path)
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


def converted_v30_release(tmp_path: Path) -> Path:
    work, _, services = accepted_v30_workspace(tmp_path)
    return convert_dataset(work, tmp_path / "converted", services=services).output


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


def test_v30_release_dispatch_reads_info_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import robo_annotate.release_validator as validator

    output = converted_v30_release(tmp_path)
    original = validator._read_object
    reads = []

    def counting_reader(path: Path):
        if path == output / "meta/info.json":
            reads.append(path)
        return original(path)

    monkeypatch.setattr(validator, "_read_object", counting_reader)
    assert validate_release(output, deep_video_stats=False).valid
    assert reads == [output / "meta/info.json"]


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

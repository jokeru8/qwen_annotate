import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from qwen_annotate.converter import convert_dataset
from qwen_annotate.release_validator import ReleaseReport, validate_release
from tests.test_converter import _fixture


def test_release_validator_is_independent_and_strict(tmp_path: Path) -> None:
    source, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    report = validate_release(output, source=source, services=services)
    assert report.valid is True and report.episode_count == 2 and report.frame_count == 40
    assert report.payload_files == sorted(report.payload_files)
    assert report.payload_checksum
    assert report.preview is not None
    assert ReleaseReport.model_validate_json(report.model_dump_json()) == report
    with pytest.raises(ValidationError):
        ReleaseReport.model_validate(report.model_dump() | {"episode_count": "2"})


@pytest.mark.parametrize("mutation", ["instruction", "boundary", "forbidden", "dagger_mixed", "episode_key"])
def test_release_validator_rejects_annotation_corruption(tmp_path: Path, mutation: str) -> None:
    source, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    path = output / "meta/lerobot_annotations.json"
    data = json.loads(path.read_text())
    if mutation == "instruction": data["episodes"]["0"]["high_level_instruction"] = "wrong"
    elif mutation == "boundary": data["episodes"]["0"]["boundaries"] = [1]
    elif mutation == "forbidden": data["episodes"]["0"]["confidence"] = .9
    elif mutation == "dagger_mixed": data["episodes"]["0"]["start_subtask_index"] = 0
    else: data["episodes"]["2"] = data["episodes"].pop("1")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate_release(output, source=source, services=services)


def test_release_validator_rejects_payload_checksum_and_extra_payload(tmp_path: Path) -> None:
    source, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    (output / "videos/chunk-000/cam.eye/episode_000000.mp4").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        validate_release(output, source=source, services=services)
    (output / "videos/chunk-000/cam.eye/episode_000000.mp4").write_bytes(b"video-0")
    extra = output / "data/chunk-000/episode_999999.parquet"
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError):
        validate_release(output, source=source, services=services)


def test_release_validator_rejects_symlink_and_duplicate_json_key(tmp_path: Path) -> None:
    source, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    info = output / "meta/info.json"
    info.write_text('{"codebase_version":"v2.1","codebase_version":"v2.1"}\n')
    with pytest.raises(ValueError):
        validate_release(output, services=services)
    info.unlink()
    info.symlink_to(source / "meta/info.json")
    with pytest.raises(ValueError):
        validate_release(output, services=services)


def test_release_validator_rejects_parquet_index_semantics(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    source, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    parquet = output / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(parquet).set_column(0, "frame_index", pa.array([1] * 20))
    pq.write_table(table, parquet)
    with pytest.raises(ValueError):
        validate_release(output, services=services)


@pytest.mark.parametrize("mutation", ["missing_episode", "wrong_name", "gap", "wrong_action"])
def test_release_validator_cross_checks_task_info(tmp_path: Path, mutation: str) -> None:
    _, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    path = output / "meta/task_info/task_0.json"
    data = json.loads(path.read_text())
    if mutation == "missing_episode": data.pop()
    elif mutation == "wrong_name": data[0]["task_name"] = "wrong"
    elif mutation == "gap": data[0]["label_info"]["action_config"][1]["start_frame"] = 11
    else: data[0]["label_info"]["action_config"][0]["skill"] = "wrong"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate_release(output, services=services)


REFERENCE = Path("/mnt/data/user/zhoukr/datasets/jokeru/arrange_orange_juice_and_green_tea_2_annotated")


@pytest.mark.skipif(not REFERENCE.is_dir(), reason="downloaded reference dataset unavailable")
def test_real_reference_dataset_is_accepted_read_only() -> None:
    rows = [json.loads(line) for line in (REFERENCE / "meta/episodes.jsonl").read_text().splitlines()]
    lengths = {row["episode_index"]: row["length"] for row in rows}
    before = (REFERENCE / "meta/lerobot_annotations.json").stat().st_mtime_ns

    def probe(path: Path):
        from qwen_annotate.lerobot import VideoProbe
        episode = int(path.stem.split("_")[-1])
        return VideoProbe(frames=lengths[episode], fps=28, width=960, height=744)

    extractor = lambda path, camera, indices, fps: [type("S", (), {"frame_index": n, "camera_key": camera})() for n in indices]
    report = validate_release(REFERENCE, services={"probe_video": probe, "extract_frames": extractor})
    assert report.valid and report.episode_count == 47 and report.mode == "complete"
    assert (REFERENCE / "meta/lerobot_annotations.json").stat().st_mtime_ns == before

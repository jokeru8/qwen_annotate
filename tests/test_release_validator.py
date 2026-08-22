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


@pytest.mark.parametrize("mutation", ["instruction", "boundary", "forbidden", "mode", "episode_key"])
def test_release_validator_rejects_annotation_corruption(tmp_path: Path, mutation: str) -> None:
    source, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    path = output / "meta/lerobot_annotations.json"
    data = json.loads(path.read_text())
    if mutation == "instruction": data["episodes"]["0"]["high_level_instruction"] = "wrong"
    elif mutation == "boundary": data["episodes"]["0"]["boundaries"] = [1]
    elif mutation == "forbidden": data["episodes"]["0"]["confidence"] = .9
    elif mutation == "mode": data["mode"] = "mystery"
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

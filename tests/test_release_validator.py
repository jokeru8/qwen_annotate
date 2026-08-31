import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from robo_annotate.converter import convert_dataset
from robo_annotate.release_validator import ReleaseReport, validate_release
from tests.test_converter import _fixture


def test_release_validator_is_independent_and_strict(tmp_path: Path) -> None:
    source, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    report = validate_release(output, source=source, services=services)
    assert report.valid is True and report.episode_count == 2 and report.frame_count == 40
    assert report.dataset_version == "v2.1"
    assert report.validation_level == "strict_deep" and report.skipped_checks == ()
    assert report.payload_files == sorted(report.payload_files)
    assert report.payload_checksum
    assert report.preview is not None
    assert ReleaseReport.model_validate_json(report.model_dump_json()) == report
    with pytest.raises(ValidationError):
        ReleaseReport.model_validate(report.model_dump() | {"episode_count": "2"})
    with pytest.raises(ValidationError):
        ReleaseReport.model_validate(report.model_dump() | {"skipped_checks": ("invented",)})


def test_release_dispatch_reads_bounded_info_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from robo_annotate.secure_tree import SecureFile

    _, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    original = SecureFile.read_bytes
    reads = []

    def counting_reader(opened: SecureFile):
        if opened.relative == "meta/info.json":
            reads.append(opened.relative)
        return original(opened)

    monkeypatch.setattr(SecureFile, "read_bytes", counting_reader)
    assert validate_release(output, services=services).valid
    assert reads == ["meta/info.json"]


def test_release_validator_requires_both_stats_artifacts(tmp_path: Path) -> None:
    """Catches deleting both stats files to bypass all statistics validation."""
    _, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    (output / "meta/stats.json").unlink()
    (output / "meta/episodes_stats.jsonl").unlink()
    with pytest.raises(ValueError, match="stats"):
        validate_release(output, services=services)


def test_release_validator_cross_checks_declared_camera_dimensions(tmp_path: Path) -> None:
    """Catches video shape metadata that disagrees with the decoded/probed payload."""
    _, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    info_path = output / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["cam.eye"]["shape"] = [5, 6, 3]
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="shape"):
        validate_release(output, services=services)


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
        from robo_annotate.lerobot import VideoProbe
        episode = int(path.stem.split("_")[-1])
        return VideoProbe(frames=lengths[episode], fps=28, width=960, height=744)

    extractor = lambda video, camera, indices: [
        type("S", (), {"frame_index": n, "camera_key": camera})() for n in indices
    ]
    with pytest.raises(ValueError, match="stats"):
        validate_release(REFERENCE, services={"probe_video": probe, "extract_frames": extractor})
    report = validate_release(
        REFERENCE,
        services={"probe_video": probe, "extract_frames": extractor},
        allow_legacy_sampled_image_stats=True,
        deep_video_stats=False,
    )
    assert report.valid and report.episode_count == 47 and report.mode == "complete"
    assert report.validation_level == "legacy_structural"
    assert set(report.skipped_checks) == {"numeric_quantile_payload_equality", "video_payload_stat_equality"}
    assert (REFERENCE / "meta/lerobot_annotations.json").stat().st_mtime_ns == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_root", "relative/source"),
        ("source_root", "/different/source"),
        ("work_dir", "relative/meta"),
        ("work_dir", "/absolute/private_workspace"),
        ("work_dir", "/absolute/private_workspace/meta"),
    ],
)
def test_release_validator_checks_public_path_provenance(
    tmp_path: Path, field: str, value: str,
) -> None:
    source, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    annotation_path = output / "meta/lerobot_annotations.json"
    annotations = json.loads(annotation_path.read_text())
    annotations[field] = value
    annotation_path.write_text(json.dumps(annotations))
    with pytest.raises(ValueError):
        validate_release(
            output,
            source=source,
            services=services,
            _expected_output_root=output,
        )


def test_expected_output_root_is_checked_during_staging_validation(tmp_path: Path) -> None:
    _, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    with pytest.raises(ValueError, match="work_dir"):
        validate_release(
            output,
            services=services,
            _expected_output_root=tmp_path / "other-release",
        )


def test_annotation_top_level_key_order_is_not_semantic(tmp_path: Path) -> None:
    _, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    path = output / "meta/lerobot_annotations.json"
    value = json.loads(path.read_text())
    path.write_text(json.dumps(dict(reversed(list(value.items())))))
    assert validate_release(output, services=services).valid


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_annotation_top_level_requires_exact_key_set(tmp_path: Path, mutation: str) -> None:
    _, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    path = output / "meta/lerobot_annotations.json"
    value = json.loads(path.read_text())
    if mutation == "missing":
        value.pop("updated_at")
    else:
        value["invented"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="top-level schema"):
        validate_release(output, services=services)


@pytest.mark.parametrize(("measured_fps", "valid"), [(9.999, True), (9.98, False)])
def test_release_video_fps_uses_import_tolerance(
    tmp_path: Path, measured_fps: float, valid: bool,
) -> None:
    _, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    from robo_annotate.lerobot import VideoProbe
    adjusted = services | {
        "probe_video": lambda path: VideoProbe(frames=20, fps=measured_fps, width=6, height=4)
    }
    if valid:
        assert validate_release(output, services=adjusted).valid
    else:
        with pytest.raises(ValueError, match="video metadata mismatch"):
            validate_release(output, services=adjusted)


def test_release_validator_recomputes_numeric_stats_instead_of_trusting_ordered_values(tmp_path: Path) -> None:
    """Catches fabricated but structurally valid zero statistics."""
    _, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    path = output / "meta/stats.json"
    stats = json.loads(path.read_text())
    for metric in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
        stats["index"][metric] = [0.0]
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="stats"):
        validate_release(output, services=services)


@pytest.mark.parametrize("mutation", ["remove", "inflate"])
def test_generated_release_requires_exact_payload_size_metadata(tmp_path: Path, mutation: str) -> None:
    _, work, services = _fixture(tmp_path)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    path = output / "meta/info.json"
    info = json.loads(path.read_text())
    if mutation == "remove":
        info.pop("data_files_size_in_mb")
    else:
        info["video_files_size_in_mb"] = 999999
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="size"):
        validate_release(
            output, services=services,
            allow_legacy_sampled_image_stats=True, deep_video_stats=False,
        )


def test_legacy_options_are_not_silently_contradictory(tmp_path: Path) -> None:
    _, work, services = _fixture(tmp_path, legacy_stats=True)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    with pytest.raises(ValueError, match="legacy.*deep"):
        validate_release(output, services=services, allow_legacy_sampled_image_stats=True)


def test_legacy_quantile_guarantee_is_explicit_and_source_backed_when_available(tmp_path: Path) -> None:
    source, work, services = _fixture(tmp_path, legacy_stats=True)
    output = tmp_path / "release"
    convert_dataset(work, output, services=services)
    path = output / "meta/stats.json"
    stats = json.loads(path.read_text())
    stats["index"]["q50"] = [(stats["index"]["q10"][0] + stats["index"]["q90"][0]) / 2 + 0.01]
    path.write_text(json.dumps(stats))

    structural = validate_release(
        output, services=services,
        allow_legacy_sampled_image_stats=True, deep_video_stats=False,
    )
    assert structural.validation_level == "legacy_structural"
    assert "numeric_quantile_payload_equality" in structural.skipped_checks
    with pytest.raises(ValueError, match="legacy release stats"):
        validate_release(
            output, source=source, services=services,
            allow_legacy_sampled_image_stats=True, deep_video_stats=False,
        )

import json
import os
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pyarrow.parquet as pq

from robo_annotate.config import AnnotationConfig
from robo_annotate.lerobot import inspect_dataset
from robo_annotate.models import FinalAnnotation
from robo_annotate.publication_metadata import (
    SelectedEpisode,
    write_public_annotations,
)
from robo_annotate.workspace import EpisodeRecord, RunManifest, WorkspaceStore
from tests.v30_fixtures import (
    make_lerobot_v30_fixture,
    make_v30_config,
    official_core_file_digests,
    source_tree_digest,
)


CONVERTED_AT = datetime(2026, 8, 28, tzinfo=UTC)
RECORDED_AT = datetime(2026, 8, 22, 12, tzinfo=UTC)


def prepared_v30_publication(
    tmp_path: Path, *, augmentation: bool = False
) -> tuple[Path, Path, RunManifest, list[SelectedEpisode]]:
    root = make_lerobot_v30_fixture(tmp_path)
    staging = tmp_path / "staging"
    shutil.copytree(root, staging)
    output = tmp_path / "published"
    base_config = make_v30_config(root, tmp_path / "work")
    config_payload = base_config.model_dump(mode="python")
    config_payload["subtasks"] = [
        {"skill": "pick", "text": "Pick up the colored block."},
        {"skill": "place", "text": "Place the colored block."},
    ]
    config_payload["sampling"]["min_segment_frames"] = 2
    config_payload["augmentation"] = {
        "enabled": augmentation,
        "language": "English",
    }
    config = AnnotationConfig.model_validate(config_payload)
    dataset = inspect_dataset(config)
    store = WorkspaceStore(config.work_dir, clock=lambda: RECORDED_AT)
    manifest = store.initialize(config, dataset, model_revision="a" * 40)
    pending = store.load_episode(1)
    accepted = EpisodeRecord.model_validate(
        pending.model_dump()
        | {
            "status": "accepted",
            "updated_at": RECORDED_AT,
            "final_annotation": FinalAnnotation(
                start_subtask_index=0,
                boundaries=[3],
            ).model_dump(),
            "decision_source": "human",
        }
    )
    selected = [
        SelectedEpisode(
            record=accepted,
            source_index=1,
            output_index=0,
            length=dataset.episodes[1].length,
        )
    ]
    return staging, output, manifest, selected


def test_v30_public_annotations_do_not_modify_info(tmp_path: Path) -> None:
    staging, output, manifest, selected = prepared_v30_publication(tmp_path)
    before = (staging / "meta/info.json").read_bytes()

    write_public_annotations(
        staging,
        output,
        manifest,
        selected,
        CONVERTED_AT,
        None,
        extend_info=False,
    )

    assert (staging / "meta/info.json").read_bytes() == before
    annotations = json.loads(
        (staging / "meta/lerobot_annotations.json").read_text(encoding="utf-8")
    )
    assert list(annotations) == [
        "source_root",
        "work_dir",
        "subtask_template",
        "episodes",
        "primary_camera",
        "updated_at",
    ]
    assert annotations["episodes"] == {
        "0": {
            "episode_index": 0,
            "boundaries": [3],
            "high_level_instruction": "Arrange the colored blocks.",
            "saved_at": "2026-08-22T12:00:00+00:00",
        }
    }
    assert annotations["subtask_template"] == [
        {"skill": "pick", "text": "Pick up the colored block."},
        {"skill": "place", "text": "Place the colored block."},
    ]
    assert annotations["work_dir"] == str(output / "meta")
    task_info = json.loads(
        (staging / "meta/task_info/task_0.json").read_text(encoding="utf-8")
    )
    assert task_info == [
        {
            "episode_id": 0,
            "task_id": 0,
            "task_name": "Arrange the colored blocks.",
            "label_info": {
                "action_config": [
                    {
                        "start_frame": 0,
                        "end_frame": 3,
                        "action_text": "Pick up the colored block.",
                        "skill": "pick",
                    },
                    {
                        "start_frame": 3,
                        "end_frame": 8,
                        "action_text": "Place the colored block.",
                        "skill": "place",
                    },
                ]
            },
        }
    ]


def test_public_annotations_map_output_identity_and_source_augmentation(
    tmp_path: Path,
) -> None:
    staging, output, manifest, selected = prepared_v30_publication(
        tmp_path, augmentation=True
    )

    write_public_annotations(
        staging,
        output,
        manifest,
        selected,
        CONVERTED_AT,
        {1: ["Lift the colored block.", "Set the colored block down."]},
        extend_info=False,
    )

    annotations = json.loads(
        (staging / "meta/lerobot_annotations.json").read_text(encoding="utf-8")
    )
    assert list(annotations["episodes"]) == ["0"]
    task_info = json.loads(
        (staging / "meta/task_info/task_0.json").read_text(encoding="utf-8")
    )
    assert task_info[0]["episode_id"] == 0
    assert [
        action["action_text"]
        for action in task_info[0]["label_info"]["action_config"]
    ] == ["Lift the colored block.", "Set the colored block down."]


def test_full_v30_conversion_preserves_core_payload_and_validates(
    tmp_path: Path,
) -> None:
    from robo_annotate.converter import convert_dataset
    from robo_annotate.release_validator import validate_release
    from tests.test_release_validator_v30 import accepted_v30_workspace

    work, source, services = accepted_v30_workspace(tmp_path)
    source_files = official_core_file_digests(source)

    report = convert_dataset(work, tmp_path / "out", services=services)

    assert report.dataset_version == "v3.0"
    assert report.annotation_schema_version == "reference-v3.0"
    assert official_core_file_digests(report.output) == source_files
    validation = validate_release(
        report.output,
        source=source,
        services=services,
    )
    assert validation.dataset_version == "v3.0"


def selectively_accepted_v30_workspace(
    tmp_path: Path,
    accepted: tuple[int, ...],
    *,
    size_limits: tuple[int, int] | None = (100, 200),
) -> tuple[Path, Path, dict]:
    source = make_lerobot_v30_fixture(tmp_path)
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if size_limits is None:
        info.pop("data_files_size_in_mb")
        info.pop("video_files_size_in_mb")
    else:
        info["data_files_size_in_mb"] = size_limits[0]
        info["video_files_size_in_mb"] = size_limits[1]
    info_path.write_text(json.dumps(info), encoding="utf-8")
    work = tmp_path / "work"
    base = make_v30_config(source, work)
    payload = base.model_dump(mode="python")
    payload["augmentation"] = {"enabled": True, "language": "English"}
    payload["sampling"]["min_segment_frames"] = 1
    config = AnnotationConfig.model_validate(payload)
    dataset = inspect_dataset(config)
    store = WorkspaceStore(work, clock=lambda: RECORDED_AT)
    store.initialize(config, dataset, model_revision="a" * 40)
    for episode_index in accepted:
        pending = store.load_episode(episode_index)
        record = EpisodeRecord.model_validate(
            pending.model_dump()
            | {
                "status": "accepted",
                "updated_at": RECORDED_AT + timedelta(seconds=episode_index),
                "final_annotation": FinalAnnotation(
                    start_subtask_index=0,
                    boundaries=[],
                ).model_dump(),
                "decision_source": "human",
            }
        )
        (work / f"episodes/episode_{episode_index:06d}.json").write_text(
            record.model_dump_json(),
            encoding="utf-8",
        )

    def augment(_config, requests):
        return {
            request.episode_index: [
                f"Perform the selected manipulation for source episode "
                f"{request.episode_index}."
            ]
            for request in requests
        }

    return work, source, {"augment_episodes": augment}


@pytest.mark.parametrize(
    ("source_limits", "expected"),
    [
        (None, (100, 200)),
        ((17, 23), (17, 23)),
    ],
)
def test_v30_accepted_only_preserves_official_integer_shard_size_limits(
    tmp_path: Path,
    source_limits: tuple[int, int] | None,
    expected: tuple[int, int],
) -> None:
    from robo_annotate.converter import convert_dataset

    work, _, services = selectively_accepted_v30_workspace(
        tmp_path,
        accepted=(0, 2),
        size_limits=source_limits,
    )

    report = convert_dataset(
        work,
        tmp_path / "accepted",
        accepted_only=True,
        services=services,
    )
    info = json.loads(
        (report.output / "meta/info.json").read_text(encoding="utf-8")
    )

    assert info["data_files_size_in_mb"] == expected[0]
    assert info["video_files_size_in_mb"] == expected[1]
    assert type(info["data_files_size_in_mb"]) is int
    assert type(info["video_files_size_in_mb"]) is int


def test_v30_accepted_only_removes_middle_episode_and_rebuilds_every_reference(
    tmp_path: Path,
) -> None:
    from robo_annotate.converter import convert_dataset
    from robo_annotate.release_validator import validate_release

    work, source, services = selectively_accepted_v30_workspace(
        tmp_path,
        accepted=(0, 2),
    )
    source_digest_before = source_tree_digest(source)

    report = convert_dataset(
        work,
        tmp_path / "accepted",
        accepted_only=True,
        services=services,
    )
    output = inspect_dataset(
        make_v30_config(report.output, tmp_path / "inspect-work")
    )

    assert report.dataset_version == "v3.0"
    assert report.episode_count == 2
    assert report.frame_count == 11
    assert [episode.episode_index for episode in output.episodes] == [0, 1]
    assert [episode.length for episode in output.episodes] == [6, 5]
    assert validate_release(report.output, services=services).valid
    assert source_tree_digest(source) == source_digest_before

    info = json.loads(
        (report.output / "meta/info.json").read_text(encoding="utf-8")
    )
    assert info["codebase_version"] == "v3.0"
    assert info["data_path"] == (
        "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    assert info["video_path"] == (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    assert info["episodes_path"] == (
        "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    assert info["data_files_size_in_mb"] == 100
    assert info["video_files_size_in_mb"] == 200
    episode_rows = pq.read_table(
        report.output / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pylist()
    assert [row["episode_index"] for row in episode_rows] == [0, 1]
    assert [row["dataset_from_index"] for row in episode_rows] == [0, 6]
    assert [row["dataset_to_index"] for row in episode_rows] == [6, 11]

    annotations = json.loads(
        (report.output / "meta/lerobot_annotations.json").read_text(
            encoding="utf-8"
        )
    )
    assert list(annotations["episodes"]) == ["0", "1"]
    task_info = json.loads(
        (report.output / "meta/task_info/task_0.json").read_text(encoding="utf-8")
    )
    assert [entry["episode_id"] for entry in task_info] == [0, 1]
    assert [
        entry["label_info"]["action_config"][0]["action_text"]
        for entry in task_info
    ] == [
        "Perform the selected manipulation for source episode 0.",
        "Perform the selected manipulation for source episode 2.",
    ]


def test_v30_accepted_only_rejects_empty_selection_without_output(
    tmp_path: Path,
) -> None:
    from robo_annotate.converter import convert_dataset

    work, source, services = selectively_accepted_v30_workspace(
        tmp_path,
        accepted=(),
    )
    output = tmp_path / "accepted-only"
    before = source_tree_digest(source)

    with pytest.raises(ValueError, match=r"at least one accepted episode"):
        convert_dataset(work, output, accepted_only=True, services=services)

    assert source_tree_digest(source) == before
    assert not output.exists()
    assert not list(tmp_path.glob("accepted-only.staging-*"))


def test_v30_accepted_only_never_cleans_a_replacement_staging_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import robo_annotate.converter_v30 as converter_v30
    from robo_annotate.converter import convert_dataset

    work, source, services = selectively_accepted_v30_workspace(
        tmp_path,
        accepted=(0, 2),
    )
    source_before = source_tree_digest(source)
    held = tmp_path / "held-owned-staging"
    replacement: Path | None = None

    def replace_staging(*args, **kwargs):
        nonlocal replacement
        staging = args[1]
        staging.rename(held)
        staging.mkdir()
        (staging / "competitor.txt").write_text(
            "must survive",
            encoding="utf-8",
        )
        replacement = staging
        raise RuntimeError("injected writer failure")

    monkeypatch.setattr(
        converter_v30,
        "write_v30_data_subset",
        replace_staging,
    )

    with pytest.raises(RuntimeError, match=r"injected writer failure"):
        convert_dataset(
            work,
            tmp_path / "accepted",
            accepted_only=True,
            services=services,
        )

    assert replacement is not None
    assert (replacement / "competitor.txt").read_text(encoding="utf-8") == (
        "must survive"
    )
    assert held.is_dir()
    assert source_tree_digest(source) == source_before
    assert not (tmp_path / "accepted").exists()


def test_v30_cleanup_restores_a_last_boundary_staging_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import robo_annotate.converter_v30 as converter_v30
    import robo_annotate.writer_publication as publication
    from robo_annotate.converter import convert_dataset

    work, _, services = selectively_accepted_v30_workspace(
        tmp_path,
        accepted=(0, 2),
    )
    held = tmp_path / "held-owned-staging"
    replacement: Path | None = None
    original_rename = publication.rename_noreplace_at

    def fail_writer(*args, **kwargs):
        raise RuntimeError("injected writer failure")

    def replace_at_move(source_fd, source_name, destination_fd, destination_name):
        nonlocal replacement
        if source_name.startswith("accepted.staging-") and replacement is None:
            staging = tmp_path / source_name
            staging.rename(held)
            staging.mkdir()
            (staging / "competitor.txt").write_text(
                "must survive",
                encoding="utf-8",
            )
            replacement = staging
        return original_rename(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )

    monkeypatch.setattr(
        converter_v30,
        "write_v30_data_subset",
        fail_writer,
    )
    monkeypatch.setattr(publication, "rename_noreplace_at", replace_at_move)

    with pytest.raises(RuntimeError, match=r"injected writer failure") as failure:
        convert_dataset(
            work,
            tmp_path / "accepted",
            accepted_only=True,
            services=services,
        )

    assert replacement is not None
    assert (replacement / "competitor.txt").read_text(encoding="utf-8") == (
        "must survive"
    )
    assert held.is_dir()
    assert any("staging cleanup" in note for note in failure.value.__notes__)
    assert not (tmp_path / "accepted").exists()


def test_v30_accepted_only_pairs_every_data_and_video_placement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import robo_annotate.converter_v30 as converter_v30
    from robo_annotate.converter import convert_dataset
    from robo_annotate.v30_video_writer import V30VideoWriteResult

    work, source, services = selectively_accepted_v30_workspace(
        tmp_path,
        accepted=(0, 2),
    )
    source_before = source_tree_digest(source)
    original = converter_v30.write_v30_video_subset

    def mismatched(*args, **kwargs):
        result = original(*args, **kwargs)
        placements = dict(result.placements)
        camera = next(iter(placements))
        changed = list(placements[camera])
        changed[1] = replace(changed[1], source_index=1)
        placements[camera] = tuple(changed)
        return V30VideoWriteResult(
            placements=placements,
            files_by_camera=result.files_by_camera,
        )

    monkeypatch.setattr(
        converter_v30,
        "write_v30_video_subset",
        mismatched,
    )

    with pytest.raises(ValueError, match=r"placements do not match"):
        convert_dataset(
            work,
            tmp_path / "accepted",
            accepted_only=True,
            services=services,
        )

    assert source_tree_digest(source) == source_before
    assert not (tmp_path / "accepted").exists()
    assert not list(tmp_path.glob("accepted.staging-*"))


def test_v30_accepted_only_decodes_each_rebuilt_video_once_for_stats(
    tmp_path: Path,
) -> None:
    from robo_annotate.converter import convert_dataset
    from robo_annotate.stats import iter_video_rgb_frames

    work, _, services = selectively_accepted_v30_workspace(
        tmp_path,
        accepted=(0, 2),
    )
    decoded_targets: list[Path] = []

    def tracked(path: Path):
        target = Path(os.readlink(path)) if str(path).startswith("/proc/self/fd/") else path
        parts = target.parts
        decoded_targets.append(Path(*parts[parts.index("videos") :]))
        yield from iter_video_rgb_frames(path)

    services["iter_video_rgb_frames"] = tracked
    report = convert_dataset(
        work,
        tmp_path / "accepted",
        accepted_only=True,
        services=services,
    )
    rebuilt = sorted(
        path.relative_to(report.output)
        for path in report.output.glob("videos/**/*.mp4")
    )

    assert rebuilt
    # One pass creates publication stats and one independent pass validates them.
    assert {path: decoded_targets.count(path) for path in rebuilt} == {
        path: 2 for path in rebuilt
    }


def test_v30_decoder_and_source_close_failures_do_not_mask_primary_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import robo_annotate.converter_v30 as converter_v30
    from robo_annotate.converter import convert_dataset

    work, _, services = selectively_accepted_v30_workspace(
        tmp_path,
        accepted=(0, 2),
    )
    decoder_closed = False
    original_tree_close = converter_v30.SecureTree.close

    class FailingDecoder:
        def __iter__(self):
            return self

        def __next__(self):
            raise RuntimeError("injected decoder failure")

        def close(self):
            nonlocal decoder_closed
            decoder_closed = True
            raise OSError("injected decoder close failure")

    def failing_tree_close(tree):
        original_tree_close(tree)
        if tree.label == "v3 composer source":
            raise OSError("injected source tree close failure")

    services["iter_video_rgb_frames"] = lambda _path: FailingDecoder()
    monkeypatch.setattr(converter_v30.SecureTree, "close", failing_tree_close)

    with pytest.raises(RuntimeError, match=r"injected decoder failure") as failure:
        convert_dataset(
            work,
            tmp_path / "accepted",
            accepted_only=True,
            services=services,
        )

    notes = "\n".join(failure.value.__notes__)
    assert decoder_closed
    assert "video decoder iterator close" in notes
    assert "injected decoder close failure" in notes
    assert "secure tree close" in notes
    assert "injected source tree close failure" in notes
    assert not (tmp_path / "accepted").exists()
    assert not list(tmp_path.glob("accepted.staging-*"))


def test_v30_source_close_only_failure_raises_after_staging_rollback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import robo_annotate.converter_v30 as converter_v30
    from robo_annotate.converter import convert_dataset

    work, _, services = selectively_accepted_v30_workspace(
        tmp_path,
        accepted=(0, 2),
    )
    original_tree_close = converter_v30.SecureTree.close

    def failing_tree_close(tree):
        original_tree_close(tree)
        if tree.label == "v3 composer source":
            raise OSError("injected source tree close failure")

    monkeypatch.setattr(converter_v30.SecureTree, "close", failing_tree_close)

    with pytest.raises(OSError, match=r"injected source tree close failure"):
        convert_dataset(
            work,
            tmp_path / "accepted",
            accepted_only=True,
            services=services,
        )

    assert not (tmp_path / "accepted").exists()
    assert not list(tmp_path.glob("accepted.staging-*"))


def test_v30_output_lock_finalizers_are_all_attempted_without_masking_primary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import robo_annotate.converter as converter

    work, _, services = selectively_accepted_v30_workspace(
        tmp_path,
        accepted=(0, 2),
    )
    original_open_lock = converter._open_lock
    original_flock = converter.fcntl.flock
    original_close = converter.os.close
    lock_descriptor = -1
    close_attempted = False

    def tracked_open_lock(path):
        nonlocal lock_descriptor
        lock_descriptor = original_open_lock(path)
        return lock_descriptor

    def failing_flock(descriptor, operation):
        if descriptor == lock_descriptor and operation == converter.fcntl.LOCK_UN:
            raise OSError("injected output lock release failure")
        return original_flock(descriptor, operation)

    def failing_close(descriptor):
        nonlocal close_attempted
        if descriptor == lock_descriptor:
            close_attempted = True
            original_close(descriptor)
            raise OSError("injected output lock close failure")
        return original_close(descriptor)

    def fail_validation(*args, **kwargs):
        raise RuntimeError("injected validation failure")

    monkeypatch.setattr(converter, "_open_lock", tracked_open_lock)
    monkeypatch.setattr(converter.fcntl, "flock", failing_flock)
    monkeypatch.setattr(converter.os, "close", failing_close)
    monkeypatch.setattr(converter, "validate_release", fail_validation)

    with pytest.raises(RuntimeError, match=r"injected validation failure") as failure:
        converter.convert_dataset(
            work,
            tmp_path / "accepted",
            accepted_only=True,
            services=services,
        )

    notes = "\n".join(failure.value.__notes__)
    assert close_attempted
    assert "output lock release" in notes
    assert "injected output lock release failure" in notes
    assert "output lock close" in notes
    assert "injected output lock close failure" in notes
    assert not (tmp_path / "accepted").exists()
    assert not list(tmp_path.glob("accepted.staging-*"))


def test_v30_cleanup_only_failure_raises_after_published_output_rollback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import robo_annotate.converter as converter

    work, _, services = selectively_accepted_v30_workspace(
        tmp_path,
        accepted=(0, 2),
    )
    original_open_lock = converter._open_lock
    original_flock = converter.fcntl.flock
    lock_descriptor = -1

    def tracked_open_lock(path):
        nonlocal lock_descriptor
        lock_descriptor = original_open_lock(path)
        return lock_descriptor

    def failing_unlock(descriptor, operation):
        if descriptor == lock_descriptor and operation == converter.fcntl.LOCK_UN:
            raise OSError("injected output lock release failure")
        return original_flock(descriptor, operation)

    monkeypatch.setattr(converter, "_open_lock", tracked_open_lock)
    monkeypatch.setattr(converter.fcntl, "flock", failing_unlock)

    with pytest.raises(ValueError, match=r"conversion finalization cleanup failed"):
        converter.convert_dataset(
            work,
            tmp_path / "accepted",
            accepted_only=True,
            services=services,
        )

    assert not (tmp_path / "accepted").exists()
    assert not list(tmp_path.glob("accepted.staging-*"))
    assert not list(tmp_path.glob(".writer-quarantine-*"))


def test_v30_accepted_only_source_change_during_stats_aborts_publication(
    tmp_path: Path,
) -> None:
    from robo_annotate.converter import convert_dataset
    from robo_annotate.stats import iter_video_rgb_frames

    work, source, services = selectively_accepted_v30_workspace(
        tmp_path,
        accepted=(0, 2),
    )
    mutated = False

    def racing(path: Path):
        nonlocal mutated
        for frame in iter_video_rgb_frames(path):
            if not mutated:
                (source / "meta/source-race.txt").write_text(
                    "changed",
                    encoding="utf-8",
                )
                mutated = True
            yield frame

    services["iter_video_rgb_frames"] = racing

    with pytest.raises(ValueError, match=r"source dataset changed"):
        convert_dataset(
            work,
            tmp_path / "accepted",
            accepted_only=True,
            services=services,
        )

    assert mutated
    assert not (tmp_path / "accepted").exists()
    assert not list(tmp_path.glob("accepted.staging-*"))


@pytest.mark.parametrize("relative", ["meta/info.json", "meta/stats.json"])
def test_v30_composer_rejects_replaced_source_json_before_outside_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative: str,
) -> None:
    import robo_annotate.converter_v30 as converter_v30
    from robo_annotate.converter import convert_dataset

    work, source, services = selectively_accepted_v30_workspace(
        tmp_path,
        accepted=(0, 2),
    )
    target = source / relative
    held = target.with_name(f"held-{target.name}")
    outside = tmp_path / f"outside-{target.name}"
    outside.write_text('{"outside":"must never be parsed"}', encoding="utf-8")
    original_open = converter_v30.SecureTree.open_file
    original_loads = converter_v30.json.loads
    swapped = False
    outside_parse_attempted = False

    def racing_open(tree, name, maximum, context):
        nonlocal swapped
        if (
            tree.label == "v3 composer source"
            and str(name) == relative
            and not swapped
        ):
            target.rename(held)
            target.symlink_to(outside)
            swapped = True
        return original_open(tree, name, maximum, context)

    def guarded_loads(document, *args, **kwargs):
        nonlocal outside_parse_attempted
        if '"outside":"must never be parsed"' in document:
            outside_parse_attempted = True
        return original_loads(document, *args, **kwargs)

    monkeypatch.setattr(converter_v30.SecureTree, "open_file", racing_open)
    monkeypatch.setattr(converter_v30.json, "loads", guarded_loads)

    with pytest.raises(ValueError, match=r"changed|symbolic link"):
        convert_dataset(
            work,
            tmp_path / "accepted",
            accepted_only=True,
            services=services,
        )

    assert swapped
    assert not outside_parse_attempted
    assert outside.read_text(encoding="utf-8") == (
        '{"outside":"must never be parsed"}'
    )
    assert not (tmp_path / "accepted").exists()


@pytest.mark.parametrize("component", ["root", "meta"])
def test_v30_composer_detects_restored_source_directory_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    component: str,
) -> None:
    import robo_annotate.converter_v30 as converter_v30
    from robo_annotate.converter import convert_dataset

    work, source, services = selectively_accepted_v30_workspace(
        tmp_path,
        accepted=(0, 2),
    )
    original_scan = converter_v30.SecureTree.scan
    original_loads = converter_v30.json.loads
    replaced = False
    outside_parse_attempted = False

    def racing_scan(tree):
        nonlocal replaced
        result = original_scan(tree)
        if tree.label == "v3 composer source" and not replaced:
            target = source if component == "root" else source / "meta"
            held = tmp_path / f"held-{component}"
            outside = tmp_path / f"outside-{component}"
            outside.mkdir()
            (outside / "info.json").write_text(
                '{"outside":"must never be parsed"}',
                encoding="utf-8",
            )
            target.rename(held)
            target.symlink_to(outside, target_is_directory=True)
            target.unlink()
            held.rename(target)
            replaced = True
        return result

    def guarded_loads(document, *args, **kwargs):
        nonlocal outside_parse_attempted
        if '"outside":"must never be parsed"' in document:
            outside_parse_attempted = True
        return original_loads(document, *args, **kwargs)

    monkeypatch.setattr(converter_v30.SecureTree, "scan", racing_scan)
    monkeypatch.setattr(converter_v30.json, "loads", guarded_loads)

    with pytest.raises(ValueError, match=r"changed"):
        convert_dataset(
            work,
            tmp_path / "accepted",
            accepted_only=True,
            services=services,
        )

    assert replaced
    assert not outside_parse_attempted
    assert not (tmp_path / "accepted").exists()


def test_full_v30_conversion_rechecks_source_digest_before_publish(
    tmp_path: Path,
) -> None:
    from robo_annotate.converter import convert_dataset
    from tests.test_release_validator_v30 import accepted_v30_workspace

    work, source, services = accepted_v30_workspace(tmp_path)
    source_task_info = source / "meta/task_info"
    source_task_info.mkdir()
    excluded_from_core_comparison = source_task_info / "task_0.json"
    excluded_from_core_comparison.write_text("source marker", encoding="utf-8")

    def mutate_source_during_validation(ref, camera, indices):
        excluded_from_core_comparison.write_text("changed marker", encoding="utf-8")
        return [
            type("Sample", (), {"frame_index": index, "camera_key": camera})()
            for index in indices
        ]

    services["extract_frames"] = mutate_source_during_validation
    output = tmp_path / "changed-source"

    with pytest.raises(ValueError, match=r"source dataset changed during conversion"):
        convert_dataset(work, output, services=services)

    assert not output.exists()
    assert not list(tmp_path.glob("changed-source.staging-*"))

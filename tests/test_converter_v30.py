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
) -> tuple[Path, Path, dict]:
    source = make_lerobot_v30_fixture(tmp_path)
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
    assert info["data_files_size_in_mb"] == pytest.approx(
        sum(path.stat().st_size for path in report.output.glob("data/**/*.parquet"))
        / (2**20)
    )
    assert info["video_files_size_in_mb"] == pytest.approx(
        sum(path.stat().st_size for path in report.output.glob("videos/**/*.mp4"))
        / (2**20)
    )
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

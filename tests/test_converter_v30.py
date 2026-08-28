import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

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


def test_v30_accepted_only_waits_for_shared_shard_repacking(
    tmp_path: Path,
) -> None:
    from robo_annotate.converter import convert_dataset
    from tests.test_release_validator_v30 import accepted_v30_workspace

    work, _, services = accepted_v30_workspace(tmp_path)
    output = tmp_path / "accepted-only"

    with pytest.raises(ValueError, match=r"shared-shard repacking"):
        convert_dataset(work, output, accepted_only=True, services=services)

    assert not output.exists()
    assert not list(tmp_path.glob("accepted-only.staging-*"))


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

"""Version-neutral publication of Robo annotation metadata."""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .augmentation import AUGMENTATION_PROMPT_VERSION
from .workspace import EpisodeRecord, RunManifest


_MAX_JSON_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class SelectedEpisode:
    """Bind one source annotation record to its published episode identity."""

    record: EpisodeRecord
    source_index: int
    output_index: int
    length: int


def write_public_annotations(
    staging: Path,
    output: Path,
    manifest: RunManifest,
    selected: Sequence[SelectedEpisode],
    converted_at: datetime,
    augmented_texts: Mapping[int, list[str]] | None,
    *,
    extend_info: bool,
) -> None:
    """Write stable Robo metadata, optionally extending legacy LeRobot info."""
    template = [item.model_dump(mode="json") for item in manifest.subtasks]
    instruction_map = {
        str(item.output_index): manifest.high_level_instruction for item in selected
    }
    if extend_info:
        info_path = staging / "meta/info.json"
        info = read_bounded_json(info_path)
        info["subtask_template"] = template
        info["high_level_instruction"] = instruction_map
        atomic_json(info_path, info)

    episodes: dict[str, dict[str, Any]] = {}
    task_info = []
    for item in selected:
        record = item.record
        annotation = record.final_annotation
        if annotation is None:
            raise ValueError("selected episode must have a final annotation")
        entry: dict[str, Any] = {"episode_index": item.output_index}
        if manifest.mode == "dagger_patch":
            entry["start_subtask_index"] = annotation.start_subtask_index
        entry.update(
            {
                "boundaries": list(annotation.boundaries),
                "high_level_instruction": manifest.high_level_instruction,
                "saved_at": record.updated_at.astimezone(UTC).isoformat(),
            }
        )
        episodes[str(item.output_index)] = entry
        starts = [0, *annotation.boundaries]
        ends = [*annotation.boundaries, item.length]
        selected_subtasks = manifest.subtasks[
            annotation.start_subtask_index : annotation.start_subtask_index
            + len(starts)
        ]
        action_texts = (
            [subtask.text for subtask in selected_subtasks]
            if augmented_texts is None
            else augmented_texts[item.source_index]
        )
        actions = [
            {
                "start_frame": start,
                "end_frame": end,
                "action_text": action_text,
                "skill": subtask.skill,
            }
            for start, end, subtask, action_text in zip(
                starts, ends, selected_subtasks, action_texts, strict=True
            )
        ]
        task_info.append(
            {
                "episode_id": item.output_index,
                "task_id": 0,
                "task_name": manifest.high_level_instruction,
                "label_info": {"action_config": actions},
            }
        )

    annotations: dict[str, Any] = {
        "source_root": str(manifest.dataset_root),
        "work_dir": str(output / "meta"),
        "subtask_template": template,
        "episodes": episodes,
        "primary_camera": manifest.effective_config["primary_camera"],
        "updated_at": converted_at.isoformat(),
    }
    if augmented_texts is not None:
        annotations["augmentation"] = {
            "enabled": True,
            "language": manifest.effective_config["augmentation"]["language"],
            "model_repo": manifest.model_repo,
            "model_revision": manifest.model_revision,
            "prompt_version": AUGMENTATION_PROMPT_VERSION,
        }
    atomic_json(
        staging / "meta/lerobot_annotations.json", annotations, sort_keys=False
    )
    task_dir = staging / "meta/task_info"
    task_dir.mkdir(exist_ok=True)
    atomic_json(task_dir / "task_0.json", task_info, sort_keys=False)


def read_bounded_json(path: Path) -> dict[str, Any]:
    """Read one bounded regular JSON object without following a symlink."""
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > _MAX_JSON_BYTES
    ):
        raise ValueError("unsafe info.json")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(constant)
            ),
        )
    except Exception as exc:
        raise ValueError("invalid source info.json") from exc
    if not isinstance(value, dict):
        raise ValueError("info.json must be an object")
    return value


def atomic_json(path: Path, value: Any, *, sort_keys: bool = True) -> None:
    """Atomically replace one JSON file and fsync its containing directory."""
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=sort_keys,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    name = f".{path.name}.{secrets.token_hex(12)}.tmp"
    fd = None
    try:
        fd = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        name = ""
        os.fsync(directory_fd)
    finally:
        if fd is not None:
            os.close(fd)
        if name:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)

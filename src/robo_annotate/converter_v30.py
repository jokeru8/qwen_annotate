"""Byte-preserving publication helpers for ordinary LeRobot v3.0 releases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from .publication_metadata import SelectedEpisode, write_public_annotations
from .workspace import EpisodeRecord, RunManifest


def write_full_v30_release(
    staging: Path,
    output: Path,
    manifest: RunManifest,
    records: Sequence[EpisodeRecord],
    converted_at: datetime,
    augmented_texts: Mapping[int, list[str]] | None,
) -> None:
    """Append only Robo metadata to an unchanged official v3.0 tree copy."""
    if manifest.dataset_version != "v3.0":
        raise ValueError("full v3 release writer requires a LeRobot v3.0 manifest")
    ordered = sorted(records, key=lambda record: record.episode_index)
    if [record.episode_index for record in ordered] != list(
        range(manifest.total_episodes)
    ):
        raise ValueError("full v3 release records must cover contiguous source episodes")
    selected = [
        SelectedEpisode(
            record=record,
            source_index=record.episode_index,
            output_index=record.episode_index,
            length=manifest.episode_lengths[record.episode_index],
        )
        for record in ordered
    ]
    write_public_annotations(
        staging,
        output,
        manifest,
        selected,
        converted_at,
        augmented_texts,
        extend_info=False,
    )

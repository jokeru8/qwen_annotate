"""Read-only inspection of LeRobot v2.1 datasets."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from robo_annotate.config import AnnotationConfig

from .lerobot import (
    DatasetIndex,
    EpisodeDataRef,
    EpisodeInfo,
    EpisodeVideoRef,
    VideoProbe,
    _camera_keys,
    _episode_tasks,
    _format_path,
    _nonnegative_int,
    _positive_int,
    _positive_number,
    _read_jsonl,
    _require_configured_cameras,
    _required_int,
    _required_string,
    _resolve_dataset_path,
    _task_texts,
    _validate_path_template,
    _verify_parquet_rows,
    video_fps_matches,
)


def inspect_v21_dataset(
    config: AnnotationConfig,
    info: dict[str, Any],
    probe: Callable[[Path], VideoProbe],
) -> DatasetIndex:
    """Validate and index a local LeRobot v2.1 dataset without writing to it."""
    root = config.source.resolve()
    version = _required_string(info, "codebase_version", "info.json")
    if version != "v2.1":
        raise ValueError(f"LeRobot codebase_version must be v2.1, got {version!r}")

    fps = _positive_number(info, "fps", "info.json")
    chunks_size = _positive_int(info, "chunks_size", "info.json")
    total_episodes = _nonnegative_int(info, "total_episodes", "info.json")
    expected_total_frames = _nonnegative_int(info, "total_frames", "info.json")
    data_path_format = _required_string(info, "data_path", "info.json")
    video_path_format = _required_string(info, "video_path", "info.json")
    _validate_path_template(data_path_format, "data_path", {"episode_index"})
    _validate_path_template(video_path_format, "video_path", {"episode_index", "video_key"})
    camera_keys = _camera_keys(info)
    _require_configured_cameras(config, camera_keys)
    task_texts = _task_texts(root / "meta" / "tasks.jsonl")
    total_tasks = _nonnegative_int(info, "total_tasks", "info.json")
    if len(task_texts) != total_tasks:
        raise ValueError(
            f"info.total_tasks is {total_tasks}, but tasks.jsonl has {len(task_texts)} rows"
        )
    episode_rows = _read_jsonl(root / "meta" / "episodes.jsonl")

    if len(episode_rows) != total_episodes:
        raise ValueError(
            f"info.total_episodes is {total_episodes}, but episodes.jsonl has {len(episode_rows)} rows"
        )
    total_chunks = _nonnegative_int(info, "total_chunks", "info.json")
    expected_chunks = (total_episodes + chunks_size - 1) // chunks_size
    if total_chunks != expected_chunks:
        raise ValueError(f"info.total_chunks is {total_chunks}, expected {expected_chunks}")

    episodes: list[EpisodeInfo] = []
    total_frames = 0
    seen_parquets: set[Path] = set()
    seen_videos: set[Path] = set()
    for expected_index, row in enumerate(episode_rows):
        episode_index = _required_int(row, "episode_index", f"episodes.jsonl row {expected_index}")
        if episode_index != expected_index:
            raise ValueError("Episode indices must be contiguous from 0 through N-1")
        length = _positive_int(row, "length", f"episodes.jsonl row {expected_index}")
        episode_tasks = _episode_tasks(row, expected_index)
        for task in episode_tasks:
            if task not in task_texts.values():
                raise ValueError(
                    f"Episode {episode_index} references task {task!r}, absent from meta/tasks.jsonl"
                )
        if len(episode_tasks) != 1:
            raise ValueError(
                "First release supports exactly one task per episode; "
                f"episode {episode_index} has {len(episode_tasks)}"
            )
        task = episode_tasks[0]
        values = {
            "episode_chunk": episode_index // chunks_size,
            "episode_index": episode_index,
        }
        parquet = _resolve_dataset_path(
            root,
            _format_path(data_path_format, values, "data_path"),
            "data_path",
        )
        if parquet in seen_parquets:
            raise ValueError(f"data_path resolves duplicate parquet path: {parquet}")
        seen_parquets.add(parquet)
        _verify_parquet_rows(parquet, length, episode_index)
        videos: dict[str, EpisodeVideoRef] = {}
        for video_key in camera_keys:
            video = _resolve_dataset_path(
                root,
                _format_path(
                    video_path_format,
                    values | {"video_key": video_key},
                    "video_path",
                ),
                "video_path",
            )
            if video in seen_videos:
                raise ValueError(f"video_path resolves duplicate video path: {video}")
            seen_videos.add(video)
            if not video.is_file():
                raise FileNotFoundError(f"Missing video for episode {episode_index}: {video}")
            video_probe = probe(video)
            if video_probe.frames != length:
                raise ValueError(
                    f"Video frame count for episode {episode_index}, camera {video_key!r} "
                    f"is {video_probe.frames}, expected {length}"
                )
            if not video_fps_matches(video_probe.fps, fps):
                raise ValueError(
                    f"Video fps for episode {episode_index}, camera {video_key!r} "
                    f"is {video_probe.fps}, expected {fps}"
                )
            videos[video_key] = EpisodeVideoRef(
                path=video,
                from_timestamp=0.0,
                to_timestamp=length / fps,
                fps=fps,
            )
        episodes.append(
            EpisodeInfo(
                episode_index=episode_index,
                length=length,
                task=task,
                data=EpisodeDataRef(
                    path=parquet,
                    dataset_from_index=0,
                    dataset_to_index=length,
                ),
                videos=videos,
            )
        )
        total_frames += length

    if total_frames != expected_total_frames:
        raise ValueError(
            f"info.total_frames is {expected_total_frames}, but episode lengths total {total_frames}"
        )
    if "total_videos" in info:
        total_videos = _nonnegative_int(info, "total_videos", "info.json")
        expected_total_videos = total_episodes * len(camera_keys)
        if total_videos != expected_total_videos:
            raise ValueError(
                f"info.total_videos is {total_videos}, expected {expected_total_videos}"
            )
    return DatasetIndex(
        root=root,
        version="v2.1",
        fps=fps,
        camera_keys=camera_keys,
        episodes=episodes,
    )

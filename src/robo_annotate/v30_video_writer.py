"""Re-encode selected LeRobot v3.0 episode video slices into fresh shards."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from string import Formatter
from typing import Any

import av
import numpy as np

from .lerobot import (
    DatasetIndex,
    EpisodeInfo,
    EpisodeVideoRef,
    video_fps_matches,
)
from .secure_tree import SecureFile, SecureTree
from .writer_publication import (
    OwnedFile,
    WriterPublication,
    _ChildDirectory,
    _CleanupFailures,
    _entry_at,
)


_MAX_VIDEO_BYTES = 16 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class VideoPlacement:
    """Describe one output episode slice in one rebuilt camera shard."""

    source_index: int
    output_index: int
    camera_key: str
    chunk_index: int
    file_index: int
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class V30VideoWriteResult:
    """Return per-camera episode placements and rebuilt MP4 paths."""

    placements: dict[str, tuple[VideoPlacement, ...]]
    files_by_camera: dict[str, tuple[Path, ...]]


@dataclass(frozen=True)
class _VideoSpec:
    width: int
    height: int
    fps: float


def write_v30_video_subset(
    staging: Path,
    dataset: DatasetIndex,
    source_indices: Sequence[int],
    info: Mapping[str, Any],
) -> V30VideoWriteResult:
    """Rebuild selected v3 video slices in deterministic caller order."""
    (
        staging,
        selected,
        cameras,
        specs,
        chunks_size,
        size_limit,
        video_template,
    ) = _validate_request(staging, dataset, source_indices, info)
    source_root = Path(os.path.abspath(dataset.root))
    placements: dict[str, tuple[VideoPlacement, ...]] = {}
    files_by_camera: dict[str, tuple[Path, ...]] = {}

    with SecureTree(source_root, "source") as tree:
        tree.scan()
        opened_sources = _open_selected_video_files(tree, selected, cameras)
        publication = WriterPublication.open(
            staging,
            tree,
            ".v30-video-",
            "v3 video publication",
            bundle_context="staging video bundle",
        )
        publication.add_close_action("source tree close", tree.close)
        committed = False
        primary_error: BaseException | None = None
        try:
            if _entry_at(publication.staging.descriptor, "videos") is not None:
                raise ValueError("unsafe staging videos entry already exists")
            directory_cache: dict[tuple[str, ...], _ChildDirectory] = {}
            directory_guards: dict[tuple[str, ...], Callable[[], bool]] = {
                (): publication.bundle_is_attached,
            }

            def ensure_directory(parts: tuple[str, ...]) -> _ChildDirectory:
                if not parts:
                    return publication.private_bundle
                existing = directory_cache.get(parts)
                if existing is not None:
                    return existing
                parent_parts = parts[:-1]
                parent = ensure_directory(parent_parts)
                parent_guard = directory_guards[parent_parts]
                child = publication.create_directory(
                    parent,
                    parts[-1],
                    f"staged video directory {'/'.join(parts)}",
                    parent_guard,
                )

                def child_guard(
                    inherited: Callable[[], bool] = parent_guard,
                    current: _ChildDirectory = child,
                ) -> bool:
                    return inherited() and current.is_attached()

                directory_cache[parts] = child
                directory_guards[parts] = child_guard
                return child

            for camera in cameras:
                groups = _pack_episodes(selected, specs[camera], size_limit)
                camera_placements: list[VideoPlacement] = []
                camera_files: list[Path] = []
                output_index_by_source = {
                    episode.episode_index: output_index
                    for output_index, episode in enumerate(selected)
                }
                for file_number, episodes in enumerate(groups):
                    chunk_index = file_number // chunks_size
                    file_index = file_number % chunks_size
                    relative = _render_video_path(
                        video_template,
                        camera,
                        chunk_index,
                        file_index,
                    )
                    parent_parts = relative.parts[:-1]
                    parent = ensure_directory(parent_parts)
                    parent_guard = directory_guards[parent_parts]
                    output = publication.create_file(
                        parent,
                        relative.name,
                        f"staged video file {relative.as_posix()}",
                        parent_guard,
                    )
                    written, episode_ranges = _encode_video_file(
                        output,
                        episodes,
                        camera,
                        specs[camera],
                        opened_sources,
                    )
                    _validate_output_video(output, specs[camera], written)
                    for episode, start, stop in episode_ranges:
                        camera_placements.append(
                            VideoPlacement(
                                source_index=episode.episode_index,
                                output_index=output_index_by_source[
                                    episode.episode_index
                                ],
                                camera_key=camera,
                                chunk_index=chunk_index,
                                file_index=file_index,
                                from_timestamp=start / specs[camera].fps,
                                to_timestamp=stop / specs[camera].fps,
                            )
                        )
                    camera_files.append(staging / relative)
                placements[camera] = tuple(
                    sorted(camera_placements, key=lambda item: item.output_index)
                )
                files_by_camera[camera] = tuple(camera_files)

            videos = directory_cache.get(("videos",))
            if videos is None:
                raise ValueError("video_path must render below videos")
            videos.verify("publication")
            publication.staging.verify("publication")
            publication.private_bundle.verify("publication")
            publication.publish(
                publication.private_bundle.descriptor,
                "videos",
                videos.identity,
                publication.staging.descriptor,
                "videos",
                publication.staging_is_attached,
                "staging videos",
            )
            source_anchor = publication.source_anchor
            if source_anchor is None:
                raise ValueError("source path is not anchored")
            source_anchor.verify("video publication")
            try:
                tree.verify()
            except ValueError as exc:
                raise ValueError("source changed during video publication") from exc
            publication.staging.verify("publication")
            if _entry_at(publication.staging.descriptor, "videos") != videos.identity:
                raise ValueError("staging videos changed during publication")
            os.fsync(publication.staging.descriptor)
            if _entry_at(publication.staging.descriptor, "videos") != videos.identity:
                raise ValueError("staging videos changed during publication")
            committed = True
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            publication.finish(primary_error, committed=committed)

    return V30VideoWriteResult(
        placements=placements,
        files_by_camera=files_by_camera,
    )


def _validate_request(
    staging: Path,
    dataset: DatasetIndex,
    source_indices: Sequence[int],
    info: Mapping[str, Any],
) -> tuple[
    Path,
    list[EpisodeInfo],
    tuple[str, ...],
    dict[str, _VideoSpec],
    int,
    float,
    str,
]:
    if not isinstance(staging, Path):
        raise TypeError("staging must be a Path")
    if not isinstance(dataset, DatasetIndex) or dataset.version != "v3.0":
        raise ValueError("video subset writer requires a LeRobot v3.0 dataset index")
    if not isinstance(info, Mapping) or info.get("codebase_version") != "v3.0":
        raise ValueError("info must declare LeRobot codebase_version v3.0")
    if isinstance(source_indices, (str, bytes)) or not isinstance(
        source_indices, Sequence
    ):
        raise TypeError("source_indices must be a sequence of integers")
    indices = list(source_indices)
    if not indices:
        raise ValueError("source_indices must select at least one episode")
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("source_indices must contain nonnegative integers")
    if len(indices) != len(set(indices)):
        raise ValueError("source_indices must be unique")

    source_root = Path(os.path.abspath(dataset.root))
    absolute_staging = Path(os.path.abspath(staging))
    if (
        absolute_staging == source_root
        or source_root in absolute_staging.parents
        or absolute_staging in source_root.parents
    ):
        raise ValueError("staging and source trees must not overlap")

    by_index: dict[int, EpisodeInfo] = {}
    for episode in dataset.episodes:
        if episode.episode_index in by_index:
            raise ValueError("dataset episode indices must be unique")
        by_index[episode.episode_index] = episode
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise ValueError(f"source_indices contains unknown episode indices: {missing}")
    selected = [by_index[index] for index in indices]

    cameras = tuple(dataset.camera_keys)
    if not cameras or len(cameras) != len(set(cameras)) or any(
        not isinstance(camera, str) or not camera for camera in cameras
    ):
        raise ValueError("dataset camera keys must be unique and nonempty")
    fps = _positive_finite_number(info.get("fps"), "info.fps")
    if not video_fps_matches(fps, dataset.fps):
        raise ValueError("info.fps does not match the inspected dataset fps")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("info.features must be an object")
    specs = {
        camera: _video_spec(camera, features.get(camera), fps)
        for camera in cameras
    }
    for episode in selected:
        if set(episode.videos) != set(cameras):
            raise ValueError(
                f"episode {episode.episode_index} camera references are incomplete"
            )

    chunks_size = info.get("chunks_size", 1000)
    if type(chunks_size) is not int or chunks_size <= 0:
        raise ValueError("info.chunks_size must be a positive integer")
    size_mb = _positive_finite_number(
        info.get("video_files_size_in_mb", 200),
        "info.video_files_size_in_mb",
    )
    video_template = info.get("video_path")
    if not isinstance(video_template, str) or not video_template:
        raise ValueError("info.video_path must be a nonempty string")
    _validate_video_template(video_template)
    return (
        absolute_staging,
        selected,
        cameras,
        specs,
        chunks_size,
        size_mb * 1024 * 1024,
        video_template,
    )


def _video_spec(camera: str, value: object, fps: float) -> _VideoSpec:
    if not isinstance(value, Mapping) or value.get("dtype") != "video":
        raise ValueError(f"video feature declaration is invalid: {camera}")
    details = value.get("info")
    shape = value.get("shape")
    if not isinstance(details, Mapping) or not isinstance(shape, list):
        raise ValueError(f"video feature declaration is invalid: {camera}")
    width = details.get("video.width")
    height = details.get("video.height")
    declared_fps = details.get("video.fps")
    if type(width) is not int or width <= 0 or type(height) is not int or height <= 0:
        raise ValueError(f"video feature dimensions are invalid: {camera}")
    if not video_fps_matches(
        _positive_finite_number(declared_fps, f"{camera} video.fps"),
        fps,
    ):
        raise ValueError(f"video feature fps does not match info.fps: {camera}")
    if shape not in ([height, width, 3], [3, height, width]):
        raise ValueError(f"video feature shape is invalid: {camera}")
    return _VideoSpec(width=width, height=height, fps=fps)


def _positive_finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{context} must be a positive finite number")
    return number


def _open_selected_video_files(
    tree: SecureTree,
    selected: Sequence[EpisodeInfo],
    cameras: Sequence[str],
) -> dict[Path, SecureFile]:
    result: dict[Path, SecureFile] = {}
    for episode in selected:
        for camera in cameras:
            path = Path(os.path.abspath(episode.videos[camera].path))
            if path in result:
                continue
            try:
                relative = path.relative_to(tree.path)
            except ValueError as exc:
                raise ValueError("video reference escapes the source tree") from exc
            result[path] = tree.open_file(
                relative,
                _MAX_VIDEO_BYTES,
                "video shard",
            )
    return result


def _pack_episodes(
    selected: Sequence[EpisodeInfo],
    spec: _VideoSpec,
    size_limit: float,
) -> list[list[EpisodeInfo]]:
    groups: list[list[EpisodeInfo]] = []
    current: list[EpisodeInfo] = []
    current_weight = 0
    for episode in selected:
        weight = spec.width * spec.height * 3 * episode.length
        if current and current_weight + weight > size_limit:
            groups.append(current)
            current = []
            current_weight = 0
        current.append(episode)
        current_weight += weight
    if current:
        groups.append(current)
    return groups


def _render_video_path(
    template: str,
    camera: str,
    chunk_index: int,
    file_index: int,
) -> PurePosixPath:
    try:
        rendered = template.format(
            video_key=camera,
            chunk_index=chunk_index,
            file_index=file_index,
        )
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError("info.video_path could not be rendered") from exc
    path = PurePosixPath(rendered)
    if (
        not rendered
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or len(path.parts) < 2
        or path.parts[0] != "videos"
        or path.suffix != ".mp4"
    ):
        raise ValueError("info.video_path must render a safe MP4 below videos")
    return path


def _validate_video_template(template: str) -> None:
    expected = {"video_key", "chunk_index", "file_index"}
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as exc:
        raise ValueError("info.video_path is malformed") from exc
    fields: list[str] = []
    for _, field, format_spec, conversion in parsed:
        if field is None:
            continue
        if (
            field not in expected
            or conversion is not None
            or "{" in format_spec
            or "}" in format_spec
        ):
            raise ValueError("info.video_path contains unsafe template fields")
        fields.append(field)
    if len(fields) != len(expected) or set(fields) != expected:
        raise ValueError(
            "info.video_path fields must be exactly "
            "['chunk_index', 'file_index', 'video_key']"
        )


def _encode_video_file(
    output: OwnedFile,
    episodes: Sequence[EpisodeInfo],
    camera: str,
    spec: _VideoSpec,
    opened_sources: Mapping[Path, SecureFile],
) -> tuple[int, tuple[tuple[EpisodeInfo, int, int], ...]]:
    rate = Fraction(spec.fps).limit_denominator(1_000_000)
    container: av.container.OutputContainer | None = None
    primary_error: BaseException | None = None
    frames_written = 0
    episode_ranges: list[tuple[EpisodeInfo, int, int]] = []
    try:
        container = av.open(str(output.proc_path), mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=rate)
        stream.width = spec.width
        stream.height = spec.height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(rate.denominator, rate.numerator)
        frame_time_base = stream.time_base
        stream.options = {"crf": "0", "preset": "fast"}
        for episode in episodes:
            episode_start = frames_written
            reference = episode.videos[camera]
            opened = opened_sources[Path(os.path.abspath(reference.path))]
            frames = _decode_episode_frames(
                opened,
                reference,
                episode.length,
                spec,
            )
            for array in frames:
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                frame.pts = frames_written
                frame.time_base = frame_time_base
                for packet in stream.encode(frame):
                    container.mux(packet)
                frames_written += 1
            episode_ranges.append((episode, episode_start, frames_written))
        for packet in stream.encode():
            container.mux(packet)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        failures = _CleanupFailures()
        if container is not None:
            failures.attempt("video container close", container.close)
        failures.finish(primary_error, "video encoding")
    os.fsync(output.descriptor)
    output.verify("video encoding")
    return frames_written, tuple(episode_ranges)


def _decode_episode_frames(
    opened: SecureFile,
    reference: EpisodeVideoRef,
    expected_length: int,
    spec: _VideoSpec,
) -> Iterator[np.ndarray[Any, np.dtype[np.uint8]]]:
    if not isinstance(reference, EpisodeVideoRef):
        raise TypeError("video reference must be an EpisodeVideoRef")
    if type(expected_length) is not int or expected_length <= 0:
        raise ValueError("expected video length must be a positive integer")
    duration_frames = (
        reference.to_timestamp - reference.from_timestamp
    ) * reference.fps
    if (
        round(duration_frames) != expected_length
        or abs(duration_frames - expected_length) > 1.0
    ):
        raise ValueError("episode video duration does not match episode length")
    if not video_fps_matches(reference.fps, spec.fps):
        raise ValueError("episode video fps does not match feature fps")

    container: av.container.InputContainer | None = None
    primary_error: BaseException | None = None
    try:
        container = av.open(str(opened.proc_path), mode="r")
        streams = [item for item in container.streams if item.type == "video"]
        if len(streams) != 1:
            raise ValueError("video shard must contain exactly one video stream")
        stream = streams[0]
        try:
            measured_fps = float(stream.average_rate)
            time_base = float(stream.time_base)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError("video shard has invalid timing metadata") from exc
        if not video_fps_matches(measured_fps, spec.fps):
            raise ValueError("video shard fps does not match feature fps")
        if not math.isfinite(time_base) or time_base <= 0:
            raise ValueError("video shard has invalid stream time base")
        if (stream.width, stream.height) != (spec.width, spec.height):
            raise ValueError("video shard dimensions do not match feature shape")

        seek_time = max(0.0, reference.from_timestamp - 1.0 / spec.fps)
        container.seek(
            math.floor(seek_time / time_base),
            backward=True,
            any_frame=False,
            stream=stream,
        )
        tolerance = 0.5 / spec.fps + 1e-9
        next_index = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("video shard contains a frame without PTS")
            media_time = float(frame.pts * stream.time_base)
            if not math.isfinite(media_time):
                raise ValueError("video shard contains a frame with invalid PTS")
            if media_time < reference.from_timestamp:
                continue
            if media_time >= reference.to_timestamp:
                break
            local_index = round(
                (media_time - reference.from_timestamp) * spec.fps
            )
            if not 0 <= local_index < expected_length:
                continue
            expected_time = reference.from_timestamp + local_index / spec.fps
            if not math.isclose(
                media_time,
                expected_time,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError("video shard contains invalid episode-local PTS")
            if local_index < next_index:
                raise ValueError("video shard contains duplicate episode-local frames")
            if local_index > next_index:
                missing = list(range(next_index, local_index))
                raise ValueError(f"video shard is missing episode frame(s): {missing}")
            array = frame.to_ndarray(format="rgb24")
            if array.shape != (spec.height, spec.width, 3):
                raise ValueError("decoded video frame has an invalid shape")
            if array.dtype != np.uint8:
                raise ValueError("decoded video frame must be uint8 RGB")
            next_index += 1
            yield array
        if next_index != expected_length:
            missing = list(range(next_index, expected_length))
            raise ValueError(f"video shard is missing episode frame(s): {missing}")
        opened.verify()
    except ValueError as exc:
        primary_error = exc
        raise
    except Exception as exc:
        error = ValueError(f"unable to decode video shard: {opened.relative}")
        primary_error = error
        raise error from exc
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        failures = _CleanupFailures()
        if container is not None:
            failures.attempt("source video container close", container.close)
        failures.finish(primary_error, "source video")


def _validate_output_video(
    output: OwnedFile,
    spec: _VideoSpec,
    expected_frames: int,
) -> None:
    output.verify("output validation")
    container: av.container.InputContainer | None = None
    primary_error: BaseException | None = None
    try:
        container = av.open(str(output.proc_path), mode="r")
        streams = [item for item in container.streams if item.type == "video"]
        if len(streams) != 1:
            raise ValueError("rebuilt video must contain exactly one video stream")
        stream = streams[0]
        try:
            measured_fps = float(stream.average_rate)
            time_base = float(stream.time_base)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError("rebuilt video has invalid timing metadata") from exc
        if not video_fps_matches(measured_fps, spec.fps):
            raise ValueError("rebuilt video fps does not match feature fps")
        if not math.isfinite(time_base) or time_base <= 0:
            raise ValueError("rebuilt video has invalid stream time base")
        if (stream.width, stream.height) != (spec.width, spec.height):
            raise ValueError("rebuilt video dimensions do not match feature shape")
        frame_count = 0
        tolerance = 0.5 / spec.fps + 1e-9
        for frame_count, frame in enumerate(container.decode(stream), start=1):
            if frame.pts is None:
                raise ValueError("rebuilt video contains a frame without PTS")
            media_time = float(frame.pts * stream.time_base)
            expected_time = (frame_count - 1) / spec.fps
            if not math.isfinite(media_time) or not math.isclose(
                media_time,
                expected_time,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError("rebuilt video is not CFR")
        if frame_count != expected_frames:
            raise ValueError("rebuilt video frame count does not match encoded frames")
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        failures = _CleanupFailures()
        if container is not None:
            failures.attempt("rebuilt video container close", container.close)
        failures.finish(primary_error, "rebuilt video validation")
    output.verify("output validation")

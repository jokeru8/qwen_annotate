"""Exact, camera-labeled frame evidence extracted from local videos."""

import base64
import io
import math
from typing import Any

from PIL import ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .lerobot import EpisodeVideoRef, video_fps_matches


class FrameSample(BaseModel):
    """A JPEG evidence frame with its source camera and exact frame timestamp."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    camera_key: str = Field(min_length=1)
    frame_index: int = Field(ge=0, strict=True)
    timestamp_seconds: float = Field(ge=0)
    jpeg: bytes = Field(min_length=1, strict=True)

    @field_validator("timestamp_seconds", mode="before")
    @classmethod
    def timestamp_must_be_a_finite_number(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("timestamp_seconds must be a finite number")
        if not math.isfinite(float(value)):
            raise ValueError("timestamp_seconds must be finite")
        return value


def uniform_indices(frame_count: int, source_fps: float, target_fps: float, max_frames: int) -> list[int]:
    """Choose evenly distributed frame indices while preserving both endpoints."""
    _require_int(frame_count, "frame_count", minimum=1)
    source_fps = _require_positive_finite_number(source_fps, "source_fps")
    target_fps = _require_positive_finite_number(target_fps, "target_fps")
    _require_int(max_frames, "max_frames", minimum=1)

    if frame_count == 1:
        return [0]
    if max_frames == 1:
        raise ValueError("max_frames must allow two endpoint samples when frame_count is greater than one")

    duration_seconds = (frame_count - 1) / source_fps
    desired_count = max(2, int(round(duration_seconds * target_fps)) + 1)
    sample_count = min(frame_count, max_frames, desired_count)
    if sample_count == 1:
        return [0]
    return [round(position * (frame_count - 1) / (sample_count - 1)) for position in range(sample_count)]


def window_indices(center: int, radius_frames: int, stride: int, frame_count: int) -> list[int]:
    """Return a clipped stride window, including its center even off the stride grid."""
    _require_int(frame_count, "frame_count", minimum=1)
    _require_int(center, "center", minimum=0)
    _require_int(radius_frames, "radius_frames", minimum=0)
    _require_int(stride, "stride", minimum=1)
    if center >= frame_count:
        raise ValueError("center must be within frame_count")

    lower = max(0, center - radius_frames)
    upper = min(frame_count - 1, center + radius_frames)
    return sorted({*range(lower, upper + 1, stride), center})


def extract_frames(
    video: EpisodeVideoRef,
    camera_key: str,
    indices: list[int],
) -> list[FrameSample]:
    """Decode requested episode-local frames from a whole or shared video file."""
    if not isinstance(video, EpisodeVideoRef):
        raise TypeError("video must be an EpisodeVideoRef")
    if not isinstance(camera_key, str) or not camera_key:
        raise ValueError("camera_key must be a nonempty string")
    requested = _validate_requested_indices(indices)
    requested_set = set(requested)
    fps = _require_positive_finite_number(video.fps, "video fps")
    episode_frame_count = round((video.to_timestamp - video.from_timestamp) * fps)
    if episode_frame_count < 1:
        raise ValueError("episode video slice must contain at least one frame")
    outside = [frame_index for frame_index in requested if frame_index >= episode_frame_count]
    if outside:
        raise ValueError(
            f"Requested frame(s) are outside episode video slice of length "
            f"{episode_frame_count}: {outside}"
        )
    if not requested:
        return []

    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to extract video frames") from exc

    try:
        container = av.open(str(video.path))
    except Exception as exc:  # PyAV exposes codec and I/O specific exceptions.
        raise ValueError(f"Unable to open video {video.path}: {exc}") from exc

    try:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"Video {video.path} has no video stream")
        try:
            measured_fps = float(stream.average_rate)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            raise ValueError(f"Video {video.path} has invalid fps") from None
        if not video_fps_matches(measured_fps, fps):
            raise ValueError(f"Video {video.path} fps does not match episode reference")
        try:
            time_base = float(stream.time_base)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            raise ValueError(f"Video {video.path} has invalid stream time base") from None
        if not math.isfinite(time_base) or time_base <= 0:
            raise ValueError(f"Video {video.path} has invalid stream time base")

        found: dict[int, FrameSample] = {}
        seen_local: set[int] = set()
        try:
            seek_time = max(0.0, video.from_timestamp - 1.0 / fps)
            seek_pts = math.floor(seek_time / time_base)
            container.seek(seek_pts, backward=True, any_frame=False, stream=stream)
            stop_time = video.to_timestamp + 1.0 / fps
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError(f"Video {video.path} contains a frame without PTS")
                media_time = float(frame.pts * stream.time_base)
                if not math.isfinite(media_time):
                    raise ValueError(f"Video {video.path} contains a frame with invalid PTS")
                if media_time > stop_time:
                    break
                if not video.from_timestamp <= media_time < video.to_timestamp:
                    continue
                local_index = round((media_time - video.from_timestamp) * fps)
                if not 0 <= local_index < episode_frame_count:
                    continue
                expected_time = video.from_timestamp + local_index / fps
                if not math.isclose(media_time, expected_time, rel_tol=0.0, abs_tol=0.5 / fps):
                    raise ValueError(f"Video {video.path} contains a frame with invalid episode-local PTS")
                if local_index in seen_local:
                    raise ValueError(
                        f"Video {video.path} contains duplicate episode-local frame {local_index}"
                    )
                seen_local.add(local_index)
                if local_index not in requested_set:
                    continue
                found[local_index] = _make_sample(frame, camera_key, local_index, fps)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Unable to decode video {video.path}: {exc}") from exc

        missing = [frame_index for frame_index in requested if frame_index not in found]
        if missing:
            raise ValueError(f"Video {video.path} is missing requested frame(s): {missing}")
        return [found[frame_index] for frame_index in requested]
    finally:
        container.close()


def as_data_url(sample: FrameSample) -> str:
    """Encode a sample JPEG for a multimodal API image input."""
    if not isinstance(sample, FrameSample):
        raise TypeError("sample must be a FrameSample")
    return "data:image/jpeg;base64," + base64.b64encode(sample.jpeg).decode("ascii")


def _require_int(value: object, name: str, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_positive_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _validate_requested_indices(indices: list[int]) -> list[int]:
    if not isinstance(indices, list):
        raise TypeError("indices must be a list of integers")
    for frame_index in indices:
        _require_int(frame_index, "indices", minimum=0)
    if len(indices) != len(set(indices)):
        raise ValueError("indices must be unique")
    return list(indices)


def _make_sample(frame: Any, camera_key: str, frame_index: int, fps: float) -> FrameSample:
    image = frame.to_image().convert("RGB")
    timestamp_seconds = frame_index / fps
    label = f"{camera_key}\nframe {frame_index}\ntime {timestamp_seconds:.3f}s"
    if image.width >= 64 and image.height >= 48:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        left, top, right, bottom = draw.multiline_textbbox((0, 0), label, font=font, spacing=2)
        padding = 3
        draw.rectangle((0, 0, right - left + 2 * padding, bottom - top + 2 * padding), fill="black")
        draw.multiline_text((padding, padding), label, fill="white", font=font, spacing=2)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return FrameSample(
        camera_key=camera_key,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        jpeg=output.getvalue(),
    )

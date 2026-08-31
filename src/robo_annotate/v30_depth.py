"""Pinned LeRobot v0.6.1 depth-map metadata and codec helpers."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np


DEPTH_QMAX = 4095
DEFAULT_DEPTH_MIN = 0.01
DEFAULT_DEPTH_MAX = 10.0
DEFAULT_DEPTH_SHIFT = 3.5
DEFAULT_DEPTH_USE_LOG = True
DEFAULT_DEPTH_PIX_FMT = "gray12le"
DEFAULT_DEPTH_UNIT = "mm"


@dataclass(frozen=True)
class DepthMetadata:
    """Parameters required to reproduce LeRobot's 12-bit depth mapping."""

    depth_min: float
    depth_max: float
    shift: float
    use_log: bool
    pix_fmt: str
    unit: str
    codec: str


def is_depth_feature(feature: object) -> bool:
    """Return whether a feature uses any official v0.6.1 depth marker."""
    if not isinstance(feature, Mapping):
        return False
    info = feature.get("info")
    video_info = feature.get("video_info")
    return bool(
        isinstance(info, Mapping)
        and (info.get("is_depth_map") or info.get("video.is_depth_map"))
        or isinstance(video_info, Mapping)
        and video_info.get("video.is_depth_map")
    )


def depth_metadata(feature: object, context: str) -> DepthMetadata:
    """Validate and return the official depth quantization declaration."""
    if not isinstance(feature, Mapping) or not is_depth_feature(feature):
        raise ValueError(f"depth feature declaration is invalid: {context}")
    info = feature.get("info")
    if not isinstance(info, Mapping):
        raise ValueError(f"depth feature info is invalid: {context}")
    depth_min = _finite_number(info.get("video.depth_min", DEFAULT_DEPTH_MIN), context)
    depth_max = _finite_number(info.get("video.depth_max", DEFAULT_DEPTH_MAX), context)
    shift = _finite_number(info.get("video.shift", DEFAULT_DEPTH_SHIFT), context)
    use_log = info.get("video.use_log", DEFAULT_DEPTH_USE_LOG)
    pix_fmt = info.get("video.pix_fmt", DEFAULT_DEPTH_PIX_FMT)
    unit = info.get("depth_unit", DEFAULT_DEPTH_UNIT)
    codec = info.get("video.codec", "hevc")
    if depth_max <= depth_min:
        raise ValueError(f"depth range is invalid: {context}")
    if type(use_log) is not bool:
        raise ValueError(f"depth logarithmic flag is invalid: {context}")
    if use_log and depth_min + shift <= 0:
        raise ValueError(f"depth logarithmic shift is invalid: {context}")
    if unit not in {"m", "mm"}:
        raise ValueError(f"depth unit must be 'm' or 'mm': {context}")
    if not isinstance(pix_fmt, str) or not pix_fmt:
        raise ValueError(f"depth pixel format is invalid: {context}")
    if not isinstance(codec, str) or not codec:
        raise ValueError(f"depth codec is invalid: {context}")
    try:
        components = len(av.VideoFormat(pix_fmt).components)
    except Exception as exc:
        raise ValueError(f"depth pixel format is unsupported: {context}") from exc
    if components != 1:
        raise ValueError(f"depth pixel format must have one channel: {context}")
    return DepthMetadata(
        depth_min=depth_min,
        depth_max=depth_max,
        shift=shift,
        use_log=use_log,
        pix_fmt=pix_fmt,
        unit=unit,
        codec=codec,
    )


def depth_codes(frame: av.VideoFrame) -> np.ndarray[Any, np.dtype[np.uint16]]:
    """Extract one decoded frame as its canonical 12-bit grayscale codes."""
    array = frame.to_ndarray(format=DEFAULT_DEPTH_PIX_FMT)
    if array.ndim != 2 or array.dtype != np.uint16:
        raise ValueError("decoded depth frame must be a uint16 grayscale image")
    if array.size and int(array.max()) > DEPTH_QMAX:
        raise ValueError("decoded depth frame exceeds the 12-bit code range")
    return array


def iter_depth_codes(
    path: Path,
    metadata: DepthMetadata | None = None,
) -> Iterator[np.ndarray[Any, np.dtype[np.uint16]]]:
    """Yield canonical depth codes from one video file."""
    with av.open(str(path), mode="r") as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if len(streams) != 1:
            raise ValueError("depth video must contain exactly one video stream")
        stream = streams[0]
        if metadata is not None and (
            stream.pix_fmt != metadata.pix_fmt
            or stream.codec.canonical_name != metadata.codec
        ):
            raise ValueError("depth video codec or pixel format disagrees with metadata")
        for frame in container.decode(stream):
            yield depth_codes(frame)


def depth_frame(
    codes: np.ndarray[Any, np.dtype[np.uint16]],
    pix_fmt: str,
) -> av.VideoFrame:
    """Build a padded PyAV depth frame without corrupting plane row strides."""
    if codes.ndim != 2 or codes.dtype != np.uint16:
        raise ValueError("depth codes must be a two-dimensional uint16 array")
    if codes.size and int(codes.max()) > DEPTH_QMAX:
        raise ValueError("depth codes exceed the 12-bit code range")
    frame = av.VideoFrame.from_ndarray(codes, format=pix_fmt)
    plane = frame.planes[0]
    stride = plane.line_size // np.dtype(np.uint16).itemsize
    destination = np.frombuffer(plane, dtype=np.uint16).reshape(codes.shape[0], stride)
    destination[:, : codes.shape[1]] = codes
    return frame


def dequantize_depth(
    codes: np.ndarray[Any, np.dtype[np.uint16]],
    metadata: DepthMetadata,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Invert LeRobot's depth quantizer in the feature's recorded unit."""
    normalized = codes.astype(np.float32) / np.float32(DEPTH_QMAX)
    if metadata.use_log:
        low = math.log(metadata.depth_min + metadata.shift)
        high = math.log(metadata.depth_max + metadata.shift)
        values = np.exp(normalized * (high - low) + low) - metadata.shift
    else:
        values = normalized * (metadata.depth_max - metadata.depth_min) + metadata.depth_min
    values = np.clip(values, metadata.depth_min, metadata.depth_max)
    if metadata.unit == "mm":
        values = np.rint(values * 1000.0)
    return np.asarray(values, dtype=np.float32)


def depth_quantization_tolerance(metadata: DepthMetadata) -> float:
    """Return a conservative absolute half-bin error in the recorded unit."""
    if metadata.use_log:
        step = (metadata.depth_max + metadata.shift) * (
            math.exp(
                (
                    math.log(metadata.depth_max + metadata.shift)
                    - math.log(metadata.depth_min + metadata.shift)
                )
                / DEPTH_QMAX
            )
            - 1.0
        )
    else:
        step = (metadata.depth_max - metadata.depth_min) / DEPTH_QMAX
    scale = 1000.0 if metadata.unit == "mm" else 1.0
    return step * scale / 2.0 + (0.500001 if metadata.unit == "mm" else 1e-7)


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"depth parameter is invalid: {context}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"depth parameter is invalid: {context}")
    return result

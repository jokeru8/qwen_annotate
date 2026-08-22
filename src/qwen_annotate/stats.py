"""Deterministic statistics for rewritten LeRobot parquet payloads."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


_QUANTILES = (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99))


@dataclass
class _Accumulator:
    width: int
    integral: bool
    expected_count: int
    spool_path: Path
    count: int = 0
    mean: np.ndarray = field(init=False)
    m2: np.ndarray = field(init=False)
    minimum: np.ndarray = field(init=False)
    maximum: np.ndarray = field(init=False)
    samples: np.memmap = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.width, dtype=np.float64)
        self.m2 = np.zeros(self.width, dtype=np.float64)
        self.minimum = np.full(self.width, np.inf, dtype=np.float64)
        self.maximum = np.full(self.width, -np.inf, dtype=np.float64)
        self.samples = np.memmap(
            self.spool_path, dtype=np.float64, mode="w+", shape=(self.expected_count, self.width),
        )

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.width:
            raise ValueError("numeric feature shape changed between parquet batches")
        if not np.isfinite(values).all():
            raise ValueError("statistics require finite numeric values")
        if not len(values):
            return
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = np.square(values - batch_mean).sum(axis=0)
        delta = batch_mean - self.mean
        combined = self.count + batch_count
        self.mean += delta * (batch_count / combined)
        self.m2 += batch_m2 + np.square(delta) * self.count * batch_count / combined
        self.samples[self.count:combined] = values
        self.count = combined
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def finish(self) -> dict[str, list[int | float]]:
        if self.count == 0:
            raise ValueError("cannot compute statistics for an empty numeric feature")
        if self.count != self.expected_count:
            raise ValueError("numeric feature row count differs between parquet files")
        convert_extreme = (
            (lambda array: [int(value) for value in array]) if self.integral
            else (lambda array: [float(value) for value in array])
        )
        result: dict[str, list[int | float]] = {
            "min": convert_extreme(self.minimum),
            "max": convert_extreme(self.maximum),
            "mean": [float(value) for value in self.mean],
            "std": [float(value) for value in np.sqrt(self.m2 / self.count)],
            "count": [self.count],
        }
        quantiles = np.quantile(
            self.samples,
            [quantile for _, quantile in _QUANTILES],
            axis=0,
            overwrite_input=True,
        )
        for (name, _), values in zip(_QUANTILES, quantiles, strict=True):
            result[name] = [float(value) for value in values]
        if not all(np.isfinite(value) for items in result.values() for value in items):
            raise ValueError("statistics must be finite")
        return result


def recompute_stats(
    parquet_paths: Iterable[Path],
    feature_names: Sequence[str] | None = None,
) -> dict[str, dict[str, list[int | float]]]:
    """Stream numeric scalar/fixed-size-list columns and return LeRobot-shaped stats."""
    paths = [Path(path) for path in parquet_paths]
    selected = set(feature_names) if feature_names is not None else None
    expected_schema: dict[str, tuple[int, bool]] | None = None
    total_rows = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        try:
            schema = parquet.schema_arrow
            numeric_schema = {
                field.name: shape
                for field in schema
                if (selected is None or field.name in selected)
                and (shape := _numeric_shape(field.type)) is not None
            }
            if expected_schema is None:
                expected_schema = numeric_schema
            elif numeric_schema != expected_schema:
                raise ValueError("numeric feature schema changed between parquet files")
            total_rows += parquet.metadata.num_rows
        finally:
            parquet.close()

    if not expected_schema:
        return {}
    with TemporaryDirectory(prefix="qwen-annotate-stats-") as temporary:
        accumulators = {
            name: _Accumulator(width, integral, total_rows, Path(temporary) / f"feature-{index}.bin")
            for index, (name, (width, integral)) in enumerate(expected_schema.items())
        }
        for path in paths:
            parquet = pq.ParquetFile(path)
            try:
                columns = list(expected_schema)
                for batch in parquet.iter_batches(columns=columns, batch_size=65_536):
                    for name, array in zip(batch.schema.names, batch.columns, strict=True):
                        width, integral = _numeric_shape(array.type)  # type: ignore[misc]
                        if array.null_count:
                            raise ValueError(f"statistics do not support null values in {name}")
                        accumulator = accumulators[name]
                        if (accumulator.width, accumulator.integral) != (width, integral):
                            raise ValueError(f"numeric feature schema changed for {name}")
                        accumulator.update(_as_matrix(array, width))
            finally:
                parquet.close()
        return {name: accumulator.finish() for name, accumulator in accumulators.items()}


def iter_video_rgb_frames(path: Path) -> Iterator[np.ndarray]:
    """Decode one video sequentially as RGB uint8 arrays."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to compute video statistics") from exc
    try:
        container = av.open(str(path))
    except Exception as exc:
        raise ValueError(f"unable to open video for statistics: {path}") from exc
    try:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"video has no video stream: {path}")
        try:
            for frame in container.decode(stream):
                yield frame.to_ndarray(format="rgb24")
        except Exception as exc:
            raise ValueError(f"unable to decode video for statistics: {path}") from exc
    finally:
        container.close()


def recompute_video_stats(
    video_paths: Sequence[Path],
    expected_frames: Sequence[int],
    declared_shape: Sequence[int],
    *,
    frame_iterator: Callable[[Path], Iterable[np.ndarray]] = iter_video_rgb_frames,
) -> dict[str, list]:
    """Compute exact RGB channel statistics with bounded 256-bin histograms."""
    if len(video_paths) != len(expected_frames) or not video_paths:
        raise ValueError("video paths and expected frame counts must be nonempty and aligned")
    if len(declared_shape) != 3:
        raise ValueError("video feature shape must be [height, width, channels]")
    height, width, channels = declared_shape
    if any(type(value) is not int or value <= 0 for value in (height, width, channels)) or channels != 3:
        raise ValueError("video feature shape must contain positive RGB dimensions")
    histogram = np.zeros((3, 256), dtype=np.int64)
    total_frames = 0
    for raw_path, expected in zip(video_paths, expected_frames, strict=True):
        path = Path(raw_path)
        if type(expected) is not int or expected <= 0:
            raise ValueError("expected video frame count must be positive")
        decoded = 0
        for frame in frame_iterator(path):
            array = np.asarray(frame)
            if array.dtype != np.uint8 or array.shape != (height, width, 3):
                raise ValueError(f"decoded RGB frame shape or dtype differs from metadata: {path}")
            for channel in range(3):
                histogram[channel] += np.bincount(array[..., channel].reshape(-1), minlength=256)
            decoded += 1
        if decoded != expected:
            raise ValueError(f"video frame count differs from metadata: {path}")
        total_frames += decoded
    count = total_frames * height * width
    if count <= 0 or np.any(histogram.sum(axis=1) != count):
        raise ValueError("decoded camera coverage is incomplete")

    bins = np.arange(256, dtype=np.float64) / 255.0
    mean = (histogram * bins).sum(axis=1) / count
    variance = (histogram * np.square(bins)).sum(axis=1) / count - np.square(mean)
    minimum = np.argmax(histogram > 0, axis=1).astype(np.float64) / 255.0
    maximum = (255 - np.argmax((histogram > 0)[:, ::-1], axis=1)).astype(np.float64) / 255.0
    metrics: dict[str, np.ndarray] = {
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": np.sqrt(np.maximum(variance, 0.0)),
    }
    cumulative = np.cumsum(histogram, axis=1)
    for name, quantile in _QUANTILES:
        rank = (count - 1) * quantile
        lower = int(np.floor(rank))
        upper = int(np.ceil(rank))
        fraction = rank - lower
        lower_values = np.array([
            np.searchsorted(cumulative[channel], lower, side="right") for channel in range(3)
        ], dtype=np.float64)
        upper_values = np.array([
            np.searchsorted(cumulative[channel], upper, side="right") for channel in range(3)
        ], dtype=np.float64)
        metrics[name] = (lower_values + fraction * (upper_values - lower_values)) / 255.0
    result: dict[str, list] = {
        name: [[[float(value)]] for value in values]
        for name, values in metrics.items()
    }
    result["count"] = [count]
    if not all(np.isfinite(value) for name, values in result.items() if name != "count" for channel in values for row in channel for value in row):
        raise ValueError("video statistics must be finite")
    return result


def _numeric_shape(dtype: pa.DataType) -> tuple[int, bool] | None:
    if pa.types.is_fixed_size_list(dtype):
        value = dtype.value_type
        if pa.types.is_integer(value) or pa.types.is_floating(value):
            return dtype.list_size, pa.types.is_integer(value)
        return None
    if pa.types.is_integer(dtype) or pa.types.is_floating(dtype):
        return 1, pa.types.is_integer(dtype)
    return None


def _as_matrix(array: pa.Array, width: int) -> np.ndarray:
    if pa.types.is_fixed_size_list(array.type):
        return np.asarray(array.values.to_numpy(zero_copy_only=False)).reshape(len(array), width)
    return np.asarray(array.to_numpy(zero_copy_only=False)).reshape(len(array), 1)

"""Deterministic statistics for rewritten LeRobot parquet payloads."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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

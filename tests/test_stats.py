import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qwen_annotate.stats import recompute_stats


def test_recompute_stats_preserves_scalar_and_fixed_list_shapes(tmp_path: Path) -> None:
    """Catches flattening vectors or emitting scalar JSON values instead of lists."""
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    vector_type = pa.list_(pa.float32(), 2)
    pq.write_table(pa.table({
        "scalar": pa.array([1, 2], type=pa.int64()),
        "vector": pa.array([[1, 10], [3, 30]], type=vector_type),
    }), first)
    pq.write_table(pa.table({
        "scalar": pa.array([3, 4], type=pa.int64()),
        "vector": pa.array([[5, 50], [7, 70]], type=vector_type),
    }), second)

    stats = recompute_stats([first, second])

    assert stats["scalar"] == {
        "min": [1], "max": [4], "mean": [2.5],
        "std": [math.sqrt(1.25)], "count": [4],
        "q01": [1.03], "q10": [1.3], "q50": [2.5],
        "q90": [3.7], "q99": [3.9699999999999998],
    }
    assert stats["vector"]["count"] == [4]
    expected_vector = {
        "min": [1.0, 10.0], "max": [7.0, 70.0], "mean": [4.0, 40.0],
        "std": [math.sqrt(5.0), math.sqrt(500.0)],
        "q01": [1.06, 10.6], "q10": [1.6, 16.0], "q50": [4.0, 40.0],
        "q90": [6.4, 64.0], "q99": [6.94, 69.4],
    }
    for key, expected in expected_vector.items():
        assert stats["vector"][key] == pytest.approx(expected)
    assert all(math.isfinite(number) for feature in stats.values() for values in feature.values() for number in values)


def test_recompute_stats_rejects_nonfinite_and_nonnumeric_values(tmp_path: Path) -> None:
    """Catches publication of NaN and silently inventing statistics for strings."""
    bad = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"value": [1.0, float("nan")]}), bad)
    with pytest.raises(ValueError, match="finite"):
        recompute_stats([bad])

    text = tmp_path / "text.parquet"
    pq.write_table(pa.table({"label": ["a", "b"]}), text)
    assert recompute_stats([text]) == {}


def test_recompute_stats_rejects_numeric_schema_drift(tmp_path: Path) -> None:
    """Catches silently aggregating a feature from only some selected episodes."""
    first, second = tmp_path / "first.parquet", tmp_path / "second.parquet"
    pq.write_table(pa.table({"value": pa.array([1], type=pa.int64())}), first)
    pq.write_table(pa.table({"other": pa.array([2], type=pa.int64())}), second)
    with pytest.raises(ValueError, match="schema"):
        recompute_stats([first, second])

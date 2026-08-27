import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import numpy as np

from robo_annotate.stats import iter_video_rgb_frames, recompute_stats, recompute_video_stats


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


def test_recompute_video_stats_has_reference_shape_and_exact_streaming_quantiles(tmp_path: Path) -> None:
    """Catches omitted camera stats, channel flattening, or dataset-sized frame retention."""
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    first.touch()
    second.touch()
    frames = {
        first: [
            np.array([[[0, 10, 20], [30, 40, 50]]], dtype=np.uint8),
            np.array([[[60, 70, 80], [90, 100, 110]]], dtype=np.uint8),
        ],
        second: [np.array([[[120, 130, 140], [150, 160, 170]]], dtype=np.uint8)],
    }

    stats = recompute_video_stats(
        [first, second], [2, 1], [1, 2, 3], frame_iterator=lambda path: iter(frames[path]),
    )

    assert stats["count"] == [6]
    for metric in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
        assert len(stats[metric]) == 3
        assert all(len(channel) == 1 and len(channel[0]) == 1 for channel in stats[metric])
    assert [channel[0][0] for channel in stats["min"]] == [0.0, 10 / 255, 20 / 255]
    assert [channel[0][0] for channel in stats["max"]] == [150 / 255, 160 / 255, 170 / 255]
    assert [channel[0][0] for channel in stats["mean"]] == pytest.approx([75 / 255, 85 / 255, 95 / 255])
    assert [channel[0][0] for channel in stats["q50"]] == pytest.approx([75 / 255, 85 / 255, 95 / 255])


def test_recompute_video_stats_checks_shape_and_frame_count(tmp_path: Path) -> None:
    """Catches camera coverage that disagrees with LeRobot metadata."""
    path = tmp_path / "video.mp4"
    path.touch()
    with pytest.raises(ValueError, match="frame count"):
        recompute_video_stats(
            [path], [2], [1, 1, 3],
            frame_iterator=lambda _: iter([np.zeros((1, 1, 3), dtype=np.uint8)]),
        )
    with pytest.raises(ValueError, match="shape"):
        recompute_video_stats(
            [path], [1], [2, 1, 3],
            frame_iterator=lambda _: iter([np.zeros((1, 1, 3), dtype=np.uint8)]),
        )


def test_real_video_iterator_closes_container_when_consumer_stops_early(monkeypatch, tmp_path: Path) -> None:
    """Catches leaking a decoder when deep validation exits before consuming every frame."""
    av = pytest.importorskip("av")
    path = tmp_path / "tiny.mp4"
    try:
        container = av.open(str(path), mode="w")
        stream = container.add_stream("mpeg4", rate=2)
        stream.width = 4
        stream.height = 4
        stream.pix_fmt = "yuv420p"
        for value in (0, 80):
            frame = av.VideoFrame.from_ndarray(np.full((4, 4, 3), value, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    except Exception as exc:
        pytest.skip(f"mpeg4 encoder unavailable: {exc}")

    real_open = av.open
    wrappers = []
    class TrackedContainer:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.closed = False
        def __getattr__(self, name):
            return getattr(self.wrapped, name)
        def close(self):
            self.closed = True
            self.wrapped.close()
    def tracked_open(*args, **kwargs):
        wrapper = TrackedContainer(real_open(*args, **kwargs))
        wrappers.append(wrapper)
        return wrapper
    monkeypatch.setattr(av, "open", tracked_open)

    frames = iter_video_rgb_frames(path)
    first = next(frames)
    assert first.shape == (4, 4, 3) and first.dtype == np.uint8
    frames.close()
    assert wrappers and wrappers[0].closed


def test_recompute_video_stats_closes_iterator_on_bad_frame(tmp_path: Path) -> None:
    """Catches relying on garbage collection to close a decoder after validation aborts."""
    path = tmp_path / "bad.mp4"
    path.touch()
    class BadFrames:
        def __init__(self):
            self.closed = False
            self.used = False
        def __iter__(self):
            return self
        def __next__(self):
            if self.used:
                raise StopIteration
            self.used = True
            return np.zeros((2, 2, 3), dtype=np.float32)
        def close(self):
            self.closed = True
    frames = BadFrames()
    with pytest.raises(ValueError, match="shape or dtype"):
        recompute_video_stats(
            [path], [1], [2, 2, 3], frame_iterator=lambda _: frames,
        )
    assert frames.closed

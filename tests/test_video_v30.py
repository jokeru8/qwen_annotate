import math
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from robo_annotate.lerobot import EpisodeVideoRef, inspect_dataset
from robo_annotate.video import extract_frames
from tests.v30_fixtures import (
    dominant_test_color,
    expected_color,
    make_lerobot_v30_fixture,
    make_v30_config,
)


def test_extracts_local_frames_from_middle_shared_video_slice(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    ref = dataset.episodes[1].videos["observation.images.main"]

    samples = extract_frames(ref, "observation.images.main", [0, 3, 7])

    assert [sample.frame_index for sample in samples] == [0, 3, 7]
    assert [sample.timestamp_seconds for sample in samples] == [0.0, 0.6, 1.4]
    assert [dominant_test_color(sample.jpeg) for sample in samples] == [
        expected_color(0, 1, 0),
        expected_color(0, 1, 3),
        expected_color(0, 1, 7),
    ]


def test_preserves_requested_episode_local_frame_order(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    ref = dataset.episodes[1].videos["observation.images.main"]

    samples = extract_frames(ref, "observation.images.main", [7, 0, 3])

    assert [sample.frame_index for sample in samples] == [7, 0, 3]
    assert [sample.timestamp_seconds for sample in samples] == [1.4, 0.0, 0.6]
    assert [dominant_test_color(sample.jpeg) for sample in samples] == [
        expected_color(0, 1, 7),
        expected_color(0, 1, 0),
        expected_color(0, 1, 3),
    ]


def test_rejects_episode_local_index_at_slice_length(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    ref = dataset.episodes[1].videos["observation.images.main"]

    with pytest.raises(ValueError, match=r"outside episode video slice.*8"):
        extract_frames(ref, "observation.images.main", [8])


def test_rejects_slice_when_shared_video_has_insufficient_decoded_frames(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    ref = dataset.episodes[2].videos["observation.images.main"].model_copy(
        update={"to_timestamp": 4.0}
    )

    with pytest.raises(ValueError, match=r"missing requested frame.*5"):
        extract_frames(ref, "observation.images.main", [5])


class _PtsFrame:
    def __init__(self, pts: int | float | None) -> None:
        self.pts = pts

    def to_image(self) -> Image.Image:
        return Image.new("RGB", (64, 64), (32, 96, 16))


class _PtsContainer:
    def __init__(
        self,
        pts: list[int | float | None],
        *,
        time_base: Fraction = Fraction(1, 2),
    ) -> None:
        self.streams = [
            SimpleNamespace(
                type="video",
                average_rate=Fraction(2, 1),
                time_base=time_base,
            )
        ]
        self._pts = pts
        self.closed = False

    def seek(self, offset: int, **kwargs: object) -> None:
        del offset, kwargs

    def decode(self, stream: object):
        del stream
        yield from (_PtsFrame(value) for value in self._pts)

    def close(self) -> None:
        self.closed = True


def test_rejects_decoded_pts_before_and_after_the_episode_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = _PtsContainer([0, 5])
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: container))
    ref = EpisodeVideoRef(
        path=tmp_path / "shared.mp4",
        from_timestamp=1.0,
        to_timestamp=2.0,
        fps=2.0,
    )

    with pytest.raises(ValueError, match=r"missing requested frame.*0"):
        extract_frames(ref, "observation.images.main", [0])

    assert container.closed


def test_does_not_round_neighboring_episode_pts_into_local_frame_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = _PtsContainer([3], time_base=Fraction(1, 4))
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: container))
    ref = EpisodeVideoRef(
        path=tmp_path / "shared.mp4",
        from_timestamp=1.0,
        to_timestamp=2.0,
        fps=2.0,
    )

    with pytest.raises(ValueError, match=r"missing requested frame.*0"):
        extract_frames(ref, "observation.images.main", [0])

    assert container.closed


def test_rejects_duplicate_episode_local_pts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = _PtsContainer([2, 2])
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: container))
    ref = EpisodeVideoRef(
        path=tmp_path / "shared.mp4",
        from_timestamp=1.0,
        to_timestamp=2.0,
        fps=2.0,
    )

    with pytest.raises(ValueError, match=r"duplicate episode-local frame 0"):
        extract_frames(ref, "observation.images.main", [0])

    assert container.closed


@pytest.mark.parametrize(
    ("malformed_pts", "message"),
    [(None, "frame without PTS"), (math.inf, "frame with invalid PTS")],
)
def test_rejects_malformed_pts_after_last_requested_frame(
    malformed_pts: float | None,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = _PtsContainer([2, malformed_pts])
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: container))
    ref = EpisodeVideoRef(
        path=tmp_path / "shared.mp4",
        from_timestamp=1.0,
        to_timestamp=2.0,
        fps=2.0,
    )

    with pytest.raises(ValueError, match=message):
        extract_frames(ref, "observation.images.main", [0])

    assert container.closed


def test_rejects_reference_fps_that_differs_from_video_stream(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    ref = dataset.episodes[1].videos["observation.images.main"].model_copy(
        update={"fps": 4.0}
    )

    with pytest.raises(ValueError, match=r"fps.*does not match"):
        extract_frames(ref, "observation.images.main", [0])

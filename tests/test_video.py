import base64
import io
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

from qwen_annotate.video import FrameSample, as_data_url, extract_frames, uniform_indices, window_indices


def test_uniform_indices_are_even_capped_and_keep_both_endpoints() -> None:
    indices = uniform_indices(frame_count=1000, source_fps=25.0, target_fps=1.0, max_frames=10)

    assert indices[0] == 0
    assert indices[-1] == 999
    assert len(indices) == 10
    assert indices == sorted(set(indices))


def test_uniform_indices_handles_a_single_frame_and_low_target_rate() -> None:
    assert uniform_indices(1, 30.0, 1.0, 8) == [0]
    assert uniform_indices(4, 4.0, 0.1, 8) == [0, 3]


def test_uniform_indices_preserve_both_endpoints_when_a_one_frame_cap_conflicts() -> None:
    assert uniform_indices(5, 30.0, 1.0, 1) == [0, 4]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((True, 25.0, 1.0, 10), "frame_count"),
        ((10, True, 1.0, 10), "source_fps"),
        ((10, 25.0, math.inf, 10), "target_fps"),
        ((10, 25.0, 1.0, 0), "max_frames"),
    ],
)
def test_uniform_indices_rejects_invalid_caller_values(args: tuple[object, ...], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        uniform_indices(*args)  # type: ignore[arg-type]


def test_window_indices_clip_to_available_frames_and_include_center() -> None:
    indices = window_indices(center=3, radius_frames=10, stride=2, frame_count=20)

    assert min(indices) == 0
    assert max(indices) <= 13
    assert 3 in indices
    assert indices == sorted(set(indices))


def test_window_indices_include_center_when_stride_grid_misses_it() -> None:
    assert window_indices(center=4, radius_frames=3, stride=2, frame_count=12) == [1, 3, 4, 5, 7]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((True, 1, 1, 5), "center"),
        ((1, True, 1, 5), "radius_frames"),
        ((1, 1, True, 5), "stride"),
        ((1, 1, 1, 0), "frame_count"),
        ((5, 1, 1, 5), "center"),
    ],
)
def test_window_indices_rejects_invalid_caller_values(args: tuple[object, ...], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        window_indices(*args)  # type: ignore[arg-type]


def test_frame_sample_forbids_extras_and_invalid_evidence_values() -> None:
    with pytest.raises(ValidationError):
        FrameSample(camera_key="right_eye", frame_index=-1, timestamp_seconds=0.0, jpeg=b"jpeg")
    with pytest.raises(ValidationError):
        FrameSample(camera_key="right_eye", frame_index=0, timestamp_seconds=math.nan, jpeg=b"jpeg")
    with pytest.raises(ValidationError):
        FrameSample(
            camera_key="right_eye",
            frame_index=0,
            timestamp_seconds=0.0,
            jpeg=b"jpeg",
            unexpected=True,
        )


def test_as_data_url_round_trips_the_exact_jpeg_bytes() -> None:
    sample = FrameSample(camera_key="right_eye", frame_index=7, timestamp_seconds=0.7, jpeg=b"jpeg-bytes")

    result = as_data_url(sample)

    assert result.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(result.removeprefix("data:image/jpeg;base64,")) == sample.jpeg


def _write_video(path: Path) -> list[tuple[int, int, int]]:
    av = pytest.importorskip("av")
    colors = [(index * 19 % 250, index * 37 % 250, index * 53 % 250) for index in range(12)]
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("mpeg4", rate=6)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for color in colors:
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64), color))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return colors


def test_extract_frames_decodes_exact_requested_frames_and_labels_evidence(tmp_path: Path) -> None:
    colors = _write_video(tmp_path / "frames.mp4")

    samples = extract_frames(tmp_path / "frames.mp4", "right_eye", [0, 6, 11], fps=6.0)

    assert [sample.frame_index for sample in samples] == [0, 6, 11]
    assert [sample.timestamp_seconds for sample in samples] == [0.0, 1.0, pytest.approx(11 / 6)]
    for sample, expected_color in zip(samples, [colors[0], colors[6], colors[11]], strict=True):
        image = Image.open(io.BytesIO(sample.jpeg)).convert("RGB")
        assert image.size == (64, 64)
        actual_color = image.getpixel((60, 60))
        assert max(abs(actual - expected) for actual, expected in zip(actual_color, expected_color, strict=True)) < 40


def test_extract_frames_rejects_duplicate_and_unavailable_requested_indices(tmp_path: Path) -> None:
    _write_video(tmp_path / "frames.mp4")

    with pytest.raises(ValueError, match="unique"):
        extract_frames(tmp_path / "frames.mp4", "right_eye", [1, 1], fps=6.0)
    with pytest.raises(ValueError, match="missing requested frame"):
        extract_frames(tmp_path / "frames.mp4", "right_eye", [12], fps=6.0)


@pytest.mark.parametrize(
    ("indices", "fps", "message"),
    [([True], 6.0, "indices"), ([0], True, "fps"), ([0], math.nan, "fps")],
)
def test_extract_frames_rejects_invalid_requested_indices_and_fps(
    tmp_path: Path, indices: list[object], fps: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        extract_frames(tmp_path / "unused.mp4", "right_eye", indices, fps=fps)  # type: ignore[arg-type]


def test_extract_frames_closes_the_container_after_a_decode_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FailingContainer:
        streams = [SimpleNamespace(type="video")]
        closed = False

        def decode(self, stream: object):
            del stream
            raise RuntimeError("corrupt stream")
            yield  # pragma: no cover

        def close(self) -> None:
            self.closed = True

    container = FailingContainer()
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: container))

    with pytest.raises(ValueError, match="Unable to decode video"):
        extract_frames(tmp_path / "corrupt.mp4", "right_eye", [0], fps=6.0)

    assert container.closed


def test_extract_frames_reports_a_missing_video_stream_and_closes_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class AudioOnlyContainer:
        streams = [SimpleNamespace(type="audio")]
        closed = False

        def close(self) -> None:
            self.closed = True

    container = AudioOnlyContainer()
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: container))

    with pytest.raises(ValueError, match="no video stream"):
        extract_frames(tmp_path / "audio.mp4", "right_eye", [0], fps=6.0)

    assert container.closed


_REFERENCE_DATASET = os.environ.get("QWEN_ANNOTATE_REFERENCE_DATASET")


@pytest.mark.skipif(not _REFERENCE_DATASET, reason="set QWEN_ANNOTATE_REFERENCE_DATASET to run dataset video smoke test")
def test_reference_right_eye_episode_zero_smoke() -> None:
    root = Path(_REFERENCE_DATASET)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    camera_key = next(key for key in info["features"] if "right_eye" in key)
    video = root / info["video_path"].format(
        episode_chunk=0, episode_index=0, video_key=camera_key
    )
    fps = float(info["fps"])

    samples = extract_frames(video, camera_key, [0, 1, 2], fps=fps)

    assert [sample.frame_index for sample in samples] == [0, 1, 2]
    assert [sample.timestamp_seconds for sample in samples] == [0.0, 1 / fps, 2 / fps]

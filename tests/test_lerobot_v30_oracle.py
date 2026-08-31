"""Optional compatibility checks against the official LeRobot v3.0 loader."""

from importlib import metadata
from pathlib import Path

import pytest


def _require_lerobot_version(installed: str) -> None:
    assert installed == "0.6.1", (
        "official LeRobot oracle requires version 0.6.1; "
        f"found {installed}"
    )


pytest.importorskip("lerobot", reason="requires the v3-validation extra")
_require_lerobot_version(metadata.version("lerobot"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

from robo_annotate.converter import convert_dataset
from tests.test_converter_v30 import selectively_accepted_v30_workspace
from tests.test_release_validator_v30 import accepted_v30_workspace
from tests.v30_fixtures import (
    expected_color,
    expected_depth_mm,
    make_lerobot_v30_fixture,
)


_EXPECTED_MAIN_CAMERA_FRAMES = {
    "source": {
        0: (0, 0, 0),
        5: (0, 0, 5),
        6: (1, 1, 0),
        9: (1, 1, 3),
        13: (1, 1, 7),
        14: (2, 2, 0),
        18: (2, 2, 4),
    },
    "full": {
        0: (0, 0, 0),
        5: (0, 0, 5),
        6: (1, 1, 0),
        9: (1, 1, 3),
        13: (1, 1, 7),
        14: (2, 2, 0),
        18: (2, 2, 4),
    },
    "accepted_only": {
        0: (0, 0, 0),
        5: (0, 0, 5),
        6: (1, 2, 0),
        10: (1, 2, 4),
    },
}
_DEPTH_CAMERA = "observation.depth.main"


def _decoded_dominant_rgb(image: object) -> tuple[float, float, float]:
    array = np.asarray(image)
    if array.ndim != 3:
        raise AssertionError(
            f"decoded main-camera image must have three dimensions; got {array.shape}"
        )
    if array.shape[-1] == 3:
        rgb = array
    elif array.shape[0] == 3:
        rgb = np.moveaxis(array, 0, -1)
    else:
        raise AssertionError(
            f"decoded main-camera image must have three RGB channels; got {array.shape}"
        )
    dominant = np.median(rgb.reshape(-1, 3), axis=0).astype(np.float64)
    if not np.all(np.isfinite(dominant)):
        raise AssertionError("decoded main-camera RGB must be finite")
    if (
        np.issubdtype(array.dtype, np.floating)
        and np.min(rgb) >= 0
        and np.max(rgb) <= 1.0
    ):
        dominant *= 255.0
    elif np.min(rgb) < 0 or np.max(rgb) > 255:
        raise AssertionError("decoded main-camera RGB must use the 0-1 or 0-255 range")
    return tuple(float(channel) for channel in dominant)


def _assert_decoded_color(
    image: object,
    expected: tuple[int, int, int],
) -> None:
    actual = _decoded_dominant_rgb(image)
    tolerance = 6.0
    assert all(
        abs(actual_channel - expected_channel) <= tolerance
        for actual_channel, expected_channel in zip(actual, expected, strict=True)
    ), (
        f"decoded main-camera RGB {actual} does not match expected {expected} "
        f"within codec tolerance {tolerance}"
    )


def build_or_convert_artifact(tmp_path: Path, kind: str) -> tuple[Path, int]:
    """Build one local v3.0 artifact without consulting the Hub."""
    if kind == "source":
        return make_lerobot_v30_fixture(
            tmp_path,
            depth_cameras=(_DEPTH_CAMERA,),
        ), 19
    if kind == "full":
        work, _, services = accepted_v30_workspace(
            tmp_path,
            depth_cameras=(_DEPTH_CAMERA,),
        )
        report = convert_dataset(work, tmp_path / "full", services=services)
        return report.output, 19
    if kind == "accepted_only":
        work, _, services = selectively_accepted_v30_workspace(
            tmp_path,
            accepted=(0, 2),
            depth_cameras=(_DEPTH_CAMERA,),
        )
        report = convert_dataset(
            work,
            tmp_path / "accepted-only",
            accepted_only=True,
            services=services,
        )
        return report.output, 11
    raise ValueError(f"unsupported oracle artifact kind: {kind}")


@pytest.mark.parametrize("kind", ["source", "full", "accepted_only"])
def test_official_lerobot_loads_v30_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    root, expected_length = build_or_convert_artifact(tmp_path, kind)

    dataset = LeRobotDataset(repo_id=f"local/{kind}", root=root)
    samples = [dataset[index] for index in range(expected_length)]

    assert len(dataset) == expected_length
    assert [int(sample["index"]) for sample in samples] == list(
        range(expected_length)
    )
    assert tuple(dataset.meta.stats["observation.matrix"]["mean"].shape) == (2, 2)
    for index, (
        output_episode_index,
        source_episode_index,
        local_frame_index,
    ) in _EXPECTED_MAIN_CAMERA_FRAMES[kind].items():
        assert int(samples[index]["episode_index"]) == output_episode_index
        image = samples[index]["observation.images.main"]
        assert tuple(image.shape) == (3, 24, 32)
        _assert_decoded_color(
            image,
            expected_color(0, source_episode_index, local_frame_index),
        )
        depth = np.asarray(samples[index][_DEPTH_CAMERA])
        assert depth.shape == (1, 24, 32)
        assert float(np.median(depth)) == pytest.approx(
            expected_depth_mm(0, source_episode_index, local_frame_index),
            abs=2.0,
        )


def test_oracle_rejects_unpinned_lerobot_version() -> None:
    with pytest.raises(
        AssertionError,
        match=r"official LeRobot oracle requires version 0\.6\.1",
    ):
        _require_lerobot_version("0.6.0")


@pytest.mark.parametrize(
    "image",
    [
        np.broadcast_to(
            np.asarray(expected_color(0, 1, 2), dtype=np.float32)[:, None, None]
            / 255.0,
            (3, 24, 32),
        ),
        np.broadcast_to(
            np.asarray(expected_color(0, 1, 2), dtype=np.uint8),
            (24, 32, 3),
        ),
    ],
)
def test_oracle_normalizes_official_image_layout_and_range(image: np.ndarray) -> None:
    assert _decoded_dominant_rgb(image) == pytest.approx(
        expected_color(0, 1, 2),
        abs=1.0,
    )


def test_oracle_rejects_substituted_main_camera_frame() -> None:
    substituted = np.broadcast_to(
        np.asarray(expected_color(0, 0, 1), dtype=np.float32)[:, None, None]
        / 255.0,
        (3, 24, 32),
    )

    with pytest.raises(AssertionError, match=r"decoded main-camera RGB"):
        _assert_decoded_color(substituted, expected_color(0, 0, 0))

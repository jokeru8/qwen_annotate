"""Optional compatibility checks against the official LeRobot v3.0 loader."""

from pathlib import Path

import pytest


pytest.importorskip("lerobot", reason="requires the v3-validation extra")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robo_annotate.converter import convert_dataset
from tests.test_converter_v30 import selectively_accepted_v30_workspace
from tests.test_release_validator_v30 import accepted_v30_workspace
from tests.v30_fixtures import make_lerobot_v30_fixture


def build_or_convert_artifact(tmp_path: Path, kind: str) -> tuple[Path, int]:
    """Build one local v3.0 artifact without consulting the Hub."""
    if kind == "source":
        return make_lerobot_v30_fixture(tmp_path), 19
    if kind == "full":
        work, _, services = accepted_v30_workspace(tmp_path)
        report = convert_dataset(work, tmp_path / "full", services=services)
        return report.output, 19
    if kind == "accepted_only":
        work, _, services = selectively_accepted_v30_workspace(
            tmp_path,
            accepted=(0, 2),
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
    for index in sorted({0, expected_length // 2, expected_length - 1}):
        image = samples[index]["observation.images.main"]
        assert tuple(image.shape) == (3, 24, 32)

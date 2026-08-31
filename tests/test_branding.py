"""Public package and command contracts for the Robo-annotate distribution."""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path

import pytest


def test_robo_annotate_is_the_public_python_package() -> None:
    """A source install must expose the documented import package."""
    try:
        package = importlib.import_module("robo_annotate")
    except ModuleNotFoundError:
        pytest.fail("the robo_annotate public package is not installed")

    assert package.__name__ == "robo_annotate"
    assert callable(package.load_config)


def test_robo_annotate_console_command_loads_the_cli() -> None:
    """The installed distribution must expose the branded console command."""
    commands = [
        entry_point
        for entry_point in importlib.metadata.entry_points(group="console_scripts")
        if entry_point.name == "Robo-annotate"
    ]

    assert len(commands) == 1
    cli = commands[0].load()
    assert cli.info.no_args_is_help is True


def test_documentation_explains_lerobot_version_compatibility() -> None:
    """Release docs must describe the supported LeRobot compatibility contract."""
    readme = Path("README.md").read_text(encoding="utf-8")
    operations = Path("docs/operations.md").read_text(encoding="utf-8")

    assert "LeRobot v2.1" in readme
    assert "LeRobot v3.0" in readme
    assert "自动识别" in readme
    assert "--accepted-only" in operations
    assert "重新编码" in operations
    assert "v3-validation" in operations


def test_documentation_names_the_strict_lerobot_version_metadata_key() -> None:
    """Both entry-point docs must name the exact version-detection contract."""
    required_contract = (
        "`meta/info.json` 的 `codebase_version` 仅接受精确值 "
        "`v2.1` 或 `v3.0`"
    )

    for path in (Path("README.md"), Path("docs/operations.md")):
        assert required_contract in path.read_text(encoding="utf-8")


def test_readme_defines_episode_local_frame_coordinate_contract() -> None:
    """The README must define the half-open coordinates used by model output."""
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "局部半开坐标 `[0, length)`" in readme
    assert "边界帧属于后一个 subtask" in readme

"""Public package and command contracts for the Robo-annotate distribution."""

from __future__ import annotations

import importlib
import importlib.metadata

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

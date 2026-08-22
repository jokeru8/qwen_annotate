from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from qwen_annotate.config import AnnotationConfig, load_config


def test_load_config_valid_yaml_and_stable_hash(tmp_path: Path) -> None:
    payload = {
        "source": "/data/source",
        "work_dir": str(tmp_path / "work"),
        "mode": "complete",
        "high_level_instruction": "arrange items",
        "primary_camera": "observation.images.right_eye",
        "refine_cameras": ["observation.images.right_eye"],
        "subtasks": [{"skill": "pick", "text": "Pick the item"}],
        "model": {"endpoint": "http://localhost:8000/v1"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    loaded = load_config(config_path)
    equivalent = load_config(config_path)

    assert loaded.mode == "complete"
    assert loaded.source == Path("/data/source")
    assert loaded.stable_hash() == equivalent.stable_hash()


def test_annotation_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        AnnotationConfig.model_validate(
            {
                "source": "/data/source",
                "work_dir": "/data/work",
                "mode": "complete",
                "high_level_instruction": "arrange items",
                "primary_camera": "camera",
                "refine_cameras": ["camera"],
                "subtasks": [{"skill": "pick", "text": "Pick the item"}],
                "unexpected": True,
            }
        )


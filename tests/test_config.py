import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from robo_annotate.config import (
    AnnotationConfig,
    AugmentationConfig,
    ModelConfig,
    SamplingConfig,
    Subtask,
    load_config,
)


def test_augmentation_defaults_to_disabled_english() -> None:
    payload = {
        "source": "/data/source",
        "work_dir": "/data/work",
        "mode": "complete",
        "high_level_instruction": "Arrange the items.",
        "primary_camera": "camera",
        "refine_cameras": ["camera"],
        "subtasks": [{"skill": "pick", "text": "Pick up the item."}],
    }

    default = AnnotationConfig.model_validate(payload)

    assert default.model_dump()["augmentation"] == {
        "enabled": False,
        "language": "English",
    }


def test_enabling_augmentation_is_a_material_run_change() -> None:
    payload = {
        "source": "/data/source",
        "work_dir": "/data/work",
        "mode": "complete",
        "high_level_instruction": "Arrange the items.",
        "primary_camera": "camera",
        "refine_cameras": ["camera"],
        "subtasks": [{"skill": "pick", "text": "Pick up the item."}],
    }

    disabled = AnnotationConfig.model_validate(payload)
    enabled = AnnotationConfig.model_validate(
        payload | {"augmentation": {"enabled": True}}
    )

    assert enabled.augmentation == AugmentationConfig(enabled=True, language="English")
    assert disabled.stable_hash() != enabled.stable_hash()


def test_disabled_augmentation_preserves_pre_feature_config_hash() -> None:
    config = AnnotationConfig.model_validate({
        "source": "/data/source",
        "work_dir": "/data/work",
        "mode": "complete",
        "high_level_instruction": "Arrange the items.",
        "primary_camera": "camera",
        "refine_cameras": ["camera"],
        "subtasks": [{"skill": "pick", "text": "Pick up the item."}],
    })
    legacy_payload = config.model_dump(mode="json", exclude={"model": {"api_key"}})
    legacy_payload.pop("augmentation")
    canonical = json.dumps(legacy_payload, sort_keys=True, separators=(",", ":"))

    assert config.stable_hash() == hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize("language", [" ", " English "])
def test_augmentation_language_rejects_blank_or_padded_values(language: str) -> None:
    with pytest.raises(ValidationError, match="language"):
        AugmentationConfig(language=language)


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

    reordered = dict(reversed(list(payload.items())))
    reordered["subtasks"] = [{"text": "Pick the item", "skill": "pick"}]
    reordered_path = tmp_path / "reordered.yaml"
    reordered_path.write_text(yaml.safe_dump(reordered), encoding="utf-8")
    assert loaded.stable_hash() == load_config(reordered_path).stable_hash()

    api_key_changed = dict(payload)
    api_key_changed["model"] = {"endpoint": "http://localhost:8000/v1", "api_key": "other"}
    api_key_path = tmp_path / "api-key.yaml"
    api_key_path.write_text(yaml.safe_dump(api_key_changed), encoding="utf-8")
    assert loaded.stable_hash() == load_config(api_key_path).stable_hash()

    material_change = dict(payload)
    material_change["high_level_instruction"] = "different instruction"
    material_path = tmp_path / "material.yaml"
    material_path.write_text(yaml.safe_dump(material_change), encoding="utf-8")
    assert loaded.stable_hash() != load_config(material_path).stable_hash()


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


@pytest.mark.parametrize(
    ("model_type", "payload"),
    [
        (Subtask, {"skill": "pick", "text": "Pick", "unexpected": True}),
        (AugmentationConfig, {"unexpected": True}),
        (ModelConfig, {"unexpected": True}),
        (SamplingConfig, {"unexpected": True}),
    ],
)
def test_nested_models_reject_unknown_keys(
    model_type: type[BaseModel], payload: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        model_type.model_validate(payload)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://alice:secret@127.0.0.1:8000/v1",
        "http://127.0.0.1:8000/v1?token=secret",
        "http://127.0.0.1:8000/v1#fragment",
    ],
)
def test_model_endpoint_rejects_non_path_url_components(endpoint: str) -> None:
    with pytest.raises(ValidationError, match="userinfo, query, and fragment"):
        ModelConfig.model_validate({"endpoint": endpoint})


def test_equivalent_safe_model_endpoints_have_one_canonical_serialization() -> None:
    upper = ModelConfig.model_validate({"endpoint": "HTTP://LOCALHOST:8000/v1"})
    canonical = ModelConfig.model_validate({"endpoint": "http://localhost:8000/v1"})
    assert str(upper.endpoint) == str(canonical.endpoint) == "http://localhost:8000/v1"

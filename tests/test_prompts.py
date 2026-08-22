import json
from copy import deepcopy

import pytest

from qwen_annotate.config import AnnotationConfig
from qwen_annotate.models import CoarseResult, RefineResult
from qwen_annotate.prompts import (
    PROMPT_VERSION,
    build_coarse_prompt,
    build_refine_prompt,
    coarse_json_schema,
    refine_json_schema,
)


def config(mode: str = "complete", subtasks=None) -> AnnotationConfig:
    return AnnotationConfig.model_validate(
        {
            "source": "/tmp/source",
            "work_dir": "/tmp/work",
            "mode": mode,
            "high_level_instruction": "Arrange the items.",
            "primary_camera": "cam.eye",
            "refine_cameras": ["cam.eye"],
            "subtasks": subtasks or [{"skill": "pick", "text": "pick"}],
        }
    )


def test_coarse_prompt_contains_ordered_template_and_required_rules():
    prompt = build_coarse_prompt(config("dagger_patch", [{"skill": "pick", "text": "pick"}, {"skill": "place", "text": "place"}]), 2, 20, 3)
    assert PROMPT_VERSION == "coarse-v1/refine-v1"
    assert '0: {"skill": "pick", "text": "pick"}' in prompt
    assert '1: {"skill": "place", "text": "place"}' in prompt
    assert "Do not invent or rewrite labels" in prompt
    assert "[k] or [k, k+1, ..., N-1]" in prompt
    assert "first frame of the next subtask" in prompt
    assert "episode index 2" in prompt and "20 frames" in prompt and "pass id 3" in prompt


def test_complete_prompt_requires_full_sequence_and_boundaries():
    prompt = build_coarse_prompt(config("complete", [{"skill": "a", "text": "A"}, {"skill": "b", "text": "B"}, {"skill": "c", "text": "C"}]), 0, 10, 0)
    assert "exactly [0, 1, 2]" in prompt
    assert "start at 0" in prompt
    assert "boundaries consecutive" in prompt


def test_template_escaping_preserves_unicode_and_quotes():
    prompt = build_coarse_prompt(config(subtasks=[{"skill": 'café "tool"', "text": "do \"this\""}]), 0, 2, 0)
    assert '0: {"skill": "café \\"tool\\"", "text": "do \\"this\\""}' in prompt


def test_refine_prompt_names_exact_pair_and_coarse_center():
    prompt = build_refine_prompt(config(subtasks=[{"skill": "a", "text": "A"}, {"skill": "b", "text": "B"}]), 1, 30, 0, 1, 14, 2)
    assert "from_subtask_index=0" in prompt and "to_subtask_index=1" in prompt
    assert 'skill": "a"' in prompt and 'text": "A"' in prompt
    assert "coarse center frame 14" in prompt
    assert "last_frame_before" in prompt and "first_frame_after" in prompt and "boundary_frame" in prompt
    assert "boundary_frame == first_frame_after" in prompt
    assert "no other transition" in prompt


@pytest.mark.parametrize("args", [(True, 1, 0), (-1, 1, 0), (0, 0, 0), (0, 1, -1)])
def test_coarse_args_are_strictly_validated(args):
    with pytest.raises((TypeError, ValueError)):
        build_coarse_prompt(config(), *args)


def test_refine_rejects_invalid_pair_and_coarse_frame():
    cfg = config(subtasks=[{"skill": "a", "text": "A"}, {"skill": "b", "text": "B"}, {"skill": "c", "text": "C"}])
    with pytest.raises(ValueError):
        build_refine_prompt(cfg, 0, 10, 0, 2, 3, 0)
    with pytest.raises(ValueError):
        build_refine_prompt(cfg, 0, 10, 0, 1, 10, 0)


def test_schemas_match_models_are_independent_and_strict():
    coarse = coarse_json_schema()
    refine = refine_json_schema()
    assert coarse == CoarseResult.model_json_schema()
    assert refine == RefineResult.model_json_schema()
    assert coarse is not CoarseResult.model_json_schema()
    mutated = deepcopy(coarse)
    mutated["properties"]["confidence"]["x"] = 1
    assert "x" not in coarse_json_schema()["properties"]["confidence"]
    def walk(node):
        if isinstance(node, dict):
            assert "label" not in node and "subtask_text" not in node
            for value in node.values(): walk(value)
        elif isinstance(node, list):
            for value in node: walk(value)
    walk(coarse); walk(refine)
    assert coarse["additionalProperties"] is False
    assert refine["additionalProperties"] is False
    assert coarse["$defs"]["CoarseBoundary"]["additionalProperties"] is False


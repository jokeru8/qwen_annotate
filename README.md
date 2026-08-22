# Qwen LeRobot Annotate

Typed configuration and tooling for annotating LeRobot v2.1 datasets with Qwen3.8.

## Setup

```bash
uv sync --extra dev
```

Annotation reads source datasets without modifying them. The workflow is:

`inspect` → `annotate` → `review` → `convert` → `validate`

Configuration examples are in `examples/`. The command-line interface will be added in a later step.

# Qwen LeRobot Annotate

用官方 `Qwen/Qwen3.8-27B` 为 LeRobot v2.1 数据集批量生成 subtask 边界标注。模型只能从 YAML 中给出的有序 subtask 列表选择；系统先做完整 episode 的 coarse 判断，再围绕候选边界做多相机 refine。不能通过硬约束或两次判断不一致的 episode 会进入 `needs_review`，不会被完整发布流程静默接受。

项目支持两类数据：

- `complete`：从 subtask 0 开始，依序完成全部 N 项，必须得到 N−1 个边界。
- `dagger_patch`：可从任意 subtask `k` 开始；合法结果是仅包含 `[k]` 并提前结束，或完整执行后缀 `[k, ..., N−1]`。

边界采用左闭右开语义：`boundaries: [180]` 表示下一 subtask 从原视频 frame 180 开始。

## 快速开始

```bash
cd /mnt/data/user/zhoukr/qwen_annotate
python3 -m venv --copies /mnt/data/user/zhoukr/envs/qwen-annotate
UV_PROJECT_ENVIRONMENT=/mnt/data/user/zhoukr/envs/qwen-annotate uv sync --extra dev

UV_PROJECT_ENVIRONMENT=/mnt/data/user/zhoukr/envs/qwen-annotate uv run qwen-annotate inspect examples/complete.yaml
UV_PROJECT_ENVIRONMENT=/mnt/data/user/zhoukr/envs/qwen-annotate uv run qwen-annotate annotate examples/complete.yaml --max-concurrency 1
UV_PROJECT_ENVIRONMENT=/mnt/data/user/zhoukr/envs/qwen-annotate uv run qwen-annotate status /mnt/data/user/zhoukr/annotations/arrange_orange_juice_and_green_tea_2
```

模型的固定目标是：

- 仓库：`Qwen/Qwen3.8-27B`
- 路径：`/mnt/data/user/zhoukr/models/Qwen3.8-27B`
- vLLM served name：`Qwen/Qwen3.8-27B`

配置样例见 [examples/complete.yaml](examples/complete.yaml) 与 [examples/dagger_patch.yaml](examples/dagger_patch.yaml)。完整下载、服务启动、人工复核、转换、验证、评测、扩容和故障恢复步骤见 [docs/operations.md](docs/operations.md)。

## 主流程

```text
inspect → annotate (coarse → refine) → status/review → convert → validate → evaluate
```

标注阶段只读 source，不复制或修改源 parquet/video。workspace 保存每个 episode 的原子状态、模型 revision、prompt/config/source 指纹和审计日志；`convert` 才创建发布数据集。输出目录已存在时拒绝覆盖。

常用命令：

```bash
uv run qwen-annotate annotate CONFIG.yaml --episodes 0,3,8 --max-concurrency 2
uv run qwen-annotate status WORK_DIR --json
uv run qwen-annotate review WORK_DIR
uv run qwen-annotate review WORK_DIR --apply decision_episode_000003.json
uv run qwen-annotate convert WORK_DIR --output DATASET_ANNOTATED
uv run qwen-annotate validate DATASET_ANNOTATED --source SOURCE_DATASET
```

默认完整转换要求所有 episode 都为 `accepted`。需要只发布已接受子集时，显式使用 `--accepted-only`；该模式会连续重编号 episode/frame/global index 并重新计算统计量。

## 开发验证

```bash
uv run pytest -q
uv run qwen-annotate inspect examples/complete.yaml
```

CLI 错误信息会隐藏底层服务响应和 traceback；详细阶段结果在 workspace 的 `episodes/`、`summary.json` 和 `logs/run.jsonl` 中。

## Agent 接手指南

后续 agent 应先阅读本节，再按任务需要阅读 [docs/operations.md](docs/operations.md)、相关代码和测试。不要仅凭历史设计计划推断当前实现；代码、测试以及操作手册中的已实测记录优先。

### 当前状态

截至 2026-08-23：

- 当前 prompt 版本是 `coarse-v6/refine-v2`，定义在 `src/qwen_annotate/prompts.py`。
- 已验证模型 revision 为 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`，探索性运行使用 vLLM `0.27.1`。
- refine-v1 在 episode 0--4 上只是失败基线；它促成了 refine-v2，不能代表当前版本效果。
- refine-v2 的 47 episode golden benchmark 尚未完成，不能声称 launch gates 已通过。
- 受控 8 episode 并发基准尚未完成。探索性运行中并发 2 可以推进，并发 4 出现过 120 秒超时和 deferred 请求；新配置仍应从单 episode、并发 1 开始验证。

新实测结果应更新到 `docs/operations.md`，并明确区分实测、推断和未执行事项。

### 代码地图

```text
src/qwen_annotate/
  cli.py                Typer CLI、参数和退出行为
  config.py             严格 YAML schema 与稳定配置 hash
  models.py             推理、状态和审计领域模型
  constraints.py        complete/DAgger 硬约束
  lerobot.py            LeRobot v2.1 索引和只读数据检查
  video.py              精确抽帧和视觉样本编码
  prompts.py            版本化 coarse/refine prompt 与 JSON schema
  qwen_client.py         OpenAI-compatible 异步客户端、重试和脱敏
  coarse.py              整集稀疏判断与两次一致性检查
  refine.py              broad/dense 多相机边界精修
  pipeline.py            episode/批处理编排与状态流转
  workspace.py           指纹、原子持久化、恢复与审计
  review.py              静态复核页面和人工 decision 导入
  converter.py           完整/accepted-only 数据集转换
  stats.py               parquet 和视频统计量重算
  release_validator.py   独立发布一致性验证
  evaluation.py          golden/DAgger 指标与 launch gates
  model_manager.py       固定 revision 下载和完整性验证
  templates/, static/    离线复核页面资源
```

`tests/test_*.py` 基本与模块一一对应，跨模块主流程见 `tests/test_end_to_end.py`。定位行为时先用 `rg` 查找对应实现和测试。

### Workspace 与兼容性

- `episodes/*.json` 是权威状态，`summary.json` 可恢复，`logs/run.jsonl` 是派生审计链。
- `accepted`、`needs_review` 和 `failed` 是当前运行的终态；重复 annotate 会跳过它们。
- source、work_dir、模式、subtask、模型 endpoint/revision、sampling 或 prompt 变化都会影响 provenance；不同配置不得复用同一 workspace。
- prompt、配置、模型 revision 或 source 指纹不匹配时必须 fail closed。
- 旧 schema workspace 只用于审计。重跑时创建新的空 `work_dir`，不要编辑旧 episode JSON、删除 manifest 或伪造指纹继续运行。

### 修改约束

- `inspect`、`annotate`、`review` 和 `evaluate` 不得修改 source 数据集。
- `convert` 只能写新的输出目录，已有输出必须拒绝覆盖。
- 不要删除、重写或手工“修复”用户已有 workspace、模型目录和数据集。
- 模型响应、confidence、内部 prompt 和 decision source 不得写入发布数据集。
- CLI 面向用户的错误必须脱敏，不能直接输出底层服务响应或 traceback。
- 修改 prompt schema 或语义时必须递增 `PROMPT_VERSION`，更新测试和操作手册，并使用新 workspace 验证。
- 不得为了通过 benchmark 修改 golden 数据或放宽 deterministic hard constraints。
- `--max-concurrency` 不会自动启动多 GPU worker；不要把 worker 数硬编码为 8。
- 开始和结束时运行 `git status --short`，保留用户已有修改，不回滚无关文件。

### 交付检查

代码变更至少执行对应的 focused tests；条件允许时执行全量测试：

```bash
uv run pytest tests/test_<module>.py -q
uv run pytest -q
uv run qwen-annotate --help
```

涉及真实数据读取、模型推理、转换或评测时，按 `docs/operations.md` 从最小样本逐级验证，并记录模型 SHA、vLLM 版本、命令、workspace、episode 集和结果。不要把 mock 测试、只读 inspect、partial run 或旧 prompt 的结果描述成 launch gate 成功。

交付说明应列出修改内容、实际验证结果、因数据/模型/GPU/服务不可用而未运行的检查，以及留给下一位 agent 的风险或未完成事项。

### 深入文档

- [操作手册](docs/operations.md)：当前实现的权威运行步骤、实测证据、故障恢复和发布清单。
- [设计文档](docs/superpowers/specs/2026-08-22-qwen38-lerobot-annotation-design.md)：项目最初设计与系统边界。
- [历史实现计划](docs/superpowers/plans/2026-08-22-qwen38-lerobot-annotation.md)：用于理解实现来源，不代表其中未勾选内容仍未实现。

若文档与实现冲突，先通过代码和测试确认事实，再在同一变更中修正文档。

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

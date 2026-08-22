# Qwen3.8 LeRobot 标注操作手册

本文描述当前代码已经实现的可复现流程。所有示例均以官方仓库 `Qwen/Qwen3.8-27B`、固定安装目录 `/mnt/data/user/zhoukr/models/Qwen3.8-27B` 和 LeRobot v2.1 为准。

## 1. 环境与只读预检

```bash
cd /mnt/data/user/zhoukr/qwen_annotate
python3 -m venv --copies /mnt/data/user/zhoukr/envs/qwen-annotate
UV_PROJECT_ENVIRONMENT=/mnt/data/user/zhoukr/envs/qwen-annotate uv sync --extra dev
export UV_PROJECT_ENVIRONMENT=/mnt/data/user/zhoukr/envs/qwen-annotate

uv run pytest -q
uv run qwen-annotate inspect examples/complete.yaml
```

`inspect` 在推理前严格检查 v2.1 版本、连续 episode index、parquet 行数、任务引用、相机集合、每路视频 frame/FPS/尺寸和 metadata 总数。它不创建 workspace。

2026-08-22 的只读预检结果如下，不能视为模型 smoke 结果：

- 机器可见 8 张 NVIDIA H20，每张总显存 97,871 MiB；检查时每张约有 97,284 MiB 空闲。
- `/mnt/data/user/zhoukr/envs/vllm` 中安装的 vLLM package 版本为 `0.27.1`；`vllm --version` 在 10 秒预检超时内未返回，因此启动前仍需现场确认 CLI 可用。
- 精确目标目录 `/mnt/data/user/zhoukr/models/Qwen3.8-27B` 尚不存在，尚未产生 `model-install.json`。
- 本机存在 `/mnt/data/user/zhoukr/models/Qwen3.8-27B-FP8`，但它不是本项目要求的精确 repo/path，不能替代经过本 CLI 固定 revision 并验证的安装。
- 因模型未安装，真实单 episode smoke、47 episode golden benchmark 和并发 1/2/4 吞吐测试均尚未执行；本文不填写虚构的延迟、显存或吞吐数值。

## 2. 配置

完整数据示例：

```yaml
source: /mnt/data/user/zhoukr/datasets/jokeru/arrange_orange_juice_and_green_tea_2
work_dir: /mnt/data/user/zhoukr/annotations/arrange_orange_juice_and_green_tea_2
mode: complete
high_level_instruction: arrange the orange juice and green tea neatly
primary_camera: observation.images.right_eye
refine_cameras:
  - observation.images.right_eye
  - observation.images.left_wrist
  - observation.images.right_wrist
subtasks:
  - {skill: pick, text: Pick up green tea using right hand}
  - {skill: pick, text: Take the green tea from your right hand with your left hand.}
  - {skill: place, text: Pick up orange juice using right hand and put it in the correct place}
  - {skill: place, text: Take the green tea with right hand and put it in the correct place}
model:
  name: Qwen/Qwen3.8-27B
  local_path: /mnt/data/user/zhoukr/models/Qwen3.8-27B
  endpoint: http://127.0.0.1:8000/v1
  api_key: local
sampling:
  coarse_fps: 1.0
  coarse_max_frames: 64
  refine_window_seconds: 2.5
  refine_fps: 8.0
  dense_radius_seconds: 0.5
  agreement_tolerance_frames: 12
  min_segment_frames: 8
```

DAgger 配置只把 `mode` 改为 `dagger_patch`，并为该批数据使用独立 `work_dir`。不要预先填写起始 subtask：coarse 阶段会从同一模板中判断 `k`。合法 coarse 序列只有单项 `[k]` 或后缀 `[k, k+1, ..., N-1]`；前者表示在进入下一项前结束，后者表示一直执行到任务结束。

配置禁止未知字段。`source`、`work_dir`、模式、subtask 文本/顺序、模型 endpoint/revision 或任何 sampling 参数的改变都会改变运行来源；不要在原 workspace 上混用不同配置。

## 3. 固定 revision 下载与验证

下载命令先把可变 revision 解析成 40 字符小写 commit SHA，再用 `hf download` 断点下载，以 `hf cache verify --fail-on-missing-files` 验证所有 repo 文件，最后原子写入 `model-install.json`：

```bash
uv run qwen-annotate model download \
  --repo Qwen/Qwen3.8-27B \
  --local-dir /mnt/data/user/zhoukr/models/Qwen3.8-27B \
  --max-workers 8
```

成功标准是 stdout 出现 `revision: <40-char-sha>`，且下面的检查返回同一 repo、SHA 和绝对路径：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('/mnt/data/user/zhoukr/models/Qwen3.8-27B/model-install.json')
d = json.loads(p.read_text())
assert d['repo'] == 'Qwen/Qwen3.8-27B'
assert len(d['revision']) == 40 and set(d['revision']) <= set('0123456789abcdef')
assert d['local_path'] == '/mnt/data/user/zhoukr/models/Qwen3.8-27B'
print(d)
PY
```

若网络中断，保留目标目录中的 resumable partial files，再运行同一命令；未通过 verify 时不会保留可信的 `model-install.json`。不要手工伪造该文件，也不要把 mutable `main` 当作已固定 revision。

## 4. vLLM 单 episode smoke

初始服务命令必须先以单 GPU 验证：

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/user/zhoukr/envs/vllm/bin/vllm serve \
  /mnt/data/user/zhoukr/models/Qwen3.8-27B \
  --served-model-name Qwen/Qwen3.8-27B \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --port 8000
```

另一个终端只跑参考 episode 0：

```bash
uv run qwen-annotate inspect examples/complete.yaml
uv run qwen-annotate annotate examples/complete.yaml --episodes 0 --max-concurrency 1
uv run qwen-annotate status \
  /mnt/data/user/zhoukr/annotations/arrange_orange_juice_and_green_tea_2 --json
```

检查 `episodes/episode_000000.json`：应保存两次合法 coarse attempt、每个候选边界的 broad/dense refine attempt、真实 frame index/camera 采样 provenance、`prompt_version` 和固定模型 SHA。客户端对每次结构化 completion 最多发出 4 次请求（包括瞬态重试与最多一次格式修复），默认单请求 timeout 为 120 秒。

真实 smoke 完成后在本节追加实际值，不可估算：

```text
model_revision: PENDING_MODEL_DOWNLOAD
vllm_version: 0.27.1 package present; server CLI/runtime confirmation pending
episode: 0
coarse_valid_responses: PENDING
refine_valid_responses: PENDING
visual_frame_count: PENDING
request_latency_seconds: PENDING
peak_gpu_memory_mib: PENDING
result: BLOCKED_BY_MISSING_TARGET_MODEL
```

## 5. 批量运行、两阶段 prompt 与恢复

```bash
uv run qwen-annotate annotate CONFIG.yaml --max-concurrency 1
uv run qwen-annotate annotate CONFIG.yaml --episodes 0,3,8 --max-concurrency 2
uv run qwen-annotate status WORK_DIR
uv run qwen-annotate status WORK_DIR --json
```

`--episodes` 是无空格、无重复的非负整数列表。筛选只限制本次要推进的 episode，status 仍显示整个 workspace。

每个 episode 的阶段是 `pending → coarse_done → refine_done → accepted`。coarse 使用主相机、覆盖完整时间范围的两套独立稀疏采样，必须在合法模板序列上达成一致。refine 对每个 transition 先用配置相机做 broad window，再在候选点附近做 dense 连续帧判断；最终边界是“下一 subtask 的第一帧”。模型只能引用 prompt 中带整数 index 的模板，不能发明标签。

当前 prompt 版本是 `coarse-v5/refine-v1`。coarse 使用两个分层字段：`semantic_uncertainty_codes` 是唯一阻断通道，只允许 `subtask_order_unclear`、`start_subtask_unclear`、`transition_neighborhood_unclear`；任一非空就进入 `coarse_uncertain`，且模型不得猜测。若顺序、起始项和大致过渡邻域都明确，模型必须返回最合理的粗略中心并保持该 codes 列表为空，由 refine 完成精确定位。`boundary_precision_notes` 只记录稀疏采样下的精确帧误差，不阻断 coarse/refine。旧的泛化 `uncertainties` 字段不再属于 schema。所有正常请求、瞬态重试和格式修复请求都显式设置 greedy `temperature=0`，不继承 vLLM 模型目录中可能为随机采样的 `generation_config`。v5 会使已有 v4 及更早 workspace 的 run fingerprint 不匹配并 fail closed；旧结果必须在新的空 workspace 中重新标注，不能复用。

重复执行同一命令会跳过 `accepted`、`needs_review` 和 `failed`，并从持久的 `coarse_done`/`refine_done` 继续。每个 episode 保存 source SHA-256 指纹；run fingerprint 绑定完整有效配置（API key 除外）、`PROMPT_VERSION`、model repo 和固定 revision。源 payload、prompt、模型或行为配置发生变化时，旧 workspace 会 fail closed，而不是复用过期结果。当前 CLI 没有“强制清缓存”选项；需要重跑时使用新的空 `work_dir`，保留旧目录用于审计。

workspace 内 `episodes/*.json` 是权威状态，`summary.json` 可恢复，`logs/run.jsonl` 是带 event id 的派生审计链。不要手工修改这些文件。

## 6. 状态、原因码与人工复核

状态含义：

- `pending`：尚未完成 coarse。
- `coarse_done`：两次 coarse 已通过，等待 refine。
- `refine_done`：refine 结果已持久化，等待最终原子状态转换。
- `accepted`：模型或人工结果已通过约束，可转换。
- `needs_review`：结果可审计但不能自动接受。
- `failed`：基础设施/源数据/持续 OOM 等导致阶段无法完成。

常见 `needs_review` 原因包括：

- coarse：`coarse_sequence_disagreement`、`illegal_coarse_sequence`、`coarse_boundary_count`、`coarse_boundary_order`、`coarse_uncertain`、`invalid_model_response`。
- refine/硬约束：`refine_boundary_disagreement`、`refine_transition_mismatch`、`camera_evidence_conflict`、`start_subtask_range`、`complete_start_index`、`complete_boundary_count`、`dagger_suffix_length`、`boundary_order`、`boundary_range`、`segment_too_short`。

`failed.failure_category` 为 `model_oom`、`model_call`、`source_or_video`、`unexpected_error` 或 `workspace_state`。episode JSON 保存 coarse/refine 成功解析后的结构化 attempt、规范化 category/reason 和采样 provenance；对于请求异常或无效模型响应，它不会保存客户端的 `attempt_count`、响应摘录或逐次请求诊断。CLI 同样只输出脱敏摘要。排查 timeout、5xx、限流、worker 退出或 OOM 的逐请求细节时，应同时保留 vLLM server 的 stdout/stderr 日志和外部负载均衡日志，并按 episode 运行时段与 workspace `logs/run.jsonl` 的 UTC transition 时间关联；不要期待从 workspace 还原未持久化的服务响应。

生成离线页面并导入浏览器导出的严格 decision JSON：

```bash
uv run qwen-annotate review WORK_DIR
# stdout 是 WORK_DIR/previews/needs_review/index.html
uv run qwen-annotate review WORK_DIR --apply decision_episode_000003.json
```

decision 文件必须只含 `episode_index`、`source_fingerprint`、`run_fingerprint`、`mode`、`start_subtask_index` 和 `boundaries`。导入时会重新检查指纹、边界范围/顺序、最短 segment 和 complete/DAgger 约束；通过后记录 `decision_source: human`。

## 7. 转换与独立验证

完整发布保留源 payload 字节和编号，要求每个 episode 都已接受：

```bash
uv run qwen-annotate convert WORK_DIR --output /path/to/dataset_annotated
uv run qwen-annotate validate /path/to/dataset_annotated --source /path/to/source
```

输出兼容参考公开格式：`meta/info.json` 增加 `subtask_template`/逐 episode instruction；新增 `meta/lerobot_annotations.json` 和 `meta/task_info/task_0.json`。complete episode 省略 `start_subtask_index`，DAgger episode 明确保存它（包括 0）。不会发布内部 confidence、prompt 或 decision source。

只发布 accepted 子集：

```bash
uv run qwen-annotate convert WORK_DIR --accepted-only --output /path/to/accepted_subset
uv run qwen-annotate validate /path/to/accepted_subset
```

accepted-only 会连续重编号 episode、`frame_index`/`episode_index`/全局 `index`，复制选中视频并对 parquet 与全部视频像素重新计算 stats。因此它是新数据集；原 source 不能作为同编号 payload checksum 基准，验证时通常省略 `--source`。

默认 `validate` 是 `strict_deep`：独立读取发布目录，校验 metadata/schema/index/timestamp、视频、annotation/task_info、payload 集、完整数值 quantile 和逐像素视频统计。可用 `--no-deep-video-stats` 得到 `strict_structural`，它会明确报告跳过 `video_payload_stat_equality`。

历史参考集的 image stats 是抽样统计，严格 deep 必然拒绝；仅在明确接受较弱保证时使用：

```bash
uv run qwen-annotate validate LEGACY_DATASET \
  --allow-legacy-sampled-image-stats --no-deep-video-stats
```

该模式报告 `legacy_structural`（无 source）或 `source_backed_legacy`（提供 source），并明确列出跳过的 numeric quantile/video payload stats checks。`--allow-legacy-sampled-image-stats` 与默认 deep 互斥，不能省略 `--no-deep-video-stats`。转换与验证都拒绝覆盖、symlink/特殊文件、缺失/额外 payload 和 source checksum 变化。

## 8. Golden 评测门槛

完成 47 episode workspace 后运行：

```bash
uv run qwen-annotate evaluate \
  /mnt/data/user/zhoukr/annotations/arrange_orange_juice_and_green_tea_2 \
  --golden /mnt/data/user/zhoukr/datasets/jokeru/arrange_orange_juice_and_green_tea_2_annotated \
  --output /mnt/data/user/zhoukr/annotations/arrange_orange_juice_and_green_tea_2/metrics.json
```

全部 launch gates 才返回成功：median boundary error ≤0.5 s、p90 ≤1.0 s、accepted coverage ≥0.85、constraint blocking rate =1.0、false accept count =0。当前因目标模型尚未下载，尚无真实 metrics，不能声称门槛已通过。

若失败，只调整 prompt 文本、sampling 密度/window 或接受阈值；修改 prompt 时递增 `PROMPT_VERSION`，使用新 workspace 使受影响 cache 失效，然后保存前后 `metrics.json`。不要为通过 benchmark 修改 golden 标注或硬约束。

## 9. 并发、安全扩容与记录模板

先以 `--max-concurrency 1` 验证正确性。该选项限制单进程同时处理的 episode 数和对同一 endpoint 的并发请求，并不自动启动多张 GPU 或多个 vLLM server。当前配置只有一个 endpoint；要使用多 worker，应在多个单 GPU vLLM replica 前放置一个保持相同 URL/served-model-name 的负载均衡入口。不要让多个不同 endpoint 配置写同一 workspace，因为 endpoint 属于 run fingerprint。

同一 workspace 的持久化使用文件锁和原子替换，但基准测试时仍应让外部进程使用互不重叠的 `--episodes` shard，避免重复昂贵推理。不要把 worker 数硬编码为 8。

正确性通过后，对同一固定 8 episode 列表依次测试并发 1、2、4：

```bash
uv run qwen-annotate annotate BENCH_C1.yaml --episodes 0,1,2,3,4,5,6,7 --max-concurrency 1
uv run qwen-annotate annotate BENCH_C2.yaml --episodes 0,1,2,3,4,5,6,7 --max-concurrency 2
uv run qwen-annotate annotate BENCH_C4.yaml --episodes 0,1,2,3,4,5,6,7 --max-concurrency 4
```

每个配置必须使用独立空 workspace，但固定相同模型 SHA、prompt、source、sampling 和 episode 集。记录模板：

```text
concurrency | model_revision | episodes/hour | peak GPU MiB | decode CPU % | error rate | needs_review count
1           | PENDING        | PENDING       | PENDING      | PENDING      | PENDING    | PENDING
2           | PENDING        | PENDING       | PENDING      | PENDING      | PENDING    | PENDING
4           | PENDING        | PENDING       | PENDING      | PENDING      | PENDING    | PENDING
```

默认采用零 OOM、零额外 `needs_review` 前提下最快的配置；目前没有真实数据，因此项目默认仍使用安全的并发 1。

## 10. 故障恢复

- timeout/临时 5xx/限流：客户端在有限预算内退避重试。服务恢复后再次执行相同 annotate 命令，`pending`/中间状态会恢复；`failed` 是终态，需要新 workspace 重跑。
- OOM：多相机 broad 首次 OOM 时会退化为仅主相机并加大 stride；再次 OOM 或 dense OOM 记为 `failed/model_oom`。降低 vLLM 显存压力或 `--max-concurrency`，然后用新 workspace 重跑，不要手改 episode JSON。
- 损坏/缺失视频：`inspect` 或推理记为 `source_or_video`。先在 source 所属采集/修复流程恢复合法 v2.1 payload，再重新 inspect，并使用新 workspace；source 指纹改变时旧结果会被拒绝。
- 中断/进程退出：原子 stage 状态保留；直接重复 annotate。status 的 summary 可由权威 episode 文件恢复。
- workspace/provenance 损坏：保留目录取证，使用新空 workspace。不要删除 manifest 后继续写旧 episode。
- conversion 失败：最终 output 不会部分覆盖；修复原因后换一个不存在的 output 路径重跑。只删除工具创建且确认属于本次失败的 staging 目录。

## 11. 发布前检查清单

```bash
uv run pytest -q
uv run qwen-annotate inspect examples/complete.yaml
uv run qwen-annotate status WORK_DIR --json
uv run qwen-annotate convert WORK_DIR --output DATASET_ANNOTATED
uv run qwen-annotate validate DATASET_ANNOTATED --source SOURCE_DATASET
uv run qwen-annotate evaluate WORK_DIR --golden GOLDEN_DATASET --output WORK_DIR/metrics.json
```

确认所有 episode 已接受（或明确选择 accepted-only）、source tree hash 未变、release 独立验证通过、launch gates 全部 PASS，并把实际模型 SHA、vLLM server version、单 episode smoke 与吞吐数据补回本手册后再发布。

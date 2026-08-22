# Qwen3.8 LeRobot v2.1 自动标注系统设计

## 1. 目标

在 `/mnt/data/user/zhoukr/qwen_annotate` 中实现一个可恢复、可审计的批量标注项目，使用官方 `Qwen/Qwen3.8-27B` 多模态模型为 LeRobot v2.1 数据集生成有序 subtask 边界。系统只允许从用户提供的 subtask 模板中选择标签，不让模型自由生成标签；低置信或不满足约束的 episode 标记为 `needs_review`，不会自动进入发布数据。

标注阶段不复制或修改原始数据。发布前通过独立的 `convert` 步骤，生成与 ModelScope 示例 `jokeru/arrange_orange_juice_and_green_tea_2_annotated` 兼容的完整数据集。

## 2. 已验证的参考数据

本设计以以下配对数据作为格式和效果基准：

- 原始集：`/mnt/data/user/zhoukr/datasets/jokeru/arrange_orange_juice_and_green_tea_2`
- 人工标注集：`/mnt/data/user/zhoukr/datasets/jokeru/arrange_orange_juice_and_green_tea_2_annotated`
- ModelScope 原始集：`jokeru/arrange_orange_juice_and_green_tea_2`
- ModelScope 标注集：`jokeru/arrange_orange_juice_and_green_tea_2_annotated`

已确认参考集有 47 个 episode、28 FPS、4 路相机。人工标注没有改变 parquet 或视频内容；它在 `meta/info.json` 中加入 `subtask_template` 和逐 episode 的 `high_level_instruction`，并新增 `meta/lerobot_annotations.json`。每个完整 episode 使用 3 个整数 frame boundary 表达 4 个依次执行的 subtask。

参考模板的准确内容是：

```json
[
  {"skill": "pick", "text": "Pick up green tea using right hand"},
  {"skill": "pick", "text": "Take the green tea from your right hand with your left hand."},
  {"skill": "place", "text": "Pick up orange juice using right hand and put it in the correct place"},
  {"skill": "place", "text": "Take the green tea with right hand and put it in the correct place"}
]
```

参考集指定 `observation.images.right_eye` 为主相机。项目不能假设所有未来数据都使用相同 FPS、episode 数量或相机名，这些值必须从 LeRobot metadata 读取并验证。

## 3. 支持的数据模式

### 3.1 `complete`

每个 episode 从模板第 0 项开始，严格按顺序执行模板中的全部 subtask，每项恰好一次。若模板有 N 项，每个 episode 必须产生 N−1 个递增边界。

### 3.2 `dagger_patch`

每个 episode 可从任意 subtask 中间开始采集。合法结果只有两种：

1. 仅包含当前 subtask，并在切换到下一项之前结束；
2. 从当前 subtask 开始，沿模板顺序一直执行到任务最后一项。

DAgger episode 使用可选字段 `start_subtask_index` 表示第一帧所属的模板项。`boundaries: []` 表示整个 episode 只属于起始 subtask。若有边界，边界后依次进入模板中的下一项。该字段对完整数据可省略，缺省值为 0，因此与已有完整数据格式向后兼容。

边界统一采用左闭右开语义。例如：

```json
{
  "start_subtask_index": 1,
  "boundaries": [180, 420]
}
```

表示 `[0, 180)` 属于模板项 1，`[180, 420)` 属于模板项 2，`[420, episode_length)` 属于模板项 3。边界值是下一 subtask 开始的第一帧。

## 4. 用户配置

每个数据集使用一个 YAML 配置。配置经过严格 schema 校验，不允许未知字段静默生效。

```yaml
source: /path/to/lerobot_dataset
work_dir: /path/to/annotation_workspace
mode: complete
high_level_instruction: arrange the orange juice and green tea neatly
primary_camera: observation.images.right_eye
refine_cameras:
  - observation.images.right_eye
  - observation.images.left_wrist
  - observation.images.right_wrist
subtasks:
  - skill: pick
    text: Pick up green tea using right hand
  - skill: pick
    text: Take the green tea from your right hand with your left hand.
  - skill: place
    text: Pick up orange juice using right hand and put it in the correct place
  - skill: place
    text: Take the green tea with right hand and put it in the correct place
model:
  name: Qwen/Qwen3.8-27B
  local_path: /mnt/data/user/zhoukr/models/Qwen3.8-27B
  endpoint: http://127.0.0.1:8000/v1
sampling:
  coarse_fps: 1.0
  coarse_max_frames: 64
  refine_window_seconds: 2.5
  agreement_tolerance_frames: 12
```

默认值由版本化配置 schema 提供，并完整写入本次运行的 manifest。命令行可以覆盖运行参数，但不能覆盖 subtask 内容而不使已有缓存失效。

## 5. 系统架构

项目按职责拆分为以下单元：

- CLI：提供模型下载、数据检查、标注、状态查看、复核、转换和最终验证命令。
- LeRobot reader：只读解析 v2.1 metadata、episode parquet、视频路径、FPS、frame count 和 camera keys。
- Sampler：按真实 frame index 从视频抽帧，生成带相机名、frame index 和时间戳的输入图像；不把采样序号误当成原视频帧号。
- Prompt builder：根据模式和模板生成版本化 prompt，并输出严格 JSON Schema。
- Qwen client：调用 vLLM OpenAI-compatible endpoint，处理超时、重试和结构化输出。
- Coarse annotator：整段稀疏浏览，识别 DAgger 起始项、实际出现的连续 subtask 和粗边界。
- Boundary refiner：在粗边界附近使用主相机和腕部相机密集取帧，将边界收敛到具体 frame index。
- Constraint validator：执行与模型无关的 schema、范围、顺序、模式和一致性校验。
- Workspace store：原子保存每个 episode 的中间状态、模型响应、配置指纹和最终状态。
- Review renderer：为 `needs_review` 生成本地静态 HTML 和边界预览图，并接收人工修订。
- Converter：把已确认标注转换为完整的可发布 LeRobot v2.1 数据集。
- Evaluator：在人工 golden set 和合成 DAgger 切片上计算质量指标。

## 6. 两阶段推理

### 6.1 Coarse 阶段

Sampler 默认从主相机以约 1 FPS 均匀抽帧，最多提供 64 帧。每帧必须显示真实 frame index 和时间戳。若 episode 更长，保持覆盖完整时间范围，而不是只截取开头。

Prompt 明确告诉 Qwen：

- high-level instruction；
- 带稳定整数编号的有序 subtask 模板；
- 当前模式；
- complete 和 DAgger 的合法序列；
- 边界的左闭右开定义；
- 只能引用模板编号，禁止改写或新增 subtask；
- 视觉证据不足时报告不确定，禁止猜测。

结构化输出包含 `start_subtask_index`、`observed_subtask_indices`、带转移方向和证据的 `coarse_boundaries`、模型置信度及不确定项。complete 必须返回完整模板序列；DAgger 只能返回单项 `[k]` 或后缀 `[k, k+1, ..., N-1]`。

### 6.2 Refine 阶段

每个粗边界默认截取 ±2.5 秒上下文。主相机用于理解整体任务状态，配置中的腕部相机用于确认抓取、交接和放置等局部事件。所有视觉输入都带 camera key 和真实 frame index。

Refine 在一个阶段内使用自适应密度：先用较密采样缩小边界区间，再读取候选点附近的连续帧。Qwen 只需要判断 `last_frame_before`、`first_frame_after` 和 `boundary_frame`，并给出可见证据。最终边界必须落在有效帧范围内，并与声明的 from/to 模板编号一致。

### 6.3 一致性和重试

系统不单独依赖模型自报 confidence。每个候选还要通过：

- 两次独立判断的边界差是否小于配置阈值；
- coarse 和 refine 的模板编号是否一致；
- 多相机证据是否明显冲突；
- 边界是否严格递增；
- 每段是否满足可配置的最短帧数；
- complete/DAgger 序列是否满足硬约束；
- 输出是否为合法 JSON 且没有模板外标签。

非法 JSON、临时超时或不一致结果自动重试一次。重试仍不能满足接受条件时，episode 标记为 `needs_review`。服务不可用、视频损坏或持续 OOM 等不可完成问题标记为 `failed`，并保留明确原因。

## 7. 标注工作目录

标注阶段不复制视频和 parquet，目录结构为：

```text
annotation_workspace/
├── manifest.json
├── episodes/
│   ├── episode_000000.json
│   └── episode_000001.json
├── previews/
│   └── needs_review/
├── logs/
└── summary.json
```

每个 episode 文件保存：状态、起始 subtask、coarse 结果、refine 结果、最终 boundaries、置信信息、校验结果、原始模型响应、prompt 版本、模型 revision、采样参数、源视频指纹及时间戳。

状态机为：

```text
pending -> coarse_done -> refine_done -> accepted
                              |             
                              +-----------> needs_review
          any stage ----------------------> failed
```

episode 文件先写临时文件，再在同一文件系统内原子重命名。重启后跳过指纹仍有效的已完成阶段。模型 revision、prompt 版本、subtask、模式或采样配置变化时，相关缓存失效。原始视频指纹变化时，该 episode 的全部旧结果失效。

## 8. `needs_review` 工作流

每个待复核 episode 生成静态页面，显示：

- 全局 coarse 时间线；
- 候选边界前后的主相机和腕部相机画面；
- 两次模型判断及差异；
- 触发复核的确定性原因；
- 可编辑的 `start_subtask_index` 和 boundaries。

人工结果走与自动结果相同的 schema 和硬约束校验，并记录 `decision_source: human`。发布数据不保存模型的 chain-of-thought；内部审计只保存模型返回的结构化答案和简短视觉证据。

## 9. 模型下载与服务

官方模型仓库为 `Qwen/Qwen3.8-27B`，本地目标路径为 `/mnt/data/user/zhoukr/models/Qwen3.8-27B`。下载必须固定 Hugging Face commit revision、支持断点续传，并在结束后执行文件完整性校验。机器当前 `hf` CLI 继承的 SOCKS 代理缺少 `socksio`；下载命令应只对自身清除相应代理环境变量或在项目环境内补齐依赖，不修改全局网络配置。

推理后端使用 vLLM OpenAI-compatible API。机器有 8 张约 96GB 的 NVIDIA H20，且现有环境包含 vLLM 0.27.1。第一步只在单 GPU 上验证 Qwen3.8 模型加载、视觉输入兼容性、结构化输出和实际显存占用。通过后以独立 episode 任务队列连接多个单 GPU worker；worker 数量由基准测试确定，而不是写死为 8。

## 10. CLI 工作流

目标命令为：

```bash
qwen-annotate model download --repo Qwen/Qwen3.8-27B --local-dir /mnt/data/user/zhoukr/models/Qwen3.8-27B
qwen-annotate inspect CONFIG.yaml
qwen-annotate annotate CONFIG.yaml
qwen-annotate status WORK_DIR
qwen-annotate review WORK_DIR
qwen-annotate convert WORK_DIR --output DATASET_ANNOTATED
qwen-annotate validate DATASET_ANNOTATED
```

`inspect` 在调用模型前验证 LeRobot 版本、episode metadata、视频数量、camera keys、FPS 和 frame count。`annotate` 可安全重复执行并恢复中断任务。`status` 汇总 pending、accepted、needs_review 和 failed。`review` 生成或打开静态复核页面。`convert` 生成发布数据。`validate` 独立验证生成结果。

## 11. 发布转换

默认情况下，只要存在 `pending`、`needs_review` 或 `failed`，`convert` 就拒绝生成完整数据集。输出目录已存在时拒绝覆盖；恢复不完整转换需要显式选项和匹配的转换 manifest。

转换执行以下动作：

1. 复制原始 parquet、视频和未修改 metadata；
2. 在 `meta/info.json` 中加入全局 `subtask_template` 和按输出 episode index 映射的 `high_level_instruction`；
3. 创建 `meta/lerobot_annotations.json`；
4. complete episode 省略 `start_subtask_index` 或写 0，DAgger episode 写实际起始项；
5. 不把内部 confidence、重试记录或模型响应写入发布数据；
6. 完成后重新读取并验证数据集。

可选的 `--accepted-only` 会只发布已接受或人工确认的 episode，并对 episode index、全局 frame index、metadata、视频路径和 annotations 做连续重编号。它始终生成新数据集，不修改源数据，也不会保留指向被排除 episode 的 metadata。

转换后必须验证：

- 原样复制的 parquet 和视频 SHA-256 与源文件一致；
- episode/frame/task 计数内部一致；
- 每个输出 episode 有且只有一个 annotation entry；
- 所有边界合法；
- `info.json` 与 `lerobot_annotations.json` 的模板和 high-level instruction 一致；
- 随机抽样边界可成功生成预览。

## 12. 错误处理与安全性

- 原始数据集始终以只读方式访问。
- 输出使用独立目录，不允许隐式覆盖。
- 模型调用超时、连接错误和可恢复服务错误采用有限次数、带退避的重试。
- JSON 解析错误可用同一视觉输入进行一次格式修复请求，但修复不能绕过语义校验。
- OOM 首先降低单次视觉输入规模并重试；仍失败则记录为 `failed`。
- 视频缺失、frame count 不一致、相机不存在或 metadata 损坏在 `inspect` 阶段直接阻止运行。
- 每次运行在 manifest 中记录源数据指纹、配置 hash、prompt 版本、代码版本、模型 repo 和 revision。

## 13. 测试与质量验收

单元测试覆盖配置 schema、边界区间语义、complete/DAgger 序列约束、JSON 解析、缓存失效、原子写入和发布 metadata 构建。集成测试使用一个极小的合成 LeRobot v2.1 数据集和可编程的假模型服务，验证从 inspect 到 convert/validate 的完整路径、中断恢复、needs_review 和 accepted-only 重编号。

效果评测使用已下载的 47 个人工标注 episode 作为 golden set。人工边界不提供给待评测推理请求，只用于离线计算：

- `start_subtask_index` 准确率；
- boundary 绝对帧误差的中位数和 P90；
- accepted coverage；
- needs_review 比例；
- 错误结果被接受的比例。

DAgger 评测从人工完整 episode 中创建不改动源视频的 frame-range 视图：一类从中间 subtask 开始并执行到结尾，另一类只保留某个 subtask 内的片段并提前结束。这些切片用于评估起始项识别和提前结束判断。

初始上线门槛为：

- boundary 中位误差不超过 0.5 秒；
- boundary P90 不超过 1 秒；
- 完整数据 accepted coverage 不低于 85%；
- schema、范围和顺序违规的拦截率为 100%；
- golden set 中明显错误的结果不得被标为 accepted。

若模型未达到门槛，优先调整抽帧密度、refine 窗口、prompt 和接受阈值；不通过降低校验标准来提高 accepted coverage。

## 14. 非目标

首版不训练或微调 Qwen，不让模型创造新的 subtask，不支持乱序、跳跃或重复 subtask，不在标注阶段改写 parquet，不构建多人在线审核平台，也不自动上传 ModelScope。未来若数据语义超出“有序模板的完整序列或 DAgger 后缀”约束，需要单独扩展 annotation schema 和评测集。

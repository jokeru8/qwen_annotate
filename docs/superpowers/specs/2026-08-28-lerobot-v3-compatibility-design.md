# LeRobot v3.0 兼容设计

## 背景与目标

Robo-annotate 当前只识别和发布 LeRobot Dataset v2.1。LeRobot Dataset v3.0 改为文件级分片：多个 episode 可以共用 Parquet 和 MP4，episode 的数据行区间及视频时间区间由关系型元数据定位。因此，v3.0 不能通过扩展版本字符串兼容，必须把底层存储定位从“每个 episode 一个文件”抽象为“episode 引用分片中的一个切片”。

本次工作的目标是：

- 输入自动识别 v2.1 或 v3.0，v2.1 继续作为默认、主要和文档优先格式。
- 标注、评估、增广和人工复核使用统一的 episode 本地帧语义，不感知底层版本。
- `convert` 保持源数据集版本，不提供跨版本转换。
- v3.0 首版同时完整支持普通转换和 `convert --accepted-only`。
- 不把官方 `lerobot` 设为运行时必需依赖；项目使用已有的 `pyarrow` 和 `av` 完成读取、切片、重建及独立校验。
- 项目介绍和用户文档使用中文；代码标识符、配置字段、错误消息、生成标注和默认 subtask 内容使用英文。

当前没有真实 v3.0 数据样本。因此首版以严格合成 fixture 和官方 loader 兼容性测试作为交付门槛，未来取得真实样本后补充外部验收，不改变本设计接口。

## 非目标

- 不实现 v2.1 与 v3.0 之间的格式迁移。
- 不修改官方 LeRobot 的核心 schema 来承载 Robo-annotate 标注。
- 不引入远程 Hub 下载、流式 Hub 数据集读取或上传功能。
- 不改变现有 subtask 标注语义、增广规则、接受状态机或人工复核权限。
- 不为 v3.0 建立第二套标注 pipeline。

## 官方格式依据

实现以以下官方资料和与首版固定兼容的官方包为准：

- [LeRobot Dataset v3.0 格式说明](https://github.com/huggingface/lerobot/blob/main/docs/source/lerobot-dataset-v3.mdx)
- [v2.1 到 v3.0 迁移说明](https://github.com/huggingface/lerobot/blob/main/docs/source/porting_datasets_v3.mdx)
- [官方 dataset metadata 实现](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/dataset_metadata.py)
- [官方路径和版本常量](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/utils.py)
- 可选兼容性 oracle：`lerobot[dataset]==0.6.1`

v3.0 的核心路径为：

```text
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet
data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet
videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
```

## 总体架构

读取和发布采用版本适配器，而不是先把 v3.0 临时转换成 v2.1：

```text
meta/info.json
      |
      v
VersionDetector
      |
      +--> V21DatasetAdapter --+
      |                         |
      +--> V30DatasetAdapter --+--> DatasetIndex
                                      |
                         annotation / review / evaluation
                                      |
                                      v
                              VersionedConverter
                                      |
                         +------------+------------+
                         v                         v
                    V21Publisher              V30Publisher
```

各单元职责如下：

- `VersionDetector` 只安全、有限地读取 `meta/info.json`，返回严格的 `v2.1` 或 `v3.0`。
- dataset adapter 解析各版本的官方元数据，完成路径约束和结构检查，生成统一 `DatasetIndex`。
- 标注、增广、评估和 review 只消费统一模型及 episode 本地帧编号。
- publisher 根据 manifest 中的源版本发布同版本结果。
- release validator 独立于 workspace 和 publisher 重新读取成品，避免“由写入代码证明自己正确”。

未知、缺失或格式错误的 `codebase_version` 一律拒绝，不根据目录形状猜测版本。

## 统一数据模型

在现有 `EpisodeInfo` 周围增加两个不可变引用对象：

```text
EpisodeDataRef
  path
  dataset_from_index
  dataset_to_index

EpisodeVideoRef
  path
  from_timestamp
  to_timestamp
  fps

EpisodeInfo
  episode_index
  length
  tasks
  data_ref
  videos[camera] -> EpisodeVideoRef
```

所有区间均采用左闭右开语义。`EpisodeInfo.length` 和标注边界始终是 episode 本地帧空间 `[0, length)`：

- v2.1 adapter 为独立 episode 文件生成从零开始、覆盖整个文件的引用。
- v3.0 adapter 从 episode metadata 解析数据分片、全局数据区间，以及每个 camera 的视频分片和起止时间。
- adapter 校验 episode 编号连续、数据区间长度等于 `length`、引用文件存在、摄像头集合完整，并验证时间区间可覆盖 episode 帧数。
- 底层共享分片发生变化时，所有引用该分片的 episode 都视为源数据已变化。

`RunManifest.dataset_version` 扩展为 `Literal["v2.1", "v3.0"]`。source/run fingerprint 加入版本、规范化切片元数据和被引用共享文件的身份信息，保证恢复 workspace 时不会误用旧结果。

## 读取、抽帧与人工复核

v2.1 读取和视频路径保持现状。v3.0 adapter 读取 `meta/tasks.parquet`、全部 `meta/episodes/**/*.parquet`，并使用官方 path template 和 metadata 中的 `chunk_index`、`file_index` 定位数据及视频分片。

数据读取按 `EpisodeDataRef` 截取对应行区间，并复核截取行中的 `episode_index`、`frame_index`、`index` 和 `timestamp`。模型采样和标注输出中的 frame index 仍从零开始。

视频抽帧 API 改为接收 `EpisodeVideoRef`。它把本地帧转换为共享 MP4 的目标时间位置，seek 后向前解码到目标帧，并只返回该 episode 的时间范围。`FrameSample.frame_index` 继续返回本地编号，防止版本细节泄漏到 prompt、refine 和 evaluation。

review 服务仍通过受限的 HTTP Range 路由提供已知 MP4 文件，同时在 episode payload 中返回 `from_timestamp` 和 `to_timestamp`。前端执行以下映射：

- 本地时间 `t` 对应媒体时间 `from_timestamp + t`。
- 展示帧号由本地时间和 fps 计算。
- seek、逐帧和多相机同步都使用 episode 本地时间。
- 播放达到 `to_timestamp` 时自动暂停，不能进入相邻 episode。

这样预览不需要重新编码共享视频，accepted-only 发布时才进行视频重建。

## 普通转换

输出版本始终等于 `RunManifest.dataset_version`，CLI 不增加 `--output-version`。

v2.1 普通转换保持现有行为，包括现有 `info.json` 扩展字段、`meta/lerobot_annotations.json` 和 `meta/task_info/task_0.json`。

v3.0 普通转换执行以下操作：

1. 原样复制官方 Parquet、MP4 和核心 metadata。
2. 不向 v3.0 `info.json` 注入 `subtask_template` 或 `high_level_instruction` 等自定义字段。
3. 把公共标注、每 episode 的本地边界、高层指令、增广来源及英文 action text 写入 `meta/lerobot_annotations.json` 和 `meta/task_info/task_0.json`。
4. 保持原始 `tasks.parquet` 和数据行中的官方 task 不变；subtask 增广不会重写原始数据集任务。
5. 对成品执行版本感知的独立 release validation 后再原子发布。

`ConversionReport` 增加 `dataset_version`。现有 v2.1 报告继续使用 `reference-v2.1`，v3.0 使用 `reference-v3.0`；公共 annotation path 仍是 `meta/lerobot_annotations.json`。

## v3.0 accepted-only 重建

共享分片中可能同时包含 accepted 和未接受的 episode，因此 v3.0 的 `convert --accepted-only` 必须重建一个独立数据集，不能只复制或删除文件。

### 选择与编号

- 按源 `episode_index` 升序选择 accepted records。
- 输出 episode 重新编号为连续的 `0..N-1`。
- 输出全局 `index` 从零连续累加；每 episode 的 `frame_index` 从零开始。
- annotation、task info、episode metadata 和数据行统一使用输出编号。
- 若没有 accepted episode，保持现有 fail-closed 行为。

### Parquet 和 task

- 按 `EpisodeDataRef` 从源共享 Parquet 截取完整 episode 行。
- 保留用户 observation、action 和其他 feature 列及其 Arrow 类型和 schema metadata。
- 重写 `episode_index`、`frame_index`、全局 `index`。
- 收集被保留行实际引用的官方 task，按源 task 顺序压缩为连续 task index，同时重写数据行中的 `task_index` 并生成新的 `meta/tasks.parquet`。
- episode 不跨输出数据文件。writer 按 `data_files_size_in_mb` 的近似序列化大小装箱；单 episode 超限时允许独占文件。
- `chunks_size` 在 v3.0 中按“每 chunk 最大文件数”解释，输出文件编号据此映射到 chunk/file index。

### 视频

- 对每个 camera 解码源 `EpisodeVideoRef` 的半开区间，只保留与 `length` 对应的帧。
- 视频保持源 feature 声明的 fps、分辨率和颜色通道；输出使用项目支持且官方 loader 可读取的 MP4 编码。
- 按 episode 顺序把视频段连接到输出分片，episode 不跨视频文件。
- writer 以 `video_files_size_in_mb` 为近似上限；单 episode 超限时允许独占文件。
- 每写入一个 episode 就依据实际输出帧位置记录新的 `from_timestamp` 和 `to_timestamp`，误差不得超过一帧。
- 普通 v3.0 转换保持源视频字节不变；accepted-only 明确属于重新编码，报告和日志不得描述为无损复制。

### 元数据和统计

重建以下官方 metadata：

- `meta/episodes/**/*.parquet`：输出 episode index、tasks、length、数据 chunk/file、dataset 区间、各视频 chunk/file 和时间区间，以及扁平化 episode stats。
- `meta/info.json`：官方 path template、splits、版本、feature 描述、总 episode/帧/视频/文件/chunk 计数和分片大小字段。
- `meta/stats.json`：从输出 Parquet 和完整解码的视频重新计算 aggregate stats。
- `meta/tasks.parquet`：只包含输出实际使用的官方 task。
- Robo-annotate 公共标注文件：边界继续使用重编号后 episode 的本地帧空间，增广后的每个 episode subtask action text 一并写入。

统计不能从被选择 episode 的旧 aggregate stats 做数学删减；必须从输出 payload 重算，以避免共享分片、浮点累计和视频采样差异。

## 发布一致性与原子性

现有发布安全协议对两个版本保持一致：

1. 获取目标输出锁并拒绝已存在的输出。
2. 重新校验 workspace、manifest、accepted records 和 source fingerprint。
3. 记录源目录摘要，在唯一 staging 目录中完成复制或重建。
4. 对 staging 执行独立 release validation。
5. 再次检查源目录摘要未变化。
6. 使用不可覆盖的原子 rename 发布并 fsync 父目录。
7. 失败时只清理本次持有的 staging 目录，绝不修改 source、workspace 或其他临时目录。

## 独立发布校验

release validator 首先识别版本，然后分派至 v2.1 或 v3.0 validator。v2.1 现有规则保持不变。v3.0 validator 不读取 workspace，独立检查：

- `info.json` 版本、模板、features、split 和所有计数。
- `tasks.parquet` schema、连续 task index 和引用完整性。
- 全部 episode metadata 行的连续编号、长度、data/video 引用和区间。
- 数据分片 schema、全局 index 连续性、episode 分区、frame index、timestamp 和 task index。
- 每个 camera 的视频文件 fps、分辨率、可解码帧数，以及 episode 时间范围的覆盖、顺序和不重叠。
- aggregate stats、扁平化 episode stats 与实际输出 payload 的一致性。
- `meta/lerobot_annotations.json`、`meta/task_info/task_0.json` 与输出 episode、本地长度和 annotation schema 的一致性。
- payload inventory、路径 containment、符号链接和摘要要求。

编码时间基允许最多一帧误差，但数据行数、annotation 边界和解码后的 episode 帧数必须严格等于 episode `length`。

## 错误处理与安全边界

- 未知或缺失版本立即报错，不回退到 v2.1。
- v2.1 template 只接受 `episode_chunk`、`episode_index`、`video_key`；v3.0 template 只接受 `chunk_index`、`file_index`、`video_key`。
- 渲染后的相对路径必须位于数据集根目录的预期 `data`、`videos` 或 `meta` 子树内。
- 数据集根目录、被引用 payload、workspace 和输出路径中的关键对象不得是符号链接。
- 缺失引用、重复编号、非连续 index、负数或反向区间、视频越界、摄像头缺失和统计不一致都在模型调用或发布前拒绝。
- 对用户可见的异常消息使用英文，并包含可操作的 version、episode、camera、path 或 field 上下文，但不暴露 API key 和未筛选模型响应。

## 依赖策略

默认安装不增加 `lerobot` 依赖。v3.0 运行路径只依赖项目已有的 `pyarrow`、`av` 和验证模型。

增加可选 extra：

```toml
v3-validation = ["lerobot[dataset]==0.6.1"]
```

它只用于兼容性 oracle 测试。固定版本避免官方 loader 行为漂移导致默认测试不稳定；升级版本需要单独验证并更新固定值。

## 测试策略

### 合成 v3.0 fixture

测试工厂生成可实际解码的微型 v3.0 数据集：

- 3 个 episode、2 个 camera。
- 至少两个 episode 共用同一数据 Parquet 和同一 camera MP4。
- 每个 episode 使用不同的数据和视频起始偏移。
- 包含多个官方 task、数值 observation/action 和真实小尺寸 CFR 视频。
- episode metadata、tasks、stats、path template 和计数严格按官方 v3.0 schema 写入。

共享分片是 fixture 的硬要求，避免实现退化为“文件名不同的 v2.1”。

### 自动测试

- adapter 单元测试：版本识别、metadata 解析、路径渲染、切片校验和 fingerprint。
- 抽帧测试：共享视频中每个 episode 的首、中、末帧映射到正确本地 frame index。
- review 测试：offset-aware seek、逐帧、多相机同步和 episode 结束暂停。
- pipeline 回归：同一套标注、增广、evaluation 逻辑可消费 v2.1 与 v3.0 `DatasetIndex`。
- 普通 v3.0 转换：官方 payload 字节保持不变，公共标注正确附加。
- accepted-only v3.0 转换：只接受第 0 和第 2 个 episode，删除共享分片中间的第 1 个 episode，再验证所有编号、区间、task、视频、统计和标注。
- 畸形输入测试：未知版本、越界区间、缺失 shard、重叠视频、错误计数、路径穿越和符号链接。
- v2.1 完整回归：默认 CLI、manifest、抽帧、review、普通转换和 accepted-only 输出保持兼容。

### 官方 loader oracle

安装 `v3-validation` extra 后，用官方 `LeRobotDataset` 加载：

1. 合成 v3.0 源数据。
2. 普通 v3.0 转换结果。
3. accepted-only v3.0 转换结果。

每个对象都读取 metadata、长度以及首、中、末样本，确认视频 feature 可解码且 episode/frame/task 索引一致。该测试可作为独立 CI job，不影响默认运行依赖。

### 真实样本后续验收

取得真实 v3.0 数据后执行 inspect、少量 episode 标注、review、普通 convert、accepted-only convert、独立 validator 和官方 loader oracle。真实样本验收是兼容性加固项，不阻塞基于官方 schema 和 oracle 的首版实现完成。

## CLI 与文档

- `inspect` 输出实际检测到的 dataset version、episode 数、camera 和共享 shard 概要。
- `run`、`review`、`convert` 和 `validate` 不增加版本参数，版本来自输入数据集或 workspace manifest。
- README 和操作文档以中文说明项目，并先展示 v2.1 示例，再说明 v3.0 自动兼容。
- 配置示例、CLI 参数、代码块、字段名、默认标注内容及生成 action text 使用英文。
- 明确记录 v3.0 accepted-only 会重新编码视频，以及可选官方 oracle 的安装和运行命令。

## 验收标准

完成实现后必须同时满足：

- 现有 v2.1 测试全部通过，未提供版本参数时仍按输入自动识别且文档默认展示 v2.1。
- 合成共享分片 v3.0 可完成 inspect、抽帧、标注、增广、review、普通转换和 accepted-only 转换。
- v3.0 普通转换保留官方 payload；accepted-only 输出只含被接受 episode，且所有数据、视频、task、metadata、统计和标注引用一致。
- 两类 v3.0 输出通过独立 release validator。
- 安装可选 extra 后，合成源和两类输出均能由固定版本官方 `LeRobotDataset` 加载并读取样本。
- 任一结构、路径、范围、fingerprint 或发布一致性错误都会 fail closed，且不会修改源数据或留下已发布的半成品。

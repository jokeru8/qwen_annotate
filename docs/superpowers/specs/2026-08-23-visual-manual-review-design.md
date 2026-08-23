# 可视化人工复核与接管设计

## 目标

在 `qwen_annotate` 内提供一个只绑定单个 workspace 的本地 Web 界面，让标注员同步查看 episode 的多路视频、当前 subtask 和边界帧，复核或修正自动标注；`pending`、`failed` 只能经过显式“人工接管”确认后完整手工标注并转为 `accepted`。

## 方案

保留现有静态 `qwen-annotate review WORK_DIR` 和离线 decision 导入兼容性，新增 `qwen-annotate review WORK_DIR --serve`。服务采用 FastAPI/Uvicorn，但所有状态仍通过现有 `WorkspaceStore`、source/run fingerprint、硬约束和事务保存进入同一份 workspace，不引入第二套标注数据库，也不复用旧工具的 exporter。

浏览器不能提交任意文件路径。服务启动时固定唯一 `WORK_DIR`，默认仅监听 `127.0.0.1`。视频接口只能返回 manifest 中当前 episode 的已知 camera 文件，并支持 HTTP Range 以便浏览器拖动。

## 界面与交互

- 左栏列出全部 episode，可按 `pending`、`coarse_done`、`refine_done`、`accepted`、`needs_review`、`failed` 过滤。
- 中间以 2×2 等大网格显示四路相机。所有画面共享播放、暂停、拖动和逐帧状态；共享帧进度条独占整行并与视频网格左右边缘对齐。
- 时间线显示彩色 subtask 区段和可拖动边界。边界语义始终是“后一 subtask 的第一帧”，逐帧时同时展示帧号与时间。
- 右栏显示不可编辑的高层任务、subtask 模板、当前区段、模型候选、置信度、证据、复核原因和校验问题。
- `needs_review` 默认载入模型候选，允许人工调整后接受。
- `pending`、`failed` 默认只读。点击“人工接管”并在确认框中再次确认后，才进入完整手工标注模式。
- `accepted` 默认只读；“修正人工结果”同样要求显式确认。首版不在界面内发布数据集，发布继续使用 `convert` 和 `validate`。
- 草稿只保存在当前浏览器内存，不触碰 workspace；点击最终提交才写入。

## 写入契约与审计

提交包含 episode、source fingerprint、run fingerprint、保存前状态、保存前 `updated_at`、模式、start subtask、边界、接管确认和人工说明。后端在保存前重新读取 authoritative record、重新检查数据集和 source fingerprint，并拒绝状态或时间戳已变化的陈旧提交。

人工写入复用 `validate_annotation`。`pending`/`failed`/`accepted` 的人工写入必须带显式接管确认；`needs_review` 可直接复核接受。保存后状态为 `accepted`、`decision_source` 为 `human`，原模型 attempts 保留。审计追加到 `sampling_details.human_decisions`，记录原状态、失败类别、复核理由、原候选、最终结果、fingerprint、时间和说明；原 `failed` 字段在 accepted record 中清空，但值保存在审计中。

状态机新增受约束的人工迁移：`pending → accepted`、`failed → accepted`、`accepted → accepted`。它们只有在存在合法的最新人工审计时才允许。pipeline outbox 若已存在则追加 `human_takeover` 或 `human_corrected` 事件；没有历史 outbox 时不伪造模型事件。

## API

- `GET /api/session`：manifest、subtask、camera、fps 和状态汇总。
- `GET /api/episodes?status=...`：轻量 episode 列表。
- `GET /api/episodes/{index}`：record 的安全展示视图和可编辑候选。
- `GET /api/episodes/{index}/videos/{camera}`：受限视频 Range 响应。
- `POST /api/episodes/{index}/decision`：严格校验并事务写入人工结果。
- `GET /` 与 `/assets/*`：项目内静态 UI。

API 不返回 API key、内部任意路径或未筛选的 model payload。错误响应给出可操作但不泄露敏感信息的消息。

## 并发和故障

`WorkspaceStore` 锁保护单次写入，decision 的 expected status/updated_at 提供乐观并发控制。如果自动标注在页面打开后推进了同一 episode，提交返回冲突并要求刷新。视频或源文件指纹变化时 fail closed。事务回滚实现必须能在 `/mnt/data` 的 FUSE 文件系统上工作，不能依赖硬链接。

## 验证

- 单元测试覆盖 takeover 门禁、硬约束、陈旧提交、失败信息审计、accepted 修正及非法状态迁移。
- API 测试覆盖状态过滤、展示 payload 脱敏、路径穿越、Range/无效 Range 和冲突响应。
- 前端行为测试覆盖多相机同步、逐帧、边界编辑、当前 subtask、高风险按钮确认和提交 payload。
- CLI 测试覆盖 `--serve` 参数；现有静态 review 与 decision 导入测试保持通过。
- 在合成 LeRobot fixture 上做端到端人工接管，并运行完整 pytest。

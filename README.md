# SymboGraph

![SymboGraph](assets/diagraph-logo.svg)

[English](README.en.md)

SymboGraph 是本地文档知识库。它把资料解析成固定片段和四层上下文图，沿图寻找证据，再从实际原文证据包生成带引用的回答。

当前架构使用 Chunk Structure、Chunk Relations/RQ、Mid Concepts 和 Coarse Concepts。LLM 分开确定问题意图与执行策略，选择粗层、中层或片段层入口；无词面走纯向量，有词面时可按 LLM 权重融合 Dense、RQ 和 BM25 分层入口。取得原文包后进行确定性来源校验、一次生成和引用绑定。

当前运行主链已迁移到该架构。Dense、RQ、BM25 只负责图入口提名与融合，最终证据必须经过图路径、结构恢复和来源准入；实现契约见[技术白皮书](docs/technical-spec.md)，设计依据见[检索调研](docs/reference/retrieval-research.md)。

Web 产品入口为概览、导入、问答、图谱和设置；独立检索页已移除，问答仍使用上述图检索链取证。问答历史由 PostgreSQL 持久化并自动恢复，失败或取消也保留完整回合；回答支持 GFM、代码和 KaTeX 公式。

## 快速开始

需要 Docker Desktop 的 Linux 容器环境和 PowerShell。首次启动时创建两份互不重叠的配置，已有文件不要覆盖：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
if (-not (Test-Path settings.json)) { Copy-Item settings.example.json settings.json }
```

编辑根 `.env`，填写本地数据库账户及 chat、graph、embedding 三组模型连接；非秘密的检索、构建、预算和产品参数位于根 `settings.json`。对话/图模型支持 OpenAI-compatible 或 Anthropic Messages，向量接口使用 OpenAI-compatible；格式见[环境说明](infra/README.md)。正常路径保持两个 fallback 开关为 false。

资源管理器中双击 `start-app.bat` 可一键启动。启动成功或失败后窗口都会保留，确认结果并按任意键才关闭；服务进程不会随启动窗口关闭。命令行使用下面的 PowerShell 入口，执行结束后直接返回：

```powershell
.\start-app.ps1 -NoBrowser
```

启动器构建 API 镜像并启动后端 Compose，Worker/Beat 复用 API 镜像；Web 使用宿主 Node.js 与锁定的 `node_modules`。`-SkipBuild` 只适用于已有匹配 API 镜像。确需从自动化调用批处理入口时，可仅为该进程设置 `SYMBOGRAPH_NO_PAUSE=1`；日常命令行优先直接调用 `.ps1`。运行说明见 [infra](infra/README.md)。

| 入口 | 默认地址 |
|---|---|
| Web | http://127.0.0.1:3000 |
| API | http://127.0.0.1:8000/api |
| Readiness | http://127.0.0.1:8000/api/ready |
| API schema | http://127.0.0.1:8000/docs |

## 仓库地图

| 目录 | 内容 |
|---|---|
| [apps/api](apps/api/README.md) | FastAPI、数据模型、解析、图构建、检索、问答和运行配置 |
| [apps/web](apps/web/README.md) | Next.js 16.2.4 界面、状态流和引用展示 |
| [apps/worker](apps/worker/README.md) | Celery Worker、Beat 和文件 watcher；复用 API service |
| `packages/shared` | 前后端共享 TypeScript 契约 |
| [infra](infra/README.md) | Docker Compose、模型桥与运行环境 |
| [scripts](scripts/README.md) | 可重复的维护、诊断、验收和性能工具 |
| [docs](docs/technical-spec.md) | 技术白皮书、精确协议参考和开发说明 |
| `data` | 被忽略的本地验收输入；应用资料存储位置见下文 |
| `output` | 临时产物，整个目录忽略，核验后清理 |

## 数据和配置

PostgreSQL 保存生命周期与审计，Qdrant/Redis 是可恢复的派生或运行态存储。应用原文、快照和向量数据位于 Docker named volume；仓库 `data/` 不是默认应用数据卷。不要删除卷来“清缓存”。

根 `.env` 与根 `settings.json` 共同组成配置权威且键集合不相交：前者只保存秘密、连接、路径、端口和进程/服务启动参数，后者保存非秘密产品、检索、预算和构建参数。设置页/API 按字段归属写入对应文件，并继续按热加载、重建或容器重建三级生命周期生效。Profile 管提示词、文案和对话偏好，不能替代工程参数或证据门禁。

真实端点、部署模型名和凭据不提交。公共示例使用占位值；真实验收输入与结果也不进入源码 fixture。

## 开发和运维

[开发与测试](docs/development.md)列出依赖安装、测试分组和验证口径。[脚本说明](scripts/README.md)区分只读诊断、显式执行和破坏性维护；不要把 dry-run 结果当成已执行。

精确架构见[技术白皮书](docs/technical-spec.md)及其参考文档；修改、验证与清理流程见[开发与测试](docs/development.md)。本地验收记录保存在 Git 忽略的工作文件中，不进入公开仓库。

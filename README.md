[English](./README.en.md) | **中文**

<p align="center">
  <img src="./assets/diagraph-logo.svg" alt="SymboGraph logo" width="132" height="132">
</p>

<h1 align="center">SymboGraph</h1>

## 项目简介

SymboGraph 是一个本地通用智能知识库系统。系统采用 Four-Layer Context Graph RAG：固定 token chunk 提供稳定索引和引用地址，Chunk Structure Graph 保存原文结构和上下文恢复路径，Chunk Relation Graph 保存可复算的底层语义关系和 RQ chunk-pair evidence。RQ fuzzy membership 只负责路由和诊断；Mid Concept Graph 与 Coarse Concept Graph 分别由 RQ L3、L2 prefix packet 经过确定性 eligibility 规则压缩产生，并保持 `Mid≤chunks`、`Coarse≤Mid`。检索与问答只使用 context package，答案必须通过 citation verification。

## 目录

| 路径 | 职责 |
| --- | --- |
| `apps/api` | FastAPI 后端，负责导入、解析、固定 chunk、四层图谱、检索、问答、设置和维护入口。 |
| `apps/web` | Next.js 16.2.4 前端，负责知识库 UI、图谱、搜索轨迹、context package、QA 和设置。 |
| `apps/worker` | Celery worker 与文件 watcher，执行长任务并复用 API service 逻辑。 |
| `packages/shared` | 前后端共享类型和契约。 |
| `infra` | Docker Compose 默认运行环境。 |
| `scripts` | 重建、对账、诊断、检索评估、质量检查和 Docker smoke。 |
| `docs` | 技术白皮书、工程清单和验收资料。 |
| `output` | 诊断、benchmark、smoke、截图和验收报告输出目录，不提交。 |

## 产品定位

SymboGraph 面向本地资料库、课程资料、技术文档和研究资料的严格引用问答与图谱检索。系统强调可重建、可审计、可追溯：PostgreSQL 保存事实源，Qdrant 和 Redis 作为 active 派生或运行态存储；答案事实只来自 context package 和 raw chunk citation span。

适合：

- 本地私有知识库。
- 需要严格引用的 QA。
- 需要展示四层图谱路径、结构上下文和检索轨迹的资料探索。
- 需要可诊断导入、重建、缓存失效和策略边界的工程环境。

## 技术栈

| 范围 | 技术 |
| --- | --- |
| 后端 | Python 3.13, FastAPI, SQLAlchemy, Alembic, Pydantic |
| 存储 | PostgreSQL 16, Qdrant 1.17.1, Redis 7 |
| 异步任务 | Celery, Redis broker |
| 检索 | Dense embedding, dense-only chunk relation graph, RQ membership, staged layered traversal |
| 模型接口 | 可选 OpenAI-compatible / Anthropic Messages 的 chat、graph endpoint；embedding 固定 OpenAI-compatible |
| 前端 | Next.js 16.2.4, React 19.2.4, TypeScript, TanStack Query, Tailwind CSS, ECharts |
| 运维 | Docker Compose, Python maintenance scripts, pytest, Vitest, ESLint |

## 主链路

```text
source files
-> parser and layout extractor
-> fixed token chunks
-> chunk structure graph
-> contextual embedding and vector index
-> optional automatic TPE operating point selection on chunk-version increments
-> independent chunk relation graph
-> RQ prefix address tree and fuzzy membership
-> RQ L3 mid concept graph
-> RQ L2 coarse concept graph
-> active context graph state
-> conversation state and query intent
-> Layered P&E Agent typed strategy
-> layered graph retrieval
-> context package
-> grounded answer and citation verification
-> reward and policy update
```

## 环境配置

首次启动前从模板创建 `.env`：

```powershell
Copy-Item .env.example .env
```

至少配置 PostgreSQL、chat、graph 和 embedding endpoint。数据库密码必须是本地生成值，不能提交：

```env
POSTGRES_USER=symbograph
POSTGRES_PASSWORD=<local-random-password>
POSTGRES_DB=symbograph
DATABASE_URL=postgresql+psycopg://symbograph:<local-random-password>@localhost:5432/symbograph

CHAT_API_KEY=...
CHAT_API_PROTOCOL=openai
CHAT_BASE_URL=https://your-chat-endpoint/v1
CHAT_MODEL=your-chat-model

GRAPH_API_KEY=...
GRAPH_API_PROTOCOL=openai
GRAPH_BASE_URL=https://your-graph-endpoint/v1
GRAPH_MODEL=your-graph-model

EMBEDDING_API_KEY=...
EMBEDDING_API_PROTOCOL=openai
EMBEDDING_BASE_URL=https://your-embedding-endpoint/v1
EMBEDDING_MODEL=your-embedding-model
EMBEDDING_DIMENSIONS=1024
```

三个协议字段彼此独立。`CHAT_API_PROTOCOL` 与 `GRAPH_API_PROTOCOL` 可分别选择 `openai` 或 `anthropic`；选择 `anthropic` 时，对应 base 必须填写 provider 根地址或路径前缀，不能以 `/v1` 或 `/v1/messages` 结尾，客户端固定追加 `/v1/messages`。选择 `openai` 时固定追加 `/chat/completions`。`EMBEDDING_API_PROTOCOL` 当前只允许 `openai`，固定追加 `/embeddings`；Anthropic Messages 没有 embedding 契约，不能用于向量模型。向量协议属于 `rebuild_required`，只能通过 candidate、shadow build、evaluation 和 promotion 生效。

正常运行路径保持 fallback 关闭：

```env
ENABLE_MODEL_FALLBACK=false
ENABLE_DATABASE_FALLBACK=false
```

设置页保存的是完整 desired 配置。`hot_reloadable` 字段会发布到 active 运行时；`rebuild_required` 与 `service_recreate_required` 字段分别等待显式重建/promotion 或服务重建，不会因页面保存而静默改写 active 图谱或容器拓扑。真实 endpoint、模型名、API key、资料 manifest 和授权回执只保存在被 Git 忽略的本地配置或 `output/` 中，仓库示例只能使用占位符或 `.invalid` 合成 fixture。

## 快速启动

```powershell
.\start-app.ps1
```

启动器会用可变本地 tag 重建 API/Web 镜像；worker 复用 API 镜像。`.env`
中的 `API_IMAGE=name@sha256:...` 只表示锁定运行引用，不能作为 Docker build
输出 tag，`rebuild-images.ps1` 会显式覆盖它：

```powershell
# 只重建本地开发镜像
.\rebuild-images.ps1

# 已存在的 digest 只允许运行，不参与 build
.\start-app.ps1 -SkipBuild `
  -ApiImage "course-kg-api@sha256:<digest>" `
  -WebImage "course-kg-web@sha256:<digest>"
```

访问：

```text
Web: http://127.0.0.1:3000
API: http://127.0.0.1:8000/api
Readiness: http://127.0.0.1:8000/api/ready
```

默认容器：

```text
course-kg-api
course-kg-worker
course-kg-web
course-kg-postgres
course-kg-redis
course-kg-qdrant
```

## 参数列表

| 分类 | 参数 |
| --- | --- |
| 应用 | `APP_NAME`, `APP_ENV`, `APP_PORT`, `API_IMAGE`, `WEB_IMAGE` |
| 端口 | `API_HOST_PORT`, `WEB_HOST_PORT` |
| 任务队列 | `INGESTION_TASK_QUEUE` |
| 基础设施 | `DATABASE_URL`, `QDRANT_URL`, `QDRANT_COLLECTION`, `REDIS_URL`, `CORS_ORIGINS`, `API_KEYS` |
| 数据目录 | `KNOWLEDGE_BASE_NAME`, `DATA_ROOT`, `STORAGE_ROOT`, `INGESTION_ROOT` |
| 模型桥接 | `MODEL_BRIDGE_ENABLED`, `MODEL_BRIDGE_PORT`, `MODEL_BRIDGE_ADMIN_TOKEN` |
| 对话模型 | `CHAT_API_KEY`, `CHAT_API_PROTOCOL`, `CHAT_BASE_URL`, `CHAT_RESOLVE_IP`, `CHAT_MODEL` |
| 图构建模型 | `GRAPH_API_KEY`, `GRAPH_API_PROTOCOL`, `GRAPH_BASE_URL`, `GRAPH_RESOLVE_IP`, `GRAPH_MODEL` |
| 向量模型 | `EMBEDDING_API_KEY`, `EMBEDDING_API_PROTOCOL` (`openai`), `EMBEDDING_BASE_URL`, `EMBEDDING_RESOLVE_IP`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `EMBEDDING_BATCH_SIZE` |
| 并发与资源 | `WORKER_CONCURRENCY`, `WORKER_MAX_TASKS_PER_CHILD`, `MODEL_REQUEST_CONCURRENCY`, `MODEL_REQUEST_TIMEOUT_SECONDS`, `INGESTION_MEMORY_SOFT_LIMIT_RATIO`, `INGESTION_MEMORY_HARD_LIMIT_RATIO`, `INGESTION_MEMORY_CRITICAL_LIMIT_RATIO` |
| 片段与上下文 | `FIXED_CHUNK_SIZE_TOKENS`, `FIXED_CHUNK_OVERLAP_TOKENS`, `CONTEXT_PACKAGE_TOKEN_BUDGET` |
| 中概念抽取 | `MID_CONCEPT_EXTRACTION_MAX_MODEL_BATCHES`, `MID_CONCEPT_EXTRACTION_MAX_CANDIDATES_PER_BATCH`, `MID_CONCEPT_EXTRACTION_MAX_TOKENS_PER_BATCH`, `MID_CONCEPT_CANDIDATE_KEEP_THRESHOLD` |
| 残差量化聚类 | `RQ_KMEANS_LEVELS`, `RQ_KMEANS_MAX_K`, `RQ_RESIDUAL_TAU` |
| 稠密关系运行点 | `DENSE_KNN_K_MIN`, `DENSE_KNN_K_MAX`, `DENSE_REVERSE_B_MIN_BASE`, `DENSE_REVERSE_B_MAX_BASE`, `DENSE_REVERSE_B_MIN_DOC`, `DENSE_REVERSE_B_MAX_DOC`, `DENSE_REVERSE_B_MIN_LANG`, `DENSE_REVERSE_B_MAX_LANG`, `DENSE_MIN_COSINE`, `DENSE_STRONG_COSINE`, `CROSS_DOC_OUT_QUOTA_MIN`, `CROSS_DOC_OUT_QUOTA_MAX`, `CROSS_DOC_MIN_COSINE`, `CROSS_LANGUAGE_OUT_QUOTA_MIN`, `CROSS_LANGUAGE_OUT_QUOTA_MAX`, `CROSS_LANGUAGE_MIN_COSINE` |
| 自动 TPE 运行点 | `ENABLE_AUTO_TPE`, `TPE_TRIAL_BUDGET`, `TPE_STARTUP_RANDOM_TRIALS`, `TPE_GOOD_QUANTILE_GAMMA`, `TPE_PROBE_QUERY_BUDGET`, `TPE_TRIAL_TIMEOUT_SECONDS`, `TPE_CANDIDATE_POOL_SIZE` |
| 运行点硬门限 | `OPERATING_POINT_HARD_GATE_MAX_EDGE_DENSITY`, `OPERATING_POINT_HARD_GATE_MAX_ISOLATED_RATIO`, `OPERATING_POINT_HARD_GATE_MAX_HUBNESS_RATIO`, `OPERATING_POINT_HARD_GATE_MIN_STRUCTURE_RECOVERY_RATE`, `OPERATING_POINT_HARD_GATE_MAX_CANDIDATE_LATENCY_P95_MS` |
| 智能体遍历预算 | `AGENT_COARSE_INITIAL_BUDGET`, `AGENT_COARSE_TOP_K`, `AGENT_MID_PER_COARSE_BUDGET`, `AGENT_COARSE_DRILLDOWN_MID_INITIAL_BUDGET`, `AGENT_MID_INITIAL_BUDGET`, `AGENT_MID_TOP_K`, `AGENT_CHUNK_PER_MID_BUDGET`, `AGENT_CHUNK_INITIAL_BUDGET`, `AGENT_CHUNK_TOP_K`, `CANDIDATE_POOL_DEDUPE_BUDGET` |
| 智能体路径与上下文边界 | `AGENT_MAX_DEPTH_PER_LAYER`, `AGENT_MAX_LABELS_PER_NODE`, `AGENT_MAX_EDGE_REUSE`, `AGENT_MAX_CYCLE_REWARD_PER_PATH`, `AGENT_CYCLE_REWARD_DISTANCE_THRESHOLD`, `AGENT_PATH_DISTANCE_GREEN_THRESHOLD`, `AGENT_PATH_DISTANCE_GRAY_THRESHOLD`, `AGENT_PATH_DISTANCE_HARD_THRESHOLD`, `AGENT_STRUCTURE_RESTORE_PER_CHUNK_BUDGET`, `CONTEXT_PATH_SUMMARY_BUDGET` |
| 智能体规划与验证边界 | `AGENT_PLANNING_ROUND_BUDGET`, `AGENT_MAX_TYPED_ACTIONS_PER_ROUND`, `AGENT_REPAIR_ROUND_BUDGET`, `AGENT_VERIFICATION_BUDGET` |
| 智能体兼容别名 | `AGENT_COARSE_TOTAL_BUDGET`, `AGENT_STRUCTURE_RESTORE_BUDGET` |
| 回退开关 | `ENABLE_MODEL_FALLBACK`, `ENABLE_DATABASE_FALLBACK` |

## 验证

后端：

```powershell
cd apps/api
python -m pytest tests
```

前端：

```powershell
npm run typecheck --workspace web
npm run lint --workspace web
npm run test --workspace web
```

Docker smoke：

```powershell
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api --execute
```

第一条命令只执行 GET 预检并打印精确 KB、query、`POST /search`、
`POST /qa` 目标与影响；第二条命令才允许发送会写审计状态的 POST。

提交或推送前还应执行：

```powershell
git status --short
git diff --check
git ls-files -o --exclude-standard
git ls-files -ci --exclude-standard
```

确认 `.env`、本地数据库、资料文件、`output/`、浏览器 trace、证书和私钥均未被跟踪；检查待推送的所有 commit，而不只是工作区最新 diff。仓库内测试凭据必须使用明显的 `unit-test-*` 或 `.invalid` fixture，禁止复制真实 endpoint、模型桥 token、个人文件名或私有资料 hash。

## 运维测试

常用入口：

```powershell
python scripts/diagnose_context_graph.py
python scripts/evaluate_layered_retrieval.py
python scripts/evaluate_layered_retrieval.py --query "<query>" --execute
python scripts/check_context_package_quality.py
python scripts/evaluate_agent_trace.py
python scripts/check_technical_spec_compliance.py --knowledge-base-name 贝叶斯
python scripts/reconcile_vector_records.py
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api --execute
```

`evaluate_layered_retrieval.py` 默认只重放 PostgreSQL 中已有 trace；新检索必须同时提供显式 query 和 `--execute`。写数据脚本默认提供 dry-run 或要求显式 `--execute`。生成的诊断、benchmark、smoke 输出和验收报告写入 `output/`。

## 文档

- [docs/technical-spec.md](./docs/technical-spec.md)：Four-Layer Context Graph RAG 技术白皮书。
- [docs/todo.md](./docs/todo.md)：工程落地清单。
- [apps/api/README.md](./apps/api/README.md)：API 后端说明。
- [apps/web/README.md](./apps/web/README.md)：Web 前端说明。
- [apps/worker/README.md](./apps/worker/README.md)：Worker 说明。
- [scripts/README.md](./scripts/README.md)：运维脚本说明。
- [infra/README.md](./infra/README.md)：Docker Compose 环境说明。

## 边界

- `chunks` 是索引、引用、检索、问答和图谱关联的主单位。
- 结构图只保存原文地图和上下文恢复路径，不进入 chunk relation edge 创建、保留或加权。
- Chunk relation graph 只表达内容语义关系和允许的 RQ chunk-pair evidence。
- RQ fuzzy membership 只提供路由与投影权重；Mid/Coarse concept eligibility 由确定性规则控制并产生语义压缩，上层边必须由底层 chunk relation edge support 投影。
- QA 只使用 context package，不直接使用裸 search results。
- Citation 必须指向 raw chunk span。
- Qdrant 和 Redis 均为 active 派生或运行态存储，必须能从 PostgreSQL 重建或刷新。
- Profile 只影响 prompt、UI 和对话偏好；工程参数进入 `.env` 或 runtime settings。
- 涉及 PostgreSQL、Qdrant、Redis、模型接口和无 fallback 的集成路径在 Docker 栈内验证。
- 不提交 `.env`、本地数据库、`node_modules`、`.next`、`data/`、`output/`。

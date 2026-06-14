[English](./README.en.md) | **中文**

<p align="center">
  <img src="./assets/diagraph-logo.svg" alt="SymboGraph logo" width="132" height="132">
</p>

<h1 align="center">SymboGraph</h1>

## 项目简介

SymboGraph 是一个本地通用智能知识库系统。系统采用 Four-Layer Context Graph RAG：固定 token chunk 提供稳定索引和引用地址，Chunk Structure Graph 保存原文结构，Chunk Relation Graph 保存可复算的底层关系网络，Mid/Coarse Concept Graph 提供可追溯的概念导航，检索与问答通过 context package 和 citation verification 保持答案接地。

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

SymboGraph 面向本地资料库、课程资料、技术文档和研究资料的严格引用问答与图谱检索。系统强调可重建、可审计、可追溯：PostgreSQL 保存事实源，Qdrant、BM25 和 Redis 作为派生状态；答案事实只来自 context package 和 raw chunk citation span。

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
| 检索 | Dense embedding, BM25, chunk relation graph, RQ-KMeans, layered retrieval |
| 模型接口 | OpenAI-compatible chat 和 embedding endpoint |
| 精排 | Cross-Encoder reranker，可选 |
| 前端 | Next.js 16.2.4, React 19.2.4, TypeScript, TanStack Query, Tailwind CSS, ECharts |
| 运维 | Docker Compose, Python maintenance scripts, pytest, Vitest, ESLint |

## 主链路

```text
source files
-> parser and layout extractor
-> fixed token chunks
-> chunk structure graph
-> contextual embedding and BM25
-> chunk relation graph and fine clusters
-> mid concept graph
-> coarse concept graph
-> conversation state and query intent
-> Layered P&E Agent typed strategy
-> layered graph retrieval
-> context package
-> grounded answer and citation verification
-> reward and policy update
```

## 环境配置（Cross-Encoder 可选）

首次启动前从模板创建 `.env`：

```powershell
Copy-Item .env.example .env
```

至少配置 chat 和 embedding endpoint：

```env
OPENAI_API_KEY=...
CHAT_BASE_URL=https://your-chat-endpoint/v1
CHAT_MODEL=your-chat-model

EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=https://your-embedding-endpoint/v1
EMBEDDING_MODEL=your-embedding-model
EMBEDDING_DIMENSIONS=1024
```

正常运行路径保持 fallback 关闭：

```env
ENABLE_MODEL_FALLBACK=false
ENABLE_DATABASE_FALLBACK=false
```

Cross-Encoder 精排默认关闭。需要启用时：

```env
RERANKER_ENABLED=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_MAX_LENGTH=512
RERANKER_DEVICE=cpu
HF_HUB_OFFLINE=1
```

Docker 镜像安装了 `apps/api` 的 `[rerank]` extra。模型缓存挂载到 `data/models/huggingface`；如果希望构建镜像时预热模型，可传入 `PRELOAD_RERANK_MODEL=true` 构建参数。

## 快速启动

```powershell
docker compose -f infra/docker-compose.yml up -d --build
```

访问：

```text
Web: http://127.0.0.1:3000
API: http://127.0.0.1:8000/api
Health: http://127.0.0.1:8000/api/health
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
| 模型桥接 | `MODEL_BRIDGE_ENABLED`, `MODEL_BRIDGE_PORT` |
| Chat | `OPENAI_API_KEY`, `CHAT_BASE_URL`, `CHAT_RESOLVE_IP`, `CHAT_MODEL` |
| Embedding | `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_RESOLVE_IP`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `EMBEDDING_BATCH_SIZE` |
| 并发与资源 | `WORKER_CONCURRENCY`, `WORKER_MAX_TASKS_PER_CHILD`, `MODEL_REQUEST_CONCURRENCY`, `MODEL_REQUEST_TIMEOUT_SECONDS`, `INGESTION_MEMORY_SOFT_LIMIT_RATIO`, `INGESTION_MEMORY_HARD_LIMIT_RATIO`, `INGESTION_MEMORY_CRITICAL_LIMIT_RATIO` |
| Chunk 与上下文 | `FIXED_CHUNK_SIZE_TOKENS`, `FIXED_CHUNK_OVERLAP_TOKENS`, `CONTEXT_PACKAGE_TOKEN_BUDGET` |
| Mid concept | `MID_CONCEPT_EXTRACTION_MAX_MODEL_BATCHES`, `MID_CONCEPT_EXTRACTION_MAX_CANDIDATES_PER_BATCH`, `MID_CONCEPT_EXTRACTION_MAX_TOKENS_PER_BATCH`, `MID_CONCEPT_CANDIDATE_KEEP_THRESHOLD` |
| RQ-KMeans | `RQ_KMEANS_LEVELS`, `RQ_KMEANS_MAX_K`, `RQ_RESIDUAL_TAU` |
| Agent envelope | `AGENT_COARSE_ACTIVATION_BUDGET`, `AGENT_COARSE_JUMP_BUDGET`, `AGENT_MID_ACTIVATION_BUDGET`, `AGENT_MID_EXPANSION_RADIUS_CAP`, `AGENT_FINE_CLUSTER_BUDGET`, `AGENT_CHUNK_CANDIDATE_BUDGET`, `AGENT_STRUCTURE_RESTORE_BUDGET`, `AGENT_PLANNING_ROUND_BUDGET`, `AGENT_MAX_TYPED_ACTIONS_PER_ROUND`, `AGENT_REPAIR_ROUND_BUDGET`, `AGENT_VERIFICATION_BUDGET` |
| 精排 | `RERANKER_ENABLED`, `RERANKER_MODEL`, `RERANKER_MAX_LENGTH`, `RERANKER_DEVICE`, `HF_HUB_OFFLINE` |
| Fallback | `ENABLE_MODEL_FALLBACK`, `ENABLE_DATABASE_FALLBACK` |

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
```

## 运维测试

常用入口：

```powershell
python scripts/diagnose_context_graph.py
python scripts/evaluate_layered_retrieval.py
python scripts/check_context_package_quality.py
python scripts/reconcile_vector_records.py
python scripts/reconcile_bm25_records.py
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api --worker-container course-kg-worker
```

写数据脚本默认提供 dry-run 或要求显式 `--execute`。生成的诊断、benchmark、smoke 输出和验收报告写入 `output/`。

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
- 结构图只保存原文地图，不改变固定 chunk 边界。
- Chunk relation graph 只表达可重建关系。
- Mid/Coarse concepts 必须有 support chunks、support spans、fine clusters 或 bridge chunks 支撑。
- QA 只使用 context package，不直接使用裸 search results。
- Citation 必须指向 raw chunk span。
- Qdrant、BM25 和 Redis 均为派生状态，必须能从 PostgreSQL 重建。
- Profile 只影响 prompt、UI 和对话偏好；工程参数进入 `.env` 或 runtime settings。
- 涉及 PostgreSQL、Qdrant、Redis、模型接口和无 fallback 的集成路径在 Docker 栈内验证。
- 不提交 `.env`、本地数据库、模型缓存、`node_modules`、`.next`、`data/`、`output/`。

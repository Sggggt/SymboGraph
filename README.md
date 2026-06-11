[English](./README.en.md) | **中文**

<p align="center">
  <img src="./assets/diagraph-logo.svg" alt="SymboGraph logo" width="132" height="132">
</p>

<h1 align="center">SymboGraph</h1>

SymboGraph 是一个本地通用智能知识库系统。当前 active 链路以可追溯证据为事实源：文件先解析为 `EvidenceAtom`，再形成只表达观测关系的 `EvidenceGraph`，随后产生 graph-grounded chunk candidates、`QualityDecision`、`ActiveChunk`、检索轨迹、引用核验、问答会话与策略奖励。

> **迁移说明**：产品代码与前端上下文已迁移到 `KnowledgeBase` / `partition` / evidence-first 语义。默认 PostgreSQL 数据库名已迁移为 `symbograph`。Docker Compose 服务名、容器名、镜像名、队列名仍保留 `course-kg-*` 等历史基础设施命名；这些名称会在独立基础设施变更中处理，不代表默认产品语义。

## 目录

- [快速概览](#快速概览)
- [技术栈](#技术栈)
- [核心能力](#核心能力)
- [主链路](#主链路)
- [运行方式](#运行方式)
- [环境参数](#环境参数)
- [仓库结构](#仓库结构)
- [测试与验收](#测试与验收)
- [使用边界](#使用边界)

## 快速概览

| 维度 | 当前设计 |
| --- | --- |
| 系统定位 | 本地通用智能知识库，服务于证据检索、grounded QA、引用核验和策略闭环 |
| 事实源 | PostgreSQL 中的 document versions、source spans、evidence atoms、active chunks、retrieval traces、answer sessions |
| 图基座 | Evidence graph，只保存相邻、包含、布局连续、引用依赖、语义相似、模态连接、话题转折等观测边 |
| 切块 | Evidence atoms 先进入图基座，再由图特征、社区边界、质量门禁与策略状态共同决定 active chunks |
| 检索问答 | Dense/lexical evidence recall、community/graph expansion、rerank、context assembly、citation verification |
| 策略 | `policy_states` 与 contextual bandit 管理 operating point；HPO 仅用于 experiment/offline replay 对照 |
| 默认运行 | Docker Compose 启动 API、worker、web、PostgreSQL、Redis、Qdrant |

## 技术栈

| 层级 | 技术 | 职责 |
| --- | --- | --- |
| API | FastAPI、SQLAlchemy、Alembic、Pydantic | 导入编排、证据图、切块质量、检索问答、运行时设置 |
| Worker | Celery、Redis broker | 文件解析、长任务、图构建、向量写入、可恢复批次 |
| Web | Next.js 16.2.4、React 19、TypeScript、TanStack Query | 本地知识库 UI、导入、证据图、检索、问答、设置 |
| Metadata | PostgreSQL | 生命周期状态、证据事实、策略状态、trace 与引用核验 |
| Vector | Qdrant | active chunk 向量与派生检索索引 |
| Runtime | Redis | 缓存、运行时设置广播、worker 协调 |
| Models | OpenAI 兼容 chat / embedding endpoint | grounded answer generation、测量型 prompt、embedding |

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 证据级导入 | 保守解析 heading、paragraph、list item、table block、code block、formula、caption、page block 等 atoms |
| 可审计图谱 | `EvidenceEdge` 只表达观测关系，不把 LLM durable ontology 当作事实源 |
| 图基座切块 | `ChunkCandidate` 保存 atom ids、span union、graph features、cost estimate 和 generator version |
| 质量决策 | `QualityDecision` 同时承担 gate、reward、feedback，并记录硬约束与诊断 |
| Active chunks | 每个 active chunk 保存 source span union、graph state hash、quality decision、policy state 和 community ids |
| 检索轨迹 | `RetrievalTrace` 记录候选、扩展、重排、缓存和风险审计信息 |
| 引用核验 | 答案引用必须回到 `active_chunk_id`、`evidence_atom_id` 或 source span |
| 策略闭环 | `RewardEvent` 回溯检索、问答、chunk 和 policy state，用于 contextual bandit 更新 |
| 热加载设置 | `.env` 更新后通过 Redis runtime settings version 广播，API 与 worker 在任务边界刷新 |

## 主链路

```text
文件解析
-> evidence atoms
-> evidence graph
-> graph-grounded chunk candidates
-> quality gate
-> active chunks
-> retrieval / QA / citation verification
-> reward events
-> contextual bandit policy update
```

```mermaid
flowchart LR
    F["文件解析"] --> A["EvidenceAtom"]
    A --> G["EvidenceGraph"]
    G --> C["ChunkCandidate"]
    C --> Q["QualityDecision"]
    Q --> X["ActiveChunk"]
    X --> R["RetrievalTrace"]
    R --> S["AnswerSession / CitationVerification"]
    S --> E["RewardEvent"]
    E --> P["PolicyState"]
    P --> C
```

算法细节、约束公式、策略 operating point 和质量决策字段见 [docs/technical-spec.md](./docs/technical-spec.md)。README 只保留工程入口和运行边界。

## 运行方式

默认使用 Docker Compose：

```powershell
docker compose -f infra/docker-compose.yml up --build
```

默认服务：

| 服务 | 默认地址 | 说明 |
| --- | --- | --- |
| Web | `http://127.0.0.1:3000` | Next.js UI |
| API | `http://127.0.0.1:8000/api` | FastAPI |
| PostgreSQL | `127.0.0.1:5432` | 元数据事实源 |
| Redis | `127.0.0.1:6379` | 缓存、broker、广播 |
| Qdrant | `http://127.0.0.1:6333` | active chunk 向量 |

Compose 服务名目前仍是 `course-kg-api`、`course-kg-worker`、`course-kg-web`、`course-kg-postgres`、`course-kg-redis`、`course-kg-qdrant`。不要把这些历史基础设施名当作产品语义。

## 环境参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `APP_NAME` | `KnowledgeBase Knowledge Base API` | API 应用显示名 |
| `APP_ENV` | `development` | 运行环境标识 |
| `APP_PORT` | `8000` | API 进程监听端口 |
| `API_IMAGE` | `course-kg-api:local` | API/worker Docker 镜像名；历史基础设施名，不代表产品语义 |
| `WEB_IMAGE` | `course-kg-web:local` | Web Docker 镜像名；历史基础设施名，不代表产品语义 |
| `INGESTION_TASK_QUEUE` | `course-kg-main-ingestion` | Celery 导入队列名；历史基础设施名，不代表产品语义 |
| `API_HOST_PORT` | `8000` | API 映射到宿主机的端口 |
| `WEB_HOST_PORT` | `3000` | Web 映射到宿主机的端口 |
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/symbograph` | PostgreSQL 连接串；默认数据库名为 `symbograph` |
| `ENABLE_DATABASE_FALLBACK` | `false` | active path 禁用数据库 fallback |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant 地址 |
| `QDRANT_COLLECTION` | `knowledge_chunks` | active chunk 向量集合名 |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis 地址 |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | API 允许的前端来源 |
| `API_KEYS` | 空 | 可选 API key 列表，逗号分隔 |
| `KNOWLEDGE_BASE_NAME` | `Sample KnowledgeBase` | 默认知识库名称 |
| `DATA_ROOT` | `./data` | 本地数据根目录 |
| `STORAGE_ROOT` | `./data/Sample KnowledgeBase/storage`（示例中注释） | 可选源文件存储目录覆盖值 |
| `INGESTION_ROOT` | `./data/Sample KnowledgeBase/ingestion`（示例中注释） | 可选导入暂存目录覆盖值 |
| `MODEL_BRIDGE_ENABLED` | `true` | 是否启用本地模型桥 |
| `MODEL_BRIDGE_PORT` | `8765` | 本地模型桥端口 |
| `OPENAI_API_KEY` | 空 | OpenAI 兼容接口密钥 |
| `CHAT_BASE_URL` | 空 | OpenAI 兼容 chat endpoint |
| `CHAT_RESOLVE_IP` | 空 | 可选 chat endpoint DNS 解析覆盖 IP |
| `CHAT_MODEL` | `qwen-plus` | chat 模型名 |
| `EMBEDDING_BASE_URL` | 空 | embedding endpoint；不会回退到 chat endpoint |
| `EMBEDDING_RESOLVE_IP` | 空 | 可选 embedding endpoint DNS 解析覆盖 IP |
| `EMBEDDING_API_KEY` | 空 | embedding endpoint 专用密钥 |
| `EMBEDDING_MODEL` | `text-embedding-v4` | embedding 模型名 |
| `EMBEDDING_BATCH_SIZE` | `10` | embedding 批大小 |
| `EMBEDDING_DIMENSIONS` | `1024` | embedding 维度 |
| `WORKER_CONCURRENCY` | `3` | worker 并发配置 |
| `INGESTION_FILE_CONCURRENCY` | `3` | 文件导入并发上限 |
| `MODEL_REQUEST_CONCURRENCY` | `3` | 模型请求并发上限 |
| `MODEL_REQUEST_TIMEOUT_SECONDS` | `240` | 模型请求超时秒数 |
| `CHUNK_TOKEN_BUDGET` | `2400` | active chunk token 预算 |
| `ENABLE_GRAPH_COMMUNITY_SUMMARIES` | `true` | 是否生成社区摘要视图 |
| `SIGNAL_EXTRACTION_MAX_MODEL_BATCHES` | `4` | signal extraction 模型批次数上限 |
| `SIGNAL_EXTRACTION_MAX_CANDIDATES_PER_BATCH` | `40` | 单批 signal candidate 数量上限 |
| `SIGNAL_EXTRACTION_MAX_TOKENS_PER_BATCH` | `6000` | 单批 signal measurement token 预算 |
| `SIGNAL_CANDIDATE_KEEP_THRESHOLD` | `0.62` | signal candidate 保留阈值 |
| `COMMUNITY_LOUVAIN_RESOLUTION` | `1.0` | Louvain 社区分辨率 |
| `COMMUNITY_MIN_MODULARITY_WARN` | `0.18` | 模块度告警阈值 |
| `GRAPH_OVERVIEW_MAX_NODES` | `260` | 图谱 overview 节点上限 |
| `GRAPH_OVERVIEW_MAX_EDGES` | `800` | 图谱 overview 边上限 |
| `ENABLE_MODEL_FALLBACK` | `false` | active path 禁用模型 fallback |
| `RERANKER_ENABLED` | `false` | 是否启用 reranker |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | reranker 模型 |
| `RERANKER_MAX_LENGTH` | `512` | reranker 输入最大长度 |
| `RERANKER_DEVICE` | `cpu` | reranker 运行设备 |
| `HF_HUB_OFFLINE` | `1` | HuggingFace Hub 离线模式 |
| `SEMANTIC_CHUNKING_ENABLED` | `false` | 是否启用语义切块候选 |
| `SEMANTIC_CHUNKING_MIN_LENGTH` | `2000` | 语义切块最小文本长度 |
| `RETRIEVAL_LAYER_ENABLED` | `true` | 是否启用检索分层 |
| `RETRIEVAL_CACHE_TTL_SECONDS` | `120` | 检索缓存 TTL 秒数 |
| `ENABLE_AGENTIC_REFLECTION` | `true` | 是否启用 Agent 反思 |
| `ENABLE_POST_GENERATION_REFLECTION` | `false` | 是否启用生成后反思 |
| `CITATION_VERIFICATION_SAMPLE_MAX` | `3` | 引用核验采样上限 |
| `REFLECTION_MAX_RETRIES` | `2` | 反思重试上限 |

完整参数列表以根目录 `.env.example` 为准；`apps/api/.env.example` 是 API 本地运行子集。`.env` 应使用 `SIGNAL_EXTRACTION_*` 和 `SIGNAL_CANDIDATE_KEEP_THRESHOLD`。旧 `CONCEPT_PROJECTION_*` / `CONCEPT_KEEP_POSTERIOR_THRESHOLD` 参数名已经废弃，不再作为 active 配置入口。

## 仓库结构

```text
apps/api        FastAPI、SQLAlchemy、Alembic、证据图、检索、问答、运行时设置
apps/web        Next.js 16.2.4、React 19、本地知识库 UI
apps/worker     Celery worker 与 watcher，复用 apps/api 服务逻辑
packages/shared 前后端共享 TypeScript 类型
infra           Docker Compose 默认运行栈
scripts         smoke、质量门禁、重嵌入、重解析、策略评估和维护脚本
docs            架构计划、技术规格、待办说明
local_light_tests 被 .gitignore 忽略的轻量真实资料库验收脚本
output          测试报告、截图、benchmark、smoke 输出
data            本地运行数据，禁止提交
```

`docs/todo.md` 用于记录后续工程任务；`local_light_tests/` 用于临时真实资料采样，不替代正式回归测试。

## 测试与验收

常用检查：

```powershell
python -m py_compile apps/api/app/core/config.py
python -m pytest tests
npm run typecheck --workspace web
npm run lint --workspace web
npm run test --workspace web
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api
```

后端纯单元测试应从 `apps/api` 执行。涉及 PostgreSQL、Qdrant、Redis、模型接口、导入链路、检索或问答时，应在 Docker 栈内验收，并把报告写到 `output/`。

## 使用边界

- SymboGraph 是通用本地知识库，不把课程、章节、作业、考试等领域语义写成系统真理。
- Evidence-first 是最高优先级。检索、问答、质量判断、社区摘要和策略奖励必须能回到 document version、source span、evidence atom 或 active chunk。
- LLM 是测量者和 grounded answer generator，不是默认本体构建器。
- 社区摘要是派生视图，不能替代引用，也不能成为事实源。
- 默认禁用 fallback。关键依赖不可用时应快速失败，并给出可行动错误上下文。
- PostgreSQL 是持久事实源；Qdrant 与 Redis 是派生或运行态存储，必须能从持久记录修复。

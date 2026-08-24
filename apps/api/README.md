# API 后端

## 项目简介

`apps/api` 是 SymboGraph 的 FastAPI 后端，负责知识库、上传、解析、固定 token chunk、Chunk Structure Graph、Chunk Relation Graph、RQ membership、Mid/Coarse Concept Graph、layered retrieval、context package、QA、citation verification、runtime settings 和维护入口。

首次构图与普通重建复用同一 ingestion/context-graph 事务、补偿和恢复协议；模型配置只读取根 `.env` 中已按生命周期生效的 Runtime Settings。

## 目录

| 路径 | 职责 |
| --- | --- |
| `app/api.py` | API 路由汇总入口。 |
| `app/main.py` | FastAPI 应用、CORS、API key 和启动检查。 |
| `app/models.py` | SQLAlchemy 数据模型、生命周期状态和审计表。 |
| `app/schemas.py` | Pydantic 请求与响应契约。 |
| `app/core/config.py` | `.env`、runtime settings 和配置边界。 |
| `app/services/ingestion.py` | 导入、解析、版本、取消和重建编排。 |
| `app/services/context_graph.py` | 结构图、独立关系图、RQ address/membership、L3/L2 concept projection、layered retrieval 和 context package。 |
| `app/services/auto_tpe.py` | chunk 最高版本递增时的自动 TPE 底层关系图工作点选择、trial 诊断和只读状态。 |
| `app/services/agent_graph.py` | QA、typed action、Agent trace、citation verification 和 reward/policy audit。 |
| `app/services/retrieval.py` | Layered search facade。 |
| `app/services/runtime_settings.py` | `.env` 写入、Redis version broadcast、热加载。 |
| `app/services/maintenance.py` | 清理、对账、补偿和维护操作。 |
| `app/services/error_sanitizer.py` | 外部模型、embedding、runtime probe 错误脱敏，防止 API key、Authorization header 或 provider 原始响应进入日志/API 响应。 |
| `migrations/` | Alembic schema。 |
| `tests/` | API 回归测试。 |

## 产品定位

API 层是生产形态的编排层。PostgreSQL 是事实源，Qdrant 和 Redis 是 active 派生或运行态存储；Worker 只复用这里的 service 逻辑，不复制解析、索引、图谱、检索或问答实现。

## 技术栈

| 范围 | 技术 |
| --- | --- |
| Web API | FastAPI, Pydantic |
| ORM / Migration | SQLAlchemy, Alembic |
| 存储 | PostgreSQL, Qdrant, Redis |
| 异步任务 | Celery 调用 API service logic |
| 检索 | Dense embedding, dense-only chunk relation graph, RQ membership, staged layered traversal |
| QA | OpenAI-compatible chat endpoint, typed action validator, citation verification |

## 主链路

```text
upload / source files
-> parser and layout extractor
-> fixed token chunks
-> chunk structure graph
-> contextual embedding and vector index
-> optional automatic TPE operating point selection on chunk-version increments
-> independent chunk relation graph
-> RQ primary prefix address tree and membership
-> RQ L3 mid concept graph / RQ L2 coarse concept graph
-> context graph state
-> layered search or Agent QA
-> context package
-> citation verification
-> reward and policy state
```

## 环境配置

API 从仓库根目录 `.env` 和 `apps/api/.env` 读取配置。Docker Compose 使用容器网络地址作为服务端连接：

```text
DATABASE_URL=postgresql+psycopg://<user>:<local-password>@postgres:5432/<database>
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379/0
DATA_ROOT=/app/data
```

对话、图谱、向量分别使用独立协议字段。对话与图谱可各自选择 `openai` 或 `anthropic`；向量当前只允许真实的 OpenAI-compatible 协议：

```text
CHAT_API_KEY
CHAT_API_PROTOCOL
CHAT_BASE_URL
CHAT_MODEL
GRAPH_API_KEY
GRAPH_API_PROTOCOL
GRAPH_BASE_URL
GRAPH_MODEL
EMBEDDING_API_KEY
EMBEDDING_API_PROTOCOL
EMBEDDING_BASE_URL
EMBEDDING_MODEL
EMBEDDING_DIMENSIONS
```

chat/graph 的 `anthropic` 模式使用 Anthropic Messages 契约：base 填 provider 根地址或路径前缀，不能以 `/v1` 或 `/v1/messages` 结尾，客户端固定追加 `/v1/messages`，并仅发送 `X-Api-Key`、固定 `Anthropic-Version` 与 JSON 头。chat/graph 的 `openai` 模式固定追加 `/chat/completions`。`EMBEDDING_API_PROTOCOL` 当前闭合为 `openai`，固定追加 `/embeddings`；它属于 `rebuild_required` candidate，不能通过普通 active settings PUT 热切换，也不能填写 `anthropic`。

正常 active path 保持 `ENABLE_MODEL_FALLBACK=false` 和 `ENABLE_DATABASE_FALLBACK=false`。

## 快速启动

推荐从仓库根目录启动 Docker 栈：

```powershell
docker compose -f infra/docker-compose.yml up -d --build api worker postgres redis qdrant
```

健康检查：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health
```

## 参数列表

| 分类 | 参数 |
| --- | --- |
| 基础设施 | `DATABASE_URL`, `QDRANT_URL`, `QDRANT_COLLECTION`, `REDIS_URL`, `CORS_ORIGINS`, `API_KEYS` |
| 对话模型 | `CHAT_API_KEY`, `CHAT_API_PROTOCOL` (`openai`/`anthropic`), `CHAT_BASE_URL`, `CHAT_RESOLVE_IP`, `CHAT_MODEL` |
| 图谱模型 | `GRAPH_API_KEY`, `GRAPH_API_PROTOCOL` (`openai`/`anthropic`), `GRAPH_BASE_URL`, `GRAPH_RESOLVE_IP`, `GRAPH_MODEL` |
| 向量模型 | `EMBEDDING_API_KEY`, `EMBEDDING_API_PROTOCOL` (`openai`), `EMBEDDING_BASE_URL`, `EMBEDDING_RESOLVE_IP`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `EMBEDDING_BATCH_SIZE` |
| 模型桥接 | `MODEL_BRIDGE_ENABLED`, `MODEL_BRIDGE_PORT`, `MODEL_BRIDGE_ADMIN_TOKEN` |
| 片段与图构建 | `FIXED_CHUNK_SIZE_TOKENS`, `FIXED_CHUNK_OVERLAP_TOKENS`, `RQ_KMEANS_LEVELS`, `RQ_KMEANS_MAX_K`, `RQ_RESIDUAL_TAU` |
| 稠密关系运行点 | `DENSE_KNN_K_MIN`, `DENSE_KNN_K_MAX`, `DENSE_REVERSE_B_MIN_BASE`, `DENSE_REVERSE_B_MAX_BASE`, `DENSE_REVERSE_B_MIN_DOC`, `DENSE_REVERSE_B_MAX_DOC`, `DENSE_REVERSE_B_MIN_LANG`, `DENSE_REVERSE_B_MAX_LANG`, `DENSE_MIN_COSINE`, `DENSE_STRONG_COSINE`, `CROSS_DOC_OUT_QUOTA_MIN`, `CROSS_DOC_OUT_QUOTA_MAX`, `CROSS_DOC_MIN_COSINE`, `CROSS_LANGUAGE_OUT_QUOTA_MIN`, `CROSS_LANGUAGE_OUT_QUOTA_MAX`, `CROSS_LANGUAGE_MIN_COSINE` |
| 自动 TPE 运行点 | `ENABLE_AUTO_TPE`, `TPE_TRIAL_BUDGET`, `TPE_STARTUP_RANDOM_TRIALS`, `TPE_GOOD_QUANTILE_GAMMA`, `TPE_PROBE_QUERY_BUDGET`, `TPE_TRIAL_TIMEOUT_SECONDS`, `TPE_CANDIDATE_POOL_SIZE` |
| 运行点硬门限 | `OPERATING_POINT_HARD_GATE_MAX_EDGE_DENSITY`, `OPERATING_POINT_HARD_GATE_MAX_ISOLATED_RATIO`, `OPERATING_POINT_HARD_GATE_MAX_HUBNESS_RATIO`, `OPERATING_POINT_HARD_GATE_MIN_STRUCTURE_RECOVERY_RATE`, `OPERATING_POINT_HARD_GATE_MAX_CANDIDATE_LATENCY_P95_MS` |
| 智能体遍历预算 | `AGENT_COARSE_INITIAL_BUDGET`, `AGENT_COARSE_TOP_K`, `AGENT_MID_PER_COARSE_BUDGET`, `AGENT_COARSE_DRILLDOWN_MID_INITIAL_BUDGET`, `AGENT_MID_INITIAL_BUDGET`, `AGENT_MID_TOP_K`, `AGENT_CHUNK_PER_MID_BUDGET`, `AGENT_CHUNK_INITIAL_BUDGET`, `AGENT_CHUNK_TOP_K`, `CANDIDATE_POOL_DEDUPE_BUDGET` |
| 智能体路径与上下文边界 | `AGENT_MAX_DEPTH_PER_LAYER`, `AGENT_MAX_LABELS_PER_NODE`, `AGENT_MAX_EDGE_REUSE`, `AGENT_MAX_CYCLE_REWARD_PER_PATH`, `AGENT_CYCLE_REWARD_DISTANCE_THRESHOLD`, `AGENT_PATH_DISTANCE_GREEN_THRESHOLD`, `AGENT_PATH_DISTANCE_GRAY_THRESHOLD`, `AGENT_PATH_DISTANCE_HARD_THRESHOLD`, `AGENT_STRUCTURE_RESTORE_PER_CHUNK_BUDGET`, `CONTEXT_PATH_SUMMARY_BUDGET` |
| 智能体规划与验证边界 | `AGENT_PLANNING_ROUND_BUDGET`, `AGENT_MAX_TYPED_ACTIONS_PER_ROUND`, `AGENT_REPAIR_ROUND_BUDGET`, `AGENT_VERIFICATION_BUDGET` |
| 智能体兼容别名 | `AGENT_COARSE_TOTAL_BUDGET`, `AGENT_STRUCTURE_RESTORE_BUDGET` |
| 运行边界 | `ENABLE_MODEL_FALLBACK`, `ENABLE_DATABASE_FALLBACK`, `MODEL_REQUEST_CONCURRENCY`, `MODEL_REQUEST_TIMEOUT_SECONDS` |

## 验证

从 `apps/api` 执行：

```powershell
python -m pytest tests
```

容器内执行：

```powershell
docker exec -w /app/apps/api course-kg-api python -m pytest tests
```

## 运维测试

从仓库根目录执行：

```powershell
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api --execute
python scripts/diagnose_context_graph.py
python scripts/evaluate_layered_retrieval.py
python scripts/evaluate_layered_retrieval.py --query "<query>" --execute
python scripts/check_context_package_quality.py
python scripts/check_technical_spec_compliance.py --knowledge-base-name "<knowledge-base-name>"
python scripts/reconcile_vector_records.py
```

Docker smoke 第一条命令只执行 GET 预检；确认报告中的精确 KB/query/POST
目标后才运行带 `--execute` 的第二条。Layered retrieval evaluator 默认只重放
已持久化 trace，新检索必须显式提供 query 与 `--execute`。

## 文档

- [../../docs/technical-spec.md](../../docs/technical-spec.md)：Four-Layer Context Graph RAG 技术白皮书。
- [../../scripts/README.md](../../scripts/README.md)：运维脚本说明。
- [tests/README.md](tests/README.md)：API 测试说明。

## 边界

- API 层负责后端编排，Worker 不复制实现。
- PostgreSQL 保存事实源、生命周期、trace、answer audit、citation verification、reward 和 compensation。
- Qdrant 和 Redis 必须可从 PostgreSQL 重建或刷新。
- 结构图只用于原文地址和上下文恢复，不进入 chunk relation graph。
- Mid concepts 必须由 RQ L3 prefixes 投影，Coarse concepts 必须由 RQ L2 prefixes 投影。
- Mid/coarse edges 必须由底层 chunk relation edge support 投影。
- 外部副作用失败必须写 compensation log 并抛出可行动错误。
- QA 事实只来自 context package 和 raw chunk citation span。
- 无 fallback 的集成路径必须在 Docker 栈内验证。

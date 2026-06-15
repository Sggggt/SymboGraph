# API 后端

## 项目简介

`apps/api` 是 SymboGraph 的 FastAPI 后端，负责知识库、上传、解析、固定 token chunk、Chunk Structure Graph、Chunk Relation Graph、Fine/RQ、Mid/Coarse Concept Graph、layered retrieval、context package、QA、citation verification、runtime settings 和维护入口。

## 目录

| 路径 | 职责 |
| --- | --- |
| `app/api.py` | API 路由汇总入口。 |
| `app/main.py` | FastAPI 应用、CORS、API key 和启动检查。 |
| `app/models.py` | SQLAlchemy 数据模型、生命周期状态和审计表。 |
| `app/schemas.py` | Pydantic 请求与响应契约。 |
| `app/core/config.py` | `.env`、runtime settings 和配置边界。 |
| `app/services/ingestion.py` | 导入、解析、版本、取消和重建编排。 |
| `app/services/context_graph.py` | 结构图、关系图、fine clusters、RQ、concept graph、layered retrieval 和 context package。 |
| `app/services/agent_graph.py` | QA、typed action、Agent trace、citation verification 和 reward/policy audit。 |
| `app/services/retrieval.py` | Layered search facade。 |
| `app/services/runtime_settings.py` | `.env` 写入、Redis version broadcast、热加载。 |
| `app/services/maintenance.py` | 清理、对账、补偿和维护操作。 |
| `migrations/` | Alembic schema。 |
| `tests/` | API 回归测试。 |

## 产品定位

API 层是生产形态的编排层。PostgreSQL 是事实源，Qdrant、BM25 和 Redis 是派生状态；Worker 只复用这里的 service 逻辑，不复制解析、索引、图谱、检索或问答实现。

## 技术栈

| 范围 | 技术 |
| --- | --- |
| Web API | FastAPI, Pydantic |
| ORM / Migration | SQLAlchemy, Alembic |
| 存储 | PostgreSQL, Qdrant, Redis |
| 异步任务 | Celery 调用 API service logic |
| 检索 | Dense embedding, BM25, chunk relation graph, RQ-KMeans, layered traversal |
| QA | OpenAI-compatible chat endpoint, typed action validator, citation verification |

## 主链路

```text
upload / source files
-> parser and layout extractor
-> fixed token chunks
-> chunk structure graph
-> contextual embedding and BM25
-> chunk relation graph / fine / RQ
-> mid and coarse concept graph
-> context graph state
-> layered search or Agent QA
-> context package
-> citation verification
-> reward and policy state
```

## 环境配置

API 从仓库根目录 `.env` 和 `apps/api/.env` 读取配置。Docker Compose 会把服务端连接改成容器网络地址：

```text
DATABASE_URL=postgresql+psycopg://postgres:postgres@postgres:5432/symbograph
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379/0
DATA_ROOT=/app/data
```

模型 endpoint 使用 OpenAI-compatible 协议：

```text
OPENAI_API_KEY
CHAT_BASE_URL
CHAT_MODEL
EMBEDDING_API_KEY
EMBEDDING_BASE_URL
EMBEDDING_MODEL
EMBEDDING_DIMENSIONS
```

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
| 模型 | `OPENAI_API_KEY`, `CHAT_BASE_URL`, `CHAT_MODEL`, `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` |
| Chunk / graph | `FIXED_CHUNK_SIZE_TOKENS`, `FIXED_CHUNK_OVERLAP_TOKENS`, `RQ_KMEANS_LEVELS`, `RQ_KMEANS_MAX_K`, `RQ_RESIDUAL_TAU` |
| Agent envelope | `AGENT_COARSE_ENTRY_BUDGET`, `AGENT_MID_ENTRY_BUDGET`, `AGENT_FINE_ENTRY_BUDGET`, `AGENT_FRONTIER_EXPANSION_BUDGET`, `AGENT_REPAIR_ROUND_BUDGET`, `AGENT_VERIFICATION_BUDGET` |
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
python scripts/diagnose_context_graph.py
python scripts/check_context_package_quality.py
python scripts/check_technical_spec_compliance.py --knowledge-base-name 贝叶斯
python scripts/reconcile_vector_records.py
python scripts/reconcile_bm25_records.py
```

## 文档

- [../../docs/technical-spec.md](../../docs/technical-spec.md)：Four-Layer Context Graph RAG 技术白皮书。
- [../../scripts/README.md](../../scripts/README.md)：运维脚本说明。
- [tests/README.md](tests/README.md)：API 测试说明。

## 边界

- API 层负责后端编排，Worker 不复制实现。
- PostgreSQL 保存事实源、生命周期、trace、answer audit、citation verification、reward 和 compensation。
- Qdrant、BM25、Redis 必须可从 PostgreSQL 重建或刷新。
- 外部副作用失败必须写 compensation log 并抛出可行动错误。
- QA 事实只来自 context package 和 raw chunk citation span。
- 无 fallback 的集成路径必须在 Docker 栈内验证。

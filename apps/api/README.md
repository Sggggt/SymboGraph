# API 后端

`apps/api` 是 SymboGraph 的 FastAPI 后端，负责知识库、上传、解析、固定 token chunk、Chunk Structure Graph、Chunk Relation Graph、Fine Clusters、RQ-KMeans、Mid/Coarse Concept Graph、layered retrieval、context package、QA、citation verification、runtime settings 和维护入口。

## 目录

| 路径 | 职责 |
| --- | --- |
| `app/api.py` | API 路由汇总入口。 |
| `app/main.py` | FastAPI 应用、CORS、API key 和启动检查。 |
| `app/models.py` | SQLAlchemy 数据模型。 |
| `app/schemas.py` | Pydantic 请求与响应契约。 |
| `app/core/config.py` | `.env`、runtime settings 和配置边界。 |
| `app/services/ingestion.py` | 导入、解析、版本、取消和重建编排。 |
| `app/services/context_graph.py` | 结构图、关系图、fine clusters、RQ、concept graph 和 context package。 |
| `app/services/retrieval.py` | Layered search facade。 |
| `app/services/agent_graph.py` | QA、Agent trace、citation verification。 |
| `app/services/runtime_settings.py` | `.env` 写入、Redis 版本广播、热加载。 |
| `app/services/maintenance.py` | 清理、对账和维护操作。 |
| `migrations/` | Alembic schema。 |
| `tests/` | API 回归测试。 |

## 运行

推荐从仓库根目录启动完整 Docker 栈：

```powershell
docker compose -f infra/docker-compose.yml up -d --build api worker postgres redis qdrant
```

健康检查：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health
```

容器内运行测试：

```powershell
docker exec -w /app/apps/api course-kg-api python -m pytest tests
```

## 环境

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

Cross-Encoder 精排由 `RERANKER_ENABLED=true` 启用；未启用时不会静默降级为其他精排路径。

## 数据与索引边界

- PostgreSQL 保存事实源、生命周期、审计、trace 和补偿记录。
- Qdrant、BM25 和 Redis 是派生状态。
- 外部副作用前先写状态、hash 或补偿入口。
- Qdrant 和 BM25 写入失败必须可对账、可重试、可诊断。
- QA 事实只来自 context package 和 raw chunk citation span。

## 常用命令

从 `apps/api` 执行：

```powershell
python -m pytest tests
python -m pytest tests/test_context_graph_pipeline.py -q
python -m pytest tests/test_routes_and_maintenance.py -q
```

从仓库根目录执行：

```powershell
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api
python scripts/diagnose_context_graph.py
python scripts/check_runtime_settings_contract.py
```

## API 入口

```text
GET  /api/health
GET  /api/knowledge-bases
POST /api/ingestion/upload
POST /api/ingestion/rebuild-context-graph
GET  /api/knowledge-bases/{id}/context-graph/stats
GET  /api/knowledge-bases/{id}/graph/chunk-structure
GET  /api/knowledge-bases/{id}/graph/chunk-relation
GET  /api/knowledge-bases/{id}/graph/mid-concepts
GET  /api/knowledge-bases/{id}/graph/coarse-concepts
POST /api/search/layered
POST /api/qa/context-graph
GET  /api/settings/runtime
PATCH /api/settings/runtime
```

## 验收

- 后端纯单元测试从 `apps/api` 执行。
- 涉及 PostgreSQL、Qdrant、Redis、模型接口和无 fallback 的集成路径在 Docker 栈内执行。
- 数据、检索、embedding、四层图谱、概念图或导入链路变化时，同步维护 `tests`、`scripts` 和 `output/` 验收记录。

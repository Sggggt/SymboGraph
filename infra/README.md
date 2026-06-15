# Infra

## 项目简介

`infra` 保存 SymboGraph 默认 Docker Compose 运行环境，包含 API、Worker、Web、PostgreSQL、Redis 和 Qdrant。

## 目录

| 路径 | 职责 |
| --- | --- |
| `docker-compose.yml` | 默认本地生产形态栈。 |
| `README.md` | Compose 运行说明。 |

## 产品定位

默认运行路径是 Docker Compose。涉及 PostgreSQL、Qdrant、Redis、模型接口和无 fallback 的集成路径必须在该栈内验证；`experiment` profile 不属于默认运行路径。

## 技术栈

| 服务 | 技术 |
| --- | --- |
| API | FastAPI container |
| Worker | Celery container |
| Web | Next.js container |
| Metadata | PostgreSQL 16 |
| Cache / broker | Redis 7 |
| Vector index | Qdrant 1.17.1 |

## 主链路

```mermaid
flowchart TB
    WEB["course-kg-web"] --> API["course-kg-api"]
    API --> PG["course-kg-postgres"]
    API --> R["course-kg-redis"]
    API --> Q["course-kg-qdrant"]
    W["course-kg-worker"] --> R
    W --> PG
    W --> Q
    API --> M["OpenAI-compatible models"]
    W --> M
```

## 环境配置

首次启动前创建 `.env`：

```powershell
Copy-Item .env.example .env
```

关键模型配置：

```env
OPENAI_API_KEY=...
CHAT_BASE_URL=https://your-chat-endpoint/v1
CHAT_MODEL=your-chat-model
EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=https://your-embedding-endpoint/v1
EMBEDDING_MODEL=your-embedding-model
EMBEDDING_DIMENSIONS=1024
ENABLE_MODEL_FALLBACK=false
ENABLE_DATABASE_FALLBACK=false
```

容器内固定覆盖：

```text
DATABASE_URL=postgresql+psycopg://postgres:postgres@postgres:5432/symbograph
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379/0
DATA_ROOT=/app/data
```

## 快速启动

启动完整栈：

```powershell
docker compose -f infra/docker-compose.yml up -d --build
```

查看服务：

```powershell
docker compose -f infra/docker-compose.yml ps
```

## 参数列表

| 分类 | 参数 |
| --- | --- |
| 镜像与端口 | `API_IMAGE`, `WEB_IMAGE`, `API_HOST_PORT`, `WEB_HOST_PORT` |
| 数据服务 | `DATABASE_URL`, `QDRANT_URL`, `QDRANT_COLLECTION`, `REDIS_URL` |
| 数据目录 | `DATA_ROOT`, `STORAGE_ROOT`, `INGESTION_ROOT` |
| 模型 | `MODEL_BRIDGE_ENABLED`, `MODEL_BRIDGE_PORT`, `CHAT_*`, `EMBEDDING_*` |
| Worker | `WORKER_CONCURRENCY`, `INGESTION_TASK_QUEUE` |
| Fallback | `ENABLE_MODEL_FALLBACK`, `ENABLE_DATABASE_FALLBACK` |

## 验证

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:3000
docker exec -w /app/apps/api course-kg-api python -m pytest tests
```

## 运维测试

```powershell
docker compose -f infra/docker-compose.yml logs -f api
docker compose -f infra/docker-compose.yml logs -f worker
docker compose -f infra/docker-compose.yml restart api worker
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api --worker-container course-kg-worker
```

## 文档

- [../README.md](../README.md)：仓库总览。
- [../apps/api/README.md](../apps/api/README.md)：API 后端。
- [../apps/worker/README.md](../apps/worker/README.md)：Worker。
- [../scripts/README.md](../scripts/README.md)：运维脚本。

## 边界

- 不绕过 Docker 直接修改生产形态 PostgreSQL、Redis 或 Qdrant。
- Compose 默认服务名保持 `course-kg-*`。
- API 容器工作目录是 `/app/apps/api`，脚本挂载到 `/app/scripts`，数据目录挂载到 `/app/data`。
- 日志、smoke 输出和临时报告写入仓库根目录 `output/`。
- 改动端口、镜像依赖、Celery pool/fork 规模等需要 service recreate 或 rebuild。

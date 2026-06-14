# Infra

`infra` 保存 SymboGraph 默认 Docker Compose 运行环境，包含 API、Worker、Web、PostgreSQL、Redis 和 Qdrant。

## 启动

首次启动前创建 `.env`：

```powershell
Copy-Item .env.example .env
```

启动完整栈：

```powershell
docker compose -f infra/docker-compose.yml up -d --build
```

## 服务

| Compose 服务 | 容器名 | 职责 |
| --- | --- | --- |
| `api` | `course-kg-api` | FastAPI 后端。 |
| `worker` | `course-kg-worker` | Celery 长任务和 watcher。 |
| `web` | `course-kg-web` | Next.js 前端。 |
| `postgres` | `course-kg-postgres` | PostgreSQL 事实源。 |
| `redis` | `course-kg-redis` | Celery broker、共享缓存、runtime settings version broadcast。 |
| `qdrant` | `course-kg-qdrant` | Chunk 向量派生索引。 |

```mermaid
flowchart TB
    WEB["course-kg-web"] --> API["course-kg-api"]
    API --> PG["course-kg-postgres"]
    API --> R["course-kg-redis"]
    API --> Q["course-kg-qdrant"]
    W["course-kg-worker"] --> R
    W --> PG
    W --> Q
```

## 关键环境

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

## 常用命令

```powershell
docker compose -f infra/docker-compose.yml ps
docker compose -f infra/docker-compose.yml logs -f api
docker compose -f infra/docker-compose.yml logs -f worker
docker compose -f infra/docker-compose.yml restart api worker
docker compose -f infra/docker-compose.yml down
```

## 健康检查

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:3000
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api --worker-container course-kg-worker
```

## 边界

- 涉及 PostgreSQL、Qdrant、Redis、模型接口和无 fallback 的集成路径在 Docker 栈内运行。
- 不绕过 Docker 直接修改生产形态 PostgreSQL、Redis 或 Qdrant。
- `experiment` profile 不属于默认运行路径。
- 日志、smoke 输出和临时报告写入仓库根目录 `output/`。

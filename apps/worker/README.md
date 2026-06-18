# Worker

## 项目简介

`apps/worker` 提供 Celery worker 与文件 watcher。Worker 只执行后台任务，复用 `apps/api` 的 service 逻辑，不复制解析、切块、索引、图谱构建、检索或问答实现。

## 目录

| 路径 | 职责 |
| --- | --- |
| `celery_app.py` | Celery app 配置。 |
| `tasks.py` | 导入、解析、图谱重建和维护任务入口。 |
| `watcher.py` | 文件 watcher 入口。 |
| `README.md` | 本说明。 |

## 产品定位

Worker 是长任务执行器，不是第二套业务实现。它从 Redis broker 取任务，在任务边界刷新 runtime settings，调用 API service layer 完成导入、索引、图谱和补偿。

## 技术栈

| 范围 | 技术 |
| --- | --- |
| 任务队列 | Celery |
| Broker | Redis |
| 服务逻辑 | 复用 `apps/api/app/services/*` |
| 运行环境 | Docker Compose service `worker` |

## 主链路

```mermaid
flowchart LR
    API["API enqueue"] --> R["Redis broker"]
    R --> W["course-kg-worker"]
    W --> S["API service layer"]
    S --> PG["PostgreSQL"]
    S --> Q["Qdrant"]
    S --> BM["BM25 records"]
    S --> C["Compensation logs"]
```

Worker 执行：

- 文件导入批次。
- 文件解析与固定 token chunk。
- Chunk Structure Graph、坐标和映射。
- Contextual embedding、Qdrant upsert 和 BM25 records。
- Chunk Relation Graph、RQ address/membership、RQ L3 Mid Concept Graph、RQ L2 Coarse Concept Graph、Context Graph state。
- 批次取消、补偿记录、可重试失败和 heartbeat。

## 环境配置

Worker 使用同一份 `.env` 和 Docker Compose 覆盖：

```text
DATABASE_URL=postgresql+psycopg://postgres:postgres@postgres:5432/symbograph
REDIS_URL=redis://redis:6379/0
QDRANT_URL=http://qdrant:6333
DATA_ROOT=/app/data
```

任务并发由 `WORKER_CONCURRENCY` 与 Compose 启动命令控制；运行中修改并发需要重启或重建 worker 服务。

## 快速启动

```powershell
docker compose -f infra/docker-compose.yml up -d --build worker
```

容器名：

```text
course-kg-worker
```

## 参数列表

| 分类 | 参数 |
| --- | --- |
| Celery | `INGESTION_TASK_QUEUE`, `WORKER_CONCURRENCY`, `WORKER_MAX_TASKS_PER_CHILD` |
| Runtime reload | `REDIS_URL`, `RUNTIME_SETTINGS_VERSION` 相关 Redis broadcast |
| 资源保护 | `INGESTION_MEMORY_SOFT_LIMIT_RATIO`, `INGESTION_MEMORY_HARD_LIMIT_RATIO`, `INGESTION_MEMORY_CRITICAL_LIMIT_RATIO` |
| 模型调用 | `MODEL_REQUEST_CONCURRENCY`, `MODEL_REQUEST_TIMEOUT_SECONDS` |

## 验证

```powershell
docker ps --filter "name=course-kg-worker"
docker logs --tail 100 course-kg-worker
docker exec -w /app/apps/api course-kg-api python -m pytest tests
```

## 运维测试

```powershell
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api --worker-container course-kg-worker
python scripts/check_runtime_settings_contract.py
python scripts/runtime_hot_reload_probe.py
```

## 文档

- [../../README.md](../../README.md)：仓库总览。
- [../../apps/api/README.md](../../apps/api/README.md)：API service layer。
- [../../infra/README.md](../../infra/README.md)：Docker Compose 环境。
- [../../scripts/README.md](../../scripts/README.md)：运维脚本。

## 边界

- Worker 不复制业务逻辑，只调用 API service。
- 任务开始前刷新 `.env`/Redis runtime settings version；长任务关键阶段再次检查版本。
- 外部副作用失败必须写 compensation log 并暴露可行动错误。
- 不能把跨进程正确性建立在 worker 内存状态上。
- 涉及 PostgreSQL、Qdrant、Redis 和模型接口的路径必须在 Docker 栈内验证。

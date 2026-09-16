# Docker 运行环境

Compose 定义见 [docker-compose.yml](docker-compose.yml)。根启动器负责 API 镜像、后端服务与就绪检查；Web 使用宿主原生 Node.js。

## 服务与数据

| Compose service | 容器 | 作用 |
|---|---|---|
| `api` | `course-kg-api` | FastAPI，开发模式挂载源码 |
| `worker` | `course-kg-worker` | 有界后台任务，复用 API 镜像 |
| `beat` | `course-kg-beat` | 独立 Celery 调度器 |
| `postgres` | `course-kg-postgres` | 元数据、生命周期和审计 |
| `redis` | `course-kg-redis` | 队列、缓存和运行时版本 |
| `qdrant` | `course-kg-qdrant` | 向量索引 |
| `model-bridge` | `course-kg-model-bridge` | 可选私有模型路由桥 |

应用数据卷 `symbograph-data` 挂载为 `/app/data`，其他持久卷为 `postgres-data`、`redis-data`、`qdrant-data`。Docker 会加 Compose project 前缀。仓库 `data/` 只用于忽略的本地输入，不是默认应用卷。

恢复已有环境必须沿用根 `.env` 中的 Compose project 配置；改变 project identity 可能连到另一组空卷。

## 配置

根 `.env` 只保存秘密、连接、路径、端口和进程/服务启动参数；根 `settings.json` 只保存非秘密产品、检索、预算和构建参数。Compose 将两者分别挂载到 `/workspace/.env` 和 `/workspace/settings.json`，服务通过 `RUNTIME_ENV_FILE` 与 `RUNTIME_SETTINGS_FILE` 读取。两个文件不能含同名键。容器内数据库和派生存储使用 Compose 网络地址；不要把宿主 `localhost` 当作另一个容器。

| 模型用途 | 协议与地址规则 |
|---|---|
| chat、graph | 各自独立选择 `openai` 或 `anthropic` |
| OpenAI-compatible | base 后追加 `/chat/completions`；例如 `https://models.invalid/v1` |
| Anthropic Messages | base 使用 provider 根或前缀，不能以 `/v1` 或 `/v1/messages` 结尾；客户端追加 `/v1/messages` |
| embedding | 当前仅 `openai`，base 后追加 `/embeddings` |

三组模型配置、密钥和协议互不替代。模型桥不改变来源、认证和错误边界；真实秘密和部署连接只存根 `.env`，本机 `settings.json` 与 `.env` 均被 Git 忽略，仓库只保存脱敏 example。旧单文件环境先运行 `python scripts/migrate_runtime_config.py` 查看只读计划，核对后再显式加 `--execute`。设置变更的生命周期见[白皮书](../docs/technical-spec.md#配置与运行)。

## 常用操作

资源管理器双击根目录 `start-app.bat` 可一键启动完整环境；仅启动 Web 可双击 `start-web.bat`。两个窗口在成功或失败后都会显示明确终态并等待按键，不再自动关闭。服务在后台运行，关闭启动窗口不会停止服务。命令行调用使用 PowerShell 入口：

```powershell
.\start-app.ps1 -NoBrowser
.\start-web.ps1 -NoBrowser
.\rebuild-images.ps1
docker compose -p knowledgegraph-dev-20260820 --env-file .env -f infra/docker-compose.yml ps
```

自动化若必须调用 `.bat`，可在当前进程设置 `SYMBOGRAPH_NO_PAUSE=1`；该变量只控制批处理窗口等待，不进入根 `.env`，也不改变服务配置、Compose project 或数据卷。

重建脚本只构建 API 镜像，Worker/Beat 复用该镜像。digest 引用只能用于运行，不能作为 build 输出 tag。已有匹配镜像可显式指定：

```powershell
.\start-app.ps1 -SkipBuild -NoBrowser -ApiImage 'course-kg-api:dev'
```

此命令不会证明镜像与当前源码一致。新依赖或正式发布应重新构建并验证；当前 Compose 的源码挂载与隔离镜像验收是不同范围。

停止宿主 Web 并保留后端：

```powershell
.\stop-app.ps1 -KeepBackend
```

停止当前项目而保留数据：

```powershell
docker compose -p knowledgegraph-dev-20260820 --env-file .env -f infra/docker-compose.yml --profile model-bridge stop
```

不要用 `down -v`、全局 `docker system prune` 或删卷来处理普通故障。只删除已确认无容器引用的本项目临时候选镜像；不要清理其他项目。

## 就绪与问题排查

默认宿主 Web 为 `http://127.0.0.1:3000`，API readiness 为 `http://127.0.0.1:8000/api/ready`。Web PID 与日志写入忽略的 `output/`；API 容器工作目录 `/app/apps/api`，脚本挂载 `/app/scripts`，临时结果挂载 `/app/output`。

启动失败先看对应容器状态和安全日志，再用[运维脚本](../scripts/README.md)核对 schema、批次恢复、freshness 或向量对账。实例不可用、空数据卷、模型错误和资料不足分别处理。检查环境时不输出完整 `.env`、容器 Env 或模型响应。

## 目标检索架构的部署边界

当前 Compose 不包含 Web service，且固定 Qdrant v1.17.1；最新官方文本检索文档中的新版本字段不能直接假定在此镜像可用。

BM25 复用 PostgreSQL 保存原文 postings、词典和版本化统计，由 Docker 内 API service 评分，不新增必需搜索容器。若后续采用 Qdrant sparse 加速或升级镜像，需先验证同 KB 统计、分词、融合秩和结果等价性，再走明确迁移/发布流程。

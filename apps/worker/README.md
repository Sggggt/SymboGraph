# Worker 与 Beat

Worker 从 Redis 消费导入、构图和维护任务，调用 API service。Beat 独立运行，定时发布恢复检查；Worker 不内嵌调度器。

| 文件 | 职责 |
|---|---|
| `worker_app/celery_app.py` | Celery、队列、并发和调度配置 |
| `worker_app/tasks.py` | 后台任务入口 |
| `worker_app/watcher.py` | 受控文件监听 |
| `worker_app/bootstrap.py` | API service 导入路径 |

## 运行与恢复

默认容器为 `course-kg-worker` 和 `course-kg-beat`，共享 API 镜像及 `/app/data` 数据卷。任务开始和阶段边界刷新根 `.env`、根 `settings.json` 与 Redis 组合运行时版本；两份文件分别拥有不重叠的配置键。

Beat 每 60 秒发布中断导入批次的对账任务。长期任务保存进度、原因和补偿记录；重启不从内存猜测应恢复的版本。

```powershell
docker logs --tail 100 course-kg-worker
docker logs --tail 100 course-kg-beat
```

启动或停止使用[现有 Compose project](../../infra/README.md)。修改进程池大小、镜像或容器形态需要 recreate，不能只靠热加载宣称生效。

## 约束与测试

不复制 API 的解析、索引、图构建或问答逻辑。模型和 I/O 使用有界并发，任务预取和进程回收按工程配置管理。数据库连接在 fork 后正确重建；提交与外部副作用遵循原服务的事务/恢复协议。

BM25 索引准备、发布、恢复复用 API service，按 KB 资源锁、有界批次和源版本工作；Worker 不维护专属分词或统计公式。作业保存 predecessor、batch、publish 与 diagnostics 身份，失败 candidate 不改写 active 指针。

回归入口在 API tests，包括 Worker fork、取消补偿、队列就绪和版本恢复。见[开发与测试](../../docs/development.md)。生产无 fallback 验证在 Docker 内执行，日志不写凭据或 provider 原文。

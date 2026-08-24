# API 测试

## 项目简介

`apps/api/tests` 覆盖 Four-Layer Context Graph RAG 的后端回归，重点验证导入、四层图谱、检索、QA、runtime settings、维护入口和 migration 兼容性。

## 目录

| 路径 | 职责 |
| --- | --- |
| `test_context_graph_pipeline.py` | fixed chunk、structure graph、chunk relation/RQ membership、mid/coarse、context package。 |
| `test_auto_tpe.py` | 自动 TPE sampler、版本递增触发、hard gate 阻断和 active relation graph 绑定。 |
| `test_cache_manager.py` | Redis cache miss/no-op 边界，防止进程内缓存替代共享 Redis。 |
| `test_error_sanitizer.py` | 外部服务错误脱敏，防止 API key、Authorization header 或 provider 原始响应泄露。 |
| `test_ingestion_logs.py` | 导入日志结构、阶段和状态文案契约。 |
| `test_model_bridge.py` | 模型桥接配置、reload 和自指向防护。 |
| `test_squashed_migration_baseline.py` | Alembic baseline、head、当前 SQLAlchemy metadata 一致性，以及 RQ primary-chain schema 的显式迁移门禁。 |
| `test_manage_migrations.py` | 迁移 preflight、upgrade 与 downgrade 安全门。 |
| `test_rq_primary_membership.py` | RQ 唯一 L1/L2/L3 主链、置信度、角色与确定性 state hash。 |
| `test_scoped_rq_protocol_migration.py` | RQ scoped rebuild 只接纳被替换层的有界 staleness。 |
| `test_routes_and_maintenance.py` | API routes、runtime settings、maintenance 与 reconciliation。 |
| `test_*.py` | 其他后端回归测试。 |
| `README.md` | 本测试说明。 |

## 主链路

测试覆盖：

```text
fixed token chunks
-> structure graph
-> contextual embedding/vector metadata
-> automatic TPE bottom relation operating point
-> chunk relation/RQ membership
-> RQ L3 mid graph / RQ L2 coarse graph states
-> layered retrieval trace
-> context package
-> QA/citation verification audit
-> runtime settings and maintenance
```

本地从 `apps/api` 执行；涉及 PostgreSQL、Qdrant、Redis、模型接口和 no-fallback 的真实集成路径仍以 Docker smoke 和诊断脚本补足。白皮书强约束包括结构图不进入 relation graph、RQ 每个 chunk 只持久化唯一三层主链、Mid=L3、Coarse=L2 和 Redis 不可用时缓存不落入进程内状态。

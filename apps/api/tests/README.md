# API 测试

## 项目简介

`apps/api/tests` 覆盖 Four-Layer Context Graph RAG 的后端回归，重点验证导入、四层图谱、检索、QA、runtime settings、维护入口和 migration 兼容性。

## 目录

| 路径 | 职责 |
| --- | --- |
| `test_context_graph_pipeline.py` | fixed chunk、structure graph、chunk relation/RQ membership、mid/coarse、context package。 |
| `test_cache_manager.py` | Redis cache miss/no-op 边界，防止进程内缓存替代共享 Redis。 |
| `test_routes_and_maintenance.py` | API routes、runtime settings、maintenance 与 reconciliation。 |
| `test_*.py` | 其他后端回归测试。 |
| `README.md` | 本测试说明。 |

## 主链路

测试覆盖：

```text
fixed token chunks
-> structure graph
-> contextual embedding/BM25 metadata
-> chunk relation/RQ membership
-> RQ L3 mid graph / RQ L2 coarse graph states
-> layered retrieval trace
-> context package
-> QA/citation verification audit
-> runtime settings and maintenance
```

本地从 `apps/api` 执行；涉及 PostgreSQL、Qdrant、Redis、模型接口和 no-fallback 的真实集成路径仍以 Docker smoke、诊断脚本和真实资料验收补足。白皮书强约束包括结构图不进入 relation graph、RQ fallback pair 不补边、Mid=L3、Coarse=L2 和 Redis 不可用时缓存不落入进程内状态。

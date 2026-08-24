# 运维脚本

`scripts/` 只保留 Four-Layer Context Graph RAG 主链需要的重建、诊断、对账、迁移和 smoke 入口。

## 共同约束

- 默认只读；写操作必须要求 `--execute` 或等价的显式确认。
- 破坏性操作必须先输出完整、有界的目标清单和 identity hash。
- PostgreSQL、Qdrant、Redis 和真实 provider 操作只允许在 Docker Compose 内执行。
- 禁止记录 endpoint、API key、Authorization header、provider 原文、个人文件名或私有资料 hash。
- 临时诊断写入被 Git 忽略的 `output/`，核验后清空。
- CLI 的 `--help` 必须在导入数据库、模型或向量客户端之前可用。

## 常用入口

### 构建与重建

| 脚本 | 用途 |
| --- | --- |
| `rebuild_chunks.py` | 从持久化 source root 重建固定 chunk、结构、向量和四层图。 |
| `rebuild_structure_graph.py` | 重建 Chunk Structure Graph。 |
| `rebuild_chunk_relation_graph.py` | 重建 Chunk Relation Graph 及依赖状态。 |
| `rebuild_rq_membership_graph.py` | 重建 RQ membership、Mid、Coarse 和 Context Graph。 |
| `rebuild_mid_concept_graph.py` | 从 active RQ 状态重建 Mid 及下游层。 |
| `rebuild_coarse_concept_graph.py` | 从 active Mid 状态重建 Coarse 及 Context Graph。 |
| `rebuild_context_graph_all.py` | 执行完整 contextual-index / graph-only 分阶段重建。 |
| `retry_versioned_graph.py` | 重试一个已持久化、可恢复的版本化构图任务。 |

### Runtime Settings 与向量生命周期

| 脚本 | 用途 |
| --- | --- |
| `manage_runtime_settings_candidate.py` | 管理 rebuild-required candidate、shadow build、evaluation 和 promotion。 |
| `runtime_hot_reload_probe.py` | 验证 active runtime version、Redis broadcast 和单例刷新。 |
| `manage_vector_shadow.py` | 管理向量 shadow build、promotion、rollback 和 abandon。 |
| `reconcile_vector_records.py` | 对账 PostgreSQL vector records、Qdrant points 和 durable outbox。 |
| `reconcile_scoped_rebuild_cache_invalidations.py` | 重放构图提交后的缓存失效 intent。 |
| `reconcile_versioned_graph_completion.py` | 对账已提交版本的图状态和缓存发布。 |
| `refresh_context_protocol_identity.py` | 刷新 Context Graph 协议身份。 |

### 数据与恢复

| 脚本 | 用途 |
| --- | --- |
| `reconcile_ingestion_batch_recoveries.py` | 检查导入批次 before-image、补偿和恢复状态。 |
| `reconcile_source_snapshots.py` | 校验并修复 immutable source snapshot 引用。 |
| `source_snapshot_gc.py` | 清点并按 retention gate 清理孤立 source snapshot。 |
| `cleanup_stale_data.py` | 清点 stale vector / inactive chunk 派生数据。 |
| `cleanup_vector_collection.py` | 通过 durable intent 清理无 active 引用的 Qdrant collection。 |

### 诊断与验收

| 脚本 | 用途 |
| --- | --- |
| `diagnose_context_graph.py` | 输出当前四层图质量诊断。 |
| `evaluate_layered_retrieval.py` | 重放或显式执行 layered retrieval。 |
| `evaluate_agent_trace.py` | 重放 Agent、Context Package、citation 和 reward 审计。 |
| `check_context_package_quality.py` | 检查 Context Package 去重、结构恢复和引用 span。 |
| `check_runtime_settings_contract.py` | 检查单一根 `.env` 与三类 lifecycle 契约。 |
| `check_technical_spec_compliance.py` | 检查白皮书强不变量。 |
| `probe_embedding_provider.py` | 在 Docker 内执行脱敏 embedding 连通性检查。 |
| `docker_smoke.py` | GET-only dry-run；`--execute` 后执行 Search/QA smoke。 |

### 迁移

| 脚本 | 用途 |
| --- | --- |
| `manage_migrations.py` | 在 Docker API 环境执行 Alembic preflight、upgrade 和 destructive gate。 |

## 内部 helper

以下模块由上述入口复用，不是独立运维命令：

- `_context_graph_maintenance.py`
- `_destructive_cleanup_guard.py`
- `_gray_zone_audit.py`
- `_quality_gate.py`

## 提交前检查

```powershell
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api
python scripts/check_runtime_settings_contract.py
python scripts/check_technical_spec_compliance.py
```

只有在审核 dry-run 目标后，才允许增加 `--execute`。

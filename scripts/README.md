# 持久运维与验收脚本

这里保留可重复使用的入口，不保存一次性调试程序或验收报告。`intent_execution_retrieval_v1` 的读取器和验收器已接入当前主链；旧协议工具只用于历史记录重放。脚本复用 API service；涉及数据库、向量、Redis 或模型时优先在 Docker 服务环境执行。

## 执行原则

- 先运行 `--help` 和默认只读计划，核对资料库、作用范围、版本影响和时限。
- 写操作需要 `--execute` 或脚本声明的专用执行 flag；破坏性清理还要求目标/清单确认，不能只凭文件名猜测用途。
- 一次验收开始后冻结输入和代码，失败/取消/未运行分列。技术失败按计划止损，不自动重发同一题。
- `output/` 只放临时结果，核验后将必要结论写入 Git 忽略的本地验收记录并清理。不要输出凭据、provider 原始响应或私有内容指纹。

所有可执行 Python 入口均列在下表；以下划线开头的模块为内部 helper。

## 构建与分层重建

| 脚本 | 用途 |
|---|---|
| `rebuild_chunks.py` | 重新解析和固定切块；按版本规则管理目标文件。 |
| `rebuild_structure_graph.py` | 重建原文结构图及依赖状态。 |
| `rebuild_chunk_relation_graph.py` | 重建片段关系及下游状态。 |
| `rebuild_rq_membership_graph.py` | 重建 RQ 主成员关系和相关高层图。 |
| `rebuild_mid_concept_graph.py` | 重建中层概念及下游图。 |
| `rebuild_coarse_concept_graph.py` | 重建粗层概念及上下文图。 |
| `rebuild_context_graph_all.py` | 完整 contextual-index/graph-only 分阶段重建。 |
| `retry_versioned_graph.py` | 重试已持久化且可恢复的版本化构图任务。 |

## 数据、补偿与维护

| 脚本 | 用途 |
|---|---|
| `reconcile_ingestion_batch_recoveries.py` | 收敛导入 before-image、取消补偿和中断恢复。 |
| `reconcile_source_snapshots.py` | 核验及修复不可变源快照引用。 |
| `reconcile_vector_records.py` | 核对 PostgreSQL、Qdrant 与向量 outbox。 |
| `reconcile_scoped_rebuild_cache_invalidations.py` | 重放构图后的缓存失效意图。 |
| `reconcile_versioned_graph_completion.py` | 核对已提交版本的图状态和发布完成情况。 |
| `source_snapshot_gc.py` | 盘点快照引用，再按保留规则清理孤立快照。 |
| `cleanup_stale_data.py` | 盘点并清理满足删除条件的 stale 派生数据。 |
| `cleanup_vector_collection.py` | 通过持久意图清理无 active 引用的集合。 |
| `coalesce_maintenance_queue.py` | 合并可合并的扫描型维护任务，保留持久事实。 |
| `maintain_graph_statistics.py` | 对固定表 allowlist 执行数据库统计维护。 |

## 配置、迁移与发布

| 脚本 | 用途 |
|---|---|
| `manage_migrations.py` | Alembic preflight、升级和受保护的降级入口；当前公开升级链从上一推送版 `20260824_0044` 直接进入合并后的 `20260916_0045`。 |
| `check_release_schema.py` | 核对迁移 head 与数据库 schema 漂移。 |
| `migrate_runtime_config.py` | 默认只读规划 `.env`/`settings.json` 权威拆分；`--execute` 后原子迁移非秘密运行参数。 |
| `manage_runtime_settings_candidate.py` | 重建型配置的 candidate、shadow、evaluation 和 promotion。 |
| `manage_vector_shadow.py` | 向量 shadow 构建、提升、回滚和放弃。 |
| `runtime_hot_reload_probe.py` | 显式验证版本广播、单例刷新和热加载。 |
| `check_runtime_settings_contract.py` | 检查根 `.env`/`settings.json` 键归属、example 完整性与三级生命周期契约。 |
| `refresh_context_protocol_identity.py` | 受控刷新上下文协议身份；不能代替必要的数据重建。 |
| `prepare_release_snapshot.py` | 导出隔离的源码快照，不修改真实 Git index。 |

## 只读诊断与功能验收

| 脚本 | 用途 |
|---|---|
| `diagnose_context_graph.py` | 共享只读准备后检查四层图；抽样显示与完整质量检查分列。 |
| `check_context_package_quality.py` | 核验装包、去重、结构恢复与来源 span。 |
| `check_technical_spec_compliance.py` | 扫描静态边界并核对持久图/协议不变量。 |
| `evaluate_layered_retrieval.py` | 重放已有检索，或显式执行新的检索。 |
| `evaluate_agent_trace.py` | 按记录版本重放目标或历史执行、上下文和来源审计。 |
| `evaluate_intent_execution.py` | 目标协议五题冻结 gold 验收；默认 dry-run，`--execute` 顺序调用 QA 并重放持久硬门禁，记录 v2 evidence/context plan、逐模型调用 input/output/cache token、TTFT、各阶段、provider 往返区间与非模型墙钟。 |
| `evaluate_intent_conversations.py` | 目标协议十轮会话验收，覆盖能力卡、同会话复用、安全转检索、跨来源、证据不足与结构题。 |
| `probe_embedding_provider.py` | 脱敏的真实 embedding 连通性探测。 |
| `docker_smoke.py` | 默认 GET-only 计划；显式执行 Search/QA 并核对契约。 |
| `check_repository_hygiene.py` | 检查本地文档链接、Git 忽略规则及仓库卫生，不访问外部服务。 |

## 性能与校准

| 脚本 | 用途 |
|---|---|
| `benchmark_build_pipeline.py` | 全量冷构建计划与显式执行，复用生产 service/Worker。 |
| `benchmark_graph_kernels.py` | 公开合成规模的数值核心与标量参考对比。 |
| `benchmark_rq_build.py` | 完整 RQ 准备、构建、支撑和诊断计时。 |
| `benchmark_vector_validation.py` | 向量规范化与校验的轻量性能检查。 |
| `monitor_build_resources.py` | 对目标构建采样进程/容器资源。 |
| `revalidate_build_resources.py` | 按既定口径复核资源采样，保留原失败结果。 |

## 典型流程

从仓库根运行帮助，确认当前入口参数：

```powershell
docker exec course-kg-api python /app/scripts/reconcile_vector_records.py --help
docker exec course-kg-api python /app/scripts/benchmark_build_pipeline.py --help
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api
```

确认 smoke 计划后才增加 `--execute`。只读模式不会发送 Search/QA POST；结构/来源检查通过与答案语义质量分别判断。

目标五题 QA 使用 `evaluate_intent_execution.py`，输入必须在执行前冻结且恰好包含五题。先运行默认 dry-run，再使用 `--execute`、目标资料库和 `output/` 新报告路径。脚本继续记录单题失败并核对 run、target trace、Context Package、零模型来源准入、一次生成、来源绑定和零在线奖励。

目标十轮会话使用 `evaluate_intent_conversations.py`，输入必须在执行前冻结且恰好包含十例。`session_from` 只能引用前序用例，用于构造同会话链。普通检索题必须完成来源绑定，只有 insufficient 类允许有界 gap。评分依据是独立 gold 和真实原文，不是模型自评分。私有 gold 放忽略的数据目录，不能成为公共 fixture。

冷构建使用 `benchmark_build_pipeline.py`：默认只读；执行需要目标资料库、`--execute`、`--cold`、全量重解析标记及截止秒数。是否通过还取决于全量成功、图/来源/向量/freshness 和资源检查，不只看计算结束时间。

## 审计口径

目标读取器须重放 Task/Intent/ExecutionStrategy、合法空词面、各层通道列表/权重/融合、原文过滤域、RQ/BM25 映射、路径和确定性来源准入。原始与生效计划、索引快照及候选截断必须可追踪；prepared 是持久意图，不代表模型已经返回。

目标采集器按 Search/同步/SSE 分别核对同一策略契约；记录部分回答、有效拒答、技术失败和完整答案，不能以来源绑定通过代替语义 gold。旧阶段/字段只按兼容协议读取，详细隔离见[历史兼容](../docs/reference/compatibility.md)。

RQ 对账读取完整所需成员字段与支撑，不通过裁剪审计达标。向量维护将 active/ready 与历史 inactive/stale 分开；有界 inventory 若标记 truncated，不能宣称全清单无孤儿。

## 内部模块

`_context_graph_maintenance.py` 管运行上下文和通用维护；`_destructive_cleanup_guard.py` 管删除门禁；`_gray_zone_audit.py`、`_quality_gate.py`、`_rq_acceptance.py` 供入口复用；`_graph_scalar_reference.py` 仅为合成数值回归提供标量参考。不要将 helper 当成独立命令。

## 维护检查

新增或移除入口时更新本表及相关 tests。检查 `--help` 不应依赖数据库或模型可用性。纯文档/仓库清理运行卫生检查及相关脚本契约测试，不自动启动模型验收。

```powershell
python scripts/check_repository_hygiene.py
git diff --check
```

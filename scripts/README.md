# Operations Scripts

## 项目简介

`scripts` 保存 SymboGraph 的重建、破坏性清理、对账、诊断、检索评估、质量检查、runtime probe 和 Docker smoke 脚本。脚本必须能从仓库根目录重复执行，并能在 API 容器中通过 `/app/scripts` 运行。

## 目录

| 脚本 | 职责 |
| --- | --- |
| `rebuild_chunks.py` | 将 source files 重解析为 fixed token chunks、structure graph、contextual embeddings、BM25 和四层图。写入需要 `--execute`；已有解析数据还需要 `--full-reparse`。 |
| `rebuild_structure_graph.py` | 重建 structure nodes、edges、mappings 和 coordinates。写入需要 `--execute`；已有解析数据还需要 `--full-reparse`。 |
| `rebuild_chunk_relation_graph.py` | 重建 independent chunk relation graph、RQ address/membership 以及依赖的 mid/coarse/context states。写入需要 `--execute`。 |
| `rebuild_rq_membership_graph.py` | 重建 active RQ address/membership 以及依赖的 mid/coarse/context states。写入需要 `--execute`。 |
| `rebuild_mid_concept_graph.py` | 重建 RQ L3 投影的 LLM-grounded mid concepts 和依赖状态。写入需要 `--execute`。 |
| `rebuild_coarse_concept_graph.py` | 重建 RQ L2 投影的 coarse concepts 和 active context graph state。写入需要 `--execute`。 |
| `rebuild_context_graph_all.py` | 从当前 chunks 重建所有 active 四层派生图状态。写入需要 `--execute`。 |
| `destroy_legacy_derived_data.py` | 清理 legacy derived state、legacy profile strategy keys 和可选 legacy score audit。写入需要 `--execute --confirm-destroy-legacy`。 |
| `cleanup_stale_data.py` | 清理 stale vector/BM25/Qdrant state；`--execute --delete-inactive-chunks` 会删除 inactive chunk versions 和依赖项。 |
| `reconcile_vector_records.py` | 对账 PostgreSQL vector records 与 Qdrant points。 |
| `reconcile_bm25_records.py` | 对账 BM25 records 与 active chunks/contextual texts；`--execute` 创建缺失记录，`--delete-stale` 删除陈旧行。 |
| `diagnose_context_graph.py` | 输出 counts、freshness、grounding、RQ diagnostics 和四层 payload sample。 |
| `evaluate_layered_retrieval.py` | 运行真实 layered retrieval queries 并输出 entry/frontier/path diagnostics。 |
| `evaluate_agent_trace.py` | 运行真实 QA 请求并验证 Agent trace nodes、citations 和非降级执行。 |
| `check_context_package_quality.py` | 验证 context package closure、citation spans、graph paths、RQ metrics 和 token budget metadata。 |
| `check_runtime_settings_contract.py` | 验证 runtime settings lifecycle、hot reload contract 和边界规则。 |
| `runtime_hot_reload_probe.py` | 探测 Redis runtime settings version publication 和本地 singleton refresh；发布 probe version 需要 `--execute`。 |
| `check_technical_spec_compliance.py` | 检查实现/报告证据是否满足技术白皮书强不变量。 |
| `run_bayes_chain_acceptance.py` | 运行真实 Bayes corpus search 和 QA acceptance。 |
| `docker_smoke.py` | HTTP smoke，覆盖 health、graph stats、graph layers、retrieval trace、QA 和 context package endpoints。 |

## 产品定位

脚本是派生状态修复、白皮书合规、真实语料验收和 Docker smoke 的运维入口。写数据脚本必须 dry-run 或显式 `--execute`；破坏性脚本必须打印目标对象和影响范围，并要求确认 flag。脚本从宿主机运行时会使用宿主可达的模型桥地址，容器内运行时使用容器网络地址。

## 技术栈

| 范围 | 技术 |
| --- | --- |
| 执行 | Python scripts |
| 数据 | PostgreSQL source of truth, Qdrant, BM25 records, Redis |
| 报告 | JSON reports under `output/` |
| 验收 | pytest fixture, HTTP smoke, real corpus acceptance |

## 主链路

```text
diagnose / evaluate / reconcile
-> read PostgreSQL facts
-> inspect derived Qdrant/BM25/Redis state
-> emit report under output/
-> optionally repair with --execute
-> rerun diagnostics or smoke
```

## 环境配置

从仓库根目录运行时读取 `.env`。在 API 容器内运行时使用 `/app/apps/api` 工作目录和 `/app/scripts` 脚本路径。

```powershell
python scripts/diagnose_context_graph.py --knowledge-base-name 贝叶斯
docker exec course-kg-api python /app/scripts/check_technical_spec_compliance.py --knowledge-base-name 贝叶斯
```

## 快速启动

只读诊断：

```powershell
python scripts/diagnose_context_graph.py
python scripts/evaluate_layered_retrieval.py --query "Metropolis Hastings"
python scripts/check_context_package_quality.py --query "posterior prior likelihood"
```

写入修复：

```powershell
python scripts/reconcile_bm25_records.py --execute --delete-stale
python scripts/runtime_hot_reload_probe.py --execute
```

破坏性清理：

```powershell
python scripts/destroy_legacy_derived_data.py --execute --confirm-destroy-legacy
```

## 参数列表

| 分类 | 参数 |
| --- | --- |
| 目标选择 | `--knowledge-base-id`, `--knowledge-base-name`, `--query`, `--base-url` |
| 写入控制 | `--execute`, `--dry-run`, `--confirm-destroy-legacy`, `--delete-stale`, `--delete-inactive-chunks` |
| Docker smoke | `--base-url`, `--worker-container`, `--knowledge-base-id`, `--qa-timeout-seconds` |
| 报告 | `--output-dir` 或默认 `output/` |

## 验证

脚本变更后至少运行相关只读入口：

```powershell
python scripts/diagnose_context_graph.py --knowledge-base-name 贝叶斯
python scripts/check_technical_spec_compliance.py --knowledge-base-name 贝叶斯
```

Docker 集成：

```powershell
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api --worker-container course-kg-worker
```

## 运维测试

Bayes 全链路验收常用组合：

```powershell
python scripts/destroy_legacy_derived_data.py --knowledge-base-name 贝叶斯 --execute --confirm-destroy-legacy --delete-inactive-chunks
python scripts/reconcile_vector_records.py --knowledge-base-name 贝叶斯
python scripts/reconcile_bm25_records.py --knowledge-base-name 贝叶斯
python scripts/evaluate_layered_retrieval.py --knowledge-base-name 贝叶斯 --query "Metropolis Hastings"
python scripts/evaluate_agent_trace.py --knowledge-base-name 贝叶斯
python scripts/run_bayes_chain_acceptance.py --base-url http://127.0.0.1:8000/api
```

## 文档

- [../README.md](../README.md)：仓库总览。
- [../docs/technical-spec.md](../docs/technical-spec.md)：技术白皮书。
- [../apps/api/README.md](../apps/api/README.md)：API service 和数据模型。
- [../infra/README.md](../infra/README.md)：Docker Compose 环境。

## 边界

- PostgreSQL 是事实源；Qdrant、BM25、Redis 只能对账或重建。
- 写数据脚本默认 dry-run 或要求 `--execute`。
- 破坏性 legacy cleanup 必须同时要求 `--execute` 和确认 flag。
- 报告必须包含目标知识库、写入模式、影响范围和可审计计数。
- RQ missing edge type 只能记录 diagnostics，不允许用 fallback pair 补边。
- Mid/Coarse 重建脚本必须保持 RQ L3/L2 投影边界。
- 不新增依赖个人绝对路径、未声明服务或本地隐藏状态的脚本。

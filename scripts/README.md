# 运维脚本

`scripts` 保存 SymboGraph 的重建、对账、诊断、检索评估、质量检查和 Docker smoke 入口。脚本从仓库根目录运行，也适配 API 容器内的 `/app/scripts` 路径。

## 通用规则

- 写数据脚本提供 dry-run 或要求显式 `--execute`。
- 破坏性清理必须打印目标对象和影响范围。
- 诊断、benchmark、smoke 输出和验收报告写入 `output/`。
- 脚本使用 PostgreSQL 作为事实源修复 Qdrant、BM25 和 Redis 派生状态。

## 脚本列表

| 脚本 | 职责 |
| --- | --- |
| `rebuild_chunks.py` | 从 source files 重建固定 token chunks、结构图、contextual embeddings、BM25 和四层图谱。写入需要 `--execute`；重解析已有资料需要 `--full-reparse`。 |
| `rebuild_structure_graph.py` | 重建 structure nodes、edges、mappings 和 coordinates。写入需要 `--execute`；重解析已有资料需要 `--full-reparse`。 |
| `rebuild_chunk_relation_graph.py` | 重建 chunk relation graph，并刷新依赖的 fine/mid/coarse/context states。写入需要 `--execute`。 |
| `rebuild_mid_concept_graph.py` | 重建 LLM-grounded mid concepts 和依赖状态。写入需要 `--execute`。 |
| `rebuild_coarse_concept_graph.py` | 重建 coarse concepts 和 active context graph state。写入需要 `--execute`。 |
| `rebuild_context_graph_all.py` | 从当前 chunks 重建所有四层图谱派生状态。写入需要 `--execute`。 |
| `reconcile_vector_records.py` | 对账 PostgreSQL `vector_records` 与 Qdrant points。 |
| `reconcile_bm25_records.py` | 对账 `bm25_records` 与当前 chunks/contextual texts；`--execute` 创建缺失记录，`--delete-stale` 删除过期记录。 |
| `cleanup_stale_data.py` | 清理过期 Qdrant/vector/BM25 状态。默认 dry-run；`--execute --delete-inactive-chunks` 删除 inactive chunks、inactive document versions 和过期 chunk_versions。 |
| `diagnose_context_graph.py` | 输出 counts、freshness、grounding、retrieval contribution 和四层采样 payload。 |
| `evaluate_layered_retrieval.py` | 执行 layered retrieval 查询并输出 trace 诊断。 |
| `check_context_package_quality.py` | 检查 context package closure、citation spans、graph paths 和 token budget metadata。 |
| `check_runtime_settings_contract.py` | 检查 runtime settings 生命周期、热加载和配置边界。 |
| `run_bayes_chain_acceptance.py` | 执行真实资料链路验收。 |
| `docker_smoke.py` | HTTP smoke，覆盖 health、graph stats、四层图谱、retrieval trace、QA 和 context package endpoints。 |

## 示例

```powershell
python scripts/diagnose_context_graph.py
python scripts/evaluate_layered_retrieval.py
python scripts/check_context_package_quality.py
python scripts/reconcile_vector_records.py
python scripts/reconcile_bm25_records.py --execute
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api --worker-container course-kg-worker
```

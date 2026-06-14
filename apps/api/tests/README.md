# API 测试

`apps/api/tests` 覆盖 Four-Layer Context Graph RAG 的后端行为。

## 覆盖范围

- 固定 token chunk，保护表格、公式、代码块、标题和图注。
- Chunk Structure Graph、坐标和映射记录。
- Contextual embedding records、BM25 records 和 vector lifecycle metadata。
- Chunk Relation Graph、Fine Clusters、RQ-KMeans、Mid Concepts、Coarse Concepts 和 active context graph state。
- Layered retrieval traces 和 context packages。
- Grounded QA、Agent trace 和 citation verification。
- 四层 schema migration。
- Runtime settings、ingestion logs、maintenance reconciliation 和 Docker smoke 相关路径。

## 运行

从 `apps/api` 执行：

```powershell
python -m pytest tests
```

从仓库根目录的 Docker 栈执行：

```powershell
docker exec -w /app/apps/api course-kg-api python -m pytest tests
```

## 约束

- Bug fix 默认补回归测试。
- 数据、检索、embedding、四层图谱、概念图或导入链路变化时，同步维护脚本与 smoke。
- 生成的测试日志、诊断和验收报告写入 `output/`。

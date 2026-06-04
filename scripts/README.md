# Scripts 维护说明

本目录只保留面向当前 Docker 架构的维护脚本。默认运行口径是：在 Docker 服务已启动的情况下执行，使用容器内系统 Python，不创建虚拟环境，模型和数据库 fallback 均关闭。

## 运行环境

推荐从项目根目录执行：

```powershell
docker exec course-kg-api python /app/scripts/quality_gate.py --course-name "课程名称"
```

涉及数据库、Qdrant 或模型 API 的脚本应在 `course-kg-api` 容器内运行。容器通过 compose 注入 `.env`，脚本不读取或输出真实密钥值。

## 脚本清单

| 脚本 | 用途 | 是否写数据 |
| --- | --- | --- |
| `quality_gate.py` | 检查 active chunks、Qdrant points、零向量、孤儿向量、父子块健康状态 | 可选删除 orphan zero vectors |
| `analyze_chunk_quality.py` | 统计 chunk 类型、长度、表格/公式、embedding version、重复内容 | 否 |
| `cleanup_aliases.py` | 清理并删除重复、冲突或失效的实体别名，维护别名表一致性 | 是 |
| `reembed_all_chunks.py` | 只修复零向量 chunks，使用当前 contextual embedding input | 是 |
| `reembed_with_enhancement.py` | 按当前 embedding input 重嵌入 active chunks，并更新 embedding version | 是 |
| `reingest_all_courses.py` | 通过正式 ingestion pipeline 重解析课程 storage 文件；已有 active chunks 时按 full reparse 规则推进 chunk 版本 | 是 |
| `repair_parent_child_links.py` | 修复历史 child chunks 缺失的 `parent_chunk_id`，可选择重嵌入 | 是 |
| `tune_graph_hyperparameters.py` | 对单门课程的 LLM/embedding 组合运行本地 TPE 图谱超参标定，报告写入 `output/` | dry-run 否，否则是 |
| `docker_smoke.py` | 通过 HTTP API 创建临时课程，验证上传、解析、检索、问答链路 | 是，结束时清理 |
| `evaluate_evidence_first_retrieval.py` | 评估并基准测试 evidence-first 混合检索性能与召回精度 | 输出报告于 `output/` |
| `evaluate_existing_quality.py` | 使用 judge model 对现有课程 search/QA 做轻量质量评估 | 输出报告于 `output/` |
| `evaluate_quality_decisions.py` | 针对切分质量和决策的合理性进行自动化打分评估 | 输出报告于 `output/` |

## 常用命令

检查单门课 DB/Qdrant 健康：
```powershell
docker exec course-kg-api python /app/scripts/quality_gate.py --course-name "课程名称"
```

分析单门课切块质量：
```powershell
docker exec course-kg-api python /app/scripts/analyze_chunk_quality.py --course-name "课程名称"
```

清理失效别名：
```powershell
docker exec course-kg-api python /app/scripts/cleanup_aliases.py --course-name "课程名称"
```

预览重嵌入影响：
```powershell
docker exec course-kg-api python /app/scripts/reembed_with_enhancement.py --course-name "课程名称" --dry-run
```

重解析单门课并清理旧数据：
```powershell
docker exec course-kg-api python /app/scripts/reingest_all_courses.py --course-name "课程名称" --cleanup-stale
```

该脚本不会绕过正式导入语义：空知识库首次解析创建 v1，已有 active chunks 时才启用 full reparse 并以当前最高 chunk 版本 + 1 为目标版本。

运行端到端 smoke 测试：
```powershell
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api
```

评估 evidence-first 混合检索性能：
```powershell
docker exec course-kg-api python /app/scripts/evaluate_evidence_first_retrieval.py --course-name "课程名称"
```

评估切分质量决策：
```powershell
docker exec course-kg-api python /app/scripts/evaluate_quality_decisions.py --course-name "课程名称"
```

## 安全约束

- 不在脚本中硬编码本机绝对路径、个人用户名、课程数据文件名或真实密钥。
- 不从宿主机绕过 Docker 直接写 Qdrant 或数据库。
- 不启用 `ENABLE_MODEL_FALLBACK` 或 `ENABLE_DATABASE_FALLBACK`。
- 写数据脚本必须提供可读输出，说明处理课程、处理数量和失败原因。
- 质量门禁失败时应直接退出非零状态，不能静默降级。

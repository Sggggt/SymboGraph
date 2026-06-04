# API Tests 维护说明

本目录覆盖 API 后端的核心行为、数据一致性、检索质量门禁和 no-fallback 运行路径。默认测试应可在 `course-kg-api` 容器内用系统 Python 运行，不依赖虚拟环境。

## 默认运行

```powershell
docker exec course-kg-api python -m pytest
```

默认 `pytest` 排除以下两类测试：

- `fallback_compat`：只用于显式兼容覆盖，不进入 no-fallback 验收路径。
- `no_fallback_e2e`：会调用真实模型 API、PostgreSQL 和 Qdrant，需要显式开启。

## 关键测试集

| 文件 | 覆盖范围 |
| --- | --- |
| `test_retrieval.py` | BM25、dense search、weighted fusion、rerank、零向量告警、child-to-parent context |
| `test_enhanced_chunking.py` | 父子切块、上下文增强 embedding input、表格/公式 metadata |
| `test_p0_graph_security.py` | graph-enhanced search、multi-hop 路由、API key middleware |
| `test_p0_consistency.py` | ingestion 事务一致性、向量写入补偿、重复文档处理 |
| `test_maintenance.py` | inactive 数据、孤儿向量、图谱引用和课程删除清理 |
| `test_runtime_settings.py` | `.env`/`.env.example` 参数名一致性、运行时检查、热更新 |
| `test_embeddings.py` | OpenAI-compatible embedding/chat 请求、零向量拒绝、错误重试策略 |
| `test_ingestion_batches.py` | chunk 版本推进、全量重新解析、取消补偿、batch phase/heartbeat 和上传解析契约 |
| `test_parsers.py` | Markdown、HTML、Notebook、PDF、DOCX、PPTX 解析和 fallback 禁用行为 |
| `test_agent_graph.py` | Agent 路由、检索问答、trace 流式事件 |
| `test_concept_graph.py` | 概念抽取、图谱合并、章节归一化和模型输出容错 |

## 显式测试命令

检索、图谱和切块核心测试：

```powershell
docker exec course-kg-api python -m pytest tests/test_retrieval.py tests/test_p0_graph_security.py tests/test_enhanced_chunking.py
```

运行配置、维护和模型请求单测：

```powershell
docker exec course-kg-api python -m pytest tests/test_runtime_settings.py tests/test_maintenance.py tests/test_embeddings.py tests/test_ingestion_batches.py
```

显式运行 fallback 兼容测试：

```powershell
docker exec course-kg-api python -m pytest -m fallback_compat
```

显式运行真实 no-fallback E2E：

```powershell
docker exec -e RUN_NO_FALLBACK_E2E=1 course-kg-api python -m pytest -m no_fallback_e2e
```

## 编写测试的约束

- 默认测试不调用真实外部模型；真实 API 调用必须使用 marker 或环境变量显式开启。
- 默认测试必须保持 `ENABLE_MODEL_FALLBACK=false` 和 `ENABLE_DATABASE_FALLBACK=false`。
- 不在测试中写入本机绝对路径、个人用户名、真实课程数据文件名或真实密钥。
- 需要文件系统时使用 `tmp_path` 或容器内测试数据。
- 需要数据库时使用 `db_session` fixture；测试结束应清理创建的课程、chunks、向量和会话数据。
- 检索相关断言应覆盖 `model_audit`、`parent_chunk_id`、`parent_content`、`retrieval_granularity` 和零向量保护。

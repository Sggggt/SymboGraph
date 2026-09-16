# 后端测试

正式回归使用公开合成数据，覆盖功能、事务、来源、版本、取消及审计。私有资料只用于明确的本地验收，不复制到 fixture。

## 运行

```powershell
docker exec course-kg-api python -m pytest tests
```

纯单元测试可从隔离环境的 `apps/api` 目录运行。默认 pytest 配置排除 `fallback_compat` 和 `no_fallback_e2e`，因此必须报告跳过/未执行范围；不能把默认套件通过当作真实外部服务验收。

## 按改动选择测试

| 范围 | 代表性文件或前缀 |
|---|---|
| 解析/结构/版本 | `test_parser_*`、`test_pdf_*`、`test_ingestion_*`、`test_storage_*` |
| TPE/RQ/图 | `test_auto_tpe.py`、`test_graph_*`、`test_rq_*`、`test_context_graph_pipeline.py` |
| 目标检索/范围/来源 | `test_intent_*`、`test_layered_execution_v1.py`、`test_entry_ranking.py`、`test_lexical_*`、`test_evidence_scope.py`、`test_source_*`、`test_scope_*` |
| 回答/会话/SSE/历史重放 | `test_answer_sources*`、`test_conversation_state.py`、`test_agent_direct_answer.py`、`test_qa_admission_timing.py`、`test_agent_admission.py`、`test_reflection_*` |
| 配置/缓存/恢复 | `test_runtime_*`、`test_*runtime_settings*`、`test_cache_manager.py`、`test_qdrant_*` |
| 脚本/发布 | `test_script_write_gates.py`、`test_*scripts.py`、`test_locked_runtime_image_contract.py`、迁移相关测试 |

目标协议回归已覆盖 Intent/ExecutionStrategy、coarse/mid/chunk 根入口、空词面、Dense/RQ/BM25 独立提名与图内通道保留、同一 chunk 多路径后的稳定结果去重、索引发布、缓存身份与失效；包括负权重、空分词、零命中、不同统计域、跨语言、同分和父节点投影。历史测试仍按 fixture 显式隔离。

目标 QA 验证确定性来源准入后一次生成、无额外结果判定调用、部分回答/不足与技术终态分列。SSE 回归验证保活注释不进入 trace、响应禁止缓存/缓冲、观察者关闭不取消 owner、显式取消才结束任务并释放租约，以及失败/取消会话状态只由最新 run 收敛。会话回归还覆盖 run 接纳后立即持久化用户问题、终态配对、空检索会话不进入问答历史，旧来源重放失败时历史仍可列出、阅读并继续正式检索，以及逆序多来源 UUID 必须先 canonicalize 再校验。历史协议测试保护已有记录可读；不能为保持旧测试通过而把旧执行机制带回新入口。真实 LLM 调权与全局覆盖需独立验收，mock 计划通过只能证明执行契约。

生成提示回归固定检查 GFM 与 `$...$`/`$$...$$` LaTeX 定界符要求；前端 Markdown 回归同时覆盖标题、列表、强调、表格和行内/块 KaTeX。提示存在和组件通过不能证明外部模型每次都遵循格式，真实生成仍按成功、schema 失败和上游技术失败分列。

Web 会话生命周期回归包含列表缓存项在读取时返回 404：客户端须保持原 activeSessionId、移除过期项、刷新查询并显示自然语言提示；该 Promise 不得成为浏览器 unhandled rejection，真实失败/取消提示也不能被随后历史水合清除。

改完先跑受影响测试，通过后按风险补充集成检查。重复测试需有新改动或未解决问题作为理由。写数据或调用真实模型的检查先给出计划和目标。

结果只暂存 `output/`，核验后将必要结论写入 Git 忽略的本地验收记录。不把历史测试数量相加伪装为当前完整套件通过。

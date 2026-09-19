# API 后端

FastAPI 负责资料库、导入、四层图、检索、QA 和运行配置。Worker 调用同一 service 层，不维护第二套业务实现。

## 入口

| 文件或模块 | 职责 |
|---|---|
| `app/main.py`、`app/api.py`、`app/routers/` | 应用启动、中间件和路由 |
| `app/models.py`、`app/schemas.py`、`migrations/` | 数据、请求/响应、迁移与约束 |
| `app/core/config.py`、`app/core/runtime_config.py`、`services/runtime_settings.py` | 双文件配置解析、原子更新和三级生命周期 |
| `services/ingestion.py`、`parsers.py`、`source_parse_pipeline.py` | 文件解析、版本和补偿 |
| `services/context_graph.py`、`auto_tpe.py`、`graph_build_workspace.py` | 四层图和构建计算 |
| `services/intent_execution_agent.py`、`intent_planning.py`、`layered_execution_v1.py` | 当前 Search/QA/SSE 规划、图执行与终态 |
| `services/retrieval_agent.py`、`retrieval_fsm.py`、`reflection_*` | 历史记录重放与兼容测试，不是目标 serving 主链 |
| `services/evidence_scope.py`、`source_location.py`、`source_use.py` | 原文范围、位置和用途 |
| `services/answer_sources.py`、`citation_provenance.py`、`agent_pe_audit.py` | 来源绑定与审计 |
| `intent_contracts.py`、`services/entry_ranking.py` | 当前意图/策略契约、RQ 重构与 `weighted_rrf_entry_v2` 图入口融合 |
| `services/lexical_index.py`、`lexical_storage.py` | 原文 BM25、快照重放、候选与发布事务 |
| `services/qa_performance.py`、`build_performance.py` | 阶段计时 |
| `services/qdrant_outbox.py`、`maintenance.py` | 派生状态、恢复与维护 |

目标协议已成为当前运行入口：LLM 在一次规划中选择三个根入口并返回 `intent_execution_strategy_v2`；开启双语词面时，可翻译概念组同时携带中英文 surface，标识符和数值保持 neutral。空词面使用 Dense-only，混合计划按层执行 Dense/RQ/BM25 独立提名和图路径遍历。Context Package 经确定性来源准入后只生成一次回答。

一次回答使用 `grounded_markdown_inline_citations_v1` 原生文本流。模型直接生成最终 GFM，不再返回外层 JSON，并可用 `⟦cite:src_n⟧` 标注原文来源；合法标记在流中确定性转换为序号链接，前端显示为浅灰胶囊，非法或未知标记原样输出而不使回答失败。标题、列表、表格、代码块和 `$...$`/`$$...$$` 都是最终正文的一部分，不经过完成后正文替换。正文完成后，服务端把本轮实际送入模型的已准入 Context Package 来源绑定到完整答案跨度，并单独发送引用列表。

完整语义见[技术白皮书](../../docs/technical-spec.md)。`reflection_*` 中仍有当前来源恢复/重放依赖，兼容边界见[说明](../../docs/reference/compatibility.md)。

FastAPI 路由已注册不等于当前产品 Web 正在调用。接口变更时同步核对 App Router 可达组件、`api.ts`、router、scripts 和兼容调用方。

## 运行

使用根启动器或现有 Compose project。API 容器工作目录为 `/app/apps/api`；脚本在 `/app/scripts`，原文数据在 `/app/data`，共享配置绑定为 `/workspace/.env` 与 `/workspace/settings.json`。

根 `.env` 只维护秘密、连接和服务启动字段；根 `settings.json` 只维护非秘密运行、检索与构建字段。两者键集合不相交，不创建 `apps/api/.env`、模块级 JSON 或数据库 active/desired 参数副本。迁移和核对入口见[脚本说明](../../scripts/README.md)。

Alembic 保留上一推送版本已有的 baseline、`20260822_0043` 和 `20260824_0044`，当前版本只增加 `20260916_0045_upgrade_from_v7_1_2.py`。该迁移从上一推送 schema 一次创建来源绑定、保留/扩展、lexical policy/reward 与 BM25 生命周期表，并收敛既有约束和索引；不依赖已删除的中间开发 revision。

```powershell
docker exec course-kg-api python -m pytest tests
docker exec course-kg-api python /app/scripts/check_release_schema.py --help
```

后端支持 Python 3.11 以上，Docker 镜像使用锁定的 Python 3.13 环境和 `uv.lock`。纯单元测试可以使用隔离开发环境；生产形态 PostgreSQL/Qdrant/Redis/模型集成须在 Docker 内执行。

## 修改与验证

更新行为时同步 schema、shared types、相关 scripts 与 tests。来源写入保持显式事务；外部副作用先保存意图。出错时保留安全原因，不能吞掉关键审计或启用 fallback 伪装成功。

`/qa/stream` 在长规划、图检索和生成等待期间发送 10 秒间隔的 SSE 注释保活，并关闭代理缓冲。保活帧不进入 Agent trace。请求接纳后由独立 run owner 持有数据库会话与租约，SSE 连接只是观察者；页面离开、刷新或网络断线只结束观察，不取消执行。显式取消才会设置取消信号。目标生成使用 provider 原生文本 delta，并直接发布 append-only 可见正文；正文结束后单独发送由实际 Context Package 派生的引用列表与 final，不发送 `answer_replace`。首个可见增量进入 `first_response_ms`/`first_token_ms`。所有完成、失败和取消终态都持久化，客户端可按 run 恢复。

问答请求接纳并创建 run 后，先将用户问题写入 PostgreSQL 会话，再开始规划、检索或模型调用。成功时追加回答；失败或取消时追加安全终态文案，因此已接纳的问题不会因执行异常丢失。会话列表逐条校验当前公共 schema，只返回兼容会话；一个旧协议或损坏会话只产生安全排除计数，不能使整个列表返回 500。被排除记录不删除、不改写，按 id 读取返回 409。历史来源的严格物理重放仍在 `verified_context_reuse` 前执行，失败时禁用复用并进入新的正式图检索。

显式删除会话会移除 transcript/state，并把关联 run 与 AnswerSession 的 session 外键置空；run、来源绑定、来源准入 observation 和其他审计事实不随 UI 会话删除。PostgreSQL 路径使用单条 `DELETE ... RETURNING`，由两个 `ON DELETE SET NULL` 外键在同一语句内解绑；SQLite 测试/兼容路径显式执行等价更新。

图遍历允许同一 chunk 从不同根路径到达；公开 Search/QA 结果在 `top_k` 前按稳定路径优先级去重，并记录原始路径候选、唯一节点和重复路径数量。`SearchResponse` 对重复 `chunk_id` fail closed，前端可以安全使用 chunk id 作为组件身份。

在线请求使用有界图准入核对 active 指针、协议、freshness 与各层精确计数；构建、promotion、reconcile 和质量验收继续使用逐项深度准入。多 requirement 的 BM25 共享一次原文域/统计核验和联合 postings 读取，但每个 requirement 与完整查询分别计分、排序和截断，结果与逐视图执行等价。

新增耗时阶段先加入强类型 allowlist。新模型调用先保存 prepared 意图，完成后才写 completed；技术超时与证据不足分开。测试分组见 [tests](tests/README.md) 和[开发说明](../../docs/development.md)。

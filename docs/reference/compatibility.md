# 历史兼容边界

目标主链见[检索协议](retrieval-and-qa.md)。本文只列必要的数据读取边界，不展开已退出的算法、流程、预算或目标函数。

| 旧对象或字段族 | 保留用途 | 新协议边界 |
|---|---|---|
| `retrieval_granularity` 及旧 coarse/mid 请求 | 重放旧请求与 UI 历史 | 新请求不接收或默默映射为 LLM 策略；旧接口必须显式版本隔离 |
| 旧 Task/词面规划与 `retrieval_fsm_v1` | 保持原字段、hash 与事件顺序 | 新 run 使用 Intent/ExecutionStrategy 和新状态版本 |
| `intent_execution_strategy_v1.lexical_terms` | 只读重放已经持久化的目标主链早期计划 | 新 run 使用 `intent_execution_strategy_v2` 的双语 lexical groups；不得把旧 term 数组补写成伪造语言或翻译事实 |
| `retrieval_evidence_sufficiency_v1`、`source_addressed_assessment_v1`、旧检索门禁与修正记录 | 只读重放已有调用、来源和判决 | 不作为新来源准入字段或恢复执行步骤 |
| `agent_answer_reflection_v1`、`CitationVerification` 及旧语义验证 | 显示历史回答与引用 | 不从新请求触发旧调用 |
| reward、policy、posterior、cycle reward 字段族 | 核验旧审计与必要迁移 | 新请求不读取策略先验，不产生更新，不参与排序或缓存身份 |
| evidence atom、signal graph 与旧 lexical/BM25 产物 | 受控迁移和清理 | 新 `source_chunk_bm25_v1` 独立构建和核验，不靠旧产物提供正确性 |

旧记录按当时协议、原文 manifest、参数和来源重放，不重跑模型，不用新结果回写旧判决。缺失、篡改、跨库/会话或版本不一致时明确失败。暂停后的恢复点不能自动续跑已退出阶段，应转为明确的需迁移状态。

## 共享模块

模块名不能决定删除范围。`reflection_context.py`、`reflection_sources.py`、`reflection_expansion.py` 等仍可能被原文恢复、不可变谱系和来源绑定引用。重构时先核对调用和外键，再迁移共享能力；保留历史读取所需的完整校验。

历史构造用 test-only 入口不得开放给新请求。新字段不能回填到旧记录中伪造协议通过；旧配置可为读取兼容保留，但不进入新运行参数表和执行契约。

## 迁移与删除

版本化迁移应逐项列出数据、接口、缓存、脚本与历史读取影响。删除或移动前盘点引用、给出 dry-run/目标/影响，再用显式执行 flag 对账。不能因新架构暂不使用某张表就直接删除共享记录或数据卷。

私有验收输入和本地状态记录留在忽略目录，旧临时产物不成为产品运行依赖。

# SymboGraph 实验报告与使用建议

**更新日期**：2026-05-21  
**报告目的**：记录当前中英文全量实验结果，统一对比 S1/S2/S3/S4 的检索、问答与图谱质量，并给出系统优劣势、cross-encoder 开关建议和用户侧使用建议。

## 1. 数据来源与口径

本报告只采用以下指定 run 作为正式数据源：

| 数据项 | 中文数据源 | 英文数据源 |
| --- | --- | --- |
| S1-S4 检索 | `comparative_experiment/data/reports/full_zh_no_crossencoder_20260520` | `comparative_experiment/data/reports/full_en_lightrerank_s1s4_20260518` |
| S1/S2/S3 QA | `comparative_experiment/data/reports/full_zh_no_crossencoder_20260520` | `comparative_experiment/data/reports/full_en_lightrerank_s1s4_20260518` |
| S4 QA | `comparative_experiment/data/reports/full_zh_no_crossencoder_20260520` | `comparative_experiment/data/reports/full_en_fair_graphqa_s3s4_20260519` |
| S1 图谱评测 | `comparative_experiment/data/reports/full_zh_no_crossencoder_20260520` | `comparative_experiment/data/reports/full_en_lightrerank_s1s4_20260518` |
| S3/S4 图谱评测 | `comparative_experiment/data/reports/full_zh_no_crossencoder_20260520` | `comparative_experiment/data/reports/full_en_fair_graphqa_s3s4_20260519` |

颜色约定：**蓝色 = 中文 ZH**，**橙色 = 英文 EN**。中英文不再拆成两套叙述，而是在同一指标下并列展示。

系统定义：

| 编号 | 系统 | 定位 |
| --- | --- | --- |
| S1 | SymboGraph-Full / SymboGraph Full | 完整系统：混合召回、轻量或 cross-encoder rerank、parent context、evidence-first agent、课程图谱增强 |
| S2 | SymboGraph-NoGraph | 无图基线：保留 dense + lexical + rerank + QA，但不使用知识图谱 |
| S3 | LightRAG | 轻量 GraphRAG 基线 |
| S4 | MS-GraphRAG | Microsoft GraphRAG 风格基线 |

关键限制：

- 中文 run `full_zh_no_crossencoder_20260520` 明确关闭 cross-encoder；S1 raw audit 显示 `reranker_enabled=false`。报告里的 `RerankRate` 包含 lightweight rerank，不等同于 cross-encoder 调用率。
- S2 NoGraph 适配器存在历史审计字段硬编码问题，可能显示 `reranker_enabled=true`；实际路径只做 term-overlap lightweight rerank，不调用 cross-encoder。
- S1 英文 QA 在 `full_en_lightrerank_s1s4_20260518` 中为 32 条样本，其余系统多数为 33 条；跨系统比较时要注意样本数差异。
- S3/S4 的输出协议和 S1/S2 不完全等价，尤其 retrieval 可能更接近图摘要或 answer-like context，而不是严格 chunk list。因此 S3/S4 分数应作为系统行为参考，不宜直接解释成同构检索精度。
- Graph 评测受语言、术语对齐、节点粒度、边定义和 evidence 约束影响。跨语言趋势可参考，但不能当作严格学术结论。

## 2. 检索结果

### 2.1 Overall Quality

![Retrieval overall](assets/experiment-report/retrieval-overall.svg)

| 系统 | ZH Overall | EN Overall | ZH HalluRate | EN HalluRate | ZH Latency ms | EN Latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 Full | 4.13 | 4.19 | 0.02 | 0.02 | 2244.5 | 3899.4 |
| S2 NoGraph | 3.20 | 2.68 | 0.04 | 0.00 | 1447.8 | 1702.3 |
| S3 LightRAG | 4.21 | 4.31 | 0.33 | 0.30 | 2012.4 | 7152.6 |
| S4 MS-GraphRAG | 4.24 | 4.40 | 0.33 | 0.29 | 32571.7 | 35549.9 |

### 2.2 Precision / Noise Control

![Retrieval precision](assets/experiment-report/retrieval-precision.svg)

| 系统 | ZH Precision | EN Precision | 解释 |
| --- | ---: | ---: | --- |
| S1 Full | 3.33 | 3.38 | 覆盖率和证据支撑强，但候选中仍有冗余、相邻概念和 parent context 噪声 |
| S2 NoGraph | 2.49 | 1.94 | 无图结构约束，复杂问题更依赖大上下文拼接，噪声控制较弱 |
| S3 LightRAG | 4.26 | 4.37 | judge 认为答案/摘要聚焦度较高，但其 retrieval 输出和 S1/S2 chunk retrieval 不完全等价 |
| S4 MS-GraphRAG | 4.44 | 4.60 | retrieval summary 聚焦度高，但延迟显著更高，且后续 QA 幻觉率偏高 |

检索结论：

- S1 的优势是课程 grounded、evidence coverage 和 citation support 稳定；短板是 precision/noise control 不够高。其策略偏召回优先，适合 QA，但会牺牲候选列表纯净度。
- S2 的检索质量明显低于 S1，尤其英文 run 中整体质量只有 2.68，说明图谱和 evidence-first 编排对检索判断有收益。
- S3/S4 检索 overall 高，但 hallucination rate 也高，并且与 S1/S2 的 chunk-based retrieval 协议不完全一致。因此不能只看 retrieval overall 判断它们更适合课程问答。

## 3. QA 结果

### 3.1 Overall Quality

![QA overall](assets/experiment-report/qa-overall.svg)

| 系统 | ZH QA Overall | EN QA Overall | ZH HalluRate | EN HalluRate | ZH Latency ms | EN Latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 Full | 5.00 | 5.00 | 0.00 | 0.00 | 39402.2 | 38187.0 |
| S2 NoGraph | 5.00 | 4.82 | 0.00 | 0.09 | 14099.5 | 15692.8 |
| S3 LightRAG | 4.00 | 3.55 | 0.30 | 0.42 | 1351.5 | 1365.1 |
| S4 MS-GraphRAG | 3.24 | 3.36 | 0.67 | 0.58 | 33450.4 | 33124.2 |

### 3.2 Hallucination Rate

![QA hallucination](assets/experiment-report/qa-hallucination.svg)

### 3.3 按题型拆分

| 系统 | ZH Simple | ZH Complex | ZH Open | EN Simple | EN Complex | EN Open |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 Full | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 |
| S2 NoGraph | 5.00 | 5.00 | 5.00 | 5.00 | 4.56 | 4.78 |
| S3 LightRAG | 4.27 | 4.33 | 3.22 | 3.93 | 4.33 | 2.11 |
| S4 MS-GraphRAG | 3.67 | 3.67 | 2.11 | 3.60 | 4.11 | 2.22 |

QA 结论：

- S1 在中英文 QA 中均为 5.00，且幻觉率为 0。它的高分主要来自 evidence-first QA、parent context、课程证据约束和更稳的检索编排，而不是单靠图谱。
- S2 中文 QA 出现分数饱和，simple/complex/open 都为 5.00；但英文 run 中 complex 和 open 已低于 S1，且检索质量显著弱于 S1。说明现有中文 QA judge 对 S1/S2 的复杂差异区分度不足，需要 hard-case 和 pairwise 评测补充。
- S3/S4 的 QA 明显低于 S1/S2。核心原因是它们偏 GraphRAG 摘要与图查询，弱于 evidence-first 课程问答；开放题更容易引入通用知识、弱证据或图摘要幻觉。
- S4 英文 QA 使用 `full_en_fair_graphqa_s3s4_20260519` 的修正结果；旧 run 中 S4 QA=1.00 的结果不作为正式结论。

## 4. 图谱评测

S2 是无图基线，不参与图谱质量比较。图谱评价重点看覆盖、关系质量、噪声控制、证据支撑、结构可用性和幻觉标记。

### 4.1 Graph Overall

![Graph overall](assets/experiment-report/graph-overall.svg)

| 系统 | ZH Graph Overall | EN Graph Overall | ZH HalluRate | EN HalluRate |
| --- | ---: | ---: | ---: | ---: |
| S1 Full | 2.33 | 2.67 | 0.67 | 0.67 |
| S3 LightRAG | 2.33 | 2.33 | 0.00 | 0.33 |
| S4 MS-GraphRAG | 2.67 | 2.00 | 0.33 | 0.33 |

### 4.2 Graph Precision / Noise Control

![Graph precision](assets/experiment-report/graph-precision.svg)

| 系统 | ZH Precision | EN Precision | 图谱形态 |
| --- | ---: | ---: | --- |
| S1 Full | 2.33 | 2.00 | 图谱较克制，但有重复节点、弱证据边和部分 hallucination/unsupported 标记 |
| S3 LightRAG | 1.33 | 1.33 | 节点规模膨胀，泛化实体和碎片化组件较多 |
| S4 MS-GraphRAG | 1.67 | 1.67 | 覆盖面较广，但边密度高，社区/摘要关系容易混入弱证据 |

图谱结论：

- 三个有图系统的图谱精确度都不高。图谱模块目前更适合作为检索辅助和概念导航，不适合单独作为答案事实源。
- S1 图谱相对克制，但 precision 仍只有 2.00-2.33，说明 RNN+KNN 稀疏构图、LLM 校正和 evidence 约束仍需要继续加强。
- S3/S4 图谱规模明显更大，但噪声控制较弱。GraphRAG 的强项是全局摘要和主题聚合，不是严格课程证据约束下的逐题事实回答。
- 图谱评测受语言影响：中英文术语、公式符号、缩写和课程材料语言都会影响 judge 对实体等价、关系正确性和覆盖度的判断。

## 5. 架构优势与不足

### 5.1 相比主流 GraphRAG 的优势

- **课程资料约束更强**：S1 不把图谱当成事实源，而是用课程 chunk、parent context 和 citation 约束答案。
- **QA 幻觉率更低**：S1 中英文 QA hallucination rate 均为 0，明显优于 S3/S4。
- **复杂问题证据组织更稳**：S1 的检索覆盖、citation support 和 QA 稳定性优于 S2，说明图谱和 agent 编排对复杂题有实际帮助。
- **工程可审计性更好**：S1/S2 能输出 chunk、parent context、model audit、stage latency 等元数据，更适合定位错误和做质量门禁。

### 5.2 当前不足

- **检索 precision 不够高**：S1 偏召回优先，parent context 和图谱邻居扩展会带来冗余候选。
- **图谱精确度偏低**：S1/S3/S4 的 graph precision 都低于 2.5，说明图谱构建和关系证据约束仍是主要短板。
- **QA judge 存在分数饱和**：中文 S1/S2 QA 都为 5.00，不能充分区分复杂/开放题中的真实可用性差异。
- **跨系统协议不完全等价**：S3/S4 的 retrieval 和 graph 输出形态与 S1/S2 不完全一致，直接横向比较时需要保留 caveat。

## 6. Cross-Encoder 开关建议

当前中英文正式 run 是 cross-encoder 关闭后的结果。结论不是“永远关闭更好”，而是：

| 场景 | 建议 |
| --- | --- |
| 普通课程问答、资源有限、需要较低部署复杂度 | 可以关闭 cross-encoder，使用 lightweight rerank；S1 中文 QA 仍保持 5.00/0 hallucination |
| 检索候选必须更干净、用户经常查看候选列表、需要降低噪声 | 建议开启 cross-encoder，并重点观察 S1 retrieval precision 是否提升 |
| 大批量导入后做质量评估或 benchmark | 建议分别跑 cross-encoder on/off A/B，同语言、同题集比较，不能混用中英文 run 推断开关收益 |
| 低延迟交互 | 谨慎开启 cross-encoder；应设置 batch、timeout 和降级策略，但不能静默 fallback |

本次修复记录：关闭 cross-encoder 后暴露了 `lightweight_rerank` 对 `fused=None` 的处理问题，已将分数读取改为复用 `_result_score()`，并补充回归测试 `test_lightweight_rerank_handles_none_fused_score`。

## 7. 用户使用建议

- **优先使用 S1 Full 作为课程问答主路径**：它在中英文 QA 中最稳，幻觉率最低，适合实际用户问答。
- **S2 可作为快速无图基线**：适合验证“没有图谱时能不能答”，但复杂/开放题的证据组织能力不如 S1。
- **S3/S4 更适合全局摘要与主题探索**：不建议直接作为 evidence-first 课程问答主路径，尤其不适合要求严格引用的答案。
- **不要把图谱可视化当成事实证明**：当前图谱 precision 偏低，图谱边需要结合课程 chunk 证据查看。
- **复杂/开放题要看引用质量，不只看答案流畅度**：S3/S4 往往语言流畅但 hallucination rate 高，用户应优先检查引用是否直接支撑结论。

## 8. 后续评测改进

- 增加 hard-case QA：多跳、跨章节、概念比较、反例、证明依赖、公式推导。
- 增加 pairwise judge：让 judge 在 S1/S2/S3/S4 答案之间直接比较，而不是只给单答案绝对分。
- 增加人工 gold chunks：让 retrieval precision/recall 不完全依赖 LLM judge。
- 对 graph 做实体对齐评测：中英文同义词、缩写、公式符号、重复节点合并应单独计分。
- 对 cross-encoder 做同语言 A/B：固定题集、固定模型、固定索引，只切换 reranker，才能可靠判断开关收益。

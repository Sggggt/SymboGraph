# LLM 策略与分层混合检索调研

本文为 SymboGraph 的 `intent_execution_retrieval_v1` 提供设计依据。研究范围是问题意图、图入口、空词面、Dense/RQ/BM25 融合、来源与执行成本。论文结论、官方接口能力和本项目设计判断分别说明；验证方法见[开发与测试](../development.md)。

## 研究结论

可以将“理解问题”与“选择检索方式”放在一次规划输出中，同时保持两个独立契约。问题是否全局、是否要求精确标识符、是否有明确来源，比用户手工选择普通或摘要模式更接近入口决策需要的信息。但现有研究不能证明：仅凭问题文字让 LLM 给三路权重，就一定优于固定策略。

推荐的目标协议是：保留完整任务；LLM 选择 coarse/mid/chunk、是否生成词面和逐层权重；合法空词面运行纯向量入口；混合时三个通道独立提名，通过固定融合算子选根层和下钻入口。物理图边、距离与来源校验继续由确定性执行器控制。回答只使用恢复后的实际原文包。

这里的入口排名是执行算法，不是新定义的答案质量、奖励或优化目标。采用的初始常数均为工程起点，后续效果必须通过独立材料测量；不把文献的实验提升外推到当前资料库。

## 问题意图与执行策略

Adaptive-RAG 根据问题复杂度选择不同的检索增强策略，说明不同任务需要不同处理成本。它使用训练出的分类器与特定任务标签，不能直接证明一个未经训练的通用规划提示能可靠决定本项目的图层和融合权重。可借鉴的是问题与执行成本的分离；不直接采用其训练流程。[Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/)

SymboGraph 的意图契约应覆盖 summarize、overview、define、fact_lookup、enumerate、compare、explain、procedure、analyze、relationship 和 source_lookup 等，并保留组合意图。执行策略则单独保存根入口、通道、权重和预算。意图用于解释任务，策略用于调度；二者不能压成同一个模式字段。

LLM 只能在服务器给定的可用层、索引与预算中选择。一次规划后冻结策略可以使行为可重放，但确定性执行并不意味着 LLM 的选择总是正确。因此应分别测试计划是否合法、选择是否有效以及整体答案是否达标。

## 全局问题与空词面

GraphRAG 研究直接讨论了面向整个语料的主题概括问题，并通过社区组织与预生成摘要支持全局问答。其方法包含针对社区的部分回答再汇总，评估主要针对特定规模与类型的语料；不能把论文的改善归因于“选择了高层入口”一个因素。[GraphRAG 论文](https://arxiv.org/html/2404.16130v2)

本项目保留的是层级组织和广度意识。全局问题允许不生成词面，用完整问题及来源范围构造向量输入；选择 coarse 或 mid 仍由 LLM 决定。概念只用于导航，最终事实来自原文，不引入多次社区生成过程。

纯向量解决的是“没有可靠词面也能开始检索”。它不会自动解决主题覆盖：宽泛 query 可能令多个概念得分接近，top-k 仍可能集中到少数区域。因此执行策略提供 broad 父节点分配，记录候选、展开与未观察范围；不能把非空结果说成全库完整总结。

需要验证的情形包括：问题只问资料库有什么、问题明确指定一份长报告、同一个总结任务要求不同主题、资料量超过输入容量，以及高层概念缺失但 chunk 索引可用。最后一种情形应由能力清单约束规划，不能事后默默换层。

## 混合评分方案比较

RRF 通过各检索器的名次进行融合，避免直接比较不同量纲的分数。原始论文在所用信息检索实验中报告了有效性，但其结果不是所有任务的普遍优越性证明。[RRF 原始论文](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)

Bruch 等对词面与语义融合的研究发现，在他们的实验设置中，调节后的凸组合可优于 RRF，RRF 本身也对参数敏感。这与“RRF 无需关心参数且始终最佳”的说法不符。该研究支持保留融合方法的比较，但不意味着其训练效果会迁移到第三路 RQ 信号。[融合函数分析](https://arxiv.org/html/2210.11934v2)

| 方法 | 优点 | 主要限制 | 本项目定位 |
|---|---|---|---|
| 原始分数加权 | 实现简单 | cosine、RQ 距离和 BM25 量纲不同，权重含义漂移 | 不采用 |
| Min-max 后凸组合 | 权重直接反映归一化幅度 | 候选域、极值、零方差会影响结果 | 独立比较方案 |
| DBSF | 使用每次结果分布缩放 | 小 top-k 与异常值影响统计，不能跨请求当概率 | 独立比较方案 |
| 加权 RRF | 权重不被原始分数尺度支配，易审计 | 丢失分数间距；对 k、候选深度及秩规则敏感 | 首版执行协议 |

Qdrant 官方说明同样强调融合前尺度处理，并记录 DBSF 使用返回列表的统计，而非全语料历史统计；weighted RRF 从 v1.17.0 起可用。[Qdrant Hybrid Queries](https://qdrant.tech/documentation/search/hybrid-queries/) 这表明工具可承载相关组合，但并不能替代本项目的父域投影、tie 规则和来源审计。

首版用加权 RRF 是可解释的工程选择。LLM 选择 dense/rq/bm25 权重，服务器固定候选提名、秩、平滑常数和合并规则。每次只在同一层、同一父域合并；不让不同层节点在同一全局队列中争抢预算。完整数学与空通道边界见[入口协议](retrieval-and-qa.md#候选合并与加权融合)。

## LLM 动态权重的证据边界

DAT 研究提出按查询动态融合 Dense 和 BM25，但其 LLM 会读取两路 top-1 结果并评价，再计算权重。论文报告的收益依赖这一信息条件。[DAT](https://arxiv.org/html/2503.23013v1) 本项目由检索前规划决定权重，且不采用结果驱动调权，两者不可视为等价实验。

因此允许 LLM 调权，应解释为赋予受限策略选择能力，而不是已经获得自适应最优策略。模型收到的 capability manifest 可以包含语种、层级、索引与范围信息，但不能把历史答案或未经核验的概念事实塞进其中作为答案线索。

至少需要保留三种对照：同候选预算下的固定融合、同条件的 LLM 权重融合，以及逐路关闭的对照。还应检查同义问法是否令层级与权重无理由大幅改变。采用多个观测指标，暂不定义综合目标或自动更新规则。

## RQ 能提供什么

Faiss 官方资料说明残差量化按阶段编码，重构依赖多个码本中心的加和；贪心编码具有近似性，扩大 beam 会在精度和成本间形成取舍。[Faiss Additive quantizers](https://github.com/facebookresearch/faiss/wiki/Additive-quantizers) 它并未将 RQ 地址解释为自然语言主题真值，也没有证明相同编码的片段可以互相替代。

本项目已有三层 primary chain，不需要因新增 RQ 查询通道改变持久成员数。查询关联度可由 query 到候选前缀重构向量的距离计算，coarse 使用 L2，mid/chunk 使用 L3；同层内排序后参与融合。禁止直接将原 query 与末层残差中心作完整语义比较，也不要求 query 与来源必须具有相同硬编码地址。

Dense 与 RQ 都来自同一 embedding 系统，其结果相关。第三路可能有助于组织入口，也可能只重复 Dense 的偏好；必须保留 rq=0 对照。多前缀候选用于提名，不能成为新增主链或物理边。

Hybrid Inverted Index 的研究讨论了词面与聚类索引结合，并在其密集检索设置中检查加速与稳健性。[Hybrid Inverted Index](https://aclanthology.org/2023.emnlp-main.116/) 它为“候选域组织值得独立验证”提供参考，但其索引设计与本项目四层图不同，不能以此认定 RQ 混合评分必然有效。

## BM25 的算法与工程落点

BM25 将词频饱和、文档长度和逆文档频率结合。本文采用的正 IDF 形式及 k1/b 参数含义可对应 Lucene 的公开实现说明；1.2/0.75 只是其常见默认，不构成中文混合资料的最佳参数证据。[Lucene BM25Similarity](https://lucene.apache.org/core/9_12_1/core/org/apache/lucene/search/similarities/BM25Similarity.html)

PostgreSQL 内置的 ts_rank 与 ts_rank_cd 分别基于词项频率及 cover density，并非这里规定的 BM25 公式。[PostgreSQL 排名函数](https://www.postgresql.org/docs/17/textsearch-controls.html) 因此不能把现有全文搜索函数改名后宣称已实现 BM25。初版可在现有 PostgreSQL 中持久化 postings 与统计，由 Docker 内执行器计算确定性 BM25，避免在文档阶段强制新增搜索服务。

统计域必须明确。N、df、avgdl 来自同 KB active chunk 快照；请求过滤约束候选域，但不悄悄改变 IDF。多个 KB 共享物理集合时，不能直接套用集合级统计。分词与原文位置还需同时支持中文、英文、编号、大小写敏感标识符和数字单位。

BM25 应从原文 chunk 生成。上层通过实际 membership/概念支撑投影，保存命中 chunk 和词项见证；大簇不能靠简单累加大量成员词频占优。先在 chunk top-k 截断再投影会导致候选父节点缺失，因此聚合顺序必须单独验收。

Qdrant 当前官方文本检索文档已包含新版本专属配置，例如标注 v1.19.0 的语言中性处理能力。[Qdrant Full-Text Search](https://qdrant.tech/documentation/search/text-search/full-text-search/) 仓库 Compose 固定 v1.17.1；不能照抄最新文档就宣称当前容器支持这些字段。若后续采用服务端 BM25 或 sparse 加速，应先按锁定版本验证 tokenizer、IDF、长度处理和客户端字段。[Qdrant v1.17.1](https://github.com/qdrant/qdrant/releases/tag/v1.17.1)

## 原文、预算与回答

Lost in the Middle 的实验显示，相关信息所在位置会影响被测模型使用长上下文的表现。[Lost in the Middle](https://arxiv.org/abs/2307.03172) 它支持在独立测试中改变证据位置，但不能据此解释当前某次调用的具体超时原因，也不能将结论无条件外推至所有模型。

目标架构将概念、索引和入口分数用于定位，事实只由实际包提供。来源准入检查身份、范围、支撑与容量；一次生成依据原文作答或说明不足。出处可靠不等于语义已经被证明，生成后的确定性绑定仍不能发现所有漏项或错误概括。

取消额外判定调用后，总串行成本减少，但真实效果需要重新测量。长输入成本、来源比较难度和模型输出仍然存在。输入预算必须包含提示/schema和范围指引，而非只计算 chunk 正文；不能把增大输出上限当作处理输入的加速手段。

## 独立验证设计

BEIR 跨多种数据集和任务比较检索方法，体现了在异质任务上评估泛化的必要性。[BEIR](https://arxiv.org/abs/2104.08663) 本项目应据此坚持跨问题族与文档划分，而不把一组固定问答的改进当作所有意图的完成。

| 维度 | 独立样例 | 需要检查 |
|---|---|---|
| 空词面 | “概括这个资料集涉及的方向” | 词面可空、保留任务、仅 Dense 起点、主题范围可见 |
| 精确词面 | 公开合成的参数名、接口编号 | BM25 独有候选能进入并集，标识符不被错误切分 |
| 总结 | 单章节与跨文档主题总结 | 同一意图允许不同入口；不写死粗层 |
| 列举 | 完整列表与仅举例的不同材料 | 部分回答和完整答案分列 |
| 比较 | 两文档同一属性但单位/版本不同 | 来源各自绑定，装包保留两侧 |
| RQ 边界 | 向量邻近但 primary 不同，primary 相同但内容不同 | 无硬地址门槛，不将簇关系当事实 |
| 多语 | 中文问法、英文正文及中英混排 | 分别观察 Dense 与词面覆盖 |
| 候选域 | 一路排名靠后、另一路排名靠前 | 各通道独立提名，投影先于本层截断 |
| 全局容量 | 多主题语料超过上下文预算 | 记录未观察主题，范围不足不包装成全库概括 |
| 失效 | 更新、删除、改分词、改 codebook | 混合缓存失效，旧索引不回流 |
| 模型计划 | 合法不同权重、负权重、空词面混合矛盾 | 合法计划执行，非法计划明确失败 |
| 生命周期 | 超时、取消、复用失效、SSE 结束 | 不重复生成、不误报无资料、不重返加载 |

这些样例是独立问题族，不是具体词串的产品分支。测试可固定一份计划验证执行器，但真实规划验收应允许多种合理层级选择，通过来源和结果判断，而不是只看 LLM 是否选中了预定枚举值。

逐项记录 Recall@k、nDCG@k、来源/主题覆盖、答案要点、错误引用、部分回答、拒答、失败、时延和实际 token。冻结样例与执行版本，保留通道/固定权重/LLM权重对照。当前不把这些指标加权成一个优化目标，也不把模型自述作为效果标签。

## 设计决策与待验证假设

| 决策 | 依据类型 | 尚未得到的保证 |
|---|---|---|
| 意图与执行策略分离 | 工程契约，参考自适应检索研究 | LLM 能稳定选对层级 |
| 空词面进入纯向量 | 满足全局查询语义的设计选择 | 自动覆盖全部主题 |
| 加权 RRF 首版 | 融合量纲与可重放性考虑 | 比归一化凸组合更准确 |
| RQ 前缀重构距离 | 量化表示的定义及已有数据结构 | 独立于 Dense 的增益 |
| 原文 chunk BM25 与上层投影 | 标准词面检索加图归属约束 | 中文分词和父域覆盖已经达标 |
| LLM 在规划时给权重 | 请求级策略能力 | 无结果反馈也能给出最优权重 |
| 确定性来源准入与一次生成 | 来源、事务与有限执行边界 | 全部回答的语义正确与完整 |

完整执行协议见[检索与问答](retrieval-and-qa.md)，持久化和缓存见[数据与生命周期](data-and-lifecycle.md)。实现前应将上述假设落实为验收计划，不能用文献引用代替本项目证据。

## 来源

以下均为原始论文或官方文档；核对日期为 2026-09-14。论文年份与网页抓取时间分开，最新接口文档不能替代锁定版本核验。

1. Jeong 等，2024，NAACL。[Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/)。用于问题复杂度与执行策略分离。
2. Edge 等，2024，2025-02 修订 v2。[From Local to Global: A GraphRAG Approach to Query-Focused Summarization](https://arxiv.org/html/2404.16130v2)。用于全局问题、社区组织及实验范围限制。
3. Cormack、Clarke、Büttcher，2009，SIGIR。[Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)。用于排名融合原理。
4. Bruch、Gai、Ingber，2023 修订 v2。[An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/html/2210.11934v2)。用于 RRF 与凸组合的条件性比较。
5. Qdrant，官方在线文档。[Hybrid Queries](https://qdrant.tech/documentation/search/hybrid-queries/)。用于融合尺度、DBSF 和 weighted RRF 版本边界。
6. Hsu、Tzeng，2025-03，预印本 v1。[DAT: Dynamic Alpha Tuning](https://arxiv.org/html/2503.23013v1)。用于区分检索前规划与读取结果后动态调权。
7. Meta/Faiss，官方文档。[Additive quantizers](https://github.com/facebookresearch/faiss/wiki/Additive-quantizers)。用于残差编码、重构及近似边界。
8. Zhang 等，2023，EMNLP。[Hybrid Inverted Index Is a Robust Accelerator for Dense Retrieval](https://aclanthology.org/2023.emnlp-main.116/)。用于候选域组织的比较边界。
9. Apache Lucene，9.12.1 API。[BM25Similarity](https://lucene.apache.org/core/9_12_1/core/org/apache/lucene/search/similarities/BM25Similarity.html)。用于 IDF、参数含义及默认值。
10. PostgreSQL，17 文档 §12.3.3。[Controlling Text Search](https://www.postgresql.org/docs/17/textsearch-controls.html)。用于内置排名函数与 BM25 的区别。
11. Qdrant，官方在线文档。[Full-Text Search](https://qdrant.tech/documentation/search/text-search/full-text-search/)。用于分词及最新配置的版本限制。
12. Qdrant，固定版本发布记录。[v1.17.1](https://github.com/qdrant/qdrant/releases/tag/v1.17.1)。与仓库 Compose 固定版本对照，不等于应用已集成所有服务能力。
13. Liu 等，2023/2024。[Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)。用于长上下文位置测试的依据。
14. Thakur 等，2021。[BEIR](https://arxiv.org/abs/2104.08663)。用于跨数据集和任务的独立检索评测依据。

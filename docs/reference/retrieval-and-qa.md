# 检索与问答协议参考

本协议定义当前 `intent_execution_retrieval_v1` serving 架构。意图描述任务，执行策略决定入口层级、词面和评分通道；所有来源最终回到 chunk 原文。设计依据见[检索架构调研](retrieval-research.md)，验证方法见[开发与测试](../development.md)。

## 请求与规划

### 统一入口

后端 Search、同步 QA 和 SSE QA 共用同一规划与分层检索契约。Search 在结果与来源返回后结束，QA 再生成一次回答。前端不提供独立 Search 产品页，也不提供普通/摘要模式选择；问答仍使用后端共享检索契约，回答风格不控制图入口。

LLM 可直接在一次规划输出中分别返回 `intent` 与 `execution_strategy`；也可先通过下述有界只读观察，再在最后一次输出中同时返回两者。它只能使用原问题、用户明确的会话约束、非事实性会话摘要、服务器提供的有界 capability manifest 和本轮获准的粗层导航材料。manifest 包含可用层、版本、索引状态、文档/语言数量、可用范围与预算上限，不包含旧答案或旧模型判决。

计划必须在检索前通过闭合 schema、本地权限和预算校验并持久化。模型仅能声明 `resource/read` 的两个受控动作；服务端执行读取。模型不能调用其他工具、选择不存在的字段、修改图边、放宽用户范围或直接提交数据库 ID。执行器依据声明式字段行动。

资料型 active 图/BM25 可用性核验与规划模型往返彼此独立时可以并行：核验使用单独只读数据库会话和有界 I/O 槽，正式检索前必须 join 并核对当前 active 身份。系统能力、澄清及已通过来源重放的复用路由不以该资料型核验授权；并行失败不能静默改通道或权重。

### `resource/read` 状态机

执行器把实际完成的标题目录、详情读取和一次格式反馈按发生顺序写入 Agent run 轨迹，最终 plan 校验通过后再写规划完成事件。轨迹仅提供有限动作/计数/耗时审计；原文目录与摘要只进入本轮有界规划输入，不作为公开轨迹事实或回答证据。直接 plan 无读取事件；历史与断线恢复从同一持久事件序列读取。

`coarse_resource_read_v1` 的初始响应是完整计划或 `{action:"resource_read",mode:"titles"}`。标题目录返回后，响应是完整计划或 `{action:"resource_read",mode:"details",keys:[...]}`；详情返回后只能是完整计划。`keys` 为本轮目录临时键、1–4 个、不重复，不能使用数据库 ID。最多两次读，正常路径最多三次规划模型调用；无粗层时只能直接规划。仅当最终完整计划未通过闭合 schema 时，服务端可再给模型一次不含原始响应的安全字段路径/错误码反馈，要求重提**完整计划**，总调用上限为四次。该重试不读取检索结果、不生成词面、不代改策略；第二次仍非法则在正式检索前失败。无效读动作、越权键、图身份漂移或资源预算超限不进入格式重试。

总硬时限覆盖上述最坏四次规划、一次回答生成和非模型检索/准入余量。当前默认值为 540 秒，规划每次最多 60 秒、生成最多 240 秒；各调用的预算不能简单相加超过总时限后仍声称循环可完成。SSE/同步共享执行 owner 与超时终态。

目录只查询同 KB 且同 active 图的粗节点，按 `node_weight DESC, id ASC` 返回全部标题、临时键、权重、置信度及支持计数等有界元信息。输入超过上限时显式失败并报告条数/大小，不能以部分目录作为完整目录。详情只返回所选节点的摘要、定义、范围、包含/排除条件等有界导航文本；字段截断明确标记并限制总输入。用户过滤后，只有节点的全部支撑 chunk 均可见才返回该节点，避免聚合摘要夹带范围外材料；无法证明映射时拒绝读取。每次动作、snapshot hash、目录和选中项 hash、耗时、模型调用数都进入 run 审计；正式检索前重核 active 图身份。目录和详情都不是事实来源，不能进入 Context Package、来源引用或历史复用包。

### 意图契约

`IntentContract` 包含 `primary` 与至多3个不重复的 `secondary`，secondary 不重复 primary；它引用 Task 中的用户责任、来源范围与回答形式，不保存另一套可独立修改的约束。首版意图枚举如下；扩展需升级契约及独立样例，不能以任意字符串绕过 validator。

| 意图 | 含义 | 需要保留的责任 |
|---|---|---|
| `summarize` | 概括指定材料或主题 | 范围、概括重点、必要限制 |
| `overview` | 浏览资料库主题或形成全局认识 | 主题广度与已检查范围 |
| `define` | 定义术语、解释概念 | 指代、限定条件 |
| `fact_lookup` | 查询属性、数值、时间或具体事实 | 对象、属性、单位、时间、否定 |
| `enumerate` | 列举类别、成员或条目 | 完整列举与举例的区别 |
| `compare` | 比较对象或来源 | 两侧对象、属性及各自来源 |
| `explain` | 解释原理或关系 | 解释对象、因果与相关的区别 |
| `procedure` | 步骤、使用方法或算法流程 | 顺序、前置条件与适用范围 |
| `analyze` | 基于材料分析、综合或计算 | 数据、推导依据及限制 |
| `relationship` | 查询对象之间的关系 | 两端对象和关系要求 |
| `source_lookup` | 定位文档、章节、表格、公式或代码 | 来源身份、结构地址及表示完整性 |
| `system_capability` | 系统自身能力或用法 | 服务器能力卡边界 |
| `clarify` | 当前任务存在关键歧义 | 需要明确的用户约束 |

意图之间允许组合，例如 `compare` 加 `summarize`。`primary` 不唯一决定 `entry_layer`、是否生成词面、是否混合或回答长短；不得重新建立“总结必走粗层、其余必走中层”的固定映射。意图属于问题解释，不是效果标签。

用户要求保存为 `TaskContract`：完整原问题、实体/标识符、属性、数值/单位、时间、否定、比较对象、来源责任、回答约束。没有可用词面时，任务责任仍然存在；禁止用空词面清空问题、扩大范围或删除困难要求。用户给定的指代约定和输出格式不另增取证任务；身份核实、真实比较和来源政策仍需原文。

### 执行策略契约

`ExecutionStrategy` 的 active 最小字段：

```text
protocol_version = intent_execution_strategy_v2
route = retrieve | verified_context_reuse | system_capability | clarify
entry_layer = coarse | mid | chunk | null
semantic_query = nonempty text when route=retrieve
generate_lexical = boolean
lexical_groups = bounded array of {
  group_id,
  requirement_ids,
  kind = concept | identifier | number_unit | quoted_literal,
  surfaces = bounded array of {text, language = zh | en | neutral, provenance}
}
hybrid = boolean
layer_weights = {each visited layer: {dense, rq, bm25}}
selection_scope = focused | broad
budget_request = optional bounded requests
reason_code = closed planning reason
```

active 容量：semantic_query 为1–4000字符；lexical groups 至多12组，每组至多4个 surface，展平并去重后至多24项，每项1–160字符；任务责任至多8项，超出时明确说明能力边界而不丢项。group id 在单计划内唯一并使用稳定有界标识；每组绑定至少一个已存在 requirement。reason_code 允许 `broad_scope/precise_terms/semantic_paraphrase/mixed_signal/source_locality/existing_evidence/system_request/ambiguous_request`。budget_request 只允许已注册的通道候选数、层/父节点起点数、最大深度和结构恢复额度，各值为非负整数且不得超过 capability manifest 的对应上限。具体预算键为 `dense_candidates/rq_candidates/bm25_candidates/root_entries/per_parent_entries/layer_entries/max_depth/restore_per_hit`；前三项表示各通道预取规模，层/父入口字段分别限制各自候选入口，后两项允许0以关闭额外展开/恢复。schema 扩展必须升版本。

`provenance=user_text|model_query` 区分用户原词与模型提出的查询表达；后者只是检索假设，不是已证实别名。`language` 是检索面的语言审计，不是文档语言事实。保留用户原始实体、否定、数值和时间；模型不得根据记忆生成答案词或伪装成原文见证。空 group 数组是一等合法结果。

`bilingual_lexical_groups_v1` 在同一次规划内控制双语：开关开启时，kind=`concept` 的每组必须同时包含至少一个 `zh` 和一个 `en` surface，并通过本地 Unicode/script、空白、控制字符和重复项校验。`identifier/number_unit/quoted_literal` 允许 neutral surface，不为代码标识符、编号、单位或逐字引用强造翻译。开关关闭时不要求成对，但仍允许标准技术别名。执行器按 group 顺序和组内 surface 顺序稳定展平，再交给 tokenizer；group、surface、语言、开关与展平 hash 全部进入计划/trace/cache。不得为双语另发模型请求，也不得调用遗留 query-facet 链。

服务器的 capability manifest 列明 `coarse/mid/chunk` 中真实可用的层。首版只选择一个根入口层；向下的入口属于同次分层检索。RQ 是地址与评分信号，结构图是地址恢复层，均不是可选择的第四或第五个语义入口。

retrieve 或带正式检索后备计划的 verified_context_reuse 必须提供 entry_layer 与其完整下游 layer_weights；成功复用不执行这些通道。能力卡/澄清必须 entry_layer=null、semantic_query 为空、layer_weights 为空，且不申请图预算。

`layer_weights` 在规划时为根层及其下游层分别确定，在同次检索中冻结；同层各父节点使用同一组权重。LLM 可以逐请求自行调整有限权重，但不能按检索结果再次调权，也不从历史策略状态加载训练先验。预算申请受根配置硬上限约束；未给出的预算由执行器使用版本化默认值。

### 组合合法性

| 组合 | 词面 | 生效通道及权重 |
|---|---|---|
| 不生成词面 | groups 必须 `[]` | `hybrid=false`；dense=1、rq=0、bm25=0 |
| 生成词面但不混合 | groups 非空且通过单语/双语门禁 | dense=1、rq=0、bm25=0；词面可辅助构造语义查询，不运行 BM25 |
| 混合检索 | 必须生成非空 groups | dense>0、bm25>0、rq≥0，三项之和为1 |
| 能力卡或澄清 | groups 必须 `[]` | `entry_layer=null`，不创建图执行策略 |

所有权重有限、非负、≤1；归一化前总和须在 `1±1e-6` 内，再由确定性有限浮点协议消除尾差并保存模型值/生效值。超过容差、NaN、Inf、负值、未知通道、缺层或多余层均拒绝；不能替模型截断出一份不同策略。

`generate_lexical=false` 且 `hybrid=true` 是矛盾计划，不得自动生成词面修补。`generate_lexical` 必须等于 group 是否非空；group/surface 总数、语言配对或 requirement 绑定不合法时在任何检索前返回 `strategy_invalid`。词面经版本化 tokenizer 后全部为空也应拒绝。有效词面在健康索引中零命中是正常结果，保留零词面贡献；索引不可用则是 `index_unavailable`，不得静默改成纯向量。

全局问题可直接用完整问题和已授权范围构造 `semantic_query`，不强造可匹配 facet。原问题始终进入任务身份和回答输入；模型压缩后的语义查询不能替代任务。

### 策略示例

以下为公开合成问题的合法计划片段，用于解释字段组合，不是按词串选择入口的产品规则。正式计划还需 Task、能力清单、预算和身份审计。

```json
{
  "intent": {"primary": "overview", "secondary": ["summarize"]},
  "execution_strategy": {
    "protocol_version": "intent_execution_strategy_v2",
    "route": "retrieve",
    "entry_layer": "coarse",
    "semantic_query": "概括资料集涵盖的主题及其联系",
    "generate_lexical": false,
    "lexical_groups": [],
    "hybrid": false,
    "layer_weights": {
      "coarse": {"dense": 1, "rq": 0, "bm25": 0},
      "mid": {"dense": 1, "rq": 0, "bm25": 0},
      "chunk": {"dense": 1, "rq": 0, "bm25": 0}
    },
    "selection_scope": "broad",
    "reason_code": "broad_scope"
  }
}
```

```json
{
  "intent": {"primary": "fact_lookup", "secondary": []},
  "execution_strategy": {
    "protocol_version": "intent_execution_strategy_v2",
    "route": "retrieve",
    "entry_layer": "chunk",
    "semantic_query": "参数 queue_limit 的定义及其单位",
    "generate_lexical": true,
    "lexical_groups": [
      {
        "group_id": "l1",
        "requirement_ids": ["f1"],
        "kind": "identifier",
        "surfaces": [
          {"text": "queue_limit", "language": "neutral", "provenance": "user_text"}
        ]
      }
    ],
    "hybrid": true,
    "layer_weights": {"chunk": {"dense": 0.45, "rq": 0.15, "bm25": 0.40}},
    "selection_scope": "focused",
    "reason_code": "precise_terms"
  }
}
```

示例权重仅展示合法结构；不规定所有精确查询必须采用该配比。规划必须允许同意图不同层、不同权重和不同词面选择。

## 分层入口评分

### 三个分数域

| 信号 | 计算对象 | 作用 |
|---|---|---|
| Dense | 查询向量与同层候选向量 | 直接语义入口排序 |
| RQ | 查询向量与候选关联的 RQ 前缀重构向量 | 聚类地址邻近度 |
| BM25 | 查询词面与 active chunk 原文词项 | 词面入口与上层投影 |

三者都只参与根层起点及逐父节点下钻时的子层起点选择。它们不生成物理边，不改写边强度和距离，也不构成事实充分性分数。node weight 不属于第四个相关性通道。

### Dense 通道

查询向量使用当前 embedding 模型与文本协议。候选向量必须同模型、同维度、同数值预处理身份；粗/中概念使用已有有原文支撑的概念向量，chunk 使用 active 索引向量。

`dense(q,v)=cos(T(q),T(v))`，其中 `T` 完全复用冻结构建的预处理。零模、缺失、非有限或版本混杂是技术错误。纯向量策略直接按该分数在同层排序，不要求任何词面命中，不运行 BM25 或 RQ 相关性评分；下钻仍核验 RQ primary 归属。

### RQ 簇相关性

采用 `rq_reconstruction_entry_v1`。候选 v 的关联前缀 p 由实际 primary 链及概念支撑映射确定：coarse 使用 L2，mid 使用 L3，chunk 使用自己的 L3 primary 前缀。

设前缀重构向量：

$$
\hat{x}_p=\sum_{j=1}^{depth(p)}c^{(j)}_{z_j(p)},\qquad
s_R(q,v)=-\min_{p\in P(v)}\|T(q)-\hat{x}_p\|_2^2.
$$

`P(v)` 只含当前候选有持久支撑的前缀；通常为一个。多个前缀时按最小距离、稳定前缀键破平并保存见证，不通过前缀数量累加分数。L2 与 L3 的原始数值不跨层比较。

不能直接对查询向量和第三层残差中心做 cosine：残差中心并非完整语义中心。不能把查询与 chunk 的地址相同当作必然相关，或只保留查询的一条硬编码地址。候选通道可按前缀距离提名多个有成员的前缀；每个 chunk 仍只持久化一条 primary chain，不把查询候选前缀写成新 membership。

Dense 与 RQ 源于同一向量系统，存在相关性；融合不把它们解释为两份独立证据。无效 codebook/支撑映射使已启用通道失败，不能以0分掩盖。是否保留非零 RQ 权重须由独立比较验证，不能宣称 RQ 必然增加召回。

### BM25 通道

新建 `source_chunk_bm25_v1`，从 active chunk 原文建立版本化倒排索引。初版不混入模型概念命名、生成摘要或历史词面产物。解析标题/章节定位可使用既有结构目录；它们不冒充原文 BM25 命中。

对通过 group 门禁并稳定展平、再经 tokenizer 去重后的查询词项集合 Q：

$$
s_B(q,c)=\sum_{t\in Q}IDF(t)\frac{f(t,c)(k_1+1)}{f(t,c)+k_1(1-b+b|c|/avgdl)},
$$

$$
IDF(t)=\log\left(1+\frac{N-df(t)+0.5}{df(t)+0.5}\right).
$$

`N/df/avgdl` 来自同 KB、同 active 索引快照的 chunk 集合。请求过滤先限制候选，但不临时重算这套统计；统计域与候选域分别保存，跨 KB 不共用 IDF。`k1=1.2,b=0.75` 为明确的工程起点，不代表已经调优；改变它们或统计/分词协议必须产生新身份。

首版 tokenizer 为 `source_jieba_nfkc_identifiers_v1`：先保持原文地址切分标识符、Unicode 单词和数字，再用项目已依赖的 Jieba 默认词典对中文段落分词，HMM=false、无用户词典、无停用词过滤。逐 token NFKC/casefold 不改变原文 offset；包含分词库版本与完整词典 hash。中英相邻文本、分解重音与兼容字符必须独立回归。中英文 tokenizer、大小写、Unicode、数字单位、代码标识符及停用词规则版本化。标识符的原形和 token 位置可回到原文跨度；重复查询词不重复加权。无有效 token 的计划在执行前被拒绝；已完成查询的零命中产生空列表。

BM25 在 chunk 层直接提名。在 mid/coarse 层，按当前可验证的 primary/概念支撑归属投影：

$$
s_B(q,v)=\max_{c\in Desc(v)\cap EligibleScope}s_B(q,c).
$$

取最大值是入口提名规则，避免把大簇的词频简单相加；它不证明整个概念或答案完整。保存获胜 chunk、词项和映射见证。无上层映射的命中单列 `unprojected_count`，不得伪造父概念。按 chunk top-k 先截断再投影可能丢失不同父节点，必须聚合到本层候选后再按本层预算截断，或报告明确的扫描截断。

PostgreSQL 保存可重建的原文、索引版本、文档长度、词典/df 和 postings。首版执行器在 Docker 内查询倒排索引并计算上述公式；不新增必需外部搜索服务。PostgreSQL `ts_rank/ts_rank_cd` 不是此 BM25 实现。Qdrant sparse 加速只在分词、IDF、长度归一化和结果等价核验后接入，不用 collection 全局 IDF 替换 KB 统计。

同一请求若同时需要完整查询和多个 requirement 的 BM25 视图，执行器只读取并核验一次 active 原文域、文档统计和查询词项并集 postings，再对每个视图分别累计贡献、排序和截断。共享读取不得合并各视图的分数、改变 df/平均长度或让完整查询的 top-k 预裁剪 requirement 候选；批量结果必须与相同身份下逐视图独立执行等价，posting 预算按实际联合扫描计量。

### 候选合并与加权融合

当前采用 `weighted_rrf_entry_v2`。它保留竞争秩融合，并在预算允许时保留每个健康通道的首个候选，防止低权重通道被完全饿死。该保留只适用于当前图层候选域；最终 chunk 还必须属于已遍历集合并有持久 path label，不能作为 BM25 直达答案的旁路。后续比较方案见[调研](retrieval-research.md)。

在同一 `layer + parent_scope` 下，各启用通道独立提名有界 top-n 列表；先完成各自资格/过滤/上层投影，再取候选并集，不能先按 Dense 裁剪所有通道。RRF 只对实际返回列表赋秩，未进入某通道列表的候选在该通道贡献0，不能伪造末位或将未扫描区域解释为无关。

$$
S(v)=\sum_{j\in\{D,R,B\}}w_j\frac{k+1}{k+rank_j(v)},\qquad k=60.
$$

秩从1开始；同原始分数使用同竞争秩 `1+严格更高分候选数`，列表截断及融合最终同分由稳定业务键破平。`(k+1)` 使每个通道首名贡献为其权重，不改变排序。`k` 与排名协议由服务器版本化，LLM 只选择 `w_j`。Qdrant 原生 RRF 的秩起点、平滑常数与 tie 规则须单独核验，不能按相同名字认定逐值等价。

健康通道返回空列表时不重新分配权重，防止执行结果改变模型策略。通道超时、索引缺失或协议漂移返回技术终态。零权重通道不运行；其状态不应阻断本次纯向量策略。

每个候选记录原始分数、通道秩、单项贡献、融合分、支撑映射与未入列原因。审计绑定完整候选域身份、各通道预取预算、排名/融合版本和 LLM 生效权重。候选截断、后端近似召回及父节点投影损失分别报告；最终 top-k 不等于全候选域已完整评估。

## 分层执行与全局问题

### 从 LLM 选择的层开始

```text
coarse: 粗层起点 → 粗层遍历 → 逐粗节点提名 mid → 合并 mid → 中层遍历
        → 逐中节点提名 chunk → 合并 chunk → 片段遍历 → 结构恢复
mid:    中层起点 → 中层遍历 → 逐中节点提名 chunk → 合并 chunk → 片段遍历 → 结构恢复
chunk:  片段起点 → 片段遍历 → 结构恢复
```

每个箭头都属于一轮固定策略执行。进入某层后不重新让 LLM 根据检索结果改层或调权。父节点探索分别使用自己的预算；完成父节点候选收集后再做同层合并、去重与输出限制。`top_k` 是最终 hit chunk 输出预算，结构恢复和 Context Package 有各自预算，不能把全库裸 top-k 包装成分层图检索。

BM25 在任一层提名到的候选也必须是合法 active 节点，并通过相同路径与来源约束。chunk 直接入口的查询向量/BM25/RQ 提名记录是起点证明，不捏造一条图边；后续只走真实关系边。

### 全局问题的覆盖

`selection_scope=broad` 表示优先分布到不同已提名父节点、主题与文档。纯向量起点仍由 Dense 排名提名，之后以版本化轮转分配各父节点预算，不能追加隐藏 BM25、随机事实或模型猜测主题。

“概括全库主题”是 broad 观察任务，不自动产生“所有原文字符必须装入”的 complete 责任；只有用户明确要求穷尽、逐项核对或完整引用指定对象时，才建立相应完整范围义务。不能通过把所有总结都解释为 complete 来重新阻断全局问题。

宽范围任务记录可用/已提名/已展开父节点数、文档数、主题数和预算截断原因。这些是观察计数，不是答案覆盖率。主题很多而预算很小时，粗层向量 top-k 仍会漏主题；不得宣称“总结意图 + 粗层入口”意味着穷尽全库。若用户要求完整范围而无法在硬预算内装入，返回范围限制或建议缩小范围，不伪装成完整总结。

概念文本是导航线索，回答事实仍取自下钻和结构恢复得到的原文。一次生成不采用社区级多次生成再汇总的隐式流程。

## 路径、灰区与终止

### 物理路径

新遍历协议为 `layered_distance_traversal_v2`。路径标签保存 layer、根入口、父节点、ordered edge path、累计距离、深度、visit counts、source/support refs 和来源角色。入口融合分与物理路径距离分列。

遍历 dominance 以 `(root_node_id,node_id)` 保存每个根下的最佳状态，因此一个 chunk 可以拥有多条合法根路径。最终结果必须在 `top_k` 前按全局稳定遍历顺序以 `chunk_id` 去重，只保留首条最佳路径；trace 同时记录 `path_candidate_count`、唯一 `candidate_count` 和 `duplicate_path_candidate_count`。路径去重不删除图边、不改变 hard threshold，也不允许未遍历候选补位。

$$
D(P')=D(P)+d_e+Penalty(P,e),\qquad d_e\ge0,\quad Penalty(P,e)\ge0.
$$

同层同父探索使用 `Key(P)=(D(P),depth(P),stable_path_key)`。来源范围与支撑先做资格检查；BM25、RQ、LLM 权重和主题计数不抵扣累计距离。根入口本身 `D=0`，另存真实入口分与提名见证，不把零步路径解释为已证明相关。分层父入口重置本层距离时，完整跨层谱系仍保留。

### 距离分区与本地灰区

沿用版本化距离门槛且满足 `τgreen≤τgray≤τhard`：green 为 `D≤τgreen`，gray 为 `τgreen<D≤τgray`，red 为 `τgray<D≤τhard`，hard stop 为 `D>τhard`。red 停止展开，hard stop 来源不能借重新装包进入答案。

green 若具有 `semantic_uncertain` 或 `crossing_rq_boundary` 也进入本地灰区规则。新 `deterministic_support_progress_v2` 适配合法空词面，不依赖模型评价，不读取 Profile、历史策略或 provider 状态。

本地闭合输入包括 layer、distance zone、完整支撑承诺、前后独立 support 集合、来源角色、结构恢复状态、已验证入口锚点、RQ 边界与语义不确定标志。词面可为空；无词面时，`query_anchor_preserved` 依据从已验证向量入口连续的合法支撑路径，不把空匹配集判为任务缺失。

`progress` 只表示新增独立支撑、合法新路径贡献或所需来源角色；重复访问不是新增信息。按以下优先级决定：

1. 支撑失败，或跨边界且锚点/来源角色均未保持：停止。
2. 有新支撑而原文结构上下文缺失：请求结构恢复。
3. 有新支撑且桥接边合格：沿桥接边继续。
4. 有新支撑且下一层候选合格：下钻。
5. 有新支撑：继续当前路径。
6. 其余：停止。

每次决定保存规则版本、输入 hash、命中规则、决定和 hard-interrupt state，模型调用数为0。expanded/compact 审计从相同最小 replay card 派生；显示频率不能改变本地决定。新协议不能重写旧协议审计。

### 去重与停止

相同 chunk 可以由不同路径到达，但正文只装一次，独立路径分别保留。重复 support 不增加贡献；相同节点、父域和支撑状态中更长路径被支配剪枝。环通过 visited state、标签数、最大深度和每路径 edge reuse 上限限制，不能靠重复绕行提高优先级。

邻接缓存只对完整读取的节点标记 complete；看到某条边的邻居不代表读过该邻居全部邻接。候选去重上限按具名父域/合并池分别计数；达到上限只拒绝新候选，不删除已接纳来源。

停止原因包括 frontier 耗尽、距离停止、来源约束、候选/深度/边/父节点预算、总时限、取消和技术错误。预算停止不等于语料不存在；检索评分不参与生成授权的语义完整性判定。

## 原文范围与 Context Package

### 来源约束

来源声明区分具体文档与资料族，不能把“某主题的报告”擅自缩成同名文件。过滤一致应用于各通道、所有层、结构恢复、复用和来源提交。字段包括 KB、document/version、source path/type、标签、表示类型与页码；未知页码不能满足具体页码限制。

结构定位使用解析产生的标题、编号、角色、父子关系和原文区间；来源文字按非指令数据处理。用户明确的文档、章节、表格、公式和代码位置须经确定性范围代数核验，必要时可有一次检索前的有界 LLM 候选位置选择。此调用只能在服务器给定候选中定位，不判断答案，不启动结果修正循环。

已验证的来源区间带有文档版本身份。范围目标规划只对区间实际涉及的版本做跨度求交，并在同一请求中复用每个 chunk 的 token 成本；候选顺序、预算、见证及完整范围审计保持原协议。不得为了省时把明确来源责任扩成全库裸 top-k 或省略未覆盖区间。

`intersection/union` 描述位置交并，`all/any` 描述多项责任必须分别满足或允许备选；两者不可混用。`complete` 是物理表示完整覆盖，`overlap` 是有合法交集；范围完整不代表语义可答。比较两侧保留独立来源，不能借同一 chunk 中出现两个标题替代各自区间验证。

### 装包

Context Package 是回答唯一事实输入，包含实际原文、chunk/版本/跨度、结构恢复结果、入口谱系和真实路径支撑。索引文本、概念命名、会话摘要及裸通道命中不直接成为事实。

结构恢复按 hit chunk 处理 previous、next 和已有合法桥接来源，沿用每 hit 有界预算。完整 chunk 是稳定引用单位；额外恢复不能越过用户过滤域或吞掉 scope ambiguity。表格跨页、公式、图像或代码若解析不完整，保留 `representation_incomplete`。

装包按请求已冻结的范围与预算执行一次确定性选择：明确的必需来源优先，比较侧与 broad 父节点保留各自配额，候选在各域内按来源顺序及稳定检索顺序组织。所有相交且能装入的完整 chunk 保留；装不下时明确记录未装入范围与 token 截断，不能用相似度最大值不变证明信息无损。

总模型输入预算包含 system/schema、完整问题、范围指引、来源标签、正文和有界历史摘要，并预留一次生成输出。正文的本地估算 token 与 provider 实际 input token 分开记录；字符数或正则 token 数不能冒充模型窗口精确计量。预算不足在调用前返回 `context_budget_exhausted`，不静默截断必需原文。

### 确定性来源准入

`source_integrity_admission_v1` 检查：非空实际包、当前过滤/版本身份、原文跨度重放、结构表示状态、路径支撑、明确来源责任、packaging manifest 与剩余执行预算。它绑定 Task、ExecutionStrategy、检索 trace、来源 manifest 和准入审计。

该准入不把 cosine、RQ、BM25、融合排名或旧覆盖阈值解释为答案充分性。若必需物理来源未确认、包为空或整体范围装不下，返回有界缺口；资料合法且进入实际包则可进入一次生成，由生成器根据原文回答能支持的内容并说明不足。

## 一次回答、直接路由与会话

### 回答与引用

生成器只读当前 Task、意图、回答约束、完整 Context Package 及来源指引。目标回答使用 `grounded_markdown_inline_citations_v1`：模型直接生成最终 GFM 文本，不返回 JSON，并在相关文字后使用 `⟦cite:src_n⟧` 或多 handle 形式标注原文来源。枚举不能把几个例子写成全集，比较不能混用版本和单位，分析中的推导要指明原文依据。

允许回答已支持部分并明确剩余缺口，但不得将部分完成计为完整事实答案。完全缺乏支持时可返回闭合的 `insufficient_evidence` 结果，由本地模板呈现；该生成结果是终态，不再触发检索。

正文不再包裹在模型 JSON 中，也不在结束时接受第二份正文。流转换器仅把合法引用标记改成稳定序号链接；错误或未知格式按原字符进入 append-only 正文，不构成技术失败。持久答案必须与 SSE 已释放的转换后字符逐字相同。正文完成后，执行器把本轮实际 Context Package 中全部已准入来源绑定到完整答案跨度，并生成独立来源列表。该列表证明生成材料的身份和可重放性，不宣称逐句语义证明。模型输出截断、取消和超时保持独立终态，不自动再生成，也不把失败流提升为可复用事实。

### 直接路由

`system_capability` 只读版本化服务器能力卡，零检索、零引用、零 context/retrieval id。`verified_context_reuse` 只接受同 KB、同会话且 provenance replay 通过的完整原文包；本轮以当前任务重新做确定性范围、来源和预算检查。复用失败可在尚未执行正式检索时进入同一冻结计划，不能引用旧答案或旧判决。

需要澄清时询问真实歧义，不用模型记忆填补事实。会话摘要只帮助理解当前问题；当前用户要求优先，模型不能把历史 prose 升为新的来源责任。

## 状态、审计与失败

```text
接纳 → 可选粗层 resource/read → 最终规划 → 本地策略校验 → 来源定位/复用检查
    → 分层入口与遍历 → 结构恢复及装包 → 确定性来源准入
    → 一次生成 → 来源绑定 → 完成
```

Search 在来源准入后返回检索结果。能力卡和澄清有独立终态。每阶段可以失败或取消；生成后没有返回检索的边。新 run 使用独立协议版本，不复用旧 FSM 的含义。

审计记录模型提出的意图/策略、服务器校验及生效预算、通道输入身份与完整性、每层/父候选列表与融合贡献、路径和灰区决定、装包范围、来源准入、最终绑定、各阶段耗时。模型调用前提交 `prepared` 意图，成功后才记 `completed`；prepared 不代表模型已完成。

| 终态或原因 | 可对外表达 |
|---|---|
| `completed` | 返回已生成且来源已绑定的回答 |
| `partial_answer` | 返回支持部分并说明未确认范围 |
| `insufficient_evidence` | 本次材料不足，未断言全库不存在 |
| `scope_ambiguous` | 需要用户明确主体或来源 |
| `representation_incomplete` | 指定来源的解析表示不完整 |
| `context_budget_exhausted` | 必需范围超出本次装包容量 |
| `strategy_invalid` | 计划不符合可执行契约 |
| `entry_unavailable/index_unavailable` | 计划所需图层或索引不可用 |
| `technical_failure/cancelled` | 依赖失败或取消 |

`partial_answer/insufficient_evidence` 与成功事实答案分开统计。技术错误不能显示成资料缺失。SSE 和同步终态含义一致，UI 不能在终态后重新进入加载。

SSE 在等待规划、图检索或一次生成时，每 10 秒发送一次 `: keep-alive` 注释，并返回 `Cache-Control: no-cache, no-transform`、`Connection: keep-alive` 与 `X-Accel-Buffering: no`。该注释只维持传输连接，不是 Agent event，不进入 trace、Context Package、缓存身份、会话历史或模型上下文。目标生成流只有 append-only 正文增量、引用列表和 final，不使用完成后的正文替换帧。执行 owner 与 SSE observer 分离：owner 持有数据库会话、并发租约、硬时限和终态责任；observer 关闭只取消订阅和传输等待，不调用 task cancel、不释放 owner lease、不把 run 写成 cancelled。客户端收到 run id 后若传输中断，必须继续读取 PostgreSQL 持久 run 状态，直到完成、澄清、失败或取消，不能把 EOF 直接猜成事实终态。

浏览器显式取消通过独立 cancel endpoint 定位 run owner；迟到的旧取消响应不得覆盖新 run。服务进程退出时必须把不能继续的 owner 收敛为独立技术失败或取消终态，并保留已接纳问题；不能依赖观察者 finally 完成业务补偿。

失败或取消 run 必须把所属会话的 task state 收敛为 `failed/failed` 或 `cancelled/cancelled`。写入前确认该 run 仍是该会话最新 run，防止迟到的旧连接覆盖新一轮状态；下一轮正式请求将这些终态重置为 `active/answering`。传输断开只释放观察者资源，run owner 继续持有执行任务和准入租约并最终持久化终态；用户显式取消、服务关闭或 owner 自身失败才结束 owner。任何路径都不得留下无 owner 的运行中任务。

## 验证边界

开发与验收使用独立跨文档、跨问题族、跨语言和跨表示材料。至少覆盖：

- 总结、浏览、精确查询、列举、比较、步骤、分析及其组合；同一意图选择不同层。
- 无词面纯向量、生成词面但纯向量、三通道混合、RQ 零权重及各通道故障。
- 中英文术语、编号/单位/代码标识符、空分词、词面零命中、全部原始分数同分。
- 通道独立提名、BM25 上层映射、同层融合、逐父预算、无概念映射的命中。
- codebook/索引/用户范围变化、缓存失效、来源篡改及持久重放。
- 全局问题的主题分布、预算下未覆盖主题、不同上下文位置及长度。
- 无额外模型结果判定调用、来源准入后一次生成、终态一致及历史隔离。

保留各通道、固定权重融合与 LLM 权重融合的独立对照，分别观察 Recall@k、nDCG@k、来源完整性、主题覆盖、答案质量、时延、token 和失败。这里规定测量维度，不定义新的综合优化目标或训练信号。

性能复杂度包括：查询 embedding、同层 Dense/RQ 提名、BM25 postings 扫描及上层投影、候选合并排序、图遍历、结构恢复、装包序列化、模型等待与来源提交。设通道列表并集大小为 M，融合与排序为 `O(M log M)`；倒排扫描成本和图探索成本不能隐藏到该式之外。任何 bounded/approximate 阶段都记录实际观察域，不能把局部完成当作全库完成。

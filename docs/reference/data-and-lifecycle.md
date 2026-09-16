# 数据与生命周期协议参考

本文件说明跨层关联、持久对象、配置生命周期、缓存身份和事务约束。表格是字段语义索引；完整类型及数据库约束以[models.py](../../apps/api/app/models.py)、[schemas.py](../../apps/api/app/schemas.py)和迁移文件核对。运行结果不写入本文件。

## 跨层对象协议

### 关系

跨层对象协议将每层关系表示为稀疏 membership 矩阵：

$$
M^{C\to R_3}_{cp}\in[0,1],\quad
M^{C\to R_2}_{cp}\in[0,1],\quad
M^{R_3\to M}_{pm}\in\{0,1\},\quad
M^{R_2\to K}_{pk}\in\{0,1\}
$$

从 chunk 到 coarse concept 的派生支撑强度为：

$$
M^{C\to K}
=
M^{C\to R_2}M^{R_2\to K}
$$

从 chunk 到 mid concept 的派生支撑强度为：

$$
M^{C\to M}
=
M^{C\to R_3}M^{R_3\to M}
$$

这些 membership 只作为路由、投影和解释信号，不能替代 citation span。事实约束为：

$$
\operatorname{Fact}(x)
\Rightarrow
\exists c\in V_C,\ \exists s=(char\_start,char\_end): x\leftarrow(c,s)
$$

### 边证据投影

四层图需要边证据投影协议。上层边必须能回到底层 chunk relation edge 集合：

$$
E_M(m_a,m_b)
\Leftarrow
\{e_{cc}\in E_C:\mu_{c_i,m_a}>0,\ \mu_{c_j,m_b}>0\}
$$

$$
E_K(k_a,k_b)
\Leftarrow
\{e_{cc}\in E_C:\mu_{c_i,k_a}>0,\ \mu_{c_j,k_b}>0\}
$$

mid edge、coarse edge 的存在性由底层 edge support 决定。RQ prefix adjacency、membership overlap、child concept adjacency 和 LLM edge explanation 只能进入 projection diagnostics，不能在没有底层 support chunk edge 时创建 active edge。

$$
\operatorname{ExistsEdge}(u,v)
\Rightarrow
|support\_chunk\_edge\_ids(u,v)|>0
$$

任意上层边 \(e^l\) 必须保存：

```text
distance
projected_distance_raw
projected_strength_raw
raw_strength_summary
projection_normalization_stats_json
support_child_edge_ids
support_chunk_edge_ids
support_chunk_ids
edge_type
source_algorithm
protocol_version
edge_projection_protocol_hash
diagnostics_json
```

投影聚合会改变距离分布，因此 mid/coarse edge 不能只保存下层距离聚合值后直接套用阈值。目标协议必须先从下层 normalized distance 聚合得到 raw projected distance，再按 `layer + edge_type` 做投影校准：

$$
d_{\mathrm{proj}}^{raw}(e^l)
=
\operatorname{Agg}_l
\left(
\{d(e):e\in support(e^l)\},
\{support(e)\},
protocol_l
\right)
$$

$$
s_{\mathrm{proj}}^{raw}(e^l)
=
\exp(-d_{\mathrm{proj}}^{raw}(e^l))
$$

$$
s_{\mathrm{proj}}(e^l)
=
\operatorname{Calib}_{l,t}
\left(
s_{\mathrm{proj}}^{raw}(e^l);
\operatorname{ProjectionStats}_{l,t},
\operatorname{ProjectionProtocol}_{l,t}
\right)
\in(0,1]
$$

$$
d(e^l)
=
-\log(\max(\epsilon,s_{\mathrm{proj}}(e^l)))
$$

其中 \(l\in\{mid,coarse\}\)，\(t=edge\_type(e^l)\)。`projected_distance_raw` 和 `projected_strength_raw` 用于诊断；active traversal、green/gray/hard threshold 使用校准后的 `distance`。该校准不是全局 weighted fusion，也不允许丢弃下层 support ids；它只解决投影聚合后 mid/coarse layer 的距离分布漂移问题，使阈值可按层稳定设置。

投影不允许断链：

$$
e_K
\Rightarrow
\exists e_C,\exists c_i,c_j
$$

其中最终 evidence chunk 必须能回到 raw span、page range、bbox 或 structure path。

### 关联字段

跨层协议要求每个可审计对象携带主键、state id 与 hash：

$$
id(o)
=
\left(
pk,\ state\_id,\ protocol\_version,\ state\_hash
\right)
$$

核心字段：

```text
chunk_id
document_version_id
structure_node_id
rq_prefix_id
mid_concept_id
coarse_concept_id
chunk_relation_edge_id
mid_concept_edge_id
coarse_concept_edge_id
chunk_relation_graph_state_id
mid_concept_state_id
coarse_concept_state_id
context_graph_state_id
retrieval_trace_id
context_package_id
answer_session_id
source_binding_id
runtime_settings_hash
agent_operating_envelope_hash
edge_distance_protocol_hash
edge_projection_protocol_hash
traversal_protocol_hash
```

### 代码中的 state hash

目标上，任意 active context graph 的 hash 为：

$$
h_{\mathcal{G}}
=
H\left(
h_C,h_0,h_1,h_R,h_2,h_3,h_{\mathrm{runtime}},h_{\mathrm{agent}}
\right)
$$

active state hash 使用 `canonical_graph_business_facts_json_v1`。编码必须采用 UTF-8、key 排序、compact JSON 和显式 protocol version；集合先转成稳定业务事实再排序，拒绝非有限浮点数。业务事实不得包含数据库随机 UUID、创建/更新时间、SQL 返回顺序、provider 原始响应或 gray-zone decision prose。chunk 的稳定业务键由 document source/checksum/type/title、chunk version/index、char/token span、section/page 与 text hash 组成；数据库 id 只允许作为持久化引用或同业务键下的 tie-break，不进入 content identity。

地址 identity 与内容 identity 必须分离。`chunk_scope_complete_address_v2` 继续绑定 active row/document-version 地址，用于隔离真实 PostgreSQL/Qdrant owner；canonical business scope 和 contextual-index business hash 则描述同一业务事实，保证随机 UUID 重建不抖动。cache key 同时绑定地址 scope 与 business/content identity，不能用其中一个替代另一个。

构建、shadow promotion 或显式 reconcile 必须从完整持久化事实深算并保存版本化 hash card。在线 search/QA admission 只校验 card 自身的 protocol/payload hash、各层 state hash 和有界 `COUNT`，不得每次查询序列化全量 relation/concept rows。任何完整事实或 protocol identity 变化都必须改变对应 layer hash；仅 UUID、时间戳、查询行序、provider 状态或 gray-zone explanation 变化不得改变 hash。

**架构影响：**
- 影响对象：所有跨层跳转、边证据投影、检索 trace、context package、answer audit、前端图谱 payload 和运维对账脚本。
- 影响方式：跨层协议提供 id、state、edge support 与 hash 的共同坐标系，使 chunk、RQ prefix membership、mid concept、coarse concept、edge projection、context package 与 source binding / answer reflection 能在同一审计链中互相定位。
- 传播字段：`chunk_id`、`rq_prefix_id`、`mid_concept_id`、`coarse_concept_id`、`chunk_relation_edge_id`、`mid_concept_edge_id`、`coarse_concept_edge_id`、`context_graph_state_id`、`retrieval_trace_id`、`context_package_id`、`state_hash`。
- 触发条件：任一层 state id、protocol version、edge distance protocol 或 hash 变化时，下游 API payload、cache key、retrieval trace 和 UI graph view 都应使用该协议坐标。
- 验收观察点：跨层 id 不悬空、edge projection 不断链、trace step 可回放、context package 可回到 raw chunk span、answer session 可回到 source binding / answer reflection。

## 事实源与派生状态

目标系统采用事实源与派生状态分离。设 \(S_P\) 为 PostgreSQL 持久状态，\(S_D\) 为派生状态，则一致性定义为：

$$
\operatorname{Consistent}(S_P,S_D)
=
\mathbf{1}
\left[
H(\operatorname{rebuild}(S_P))=H(S_D)
\right]
$$

### PostgreSQL

PostgreSQL 保存不可丢失事实、生命周期与审计记录。目标上它是唯一可恢复源：

$$
S_{\mathrm{recover}}
=
F_{\mathrm{rebuild}}(S_{\mathrm{postgres}})
$$

PostgreSQL 必须保存 knowledge bases、documents、chunks、structure graph、relation graph、concept graphs、context graph state、retrieval traces、context packages、answer sessions、source bindings、意图/执行策略审计、BM25 索引事实和 runtime settings versions；旧表保留边界见[历史兼容](compatibility.md)。

### Qdrant

向量索引目标函数是近似最近邻：

$$
\operatorname{ANN}(q)
=
\operatorname*{arg\,topk}_{c\in C}
\cos(e_q,e_c)
$$

Qdrant 是派生索引。collection 身份必须由原始五元组
`(embedding_model, embedding_dimensions, vector_distance_metric, embedding_text_version, chunk_schema_version)` 唯一决定，不能把 sanitize 或截断后的文本当身份。active 距离度量固定为小写 ASCII `cosine`；`embedding_dimensions` 必须是正整数，canonical 值是无符号、无前导零的十进制 ASCII。冻结协议
`qdrant_collection_identity_u64be_utf8_sha256_v2` 定义 canonical byte stream 为：协议名的 ASCII 字节、单个 `0x00`，随后按上述固定字段顺序依次写入 `u64be(UTF-8 byte length) || UTF-8 bytes`；字符串字段不 trim、不 case-fold、不做 Unicode normalization。identity digest 是该 byte stream 的完整 SHA-256 小写十六进制值。任一 vector schema 参数变更都必须产生新 identity/protocol，不允许就地改写已存 collection 配置。

collection 名为 `symbograph_{readable_prefix}_{identity_digest}`。`readable_prefix` 只用于人工诊断：把五字段以 `_` 连接后 sanitize/lower，最多保留 96 个字符；为空时使用 `identity`。digest 必须保留完整 64 个十六进制字符，最终 collection 名不得超过 180 个字符。sanitize 后相同或只在截断范围之外不同的五元组必须得到不同 collection。

Qdrant payload、`VectorRecord.diagnostics_json`、contextual index state hash 和 expected collection diagnostics 必须记录/绑定 collection identity protocol、digest、dimension 与 distance metric。协议升级、v1 三元组 collection 或旧无 digest collection 属于 `rebuild_required` 派生状态：expected collection/protocol/digest/vector schema 不一致时标记 contextual index stale 并重建，不设计静默兼容读取旧 collection。

`VectorRecord` 持久唯一身份固定为 `(chunk_id, embedding_model, embedding_dimension, embedding_text_version, chunk_schema_version)`，其中 `chunk_schema_version` 必须是直接非空列，不能只存在于可变 diagnostics；这使旧 active 与新 model/dimension/text/schema shadow candidate 可并存、评估和显式 promotion。不得原地改写旧身份事实伪装 shadow rebuild，也不得用四列唯一键阻断不同 chunk schema 的派生向量共存。

向量 rebuild-required 生命周期以 PostgreSQL 的 runtime candidate、per-KB shadow build 与 active vector pointer 为事实源。stage 只冻结五元 vector schema、collection identity、active chunk scope 和旧 pointer，不修改根 `settings.json`、Redis、Qdrant collection 或 active graph；同一 KB 同时至多有一个 live candidate。若 candidate 的 `chunk_schema_version` 与当前 active chunk scope 不同，向量 builder 必须 fail closed 并要求先产生匹配 schema 的 shadow re-chunk/scope，不能把旧 chunks 重新贴标签。

补偿向量恢复协议固定为 `vector_shadow_compensated_embedding_recovery_v2`。

同一 staged build 重试可以从其 compensated durable outbox 恢复；旧 candidate 已显式 supersede/reject 且新 candidate 的 KB、完整五元 vector schema/collection identity、active chunk id 集、逐 chunk contextual text hash、local-hint hash、canonical float32 vector 与 `vector_payload_hash_v3` 全部逐项相同时，也允许把旧 outbox 仅作为 embedding 数值来源重绑定到新 candidate/build 地址。

source candidate/build 地址必须存在且进入有界 source-binding count/hash audit，但不得复制为新地址 authority；新 writer 必须以目标 candidate/build 重新生成 payload/outbox 并再次执行完整 payload-hash/Qdrant proof。任一 schema、chunk、context、hint、vector、payload hash 冲突或扫描超界都 fail closed 并退回真实 provider 路径，不能部分混用、猜测或绕过 durable intent。

shadow build 必须使用 candidate-local embedding provider，经 durable Qdrant outbox 写入 candidate collection，并构造 state=`shadow` 的完整四层图。

build ready 证明必须来自 bounded exact-point Qdrant observation：按冻结 chunk ids 逐点验证 owner、collection/schema identity、payload hash 与 point-set hash，保存 observation protocol、input/output hash 和计数；不能以 PostgreSQL 期望值自证，也不能为此扫描并猜测整个 collection 的 orphan。evaluation 必须绑定该 proof、shadow graph state/hash 和版本化 hard gates/evidence hashes。

只有全部 build `evaluation_passed` 时，promotion 才可在一个 PostgreSQL 事务中原子切换 active pointer、四层 graph state、candidate records=`ready` 与旧 records=`rollback_retained`；四层 state 切换必须同时把各自 state-id 精确绑定的 `RQPrefix`、`MidConcept`、`CoarseConcept` 行从旧 active→inactive、candidate shadow→active，不能留下“父 state active、公开子行仍 shadow”的半激活状态。

rollback 必须按冻结 previous pointer/schema/graph ids 精确反向恢复父 state 与这些子行，并把被撤回 candidate 标为 `rolled_back_retained`。这些 lifecycle state 字段不进入 UUID-free graph business hash，切换不得改写图业务事实或 Qdrant payload hash；commit 后 Redis 失效失败必须留下可重试 intent，不能撤销已提交 pointer 或静默吞掉。

旧或被放弃 collection 的删除是独立 destructive 运维协议，不属于 reconcile/orphan scan。默认只能 dry-run 一个经过 allowlist 校验的 exact collection name；执行必须同时提供显式 execute flag 和完全相同的名称确认，打印 pointer/build/outbox/record 影响，并先提交 PostgreSQL exact-delete intent 再调用无 filesystem fallback 的 Qdrant delete。active pointer、live build、active outbox 或任一 serving `ready` record 均硬阻断。

`shadow_ready`、`rollback_retained`、`rolled_back_retained` 对自动 outbox/reconcile 仍是 authoritative，只有该 exact destructive intent 可显式放弃其恢复能力并在 verified absence 后标为 `missing`。stage、rollback 与 cleanup 必须按 exact collection 共享 PostgreSQL advisory fence；pending cleanup intent 存在时不得创建同 collection candidate 或把该 collection rollback 为 active。

本地检索路径必须在 `VectorRecord.diagnostics_json` 保存 embedding vector，以便本地 layered retrieval 直接计算 dense score。

### BM25 索引生命周期

`source_chunk_bm25_v1` 为版本化可恢复派生索引；PostgreSQL 保存原文事实、active 索引清单和发布审计。应用层读取 postings 执行标准 BM25，`ts_rank/ts_rank_cd` 不替代该公式。Qdrant sparse 可作为未来等价加速实现，但不是新架构首版的额外运行依赖。

目标数据契约：

| 对象 | 必需字段及约束 |
|---|---|
| `lexical_index_states` | KB、source/chunk scope、tokenizer/text/statistics/scoring protocol、N、总长度、avgdl、完整 df/postings hash、state、predecessor、publish intent；每 KB 一个 active 快照 |
| `lexical_documents` | index state、chunk/document version、raw span、原文身份、token length；同快照同 chunk 唯一 |
| `lexical_terms` | index state、确定性 term key、df；词项与分词身份绑定，跨 KB 不混用 |
| `lexical_postings` | index state、term key、chunk、tf、可重放原文位置；同词项同 chunk 唯一，显式外键 |
| `lexical_index_jobs` | 目标 scope、before-state、prepared/completed/failed/cancelled、计数/错误和补偿意图 |

词项和位置是私有检索数据，只存受控数据库，不进入普通日志/报告。tokenizer 需固定 Unicode/大小写、中文切分、标识符与数字、停用词、版本及任何分词词典身份。N/df/avgdl 属于同 KB active chunk 快照，过滤后的候选子集不改变统计域。

首次构建、文件更新、删除、版本推进和范围重建均先冻结源清单，建立候选 postings 与统计，逐项验证长度/df/tf/原文/外键/hash，再在 KB 行锁或资源锁下发布。指针切换保持原子性，缓存失效意图与发布同事务记录；失败和取消保留旧 active 指针，候选不可查询为 active。

源 scope 改变后，旧 lexical 快照标记 stale；混合请求不能读 stale 统计，纯向量请求仍按其所需图/向量 freshness 执行。只改 LLM 权重不触发索引重建；tokenizer、文本、统计或计分参数改变创建新 lexical 身份。改变 BM25 不自动重建独立的 Dense 关系图和 RQ。

崩溃恢复依据索引 job、源版本和 publish intent 对账，不从内存推断是否完成。删除 source 时先失效 active 依赖，再按持久意图清理 postings；不能留下仍可被新请求命中的孤立词项。Qdrant sparse 若后续启用，其写删须接入同类 outbox 并与 PostgreSQL 统计核对。

旧 lexical 表不属于新快照，不能在 cache miss 或索引不完整时作为兼容 fallback。具体迁移需另行 schema 变更和独立测试，本字段表不代表数据库已经新增。

### Redis

Redis 承担 runtime version broadcast。理论上，热加载事件为：

$$
event
=
\left(h_{\mathrm{runtime}},\Delta keys,source,timestamp\right)
$$

`publish_runtime_settings_version()` 必须写入 `runtime_settings_versions`，设置 Redis key，发布 channel message，并清理 settings、cache manager、retriever 与 lexical index reader 等运行时单例。

**架构影响：**
- 影响对象：导入、索引、图构建、检索、QA、Agent、runtime settings、缓存、对账脚本和测试验收。
- 影响方式：PostgreSQL 决定可恢复事实；Qdrant 与 Redis 只能改变向量召回效率、运行态协调和热加载，不改变事实来源。BM25 postings 与统计按独立索引身份恢复。
- 传播字段：`vector_records`、`runtime_settings_versions`、`payload_hash`、`collection_name`、`status`、`diagnostics_json`。
- 触发条件：派生索引缺失、payload hash 不一致、runtime version 更新或 Redis broadcast 失败时，相关检索路径应进入重建、刷新或阻断。
- 验收观察点：Qdrant 对账通过、runtime publish 可观测、cache miss 行为正确、active 派生状态能从 PostgreSQL 重建，新 BM25 索引可用性与纯向量/混合策略依赖一致。

## Runtime Settings、Profile 与请求策略

### 运行时设置

目标可变运行参数分为三类：

$$
\Theta
=
\Theta_{\mathrm{hot}}
\cup
\Theta_{\mathrm{rebuild}}
\cup
\Theta_{\mathrm{service}}
$$

固定协议常量不属于 \(\Theta\)。当前 `rq_kmeans_levels=3` 属于 `fixed_protocol`：历史 `.env` 中出现该键只用于迁移期启动一致性断言，新 `settings.json` 不保存该键；非 3 必须 fail-fast。GET/lifecycle 可以返回只读常量 3，PUT/update payload 携带该字段必须在 schema/service 边界拒绝，并且在拒绝前不得规范化或写任何配置文件、清理 Settings/检索 cache、写 `RuntimeSettingsVersion` 或发布 Redis version message。

配置权威由两个不重叠的根文件组成。`.env` 只保存 API key、数据库/Redis/Qdrant/模型连接、数据路径、端口和进程/服务启动参数；`settings.json` 只保存非秘密的产品、在线检索、预算、并发和构建参数。每个注册键必须恰好属于一个文件，重复键、未知键、类型漂移和跨文件覆盖 fail closed。前端、API、worker、beat、Compose 和启动器不得创建第三份 active/desired 参数文件。PostgreSQL 只记录不含参数值的单文件 hash、组合 version hash、changed keys、生命周期状态、错误类型与时间审计；Redis 只广播组合 version，不成为配置事实源。secret bytes 只能存在于根 `.env`，不得进入 `settings.json`、数据库、日志、报告或缓存。

active `settings.json` 使用 `symbograph_runtime_settings_v1` 的闭合结构：顶层只允许 `protocol_version` 与 `settings`，后者只允许本地注册的 snake_case 非秘密键和 JSON scalar。数组、嵌套自由对象、NaN/Inf、重复键及非 UTF-8 输入拒绝；canonical identity 使用 UTF-8、key 排序、compact JSON 和有限数规则。仓库保存脱敏 `settings.example.json`，本机 `settings.json` 被 Git 忽略。一键启动只在文件缺失时从 example 创建，不覆盖已有文件。

每次保存必须返回同一字段的 authority file、文件写入状态与生命周期状态：`written_and_applied`、`written_pending_rebuild`、`written_pending_service_recreate` 或 `failed`。模型连接字段以 `.env` 为基线，非秘密 Runtime Settings 字段以 `settings.json` 为基线；产品不显示数据库里的第二套值。一个完整表单 mutation 可以同时包含两类字段，但每个键只能落入自己的 authority；服务在同一跨进程锁下冻结两份 before-image，分别原子替换并在任一后续步骤失败时回滚已替换文件。`hot_reloadable` 写入后立即刷新进程和广播；`rebuild_required` 写入后记录 pending，已有图/索引继续按自身冻结的构建 identity 和 provenance 服务，完成 shadow build、evaluation 和 promotion 后由新派生状态接管；`service_recreate_required` 写入后明确要求 recreate，运行中容器不伪装已重建。

三类生效规则固定为：`hot_reloadable` 在所属权威文件原子写入后立即刷新当前进程、写不含参数值的 `runtime_settings_versions` 审计、广播 Redis 并清理相关单例/cache；`rebuild_required` 立即写 `settings.json` 并标记 pending，但现有图和索引继续保持原构建身份，必须经 target-KB candidate、dry-run、shadow build、evaluation、promotion 与 activation intent 后才能关闭 pending；`service_recreate_required` 立即写 `.env`，运行中进程保留启动值，显式 recreate 后从两份根文件重新形成组合配置。一个 Runtime Settings Save 可以同时包含 hot/rebuild 字段，但不能包含 `.env` 字段；hot 子集不得被 rebuild pending 阻断。三类状态必须分别返回并可重试。

模型配置分为三组，参数、协议、密钥和运行身份分别管理：

| 用途 | 参数前缀 | 允许协议 |
|---|---|---|
| 文档/查询向量 | `EMBEDDING_*` | 当前仅 `openai`，追加 `/embeddings` |
| 概念命名、定义和高层图解释 | `GRAPH_*` | `openai` 或 `anthropic` |
| 意图/策略规划、来源定位、一次回答与 Profile 助手 | `CHAT_*` | `openai` 或 `anthropic` |

OpenAI-compatible base 后追加 `/chat/completions`，使用 Bearer Authorization。Anthropic 使用 Messages API 顶层 `system`、官方 SDK、Bearer Authorization、固定 `anthropic-version` 和有界 `max_tokens`，base 后追加 `/v1/messages`。具体 base 格式见[环境说明](../../infra/README.md)。

启用模型桥时，客户端只使用已解析的各自 `Settings.*_base_url` 并同步桥接配置，不能从旧进程变量覆盖或把 chat/graph 混成同一路由。三组配置不互相 fallback，也不共享密钥状态。向量协议扩展需先定义路由、认证、批次、维度、错误与向量生命周期身份；不能把生成文本或假向量当 embedding。gray-zone 不使用模型端点。

向量连通性诊断使用 `embedding_provider_probe_v1`。脚本默认 dry-run，只冻结 embedding protocol/model/dimension、fallback/bridge 开关和 probe 输入的 UTF-8 长度/SHA-256；只有显式 `--execute` 才允许在 API 容器内同步 model bridge runtime config。

`--arm provider` 通过生产 `EmbeddingProvider.embed_texts_with_meta` 发起一个单文本 `query|document` 请求；`--arm bridge` 使用同一冻结模型、维度和凭据对本地 Docker bridge 的 `/embeddings` 发起恰好一个请求，绕过 EmbeddingProvider retry/error mapping，但仍禁止绕过 bridge 直连真实上游。

执行结果只能记录请求耗时、provider 类型、external-called、向量数量/维度、有限性、非零性、L2 norm，或 bridge HTTP status/content-type/response byte count/allowlisted error code/route，以及 `external_failure_classification_v1` 的有界 scalar 分类和异常类型链；不得记录输入正文、向量值、endpoint、resolve IP、API key、Authorization、provider body/headers 或原始异常消息。fallback=true 时 execute 必须在网络 I/O 前拒绝，防止 fake vector 被误报为连通。

失败必须输出脱敏诊断并以非零退出；bridge 未发布 loopback 端口时不得临时扩大 Compose 暴露面。

精确 base URL、effective endpoint、模型名和资料身份属于部署侧私有配置，不得硬编码或提交到仓库。测试只能使用 RFC 2606 保留的 `.invalid` 域名、合成模型名和公开合成数据。运行时 provider identity 必须绑定根 `.env` hash、根 `settings.json` hash、已发布的组合 `runtime_settings_versions` 审计与请求 scope；数据库不得保存完整参数 snapshot，不同 scope 不得共享或回退 endpoint pin。system prompt 的稳定前缀先于动态 evidence/conversation 内容，provider cache 只有 usage 中的 `cache_read` tokens 可以计为命中。

根 `.env` 与根 `settings.json` 通过受限 bind mount 提供给 API、worker 和 beat。文件身份审计必须分别以 canonical path、内容 SHA、size 和 protocol/version hash 为主，再形成组合 hash；不得因为 Docker 容器重建后的 mount/inode 差异制造另一份配置副本。更新事务必须有共享文件锁、两文件 expected hash、完整新字节校验、durable publication 和 before-image rollback；失败时恢复已替换文件并返回可行动错误，不得把临时文件或数据库 snapshot 提升为配置真值。

`CHAT_JSON_MAX_TOKENS` 是 `hot_reloadable` 的有界完成预算，当前允许范围为
256..32768；它用于 Anthropic 结构化输出的实际 token cap 为 `min(CHAT_JSON_MAX_TOKENS, component_cap)`，不改变路由、认证、schema 或 fallback 边界。

目标 QA 只为规划、检索前来源定位和一次生成分配模型阶段预算，统一服从 `RETRIEVAL_TOTAL_TIMEOUT_SECONDS` 与 provider 全局上限。规划输出 Intent 与 ExecutionStrategy，词面可以为空；来源定位的字段和独立阶段参数随新 schema 定义，不继续用修正参数隐式代管。

| 运行参数域 | 生命周期与生效边界 |
|---|---|
| 总时限、规划/定位/生成阶段上限、并发、输出预算 | hot reload；请求接纳时冻结，本地执行器负责硬中断 |
| 候选通道预取、逐父节点/层、深度/标签/结构恢复预算 | hot reload；进入执行策略身份，模型不能超限 |
| RRF k、排名/投影规则、RQ 入口评分规则 | 版本化检索协议；清理依赖缓存，不直接改图边 |
| tokenizer、索引文本、BM25 k1/b 与统计规则 | lexical rebuild；生成并核验新索引身份 |
| LLM 本轮入口层、双语词面 group、词面开关、混合权重 | 请求数据；不写配置文件，不触发全库索引或图重建 |

完整模型输入包括提示/schema、原问题、范围、标签与实际正文，正文估算 token 只用于已标明口径的容量预筛，provider 输入/输出用量分别记录。回答采用完整结构化单位，`AGENT_ANSWER_UNIT_LIMIT` 默认12、上限32，不按英文句点拆句。

计划的 shape、层级、通道与权重矛盾在本地拒绝。transport、超时、截断和无效输出保持技术终态，不在同一请求启动额外模型补救。阶段调用先保存 prepared 意图，实际完成后才保存 typed/hash 审计；provider 原始响应不持久化。

启用宿主 model bridge 时，bridge 只能监听经进程内校验的 literal loopback，直接传入 `0.0.0.0`、`::`、hostname 或其他非回环 bind 必须在 server construction 前失败；upstream 必须是无内嵌凭据、无 query/fragment 的 `https://` base URL，显式 resolve IP 只能是 globally routable public unicast IP。

provider 请求与应用层 DoH 都必须使用系统 CA 并校验原始 URL hostname，DNS override 只能改变连接 IP，不能关闭 TLS、改变 SNI/Host 或使用 `CERT_NONE`、`insecure`、`-k`。provider 与 DoH opener 必须完全禁用 redirect，任意 3xx 都按固定错误 fail closed，不能跨 origin、跨协议、改 method 或继续携带凭据。

`localhost`/`.localhost`、非规范数字地址、IPv4-mapped IPv6 与私网、回环、CGNAT、multicast、reserved、site-local literal 必须在配置门拒绝；没有显式 resolve IP 时只能接受经 HTTPS DoH 验证并钉住的 public A record，DoH response bytes、JSON 顶层类型和 Answer count 必须有独立小 hard bound，DoH 没有 public answer 时必须失败，不能回落到系统 DNS。

模型连接 active update 与 rebuild candidate 必须在 `.env`、`settings.json`、数据库、cache、Redis 或 bridge 副作用前复用同等 HTTPS/public-unicast 上游门禁，不能只依赖 bridge reload 后置拒绝。

proxy 固定允许 `/embeddings`，并且聊天路由必须与当前 `CHAT_API_PROTOCOL` 精确互斥：`openai` 只允许 `/chat/completions`，`anthropic` 只允许 `/v1/messages`；两种路由都只转发其官方客户端生成的 Bearer Authorization，Anthropic 路由还固定转发 `anthropic-version`，并显式拒绝 legacy `x-api-key`，两者只额外生成必要 JSON/Accept/Host header，不得跨协议转发认证头。

bridge 必须在读取 body 前执行 Content-Type、JSON-object 与字节 hard bound；管理 reload 也必须先鉴权再执行独立小 body bound。成功 provider response 必须在压缩读取和有界解压两个阶段限制字节数并验证为 JSON object，provider error body 不得透传。未知 path、协议与 path 不匹配、私网/回环/CGNAT/site-local resolve、非 HTTPS、URL userinfo、无效证书或 hostname mismatch 一律 fail closed。

bridge 管理凭据必须显式非空、不得使用任何固定已知默认值，只能通过进程环境继承，不能进入 argv、公开状态、日志或报告；进程在 token 缺失、空白、含控制字符或命中已知默认 denylist 时必须在 server bind 前失败，不能只依赖外层 launcher 检查。bridge 的 access/error 日志和 4xx/5xx 只能返回固定 route/error code 或异常类型，不得包含完整 path/query、header、body、provider response、临时路径或原始异常文本。

不得把 Authorization、`x-api-key`、模型 payload 或 provider response 写入 curl/config/body/response 临时文件；若安全转发不可用，API/worker 启动或请求必须失败，不能直连绕桥或启用 model fallback。上述传输门不进入 gray observation/rule/hash，也不改变 `model_call_count=0`。

目标 settings 还必须显式覆盖：

```text
edge_distance_protocol
rq_membership_protocol
rq_membership_temperature
edge_projection_protocol
graph_operating_point_protocol
graph_operating_point_optimizer = tpe
enable_auto_tpe
tpe_trial_budget
tpe_startup_random_trials
tpe_good_quantile_gamma
tpe_probe_query_budget
tpe_trial_timeout_seconds
tpe_candidate_pool_size
operating_point_hard_gate_max_edge_density
operating_point_hard_gate_max_isolated_ratio
operating_point_hard_gate_max_hubness_ratio
operating_point_hard_gate_min_structure_recovery_rate
operating_point_hard_gate_max_candidate_latency_p95_ms
dense_knn_k_min
dense_knn_k_max
dense_reverse_b_min_base
dense_reverse_b_max_base
dense_reverse_b_min_doc
dense_reverse_b_max_doc
dense_reverse_b_min_lang
dense_reverse_b_max_lang
dense_min_cosine
dense_strong_cosine
cross_doc_out_quota_min
cross_doc_out_quota_max
cross_doc_min_cosine
cross_language_out_quota_min
cross_language_out_quota_max
cross_language_min_cosine
edge_type_calibration_protocol
agent_coarse_initial_budget
agent_coarse_top_k
agent_mid_per_coarse_budget
agent_coarse_drilldown_mid_initial_budget
agent_mid_initial_budget
agent_mid_top_k
agent_chunk_per_mid_budget
agent_chunk_initial_budget
agent_chunk_top_k
agent_structure_restore_per_chunk_budget
label_dominance_budget
path_distance_thresholds
gray_zone_rule_protocol
gray_zone_observation_cadence
traversal_observation_budget
context_path_summary_budget
```

四个可选协议字段只能引用本地实现 allowlist，不能保存 prompt、模型名、LLM 输出或自由表达式。当前 active identity 固定为：

```text
edge_distance_protocol = edge_distance_log_calibrated_strength_v2
rq_membership_protocol = rq_primary_chain_v1
edge_projection_protocol = membership_q15_layer_type_calibrated_v3
edge_type_calibration_protocol = type_local_winsorized_minmax_v1
```

上述协议字段、`rq_membership_temperature` 和 `rq_residual_tau` 都属于 `rebuild_required`；active settings PUT 不得把相同字段当热加载写入或广播。builder 必须在落库前验证 selected setting 与本地实现一致，并把 protocol/runtime identity hash 传播到 relation、RQ membership、mid/coarse projection、context graph、retrieval cache 与 freshness admission。RQ 主链协议不存在候选数量或概率裁剪设置。

其中改变 chunking、embedding、dynamic dense KNN、bridge quota、edge type calibration、relation graph、RQ codebook、RQ membership protocol、edge projection、graph model endpoint 或 concept graph 的参数属于 `rebuild_required`；改变 chat model endpoint、staged traversal budget、layer top-k、result top-k default、label/cycle/path distance threshold、`gray_zone_rule_protocol`、gray-zone observation cadence 等不改变 active graph 的参数属于 `hot_reloadable`，需要刷新 traversal protocol hash 并失效检索与 QA cache。

`gray_zone_rule_protocol` 只能从本地实现的 allowlist 中选择，不能保存 prompt、模型名或自由表达式。`concept_i18n_enabled` 是热加载功能开关：保存后立即控制检索是否使用已有成功翻译文本，并控制下一次构图是否执行双语派生；它不会自动改写已有 active graph。`query_facet_bilingual_enabled` 是热加载功能开关：保存后立即控制下一次 QA/search planning 的 LLM query facet packet 是否要求中英双语 aliases；它不写 concept graph，不触发 Qdrant 或 graph rebuild。

预算类参数只作为 hard interrupt 或层间输出上限，不参与路径价值排序。

TPE settings 分两层处理。`enable_auto_tpe`、`tpe_trial_budget`、`tpe_startup_random_trials`、`tpe_good_quantile_gamma`、`tpe_probe_query_budget`、`tpe_trial_timeout_seconds` 和 `tpe_candidate_pool_size` 是 automatic optimizer envelope，保存后热加载到下一次 graph build 或下一 trial 边界；它们不直接改写 active graph。

dense KNN、bridge quota、threshold 和 edge calibration 改变 active graph 语义，必须只在 graph build 阶段由自动 TPE 或版本化默认 theta 选择，并在最终 active bottom relation graph 写入时一次性落库。前端导入页在清理数据库/文件数量附近提供自动 TPE 开关、可折叠 envelope 参数和最近一次 auto TPE run/blocking reason；设置页不提供启动、取消、手动切换或独立手动调参入口。

根 `.env` 的写入固定走 `runtime_env_file_cas_v1` + `runtime_env_file_recovery_v1`；根 `settings.json` 使用对称的 `runtime_settings_json_cas_v1` + `runtime_settings_json_recovery_v1`。两者都先在同目录写临时文件并 fsync，再做原子 namespace replace，随后验证 exact size/SHA-256、协议与 path identity；recovery journal、before-image、audit 和 resolved-name cleanup 使用同一 durability contract。两个 writer 使用同一跨进程配置锁；若一个表单同时触及两类 authority，必须先冻结两份 identity/before-image，再按确定顺序替换，并在任一步失败时恢复已经替换的文件。POSIX 的 namespace durability barrier 是 `rename/replace + parent directory fsync`。

Windows 不得把缺失 `O_DIRECTORY` 当成功，也不得用“重开目标文件后 fsync”冒充 rename durability；replace 必须调用 `MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)`，随后以 `O_RDWR | O_BINARY | O_NOINHERIT` 重开目标、fsync、重放 exact bytes，并复核打开前后 path/device/inode/size/mtime。

Windows 的 authoritative unlink 固定先把原名 write-through rename 为同目录 tombstone，再删除非 authoritative tombstone；崩溃残留只允许在同一跨进程 writer lock 内清扫。任一 replace、barrier、identity/content replay 或 recovery cleanup 失败都必须进入 typed rollback/recovery 状态，不得静默继续，审计不得包含 `.env` 或 `settings.json` 原值。

`EMBEDDING_API_PROTOCOL` 即使当前 allowlist 只有单值 `openai`，也属于 `rebuild_required`：普通 active settings PUT 不得提交该字段；它只能随 embedding candidate 进入 shadow vector build、evaluation 与 promotion，并绑定 candidate hash、冻结 vector schema/provider identity、bridge config 与 cache freshness。

由于 current-source allowlist 只有 `openai`，现有 Qdrant collection/payload/outbox 协议不得仅为增加这个同值字段而伪造多协议迁移；一旦开放第二个向量协议，必须先升级并迁移 collection identity、payload hash、outbox target、active pointer、TPE reuse 与 query cache identity，旧 identity 不得跨协议复用。`CHAT_API_PROTOCOL` 仍为独立的 `hot_reloadable` 字段，`GRAPH_API_PROTOCOL` 仍为独立的 `rebuild_required` 字段；三者的 lifecycle 与 payload 不得串写。

### 候选重建生命周期

通用 `rebuild_required` 更新使用 `runtime_settings_candidate_v2`，并以 `runtime_settings_shadow_builds` 保存每个知识库的冻结 before-state、candidate chunk scope、四层 shadow state、构建指标、evaluation evidence/hash 和 promotion/rollback audit。普通 Runtime Settings Save 的全部合法字段写入根 `settings.json`；其中 rebuild 子集写入后必须返回 pending candidate，旧 graph/index 继续只按其冻结构建身份服务，不能把新参数偷偷套到旧派生状态上。

实际变化的 `service_recreate_required` 字段写入同一文件后返回 `requires_service_recreate`，只在显式 recreate 后改变容器形态。candidate/intent 表只保存有界执行计划和审计，不得成为另一份全局 Runtime Settings 真值。

首次构图与普通重建使用同一 candidate/ingestion 事务协议。不存在 active graph 时，executor 以版本化空 before-state 建立首个 relation/RQ/Mid/Coarse/Context Graph。模型 endpoint 与凭据只来自 active Runtime Settings，provider side effect 仍受 fallback=false、预算、事务和 compensation gate 约束。

首次导入或版本化构图恢复以 PostgreSQL batch/recovery/outbox intent 为唯一执行权。Celery/Redis 只提供投递与可见性，不是完成事实；重复投递必须复用 durable task identity，成功必须回放已提交的 chunk/vector/four-layer state，失败必须保留 before-image、write-set 和可重试分类。

candidate 流程固定为：

```text
side-effect-free dry-run
  -> durable stage
  -> bounded per-KB shadow build
  -> measured hard-gate evaluation
  -> one PostgreSQL transaction promotion
  -> durable post-commit activation intent
```

dry-run 冻结 active runtime rebuild slice、chunk scope、vector pointer 和四层 graph ids/hash，并检查知识库数量、文档数量、chunk 数量、immutable source 可用性及 candidate-local settings 合法性；不得写 PostgreSQL、Qdrant、Redis、`.env` 或 `settings.json`。

改变 chunk 参数时 builder 必须从 active immutable `DocumentVersion.storage_path` 真正重新解析和固定切块，产生新的 shadow `DocumentVersion`、`ChunkVersion` 与 chunk ids；改变 vector schema 时复用 vector shadow lifecycle，通过 durable Qdrant outbox 写 candidate collection；仅改变 graph 参数时可以只读复用冻结 active vectors，但不得改写其 record、payload 或 active pointer。

除下述 concept-only 作用域外，两类路径最终都必须构造完整 shadow structure/relation/RQ/mid/coarse/context bundle，不能把旧 chunk 改标签伪装 rebuild。

当 changed keys 严格属于本地 allowlist `mid_concept_extraction_max_model_batches`、`mid_concept_extraction_max_candidates_per_batch`、`mid_concept_extraction_max_tokens_per_batch`、`mid_concept_candidate_keep_threshold`，且 candidate 不改变 chunk/vector/relation operating point 时，允许使用 `runtime_settings_concept_only_scoped_shadow_v1`：冻结并只读复用 staged base 的 exact active relation/RQ state 与 TPE-selected operating point，只构造 state=`shadow` 的 Mid、Coarse 和 Context。

共享 relation id/hash 必须逐字等于 staged base，evaluation、promotion 与 rollback 必须显式声明 `reused_active_graph_layers=["relation"]`；promotion/rollback 都不得切换、降级或改写该共享 relation。若 candidate 不改变 relation operating-point keys，通用 graph-only shadow 同样必须复用 staged base 的 exact operating point，禁止退回版本化默认 theta 造成无关 relation/RQ 漂移。

concept-only candidate 必须以 staged base 的 admitted Mid/Coarse state 作为 `concept_definition_semantic_reuse_v8` 来源，并保存 `runtime_settings_concept_provider_evidence_v1` 的逐层 hit/miss/request 计数。上述 allowlist 路径要求 exact semantic reuse：任一 packet miss、复用审计失败或 source admission 失败都必须在 provider 网络 I/O 前 fail closed；成功 build 的 provider request count 必须为 0，且不得持久化 provider response。该零调用约束只适用于这一语义复用路径，不能伪装成所有 concept rebuild 都禁止 provider。

同一 vector schema 的候选因 latency/resource hard gate 被阻断后，允许下一候选按 `vector_shadow_terminal_concept_semantic_reuse_v2` 复用其已经生成的 Mid/Coarse 定义，但 source 必须先显式进入 `rejected` 或 `superseded`，且 source candidate/build、knowledge base、candidate vector schema hash、完整 chunk scope、attested shadow context 及 Mid/Coarse state 引用必须逐项存在并一致；扫描必须有界并按 `(created_at,id)` 确定性选取。

该路径只把 terminal shadow state 交给 `concept_definition_semantic_reuse_v8` 的 exact packet/profile/protocol 校验，不继承 source lifecycle authority、Qdrant proof 或 evaluation 结论。

选中 terminal source 后必须要求所有 packet exact hit，任一 miss 在 provider client 构造前 fail closed，成功重建的 concept provider request count 必须为 0；若无合格 terminal source，才回到 staged active pointer source 并允许 miss 走受预算约束的 provider。审计必须记录 source kind/candidate/build/state ids、bounded scan count、exact-reuse-required 与 provider-response-persisted=false。

evaluation 只读取冻结 build/result proof，至少计算 vector record coverage、structure recovery、relation retrieval coverage、raw-span citation coverage、构建延迟与估算资源占用，并保存数值 metrics、版本化 hard gates、逐项 evidence hash、完整 input/result hash。任一 gate 失败时状态为 `promotion_blocked`；gray-zone 路径判定仍只属于 deterministic local rule，dry-run/build/evaluation/promotion 的 `gray_zone_rule_decision_model_call_count` 必须为 0，不能以 LLM 评审替代 hard gate。

全部目标知识库的 build 都为 `evaluation_passed` 且冻结 base facts 未漂移时，promotion 才能在同一 PostgreSQL 事务中切换 candidate chunk/version（如有）、vector pointer（如有）和 candidate 实际重建的 graph state；共享 active layer 必须保持原 id/state/hash。`DocumentVersion.is_active` 交接必须在事务内显式执行“旧版本停用并 flush，再激活 candidate”的两阶段切换，避免 partial unique constraint 的瞬时冲突。

事务提交后由 `runtime_settings_activation_intent_v1` 幂等发布新的进程 active slice、组合 runtime version 与 Redis/cache；它不重写已经持久化的 authority file。失败保留 `failed/applying` intent 供 reconcile 重试，不撤销已提交 serving pointer。rollback 反向恢复冻结 chunk/vector/graph scope并创建 rollback activation intent；旧 Qdrant candidate/retained data 的物理删除仍只能走独立 destructive cleanup gate。

未 promotion 的 graph-only candidate 若处于 staged/building/evaluating/evaluation-passed/promotion-blocked/failed，可通过 rollback 语义显式放弃：锁内重验 staged base 未漂移，只把非 base 的 shadow graph rows 退为 inactive，并记录 `unpromoted_abandon=true`。该分支不得修改 active pointer、权威配置文件、runtime version、Redis 或 cache，也不得创建伪造的 activation intent。vector shadow candidate 的放弃继续服从独立 retained-data/cleanup 协议，不能复用这条 graph-only 快捷路径。

candidate builder 由有界 Celery task 执行，单个 candidate 最多处理固定上限的 per-KB build，并在任务入口及每个知识库边界刷新 runtime settings version。graph freshness 的 runtime identity 只绑定 `rebuild_required` canonical slice；hot/service 值改变不得把 active graph 错标 stale。API、设置页和运维脚本必须暴露 dry-run、stage、status、build、evaluate、promote、rollback 与 activation reconcile，并显示 hard gates、metrics、blocking reasons、hash、active side-effect 状态及 gray-zone 零模型调用审计。

### 范围内维护重建

分层维护命令只允许重放当前已接纳 active graph 的某一派生层及其下游，不能借“局部修复”静默扩大为完整 relation/TPE rebuild，也不能代替 `rebuild_required` candidate lifecycle。`scoped_context_graph_rebuild_v1` 固定三种作用域：

| requested scope | 必须复用 | 必须重建 |
| --- | --- | --- |
| `rq_membership` | chunk structure、已校准 bottom relation business facts、active vector schema、TPE operating point | RQ membership/pair diagnostics → mid → coarse → context |
| `mid_concept` | chunk structure、bottom relation、RQ membership/pair diagnostics | mid → coarse → context |
| `coarse_concept` | chunk structure、bottom relation、RQ membership/pair diagnostics、mid | coarse → context |

executor 必须先在知识库级 resource lock 内执行 active graph admission/freshness gate；任一复用层的 state/card/protocol/vector pointer/freshness 不合法时 fail closed，不得自动回退到 `rebuild_context_graph`。RQ scoped replay 通过新 relation generation 复制而非重算 bottom relation rows，复用原 operating point，再确定性重建 RQ；旧 relation generation 保持历史不可变。切换前后必须比较所声明复用层的 UUID-free business hash 与完整 row count，任一变化都回滚 candidate savepoint。

`ContextGraphState.canonical_agent_operating_envelope` 是 build-time 自审计卡，不是把历史 traversal/gray protocol 重新激活为当前执行协议的入口。

admission 必须按冻结卡中的派生 protocol hash 原样重放其 `agent_envelope_hash`、`traversal_protocol_hash` 和 `canonical_protocol_identities`，不得先用当前代码重写历史派生 hash 再误报图事实 stale；当前 retrieval 仍必须单独冻结并使用当前 Runtime Settings 与当前 executor protocol，trace/cache identity 也只绑定当前值。

历史卡只能证明该 graph generation 构建时的内部一致性，不能覆盖当前 gray rule、阈值、hard interrupt 或模型调用边界；真正改变 graph business facts 的 rebuild-required 协议仍按 shadow rebuild/promotion 生命周期处理。

所有新 downstream rows、freshness rows、active vector graph pointer、旧 active state 停用和 durable cache invalidation intent 必须位于同一个 caller-owned PostgreSQL transaction；service 不得自行 commit。builder/LLM/SQL 失败时 candidate savepoint 与 pointer switch 一起回滚。

commit 后才允许执行 Redis knowledge-base invalidation；失败保留 `scoped_context_graph_cache_invalidation_v1` pending intent，并由 `reconcile_scoped_rebuild_cache_invalidations` 幂等重试；运维脚本 `reconcile_scoped_rebuild_cache_invalidations.py` 默认只读，只有显式 `--execute` 才可重放。即使 UUID-free content hash 相同也必须失效 cache，因为 payload 可能引用被替换 generation 的地址 id。

为避免同一已接纳业务输入在 scoped maintenance 中重复支付 Mid/Coarse 定义成本，概念定义允许使用 `concept_definition_semantic_reuse_v8`。

它不是 provider response cache，也不得复制旧 membership、support、edge、node weight、grounding hash 或 graph pointer；只可从当前已通过 active admission/freshness 的 source Mid/Coarse state 投影已经持久化并受 state hash 保护的 schema-valid label/definition/summary 等语义字段，再对当前 packet 重新执行 provider output schema validator、grounded gate、deterministic grounding/membership/support/edge/weight 构建与完整 state hash。

不得保存或恢复原始 provider response、Authorization header、API key 或未知 provider 字段。Mid packet 必须显式保存 UUID-free `rq_prefix_key`；读取缺少该字段的 admitted 旧 packet 时，只能从同一 source state 绑定的 `MidConcept.support_rq_l3_prefix_id -> RQPrefix.rq_prefix_key` 外键重放，并校验 level、relation generation、state 与已存在 internal/packet key，一致后才可用于 scope lookup。

不得退回 generation-specific grounding hash 或猜测地址。

复用键必须同时绑定 layer、definition-semantic business identity、去除纯地址 id 后仍逐字段覆盖完整 bounded semantic candidate universe（最多 6 个 candidate labels、6 个 representative excerpts、6 个 child Mid label/definition/summary excerpts）的 hash 与各类 count、当前 Profile business hash、**实际生效的完整 system prompt hash**（editable Profile system prompt 与服务端不可编辑 output contract 拼接后的 exact UTF-8 bytes）、prompt/schema/projection/schema-repair/reuse protocol、Graph provider protocol/model/credential-free target identity 与 timeout。

reuse identity 不得绑定一次 provider packing 偶然选择的 selected/omitted 子集；同一完整候选集可能因不透明 lineage digest 或容量边界选择不同 sample，但完整语义候选相同才允许复用，任一候选文本/原文 span 变化仍必须 miss。

Mid 定义 identity 必须绑定完整 UUID-free RQ primary membership 语义事实（prefix/chunk business key、score、role、RQ path 与 primary encoding），但剔除每条 membership 内仅用于审计寻址的有界 `support_chunk_edges` 样本；还必须绑定该 prefix 全部 incident bottom-edge 的 UUID-free 事实，但剔除 relation generation 的 `graph_state_hash`。

原始 membership/support-edge hashes、地址 id、grounding hash 与 full packet business hash 继续留在完整 packet、address `identity_card`、projection audit 和新 state hash，不能被定义 identity 替代。

Coarse 键同样必须使用剔除 relation generation hash 的完整 UUID-free incident bottom-edge 事实，并保留原文 chunk business support、structure/source-span business facts、完整 bounded child semantic text、membership/edge count 与分布，但不得把模型无法解释且由新 generation 必须重算的 child grounding digest、membership digest、edge digest 或 full lineage hash 当作 definition semantic identity；这些完整 graph business hash 仍需单独保留在 audit、按当前协议重算并进入新 state hash。

system prompt、Profile、output contract、provider model/protocol、模型可见语义事实、bounded semantic candidate universe 或任一复用协议只要变化一个字节都必须 cache miss；禁止只按 Profile id、concept id、packet id、地址 UUID 或旧 `prompt_protocol_version` 命中。source state 缺少完整审计、重复业务键、持久化 hash/validator 失败、输出字段无法无损投影或当前 active admission 失败时同样 miss/fail closed，不能打开 fallback。

每个新 concept 的 `llm_audit_json.provider_output_audit` 必须记录 reuse protocol、semantic input hash、effective system prompt hash、source state/concept id、source output hash、`reuse_hit`、`provider_called`、provider request count 与 `provider_response_persisted=false`；Mid/Coarse state 与 scoped rebuild audit 汇总 hit/miss/provider request count。全命中时概念定义 provider request count 必须为 0；gray-zone model-call count始终独立为 0，概念定义复用不得取得 gray-zone 权限。

RQ scoped maintenance 允许由运维入口设置 `concept_provider_request_budget_v1` 全链硬上限，并由同一个 request-budget 对象贯穿 Mid 与 Coarse。每个发生 semantic miss 的 provider batch/group 必须在任何网络 I/O 前预留该组 schema repair 的最坏 2 次请求；剩余预算不足时必须抛出 typed budget-exhausted failure、回滚整个 candidate，并保留仅含 layer、miss count、max/reserved/observed request count 与 hash 的安全诊断。预算按最坏请求数预留而不是只在响应后记账，不能因并发窗口、transport 等待或未落库的 candidate 绕过；报告不得把 failure 时的 observed count 推断成 provider 一定收到的完整请求数，也不得持久化 provider response。

无 `--execute` 的 dry-run 只能读取并报告 requested scope、复用层、重建层、目标表和当前 stats，不得初始化模型 I/O、写 PostgreSQL、调用 Qdrant/Redis 或提交事务。

每次 execute 的 `scoped_rebuild_audit` 至少记录 protocol、scope、reused/rebuilt layers、前后 upstream snapshot hash、row counts、transaction/intent、resource lock，以及 `gray_zone_rule_inputs_modified=false`、`gray_zone_thresholds_modified=false`、`gray_zone_rule_protocol_modified=false`、`gray_zone_model_call_count=0`。

概念定义所需的 LLM 不得被计作或取得 gray-zone 裁决权；path-distance 阈值、local rule、observation 与 hard interrupt 语义保持不变。

### 热加载

目标 runtime version：

$$
h_{\Theta}
=
H(\Theta,t,\Delta keys)
$$

`publish_runtime_settings_version()` 写 `RuntimeSettingsVersion`，把 hash 写入 Redis，并发布消息：

$$
msg
=
(h_{\Theta},\Delta keys,source,created\_at)
$$

本地刷新会清理 settings cache、cache manager、retriever 与 lexical index reader 等运行时单例。

### 资料库 Profile

目标 profile 是资料库级 prompt registry 与交互偏好配置：

$$
profile
\to
(system\ prompts,ui,conversation\ preference)
$$

默认 Profile 必须保存当前实现中所有可注册 system prompt 的默认文本，包括 answer generation、JSON response fallback、query rewrite、question perception、query facet extractor、Agent planner、whole-answer reflection reviewer、mid/coarse concept definition、concept i18n、concept edge i18n 和 profile assistant。自定义 Profile 可以按资料库覆盖这些 prompt，以适配不同资料类型、术语风格和生产效果。这里的 answer/reflection prompt 是 editable guidance，不是完整的安全边界；不可编辑 envelope 由服务端代码独立拥有，`answer_grounding_envelope`、`citation_grounding_envelope` 等 Profile key 不属于 active schema。

Profile 不保存 chunking、embedding、dynamic dense KNN、bridge quota、TPE graph operating point、model endpoint、fallback、database、vector-store、cache TTL、worker concurrency 或 budget 参数；这些仍属于 Runtime Settings。Profile 也不能替代 typed action validator、context package、citation verification、support span 校验、graph grounded gate 或 destructive operation guard。

Profile 的写入与公开读取必须 fail closed。`profile_json` 只接受 `user_profile_v1` 的固定顶层字段；`prompt_pack` 只接受本地注册键且值为字符串，`ui_labels` 的值也必须是字符串。任意层级出现 API key、authorization/token/credential、raw provider response/payload，或在注册 `prompt_pack` 之外伪装 `system_prompt/system_content/profile_json` 的键时，整次写入必须拒绝，不能仅依赖后续 response filtering。

详情/复制/绑定等读取在公开或激活持久化 Profile 前，必须重新校验 canonical `profile_json`、`profile_hash` 与独立 `library_type` 事实；同步改写 JSON 和 hash 仍不能绕过敏感键校验，完整性失败返回冲突且不得回显值。create/update/delete/bind 必须先对完整候选及既有持久化 Profile 做上述校验，才允许修改 ORM fact、提交事务、写 lifecycle event、发布 runtime version 或失效 cache；任一校验失败必须 rollback 且所有外部副作用计数为零。

Profile 列表只能经独立的正向 allowlist `profile_summary_to_payload` 投影 `id/name/library_type/is_builtin/is_active/profile_hash/knowledge_base_ids/timestamps`，不得通过“详情 payload 再删除一个字段”构造 summary。

递归敏感字段分类使用版本化 `semantic_sensitive_field_key_segments_v1`。键先做 Unicode NFKC，再按 camel/acronym、数字与非字母数字分隔符切成精确 semantic segments，并规范化已声明复数；分类器必须识别 `auth/authorization`、credential、token、API+key、provider+raw response/payload、system+prompt/content、profile+json 及 password/secret 的大小写、分隔符、复合词与 `archive/backup/blob/bundle/copy/snapshot/value` 后缀组合。

危险存储后缀的优先级高于 count/hash/status/exposed 等观测词，不能借尾缀豁免。普通 token accounting（例如 `token_count`、token budget、tokenizer、chunk size token）只能通过协议内固定 operational allowlist 放行，不能用自由正则或调用方 prompt 扩展。Profile validator 与所有公共 response 最外层必须复用同一分类器及相同递归深度/条目 hard bound；分类器只检查结构键，不读取、记录或回显字段值。

Search、QA/Agent、task status、knowledge-base summary、model settings 与 runtime check 等公共 response schema 必须采用 `extra=forbid` 的闭合契约，并在最外层对嵌套 payload 递归拒绝 `profile_json/prompt_pack/system_prompt`、凭据与 raw-provider 字段。Search 的内部 retrieval row 必须先经声明字段的正向投影，再交给闭合 `SearchResult` 校验；未知顶层字段不得因为 Pydantic 默认忽略而进入响应，`metadata/graph_path` 等开放业务容器仍须执行递归敏感键扫描。

Runtime Settings 只能公开 `has_*_api_key` 布尔状态、protocol/hash/lifecycle 与已声明的运行参数，不能公开密钥值或 provider 原始响应；固定形状的 env-sync、infrastructure、issue、model-bridge 与 lifecycle 子对象也必须使用闭合 schema。Pydantic 与 `packages/shared` 的字段集合必须同步，新增合法公共字段时先显式扩展两端契约，不能重新打开任意 extra；Profile bind 的 `knowledge_base_id` 在两端都为必填字段。

Profile prompt 对链路的影响按生命周期区分：

```text
answer/intent/execution strategy/source location prompt -> hot_reloadable
mid concept/coarse concept/concept i18n/concept edge i18n prompt -> rebuild_required
```

hot_reloadable prompt 更新后影响下一次 search/QA 的规划、来源定位与回答，不触发 chunk、embedding、Qdrant 或 relation graph rebuild，但必须让相关 prompt protocol hash、profile hash、retrieval trace、context package diagnostics 和 cache key 刷新。rebuild_required prompt 更新后只能影响下一次 mid/coarse concept graph rebuild 或 shadow rebuild；active concept graph 不得被静默改写。

Profile mutation/binding 使用 `strategy_profile_lifecycle_v1`。服务端必须按 effective value 对 `prompt_pack`、`ui_labels`、`conversation_preferences` 与 `library_type` 做分类 diff；独立列与 `profile_json.library_type` 必须规范化为同一事实。每个受影响知识库在同一 PostgreSQL 事务中写入 immutable lifecycle card/hash 及冻结的 before/after replay inputs。

事务提交后才执行知识库级 Redis cache invalidation 与 version broadcast；失败保持 `pending_dispatch`，由 API startup、worker beat 或同一事件的显式重试幂等恢复，不能把已提交 Profile 变更伪装成副作用成功。

dispatch 必须以 event row lock 串行化；event replay 必须从冻结输入重新计算 effective diff，并校验 knowledge-base scope、before/after Profile identity、changed paths、lifecycle hash、最新 active binding/Profile fact 与 concept rebuild marker，不能只验证调用方提供的摘要或自洽 hash。审计固定 `gray_zone_rule_inputs_modified=false`、`gray_zone_model_call_count=0`。

代码内置默认 Profile 的版本升级只能由专用 builtin lifecycle reconciler 执行。schema ensure、普通 list/get、delete/bind fallback 等路径只可创建缺失 builtin 或读取既有 before-image，不得先覆盖旧 digest 使 reconciler 丢失 diff。API startup 必须在 schema/binding 完成后执行该 reconciler；Worker 在每个任务边界、任何 Profile 读取或模型/业务副作用前执行 builtin 与 pending lifecycle reconcile，失败即 fail closed，beat 只作为额外周期恢复器。

concept prompt diff 使用 `profile_concept_prompt_rebuild_marker_v1` 在 active context state 上登记 `rebuild_required` 地址和 immutable lifecycle hash，但不修改 active graph business facts、state hash、freshness row 或 pointer，也不就地调用 LLM 重写概念；下一次显式 graph rebuild/shadow rebuild 才读取新 prompt。hot/UI/preference-only diff 不登记 concept rebuild marker。两类 diff 都必须失效绑定知识库的 retrieval/QA/UI/conversation cache，并使下一请求使用新的 Profile/prompt identity。

`conversation_preferences` 的 active allowlist 为：`default_language ∈ {auto,en,zh}`、`citation_strictness ∈ {strict,compact,explain_failures}`、`clarification_style ∈ {concise,detailed}`。`default_language` 决定下一次回答和 deterministic no-context/clarification 文案语言；`clarification_style` 决定证据不足时澄清请求的详略；`citation_strictness` 只控制引用表达的显式程度。任何 preference 都只能作为交互风格输入，不能减少验证预算、改变来源准入和范围约束、绕过 raw span/Context Package，或进入 gray-zone observation/rule/decision hash。

context package 保存 active `profile_hash`，answer prompt、intent/strategy planner、source locator 和 graph concept generator 读取 active profile JSON。凡读取 Profile system prompt 的组件，必须把 `profile_hash` 或由 Profile 派生的 `prompt_protocol_hash` 写入 trace、state 或 diagnostics。

answer/citation 组合协议还必须分别记录 immutable `grounding_envelope_protocol_version`、`grounding_envelope_hash`、editable `profile_hash` 与 composite `prompt_protocol_hash`；envelope hash 只随服务端协议变化，Profile 改动只改变 profile/composite hash。

answer 的这些字段必须进入 `AnswerSession.model_json/diagnostics_json/prompt_protocol_version` 与对应来源准入审计，citation 字段进入每条来源绑定 diagnostics。deterministic gray-zone rule 不读取 Profile 或 grounding envelope，因此其 decision hash 只绑定 gray-zone rule/traversal/runtime protocol 与规范化 observation input，不得绑定可变 prompt 文本；同一 gray observation 的模型调用数仍为零。

Provider 侧 system-prompt cache 使用 `provider_system_prompt_cache_v1`。Anthropic 协议必须始终经官方 Anthropic SDK/Bearer，并把实际生效的稳定 system prompt 作为单独的 system text block，显式附加 `cache_control={"type":"ephemeral"}`；OpenAI-compatible 协议保留标准 system message，依赖 provider 的 exact-prefix 自动缓存。

可编辑 Profile guidance 与服务端 immutable contract 拼成 exact UTF-8 前缀，问题、history、context package、concept packet、request id、时间戳和其他逐请求内容只能位于后续 user/assistant messages，不能污染稳定 system 前缀。answer prompt 只允许由语言、evidence-quality 等有限、可审计变体形成不同缓存键；Profile 或 immutable contract 任一字节变化必须有意失效旧前缀。schema repair 的动态拒绝卡不得伪装为首轮缓存命中。

缓存审计只允许保存 cache mode、system prompt SHA-256/UTF-8 byte count、provider-reported input/output/cache-creation/cache-read/total token counters、延迟与 `provider_response_persisted=false`，不得保存 system prompt 正文、API key、Authorization header 或完整 provider response。

`cache_read_input_tokens > 0` 才能证明 provider cache hit；仅发送 `cache_control`、请求成功或本地语义复用命中都不能代替真实 provider usage 证据。Anthropic-compatible 网关可能把 `input_tokens` 与 cache 字段定义为子集或互斥集合，审计采用 `provider_reported_anthropic_fields_no_cross_field_inference_v1`，不得在没有 provider 明确计费语义时自行相加或计算命中百分比。

### 请求策略与审计

Task、Intent 与 ExecutionStrategy 属于一次请求的数据。LLM 提出入口层、词面和逐层混合权重，本地校验后冻结，执行器不可根据未授权的历史状态替换权重、改变意图或追加模式默认值。策略包含版本、完整任务身份、capability manifest、词面/语义查询身份、启用通道、逐层权重和生效预算。

服务器 Runtime Settings 定义协议、硬预算和索引生命周期，Profile 只提供兼容的提示词与文案。请求策略不回写根配置，不触发后续请求的学习更新；当前只保存行为、耗时、来源与失败观察。

`agent_trace_events.sequence_index` 在每个 run 内从0连续递增，由唯一/非负约束和 AgentRun 行锁保护。所有读取按该序号排序，不用可能相同的 created_at 推断先后。调用意图、策略接纳、入口提名、路径/装包和完成绑定保持同 run/同 scope 外键。

迁移需要隔离旧记录、旧配置与旧执行入口，保留范围见[历史兼容](compatibility.md)。暂停机制的字段和公式不作为新请求契约。

## Freshness、缓存与热加载

### 身份 hash

目标 freshness 由 hash 等式判断：

$$
fresh(layer)
=
\mathbf{1}
\left[
h_{layer}^{stored}=h_{layer}^{current}
\right]
$$

Context graph state 保存 address chunk scope、business chunk scope、contextual-index address/business identity、structure、relation、RQ membership/address、RQ pair aggregate、mid、coarse、runtime、agent protocol、profile、prompt protocol、edge distance protocol、edge projection protocol、traversal protocol、graph runtime identity 与 vector identity hashes。

各层 hash card 至少满足以下覆盖规则：

- structure 绑定完整 structure node/edge/mapping、raw span、page/reading order、bbox 与 parser/layout protocol facts；
- relation 绑定完整 bottom-edge facts、RQ codebook/prefix/membership role 事实、RQ pair aggregate、operating point/TPE/calibration、协议与 vector identity；
- mid/coarse 绑定完整 concept、membership、edge、projection、grounding、definition 与 i18n 派生状态；
- context composite 绑定 business contextual index、所有 layer hashes、runtime/profile/prompt/agent protocol、edge/projection/traversal/graph-runtime protocol 与 vector identity。

gray-zone 累计距离裁决不参与 graph hash 生成。gray rule protocol identity 必须显式记录 `model_call_count=0`；单条 decision 文案、provider/model 状态和 conversation prose 不得进入 state hash，也不得改变 green/gray/red/hard-stop 判定。

旧 composite 与 freshness 记录按原版本只读重放，不能原地删掉旧身份字段再计算新 hash。新构建写入新协议；复用旧构建需显式验证其冻结事实与当前图依赖兼容。请求级 Intent/Strategy 与 BM25 快照另绑定检索身份，不反写图业务事实。

### 可用性核验

目标 stale reasons：

$$
R_{\mathrm{stale}}
=
\{r_i: h_i^{stored}\ne h_i^{current}\}
$$

`ContextGraphFreshness` 保存 layer、state hash、is stale、stale reasons、checked at、canonical hash card 和 diagnostics。`context_graph_stats()` 返回 counts、freshness、grounding、canonical protocol/identity cards 和 traversal contribution。active admission 必须先验证持久化 composite 与各层 card 的内部一致性，再以有界 count proof 防止行数漂移；hash/card/count 不一致时 fail closed，并给出 rebuild/reconcile 指引。

目标 freshness row 协议为 `context_graph_freshness_canonical_row_v2`，一次
context state 必须恰好保存以下九类行：
`contextual_index`、`contextual_index_business`、`chunk_structure`、
`chunk_relation`、`rq_membership`、`rq_prefix_pairs`、`mid_concepts`、
`coarse_concepts`、`context_graph`。BM25 以独立 lexical freshness 核验，不增加物理图层；执行策略只依赖实际启用的索引。每行必须保存 layer-specific state hash、
context graph hash、canonical source card/source-card hash、canonical row-card hash，
以及 `context_graph_freshness_evaluation_v2` 的最近一次检查结果。evaluation card 自身
必须有 canonical hash，并绑定对应 row-card hash、checked-at、is-stale、完整 reasons
和 gray/model-zero 边界。source card 必须引用
构图时已冻结的 canonical layer card 或 contextual/vector identity card，不能用一个
context hash 替代所有层，也不能从当前 hot Profile/provider 状态重写历史 card。

公开 freshness 与 active retrieval/Agent admission 必须消费同一只读 evaluator。该
evaluator 检查九类行的缺失、重复、unexpected layer、显式 stale、state-hash mismatch、
row/source card replay、active pointer 与 layer binding、bounded count proof、协议身份、
active VectorRecord/Qdrant freshness proof，并返回完整、排序、去重的 reasons。历史或
inactive context state 必须显式返回 `context_graph_state_not_active`，不能借当前 active
state 的通过结果宣称自己 fresh。admission exception 冻结 state id、context hash、九层
expected hashes、完整 reasons 与 checked-at；面向客户端的 409 仍使用脱敏 typed contract。

构图/提升在原事务内写入九层 fresh rows。普通 graph GET 和 search 只执行 evaluator，
不得隐式提交 freshness 写事务；完整 mismatch 集合通过显式
`POST /knowledge_bases/{knowledge_base_id}/context-graph/freshness/reconcile`，或 Agent
本身已有的失败审计事务持久化。显式 reconcile 把同一份完整 reasons、统一 checked-at 和
最近一次 evaluation card 写入九层 rows；缺失 row 只能重建为 stale audit placeholder，
不能把底层 graph 自动修成 fresh。请求回滚后的审计重放必须再次核对冻结 state/context
identity 与完整九层 expected-hash 集合，防止把旧失败写到新 active state。

freshness evaluator、row diagnostics、公开 payload 与 reconcile 固定
`model_call_count=0`、`gray_zone_rule_inputs_modified=false`。Profile、provider、
conversation、cache 内容不得进入、覆盖或补判 gray-zone observation/rule input；
资料型检索使用图与索引前完成对应准入；问题理解可以先执行，system capability 直答不依赖资料图的 freshness。

hot-reloadable Runtime Settings 或 Profile 当前值改变时，不应把既有 graph business facts 误判为损坏：持久化 composite snapshot 仍按其构建时 card 做内部校验，当前 hot runtime/profile hashes 进入下一次 retrieval/QA cache key、trace 与 prompt protocol。只有改变 active graph/派生索引语义的 `rebuild_required` candidate 经 shadow rebuild、evaluation 和 promotion 后，才以新 snapshot 替换 active graph card。

### 缓存身份

目标缓存键绑定：

```text
KB / conversation / current user question / filters
Task / Intent / ExecutionStrategy / capability manifest
entry_layer / semantic_query / bilingual lexical groups and surfaces / generate_lexical / hybrid
per-layer weights / prefetch budgets / parent budgets / broad-or-focused
embedding and index text / chunk and structure / relation and RQ / mid and coarse
BM25 active snapshot + tokenizer + df/statistics + scoring (only when enabled)
RQ entry / projection / ranking / fusion / traversal protocols
runtime / Profile / source manifest and representation
```

计划缓存绑定原问题与可用层/索引清单，检索缓存还绑定已接纳的具体策略；不能先命中一个旧结果再伪造本轮 LLM 计划。纯向量无需 BM25 readiness，混合索引变化必须使对应缓存失效。

同一融合权重在不同候选列表上不具有同一数值含义，通道预取、过滤及投影协议必须进入键。源文、代码本、分词、文档版本或快照发布变化时重验来源与 freshness。返回缓存结果仍需确定性来源准入；缓存不存在时执行同一正确路径。

旧请求模式与旧控制状态不进入新缓存命名空间，历史 replay 使用原冻结身份。Cache identity 记录于 retrieval trace，且与模型计划及 active 索引来自同一次冻结快照。

## 数据模型

### 片段与结构表

关系不变量可写作函数依赖：

$$
(document\_version\_id,chunk\_version,chunk\_index)
\to
chunk\_id
$$

目标表：

```text
chunk_versions
chunks
chunk_spans
chunk_coordinates
chunk_context_texts
chunk_structure_nodes
chunk_structure_edges
chunk_structure_mappings
```

### 关系与成员表

目标关系不变量：

$$
edge\in E_{CC}
\Rightarrow
source,target\in chunks
$$

$$
membership(c,p)
\Rightarrow
c\in chunks,\ p\in rq\_prefixes
$$

目标表：

```text
chunk_relation_graph_states
chunk_relation_edges
rq_prefixes
rq_prefix_memberships
rq_prefix_diagnostics
```

目标字段闭环：

```text
documents / document_versions:
  language
  language_source
  language_confidence
  language_detection_protocol_version
  language_detection_hash
  language_metadata_json

chunk_relation_graph_states:
  state_hash
  graph_operating_point_hash
  graph_operating_point_json
  edge_distance_protocol_hash
  edge_type_calibration_protocol_hash
  diagnostics_json

chunk_relation_edges:
  edge_type
  distance
  raw_strength
  features_json
  normalization_stats_json
  source_algorithm
  protocol_version
  edge_distance_protocol_hash
  source_language
  target_language
  is_cross_document
  is_cross_language
  bridge_quota_reason

rq_prefixes:
  rq_level = 1 | 2 | 3
  rq_path_prefix
  parent_rq_prefix_id
  codebook_version
  diagnostics_json

rq_prefix_memberships:
  membership_score
  membership_role
  residual_norm
  membership_entropy
  rank
  membership_origin = primary_chain
  diagnostics_json

rq_prefix_diagnostics:
  diagnostic_type
  diagnostic_strength
  support_membership_mass
  support_chunk_ids_sample
  protocol_version
```

目标 relation/RQ 不变量：

$$
membership(c,p)
\Rightarrow
\mu_{c,p}\in[0,1]
$$

$$
distance(edge)\ge 0,\quad raw\_strength(edge)\in(0,1]
$$

### 概念表

目标 concept 支撑不变量：

$$
\forall m\in V_M,\quad |support(m)|>0
$$

$$
\forall m\in V_M,\quad RQPrefixLevel(m)=3
$$

$$
\forall k\in V_K,\quad RQPrefixLevel(k)=2
$$

目标表：

```text
mid_concept_states
mid_concepts
mid_concept_memberships
mid_concept_edges
mid_concept_definitions
coarse_concept_states
coarse_concepts
coarse_concept_memberships
coarse_concept_edges
coarse_concept_definitions
```

目标字段闭环：

```text
mid_concepts:
  support_rq_l3_prefix_id
  parent_rq_l2_prefix_id
  parent_rq_l1_prefix_id
  support_chunk_ids
  support_chunk_edge_ids
  representative_chunk_ids
  core_chunk_ids
  boundary_chunk_ids
  bridge_chunk_ids
  outlier_chunk_ids
  display_terms_json
  summary
  internal_state_json
  raw_node_weight
  node_weight
  node_weight_normalization_scope
  node_weight_diagnostics_json
  grounding_hash

mid_concept_edges:
  edge_type
  distance
  projected_distance_raw
  projected_strength_raw
  raw_strength_summary
  projection_normalization_stats_json
  edge_projection_protocol_hash
  support_rq_prefix_ids
  support_chunk_edge_ids
  support_chunk_ids
  diagnostics_json

coarse_concepts:
  support_rq_l2_prefix_id
  parent_rq_l1_prefix_id
  child_rq_l3_prefix_ids
  included_mid_concept_ids
  bridge_mid_concept_ids
  boundary_mid_concept_ids
  outlier_mid_concept_ids
  display_terms_json
  summary
  internal_state_json
  raw_node_weight
  node_weight
  node_weight_normalization_scope
  node_weight_diagnostics_json
  grounding_hash

coarse_concept_edges:
  edge_type
  distance
  projected_distance_raw
  projected_strength_raw
  raw_strength_summary
  projection_normalization_stats_json
  edge_projection_protocol_hash
  support_child_mid_edge_ids
  support_chunk_edge_ids
  support_chunk_ids
  cross_prefix_weak_support
  diagnostics_json
```

目标 concept edge 不变量：

$$
edge_M(m_a,m_b)
\Rightarrow
|support\_chunk\_edge\_ids(edge_M)|>0
$$

$$
edge_K(k_a,k_b)
\Rightarrow
|support\_chunk\_edge\_ids(edge_K)|>0
$$

### 检索、问答与策略表

目标审计链：`agent_run → Task/Intent/ExecutionStrategy → retrieval_trace → context_package → source_integrity_admission → answer_session → answer_source_bindings`。Search 在来源准入后返回；能力卡路由独立记录卡片身份。

以下是目标逻辑字段契约，实际表名/迁移与现有 JSON 列的映射须在实现阶段冻结；新逻辑字段不能被写成数据库已存在。

| 对象 | 必需记录 |
|---|---|
| agent run/plan | 协议版本、Task/Intent、原始与生效策略、capability manifest、校验结果、生效预算、阶段和模型 prepared/completed |
| retrieval trace | 策略身份、根入口层、逐层权重、全部依赖索引版本、通道列表及候选池身份、过滤域、已观察范围、终止原因 |
| graph retrieval step | layer/parent、候选/入列/合并结果、三通道 raw score/rank/contribution、fusion score、RQ/BM25 映射见证、路径标签、灰区决定和硬预算 |
| context package | 实际完整 chunk 原文、版本/span、来源角色、结构恢复、来源谱系、未装入范围、估算 token 与实际模型输入计量 |
| source admission | Task/Strategy/trace/package/manifest 关联，原文/版本/范围/路径/预算核验结果及 hash |
| answer session/bindings | 完整或部分回答/不足终态、生成调用、事实单元、handle、答案和原文跨度、文档版本、准入关联及事务审计 |
| runtime / prompt versions | 配置版本与 changed keys、生命周期、提示及 schema 身份；不保存另一份工程参数值 |

每个进入包的图来源都必须有合法入口和路径/结构恢复谱系；chunk 直接入口允许零物理边，但需真实提名记录。每个事实回答必须绑定实际 Context Package 与来源准入，能力卡不能伪造这些对象。

JSON 契约闭合并拒绝未知字段，新增字段必须同步 schema、service、API、shared types、Web、scripts、迁移与 tests。缺失关联、篡改或版本混杂不能通过表面 hash 重算获得准入。旧表和旧字段只按兼容协议读取，不作为新运行流程依赖。

## 事务、并发与安全

### 事务

目标 ACID 约束：

$$
\operatorname{commit}(T)
\Rightarrow
\operatorname{valid}(I(S_T))
$$

其中 \(I\) 是跨表不变量集合。外部副作用前应先写入意图：

$$
external\_write
\Rightarrow
intent\_logged
$$

恢复链路必须通过 batch/job state、heartbeat、diagnostics、compensation logs 和 reconcile scripts 管理恢复。

问答 run 与会话 task state 也遵循持久终态：run 创建并 flush 后、任何检索或模型工作开始前，同一事务先追加带 `run_id` 的 user 消息。尾部待完成 user 是唯一允许的奇数消息形态，prompt history 在终态形成前排除它。成功时追加对应 assistant；run 进入 `failed` 或 `cancelled` 时追加安全终态文案，并且仅当该 run 仍是会话最新 run 才更新 task state、revision 和 state hash。客户端断线后的界面恢复读取这些 PostgreSQL 状态；SSE transport keep-alive 不持久化，也不具有证据或执行授权。

会话列表与消息读取校验 canonical transcript、角色顺序、引用结构和 state hash，但不要求历史来源仍可完成物理重放。来源漂移或旧协议审计失效不能隐藏整个历史，也不能阻止下一轮新检索；`verified_context_reuse` 仍独立重放 AnswerSourceBinding、Context Package、物理来源和本轮准入，任一失败即不授权复用。

新增回答引用必须先通过 `canonical_history_references` 统一 UUID 列表顺序，再把 canonical 的新增项交给来源重放校验。禁止一边持久化 canonical 引用、一边校验原始局部对象；多来源 UUID 的创建顺序没有语义，也不能决定事务是否通过。

单文件删除是 PostgreSQL/文件系统跨系统 mutation，固定使用
`source_file_delete_v1`。入口必须先取得 knowledge-base resource lock；
`source_file_delete` compensation row 以 payload/schema hash 绑定 KB、绝对且
no-follow containment 校验后的 source path、删除前文件 checksum/size、Document
身份、删除前 active document chunk scope hash，以及最多 64 个按 id 排序的 active
`SourceFile` identity cards。超过上限、重复 Document logical path 或任一 before-state
漂移时必须在 unlink 前 fail closed。状态顺序固定为：

```text
intent_committed
-> durable_unlink + parent directory fsync
-> external_applied
-> Document inactive + Chunk deleted + SourceFile deleted
   + complete active ChunkVersion scope rewrite；空库 current version=0
     且 active ChunkVersion 全部停用
   + ContextGraphFreshness stale（同一 PostgreSQL 事务）
-> database_committed/cache_invalidation_pending
-> post-commit strict KB cache invalidation
-> completed/committed
```

unlink/目录 fsync 失败或结果不确定时不得改变 PostgreSQL active scope，也不得吞掉
`OSError`；intent 保持 active，目标缺失可作为同一 intent 上一次 unlink 已发生的
幂等恢复观察，但当 frozen `file_before.exists=true` 时必须先重发 parent directory
sync，不能仅凭 missing observation 推进。目标重新出现或 checksum 改变、同一 KB/path 对应多个 Document、
payload/hash/protocol/owner 不匹配时必须 fail closed。active intent 是中央 KB lock
fence：只有相同 source path、相同 intent owner 和允许的 recovery operation 可以
重入；ingest、rebuild 和其他 delete 不能旁路。删除 scope 事务不得声称同步删除了
Qdrant points；旧向量属于派生 stale/后续 durable reconcile 范围。Redis client
缺失或 SCAN/DELETE 部分失败不能按 cache success 处理，intent 必须保持
`cache_invalidation_pending` 并由 exact-owner retry 重发；普通 cache read 的
cache-miss-correct 宽松语义不因此改变。

### 并发

目标并发控制：

$$
\sum_{i=1}^{n} active_i
\le
B_{\mathrm{resource}}
$$

配置项包括 worker concurrency、model request concurrency、timeout 和 ingestion memory watermarks。长任务应在关键阶段刷新 runtime settings。

### 安全

上传与存储路径必须规范化并验证 storage-root containment；凭据、provider 原文和私有内容指纹不得进入源码、普通日志或报告。模型输出及客户端 metadata 均视为不可信数据。确定性来源和索引校验失败须保持安全错误，不用 fallback 伪装成功。

product path 默认 `ENABLE_MODEL_FALLBACK=false`、`ENABLE_DATABASE_FALLBACK=false`。Settings payload 只暴露 chat、graph、embedding key 是否存在，不输出密钥。

**架构影响：**
- 影响对象：ingestion、indexing、graph rebuild、runtime settings publish、QA 来源准入与绑定提交、Qdrant/Redis side effects、BM25 索引发布/清理 和 destructive scripts。
- 影响方式：事务边界决定 PostgreSQL 状态是否可提交；补偿记录决定外部副作用失败后如何恢复；并发上限决定导入、检索和模型调用是否稳定。
- 传播字段：job state、batch id、compensation logs、status fields、diagnostics_json、runtime settings version、side-effect payload hash。
- 触发条件：长任务取消、外部写入失败、并发资源耗尽、fallback 开关变化、路径校验失败或密钥状态变化时，流程必须阻断、补偿或降级为可审计错误。
- 验收观察点：半提交状态不存在、补偿记录可重试、destructive flag 明确、fallback 默认关闭、日志不含密钥、路径限制在 storage root 内。

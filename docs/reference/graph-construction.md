# 构建协议参考

本文件说明解析、索引、关系、RQ 和概念构建的精确约束。先读[技术白皮书](../technical-spec.md)了解全链路；配置操作见[运维说明](../../infra/README.md)。

## 解析与结构映射

### PreparedDocument

目标解析产物定义为：

$$
D
=
(T,L,S,M)
$$

其中 \(T\) 是文本序列，\(L\) 是布局坐标，\(S\) 是结构对象集合，\(M\) 是 parser metadata。解析目标不是创造语义事实，而是为 chunk 和结构恢复提供可定位地址。

`PreparedDocument` 使用 `prepared_document_layout_v1`。parser 输出的每个 layout item 至少包含稳定 `layout_id`、清洗后文本中的 `char_start/char_end`、`page_number`、`reading_order`、`region_type`、`coordinate_system`、`confidence`、可用时的 `bbox{x0,y0,x1,y1}`，以及保留原坐标/页面尺寸的 metadata。

每个 structure object 至少包含稳定 `structure_id`、`object_type`、清洗后 char span、page、reading order、parent/path、可用时的 bbox 与 parser source。`parser_metadata` 必须记录 parser/source type、native layout 是否可用、清洗后 span remap 方法和 layout/structure 数量。PDF/PPTX/image 等有原生几何的格式不得以等分页 bbox 替代；纯文本流允许 `text_flow_v1` 坐标且 bbox 为空，不得伪造几何。

### 固定 Token Chunk

目标固定切块函数：

$$
C
=
\operatorname{FixedChunk}(D;B,O,\Omega)
$$

其中 \(B\) 是 chunk token budget，\(O\) 是 overlap，\(\Omega\) 是保护对象集合。chunk 边界满足：

$$
|c_i|\le B+\epsilon_{\Omega}
$$

相邻 chunk 的覆盖关系为：

$$
c_i\cap c_{i+1}
\approx
O
$$

### 结构图写入

目标结构映射权重：

$$
w(c,s)
=
\alpha
\frac{|span(c)\cap span(s)|}{|span(c)|}
+(1-\alpha)\operatorname{LayoutOverlap}(c,s)
$$

结构映射的基础可复算项为 char overlap：

$$
overlap(c,s)
=
\max
\left(
0,\min(c_e,s_e)-\max(c_b,s_b)
\right)
$$

coverage ratio：

$$
coverage(c,s)
=
\frac{overlap(c,s)}
{\max(1,c_e-c_b)}
$$

active mapping protocol 为 `structure_mapping_span_bbox_path_v2`，其中准入子协议为 `structure_mapping_address_admission_v2`：

```text
SpanOverlap = char_overlap / max(1, chunk_char_length)
BBoxIoU = max IoU over chunk coordinates and structure coordinates
          constrained to the same page and coordinate system
PathMatch = |normalized chunk path segments ∩ normalized structure path segments|
            / max(1, |normalized chunk path segments|)
```

默认权重为 `alpha_span=0.55`、`alpha_bbox=0.30`、`alpha_path=0.15`。某个分量因源格式没有原生信息而不可计算时，只在可用分量上重新归一化；“不可用”与数值 0 必须区分。`chunk_structure_mappings` 显式持久化 `span_overlap`、`bbox_iou`、`path_match`、`mapping_weight`、`mapping_protocol_version`，并在 metadata 保存 effective weights 与输入 layout ids；结构恢复按 `mapping_weight DESC, depth DESC` 排序。chunk coordinates 必须保存参与映射的 parser-native bbox/coordinate system，而不是空 bbox 或合成等分页坐标。

准入必须先于上述权重计算，并遵守以下 fail-closed 地址规则：

- 所有 mapping 必须属于同一 knowledge base、document 和 document version；跨 scope 即使 span、bbox 或 path 命中也不得准入。
- `paragraph`、`list`、`table`、`formula`、`caption`、`code_block`、`page`、`region` 以及未来未显式声明为容器的节点，只有 `SpanOverlap > 0` 或同页同坐标系的原生 `BBoxIoU > 0` 才能准入。纯 PathMatch 不能创建叶子或布局地址。
- `document` 保持同 document/version 的容器映射。
- `section` 仅在 span 与兼容原生 bbox 都不可计算、chunk section path 与结构 section 的完整 canonical 地址精确一致且该地址在当前 document/version 唯一可解析时，允许纯路径容器 fallback。数值 0 不等于不可计算；重复 section、共享标签、部分 segment 命中或歧义必须拒绝。
- PathMatch 仍参与已经通过地址准入的 mapping 权重与排序，但没有准入、补判或扩大 mapping 集合的权限。

协议升级不得把旧 v1 mapping 静默重标为 v2。改变准入集合会改变 structure hash、contextual index freshness、relation/RQ/Mid/Coarse/context graph、cache 与 citation replay，因此必须通过 clean/shadow rebuild 生成新的完整有效 mapping 集合。完整有效集合不得采样、截断或用代表项替代；消费者需要通过聚合、流式或有界批处理降低内存，而不是简化证据。

### PDF 原生表格地址

`pdf_ruled_table_geometry_v1` 在 PDF 解析时从已冻结快照的原生矢量线检测表格，固定使用 PyMuPDF `find_tables(strategy="lines")`；至少两个行带、两列且存在非空原生文字才建立表格结构对象。它只证明检测区域属于有线表格，不保证已完整识别无边框、扫描或所有单元格。检测器版本、区域、行列数和原生块地址进入 parser metadata 与结构身份，未检测到不能作为原文没有表格的证明。

表格对象只覆盖完全位于检测区域内的原生文字块；不把标题、邻近段落或整页自动当表格。按既有阅读序分成连续块段，每段引用现有正文的精确字符跨度，不重排、重复追加或改写正文；跨栏交错时不能用一个包络跨度吸收其他正文。版面文字、段落对象和原始快照保持原样，表格对象是额外的可追溯地址。检测前后检查取消，失败使用安全阶段错误，禁止吞掉错误后声称结构完整。

旧 active 结构不会因为 parser 协议改变而就地增补或重标；必须通过既有显式重解析或 shadow rebuild 更新结构、后续图身份及缓存。检索门禁发现来源角色缺口时保留该诊断，不能降低角色约束让旧数据通过。验收使用合成有线表格、普通分栏、交错段落及原文跨度/结构落库回归；真实资料只作本地只读核对，不写入 fixture。

### 版本与取消边界

目标版本语义用知识库内最高 chunk version 表示：

$$
v_{\mathrm{target}}
=
\begin{cases}
1,& v_{\max}=0\\
v_{\max}+1,& full\ rebuild\\
v_{\max},& selected\ parse
\end{cases}
$$

取消恢复不能使用 \(v-1\) 推断，应使用解析前记录的 active version：

$$
rollback\_version
=
v_{\mathrm{before\_batch}}
$$

批次取消与进程重启恢复使用
`ingestion_batch_cancel_compensation_v1`。执行器在取得同一知识库的跨进程
resource fence 后、任何文件 mutation 前，必须先提交一条 durable batch recovery，
冻结 `v_before_batch`、完整 active chunk scope/hash、active DocumentVersion 集、
`ChunkVersion` descriptor before-image、四层 active graph state 与 active vector-runtime
graph pointer。每个文件开始前再提交 `ingestion_file_before_scope_v1`；文件成功时，
必须在 Document metadata、DocumentVersion、chunk/structure、VectorRecord 与 Qdrant
owner intent 所属 PostgreSQL 提交中原子写入
`ingestion_file_committed_write_set_v1`。before-image 或 write-set hash 不一致、同一
source path 重复、跨 KB id、未知 Qdrant owner、worker release 未证明或 recovery row
缺失时一律 fail closed，不能由当前 active scope 猜测旧状态。

取消边界按 durable `parse_committed` 分成两段：

1. `parse_committed=false` 时，按文件逆序消费 committed write-set。每组 candidate
   Qdrant point 必须先提交 owner-fenced delete intent；外部删除成功后才在一个
   PostgreSQL 事务中停用 candidate chunks/DocumentVersion/VectorRecord、恢复该文件
   精确旧 metadata、SourceFile、active DocumentVersion 与 active chunks，最后恢复
   `v_before_batch` 和原 `ChunkVersion` descriptor，并以全库 active scope/hash 再验证。
   不允许使用 `target_version-1`、`current_version-1` 或另一个文件的 before-image。
2. 在所有成功文件和失败文件的 per-file 状态均已 durable 后，执行器以一个事务把
   `parse_committed=true` 与 batch phase=`graph_building` 一起提交。此后取消、异常、
   SIGTERM 或重启不得回滚已提交 chunk、structure、contextual vector 或 KB version；
   只能回滚当前图事务，或按 frozen graph before-state 补偿 relation/RQ、mid、coarse、
   context state 及 vector-runtime graph pointer。进入该边界后把 parse write-set 标为
   `retained_after_parse_commit`，禁止再次当成 cancellation delete target。

Qdrant delete 的 intent、外部结果不确定状态和 recovery→intent 绑定必须独立耐久；
重启只能重放同一个 exact-owner intent，不能创建扩大 scope 的新删除。PostgreSQL
事实恢复完成后，Redis/cache invalidation 以独立 durable `pending -> dispatched`
状态重试；Redis 失败不得回滚已恢复事实，也不得被记录为成功。API startup 与 worker
定时 reconcile 都必须先证明旧 worker 已释放，再在同一 KB fence 下幂等消费 pending
metadata intent、batch recovery、Qdrant delete intent 和 cache dispatch。全量重建全部
文件失败时不得设置 `parse_committed`，不得进入图阶段，也不得推进知识库最高版本。

同一 `chunk_version` 可以存在多个 parse attempt，但每个 document 在任一提交态至多有一个 active `DocumentVersion`；active chunk 必须属于该 active attempt，且其 `knowledge_base_id`、`document_id`、`chunk_version` 必须与引用的 Document / DocumentVersion 一致。上述约束既是 service promotion 事务的不变量，也是 PostgreSQL 的 fail-closed 门禁，不能只依赖进程内锁。

用户可见的 upload path 是逻辑 source slot，不是版本事实地址。每次解析必须先把输入固定为 checksum-addressed immutable source snapshot，parser、DocumentVersion、chunk span、context package 与 citation 都绑定该 snapshot/checksum；后续同名上传不得改变旧 attempt 的 raw source。snapshot commit 必须经过 durable rename/目录持久化屏障并应用跨平台只读保护；权限位只是误写防护，不能替代 checksum 验证。

引用生成必须验证 snapshot containment、存在性与内容完整性，不能把 mutable slot 或未经验证的数据库字符串当 citation source。checksum-addressed `source_slots/<digest>` 和 snapshot path 只承担存储身份，绝不是用户可见文件名或目录分组。

upload admission 必须把通过校验的原始 filename 绑定到 logical source slot，并将其不带扩展名的 display title 持久化到 Document；后续重解析或全量重建没有新的 `display_filename` 时必须从 existing Document、历史 metadata intent 或 upload logical slot 依次恢复该标题，禁止用物理 hash path 覆盖。

文件名协议 `nfkc_security_shadow_display_colon_preservation_v3` 使用完整 NFKC security shadow 识别 Windows reserved stem，继续拒绝 ASCII 非法字符、控制字符、真实路径分隔符及版本化 separator-confusable denylist；用户可见 display normalization 只对安全 allowlist 中的 U+FF1A FULLWIDTH COLON 保留原字符，避免合法中文标点因 NFKC 变成 ASCII `:` 后被误拒。该保留字符不得进入物理路径，upload slot 与 snapshot 仍只使用 checksum-addressed 物理名称。

flat upload 的 product partition/tag 必须由同一 display title 派生，不能从 digest stem 或 hash shard 目录派生；raw operator import 仍按其真实相对目录计算 partition。文件列表、目录树、Search source/filter、Context Package document 和 Citation 必须投影同一个 display title；普通产品 UI 不得以 hash、UUID、`本地文件`、`本地资料` 等占位词掩盖缺失身份，身份恢复失败必须进入后端诊断与验收 RED。

上述目录持久化协议必须按实际 `DATA_ROOT` / storage root 的文件系统能力门禁，而不能只按容器操作系统推断。配置加载必须是零目录写的纯读取；`DATA_ROOT` mount point 由部署预置，完整门禁必须发生在 router、数据库 engine/connect、Redis、Qdrant 或模型网络副作用之前，成功后才用 durable mkdir 逐层创建默认 KB/storage/ingestion 子目录并逐级 fsync。

生产启动恢复、worker task 和高层写入口必须先在同一 mount 上完成有界 `file fsync -> rename -> source/target parent fsync -> unlink -> parent fsync` 探针；POSIX 路径必须逐级 `openat + O_NOFOLLOW`，探针与 mkdir/rename/unlink 必须绑定同一 pinned directory descriptor。能力缓存键必须包含 process id、root、device/inode、完整 mount signature 和 protocol version，并使用有界 TTL；fork 子进程不得继承授权。

worker 每个 mutation task 必须在访问 Redis/model bridge/DB 前重新 no-follow 打开根并核对 PID/device/inode/mount signature；fork、identity/mount 变化、首次使用或 TTL 到期时必须执行完整探针，不要求热任务每次重复创建探针文件。`/proc/self/mountinfo` 缺失、读取失败、未找到匹配项、设备不一致或超过有界行数/字节数一律 fail closed；Windows shared bind、FUSE、9p、virtiofs、NFS、SMB、overlay/tmpfs 等未证明 crash-durability 的 family/source 不得仅凭 syscall 成功放行。

native Windows 普通服务进程当前没有已证明的 namespace barrier，不允许通过原始卷权限或环境变量伪装测试能力；测试 fake 只能由显式 fixture 注入，且 production 不得用 fake protocol 创建新 intent。默认 Compose 的 API/worker `/app/data` 使用共享 Docker managed volume；切换已有数据卷必须另行执行显式迁移和恢复验证，不能静默替换在线数据目录。

只读 operator import 与可写 `DATA_ROOT` 必须使用不同的能力协议。`posix_readonly_import_openat_nofollow_fstat_v1` 只允许读取 manifest 完整 allowlist 中的文件，并要求生产 mount options 显式包含 `ro`；目录和每级相对路径必须由 pinned descriptor 逐级 `openat + O_NOFOLLOW` 打开，最终文件必须是单链接 regular file。

读取前后必须重放 root/file device、inode、link count、size、mtime、ctime 和 regular-file identity，manifest checksum 必须与最终 descriptor 读取的 bytes 一致。只读 import 不要求在 source mount 上执行 rename/unlink durability probe，因为该 mount 不是写入事实源；但它也绝不能借此授权任何 `DATA_ROOT` mutation、snapshot commit 或 intent 写入。

`rw` mount、symlink、路径逃逸、manifest/文件身份漂移、缺失 mount proof 或生产 fake adapter 一律在 upload/数据库/模型副作用前 fail closed。

existing document 的 candidate metadata 只能先写 durable intent，不得在 parse 成功前覆盖 active Document。candidate metadata、DocumentVersion、chunks 与 active scope 的 promotion 必须同事务提交；失败、取消与已确认 worker terminate 必须按 intent 恢复解析前 per-document scope。KB 全局版本只允许单调前进，单文件恢复不得回退其他文件已经提交的版本。

同名 upload 的文件替换属于跨 PostgreSQL/文件系统副作用：执行 rename 前必须持久化 target、candidate、backup、checksum 与 phase；commit/kill/restart 后由同一协议幂等 reconcile。target/candidate/backup 的路径规划必须是零 namespace 写的纯计算，首个 durable intent commit 成功前不得创建知识库目录、hash 分片目录或临时文件；commit 成功后才允许按已冻结路径 durable mkdir/write。只依赖 Python `try/except` 或随机临时文件不构成 crash recovery。

上传路由的 path 校验不能替代 worker/executor 校验。异步任务在取得 knowledge-base resource lock 后、读取任何 source bytes 前，必须重新验证 absolute、regular-file、non-symlink、lexical/resolved storage-root containment；队列参数、job metadata 与数据库 path 均不可信。

active chunk scope hash 的冻结协议为 `chunk_scope_complete_address_v2`。

每个 active chunk 的 canonical card 必须绑定 `knowledge_base_id`、`chunk_id`、`document_id`、`document_version_id`、`chunk_version`、`chunk_index`、char/token span、规范化 `section_path`、page range、`text_hash`，以及该 chunk 写入时持久化的 `chunk_schema_version`、`tokenizer_version`、`chunk_size`、`chunk_overlap` protocol descriptor。

card 按 `(knowledge_base_id, document_id, document_version_id, chunk_version, chunk_index, chunk_id)` 排序后与 protocol version 一起计算 SHA-256；调用方传入顺序不得改变 hash。历史 chunk 缺失或带有非法 protocol metadata 时必须作为显式 `missing`/`invalid` descriptor 进入 hash 和 diagnostics，不能用当前设置静默补值。

`chunk_versions` 的 active scope state 使用 `chunk_version_active_scope_state_v2`，必须把“本次 target build descriptor”与“全库 active version/protocol distribution”分开记录。state hash 同时绑定 `chunk_scope_complete_address_v2` hash、按 active chunk version 与 protocol descriptor 分组的冻结 distribution hash，以及 target build descriptor。部分重建失败而保留旧 active chunks 时，diagnostics 必须显示 mixed/missing/invalid protocol counts；不得把混合 scope 声明为单一 chunk size、overlap、tokenizer 或 schema operating point。

**架构影响：**
- 影响对象：contextual index、Qdrant、chunk relation graph、RQ membership、concept graphs、retrieval trace、context package 和 source binding / answer reflection。
- 影响方式：固定 chunk 定义全系统最小地址单位；结构图定义上下文恢复路径；版本策略决定下游索引、图状态与引用审计是否仍然有效。
- 传播字段：`chunk_id`、`chunk_version`、`chunk_index`、`char_start`、`char_end`、`token_start`、`token_end`、`text_hash`、`section_path`、`page_range`。
- 触发条件：chunk size、overlap、tokenizer、parser output、chunk span、document version 或 active chunk scope 变化时，contextual index、relation graph、concept graph 和 cache 都需要刷新或重建。
- 验收观察点：active chunk 版本一致、span 不越界、结构映射覆盖率达标、取消恢复回到 batch 前版本、citation 能回到 raw chunk。

## 索引文本与向量

目标是分离“索引用文本”和“引用用文本”。定义：

$$
x_c^{ctx}
=
\operatorname{concat}
\left(
title(d),
section(c),
page(c),
hint(c),
x_c
\right)
$$

raw citation 仍然是：

$$
cite(c)
=
(document\_version\_id,chunk\_id,char\_start,char\_end,page\_range)
$$

### 上下文化文本

contextual text 改变必须改变 context hash；active hash 不能只绑定 raw chunk 或旧文本版本，完整输入按下文 v2 协议执行。

`hint(c)` 的 active 协议固定为 `local_context_hint_previous_next_nonoverlap_v1`。它是由 executor 从 Chunk Structure Graph 的 `previous_chunk_id` / `next_chunk_id` 确定性生成的索引提示，不调用 LLM，也不是事实证据或 citation 来源。候选邻居必须与当前 chunk 属于同一 `document_version_id` 且处于 active 状态；指针存在但目标缺失、跨文档版本、指向非 active chunk 或 previous/next 顺序与 chunk/span 顺序矛盾时必须快速失败，不能静默改用文本相似度或其他文档补位。文档首尾没有某一侧邻居属于合法空侧。

为避免固定 chunk overlap 被重复写入 embedding 输入，生成器先从相邻 chunk raw span 中扣除与当前 `[char_start,char_end)` 相交的范围。previous 侧只取扣除后 raw span 的末尾 48 个 `symbograph_regex_tokenizer_v1` token，next 侧只取开头 48 个 token；输出顺序固定为 previous、next，总量不超过 96 token。空白按 `normalize_text` 规范化，文本格式固定为：

```text
Previous context: <previous raw-span excerpt>
Next context: <next raw-span excerpt>
```

没有可用非重叠 span 时不写对应行；两侧都为空时 `hint(c)` 为空。每个非空侧必须保存 `role`、`source_chunk_id`、`document_version_id`、选中 raw `char_span`、token count 与 excerpt hash。hint hash 定义为：

$$
h_{\mathrm{hint}}(c)
=
H(
protocol, tokenizer, budgets,
ordered\ source\ ids,
ordered\ raw\ spans,
normalized\ excerpts
)
$$

active contextual text protocol 升级为 `contextual_text_v2`，context hash 的完整输入固定为：

$$
h_{\mathrm{ctx}}(c)
=
H(
x_c^{ctx},
embedding\_text\_version,
local\_hint\_protocol,
h_{\mathrm{hint}}(c)
)
$$

`chunk_context_texts.metadata_json` 必须保存 local hint protocol/hash/source cards 与 context hash protocol；Qdrant payload 和 `vector_records.diagnostics_json` 必须保存同源 `context_hash`、local hint protocol/hash。active chunk 集合的 contextual-index hash 由按 `chunk_id` 排序的 context hash、hint hash、embedding model/text version、vector payload hash/status 计算，并进入 relation state diagnostics/hash、context graph diagnostics/hash 与 retrieval cache key。

生产导入必须在 G0 chunk、previous/next 与 structure mapping 写入后生成 hint，再写 contextual index。独立 graph rebuild 在构建 relation graph 前必须重新生成并比较期望 hint/context hash；缺失或不一致的 context/vector 必须先重嵌入并 upsert Qdrant。hint 文本、source span、生成协议、tokenizer/budget 或 embedding text version 任一变化时，必须把旧 contextual index、relation/RQ/mid/coarse/context graph 标记为 stale，失效该知识库的 retrieval/QA cache，并按 `Qdrant -> relation -> RQ -> mid -> coarse -> context state` 顺序重建；不能只改 `chunk_context_texts` 后继续复用旧图。

### 向量记录

目标 embedding：

$$
e_c
=
f_{\mathrm{emb}}(x_c^{ctx};\theta_{\mathrm{emb}})
$$

vector payload hash：

$$
h_{\mathrm{vec}}
=
H_{\mathrm{vector\_payload\_hash\_v3}}(
e_c^{f32},
chunk\_id,
embedding\_model,
embedding\_dimensions,
\texttt{cosine},
embedding\_text\_version,
chunk\_schema\_version,
context\_hash\_protocol\_version,
context\_hash,
local\_hint\_protocol\_version,
local\_hint\_hash,
collection\_identity\_protocol\_version,
collection\_identity\_digest
)
$$

`vector_payload_hash_v3` 采用独立冻结字节协议，不能委托通用 JSON/stable-hash helper。输入向量的每个元素必须是非 `bool` 的有限实数；先按 IEEE-754 binary32 round-to-nearest-even 规范化，拒绝溢出后得到 `±Inf` 的值，并把 `-0.0` 统一为 `+0.0`。canonical 向量的 L2 范数必须严格大于 `1e-12`，禁止零/近零向量进入 active path。该 binary32 向量同时作为 outbox target 与 Qdrant upsert 的唯一数值表示，使 Qdrant float32 往返后可以重算相同 hash，而不是依赖容差掩盖身份漂移。

canonical byte stream 固定为：协议名 `vector_payload_hash_v3` 的 ASCII 字节、单个 `0x00`；随后依次对 `chunk_id`、`embedding_model`、正 `embedding_dimensions` 的无前导零十进制 ASCII、固定小写 ASCII `cosine`、`embedding_text_version`、`chunk_schema_version`、`context_hash_protocol_version`、`context_hash`、`local_hint_protocol_version`、`local_hint_hash`、`collection_identity_protocol_version`、`collection_identity_digest` 写入 `u64be(UTF-8 byte length) || UTF-8 bytes`；最后写入 `u64be(4 * embedding_dimensions)`，再按向量顺序写入每个 canonical 元素的 4-byte IEEE-754 big-endian binary32。

字符串不 trim、不 case-fold、不做 Unicode normalization，且上述字符串均不得为空。metric 必须是固定小写 `cosine`，collection identity protocol/digest 必须按本节冻结的 active collection identity 重新计算并完全一致。digest 是该完整 byte stream 的 SHA-256 小写十六进制值；必须用 golden vector 与逐字段 mutation 锁定实现。

历史 `vector_payload_hash_v2` 的冻结字节流保持不变：`vector_payload_hash_v2` ASCII、`0x00`，依次长度分帧 `chunk_id`、model、dimension、固定 `cosine`、embedding text version，最后长度分帧 canonical binary32 vector。该 helper 只能用于按显式 v2 protocol 恢复旧 durable intent；active writer、freshness promotion 与新 outbox prepare 均不得产出或接纳 v2。

Qdrant payload、`VectorRecord.payload_hash/diagnostics_json` 与 `qdrant_side_effect_outbox_v2.target_points` 必须保存同一 active protocol/hash/schema/context/hint/collection card。

v2 outbox validator 必须在 durable intent 提交前从 target 的 canonical vector 与 payload 字段重算 collection identity 和 `vector_payload_hash_v3`，并验证所有 target points 的 model、dimension、metric、text/schema version 和 collection name 一致；PostgreSQL freshness/reconcile、committed outbox owner 与 Qdrant point 必须分别从各自保存的卡片重算，并要求三方 hash 完全相同。

历史 `vector_payload_hash_v2` 保持原冻结字节语义，仅允许 v2 outbox recovery decoder 按显式 protocol 读取，不能进入新的 durable write；历史 outbox v1 也只能走独立冻结 decoder，不得借用 active 规则重解释。

Qdrant 点缺失或 `VectorRecord.vector_status` 因已验证补偿被标为非 ready 时，contextual-index repair 必须先尝试从 PostgreSQL 事实源重放，不得无条件重新调用 embedding provider。

只有当前 context/hint/chunk/vector schema、collection identity、canonical embedding bytes 和重算 `vector_payload_hash_v3` 全部一致，且 record protocol reasons 为空或仅为 `vector_status_not_ready`、外部 freshness reasons 仅为 `qdrant_point_missing/vector_not_ready` 时，才允许通过新的 durable v2 outbox intent 精确 upsert 同一 point，并把 record 恢复为 ready。

该路径必须记录 `postgresql_vector_record_qdrant_replay_v1`、`embedding_provider_call_count=0`、目标/恢复计数和 outbox audit；不得改变 embedding bytes、contextual-index business hash、active graph 或 gray-zone 输入。任一额外 stale/protocol/hash/dimension/owner reason 必须退出 replay-only 路径并走正常重嵌入或 fail closed，不能用 PostgreSQL 旧向量掩盖语义漂移。

Outbox envelope 的编码也按协议版本冻结。历史 `qdrant_side_effect_outbox_v1` 使用 `qdrant_outbox_json_utf8_sorted_compact_default_str_v1`：`json.dumps(ensure_ascii=false, sort_keys=true, separators=(",", ":"), default=str)` 的 UTF-8 字节；其 schema hash 固定为 `910a87d94eefd2f81adf1f4ee69fea9202cbde0263a20ded1f195e4bcac9f666`，只能用于只读恢复。

`qdrant_side_effect_outbox_v2` 使用 `qdrant_outbox_json_utf8_sorted_compact_strict_json_v1`：同样的 UTF-8/sorted/compact JSON，但禁止 `default=str`、非字符串 object key、非有限数值和非 JSON 类型；其 schema hash 固定为 `fa90a6e862a11d35bae9c554ff36dc6f8899f99e2727ae50e0212e5475d6b8ba`。

v2 manifest 的 target schema 是不变 core；其中 `vector_payload_hash_protocol` 显式分派独立版本化的 vector identity extension，active v3 extension 要求 context/hint 字段，历史 vector hash v2 extension 只能由 recovery decoder 读取。target 与 before-image 各自按所属版本计算 SHA-256；decoder 必须先按所属版本复核 hash，再执行对应 schema validator。

任何 core 字段语义、extension 分派规则或 canonical bytes 变化必须发布新的 outbox protocol version，不能在 v1/v2 名称下静默改码。

Outbox writer 每个 durable upsert intent 最多保存 256 个 target point；更大的调用批次必须拆成多个独立 intent。`reconcile_qdrant_outbox_sync` 对 active upsert 使用 `qdrant_outbox_active_intent_pk_keyset_v1`：先冻结当前最大 intent primary key 作为 high-water mark，再按 `id ASC`、每页最多 32 行扫描；每页使用独立短事务，并在页边界释放 ORM rows 与 collection client cache。

reconcile 的 action diagnostics 最多保留 128 条，delete recovery diagnostics 最多保留 64 条，同时返回总数与 truncated count。分页、采样和拆批只限制资源占用，不得削弱 committed owner、knowledge-base/chunk scope、collection identity、point mutation lock 或 uncertainty-watch fence。

### 原文 BM25 索引

`source_chunk_bm25_v1` 在 active chunk 原文上建立倒排索引，为请求时的 Dense/RQ/BM25 分层入口提供词面通道。BM25 的分词、统计域、公式、上层投影及融合见[检索协议](retrieval-and-qa.md#bm25-通道)，持久化/发布见[数据协议](data-and-lifecycle.md#bm25-索引生命周期)。

索引只接受有版本和 raw span 的 active chunk。按同 KB 索引快照统计 N、df、文档长度及 avgdl；不把历史 inactive chunk、模型命名或不同 KB 的语料混入统计。解析失败或取消的候选不能发布，旧 lexical 产物不能只更换协议名后直接使用。

倒排 postings、chunk 清单、tokenizer 与统计 hash 完整核对后原子切换 active 索引指针。增删改和版本切换失效该 KB 的混合入口缓存；纯向量请求不读取词面通道。首次启用 BM25 的索引准备可复用已验证原文，不要求重新 embedding 或重建与词面无关的图。

BM25、词面共现及混合提名不创建或加权 chunk 语义边，也不进入 TPE/RQ/概念的底层支撑公式。RQ 的查询相关性只读取已有 codebook 和 primary 映射，不修改构建编码。

## 片段关系图与 TPE

### 状态身份

目标 relation graph state：

$$
S_1
=
(V_C,E_C,\mathcal{R},\mathcal{M}_R,h_1,p_1)
$$

其中 \(E_C\) 是底层 chunk relation edges，\(\mathcal{R}\) 是 RQ prefix address tree，\(\mathcal{M}_R\) 是 chunk 到 RQ prefixes 的 primary membership。\(h_1\) 是 state hash，\(p_1\) 是 protocol version。protocol 是 `chunk_relation_rq_membership_v3`。

目标 state hash：

$$
h_1
=
H(scope_{business}(C),facts(E_C),codebook(RQ),facts(\mathcal{M}_R),pair(RQ),graph\_operating\_point,\ edge\_calibration,\ protocol\_identity,\ vector\_identity,\ p_1)
$$

其中 `facts(E_C)` 覆盖边端点业务键、edge type、方向、raw/calibrated strength/distance、support 与 signal/quota/calibration cards；`codebook(RQ)` 与 `facts(\mathcal{M}_R)` 覆盖 canonical codebook/prefix path、membership score/rank、role、entropy/boundary/residual 诊断；`pair(RQ)` 是独立 prefix-pair diagnostics aggregate。关系层 hash 不得只绑定 count、汇总 stats 或 prefix UUID。

### 候选与边构造

目标候选边只来自多语言 dense semantic evidence：

$$
E_{\mathrm{cand}}
=
E_{\mathrm{dense\_base}}
\cup E_{\mathrm{dense\_cross\_doc}}
\cup E_{\mathrm{dense\_cross\_lang}}
$$

结构邻接、同页、同标题、表格闭包、公式闭包和图注闭包属于 Chunk Structure Graph，不进入 relation edge 的创建和保留。它们在 context package 阶段根据 hit chunk 恢复，保证信息不丢失，同时避免把版面关系误写成语义关系。BM25、term overlap、lexical co-hit、dense+BM25 co-retrieval 和 RQ pair adjacency 不创建 active chunk relation edge。

目标候选通道：

```text
base_dense_candidates: top K_i dense neighbors over active chunks
cross_document_candidates: top Q_i_doc dense neighbors with different document_id
cross_language_candidates: top Q_i_lang dense neighbors with different language
```

目标 edge types：

```text
dense_semantic
dense_cross_document_bridge
dense_cross_language_bridge
```

动态出边配额：

$$
\begin{aligned}
K_i
&=
\operatorname{clamp}(K_{\min}+\lfloor\log_2(1+16m_i)\rfloor,K_{\min},K_{\max})\\
Q_i^{doc}
&=
\operatorname{clamp}(Q^{doc}_{\min}+\lfloor\log_2(1+16m_i)\rfloor,Q^{doc}_{\min},Q^{doc}_{\max})\\
Q_i^{lang}
&=
\operatorname{clamp}(Q^{lang}_{\min}+\lfloor\log_2(1+16m_i)\rfloor,Q^{lang}_{\min},Q^{lang}_{\max})
\end{aligned}
$$

动态反向入边配额：

$$
\begin{aligned}
B_j^{base}
&=
\operatorname{clamp}(B_{\min}^{base}+\lfloor\log_2(1+16r_j)\rfloor,B_{\min}^{base},B_{\max}^{base})\\
B_j^{doc}
&=
\operatorname{clamp}(B_{\min}^{doc}+\lfloor\log_2(1+16r_j)\rfloor,B_{\min}^{doc},B_{\max}^{doc})\\
B_j^{lang}
&=
\operatorname{clamp}(B_{\min}^{lang}+\lfloor\log_2(1+16r_j)\rfloor,B_{\min}^{lang},B_{\max}^{lang})
\end{aligned}
$$

其中 \(m_i\) 表示同层归一化出边证据量，\(r_j\) 表示同层归一化入边接纳容量。二者是两个独立信号，统一配额协议为 `dynamic_knn_reverse_quota_signals_v3`，`quota_signal_scale=16`。二者只参与配额计算和确定性 tie-break diagnostics，不表示 query relevance，也不直接进入 raw strength。

`chunk_node_quality_intrinsic_v2` 的节点质量 \(q_i\in[0,1]\) 使用 availability-aware weighted mean：token sufficiency、parser coordinate confidence、finite non-zero vector integrity 和 lifecycle/text-hash integrity 的默认权重分别为 `0.35/0.25/0.25/0.15`。缺失 parser coordinate 时该分量标记 unavailable 并只在其余可用分量上重归一化；向量缺失是质量值 0，不得伪装成 unavailable。

text-hash integrity 必须按 `chunk_text_sha256_normalized_v1` 对当前 `chunk.text` 重算 `sha256(normalize_text(text))` 并与存储的 `chunks.text_hash` 精确比较；非空错误 hash 与空 hash 的完整性值均为 0，并分别记录 `mismatch` 与 `missing`，不得用“非空”代替一致性校验。

`relation_out_evidence_mass_v2` 固定包含以下类型内、query-independent 分量：

```text
chunk_quality          0.24  # token/非空字符充分度
semantic_density       0.24  # 当前 scope 中 top-4 正 cosine 的均值
span_citability        0.20  # char/token span、raw ChunkSpan 与原文地址完整性
node_quality           0.16  # chunk_node_quality_intrinsic_v2
structure_coverage     0.16  # G0 mapping strength/type coverage、coordinate confidence、previous/next continuity
```

`relation_in_acceptance_capacity_current_scope_v3` 固定包含与出边质量不同、且能从当前构图输入重算的信号：

```text
structure_coverage       0.20
bridge_coverage          0.20  # 当前 threshold-eligible dense bridge opportunity
boundary_stability       0.18  # 当前 G0 previous/next continuity
node_quality             0.18
hub_headroom             0.24  # 1 - 当前 threshold-eligible inbound pressure / scope max pressure
```

所有分量与最终值保留 6 位小数。availability-aware weighted mean 只对真实不可用分量重归一化；可计算但数值为 0 的分量仍必须参与，不能以 `or default` 抬高。`bridge_coverage` 的当前机会只来自满足对应 edge-type threshold 的跨文档/跨语言 dense 候选；`hub_headroom` 从尚未执行出边 quota 的完整 threshold-eligible 有向候选域计算。structure 分量读取当前 G0 `mapping_weight`、node type、parser coordinate confidence 与 previous/next，不允许用是否存在 `section_path` 的布尔值替代。

RQ membership 在 bottom relation graph 之后派生，因此不得形成循环 prior；兼容旧 state 的 RQ id/hash/role/entropy 可以保存在 `historical_rq_quota_diagnostics_only_v1` 审计中，但 quota diagnostics 必须明确 `historical_rq_prior_used_for_quota/edge_gate/raw_strength=false`，且这些历史值不能进入 signal-scope hash。

`NodeQualityPair(i,j)` 只能使用两个端点的 `chunk_node_quality_intrinsic_v2` 平均值；不得复用 \(m_i\)、\(r_j\) 或旧 `node_mass`。每个候选 channel 必须记录 source out-evidence mass、实际 out quota、target in-acceptance capacity、实际按 edge type 分桶的 inbound quota、各信号 component card、protocol version/hash 与 signal-scope hash。

relation state、graph operating point、TPE protocol/trial diagnostics、retrieval trace 和 cache key 必须传播三个信号协议及统一 quota protocol hash；协议变化必须改变 relation state hash，并触发正常 rebuild/shadow promotion。

下一次 relation rebuild 的 operating-point prior 只能读取 relation protocol、raw strength、node quality、out/in signal、quota protocol version/hash 及 operating-point hash 全部匹配的 state；任何一项不匹配必须记录拒绝原因并视为 unavailable，不得只迁移旧数值后静默套用新协议标签。该 operating-point 数值 prior 不授权读取上一版 RQ membership 参与 bottom-edge gate。

普通入边、跨文档入边和跨语言入边分开计数，避免同语言、同文档密集区域挤掉 bridge 候选。

edge-type 接受规则：

$$
\operatorname{accept}_t(i,j)
=
\mathbf{1}
\left[
\cos(e_i,e_j)\ge\tau_t
\land
\left(
\operatorname{mutual}_t(i,j)
\lor
\operatorname{reverseAccepted}_t(i,j)
\lor
\cos(e_i,e_j)\ge\tau_{\mathrm{strong}}
\right)
\right]
$$

目标边特征：

$$
\phi_{ij}
=
\left[
\cos(e_i,e_j),
\operatorname{RankScore}_t(i,j),
\operatorname{Reciprocity}_t(i,j),
\operatorname{NodeQualityPair}(i,j),
\operatorname{BridgeFlags}(i,j)
\right]
$$

`RankScore` 的规范协议为 `channel_percentile_rank_v1`。对 source chunk \(i\) 和候选通道 \(c\in\{base,doc,lang\}\)，先取通过该通道阈值、尚未截取出边 quota 的完整候选集合 \(\mathcal C_i^c\)，并记：

$$
N_i^c=|\mathcal C_i^c|,
\qquad
r_i^c(j)=1+\sum_{k\in\mathcal C_i^c}\mathbf 1\left[\cos(e_i,e_k)>\cos(e_i,e_j)\right]
$$

这里 \(r_i^c(j)\) 是 competition rank：未截断 cosine 完全相同的候选具有相同 rank。quota 选择顺序固定为 `cosine desc, target_chunk_id asc`；`target_chunk_id` 只解决并列候选的确定性截断，不改变 rank。被通道 \(c\) 提名的候选满足 \(N_i^c\ge1\)，其通道分数为：

$$
\operatorname{RankScore}_c(i,j)
=
\frac{N_i^c-r_i^c(j)+1}{N_i^c}
$$

若同一有向候选同时被多个通道提名，则最终分数为：

$$
\operatorname{RankScore}_t(i,j)
=
\max_{c:\,j\text{ is nominated by }c}
\operatorname{RankScore}_c(i,j)
$$

因此 RankScore 使用相对候选位置和候选总数，虽然排序原始信号来自 cosine，但不等同于 cosine semantic 分项。RankScore 持久化前四舍五入到 6 位小数；raw-strength 各连续分项先持久化到 6 位小数，再按固定系数组合、截断到 \([10^{-6},1]\) 并四舍五入到 6 位。每条边的 `features_json` 必须记录每个提名通道的 `candidate_channel/rank/ordinal/candidate_count/selected_limit/selected_count/rank_score`，并记录最终 `rank_score`、`rank_score_protocol_version/hash`。

`raw_strength_components` 必须分别记录 semantic、reciprocity、rank score、node quality pair、系数、计算结果和协议 version/hash，足以重放 raw strength。若同一无向边存在两个方向的提名，必须保留各方向分项并明确最终取胜方向，不能让分项与最终 `raw_strength` 错配。

节点证据量与接纳容量只能决定 quota 和确定性 tie-break diagnostics；node weight 不进入 RankScore，也不表示 query relevance。rank/raw-strength protocol version/hash 必须进入 `graph_operating_point_json`、relation state diagnostics、TPE protocol audit、retrieval trace diagnostics 与相关 cache key；协议变化必须改变 relation graph state hash 并触发下游派生图重算。

raw strength：

$$
\operatorname{raw\_strength}_{ij}
=
0.75\operatorname{semantic}_{ij}^{t}
+0.15\operatorname{reciprocity}_{ij}^{t}
+0.07\operatorname{rank}_{ij}^{t}
+0.03\operatorname{nodeQualityPair}_{ij}
$$

其中：

$$
\operatorname{semantic}_{ij}^{t}
=
\operatorname{clip}
\left(
\frac{\cos(e_i,e_j)-\tau_t}{\tau_{\mathrm{strong}}-\tau_t},
0,
1
\right)
$$

cross-language 和 cross-document edge 不因 bridge 身份额外降权；它们通过独立阈值、独立入边配额、mutual/reverse gate 与 edge-type calibration 控噪。quota 只负责候选召回机会，不进入 raw strength。

### TPE 工作点校准

#### 精确构建计算与性能协议

固定 chunk 与结构节点映射可使用 exact address prefilter：取 raw span 相交节点、具有同页同坐标系有效 bbox 的节点、document 容器与无 span 的 section 候选的并集，再执行原完整 admission/评分。唯一 section path 判定仍检查全体 section，不能通过预筛选丢掉歧义；跨页 span 与页面 bbox 的候选必须取并集。该索引不截断结果、不改映射协议，必须与全扫描的持久映射事实逐项一致。

canonical fact-set hashing 使用有界 UTF-8 外部归并排序，逐条序列化 canonical fact 并流式写入同一 canonical envelope；必须与原全量实现输出完全相同的 SHA-256，不改变协议或删减事实。单缓冲上限 8 MiB、归并打开文件不超过 32。

已规范化的子树可以用不可变 JSON 容器在本次序列化中复用，必须保持相同引用替换、无序集合排序、有限数校验和完整 SHA-256；不允许通过标记可变对象来跳过重新校验。

同一 trial 的节点信号 card 在候选生成时可冻结一次并由多条边共享；canonical 投影只对不可变 card 及同一不可变引用映射/字段上下文复用。更换引用环境或输入 card 必须重新规范化，数据库序列化与重读仍包含完整 card。 不可变 canonical 子树的 UTF-8 字节可以在该对象生命周期内复用；组合编码必须逐字节等价于标准 JSON 的 ensure_ascii=false、sort_keys=true、紧凑分隔符与 allow_nan=false，不可改为摘要替代子树。PostgreSQL canonical membership 审计更新按有界批次落库，保留完整原事实与事务。

数据库 JSON 传输可使用已声明的原生编码器加速，前置拒绝非有限浮点数，对大整数、非标准容器等保留标准 JSON 语义；数据库重读仍使用标准解析。该传输优化不替换 canonical hash 的标准编码器，必须验证有限 float64 精确往返及重新读取后的完整事实哈希。relation hash 的数据库读取与 membership 审计更新分批进行，业务地址索引不得为获取地址额外构造完整 edge facts。

同次构建已经 flush 的边/membership 可由所属 Session 提供不可变事实对象给 canonical hash；必须核对数据库完整行清单、标量字段、完整 JSON 传输值的数据库 SHA-256 与原事务归属，拒绝 dirty、缺失、外来或可变 JSON 对象。membership 的 canonical fact 可以在首次插入前准备；最终阶段只有在数据库行校验通过、不可变业务输入快照一致、chunk/prefix/edge 引用映射全部一致时复用该 fact。跨事务、重启与最终独立重读校验必须重新加载并计算完整事实。

`relation_node_signal_pool_v1` 在同一 relation state 内按 chunk 与 signal kind 保存完整节点信号池；边及双向 contribution 中重复的四类 signal card 改为同状态引用，边自己的方向、rank、强度、配额和支撑仍完整保留。信号池以带长度和 SHA-256 的有界无损 JSON/zlib 包存入 relation state，独立重读必须校验包完整性与全部引用；图业务哈希单独绑定完整 UUID-free 信号池事实，不能把压缩字节或地址 checksum 当业务身份。诊断 trace 按引用展开原 card；候选试验继续使用完整 card，正式持久化才做无损因子化。relation 协议升为 `dense_only_chunk_relation_graph_v10`，不存在跨状态引用或减少 contribution/support 的路径。

构建 admission 必须检查 Worker 队列就绪。周期性的无参 reconciliation/backfill 任务只表示“重新扫描当前持久事实”，每类保留最新待执行实例即可；它们必须设置不超过调度间隔的消息有效期，防止停机期间的重复维护任务阻塞新的导入。历史积压只能通过显式 dry-run/execute 运维入口合并，严格匹配已知维护任务、空 args/kwargs、无 chain/chord/callback；所有导入、带参及未知任务均保留。被取消的排队验收不得重置计时后计作同一次成功，必须另建完整冷构建尝试。

有界候选计算、RQ 和批写阶段使用 `runtime_settings_read_scope` 冻结同一根 `.env` 的已验证设置对象，避免内层每条边反复读取宿主挂载文件。它不创建 shadow override 或第二份参数文件；必须在 trial/阶段刷新前退出，下个作用域重新验证根文件身份，不能跨任务静默保留配置。

解析提交后释放已持久化的 ORM before-image，再读取控制身份进入构图；逐文件与阶段边界允许回收不可达对象和显式记录 allocator trim。RQ membership/pair 写入按 256 行分批；persisted integrity 必须从数据库投影读取所需的标量业务事实，不得为复核几个字段全量解码所有重复 feature/vector JSON。批次每五秒保存内容无关的 performance_live，异常退出时明确为未完成快照；最终 performance 只能在执行器退出并完成统计后产生，缺最终统计不能通过验收。

`graph_build_numeric_float64_v1` 使用完整向量域、float64 分块矩阵运算和稳定同分顺序；不使用 ANN、降维或减少 trial 代替优化。阈值、候选排序和最近中心的舍入临界样本必须由标量参考复核。 初始化使用的 6 位小数及输入哈希使用的 12 位小数可按块舍入；缩放整数限定在精确可表示域，靠近半值边界、超域及非有限输入必须回到原 Python round，回归要求舍入结果逐位一致。数值实现身份绑定 relation/RQ 构建身份，旧数值协议的派生状态不得冒充新构建。

每个 relation build 使用独立、可丢弃的 `GraphBuildWorkspace`：冻结 chunk/vector、语言/结构事实和运行配置，模长、两两相似度、邻居顺序及 RQ 编码每次构建准备一次；trial 仍逐轮重新计算其阈值相关信号、配额、强度和校准。选中的候选只有在输入、theta、校准及候选摘要重新核验后才能供正式落库复用。进程内缓存不承担生命周期权威，重启后从 PostgreSQL 身份重新准备。工作区默认 256 MiB、数值线程默认 2，超限采用构建专属临时映射文件，完成、取消及异常均清理；超出受控临时空间时 fail closed。

节点固有质量（文本、坐标、向量有效性及 lifecycle）、静态协议卡和同一 probe 集的结构正例在工作区内只计算一次；结构正例的复用键绑定 chunk/probe/business-key 范围，返回副本，不能被后续 trial 改写。每次进入候选计算重新核验完整 float64 向量身份，向量原位变更也必须拒绝。审计继续记录每轮候选提名数量，不能只在失败 trial 中记录。

选中候选完成身份/参数/校准核验并物化为关系边后，在完整 RQ 计时内、物化前释放 best/last trial 的候选对象；清理成本计入专项，持久化 TPE 审计和关系边事实继续完整保留。RQ 所需训练/编码缓存仍保留到当前构建结束，不能以保留不再使用的候选副本增加后续阶段的内存与回收成本。

已冻结的 JSON 在实际数据库序列化时可以记住该 serializer 对完整字节计算的 SHA-256；持久行验证仍逐行比较 PostgreSQL 保存的完整 JSON 文本摘要、标量列和行集合，只省去客户端重复序列化。缓存绑定深度不可变对象及 serializer 身份，不缓存整份大 JSON 字符串；未冻结值、独立重读或 serializer 变化必须重新计算。

TPE 共享准备耗时单列；trial latency 门禁保守计入共享准备与当前候选计算，run 墙钟只累计一次实际准备。不得把工作移到计时边界之外隐去成本。主要计算复杂度由 `O(T*n²*(d+log n))` 变为 `O(n²*d+n²*log n+T*(n²+M*log M))`；精确两两域仍为二次空间，可映射到受控临时文件。

RQ 保持三层、每层最多六个中心、最多八次 K-means 迭代、原初始化/空簇/收敛规则。 `rq_diagnostics_float64_le_base64_v1` 将每个已持久化 membership 的 residual/reconstruction 向量无损保存为 little-endian float64 字节的 Base64、维度与 SHA-256；不量化、不降维、不删向量。读取必须校验闭合协议、长度、校验和与有限性，数值消费者还原为相同 float64。规范事实哈希绑定完整打包对象；RQ index 版本切换到 `residual_quantized_kmeans_primary_v5`，每个构建记录存储协议，不能混用不同构建身份。旧数组格式只依其原冻结图身份读取和审计，新构建统一写入新格式。

训练与编码向量化，TPE 与正式 RQ 可复用相同冻结向量范围的结果；最终边支撑、成员角色和诊断必须重新依赖已选择的底层边。支撑边先按既有业务顺序排序一次，再构造 chunk→有序支撑边索引，保留原前十六条结果，把该步骤从 `O(n*E*log E)` 降为 `O(E*log E+n)`。完整 RQ 验收包含本次共享训练/编码、前缀 pair diagnostics、support integrity、边界标注、落库与 canonical membership/prefix/pair 事实哈希，不仅计 K-means 或 prefix 函数。共享训练可同时计入 TPE/RQ 各自专项预算，但总墙钟只累计一次；共同 relation/RQ 哈希无法拆分时整体保守计入 RQ。

完整 RQ 时间上界包含 `O(L*I*K*n*d + L*n*log(n) + E*log(E) + Σ P_l²*(d+log(P_l)) + L*E*log(E) + H*log(F))`。其中 P_l 是各层前缀数，H 是完整规范事实的总序列化字节数，F 是事实行数；哈希/数据库 I/O 不得从复杂度中省去。primary chain 的同层成员交集通过一次 chunk/level 索引得到，消除逐前缀对反复扫描成员集合。逻辑内存包含 `O(n*d + L*n + P*d + P² + L*E)` 以及证据载荷、固定数值块和有界数据库/归并缓冲；与 TPE 同作用域时还持有 `O(n²)` 精确相似度工作区（必要时映射到临时文件）。

构建执行控制在数值块、迭代、数据库批次和模型请求边界检查取消与硬时限，目标两秒内观察取消、十秒内停止启动新工作；补偿单独计时。每五秒及阶段切换持久化进度，关键审计写失败不得静默吞掉。性能摘要只含计数、耗时、资源和协议，禁止 provider 原文、凭据和私人资料指纹。

冷构建禁止复用旧解析、向量与概念结果，允许同次构建内部复用；服务端 prompt cache 必须如实披露。验收从任务接纳到四层图提交、缓存失效与 freshness 通过，包含排队、外部请求和重试：总时限 1800 秒，TPE 不超过 60 秒。完整 RQ 专项按 `max(60 秒, min(完整 RQ 参考耗时))` 验收：先以相同实现、数值协议、运行配置及完整输入范围顺序预跑三次，全部通过生产 RQ 的支撑、primary、前缀及持久事实校验，再取三次完整耗时的最小值并冻结门槛。参考运行复用现有向量、调用生产 relation/RQ service、最终回滚整个事务，不调用模型，不属于冷构建；每次使用新工作区，训练/编码、诊断、数据库写入和共同 relation/RQ 哈希都计入。参考报告只保留规模、计时、代码实现身份及数值资源配置，不保存私人资料指纹。冷验收接纳前读取并校验参考报告，门槛写入批次控制审计；不得依据本次验收自身耗时追涨门槛。实现、规模或数值资源配置变化时参考报告失效。TPE/全链路时限和全部质量要求不随 RQ 门槛改变。

全量文件必须成功，Worker 峰值内存目标 3 GiB、项目栈 6 GiB。文件/chunk/边吞吐与模型请求 QPS 分开计算；P50/P95/P99 使用 nearest-rank 并报告样本数，单次构建与六个 trial 不伪装稳定尾延迟。临时 benchmark 写入被忽略的 output，核验后清理。

计算块延迟分别记录 `similarity_block` 与 `rq_distance_block`，不能用阶段均值推算块的分位数。块计时嵌套在原阶段中，不另加到全链路墙钟；实际冷验收必须观察到两类成功样本并披露数量。

资源采样器与容器使用不同 UTC 时钟，实时采样年龄允许最多 2 秒的跨时钟偏差，仍要求 15 秒内更新且相邻采样间隔不超过 2 秒，并披露实际年龄。若只有该时钟边界误判，可对同次构建原始 JSONL 做独立的 `resource_window_replay_v1` 复核：窗口从接纳前冻结记录或总耗时反推的更早起点开始，到原始报告写入后结束，两端再各保守扩展 2 秒；必须有完整首尾覆盖、连续样本、全部项目服务、零采样错误，并从该窗口重新计算内存/CPU。原失败报告保留。文件、构图、质量、时间或数值门槛失败不得借资源复核改判，仍需新的完整冷构建。

`rq_best_complete_time_v2` 参考计时必须调用生产 TPE 的共享 RQ 输入校验/训练/编码入口，随后正式 RQ 复用该结果；以 `rq_shared_prepare + rq_complete` 计量，和冷验收保持同一范围。缺少共享准备的旧参考报告不得用于冻结时限；报告分别披露参考与验收的 E/P，避免把只具有相同 n/d 的边域当成完全相同工作量。

批量冷解析可以提前准备下一份不可变源快照及其解析结果，以单个有界预取槽覆盖当前文件的模型等待。预取只产生本轮可丢弃计算结果，不写文档/版本/active 派生事实；消费时重新验证源路径、快照身份与解析配置，元数据意图、before-image、结构/向量写入和版本提交仍按原文件顺序执行。取消或异常必须停止预取并清理计算对象，不能复用跨批次解析结果或共享 SQLAlchemy Session。

解析清洗必须覆盖结构对象 title 与展示 path，正文清洗后同步原有 span remap；原始 source snapshot 不改写。结构图文本列入库前拒绝残留 NUL，失败先回滚失效事务再执行恢复，安全错误记录不能包含 SQL parameters 或源文片段。

PDF 页面中已有的 `ocr_image_errors` 是结构布局诊断，公开 `ContextStructureLayoutAudit` 必须以可选字符串列表显式承接，不能因遗漏字段而拒绝整个已持久化 Context Package。它不成为引用证据或模型事实来源；其他未声明布局字段仍拒绝，读取时继续验证原始 package/citation 字节形状、span、trace 及 PostgreSQL provenance。

Dynamic KNN、reverse quota、bridge quota、semantic threshold 与 edge calibration 参数通过自动 TPE Bayesian Optimization 选择 active bottom relation graph 的 operating point。TPE 是构图阶段的轻量数值优化器，不是独立产品入口；它不调用 LLM，不重新 embedding，不构建 mid concepts，不构建 coarse concepts，也不生成任何临时候选持久图谱。

TPE 的触发只由运行环境和构图批次决定：

```text
ENABLE_AUTO_TPE=true:
  仅当本批次推进知识库最高 chunk version 时，在 chunk/vector ready 后、写 active chunk relation graph 前自动运行轻量 TPE。

ENABLE_AUTO_TPE=false:
  使用当前 active operating point；若不存在，则使用版本化默认 operating point。
```

触发锁定为 chunk 最高版本号递增：空库首次成功解析产生 v1 时可以运行；全量重建成功推进到 vN+1 时可以运行；普通选中文件重解析、同版本补解析、普通搜索、QA、Agent、导入页/设置页保存和日志抽屉打开都不得触发 TPE。前端导入页提供全局热加载开关、envelope 参数和最近一次 auto TPE run 只读状态；开启开关本身不 retroactive 触发当前版本调参，必须等待下一次最高 chunk version 递增。

TPE 只在底层关系图参数空间 \(\Theta_G\) 中选择构图常数：

```text
K_min, K_max
B_min_base, B_max_base
B_min_doc, B_max_doc
B_min_lang, B_max_lang
dense_min_cosine
dense_strong_cosine
cross_doc_out_quota_min, cross_doc_out_quota_max
cross_doc_min_cosine
cross_language_out_quota_min, cross_language_out_quota_max
cross_language_min_cosine
edge_type_calibration_protocol
calibration_params.lower_quantile
calibration_params.upper_quantile
calibration_params.min_span
calibration_params.strength_floor
```

一个 operating point 记为：

```text
θ = {
  graph_operating_point_protocol,
  edge_distance_protocol,
  edge_type_calibration_protocol,
  dense_knn,
  reverse_quota,
  bridge_quota,
  type_thresholds,
  calibration_params
}
```

`θ` 的每个字段必须可序列化、可 hash、可落库、可回放。`dense_knn` 控制 base dense candidate fan-out；`reverse_quota` 控制按 edge type 分桶的入边接受上限；`bridge_quota` 只控制 cross-document 和 cross-language candidate 进入机会；`type_thresholds` 控制不同 edge type 的最小 cosine 与 strong cosine gate；`calibration_params` 控制 raw feature 到 typed strength/distance 的单调校准。任何 sampled `θ` 若违反 `min <= max`、阈值区间、quota 上限、protocol version 或 settings lifecycle 约束，必须在 trial preflight 阶段判为 invalid，不得进入 candidate adjacency simulation。

当前落库表示使用 `tpe_graph_operating_point_search_space_v2`。概念上的 `dense_knn`、`reverse_quota`、`bridge_quota` 和 `type_thresholds` 在 `sampled_theta_json` 中展开为上文列出的标量字段，`calibration_params` 保持为闭合嵌套对象。

协议字段是本地 allowlist 的 categorical identity：当前每个 allowlist 都只有一个 active 成员，分别为 `dense_dynamic_knn_bridge_quota_edge_calibration_v2`、`type_local_winsorized_minmax_v1` 和 `edge_distance_log_calibrated_strength_v2`；对应 protocol hash 是不可调的审计伴随字段，不得伪装成数值搜索维度。四个校准参数必须是真实进入 random/TPE 采样、candidate simulation 和 objective observation 的连续维度：

```text
lower_quantile ∈ [0.00, 0.25]
upper_quantile ∈ [0.75, 1.00]
min_span ∈ [0.01, 0.50]
strength_floor ∈ [0.000001, 0.25]
```

搜索空间规范必须完整保存 integer/float/calibration bounds、categorical allowlist、跨字段约束、numeric paths，以及不参与采样但约束 θ 的完整 `immutable_identity`（rank/raw-strength、node-quality、out/in signal、relation-quota protocol/hash 与 quota signal scale），并形成 `tpe_search_space_hash`。`sampled_fields` 与 `audit_companion_fields` 必须分开，固定 hash 不得被列作 sampled dimension。

每个 trial 必须同时保存 `calibration_params_hash` 与 `edge_type_calibration_config_hash`；后者绑定 graph operating point protocol、edge calibration protocol/hash、distance protocol/hash 和完整 calibration params。preflight 必须拒绝缺字段、额外字段、旧字段名、非有限数值、越界、`min > max`、strong threshold 不高于 typed threshold 以及任何 protocol/hash 漂移。

candidate simulation 返回的校准 config identity 必须与 sampled θ 完全一致，否则 trial 失败，不能成为 TPE observation 或 best theta。

TPE 使用已完成 trial 的目标值把参数样本切分为 good set 与 bad set。设 \(y=J(\theta)\)，\(y^\*\) 为前 \(\gamma\) 分位的目标值，目标越大越好：

$$
l(\theta)=p(\theta\mid y\ge y^\*),\quad
g(\theta)=p(\theta\mid y<y^\*)
$$

采样阶段优先选择最大化 \(\frac{l(\theta)}{g(\theta)}\) 的候选点；当 trial 数不足 `tpe_startup_random_trials` 时使用有界随机采样填充初始观测。TPE 只优化 bottom relation graph operating point，不参与在线 query scoring，不替代 staged traversal priority queue，也不改变 node weight 语义。

每个 trial 执行：

```text
sample θ from TPE
-> theta preflight
-> in-memory candidate adjacency simulation
-> bottom graph diagnostics
-> lightweight probe metrics
-> hard gate
-> objective score
-> update TPE observations
```

trial 的输入只能来自当前构图批次已经确定的事实源：

```text
active chunk scope
current chunk embeddings
chunk structure graph
document/language metadata
RQ codebook inputs if already available for this build stage
previous active operating point or versioned default theta
```

trial 不写 `chunk_relation_graph_states`，不写 `chunk_relation_edges`，不写 RQ prefix，不写 mid/coarse，不写 Qdrant，不写 Redis active cache。trial 只能产生内存邻接表和可审计指标；批次失败或取消时不会留下可被 active retrieval 读取的半成品图。

自动 TPE 架构图：

trial 必须形成可审计记录，但记录的是轻量仿真结果，不是持久候选图状态：

```text
tpe_trials:
  trial_id
  knowledge_base_id
  build_batch_id
  chunk_scope_hash
  embedding_model
  embedding_text_version
  sampled_theta_json
  theta_hash
  tpe_search_space_hash
  edge_distance_protocol
  edge_distance_protocol_hash
  edge_type_calibration_protocol
  edge_type_calibration_protocol_hash
  calibration_params_json
  calibration_params_hash
  edge_type_calibration_config_hash
  sampler_state_hash
  probe_set_hash
  candidate_adjacency_hash
  diagnostics_json
  hard_gate_json
  objective_components_json
  objective_score
  status
  failure_code
  started_at
  finished_at
```

`candidate_adjacency_hash` 当前使用 `tpe_candidate_adjacency_theta_typed_gate_language_scope_v3`，由 candidate edge ids、edge type、raw strength、typed gate decision、active language identity scope hash 和 θ hash 计算；该 hash protocol 也必须进入 TPE protocol hash。即使端点与 edge type 恰好不变，Document/active DocumentVersion 的版本化语言 card 变化仍必须改变 trial identity。

trial diagnostics 同时保存完整 language identity scope card。trial 失败、取消或超时必须保留 failure code、blocking reason 和可重试边界；不得静默退回固定参数并标记成功。若所有 trial 均失败，构图批次必须使用上一版 active operating point 或版本化默认 theta，并在 batch diagnostics 中记录 `auto_tpe_status=failed_or_skipped`，不得把失败 trial 写成成功优化。

`auto_tpe_runs` 必须保存本次 `tpe_search_space_hash`；选出 best theta 后还必须把 selected edge distance protocol/hash、selected edge calibration protocol/hash、selected calibration params/hash 与 selected calibration config hash 作为直接审计字段保存，不能只依赖任意 JSON 的自洽 hash。promotion gate 必须比较 run、best completed trial、最终 active relation state 和 state `auto_tpe` diagnostics 的完整 θ、search-space 与 calibration identity；旧记录缺少任一身份字段时只能作为 historical diagnostics，不能作为 active prior 复用。

TPE runtime identity 必须拆成两个不可混用的直接事实。

`auto_tpe_runs.runtime_settings_hash` 绑定 trial 采样、预算、超时、quality proxy 与 hard-gate 等完整 optimizer envelope，并与 best trial 相等；`auto_tpe_runs.selected_graph_runtime_settings_hash` 绑定本次选中 theta 实际写图时的 `runtime_settings_rebuild_slice_v1`，并与最终 active `chunk_relation_graph_states.runtime_settings_hash` 以及 state `auto_tpe.runtime_settings_hash` 相等。

state `auto_tpe.optimizer_runtime_settings_hash` 还必须反向绑定 run 的 optimizer hash。promotion handle、根事务 commit hook 与 crash reconciler 必须按 `tpe_durable_audit_v5` 同时重放这两个身份；任一缺失、串位或在 selection 后被改写都必须 fail closed，不能因为 hot optimizer 参数与 rebuild slice 的合法差异而误判图提交失败，也不能把两者压成同一个 hash。

TPE audit 与 active graph fact transaction 必须分离。生产路径只能使用独立 PostgreSQL 事务提交 run 创建、trial 创建以及每个 trial 边界的终态；graph rebuild 的外层事务 rollback 不得删除这些记录。SQLite 只允许作为测试期显式 non-durable adapter，其他非 PostgreSQL 运行必须 fail closed。run 的 promotion 生命周期固定为：

```text
running
-> selected_pending_graph_commit
-> completed | failed
```

`selected_pending_graph_commit` 只表示已选出 best valid theta，不表示该 theta 已成为事实。

只有最外层 graph transaction commit，且最终 `chunk_relation_graph_state.state=active`，其 knowledge base、`auto_tpe_run_id`、`auto_tpe_best_trial_id`、`graph_operating_point_hash` 与 durable run、completed best trial、objective 和 theta hash 全部一致时，run 才能转为 `completed` 并绑定 relation state；SAVEPOINT release 不得消费 promotion。

外层 rollback、关联不一致、shadow state 或 graph write 后续阶段失败必须将 run 转为 `failed`，保留 failure code、blocking reason 与 retry boundary。只有带已提交 active relation state 关联且再次通过同一完整性门禁的 `completed` run 才可被后续构图复用。

run/trial 的 `running` 和 `selected_pending_graph_commit` 状态必须带有限 lease。每个 running trial 的 lease 必须独立判定，不能由更长的 aggregate run lease 掩盖。best theta 选出后到根 graph transaction 结束之间还必须持有 run owner fence；reconciler 只有同时取得 knowledge-base resource lock 与 run owner fence，才可把 lease 到期的记录视为崩溃遗留。这样即使 relation 已写入但后续 RQ/mid/coarse 长事务尚未提交，也不能因另一个连接看不到未提交 state 而误杀活跃 run。

进程在 audit commit 与 graph commit 任一窗口退出后，数据库连接释放 fence，reconciler 再检查已提交 active relation state：关联完整则补记 `completed`，关联不完整则记 `failed`；无 graph 且 lease 到期时记 process-interrupted failure。reconcile 必须幂等，不能把仍有 owner/resource fence 的活动 run 误判为失败。

自动 TPE 由 graph build worker 在 bottom relation graph 阶段执行。worker 必须在 TPE 开始前和每个 trial 边界刷新 runtime settings version；长 trial 内不得继续读取已经撤销的开关。取消批次时，TPE 必须在当前 trial 边界停止；如果单个 trial 内部执行时间超过 `tpe_trial_timeout_seconds`，该 trial 失败并进入下一 trial 或终止批次。

日志流必须把 TPE 作为构图阶段子事件展示，而不是伪装成文件解析进度：

```text
auto_tpe_started
auto_tpe_trial_started
auto_tpe_trial_completed
auto_tpe_trial_blocked
auto_tpe_best_theta_selected
auto_tpe_skipped
auto_tpe_failed
```

这些事件只描述自动 operating point 选择，不表示 mid/coarse 已完成。前端只在导入页提供自动 TPE 开关、envelope 参数和最近 run 状态；设置页不得提供 TPE 开关、参数入口、独立运行优化器或单独切换图谱参数的按钮。

硬约束使用无向简单图口径。令 \(n=|V_C|\)、\(m=|E_C|\)，同一 chunk pair 的 typed relation 仍只计一个无向 adjacency；当 \(n\le1\) 时归一化密度固定为 0：

$$
d_{norm}
=
\begin{cases}
0,&n\le1\\
\frac{2m}{n(n-1)},&n>1
\end{cases}
\in[0,1]
$$

scope-aware 稀疏边预算固定为 `tpe_scope_sparse_edge_budget_log2_v1`：

$$
B_{sparse}(n)
=
\begin{cases}
0,&n\le1\\
\min\left(\frac{n(n-1)}2,\left\lceil n\max(1,\log_2 n)\right\rceil\right),&n>1
\end{cases}
$$

TPE trial 必须同时通过归一化密度与稀疏边预算；`K_min/K_max`、reverse quota 和 bridge quota 的实际采样上界必须由当前 scope 的稀疏 out-degree allowance 约束，不能仅在 objective 末端把所有候选判死。旧 `|E|/|V|` 只允许以 `mean_edges_per_node` 作为历史/辅助诊断，不得继续绑定 `edge_density` hard gate 或 density penalty。

硬约束：

$$
d_{norm}\le \eta_E,\quad
|E_C|\le B_{sparse}(|V_C|),\quad
isolated\_ratio\le \eta_I,\quad
\frac{degree_{p95}}{\max(degree_{median},1)}\le \eta_H
$$

$$
structure\_recovery\_rate\ge \eta_S,\quad
candidate\_latency_{p95}\le \eta_L
$$

硬约束的阈值来自 构建 Runtime Settings 中的 versioned gate profile。任一 hard gate 失败时，trial 的 `status=blocked`，可记录 objective components 供诊断，但不得成为 best theta。`candidate_latency_p95` 是 candidate adjacency 构造、probe expansion 和指标计算的本地耗时，不包括 LLM latency；如果 embedding model、embedding text version 或 chunk scope 变化，旧 trial 只能作为 historical diagnostics，不能跨 scope 复用。

active 延迟统计协议固定为 `tpe_local_latency_segment_nearest_rank_p95_v1`。每个 trial 必须分别采集：当前 candidate adjacency simulation（含 typed adjacency hash）的完整本地耗时、每个 bounded probe 的 1–2 hop/structure expansion 耗时、各个确定性 metric block 的计算耗时。每相位独立按 nearest-rank `ceil(0.95*n)-1` 计算 p95，最终 hard-gate 值取三相位 p95 的最大值；不得把 trial 从开始到结束的单个 wall-clock 值直接标作 p95。

profile 必须保存 protocol、每相位有界 raw samples、sample count、min/max/mean/p50/p95 和最终 max-phase p95；非有限值、负值、空相位或超过 512 个样本必须 fail closed。trial 总耗时只作为 timeout/diagnostics，不能替代该分布。

软目标函数：

$$
\begin{aligned}
J(\theta)
=&
0.26\cdot evidence\_recall\_proxy
+0.18\cdot structure\_recovery\_rate\\
&+0.16\cdot component\_coverage
+0.12\cdot edge\_precision\_proxy\\
&+0.10\cdot bridge\_opportunity\_coverage
+0.08\cdot path\_diversity\\
&-0.12\cdot hubness\_penalty
-0.10\cdot density\_penalty\\
&-0.06\cdot latency\_penalty
\end{aligned}
$$

目标函数组件定义如下：

```text
evidence_recall_proxy:
  probe chunk 或 expected support chunk 在 candidate adjacency 中可被 1-2 跳触达的比例。
  expected support 可以来自人工 probe、上一版 verified citation spans、
  或结构邻近的 positive support set；不得来自 LLM 无支撑猜测。

structure_recovery_rate:
  candidate adjacency 能恢复 previous/next、same section、same page、
  table/formula/caption/code closure 周边证据的比例。
  这些结构边不进入 active relation graph，只作为恢复能力评估。

component_coverage:
  active chunks、document ids、语言桶和候选 RQ prefix 输入被非孤立覆盖的加权比例。

edge_precision_proxy:
  抽样 candidate relation edges 中 mutual/reverse/strong gate、typed threshold、
  support feature 和结构可回溯性均通过的比例。support feature 必须从冻结卡片
  完整重放 coefficients、semantic、reciprocity、channel rank、两端 intrinsic
  node-quality pair、out/in signal 与 quota；声明的 computed 值不能替代公式重放。

bridge_opportunity_coverage:
  cross-document 与 cross-language candidate 在独立 quota 内获得候选机会的比例。
  它只计量桥接机会，不能通过无约束加边提高该指标。

path_diversity:
  probe expansion 在 document、language、edge type 和 candidate RQ prefix 上的归一化熵。
  它计量已支撑路径的分布多样性，无支撑跳边不计入。

hubness_penalty:
  degree_p95、degree_median、top hub share 与 edge type imbalance 的归一化惩罚。

density_penalty:
  d_norm 超过目标归一化密度区间后的惩罚；输入、阈值和输出都在 [0,1]。

latency_penalty:
  candidate adjacency simulation、probe expansion 与 metric computation 三相位
  nearest-rank p95 的最大值超过预算后的惩罚；预算内（含等于预算）必须为 0。
```

Latency soft penalty 协议固定为 `tpe_latency_budget_excess_ratio_v1`。令 (L) 为三相位 nearest-rank p95 的最大值，(B>0) 为同一个 hard-gate latency budget，则

$$
latency\_penalty=\min\left(1,\frac{\max(0,L-B)}{B}\right)
$$

该 soft card 保存 `candidate_latency_p95_ms`、`budget_ms`、`excess_ms`、分子、分母、protocol 与 probe hash；hard gate 仍严格使用 (L\le B)，不得因 soft normalization 改变。

Active 质量代理总协议固定为 `tpe_expected_support_structure_coverage_diversity_v4`，并由 `auto_tpe_lightweight_graph_operating_point_v8` 绑定。expected support 的人工目标必须先验证引用并解析为 UUID-free `(probe_chunk_business_key, expected_support_chunk_business_key)`，按 canonical pair 去重、排序，再对每个 probe 截断到最多 512 个；无效、自引用或无法解析的地址不占用 512 上限，但必须用有界 sample、完整 count/hash 审计。

previous/next、same section、same page、table/formula/caption/code closure 每个 probe、每类最多保留 512 个 UUID-free business-key 排序目标；历史 verified citation 按 `(created_at desc, id desc)` 最多读取 4096 行。截断数、非法输入数、上限、输入 hash 和 `model_call_count=0` 必须进入 trial audit，不能静默丢弃或调用 LLM 补齐。

probe、expected-support pair、candidate RQ assignment 和 edge precision sample 的审计 hash 必须使用 `chunk_business_key_v1`，不能纳入随机 chunk/document/version UUID。

candidate RQ 只从已冻结向量在内存中训练、编码一次并供该 run 的全部 trial 只读复用；冻结 scope 中每个 active chunk 必须恰有一个非空、有限且维度一致的 vector，missing、extra、重复归一化 key、非法数值、维度漂移或 encode 后 assignment scope 不完整时必须 `enabled=false`，返回空 assignment，并记录有界 business-key sample、各原因 count、scope/input hash 与 `model_call_count=0`。

run diagnostics 必须记录 precompute latency、codebook/membership/input hash 与 `precomputed_once_per_run=true`。真实 TPE run 缺失完整 document/version provenance、出现 business-key collision 或 candidate RQ 输入不可用时必须 `insufficient_evaluation`/fail closed；test-double 的 local audit fallback 不得进入生产 trial。

Edge precision active 协议固定为 `tpe_typed_gate_support_feature_structure_traceability_v2`。每个抽样 edge 必须核对 raw-strength protocol/hash 与闭合字段集合，重算 semantic normalization、reciprocity、逐 channel competition-rank percentile、两端 `chunk_node_quality_intrinsic_v2` card hash/value 的均值、out/in availability-weighted signal card、source/target quota card，并按 active coefficients 重算 raw strength；任一字段缺失、非有限、越界、protocol/hash 不一致或被篡改时，该 edge 不得获得 precision hit。

Hubness soft penalty 协议固定为 `tpe_degree_ratio_top_five_percent_share_edge_type_imbalance_v1`：

```text
ratio_pressure = clip((degree_p95 / max(degree_median, 1)) / hard_hubness_ratio, 0, 1)
top_count = max(1, ceil(0.05 * |V|))
uniform_top_share = top_count / |V|
top_concentration = clip((observed_top_degree_share - uniform_top_share)
                         / max(1 - uniform_top_share, 1e-12), 0, 1)
edge_type_imbalance = 1 - H(edge_type counts) / log(|eligible edge types|)
hubness_penalty = (ratio_pressure + top_concentration + edge_type_imbalance) / 3
```

无边图的 `edge_type_imbalance=0`，由 isolated hard gate 负责阻断。eligible edge-type 集合始终包含 dense semantic；当前 scope 存在多个 document 时加入 cross-document bridge，存在两个以上已知语言桶时加入 cross-language bridge。只有一个 eligible channel 时 imbalance 为 0；出现 ineligible persisted edge type 时仍纳入 entropy buckets 并由 typed gate 另行 fail closed。card 必须保存 degree p95/median、top count/share/uniform share、eligible types、逐 edge-type count/entropy、三项等权重、样本数与 UUID-free distribution hash。

Density soft penalty 协议固定为 `tpe_normalized_undirected_density_soft_ceiling_v2`。输入固定为 `tpe_normalized_undirected_simple_graph_density_v1` 的 `d_norm`，soft target interval 为 `[0, 0.75 * hard_max_edge_density]`；区间内 penalty 为 0，之后按剩余 25% 线性增长，在 hard ceiling 为 1，超过后仍封顶 1 且由 hard gate 阻断。card 必须保存 normalized observed density、unique undirected pair count、maximum pair count、edge count、mean edges per node、scope sparse budget/ratio、target interval、hard ceiling、原始 excess/penalty span 与协议版本。

所有组件必须保存原始分子、分母、采样数量、probe set hash 和计算协议版本。没有足够 probe 时，自动 TPE 必须标记为 `insufficient_evaluation` 并回退到上一版 active/default theta；不得调用 LLM 临时补 probe。

跨语言质量在当前 operating point 中作为 lightweight diagnostics 和 bridge opportunity 组件的一部分，不作为单独 hard gate：

```text
cross_language_edge_count
cross_language_edge_ratio
cross_document_edge_count
cross_document_edge_ratio
prefix_language_entropy
prefix_language_purity
```

TPE 结束后只选择 best valid theta；真正写入 active 图谱发生一次，且只写 bottom relation graph。随后 RQ、mid concepts 和 coarse concepts 基于最终 active bottom relation graph 派生一次。mid/coarse 的 LLM 生成、双语派生、摘要和 projection calibration 不进入 TPE trial，也不参与 TPE objective。

最终 active relation graph 写入必须原子保存：

```text
graph_operating_point_hash
graph_operating_point_json
edge_distance_protocol_hash
edge_type_calibration_protocol_hash
calibration_params_hash
edge_type_calibration_config_hash
runtime_settings_hash
auto_tpe_run_id
auto_tpe_best_trial_id
diagnostics_json
```

如果 active relation graph 写入失败，当前批次失败并保持上一版 active graph state 不变；TPE run 保留为 failed diagnostics。成功写入后必须失效 relation graph、mid/coarse graph、retrieval trace、context package、QA 和 Agent 相关 cache；下游 mid/coarse projection 必须基于最终 active relation graph 重新计算。

### 距离与遍历支撑

目标关系图不输出孤立图分数，而输出可遍历距离边和路径证据。所有进入 active traversal 的边都必须先经过 edge-type normalization。设边 \(e\) 的类型为 \(t\)，原始强度或原始特征摘要为 \(a_e^{(t)}\)，则：

$$
s_e
=
\operatorname{Calib}_t
\left(
a_e^{(t)};
\operatorname{Stats}_t,
\operatorname{Protocol}_t
\right)
\in(0,1]
$$

类型内校准函数必须单调：原始证据越强，\(s_e\) 越大。active chunk relation graph 只允许前文定义的 `type_local_winsorized_minmax_v1`；其他 quantile、z-score sigmoid、isotonic 或 rank-to-strength 实现必须先定义新 protocol、迁移/重建边界和验收，不能在同一 protocol hash 下替换。chunk relation edge 的 active distance：

$$
d_e
=
-\log(\max(\epsilon,s_e))
$$

硬路径阈值使用归一化后的累计 distance，而不是 raw score。不同 edge type 不直接相加 raw score；跨类型路径只累计统一 distance，同时保留 `edge_type`、support ids 与 normalization diagnostics，供版本化 deterministic gray-zone rule 形成 bounded observation 并判断路径动作。gray-zone rule 不调用模型，也不读取 Profile prompt 或 provider response。

候选路径距离：

$$
D(P)
=
\sum_{e\in P}d_e
+\operatorname{Penalty}(P)
$$

路径贡献记录独立支撑与来源谱系，不把贡献计数折算成距离增益。`layered_distance_traversal_v2` 使用累计距离、深度和稳定路径键排序；入口处的 Dense/RQ/BM25 融合分与物理距离分别保存。循环访问不能提高优先级，node weight 不作为最终相关性分数。

底层边写入不接受无 support 的 LLM 推断：

$$
e_{ij}\in E_C
\Rightarrow
\left(
support\_features(e_{ij})\ne\varnothing
\land
protocol(e_{ij})=p_1
\land
d_e<\infty
\right)
$$

所有边必须保存 typed features、raw strength、calibration stats、support ids、distance、source algorithm、protocol version 和 diagnostics。任意特殊文本形态的局部闭包由 \(G_0\) 恢复；底层关系图只判断两个 chunk 是否存在内容语义近邻关系。

**架构影响：**
- 影响对象：RQ membership diagnostics、mid concept packet、coarse concept packet、staged priority-queue traversal、bridge traversal、context package bridge chunks 和 graph diagnostics。
- 影响方式：relation graph 把 base dense、cross-document dense bridge 和 cross-language dense bridge 统一成可遍历距离边；结构信息只在命中后恢复上下文；RQ membership 提供地址和支撑映射；查询时的 RQ 相关性按独立入口协议计算；mid/coarse 节点和边完全根据底层 chunk edges 与 membership 投影。
- 传播字段：`chunk_relation_graph_state_id`、`chunk_relation_edges`、`rq_path`、`rq_prefix_memberships`、`edge_type`、`distance`、`raw_strength`、`features_json`、`normalization_stats_json`、`edge_distance_protocol_hash`、`state_hash`。
- 触发条件：embedding、chunk scope、dynamic KNN operating point、bridge quota protocol、TPE calibrated active parameters、RQ codebook、RQ membership protocol 或 relation protocol 变化时，mid concepts、coarse concepts、retrieval trace 和 cache 需要刷新。
- 验收观察点：relation state ready、edge type 分布、bridge ratio、cross-language edge ratio、cross-document edge ratio、raw strength distribution by edge type、normalized distance distribution、hubness diagnostics、path threshold hit distribution、trace 中 staged frontier expansion steps、循环剪枝和 diagnostics hash。

## RQ 地址、成员与聚类

### 协议概览

RQ membership layer 表示 RQ residual address 与 primary membership protocol。它不是独立 active traversal layer，不承担原文结构职责，不通过社区检测决定底层边。原文层次、坐标、previous/next、表格、公式和图注闭包由 Chunk Structure Graph 负责；底层关系由 Chunk Relation Graph 负责；RQ 只定义 primary 语义地址、边界/低置信诊断、chunk seed prior 和高层节点投影基础。

RQ 层级的工程语义固定为：

```text
RQ L3 prefix -> Mid Concept node
RQ L2 prefix -> Coarse Concept node
RQ L1 prefix -> parent prior, route prior, diagnostics
```

active RQ address depth 固定为 3，不是可调参数。只要存在可用 chunk vector，即使知识库只有 1 或 2 个 chunk，也必须完整构造 L1/L2/L3；任一层都允许 `k=1`，不得再按 chunk 数缩短 address depth。`rq_kmeans_levels` 只能作为 `fixed_protocol=3` 的只读诊断字段暴露，不能进入 Runtime Settings update schema、环境写入 key map、hot reload、candidate settings、cache invalidation 或 runtime version broadcast。

为保证同层 `centroid_near` 的完整精确 pairwise 域静态有界，active `rq_kmeans_max_k` 固定允许区间为 `1..6`，默认 6；因此 L1/L2/L3 的协议上界分别为 6/36/216 个 prefix。配置、Runtime Settings request 与 UI 必须共同拒绝大于 6 的值，builder 还要独立 fail closed，不能依赖 UI 校验。该上限属于 rebuild-required RQ protocol identity；不得在同一 prefix-pair protocol hash 下扩成 64 或无界域。

RQ prefix tree 是硬树：每个 L3 prefix 只有一个 L2 parent，每个 L2 prefix 只有一个 L1 parent；每个 chunk 在每层只持久化一个主 prefix。membership score 是主链选择置信度，不授权创建第二条归属路径；一个 L3 prefix 不会被拆成多个 L2 parent，一个 L2 prefix 不会被拆成多个 L1 parent。

目标架构受 [ContextRAG](https://arxiv.org/abs/2605.19735) 的 extraction-free graph construction 启发：底层拓扑不由 LLM 抽实体和关系，而由可复算 multilingual dense embedding、dynamic KNN、bridge quota 和 typed edge calibration 构建。RQ 提供语义地址、membership、seed prior 和 diagnostics，不创建 active bottom edge。[KG2RAG](https://aclanthology.org/2025.naacl-long.449/) 的 seed expansion / graph organization 思路用于检索阶段：先定位图入口，再沿关系图扩展和组织证据。

### 片段证据图

chunk evidence graph 等同于上一节定义的独立 Chunk Relation Graph：

$$
G_C=(V_C,E_C)
$$

其中：

$$
E_C
=
E_{\mathrm{dense\_base}}
\cup E_{\mathrm{dense\_cross\_doc}}
\cup E_{\mathrm{dense\_cross\_lang}}
$$

结构边不作为 evidence feature。每条 chunk relation edge 先保存原始 evidence feature \(a_e^{(t)}\)，再按 edge type \(t\) 归一化为关系强度 \(s_e\in(0,1]\)：

$$
s_e
=
\operatorname{Calib}_t
\left(
a_e^{(t)};
\operatorname{Stats}_t,
\operatorname{Protocol}_t
\right)
$$

再写入距离 \(d_e\)：

$$
d_e
=
-\log(\max(\epsilon,s_e))
$$

关联越强，\(s_e\) 越大，\(d_e\) 越小。不同 edge type 的 raw feature 不直接比较；只有归一化后的 distance 可进入累计路径距离、green/gray/hard stop 阈值。跨类型导航仍保留 typed edge、support ids、路径证据和 deterministic gray-zone rule decision，不做全局拍脑袋加权。

### RQ 主成员关系

RQ primary membership 是 active 归属协议。可视化或诊断层可以报告完整 softmax 与边界不确定性，但不能把未落库的非主候选作为 mid/coarse 节点事实源，也不能用诊断分组边反向决定底层 chunk edge。

`rq_primary_chain_v1` 对每个 chunk 只持久化 L1/L2/L3 primary chain。任何非主 code、single-deviation leaf 或 ancestor closure 都不得写入 `rq_prefix_memberships`、Query→RQ entry、概念 packet、节点权重或边投影。禁止各层候选笛卡尔积。完整 codebook softmax、候选概率、entropy 与 margin 只保留在 encoding diagnostics 中。概念 eligibility 由 executor 根据 primary membership、support span、底层 edge、chunk scope 与确定性预算单独计算，`model_call_count=0`。

对第 \(l\) 层 codebook，chunk \(c\) 到 code \(k\) 的距离为：

$$
d_{c,l,k}
=
\left\|r_c^{(l-1)}-\mu_{l,k}\right\|_2
$$

soft assignment：

$$
p_{c,l,k}
=
\frac{\exp(-d_{c,l,k}/\tau_l)}
{\sum_h\exp(-d_{c,l,h}/\tau_l)}
$$

residual confidence：

$$
\gamma_c
=
\exp(-\rho_c/\tau_r)
$$

prefix membership：

$$
\mu_{c,p}
=
\gamma_c
\prod_{l\le depth(p)}
p_{c,l,q_p^{(l)}}
$$

Active 协议固定为 `rq_primary_chain_v1`。每层必须先对完整 codebook 计算并审计归一化 softmax，最近 code 构成的主 residual path 在三层始终完整保留；不存在候选截断或概率裁剪设置。三层时每 chunk membership 必须恰为 3 条，全库必须恰为 `3×chunks`，非主 membership 与 Cartesian expansion 数必须为 0。持久化分数继续使用完整 softmax 中 primary code 的概率乘积，不重新归一化，也不设置人工 floor。

`rq_membership_temperature` 对应所有层的 \(\tau_l\)，`rq_residual_tau` 对应 \(\tau_r\)。这两个温度和协议名属于 `rebuild_required` Runtime Settings；只有 candidate 经 shadow rebuild、evaluation 和 promotion 后才能改变 active graph。

RQ 编码按稳定 chunk id 分批，默认批上限为 256；构建诊断必须记录 codebook/protocol/encoding/membership hash、完整 softmax 归一误差、`primary=3×chunks`、`non_primary=0`、逐 chunk membership count hash、observed max/hard max=3、批次数、`cartesian_expansion_used=false`、`renormalized_after_primary_selection=false`、`artificial_membership_floor=false` 和 `model_call_count=0`。

同 codebook、向量、参数和 chunk scope 重建必须得到相同 primary membership hash。

membership role 由 \(\mu_{c,p}\)、rank、entropy、residual norm 和边界距离决定：

```text
primary_member
boundary_member
bridge_member
low_confidence_member
outlier_member
noise_candidate
```

Active role 协议固定为 `rq_membership_role_primary_entropy_boundary_v2`。第 (l) 层归一化 entropy 为 (H_{c,l}=-\sum_k p_{c,l,k}\log p_{c,l,k}/\log |K_l|)（单 codebook 时为 0）；prefix entropy 为截至该深度各层 (H_{c,l}) 的均值。每层同时记录前两名的概率 margin (Delta p_l=p_{(1)}-p_{(2)}) 与距离 margin (Delta d_l=d_{(2)}-d_{(1)})，prefix 的 boundary margin 取路径各层最小值；单 codebook 没有竞争边界，两个 margin 都使用固定非边界值 1。residual outlier threshold 取当前构建 scope 的 residual norm p95。

角色按以下 deterministic precedence 判定，并把所有同时命中的 flags 一并留在 diagnostics：

```text
noise_candidate       membership_score <= 1e-8
outlier_member        residual_norm >= scope_p95 and gamma <= 0.25
bridge_member         chunk has retained cross-document/cross-language bridge support
low_confidence_member gamma <= 0.35 or membership_score <= 0.01
boundary_member       entropy >= 0.65 or probability_margin <= 0.15 or distance_margin <= 0.05
primary_member        persisted primary prefix
```

role protocol 输入、阈值、precedence、matched flags、role hash 与 `model_call_count=0` 必须写入每条 membership diagnostics；relation state 还必须保存 role/entropy/boundary/residual 的全量分布。角色只影响 membership diagnostics、上层 packet 权重、seed prior 与 tie-break，不创建底层关系边，也不参与或覆盖 gray-zone path decision。

低置信 chunk 不被丢弃；它以低 membership、边界角色或 outlier/noise diagnostics 进入 packet 和 trace。Primary membership 权重影响高层投影，但不额外增加图层。

### RQ 前缀诊断

RQ prefix diagnostics 在 active 架构中不是独立导航边。RQ prefix 之间的 parent-child、sibling、centroid-near 和 overlap diagnostics 只服务于地址解释、entry prior、packet diagnostics 和 UI 展示；active mid/coarse edge 仍必须由底层 chunk relation edge support 投影。

RQ prefix adjacency diagnostics schema：

```text
source_rq_prefix_id
target_rq_prefix_id
edge_type
diagnostic_strength
support_membership_mass
support_chunk_ids_sample
source_algorithm
protocol_version
diagnostics_json
```

diagnostic strength：

$$
d^{diag}_{pq}
=
\operatorname{Diag}
\left(
\operatorname{PrefixRelation}(p,q),
\operatorname{CentroidDistance}(p,q),
\operatorname{MembershipOverlap}(p,q),
\operatorname{ProjectedChunkSupport}(p,q)
\right)
$$

diagnostic edge types：

```text
parent_child
sibling
centroid_near
projected_chunk_support
```

这些 diagnostics 不进入 \(D(P)\)，不参与 active graph threshold，不替代 support_chunk_edge_ids。

active prefix-pair 诊断协议固定为 `rq_prefix_pair_diagnostics_v1`，其输入、方向和强度必须可复算：

- `parent_child` 是从 hard parent 到 child 的有向事实，强度为 `child_membership_mass / parent_membership_mass`，support mass 为 child 的真实 primary membership mass；
- `sibling` 仅连接同一 hard parent（L1 使用同一隐式 root）的同层 prefix，强度为 `exp(-centroid_distance / level_tau)`；`level_tau` 是该层全部非零 reconstructed-centroid pair distance 排序后以全浮点精度计算的确定性中位数，偶数样本取中间两项均值，不允许复用带展示舍入的 quantile helper；没有非零距离时固定回退为 `1.0`；
- `centroid_near` 在每层完整、静态有界的 prefix 域上精确计算距离，每个 prefix 保留最近 3 个邻居，取无向并集，强度与 sibling 使用同一距离式；
- `projected_chunk_support` 只由已存在的底层 `ChunkRelationEdge` 投影。每个贡献质量为 `mu_source,p × mu_target,q`，support mass 是贡献质量之和，强度是该质量对底层 calibrated edge strength 的加权均值；底层 edge ids 必须完整保存。

除 `parent_child` 外，端点都按稳定 `rq_prefix_key` 排序。canonical pair hash 绑定端点业务键、层级/path、类型、强度、完整 support chunk 业务键集、底层 edge contribution 事实 hash、source algorithm、protocol hash 与公式输入；chunk 业务键由 document source/checksum/type/title 与 chunk version/index、char/token span、section/page、text hash 构成。

projected contribution 还必须绑定两端 chunk 业务键、对应 prefix 业务键、membership score 及其乘积。canonical hash 不绑定 chunk/prefix/edge 的数据库 UUID、创建时间或查询顺序。相同 graph state 的同事实重试必须复用既有行；事实不同则 fail closed。

build/retry/promotion，以及任何显式执行的 reconcile（若提供），必须复用同一 verifier：从实际持久化 canonical facts 重算逐行与 aggregate hash，并检查端点 graph-state/KB、方向、同层/同 parent 约束和完整 support-id checksum，再保存由 graph state、KB、count、aggregate/protocol hash 组成的 durable integrity proof；没有独立 reconcile active path 时不得把 rebuild 之外的入口写成已实现。

在线 search/QA admission 只做 pair row `COUNT` 与该 proof/state hash 的常数大小核对，不能在每次查询加载全量 pair JSON。底层 edge ids 在表中完整保存；support chunk/edge sample 都先按各自 UUID-free 业务键排序（数据库 id 仅作同业务键 tie-break/reference）再取前 24，concept packet 与 UI/API 同时返回完整 count/hash。

诊断表、packet 与 UI 必须显式标记 `diagnostic_only=true`、`active_relation_edge=false`、`model_call_count=0`；不得把任何 pair 写入 active chunk relation graph，不得影响累计距离 gray-zone 分区或本地裁决。主库验收必须单列在线 admission p95 与 RSS。

### 残差 K-means

目标 RQ-KMeans 将 embedding 递归量化为语义地址。对 chunk embedding \(e_c\)：

$$
r_c^{(0)}=e_c
$$

第 \(l\) 层：

$$
q_c^{(l)}
=
\operatorname*{arg\,min}_{k}
\left\|r_c^{(l-1)}-\mu_{l,k}\right\|_2
$$

$$
r_c^{(l)}
=
r_c^{(l-1)}-\mu_{l,q_c^{(l)}}
$$

RQ path：

$$
path(c)
=
\left(q_c^{(1)},q_c^{(2)},\ldots,q_c^{(L)}\right)
$$

residual norm：

$$
\rho_c
=
\left\|r_c^{(L)}\right\|_2
$$

### 前缀分组

目标上，每个 prefix 是层级地址节点：

$$
prefix_l(c)
=
(q_c^{(1)},\ldots,q_c^{(l)})
$$

prefix membership：

$$
\mu_{c,p}
=
\gamma_c
\prod_{l\le depth(p)}
p_{c,l,q_p^{(l)}}
$$

其中 \(\gamma_c=\exp(-\rho_c/\tau_r)\)。membership 不设置人工下限；低 membership 进入 boundary、outlier 或 noise diagnostics。

### 前缀间关系

RQ cluster graph 不作为 active traversal layer。prefix 关系保存在 address tree 与 diagnostics 中：

$$
E_R^{diag}
=
E_{\mathrm{parent}}
\cup E_{\mathrm{sibling}}
\cup E_{\mathrm{centroid}}
\cup E_{\mathrm{overlap}}
\cup E_{\mathrm{projected\_support}}
$$

diagnostic edge types：

```text
parent_child
sibling
centroid_near
projected_chunk_support
```

其中 `parent_child` 来自 prefix tree，`projected_chunk_support` 来自底层 chunk relation edge support 的投影统计。四类名称与 `rq_prefix_pair_diagnostics_v1` schema/allowlist 完全一致；它们只存入独立诊断表，不得以 `rq_*` 类型写入 active `ChunkRelationEdge`。诊断边不作为 mid/coarse active edge 的存在性条件。

### 片段级诊断

两个 chunk 的最长公共前缀：

$$
LCP(c_i,c_j)
=
\max
\left\{
l:\ prefix_l(c_i)=prefix_l(c_j)
\right\}
$$

RQ diagnostic weight：

$$
w_{ij}^{rq}
=
\frac{LCP(c_i,c_j)}{L}
\cdot
\exp
\left(
-\frac{\|r_i-r_j\|_2}{\tau_r}
\right)
$$

RQ pair diagnostics 可生成诊断类型：

```text
rq_hierarchy_near
rq_prefix_sibling
rq_residual_near
```

并在 diagnostics 中保存 `lcp_depth`、`residual_distance`、`rq_weight`、source/target rq path。RQ pair diagnostics 不写入 active relation graph，不参与 active edge calibration，不作为 bottom edge existence gate。诊断缺口写入 graph diagnostics，不使用 fallback pair 补边。

**架构影响：**
- 影响对象：mid concept aggregation、coarse concept aggregation、staged priority-queue graph traversal、context package bridge restoration、retrieval trace 和前端 RQ 诊断。
- 影响方式：RQ layer 提供 L3/L2/L1 地址、chunk membership、边界/低置信/outlier 诊断和 chunk seed prior；active mid concept 与 RQ L3 prefix packet 对齐，active coarse concept 与 RQ L2 prefix packet 对齐；检索在 selected mid queue 中逐父节点使用 RQ membership 选择 chunk seeds，再进入独立 chunk relation graph 并由结构图恢复上下文。
- 传播字段：`rq_prefixes`、`rq_prefix_memberships`、`rq_path`、`rq_level`、`rq_path_prefix`、`residual_norm`、`membership_score`、`membership_role`、`lcp_depth`、`residual_distance`、`rq_weight`、`support_chunk_edge_ids`。
- 触发条件：relation graph hash、embedding vectors、RQ level/codebook、RQ membership protocol、bridge support 或 residual diagnostics 变化时，mid concept hash、coarse hash、retrieval trace 和 cache 必须刷新。
- 验收观察点：RQ path availability、RQ L3-to-mid projection coverage、RQ L2-to-coarse projection coverage、primary membership 数量、membership role 分布、LCP depth 分布、bridge path coverage、chunk seed quality 和 staged traversal diagnostics。

## 中层概念

### L3 聚合

目标 active mid concept 由通过 deterministic eligibility 的 RQ L3 prefix packet 生成。设 \(\mathcal{P}_3\) 为 active RQ L3 prefixes，中粒度候选集合为：

$$
\mathcal{M}^{cand}
=
\{m_p:\ p\in\mathcal{P}_3,\ \operatorname{mass}(p)>0\}
$$

候选集合不等于 active 节点集合。active `concept_node_eligibility_primary_coverage_v3` 先验证 primary membership、raw span 和 packet business identity，再用稳定 greedy coverage 选择节点：每轮依次最大化尚未覆盖的 primary-support chunk 数、primary membership mass、primary 底层 support edge 数，最后按稳定 `rq_prefix_key` 破平。设 active chunk 数为 \(N_C\)，Mid 节点预算为：

$$
B_M(N_C)=
\begin{cases}
0,&N_C=0\\
1,&N_C=1\\
\min(|\mathcal{M}^{cand}|,\max(1,\min(N_C-1,\lceil\sqrt{N_C}\rceil))),&N_C>1
\end{cases}
$$

只对前 \(B_M\) 个入选 packet 调用定义 provider；LLM 不得改变入选、排序、support 或预算。状态必须保存 candidate/eligible/ineligible counts、coverage、budget、完整 eligibility facts hash、稳定排序 sample 和 `model_call_count=0`。因此 `|V_M|<=|V_C|`，且当 \(N_C>1\) 且存在多个候选时必须形成严格压缩；不能用缺失概念节点删除底层 primary route。

`primary_support_count>0` 是 Mid candidate 的硬资格门。Active 图只持久化唯一主链 leaf/ancestor prefix；任何没有 primary support 的候选必须记录 `no_primary_support`，并禁止创建 Mid、触发概念 provider、提供概念定义核心证据或单独创建高层边。Coarse candidate 必须至少拥有一个通过该门的 Mid 子节点。

### 查询时成员分数边界

同父节点 Frontier 按 `distance_so_far`、深度和稳定路径键排序。RQ 前缀重构相关性可按新检索协议进入根层/下钻入口融合，不能修改物理距离。primary membership、RQ 评分协议与生效权重进入相应检索缓存身份；它们不改变概念 eligibility 或持久主链。
每个候选 \(m_p\) 必须保留：

```text
support_rq_prefix = rq_l3_prefix
parent_rq_l2_prefix
parent_rq_l1_prefix
representative_chunk_ids
support_chunk_ids
core_chunk_ids
boundary_chunk_ids
bridge_chunk_ids
outlier_chunk_ids
structure_paths
membership_mass
membership_entropy
residual_norm_stats
raw_node_weight
node_weight
node_weight_normalization_scope
display_terms_json
summary
internal_state_json
```

中粒度节点权重来自 RQ L3 packet 的证据规模、归属清晰度、内部底层边密度、边界比例和摘要置信度：

$$
w_M^{raw}(m_p)
=
\operatorname{Score}
\left(
\log(1+|S_C(p)|),
\operatorname{mass}(p),
\operatorname{core\_ratio}(p),
\operatorname{density}_{E_C}(p),
1-\operatorname{boundary\_ratio}(p),
1-\operatorname{outlier\_ratio}(p),
\operatorname{summary\_confidence}(p)
\right)
$$

active `mid_node_weight_membership_structure_v1` 将七个输入先压到 \([0,1]\)，再按固定系数求和：`support_log_scale=0.15`、`membership_mass=0.20`、`core_ratio=0.15`、`internal_edge_density=0.15`、`boundary_stability=0.12`、`outlier_stability=0.08`、`summary_confidence=0.15`。

其中 `support_log_scale=min(1,log1p(support_count)/log1p(32))`，`membership_mass=mass/(1+mass)`；core、boundary 与 outlier 比率均按 membership mass 计算，不能用截断后的代表 chunk 数替代。packet 构造阶段只使用中性的 `summary_confidence=0.5` 生成预定义诊断；写入时使用经过 schema 校验的 definition confidence 重算最终 raw weight。除该 confidence 外，membership、结构、边与权重公式都由 executor 本地计算，`model_call_count=0`。

每个 mid state 内做同层归一化：

$$
w_M(m_f)
=
\operatorname{LayerNorm}_M
\left(
w_M^{raw}(m_f);
\{w_M^{raw}(m'):m'\in V_M\}
\right)
\in[0,1]
$$

active `layer_state_max_raw_v2` 使用同一 `mid_concept_state` 内的 `raw/max(raw)`，并保存完整 raw distribution、scope hash、最大 raw 值和 `layer_local_only=true`；空值、负值或非有限值 fail closed。

其中 `node_weight` 只在 mid layer 内可比较，用于预算控制、展示、入口候选辅助和同等路径下的 tie-break，不表示用户问题相关性，不与 coarse 或 chunk 权重跨层比较，不替代路径搜索，也不能形成“大节点优先”的 active retrieval 规则。

### 概念输入包

目标 concept packet：

$$
P_m
=
\left(
p_3,S_c,S_{core},S_{boundary},S_{bridge},S_{outlier},E_C^{in},E_C^{cross},D_R,W_m,X_m
\right)
$$

其中 \(p_3\) 是 RQ L3 prefix，\(S_c\) 是 membership 支撑 chunk 集合，\(S_{core}\)、\(S_{boundary}\)、\(S_{bridge}\)、\(S_{outlier}\) 是按 membership role 切分的支撑集合，\(E_C^{in}\) 是 L3 内部底层边，\(E_C^{cross}\) 是跨 L3 底层边，\(D_R\) 是 RQ residual 与 membership diagnostics，\(W_m\) 是 node weight diagnostics，\(X_m\) 是 chunk excerpts、source spans 与 structure paths。

packet 字段包括 packet id、RQ L3 prefix、candidate labels、display terms、node summary、raw/normalized node weight 与 weight card、representative chunk ids、完整 support/core/boundary/bridge/outlier/low-confidence/noise chunk ids、membership mass、role count/mass distribution、entropy/boundary/residual distribution、完整 internal/cross/support bottom edge ids 与 business-fact hashes、chunk excerpts、source spans、完整 structure mapping identity/coverage/business-fact hash 和 grounding hash。

structure mapping 由流式强 multiset hash 保存完整 UUID-free business facts 的 count/hash，并另存含 mapping/chunk/node 地址的有序 address-stream hash；可读 `structure_paths` 只是固定上限的 deterministic trace sample，必须同时保存 sample count/limit/complete。sample 不得替代完整 identity，空或空白 `mapping_protocol_version` 必须在 hash 与 provider 调用前 fail closed。

本地 (P_m) 是构图与写入的完整确定性 authority；不得为了模型上下文限制删减 support、membership、edge、structure、source-span 或 node-weight 的完整 count/hash。模型只接收版本化投影 (P_f=operatorname{Project}_{provider}(P_m))：投影绑定完整 packet address hash、UUID-free business hash、各 identity card 的完整 count/hash，以及按 declared representative 顺序选择的 raw-span/full-text-hash excerpt。每个 representative 必须恰有一个合法整数 char span，且 span 必须覆盖实际发送的投影文本；缺失、重复或过短一律 fail closed。

`concept_definition_provider_projection_v8` 使用 `concept_provider_ordered_admissible_pack_v2` 对严格 JSON wrapper `{"concept_packets":[...]}` 做 deterministic ordered-admissible packing：本地候选集不变，provider sample 只在 2400 rough-token/28800-byte 单包预算内按固定顺序尝试；Coarse 固定先尝试 child summary、再尝试 representative excerpt，Mid 先尝试 representative excerpt，candidate-label 文本在 evidence sample 之后使用剩余预算加入。

完整 candidate-label count/hash 留在 base identity，label 文本本身不得无条件挤占 evidence sample 容量。单个候选加入后超限时只把该候选记为 omitted 并继续检查后续候选，不得用首个超限候选短路掉后续可容纳样本。

selection audit 对 representative/child 逐类记录 candidate/selected/omitted count 与完整 candidate bindings hash，并按固定 child-then-representative 顺序各保存一个合并后的 selected-evidence bindings hash 与 omitted-evidence bindings hash，避免把每类可从 projection 直接重放的重复 hash 塞回 provider input。

candidate label 只使用 base 的完整 count/hash与 provider-visible selected list 重放：candidate count 固定为 `min(full_count, 6)`，selected count 是列表长度，omitted 是两者差值，不再重复携带三项 count 或 selected/omitted hash。selection hash 绑定上述完整 audit。存在 child/representative evidence 候选且扫描完整 evidence 候选集后仍一个都放不下时才拒绝调用。

代表 excerpt 使用带 raw-span binding 的固定字符上限；child summary/definition 使用 `concept_provider_bounded_text_projection_v2` 的固定字符上限并绑定完整文本 hash与投影文本 hash，但不把 provider 无法解释的 child grounding digest 放进模型可见 child block 或让它影响 ordered packing；完整 grounding/lineage 事实仍留在本地 packet、address `identity_card`、projection audit 与新 graph state hash。

v8 从 provider-visible JSON 移除可由完整 `identity_card`、`business_identity_card` 重放的两个冗余 card hash；持久化 projection audit 必须在本地重算并保留这两个 hash，`projection_hash` 仍绑定两张完整 card。该约束避免地址相关 SHA-256 的分词差异使同等业务证据在 2400-token 硬边界附近随机准入或拒绝，不能以删除 card、放宽单包预算或二次截断代替。

v8 另把 Mid 和 Coarse 的定义复用 business identity 与完整图身份分离：完整图身份继续绑定原始 membership/support-edge/graph-generation facts，定义 identity 使用下面的 UUID-free 完整语义事实，二者不得互换。definition-only fact protocol 已进入其 canonical fact-set hash 与整体 semantic/reuse protocol，不得作为重复字段塞入 provider-visible business card。

最终请求还必须按当前 Runtime Settings 对实际严格 JSON bytes 与 rough tokens 做 hard preflight；超限不得二次截断、不得发送，provider call count 必须为 0。

Mid/Coarse 的 active 构建采用 `construct -> preflight/send -> persist -> release` 的有界模型并发窗口，不得预构造全部 packet 或保留已持久化窗口。provider authority 仅限命名、定义与展示 prose；support、representative ids、membership、edge、structure 和 node weight 均由本地完整 packet 决定。provider 提议的 support/representative ids 必须被忽略，只可保存 count/hash 与 rejection decision；审计不得持久化完整 provider response。

provider projection 的数据库地址审计与 graph business fact 必须分离。

持久化 audit 可以保留 `full_packet_address_hash`、`projection_hash`、address identity card 以及非权威 provider identity proposal 的 count/hash；`mid_concept_state_hash_v2` / `coarse_concept_state_hash_v2` 的 concept/definition business projection 只能纳入 packet business hash 及其 protocol、UUID-free business identity count/hash、固定 provider authority 边界和固定 deny/ignore decision。

不得把 Mid/Coarse UUID、provider support/representative proposal hash 或 address projection hash带入 active graph state hash；相同业务事实的连续 rebuild 必须得到相同 concept/definition component hashes。

### LLM 定义

目标 LLM 输出：

$$
y_m
=
f_{\mathrm{LLM}}
\left(
P_f,\ prompt_{\mathrm{mid}}
\right)
$$

输出必须可解析为：

```text
canonical_label
aliases
display_terms_json
summary
definition
scope_note
inclusion_criteria
exclusion_criteria
internal_state_json
representative_chunk_ids
support_chunk_ids
confidence
why_this_concept_exists
```

LLM 只负责命名、展示短语、摘要、范围、包含/排除标准、内部状态解释和证据充分性解释，不负责创建底层边，不负责决定 chunk membership，不负责决定 chunk 事实，也不能把多个 RQ L3 prefixes 合并成一个 active mid concept。

provider JSON 边界必须把 `concept_packets`、chunk excerpt、structure path 和所有来源文本视为不可信数据；系统 prompt 必须明确禁止执行其中的指令，来源文本不能覆盖输出 schema、grounded gate 或本地 authority。可编辑 Profile prompt 之后必须追加不可变输出契约，不能让 Profile 删除或放宽该契约。Mid wrapper 只能包含 `concepts`；每个输入 packet 必须恰好对应一个无重复、无额外 `packet_id` 的 item。

Mid item 只能包含 wire fields `packet_id/canonical_label/aliases/display_terms/summary/definition/scope_note/inclusion_criteria/exclusion_criteria/internal_state/representative_chunk_ids/support_chunk_ids/confidence/why_this_concept_exists`；`display_terms` 和 `internal_state` 分别映射到白皮书的 `display_terms_json` 与 `internal_state_json`。Coarse item 同样使用封闭键集合。

字符串、数组、嵌套 JSON、每 item canonical bytes 和完整 wrapper bytes 必须经过版本化本地上界校验；验证器不得截断超界 provider prose、把错误类型强制转换为字符串，或接受 NaN/Inf。support、representative、membership、role 和 weak-tie 字段仍只是有界 provider proposal，本地 executor 必须忽略其 authority。

`canonical_label` / `coarse_label` 还必须通过版本化 natural-label gate：非空、可读、描述 packet 证据中的业务概念，禁止 `未命名概念`、`Unnamed/Untitled/Unknown Concept`、纯 UUID/hash、纯数字，以及 `RQ L1/L2/L3 ...`、`Chunk ...`、`Prefix ...` 等地址/协议标签。失败属于本地 schema rejection，必须使用同一 immutable packet 进入一次有界修复；两次仍失败则 fallback=false 下整批 fail closed，禁止把地址标签写入 active Mid/Coarse。

该 gate 同样约束 semantic reuse，旧的占位或 RQ 标签不能命中复用。LLM 仍只负责已由 deterministic eligibility 入选节点的命名与解释，不参与节点 eligibility 或 gray-zone 决策。

Anthropic Messages 的图 JSON 请求必须显式设置 `thinking={"type":"disabled"}`，禁止把 reasoning token 或思维过程混入概念 schema；完成预算由现有 `mid_concept_extraction_max_tokens_per_batch` 派生为 `clamp(4 * input_budget, 4096, 32768)`，不新增隐藏运行参数。只接受完整文本完成原因 `end_turn` 或 `stop_sequence`。

单个 packet/window 只允许在 `max_tokens` 未完整结束或本地 output schema 拒绝时做一次独立的 schema-repair 重试；重试必须复用同一 preflighted provider projection，只能追加不可变本地 repair instruction，最大 attempt count 为 2，并在成功定义 audit 中记录实际 attempt count。

若首轮由本地 schema 拒绝，repair instruction 必须追加由 executor 生成的 content-free rejection card，且只能含固定 failure class、allowlisted `error_code`、allowlisted `field_path`、固定数值约束与“遵守既有不可变字段上限”的命令；不得包含被拒绝字段值、完整/局部 provider response、来源 excerpt、shape hash 或任意 provider 自带文本。

为避免短路 validator 只暴露首个超界字段、修复后又在第二字段失败，第二轮还必须同时施加一套服务器固定且严格低于 validator 最大值的全字段 conservative target：label、term、summary、definition、scope、criterion、explanation、数组 cardinality 与嵌套 JSON 都必须更短，非权威 id/weak-tie proposal 允许为空。这样第二轮可一次收敛多个潜在越界字段，但仍不得截断、强制转换或放宽本地 validator。

最终失败审计必须保留本地 batch/packet 地址、attempt count、首轮 failure class 与最终 content-free schema card；不得保存异常 message 或 provider 内容。refusal、鉴权/权限失败、未知/缺失 stop reason、其他 transport/provider error 不得借该 repair budget 重试。

官方 Anthropic SDK 必须关闭 SDK 自带重试，由 executor 使用独立于 schema-repair 的有界 transport envelope：同一 preflighted request 只对 typed connection/timeout、HTTP 429 和 HTTP 5xx 重试，最多 6 个 transport attempts，采用有界退避；鉴权/权限、refusal、完成原因、JSON 或 schema 错误不得进入该 envelope。

transport retry 日志与最终 failure card 只允许保存 attempt、最大 attempt、异常类型、安全状态码、canonical error code 和 retryable，不得保存异常正文、响应 body/headers、鉴权头或凭据。两次 schema attempts 或 transport envelope 耗尽时都必须保持 fail closed。

`max_tokens`、refusal、未知/缺失完成原因、空文本、非 JSON、非 object、缺少规定 wrapper 或上述 schema 校验失败均必须 fail closed。JSON 解码或 schema 失败只能输出协议版本、字段路径/错误码、文本或 canonical JSON 字节数、SHA-256、括号/code-fence 布尔诊断和安全错误位置，不得返回 `{}` 伪装成 provider object，也不得记录完整 provider response。

### 写入规则

目标 grounded gate：

$$
\operatorname{accept}(m)
=
\mathbf{1}
\left[
S_C(m)\ne \varnothing
\land
\operatorname{RQPrefixLevel}(m)=3
\land
S_C(m)\subseteq support(prefix_3(m))
\land
\forall c\in S_C(m),\ Span(c)\ne\varnothing
\land
\operatorname{SummaryGrounded}(m)
\right]
$$

### 概念边

目标 mid concept edge 由跨 RQ L3 membership 的底层 chunk relation edges 投影而来。若两侧 support chunks 之间存在可审计 \(E_C\) 边，则写入 mid edge：

$$
E_M
=
\left\{
(m_a,m_b):
\exists c_i,c_j,\ (c_i,c_j)\in E_C
\land
\mu_{c_i,m_a}>0
\land
\mu_{c_j,m_b}>0
\right\}
$$

中粒度边距离先从底层 chunk relation edge 的 normalized distance 与 membership support 聚合，得到 raw projected distance：

$$
d_{M}^{raw}(m_a,m_b)
=
\frac{
Q_{0.15}\left(\{d_e:(i,j,e)\in S_{ab}^{C}\}\right)
}{
1+\log(1+n_{ab})
}
$$

其中：

$$
S_{ab}^{C}
=
\{(i,j,e)\in E_C:\mu_{i,m_a}>0,\ \mu_{j,m_b}>0\}
$$

$$
n_{ab}
=
\sum_{(i,j,e)\in S_{ab}^{C}}
\mu_{i,m_a}\mu_{j,m_b}
$$

\(Q_{0.15}\) 是低分位距离，避免被单条最小噪声边完全支配；\(n_{ab}\) 是 membership 加权 support mass，支持越多 raw projected distance 越短。

底层关系边按无向事实处理；对每条边同时计算两个 endpoint orientation 的 membership product，取较大者作为该边唯一贡献并记录 orientation、两端 membership、bottom distance、bottom fact hash。所有正贡献边都进入 support 集，不能取单条最短边或只使用 LLM 返回的 concept support 子集。projected edge type 由各 bottom edge type 的 membership mass 主导类型确定；当前 active bottom allowlist 将 `dense_semantic -> co_occurs_with`，`dense_cross_document_bridge|dense_cross_language_bridge -> bridge_to`，未知 active bottom type 必须 fail closed。

由于该投影聚合会改变距离分布，active mid edge distance 必须再按 `layer=mid` 与 `edge_type` 做投影校准：

$$
s_M^{raw}(m_a,m_b)
=
\exp(-d_M^{raw}(m_a,m_b))
$$

$$
s_M(m_a,m_b)
=
\operatorname{Calib}_{mid,t}
\left(
s_M^{raw}(m_a,m_b);
\operatorname{ProjectionStats}_{mid,t},
\operatorname{ProjectionProtocol}_{mid,t}
\right)
$$

$$
d_M(m_a,m_b)
=
-\log(\max(\epsilon,s_M(m_a,m_b)))
$$

active `layer_edge_type_winsorized_minmax_v1` 在每个 `layer + projected edge_type` 组内以 raw strength 的 Q0.05/Q0.95 做 winsorized min-max，并映射到 `[0.05,1]`；样本少于 2 或 quantile span 小于 `0.05` 时使用显式 identity fallback，必须保存原因，不能跨类型借用统计。校准和 gray predicate rollup 均为本地确定性计算，模型调用数为 0。

active traversal 使用 \(d_M\)，不是 \(d_M^{raw}\)。边必须保存：

```text
support_rq_prefix_ids
support_chunk_edge_ids
support_chunk_ids
distance
projected_distance_raw
projected_strength_raw
raw_strength_summary
projection_normalization_stats_json
edge_projection_protocol_hash
source_algorithm
protocol_version
state_hash
edge_type
diagnostics_json
```

每条边还必须保存逐 bottom edge 的完整 contribution cards、membership support mass、Q0.15、dominant bottom type、contribution facts hash 与 layer/type normalization stats hash。`state_hash` 在 canonical concept state hash 完成后回填为所属最终 state hash；canonical edge facts不包含该自指字段，以避免循环 hash，但包含 projection protocol 与完整 support business facts。

公开 graph overview 可以对 sampled projection edges 和每条边的 contribution cards 做确定性有界投影，但不得把样本冒充完整事实。每条有界边必须同时返回 `support_contributions_complete=false`、完整 contribution count、完整 contribution business-fact hash、projection protocol/hash 与完整分布/rollup；overview 的 sampled edge count 必须与 full edge count 分离。Retrieval Trace、Context Package、admission、freshness、quality gate 和 canonical state hash 始终读取并重放 PostgreSQL 中的完整 contribution facts，不得依赖 overview sample。

边类型由底层主导证据决定：

```text
co_occurs_with
depends_on
contrasts_with
bridge_to
same_method_family
same_evidence_region
```

LLM 可以解释边语义，但不能在没有底层 chunk relation edge evidence 时创建 active mid edge。RQ prefix sibling、centroid-near、membership overlap 和 L1/L2 parent relation 只进入 diagnostics，不进入 edge existence gate。

**架构影响：**
- 影响对象：coarse concept aggregation、coarse concept definition、concept routing、Agent planning、context package coverage、citation grounding 和 answer synthesis。
- 影响方式：mid concepts 与 RQ L3 prefixes 对齐，提供用户可读语义节点、LLM 可读摘要和稳定 chunk seed 集合；mid edges 是底层 chunk relation edges 的 membership 加权投影；support spans 决定概念能否参与检索、回答和引用验证。
- 传播字段：`mid_concept_state_id`、`mid_concepts`、`mid_concept_memberships`、`mid_concept_edges`、`mid_concept_definitions`、`display_terms_json`、`summary`、`internal_state_json`、`support_rq_prefix_ids`、`support_chunk_edge_ids`、`support_chunk_ids`、`representative_chunk_ids`、`node_weight`、`support_spans_json`、`projected_distance_raw`、`projection_normalization_stats_json`、`edge_projection_protocol_hash`、`distance`、`grounding_hash`。
- 触发条件：RQ L3 membership hash、bottom chunk edge distance、LLM prompt protocol、concept packet、support span 或 grounded gate 变化时，coarse graph、graph traversal trace、context package 和 cache 需要刷新。
- 验收观察点：RQ L3-to-mid projection coverage、mid concept grounded rate、node summary grounded rate、support chunk coverage、node weight diagnostics、edge support density、raw projected distance distribution、calibrated mid distance distribution、projection calibration diagnostics、concept path accuracy 和 unsupported concept diagnostics。

## 粗层概念

### L2 分组

目标 coarse graph 由通过 deterministic eligibility 的 RQ L2 prefix packets 生成。辅助分组只作为诊断和可视化参考，不决定 active coarse node。只有已入选 Mid 的 parent L2 prefix 可以成为 Coarse candidate；未入选 L2 仍保留 RQ routing diagnostics。设 \(\mathcal{P}_2\) 为 active RQ L2 prefixes：

$$
\mathcal{K}^{cand}
=
\{k_p:p\in\mathcal{P}_2,\operatorname{mass}(p)>0\}
$$

Coarse 使用与 Mid 同一 eligibility authority 和稳定 coverage tie-break，但覆盖对象优先为已入选 child Mid。设 \(N_M=|V_M|\)，预算为：

$$
B_K(N_M)=
\begin{cases}
0,&N_M=0\\
1,&N_M\le2\\
\min(|\mathcal{K}^{cand}|,\max(1,\min(N_M-1,\lceil\sqrt{N_M}\rceil))),&N_M>2
\end{cases}
$$

只允许为含至少一个 eligible child Mid 的 L2 prefix 构建 Coarse packet；只对前 \(B_K\) 个 packet 调用 provider。状态保存与 Mid 对称的 eligibility audit，`model_call_count=0`，并强制 `|V_K|<=|V_M|<=|V_C|`。多节点层应产生严格压缩；任何违反 cardinality 或空 child support 的构建必须 fail closed。

每个 coarse candidate 聚合其 child L3 mid summaries、chunk membership、底层边投影、边界/桥接/outlier 诊断和结构路径：

```text
support_rq_l2_prefix
parent_rq_l1_prefix
child_rq_l3_prefix_ids
included_mid_concept_ids
boundary_mid_concept_ids
bridge_mid_concept_ids
outlier_mid_concept_ids
support_chunk_edge_ids
cross_prefix_weak_support
support_chunk_ids
membership_mass
membership_entropy
residual_norm_stats
raw_node_weight
node_weight
node_weight_normalization_scope
display_terms_json
summary
internal_state_json
```

粗粒度节点权重来自 RQ L2 packet 的证据覆盖、child L3 质量、内部底层边密度、桥接状态、边界比例和摘要置信度：

$$
w_K^{raw}(k)
=
\operatorname{Score}
\left(
\log(1+|L3_k|),
\log(1+|support(k)|),
\operatorname{density}_{E_C}(k),
\operatorname{child\_quality}(k),
\operatorname{bridge\_state}(k),
1-\operatorname{boundary\_ratio}(k),
1-\operatorname{outlier\_ratio}(k),
\operatorname{summary\_confidence}(k)
\right)
$$

每个 coarse state 内做同层归一化：

$$
w_K(k)
=
\operatorname{LayerNorm}_K
\left(
w_K^{raw}(k);
\{w_K^{raw}(k'):k'\in V_K\}
\right)
\in[0,1]
$$

其中 `node_weight` 只在 coarse layer 内可比较，用于 coarse entry 候选辅助、coarse 层 hard interrupt 上限的局部分配、coarse -> mid 下钻配额、overview/survey 类问题的主题覆盖提示和同等路径下的 tie-break；它不表示查询相关性，不与 mid/chunk 权重跨层比较，也不能替代 query-entry 匹配、累计路径距离或 deterministic gray-zone rule decision。

### 粗层输入包

目标 coarse packet：

$$
P_k
=
\left(
p_2,M_{child},S_c,E_C^{in},E_C^{cross},B_k,O_k,W_k,N_k
\right)
$$

其中 \(p_2\) 是 RQ L2 prefix，\(M_{child}\) 是 child RQ L3 mid summaries，\(S_c\) 是 support chunks，\(E_C^{in}\) 是 L2 内部底层边，\(E_C^{cross}\) 是跨 L2 底层边，\(B_k\) 是 bridge diagnostics，\(O_k\) 是 outlier/noise diagnostics，\(W_k\) 是 cross-prefix weak support，\(N_k\) 是 coarse node weight diagnostics。

packet 包含 RQ L2 prefix、child L3 ids、child mid display terms、child summaries、support chunks、bridge concepts、outlier states、raw node weight、normalized node weight、normalization scope、display terms、summary、internal state 和 grounding hash。与 (P_m) 相同，本地 (P_k) 保存完整 membership/support/edge/structure/node-weight identity 与业务 hash；模型只接收上述版本化 (P_f)。

Coarse packing 固定优先 child-Mid summary/definition，再按 raw-span binding 加 representative excerpt；所有未发送样本仍由完整 candidate bindings hash、count 和本地 packet hash 约束，不获得 membership、role、support、weak-tie 或 weight authority。

### 写入规则

目标 grounded definition：

$$
k
=
f_{\mathrm{LLM}}(P_k,prompt_{\mathrm{coarse}})
$$

并满足：

$$
support(k)
=
\{c:\mu_{c,p_2(k)}>0\}
$$

LLM 输出必须包含 `display_terms_json`、`summary`、`scope_note`、`inclusion_criteria_json`、`exclusion_criteria_json` 和 `internal_state_json`。写入要求 `CoarseConcept`、`CoarseConceptMembership`、`CoarseConceptDefinition`。membership role 至少区分 `included`、`boundary`、`bridge`、`outlier` 和 `low_confidence`。

### 粗层边

目标 coarse edge 由跨 RQ L2 membership 的底层 chunk relation edges 投影而来。若两个 coarse nodes 之间存在底层 support chunk edge，则建立 coarse edge：

$$
E_K
=
\left\{
(k_a,k_b):
\exists c_i,c_j,\ (c_i,c_j)\in E_C
\land
\mu_{c_i,k_a}>0
\land
\mu_{c_j,k_b}>0
\right\}
$$

距离先从 support chunk relation edges 聚合为 raw projected distance：

$$
d_K^{raw}(k_a,k_b)
=
\frac{
Q_{0.15}\left(\{d_e:(i,j,e)\in S_{ab}^{C,K}\}\right)
}{
1+\log(1+n_{ab}^{C})
}
$$

其中：

$$
S_{ab}^{C,K}
=
\{(i,j,e)\in E_C:\mu_{i,k_a}>0,\ \mu_{j,k_b}>0\}
$$

$$
n_{ab}^{C}
=
\sum_{(i,j,e)\in S_{ab}^{C,K}}
\mu_{i,k_a}\mu_{j,k_b}
$$

coarse projection 与 mid 使用同一 `membership_weighted_bottom_support_q15_log_mass_v1`：读取两侧 RQ L2 prefix 的全部正 primary memberships，并扫描所属 relation state 的完整 bottom edge 集；每条无向 bottom edge 只采用最大 endpoint orientation product，保存完整 contribution card、bottom edge id/business fact hash 和两端 membership。coarse membership 不得退化为 included mid ids 或 LLM support 子集。

由于 coarse projection 会改变距离分布，active coarse edge distance 必须按 `layer=coarse` 与 `edge_type` 做投影校准：

$$
s_K^{raw}(k_a,k_b)
=
\exp(-d_K^{raw}(k_a,k_b))
$$

$$
s_K(k_a,k_b)
=
\operatorname{Calib}_{coarse,t}
\left(
s_K^{raw}(k_a,k_b);
\operatorname{ProjectionStats}_{coarse,t},
\operatorname{ProjectionProtocol}_{coarse,t}
\right)
$$

$$
d_K(k_a,k_b)
=
-\log(\max(\epsilon,s_K(k_a,k_b)))
$$

校准同样使用 `layer_edge_type_winsorized_minmax_v1`，但统计域固定为 `layer=coarse + projected edge_type`；Q0.05/Q0.95、`min_span=0.05`、`strength_floor=0.05` 与显式 identity fallback 的语义和 mid 一致，禁止复用 mid 或其他 edge type 的统计。

active traversal 使用 \(d_K\)，不是 \(d_K^{raw}\)。粗粒度边必须保存：

```text
support_mid_concept_ids
support_child_mid_edge_ids
support_rq_prefix_ids
support_chunk_edge_ids
support_chunk_ids
distance
projected_distance_raw
projected_strength_raw
raw_strength_summary
projection_normalization_stats_json
edge_projection_protocol_hash
source_algorithm
protocol_version
state_hash
edge_type
cross_prefix_weak_support
```

coarse edge 的 active `distance`、完整 bottom support、support RQ L2 ids、support mid ids/edges、projection normalization stats 与最终 coarse state hash 必须能一起重放；canonical state hash 避免纳入自指 `edge.state_hash`，但纳入上述 projection protocol 与 support business facts。该投影不得修改累计距离 green/gray/red/hard-stop 协议；`semantic_uncertain` 与 `crossing_rq_boundary` 仍只由既有 bottom support deterministic rollup 产生，LLM/Profile/历史策略均不参与。

粗粒度边可以很弱，但不能丢弃。图导航时弱边会因距离大而排在队列后方；若跨主题候选满足 support gate、累计距离阈值与版本化 deterministic gray-zone rule，仍可被探索。RQ L2 sibling、shared L1 parent、child mid adjacency 和 membership overlap 只进入 diagnostics；没有底层 support chunk edge 时不能创建 active coarse edge。

### 诊断字段

目标 diagnostics：

$$
D_k
=
\left(
Q,\phi,B,stability,singleton\_rate,bridge\_density
\right)
$$

coarse diagnostics 必须保存 RQ L2 coverage、child L3 coverage、membership entropy、residual norm distribution、bridge density、boundary ratio、outlier ratio、cross prefix edge count、internal edge count、raw projected distance distribution 和 projection calibration diagnostics。

**架构影响：**
- 影响对象：coarse entry selection、mid concept drilldown、cross-document synthesis、Agent coarse jump、retrieval cache、graph overview 和质量诊断。
- 影响方式：coarse concepts 决定查询先进入哪些 RQ L2 高层主题区域；coarse edges 是底层 chunk relation edges 的 membership 加权投影，保留跨主题弱边和桥接状态，供优先队列图导航探索。
- 传播字段：`coarse_concept_state_id`、`coarse_concepts`、`coarse_concept_memberships`、`coarse_concept_edges`、`coarse_concept_definitions`、`support_rq_l2_prefix`、`child_rq_l3_prefix_ids`、`bridge_mid_concept_ids`、`support_chunk_edge_ids`、`display_terms_json`、`summary`、`internal_state_json`、`raw_node_weight`、`node_weight`、`node_weight_diagnostics_json`、`projected_distance_raw`、`projection_normalization_stats_json`、`edge_projection_protocol_hash`、`distance`、`freshness_hash`。
- 触发条件：RQ L2 membership state、child L3 summaries、bottom chunk edges、coarse summary protocol、bridge diagnostics 或 traversal edge protocol 改变时，coarse hash、retrieval trace 和 graph payload 需要刷新。
- 验收观察点：RQ L2 coverage、child L3 coverage、bridge density、coarse node summary grounding、coarse node weight diagnostics、coarse entry hit rate、coarse-to-mid drilldown path、cross prefix edge count、raw projected coarse distance distribution、calibrated coarse edge distance distribution、projection calibration diagnostics 和 staged traversal contribution。

[English](./README.en.md) | **中文**

<p align="center">
  <img src="./assets/diagraph-logo.svg" alt="SymboGraph 标志" width="132" height="132">
</p>

<h1 align="center">SymboGraph</h1>

**SymboGraph 是面向严肃企业和学术场景开发的新一代 Neuro-Symbolic Agentic RAG（神经符号代理 RAG） 系统，该项目基于GraphRAG开发，但区别于传统 RAG 的“语义破碎”和笨重 GraphRAG 的“高成本与幻觉叠加”。**

本系统把 PDF、幻灯片、文档、网页、Notebook、图片和 Markdown 解析为可检索文本块、Qdrant 稠密向量、PostgreSQL 稀疏知识图谱和带引用出处的问答结果。无论你的资料是中文还是英文，系统都能统一检索；所有数据留在本地，无需上传第三方。

作为通用 GraphRAG 平台，SymboGraph 的知识库概念并不限于某一类文档——你可以将其用于课程资料、研究文献、技术手册、法律合同或任何需要结构化解构与语义关联的文本集合。

## 快速概览

| 维度       | 实现                                                                                                          |
| ---------- | ------------------------------------------------------------------------------------------------------------- |
| 运行方式   | Docker Compose，全栈容器化                                                                                    |
| 后端       | FastAPI、Pydantic、SQLAlchemy、NetworkX、LangGraph                                                            |
| 前端       | Next.js 16.2.4、React 19、TypeScript、TanStack Query、ECharts                                                 |
| 数据库     | PostgreSQL 16，保存知识库、文件版本、文本块、图谱、问答会话和运行轨迹                                         |
| 向量库     | Qdrant 1.17.1，集合 `knowledge_chunks`                                                                        |
| 缓存与协调 | Redis 7                                                                                                       |
| 模型接口   | OpenAI 兼容 Embedding / Chat API，Embedding 与 Chat 端点独立配置                                              |
| 检索       | Evidence-first 检索：dense + BM25 + rerank 基础召回，证据锚点选择，受控图导航增强，再装配 parent context      |
| 图谱       | LLM 抽候选，chunk 向量建语义图，图算法做构图、消冗、社区、中心性和隐式关系；自适应最佳优先 chunk 选择；TPE 超参自动优化；支持增量更新与全量重建            |
| 质量系统   | 信号-策略-画像-裁判四级质量架构：对 chunk、concept、relation 做自适应分级过滤与路由                           |
| 问答       | Agentic RAG：Perception → Planning → Retrieval → EvidenceEvaluator → Generation，支持跨语言检索与前置证据评估 |

## 技术栈

| 层         | 技术                                                               | 职责                                                            |
| ---------- | ------------------------------------------------------------------ | --------------------------------------------------------------- |
| 前端       | Next.js 16.2.4、React 19、TypeScript、TanStack Query、ECharts      | 知识库管理、上传解析、搜索、问答、图谱浏览、运行时配置          |
| API        | FastAPI、Pydantic、SQLAlchemy、LangGraph                           | REST / SSE 接口、强类型校验、事务编排、导入任务、检索与问答编排 |
| 图算法     | NetworkX、NumPy、SciPy                                             | 稀疏构图、连通分量、Louvain、谱聚类、中心性、Dijkstra 隐式关系  |
| 超参优化   | Optuna TPE（Tree-structured Parzen Estimator）                      | 自动搜索图算法最优阈值与权重组合，持久化最优参数                  |
| 质量系统   | 信号工程、规则策略、领域画像、LLM-as-judge                         | Chunk/Concept/Relation 分级过滤、自适应领域基线、缓存裁判       |
| 数据库     | PostgreSQL 16                                                      | 知识库、文件版本、chunks、图谱、问答会话、运行轨迹和补偿记录    |
| 向量检索   | Qdrant 1.17.1                                                      | parent / child chunk 向量、dense recall、向量健康检查           |
| 词面检索   | PostgreSQL 文本数据、BM25                                          | child chunk 词面召回与混合检索融合                              |
| 缓存与协调 | Redis 7                                                            | 运行态缓存、任务协调、服务依赖                                  |
| 解析       | PyMuPDF、PPTX / DOCX / Markdown / HTML / Notebook 解析器、OCR 路径 | 把异构文档转成结构化章节与文本                                  |
| 模型接口   | OpenAI 兼容 Embedding / Chat API                                   | 向量化、摘要、关键词、实体候选、关系候选、回答生成              |
| 精排       | 轻量精排、可选 Cross-Encoder                                       | 对融合候选按相关性重排                                          |
| 部署       | Docker Compose                                                     | 固定服务边界、依赖版本和本地持久化                              |
| 测试       | pytest、Vitest、Next build、Docker smoke                           | 行为回归、前后端类型契约、无 fallback 质量门禁                  |

## 核心能力

| 能力           | 说明                                                                                |
| -------------- | ----------------------------------------------------------------------------------- |
| 多格式解析     | 支持 PDF、PPT/PPTX、DOCX、Markdown、TXT、Notebook、HTML 和图片资料                  |
| 父子块切分     | parent chunk 保存完整上下文，child chunk 承担精确召回、精排和证据引用               |
| 语义切分       | 对长文本按结构、语义边界、句子边界和长度上限切分，可用 embedding 相似度辅助边界判断 |
| 上下文增强向量 | 向量输入包含文件元数据、章节、父块摘要、相邻子块摘要、关键词、表格和公式标记        |
| 混合检索       | Qdrant child dense recall 与 child BM25 recall 融合，统一进入精排                   |
| 跨语言检索     | LLM 翻译查询扩展至双语子查询，DocumentGrader 用 embedding similarity 跨越语言壁垒   |
| 图谱增强       | 图谱关系必须回到证据 chunk，作为检索扩展信号而不是替代证据                          |
| 图论构图       | 用稀疏图、社区、中心性、Dijkstra 和关系补全降低噪声、保留关键结构                   |
| 自适应质量系统 | 基于领域画像的信号-策略分层过滤，对 chunk、concept、relation 做差异化路由           |
| 可观测问答     | 保存检索审计、模型调用审计、agent trace、引用和失败原因                             |
| 运行时检查     | 暴露健康检查、runtime-check、fallback 状态、Qdrant 状态和模型端点状态               |
| 自动超参优化   | `ENABLE_AUTO_HPO=true` 时，图谱重建前自动运行 TPE 寻优最优阈值和权重组合            |
| 增量图谱更新   | 仅对变更文档关联的图谱局部重算，避免无必要全量重建                                   |

## 系统架构

```mermaid
flowchart TB
    USER["用户浏览器"] --> WEB["Next.js Web<br/>course-kg-web"]
    WEB -->|"HTTP / SSE"| API["FastAPI<br/>course-kg-api"]

    subgraph APP["应用层"]
        API --> INGEST["导入流水线<br/>解析 -> 父子切块 -> 质量路由 -> 知识增强 -> 向量写入"]
        API --> RETRIEVAL["检索流水线<br/>dense/BM25 -> 融合 -> 精排 -> parent context"]
        API --> GRAPH["图谱流水线<br/>LLM候选 -> 质量过滤 -> 向量相似图 -> 稀疏构图 -> 社区/中心性/推断"]
        API --> QA["Agentic 问答<br/>Perception -> Planning -> Retrieval -> EvidenceEvaluator -> Generation"]
        API -.-> QUALITY["质量系统<br/>被各流水线调用的内部库"]
    end

    subgraph STORE["存储与运行态"]
        PG[("PostgreSQL<br/>元数据、文本块、稀疏图、审计记录、质量画像")]
        QD[("Qdrant<br/>chunk 向量与相似召回")]
        RD[("Redis<br/>运行态缓存、embedding / 检索结果缓存、裁判缓存、分布式锁、Celery broker")]
        FS["data/<br/>知识库文件、解析产物、本地持久化"]
    end

    subgraph MODEL["模型接口"]
        EMB["OpenAI 兼容 Embedding API<br/>独立端点配置"]
        CHAT["OpenAI 兼容 Chat API<br/>独立端点配置"]
        MB["Model Bridge<br/>宿主机模型桥接（可选）"]
    end

    API --> PG
    API --> QD
    API --> RD
    API --> FS
    API --> EMB
    API --> CHAT
    API --> MB
```

## 数据流

```mermaid
sequenceDiagram
    participant 用户
    participant Web
    participant API
    participant 文件 as File Storage
    participant DB as PostgreSQL
    participant Vector as Qdrant
    participant Model as Model API

    用户->>Web: 上传或选择知识库文件
    Web->>API: 创建 ingestion batch
    API->>DB: 创建 batch、job、document/version 状态
    API->>文件: 保存原始文件与解析产物
    API->>API: 解析章节、页面、表格、公式和 Notebook 单元
    API->>API: 生成 parent/child chunks（内部含质量信号提取与 discard）
    API->>API: 自适应切片：按 section 动态选择 chunk_size / overlap / strategy
    API->>Model: 生成 parent chunk 摘要和关键词（ChatProvider）
    API->>API: 生成上下文增强文本（contextual_embedding_text）
    API->>Model: 生成所有 chunk 的 embeddings（EmbeddingProvider）
    API->>Vector: 写入 active chunk 向量（含零向量校验与补偿）
    API->>DB: 激活 document version 与 chunks
    API->>Model: 抽取实体/关系候选（自适应最佳优先 chunk 子集）
    API->>DB: 保存 concepts、relations、证据（内部含质量过滤）
    API->>API: 自动超参优化（可选）：TPE 迭代寻优 Dijkstra 阈值、PageRank 权重等
    alt 全量重建
        API->>API: 运行稀疏图构建、社区、中心性和推断
        API->>DB: 保存图算法字段（中心性、社区、排序）
        API->>API: 生成社区摘要
    else 增量更新
        API->>DB: 仅变更文档关联的图谱局部重算
    end
    API-->>Web: SSE 推送日志、进度、retry 和失败状态
```

导入过程使用显式 batch / job 状态和文件级锁。同一知识库同一时间只保留一个非终态导入批次。PostgreSQL 是生命周期事实源；Qdrant 和 Redis 是派生或运行态存储，失败会记录补偿或错误上下文，不做静默降级。

## 导入、切块与向量

### 分层切块

1. 解析器把源文件转成 `ParsedSection`，保留章节、页码、来源类型、表格、公式、Notebook cell 和图片 OCR 元数据。
2. 每个结构段生成 parent chunk，保存完整章节、页面段落或自然语义段。
3. parent chunk 内部继续生成 child chunks，用于精确召回、精排和证据定位。
4. Markdown 和 Notebook 优先按标题与 cell 层级切分；普通长文本按语义边界、句子边界和安全长度切分。
5. 当 `SEMANTIC_CHUNKING_ENABLED=true` 且文本长度达到 `SEMANTIC_CHUNKING_MIN_LENGTH` 时，系统可使用 embedding 相似度辅助切分边界。

> **设计意图（为什么这么做）**：普通的固定长度切块会导致严重的内容割裂（断章取义）。采用父子分层与语义切分，使得模型在检索阶段能利用子块的高精确度，而在生成阶段又能获取父块的完整上下文，彻底解耦了"召回单元"与"生成单元"。

### 自适应切片参数选择

上述分层策略并非使用全局固定的 `chunk_size` 和 `chunk_overlap`。对于每个 `ParsedSection`，系统在切块前动态分析其内容特征，从候选配置空间中选择最优的切块参数组合。该过程通过 SSE 推送 `chunk_adaptive` 事件。

**文档特征向量**：对每个 section 提取 12 维归一化特征：

| 特征维度 | 计算方式 | 说明 |
|---------|---------|------|
| 文本长度 | $\min(1, L_{\text{tokens}} / 1800)$ | 归一化 token 数 |
| 语义单元均长 | $\min(1, \bar{l}_{\text{unit}} / 80)$ | 语义单元平均 token 长度 |
| 长度变异系数 | $\min(1, \sigma_l / \max(\bar{l}_{\text{unit}}, 1))$ | 语义单元长度离散程度 |
| 定义密度 | 来自语义密度信号 | 定义性语句占比 |
| 实体密度 | 来自语义密度信号 | 命名实体占比 |
| 术语密度 | 来自语义密度信号 | 领域术语占比 |
| 唯一 token 比例 | 来自语义密度信号 | 文本去重后的词汇丰富度 |
| 公式信号 | $\min\bigl(1, \frac{12 N_{\text{formula}}}{L_{\text{tokens}}} + 0.35 \cdot \mathbf{1}_{\text{has\_formula}}\bigr)$ | 公式密集程度 |
| 表格信号 | $\mathbf{1}_{\text{has\_table}}$ | 是否包含表格 |
| 代码标记比例 | 代码行 / 总行数 | `def`/`class`/`import` 等标记占比 |
| 符号比例 | $\min(1, \frac{4 N_{\text{symbol}}}{L_{\text{chars}}})$ | 非字母数字符号占比 |
| 结构噪声 | $\max(S_{\text{structural}}, 20 R_{\text{mojibake}})$ | 结构异常与乱码风险 |

**谱形状分析**：将 section 分割成语义单元，每个单元提取特征向量，构建协方差矩阵并计算特征值与特征向量：

$$
\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge \lambda_4,\qquad
\tilde{\lambda}_k = \frac{\lambda_k}{\sum_{j=1}^4 \lambda_j}
$$

谱间隙 $\rho = \tilde{\lambda}_1 - \tilde{\lambda}_2$ 衡量语义一致性；语义曲率 $\kappa$ 衡量相邻语义单元间的特征漂移幅度。

**候选配置空间**（chunk_size, overlap, strategy）：

| chunk_size | overlap | strategy |
|-----------|---------|----------|
| 512 | 64 | `sentence_aware` |
| 640 | 96 | `sentence_aware` |
| 800 | 120 | `semantic_or_sentence` |
| 960 | 144 | `semantic_or_sentence` |
| 700 | 160 | `recursive_structure_preserving` |

**评分机制**：对每组候选参数计算综合得分。

复杂度（内容结构混乱程度）：

$$
\gamma = \min\!\Bigl(1,\; 0.22 \cdot c_v + 0.18 \cdot f + 0.16 \cdot t + 0.18 \cdot r_{\text{code}} + 0.16 \cdot r_{\text{sym}} + 0.10 \cdot \kappa\Bigr)
$$

密度（知识含量）：

$$
\delta = \min\!\Bigl(1,\; 0.35 \cdot d_{\text{term}} + 0.30 \cdot d_{\text{entity}} + 0.20 \cdot d_{\text{def}} + 0.15 \cdot r_{\text{unique}}\Bigr)
$$

目标大小与重叠（由复杂度和密度驱动）：

$$
S^* = 920 - 380\,\gamma + 160\,\delta + 100\,\rho,\qquad
O^* = 80 + 130\,\gamma + 40\,\kappa
$$

拟合度：

$$
\phi_{\text{size}} = 1 - \min\!\Bigl(1, \frac{|S - S^*|}{700}\Bigr),\qquad
\phi_{\text{overlap}} = 1 - \min\!\Bigl(1, \frac{|O - O^*|}{220}\Bigr)
$$

策略奖励：

$$
\beta =
\begin{cases}
0.08 \cdot \max(\kappa, \delta) & \text{if strategy} = \text{`semantic\_or\_sentence'} \\
0.10 \cdot r_{\text{code}} + 0.05 \cdot f & \text{if strategy} = \text{`recursive\_structure\_preserving'} \\
0 & \text{otherwise}
\end{cases}
$$

综合得分：

$$
\mathcal{F} = \max(0, \min(1,\; 0.54\,\phi_{\text{size}} + 0.26\,\phi_{\text{overlap}} + 0.14\,\delta - 0.10\,\nu + \beta))
$$

其中 $\nu$ 为结构噪声。得分最高者被选为当前 section 的切块参数，并持久化到 chunk metadata 中。

### 上下文增强向量

child 向量不只嵌入 child 原文，而是由 `contextual_embedding_text()` 生成上下文增强输入：

```text
文件元数据
章节、页码和来源类型
child chunk 正文
parent summary 或 parent content
相邻 child summaries
关键词
表格、公式和内容类型标记
```

parent chunk 保留原文、摘要和关键词。child chunk 继承 parent 语义摘要并补充邻近上下文，减少小块检索的断章取义。当前向量文本版本是 `contextual_enriched_v3`。

> **设计意图（为什么这么做）**：子块由于文本较短，在独立转换为向量时极易产生语义歧义（例如单独的"该方法"、"这一步"）。在 Embedding 前强制注入父块摘要和相邻上下文，相当于在入库前对文本做了"环境灌注"（Contextual Retrieval），极大提升了 Dense Recall 阶段的召回准确率。

### 去重与幂等

导入按知识库、规范化标题和 checksum 判断重复文件。未变化文件会以 `unchanged_checksum` 跳过；同标题同 checksum 的重复副本会以 `duplicate_document` 跳过，避免重复写入 chunks 和向量。强制重新解析会重新生成 document version、chunks、Qdrant 向量和图谱候选。

## 质量系统

SymboGraph 内置四级质量架构——信号（Signals）、策略（Policies）、画像（Profiles）、裁判（Judge）——对 chunk、concept 和 relation 做差异化分级过滤与自适应路由。质量系统不是简单的"合格/不合格"二元判断，而是为每个对象计算多维信号、输出结构化决策，并允许下游流水线根据决策采取不同行动。

```mermaid
flowchart LR
    RAW["原始对象<br/>chunk / concept / relation"] --> SIG["信号层<br/>TextQuality · StructuralRole · SemanticDensity · DomainSpecificity · EvidenceGrounding"]
    SIG --> POL["策略层<br/>ChunkPolicy · ConceptPolicy · RelationPolicy"]
    POL --> ROUTE["路由决策<br/>discard · summary_only · evidence_only · retrieval_candidate · graph_candidate · accept · candidate_only"]
    PROF["画像层<br/>分层采样 · 正负样本 · 领域术语基线"] --> POL
    JUD["裁判层<br/>LLM-as-judge · Redis缓存 · defer兜底"] --> POL
```

### 1. 信号层（Quality Signals）

信号层从原始文本和元数据中提取可量化的质量指标：

- **TextQuality**：长度、规范化长度、乱码比率（mojibake）、控制字符数、重复行比率、目录相似度
- **StructuralRole**：结构标签（章节/页码/文件名）、容器暗示、目录页、Notebook output
- **SemanticDensity**：唯一词比率、定义得分、实体密度、术语密度、公式/表格标记
- **DomainSpecificity**：通用性得分（genericity）、特异性得分（specificity）、局部 IDF
- **EvidenceGrounding**：文本片段、chunk 锚定、文档锚定、端点匹配、支持计数

对于 concept，信号层还包含 **ModelJudgment**（LLM 裁判的 verdict、score、reasons）。

### 2. 策略层（Quality Policies）

策略层将信号映射为离散的路由决策：

**ChunkQualityPolicy** 决策空间：

| 动作                  | 含义                         | 下游影响                                 |
| --------------------- | ---------------------------- | ---------------------------------------- |
| `discard`             | 机械噪声，直接丢弃           | 不嵌入、不检索、不参与图谱               |
| `summary_only`        | 目录页或结构标签             | 仅用于摘要，不参与检索和图谱             |
| `evidence_only`       | 太短或 Notebook output       | 可嵌入、可检索，但不生成摘要、不参与图谱 |
| `retrieval_candidate` | 普通内容块                   | 嵌入、检索、生成摘要，不参与图谱         |
| `graph_candidate`     | 高语义密度（定义/实体/术语） | 嵌入、检索、摘要、参与图谱抽取           |
| `embed_only`          | 无领域上下文的代码块         | 仅嵌入，不参与检索和图谱                 |

Chunk 质量得分公式：

$$
S_{\text{chunk}} = 0.30 \cdot \min\Bigl(1, \frac{L_{\text{norm}}}{600}\Bigr) + 0.25 \cdot D_{\text{term}} + 0.20 \cdot R_{\text{unique}} + 0.15 \cdot D_{\text{def}} + 0.05 \cdot \mathbf{1}_{\text{formula}} + 0.05 \cdot \mathbf{1}_{\text{table}} - 0.35 \cdot \mathbf{1}_{\text{toc}} - 0.40 \cdot \min\Bigl(1, 20 \cdot R_{\text{mojibake}}\Bigr)
$$

其中 *L*`<sub>`norm`</sub>` 为规范化长度，*D*`<sub>`term`</sub>` 为术语密度，*R*`<sub>`unique`</sub>` 为唯一词比率，*D*`<sub>`def`</sub>` 为定义得分，*R*`<sub>`mojibake`</sub>` 为乱码比率。

**ConceptQualityPolicy** 决策空间为 `accept` / `reject`：

$$
S_{\text{concept}} = \max\Bigl(S_{\text{specificity}},\; 0.35 D_{\text{def}} + 0.25 D_{\text{term}} + 0.20 D_{\text{entity}}\Bigr) - 0.35 S_{\text{structural}} - 0.25 G_{\text{genericity}}
$$

准入条件：无硬拒绝理由（过短、乱码、路径名、结构容器、低特异性、证据不足），且得分 *S*`<sub>`concept`</sub>` ≥ 0.45。

**RelationQualityPolicy** 决策空间为 `accept` / `candidate_only`：

$$
S_{\text{relation}} = 0.40 \cdot c + 0.25 \cdot \mathbf{1}_{\text{src}} + 0.25 \cdot \mathbf{1}_{\text{tgt}} + 0.10 \cdot \min\Bigl(1, \frac{n_{\text{support}}}{3}\Bigr)
$$

其中 *c* 为 LLM 置信度，**1**`<sub>`src`</sub>` / **1**`<sub>`tgt`</sub>` 为证据文本中是否出现源/目标概念。`inferred` 或 `related_to` 类型关系强制降级为 `candidate_only`。

> **设计意图（为什么这么做）**：传统的 RAG/GraphRAG 系统往往在建图前只做粗粒度过滤，导致大量目录页、乱码、重复提取噪声进入向量库和知识图谱。SymboGraph 的分级质量路由让不同类型的内容去它该去的地方——噪声被丢弃、结构标签只做摘要、高语义密度块参与图谱、普通块负责检索——从数据源头上保证下游质量。

### 3. 画像层（Domain Quality Profile）

画像层为每个知识库构建自适应的质量基线：

1. **分层采样**：按 `(content_kind, chapter)` 做分层，从每层抽取短/中/长样本，确保覆盖度
2. **正样本**：高定义得分或高术语密度的 chunk，作为领域"好内容"示例
3. **负样本**：目录页、乱码页、高结构分的 chunk，作为领域"噪声"示例
4. **领域术语**：采样中出现频率最高的 40 个长词，用于后续概念特异性计算
5. **关系 Schema 提示**：预定义 13 种允许的关系类型（`is_a`、`part_of`、`prerequisite_of`、`used_for`、`causes`、`derives_from`、`compares_with`、`example_of`、`defined_by`、`formula_of`、`solves`、`implemented_by`、`related_to`）

画像数据保存在 `quality_profiles` 表，版本化并通过 SHA256 哈希校验完整性。画像在图谱构建和 LLM 裁判时被引用，使质量判断具备领域上下文。

### 4. 裁判层（Quality Judge）

裁判层是可选的 LLM-as-judge 增强：

- 接收策略层的候选对象 + 领域画像，让 LLM 输出 `accept` / `reject` / `candidate_only` / `defer`
- 缓存 key 绑定 `(course_id, profile_version, target_type, model, candidate_hash)`，命中 Redis 时直接返回缓存结果
- LLM 不可用时降级为 `defer`，完全回退到规则策略层，保证系统可用性

> **设计意图（为什么这么做）**：规则策略层快且稳定，但面对复杂领域边界案例时不够灵活。LLM 裁判作为"慢思考"补充，只在规则层无法决断时介入；同时通过 Redis 缓存避免重复调用。这种"规则为主、LLM 为辅、缓存兜底"的三层架构兼顾了延迟、成本和准确性。

## 图谱构建

SymboGraph 的图谱生成不是单一的 KNN 或 LLM 抽取流程，而是一条 evidence-first 的多阶段构图流水线。LLM 负责发现候选实体和显式语义关系，质量门禁负责事实约束，向量相似图负责提供可控的语义候选，传统图算法负责结构分析，LLM 证据补全和 Dijkstra 推断只在受控边界内修补局部结构。PostgreSQL 是概念、关系和生命周期状态的事实源；Qdrant 只提供 chunk 向量与相似度信号，不能单独决定事实关系。

本次更新引入了**自适应最佳优先 chunk 选择**、**TPE 自动超参优化**和**增量图谱更新**三大能力：
- **自适应最佳优先**：根据文档覆盖、章节覆盖、段落覆盖、内容类型覆盖、embedding 簇代理覆盖和低频术语覆盖六维信号，动态选择最具信息增益的 chunks 进行 LLM 抽取，避免对所有 chunks 做无差别调用。
- **TPE 超参优化**：重建前可选运行 Auto HPO，用 Optuna 的 `TPESampler` 在 30 轮 trial 中自动搜索最优的 Dijkstra 语义阈值、关系置信度阈值和图算法权重组合。
- **增量更新**：仅对变更文档关联的图谱局部重算，保留未变更文档的概念和关系，大幅降低更新成本。

```mermaid
flowchart TB
    CHUNK["Active child chunks<br/>通过 ChunkQualityPolicy"] --> VEC["Qdrant chunk vectors<br/>派生向量信号"]
    CHUNK --> LLM["LLM 抽取<br/>候选实体 + 显式关系"]

    LLM --> MERGE["Concept merge<br/>规范名、别名、去重、章节引用"]
    MERGE --> CONCEPT_GATE["ConceptQualityPolicy<br/>实体质量过滤"]
    CONCEPT_GATE --> UPSERT["Upsert graph candidates<br/>concepts、relations、evidence（含 hard gate）"]

    VEC --> CENTROID["Concept vector centroid<br/>证据 chunk 向量质心 + L2 归一化"]
    UPSERT --> CENTROID
    CENTROID --> SPARSE["Semantic sparse candidates<br/>动态 KNN + 语义阈值 + 互近邻/反向入边配额"]
    SPARSE --> CANDIDATE["GraphRelationCandidate<br/>candidate_only，不直接当事实边"]

    UPSERT --> EXPLICIT["Verified explicit relations<br/>有证据支撑的事实边"]
    CANDIDATE --> GRAPH["Verified graph + candidates<br/>中心性/社区分析输入"]
    EXPLICIT --> GRAPH

    UPSERT --> HPO["Auto HPO（可选）<br/>基于 probe chunks 的 TPE 优化"]
    GRAPH --> ENRICH["enrich_course_graph<br/>build_sparse_edges → analyze → complete → dijkstra → final analyze"]
    HPO --> ENRICH
    ENRICH --> COMMUNITY["Community summaries<br/>rebuild_graph_community_summaries"]
    ENRICH --> PG["PostgreSQL graph tables<br/>概念、验证关系、候选关系、指标"]

    PG --> RETRIEVAL["Evidence-first retrieval<br/>只沿可审计证据扩展"]
    PG --> UI["Community-aware graph UI"]
```

### 0. 自适应最佳优先 Chunk 选择（Adaptive Best-First）

在 LLM 抽取之前，系统不会盲目对所有 active chunks 调用模型，而是构建一个**自适应抽取计划**，按信息增益优先选择 chunks：

$$
\Delta_{\text{cov}}(c) = \sum_{d \in \text{new}(c)} w_d + \sum_{h \in \text{new}(c)} w_h + \sum_{s \in \text{new}(c)} w_s + \sum_{k \in \text{new}(c)} w_k + \sum_{e \in \text{new}(c)} w_e + \sum_{t \in \text{new}(c)} w_t
$$

其中 $\text{new}(c)$ 表示加入 chunk $c$ 后新覆盖的文档、章节、段落、内容类型、embedding 簇代理和低频术语。各维度权重为：

| 维度 | 权重 | 说明 |
|------|------|------|
| 文档覆盖 | $0.18$ | 新文档首次被覆盖 |
| 章节覆盖 | $0.16$ | 新章节首次被覆盖 |
| 段落覆盖 | $0.10$ | 新段落首次被覆盖 |
| 内容类型覆盖 | $0.08$ | 新内容类型首次被覆盖 |
| Embedding 簇代理 | $0.12$ | 新簇首次被覆盖 |
| 低频术语覆盖 | $\min(0.18, 0.015 \cdot n_{\text{rare}})$ | 新稀有术语数量加权 |
| 图谱缺口填补 | $0.08$ | 当前图谱缺失该 chunk 关联概念时额外奖励 |

选择采用**贪心最佳优先**策略，每次从未选 chunks 中挑选 $\Delta_{\text{cov}}$ 最大者，直到满足以下任一停止条件：
- 累积覆盖率超过阈值（默认 $0.85$）
- 边际增益 $\Delta_{\text{cov}} < 0.03$
- 达到软启动预算（`GRAPH_EXTRACTION_SOFT_START_BUDGET`，默认 120）
- 所有 chunks 已选

被选中的 chunks 进入 LLM 图谱抽取，其余 chunks 仍保留在向量库和检索系统中，但不消耗模型调用预算。

### 1. 实体与证据

每个概念保留规范名、别名、章节引用、重要度、证据 chunk 数和质量审计信息。概念向量不是由概念名直接生成，而是由支撑该概念的证据 chunk 向量求质心，再做 L2 归一化：

$$
\bar{\mathbf{x}}_i = \frac{1}{|C_i|}\sum_{c \in C_i}\mathbf{z}_c,\qquad
\mathbf{x}_i = \frac{\bar{\mathbf{x}}_i}{\lVert \bar{\mathbf{x}}_i\rVert_2}
$$

其中 *C*`<sub>`i`</sub>` 是支撑概念 *i* 的 active child chunks，$\mathbf{z}_c$ 是 chunk embedding。这样做的目标是让概念向量忠实于本地课程材料，而不是偏向 LLM 对概念名的通用预训练语义。

### 2. LLM 显式关系与质量门禁

LLM 抽取出的关系不是直接入图。系统先检查关系类型 allowlist、端点是否存在、是否自环、置信度、support count、evidence chunk、端点名称是否能在证据文本中匹配，以及候选关系是否由允许的来源产生。只有通过质量门禁的关系才进入验证图，成为后续中心性、社区发现和检索路径规划的事实边。

质量门禁的职责是约束事实性：LLM 负责“提出可能关系”，门禁负责“确认这条关系是否能被课程资料支持”。这也是图谱避免静默污染的关键边界。

### 3. 动态 KNN + 语义阈值 + 互近邻/反向入边配额稀疏构图

这里不是标准 Radius-NN（按固定半径或距离阈值直接入选）的构图。当前实现先按动态 KNN 和语义相似度阈值生成候选，再保留互为近邻或通过反向入边配额的候选。该层的定位是“语义候选边筛选和规模控制”，不是最终事实关系判定。

每个概念按证据量动态决定发出候选边数量：

$$
K_i = \mathrm{clamp}\bigl(4 + \lfloor \log_2(1 + m_i) \rfloor,\, 4,\, 12\bigr)
$$

每个概念按章节覆盖动态限制可接受的反向入边配额：

$$
B_i = \mathrm{clamp}\bigl(2 + \lfloor \log_2(1 + r_i) \rfloor,\, 2,\, 8\bigr)
$$

*m*`<sub>`i`</sub>` 是证据 chunk 数，*r*`<sub>`i`</sub>` 是章节引用数。系统保留互为近邻、通过反向入边配额 *B*`<sub>`i`</sub>` 的近邻，以及高置信 LLM 显式关系，从而让边数随节点数近线性增长。

纯语义候选边会标记为 `relation_source="semantic_sparse"` 和 `candidate_only=true`。它们可以作为补边线索、审计对象或后续验证材料，但默认不会进入中心性和社区计算图，避免相似度噪声放大成事实结构。

### 4. 边权、结构分析与剪枝

边权由证据支持、关系置信度、语义相似度、共现强度和章节结构一致性组合：

$$
w_{ij}=
0.30\,s_{ij}^{\mathrm{evidence}}
+0.25\,c_{ij}^{\mathrm{relation}}
+0.20\,s_{ij}^{\mathrm{sem}}
+0.15\,s_{ij}^{\mathrm{cooccur}}
+0.10\,s_{ij}^{\mathrm{structure}}
$$

最终 *w*`<sub>`ij`</sub>` 裁剪到 [0,1]。验证图上执行：

- 连通分量分析：识别孤立结构、噪声节点和主要知识簇。
- Louvain 社区发现：用于主社区标记和前端颜色分组。
- 谱聚类：用于大连通分量和大社区的二级结构划分。
- 中心性：计算 degree、weighted degree、PageRank、betweenness、closeness，并合成 `centrality_score`。
- 图谱剪枝：优先保留高中心性节点、社区代表节点、跨社区桥接结构和高证据节点。

中心性不是在纯语义候选图上计算，而是在经过证据门禁的验证图上计算；`graph_rank_score` 还会混合概念重要度和证据量，避免单一图拓扑支配排序。

### 5. LLM 关系补全与 Dijkstra 隐式关系

结构分析完成后，系统会挑选高 `graph_rank_score` 节点的局部邻域，收集相关 evidence snippets，让 LLM 只基于这些证据做关系补全。补全结果仍要回到同一套 `RelationQualityPolicy + hard gate`，不能绕过质量门禁直接入图。

Dijkstra 在非负代价图上发现 2-3 跳隐式结构线索：

$$
\mathrm{cost}_{ij}=\frac{1}{0.05+w_{ij}}
$$

当端点语义相似度足够高且路径代价足够低时，系统可写入 `relation_source="dijkstra_inferred"` 的推断边。推断边用于导航、审计和候选补全，不应被视为无条件事实；回答仍必须回到原文 chunk 证据。

### 6. TPE 自动超参优化（Auto HPO）

当 `ENABLE_AUTO_HPO=true` 时，系统在**全量重建**前自动运行超参寻优。HPO 不直接在全量图谱上 trial（成本过高），而是基于少量 **probe chunks**（默认 5 个）的预抽取结果构建代理评估：

**Phase 1 — 候选生成与特征提取**：
从多组种子超参（保守型、激进型、均衡型及随机插值）中生成候选参数空间，对每组参数用预抽取的 probe payloads 构建 mock 图谱，提取以下特征：
- 图规模特征：节点数、边数、连通分量数、平均度
- 结构质量特征：社区模块化度、聚类系数、中心性分布熵
- 语义特征：Dijkstra 推断边占比、LLM 显式关系占比、语义稀疏边占比
- 硬约束检查：孤立节点过多、边数爆炸、社区退化等

**Phase 2 — 成对裁判（Pairwise Judge）**：
让 LLM 作为裁判对候选参数做成对比较（A vs B），输出 `winner`、`confidence` 和 `reasons`。系统收集至少 `HPO_JUDGE_MIN_LABELS`（默认 6）个有效标签后，训练一个**裁判学习代理目标函数**（judge-learned surrogate objective）：

$$
\mathcal{L}(\mathbf{f}, \mathbf{w}, b) = \sum_{(i,j) \in \mathcal{P}} \mathbb{1}[\hat{y}_{ij} = y_{ij}] \cdot \max(0, \; |s_i - s_j| - \epsilon)
$$

其中 $\mathcal{P}$ 是裁判标记的候选对集合，$\mathbf{f}$ 是特征向量，$\mathbf{w}$ 是学习权重，$s_i = \mathbf{w}^\top \mathbf{f}_i + b$ 是候选得分。代理目标函数把高维特征映射为标量质量分，使得后续 TPE 只需优化一个标量目标。

**Phase 3 — TPE 迭代优化**：
使用 Optuna 的 `TPESampler`（Tree-structured Parzen Estimator）在 30 轮 trial 中最大化代理目标函数：

$$
\theta^* = \arg\max_{\theta \in \Theta} \; g_{\text{surrogate}}(\theta; \mathcal{D}_{\text{probe}})
$$

其中 $\theta$ 是 11 维超参向量：

| 超参 | 搜索范围 | 默认值 | 说明 |
|------|----------|--------|------|
| `min_relation_confidence` | $[0.50, 0.85]$ | $0.62$ | 关系最低置信度 |
| `min_accepted_relation_weight` | $[0.45, 0.78]$ | $0.56$ | 关系最小接受权重 |
| `dijkstra_semantic_threshold` | $[0.65, 0.88]$ | $0.74$ | Dijkstra 推断语义阈值 |
| `w_pagerank` | $[0.05, 0.50]$ | $0.20$ | PageRank 权重 |
| `w_betweenness` | $[0.05, 0.50]$ | $0.20$ | Betweenness 权重 |
| `w_degree` | $[0.05, 0.50]$ | $0.25$ | Degree 权重 |
| `w_weighted_degree` | $[0.05, 0.50]$ | $0.25$ | Weighted degree 权重 |
| `w_closeness` | $[0.05, 0.30]$ | $0.10$ | Closeness 权重 |
| `w_centrality` | $[0.10, 0.80]$ | $0.50$ | 综合中心性权重 |
| `w_llm_importance` | $[0.05, 0.60]$ | $0.25$ | LLM 重要度权重 |
| `w_evidence` | $[0.05, 0.60]$ | $0.25$ | 证据量权重 |

最优参数持久化到 `course_model_hyperparameters` 表，并在全量重建时注入 `GraphHyperparameters`。如果 HPO 失败，系统回退到上次成功参数或默认值。

### 7. 增量图谱更新

对于知识库的局部变更（如修改或删除少量文档），系统支持**增量图谱更新**，避免无必要全量重建：

1. **变更检测**：对比变更文档的 chunks 与当前图谱证据，识别受影响的 concepts 和 relations。
2. **局部清理**：调用 `delete_document_graph_incremental` 删除仅由变更文档支撑的概念、别名和关系；保留其他文档支撑的概念和关系不受影响。
3. **局部重抽**：对变更文档的 active chunks 运行自适应最佳优先选择和 LLM 抽取。
4. **局部重算**：运行图算法（社区、中心性）时仅重算受影响的子图，未受影响的节点和边保持原有指标。
5. **事务一致性**：所有清理和重抽操作在显式事务中完成，失败时回滚，保证图谱不会处于半删除状态。

增量更新的时间复杂度与变更文档数量成正比，而非与全库 chunks 数成正比。当变更文档过多或首次构建时，系统会自动回退到全量重建。

### 8. 协作边界与注意事项

这套架构的核心不是让某个算法单独决定图谱，而是让多种信号互相制衡：

- LLM 负责发现候选语义，但必须接受 evidence-first 约束。
- 质量门禁负责阻止幻觉、弱证据关系和不可信关系类型进入验证图。
- 动态 KNN 稀疏构图负责控制语义候选规模，降低 hubness 和毛线球风险。
- 图算法负责社区、中心性、桥接结构和剪枝，但不能替代证据。
- LLM 补全和 Dijkstra 推断只做局部修补和导航线索，不能绕过验证流程。

局限和课程资料约束也必须明确：

- 图谱只表达本地课程资料中可被 evidence chunk 支撑的知识，不保证覆盖通用百科知识。
- 课程资料如果章节混乱、OCR 错误、chunk 过短、重复文件过多或概念抽取质量差，图谱质量会随之下降。
- 动态 KNN 仍偏向局部语义密集区域，跨章节桥接概念可能被漏召回，需要依赖 LLM 显式关系、证据补全和检索路径校正弥补。
- 中心性会放大已进入验证图的错误边，因此关系门禁、候选边隔离和回归测试必须保持严格。
- `semantic_sparse` 和 `dijkstra_inferred` 边不能直接作为答案依据；用户可见答案必须引用原文 chunk。
- 换 embedding 模型、chunk 策略或关系抽取 prompt 后，需要重新校准相似度阈值、质量门禁和图谱质量指标。

前端图展示按 Louvain 社区着色，节点大小反映中心性和图谱排序，虚线边表示 inferred 关系。用户可以筛选社区并快速打开关键实体详情，但图谱视图应被理解为“课程证据导航图”，不是不带证据的权威本体库。

## 检索与问答

SymboGraph 的问答链路采用 **Perception → Retrieval Planning → Base Retrieval → Evidence Navigation → EvidenceEvaluator → Generation** 的 evidence-first Agent 架构，以 LangGraph 编排。所有节点写入 `agent_trace_events`，前端通过 SSE 实时展示运行轨迹。

```mermaid
flowchart LR
    Q["问题"] --> PER["Perception<br/>意图识别 · 实体抽取 · 图概念匹配"]
    PER -->|"greeting / clarify"| AG["AnswerGenerator"]
    PER -->|"需要检索"| PLAN["RetrievalPlanner<br/>evidence-first 参数 · 跨语言翻译 · 查询扩展"]
    PLAN --> BASE["BaseRetrieval<br/>dense + BM25 + fusion + rerank"]
    BASE --> ANCHOR["EvidenceAnchorSelector<br/>可靠 chunk / concept 锚点"]
    ANCHOR --> PATH["EvidenceChainPlanner<br/>verified edges / community routing"]
    PATH --> GRAPH["ControlledGraphEnhancer<br/>仅沿规划路径收集证据"]
    GRAPH --> ASSEMBLE["EvidenceAssembler<br/>base + anchor + graph evidence"]
    ASSEMBLE --> GRADE["DocumentGrader<br/>0.4·overlap + 0.6·embedding_sim"]
    GRADE --> EVAL["EvidenceEvaluator<br/>前置证据充分性评估"]
    EVAL -->|"不充分 + retry<2"| PLAN
    EVAL -->|"充分 / 不充分+retry≥2"| CS["ContextSynthesizer"]
    CS --> AG
    AG --> CC["CitationChecker"]
    CC --> CV["CitationVerifier"]
    CV --> REFL["Reflection<br/>后生成反思（默认关闭）"]
    REFL --> AC["AnswerCorrector"]
    AC --> CS
```

### Perception（感知层）

Perception 节点负责理解用户意图、抽取实体并在知识库图谱中匹配相关概念：

1. **Fast-path**：问候语直接路由到 `direct_answer`；空查询或指代消解路由到 `clarify`。
2. **LLM 感知**：调用 ChatProvider 分析意图（`definition` / `comparison` / `analysis` / `application` / `procedure` 等），抽取实体列表和子查询。
3. **图概念匹配**：将抽取的实体与 `concepts` 表和 `concept_aliases` 表做规范化匹配，获取匹配概念的社区标签和一阶邻居关系。

Perception 输出包含：

- `intent`：问题类型
- `entities` / `matched_concepts`：提取实体与图谱匹配结果
- `perceived_communities`：相关社区 ID 集合
- `suggested_strategy`：建议 evidence-first 路由（`base_retrieval`、`evidence_chain`、`community`）
- `needs_graph`：是否需要图谱增强

### RetrievalPlanner（规划层）

规划层根据 Perception 输出配置 evidence-first 检索参数，并执行跨语言查询翻译：

**策略选择规则：**

| 意图                               | 条件                   | evidence-first 参数                             |
| ---------------------------------- | ---------------------- | ----------------------------------------------- |
| `definition` / `formula`           | `needs_graph=false`    | 只走基础召回和证据评估                          |
| `comparison` 或 `needs_graph=true` | —                      | 基础召回后启用 verified edge 路径规划           |
| `application` / `procedure`        | 有匹配概念             | 允许最多 3 hop 的受控证据链规划                 |
| `analysis`                         | 匹配社区或 broad query | 社区摘要只做 routing hint，答案仍引用原文 chunk |

**跨语言翻译扩展**：

系统检测查询语言（中/英），通过 LLM 翻译为对侧语言：

$$
Q_{\mathrm{bilingual}} = \{q_{\mathrm{original}},\; q_{\mathrm{translated}}\} \cup Q_{\mathrm{sub}}
$$

子查询去重后全部进入 BaseRetrieval。这让中文查询（如"最大流"）也能通过英文翻译（"max flow"）命中英文知识库材料中的 BM25 词面匹配。

> **设计意图（为什么这么做）**：多语言 Embedding 模型在跨语言对齐上常常表现不佳。通过显式的查询翻译并入双语子查询，使得后续检索可以通过多种语言形态并行探测文档库，这是比单纯依赖 Embedding 模型底层对齐更鲁棒的工程解法。

### Evidence-first 检索执行层

执行层固定先召回文本证据，再用图谱做导航：

| 阶段     | 后端/节点                                                  | 说明                                                     |
| -------- | ---------------------------------------------------------- | -------------------------------------------------------- |
| 基础召回 | `hybrid_search_chunks` / `hybrid_search_chunks_with_audit` | dense + BM25 混合召回、融合、精排                        |
| 锚点选择 | `select_evidence_anchors`                                  | 从基础召回结果中选择可靠 anchor chunks / anchor concepts |
| 路径规划 | `plan_evidence_chains`                                     | 只使用 verified graph edges；社区摘要仅用于 routing hint |
| 受控增强 | `controlled_graph_enhancement`                             | 只沿规划路径收集 evidence chunks，不做全邻居扩展         |
| 证据装配 | `assemble_evidence_documents`                              | 合并基础证据、锚点证据和图路径证据                       |

所有策略都遵循 **Small-to-Big** 原则：召回最小粒度单元（child chunks，或没有 child 的 parent chunks 自身），避免 parent/child 在同一候选池竞争，最终通过 `parent_chunk_id` 装配 parent context。

### DocumentGrader（文档评分层）

对召回文档做准入评分，融合词面重叠与向量语义相似度：

$$
\mathrm{grade\_score} = 0.40 \cdot r_{\mathrm{overlap}} + 0.60 \cdot s_{\mathrm{embedding}}
$$

其中：

- *r*`<sub>`overlap`</sub>` = |*T*`<sub>`q`</sub>` ∩ *T*`<sub>`d`</sub>`| / |*T*`<sub>`q`</sub>`|，*T*`<sub>`q`</sub>` 为查询词集合，*T*`<sub>`d`</sub>` 为文档标题+摘要+正文词集合
- *s*`<sub>`embedding`</sub>` 为查询向量与文档向量的余弦相似度；当文档未携带原始向量时，回退到检索阶段记录的 dense score

准入规则（满足任一即可通过）：

$$
\begin{cases}
\mathrm{grade\_score} \ge 0.35 & \text{（主通道）} \\
s_{\mathrm{embedding}} \ge 0.45 & \text{（跨语言桥接通道）} \\
r_{\mathrm{overlap}} \ge 0.25 \;\land\; \mathrm{original\_score} \ge 0.3 & \text{（辅助通道）}
\end{cases}
$$

跨语言桥接通道解决了一个关键问题：中文查询"最大流"与英文材料"max flow"在 `text-embedding-v4` 的向量空间中重叠较弱，但 LLM 翻译后的子查询通过 dense recall 能召回相关 chunk。此时 *r*`<sub>`overlap`</sub>` 可能接近 0，但 *s*`<sub>`embedding`</sub>` 仍然较高，桥接通道防止这类有效跨语言结果被词面匹配误杀。

> **设计意图（为什么这么做）**：这是一个专为"跨语言墙"设计的破局漏斗。中文查询和英文材料往往字面毫无交集（overlap=0），但高维语义极高。通过设立基于 *s*`<sub>`embedding`</sub>` ≥ 0.45 的跨语言桥接豁免通道，巧妙修补了纯词面匹配（BM25）在面对翻译差异时的严重误杀问题。

### EvidenceEvaluator（前置证据评估层）

**在生成答案之前**评估已检索证据是否充分：

对每份 graded document 提取 `grade_score`，计算：

$$
\bar{g} = \frac{1}{n}\sum_{i=1}^{n} g_i,\qquad g_{\max} = \max_i g_i
$$

按意图设定最低证据阈值：

$$
\begin{cases}
(n_{\min}, \bar{g}_{\min}) = (1,\, 0.25) & \text{if intent} \in \{\text{definition},\, \text{procedure}\} \\
(n_{\min}, \bar{g}_{\min}) = (2,\, 0.20) & \text{if intent} \in \{\text{comparison},\, \text{analysis}\} \\
(n_{\min}, \bar{g}_{\min}) = (1,\, 0.20) & \text{otherwise}
\end{cases}
$$

充分性判定：

$$
\mathrm{sufficient} \;\Leftrightarrow\; g_{\max} \ge 0.35 \;\land\; n \ge n_{\min} \;\land\; \bar{g} \ge \bar{g}_{\min}
$$

若仅有锚点但数量/分数边缘，则标记为 `marginal` 仍允许生成。若证据不充分且 `retry_count < 2`，路由回 `RetrievalPlanner`，`top_k` 翻倍后重新检索；若 `retry_count ≥ 2`，标记 `low_evidence=true` 并进入生成，生成器会在提示中注入免责声明，且不强制引用可能不相关的 chunk。

> **设计意图（为什么这么做）**：这是打破传统 RAG"无论搜出什么垃圾都强行生成"的关键。作为防御性评估层，它赋予了系统"知道自己不知道"的能力。与其放任 LLM 胡编乱造，不如主动拦截低质检索并触发降级免责，这对于专业知识问答的可靠性至关重要。

### 后生成闭环（默认关闭）

`ENABLE_POST_GENERATION_REFLECTION=false`（默认）。后生成节点在启用时执行：

- **CitationVerifier**：对答案中的高重要性陈述抽样做 NLI 验证。
- **Reflection**：LLM 评估答案是否存在幻觉、覆盖不足或矛盾，返回 `has_issue` / `issue_type` / `suggestion`。
- **AnswerCorrector**：根据反思结果调整策略（扩大 top_k、改写 query 或保留高置信文档重新生成）。

这些节点在 trace 中可观测，但默认不参与主链路，避免增加延迟和模型调用成本。前置 `EvidenceEvaluator` 已覆盖大多数证据不足场景。

### Evidence-first 图导航

所有问题都先经过基础召回；只有比较、推导、过程或 broad analysis 类问题才在锚点证据之后启用图导航。`semantic_sparse`、`dijkstra_inferred`、`candidate_only` 和缺少 evidence chunk 的关系不会参与默认路径规划。

可缓存的检索结果写入 Redis 时，key 必须绑定 course、query、filters、model、embedding text version 和相关配置；缓存命中仍携带审计信息。

### Small-to-Big 检索

主检索路径只让最小粒度单元进入召回和精排，最终再装配 parent context：

```text
最小粒度单元 dense recall + 最小粒度单元 BM25 recall
-> weighted fusion
-> rerank
-> load parent_chunk_id（如有）
-> 最小粒度证据 + parent context（如有）+ citations
```

检索结果携带 `retrieval_granularity=child_with_parent_context`、dense score、BM25 score、融合分数、精排分数、graph boost 和模型审计字段。

## 技术优势

| 优势             | 体现                                                                                               |
| ---------------- | -------------------------------------------------------------------------------------------------- |
| 证据优先         | 答案、关系和图谱增强都回到真实文本块与 parent context                                              |
| 上下文与精度兼顾 | child chunk 精确召回，parent chunk 提供完整解释上下文                                              |
| 图谱结构可控     | 动态 KNN、语义阈值和反向入边配额限制边数，连通分量和社区算法降低噪声                               |
| 自适应质量系统   | 信号-策略-画像-裁判四级架构，对 chunk/concept/relation 做差异化分级与路由                          |
| 领域质量画像     | 自动构建知识库特异性基线，让质量判断自适应于不同领域                                               |
| 文档资料友好     | 保留章节、页码、公式、表格、Notebook cell 和来源类型                                               |
| 可审计           | 保存 batch/job/log、模型调用、检索分数、fallback 状态和引用                                        |
| 可恢复           | PostgreSQL 保存生命周期状态，Qdrant / Redis 可从持久记录修复                                       |
| 无静默降级       | 模型、数据库、Qdrant 不可用时快速失败并给出错误上下文                                              |
| 可扩展           | 精排、语义切分、图谱增强、超参优化和模型端点通过配置与服务层隔离                                   |
| Agent 架构清晰   | Perception-Planning-Retrieval-EvidenceEvaluator-Generation 五阶段分离，每层可独立观测与调优        |
| 跨语言鲁棒       | LLM 翻译查询扩展 + embedding similarity 桥接 + 跨语言准入通道，缓解单语言 embedding 模型的对齐缺陷 |

## 数据模型

```mermaid
erDiagram
    Course ||--o{ Document : has
    Document ||--o{ DocumentVersion : versions
    DocumentVersion ||--o{ Chunk : chunks
    Chunk --o{ Chunk : children
    Course ||--o{ Concept : has
    Concept ||--o{ ConceptAlias : aliases
    Concept --o{ ConceptRelation : source
    Concept --o{ ConceptRelation : target
    Course ||--o{ IngestionBatch : batches
    IngestionBatch ||--o{ IngestionJob : jobs
    Course --o{ QualityProfile : profiles
    Course ||--o{ QASession : sessions
    QASession --o{ AgentRun : runs
    AgentRun --o{ AgentTraceEvent : traces
    Course --o{ CourseModelHyperparameter : hyperparameters
    Course --o{ GraphExtractionRun : extraction_runs
    GraphExtractionRun ||--o{ GraphExtractionChunkTask : tasks
    Course --o{ GraphHpoJudgeSample : hpo_judge_samples
    Course --o{ GraphHpoObjectiveModel : hpo_objective_models
    Concept --o{ EntityMention : mentions
    Concept --o{ EntityMergeCandidate : merge_candidates
    Concept --o{ GraphRelationCandidate : relation_candidates
    Concept --o{ GraphCommunitySummary : community_summaries
```

| 表                                                  | 作用                                                             |
| --------------------------------------------------- | ---------------------------------------------------------------- |
| `courses`                                           | 知识库工作区                                                     |
| `documents` / `document_versions`                   | 文件元数据、版本和解析产物路径                                   |
| `chunks`                                            | 父子文本块、摘要、关键词、embedding text version 和证据文本      |
| `concepts`                                          | 概念、章节引用、证据数、社区、中心性和图谱排序                   |
| `concept_aliases`                                   | 概念别名和规范化别名                                             |
| `concept_relations`                                 | 稀疏边、关系类型、证据 chunk、权重、语义相似度、支持数和推断来源 |
| `quality_profiles`                                  | 领域质量画像（版本化、分层采样、正负样本、术语基线）             |
| `ingestion_batches` / `ingestion_jobs`              | 批量导入与单文件任务                                             |
| `ingestion_logs` / `ingestion_compensation_logs`    | 事件日志与跨存储补偿记录                                         |
| `qa_sessions` / `agent_runs` / `agent_trace_events` | 问答会话、智能体运行和可观测轨迹                                 |
| `course_model_hyperparameters`                      | 课程级 HPO 超参记录（版本化、可审计、可回退）                    |
| `graph_extraction_runs` / `graph_extraction_chunk_tasks` | 自适应图谱抽取运行状态与 chunk 级任务跟踪                    |
| `graph_hpo_judge_samples` / `graph_hpo_objective_models` | HPO 裁判样本与代理目标函数模型持久化                            |
| `entity_mentions` / `entity_merge_candidates`         | 实体提及记录与 LLM 验证的合并候选                                 |
| `graph_relation_candidates`                           | 语义稀疏图产生的候选关系（非事实边）                              |
| `graph_community_summaries`                           | 社区摘要文本与社区级聚合指标                                      |

## 配置

复制配置模板：

```powershell
Copy-Item .env.example .env
```

常用配置：

| 变量                                                                                                         | 说明                                                                        |
| ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| `API_HOST_PORT` / `WEB_HOST_PORT`                                                                            | 宿主机访问端口                                                              |
| `DATABASE_URL`                                                                                               | PostgreSQL 连接地址                                                         |
| `ENABLE_DATABASE_FALLBACK`                                                                                   | 数据库降级开关，默认 `false`                                                |
| `QDRANT_URL` / `QDRANT_COLLECTION`                                                                           | Qdrant 地址和集合名                                                         |
| `REDIS_URL`                                                                                                  | Redis 地址                                                                  |
| `COURSE_NAME`                                                                                                | 默认知识库名称                                                              |
| `DATA_ROOT`                                                                                                  | 本地数据根目录                                                              |
| `OPENAI_API_KEY` / `CHAT_BASE_URL`                                                                           | OpenAI 兼容 chat / 图谱抽取模型接口                                         |
| `CHAT_RESOLVE_IP`                                                                                            | 需要固定解析 chat 模型域名时使用的目标 IP                                   |
| `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL`                                                                   | OpenAI 兼容 embedding 模型接口，独立于 chat endpoint                        |
| `EMBEDDING_RESOLVE_IP`                                                                                       | 需要固定解析 embedding 模型域名时使用的目标 IP                              |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` / `EMBEDDING_BATCH_SIZE`                                          | 向量模型、维度和批大小                                                      |
| `CHAT_MODEL`                                                                                                 | 对话与图谱抽取模型                                                          |
| `GRAPH_EXTRACTION_SOFT_START_BUDGET` / `GRAPH_EXTRACTION_CONCURRENCY` / `GRAPH_EXTRACTION_RESUME_BATCH_SIZE` | 自适应图谱抽取初始预算、并发模型调用数和每批模型抽取 chunk 数               |
| `ENABLE_MODEL_FALLBACK`                                                                                      | 模型降级开关，默认 `false`                                                  |
| `RERANKER_ENABLED` / `RERANKER_MODEL` / `RERANKER_MAX_LENGTH`                                                | Cross-Encoder 精排配置                                                      |
| `SEMANTIC_CHUNKING_ENABLED` / `SEMANTIC_CHUNKING_MIN_LENGTH`                                                 | 语义切分开关和最小文本长度                                                  |
| `RETRIEVAL_LAYER_ENABLED`                                                                                    | 检索分层开关，默认 `true`                                                   |
| `RETRIEVAL_CACHE_TTL_SECONDS`                                                                                | Redis 检索缓存 TTL，默认 `300`                                              |
| `ENABLE_AGENTIC_REFLECTION`                                                                                  | Agentic 反思与修正总开关，默认 `true`                                       |
| `ENABLE_POST_GENERATION_REFLECTION`                                                                          | 后生成反思开关（CitationVerifier/Reflection/AnswerCorrector），默认 `false` |
| `CITATION_VERIFICATION_SAMPLE_MAX`                                                                           | 每答案引用验证抽样数，默认 `3`                                              |
| `REFLECTION_MAX_RETRIES`                                                                                     | 反思触发修正的最大重试次数，默认 `2`                                        |
| `MODEL_BRIDGE_ENABLED` / `MODEL_BRIDGE_PORT`                                                                 | 宿主机模型桥接开关和端口                                                    |
| `ENABLE_AUTO_HPO`                                                                                            | 图谱重建前自动运行 TPE 超参优化，默认 `false`                               |
| `HPO_JUDGE_MAX_CANDIDATES` / `HPO_JUDGE_MAX_PAIRS` / `HPO_JUDGE_MIN_LABELS`                                  | HPO 裁判候选数、成对比较数、最小有效标签数                                   |
| `HPO_JUDGE_MAX_TOKENS_PER_PAIR` / `HPO_JUDGE_CONCURRENCY`                                                    | 每对裁判最大 token 数和并发数                                               |
| `GRAPH_EXTRACTION_SOFT_START_BUDGET` / `GRAPH_EXTRACTION_RESUME_BATCH_SIZE`                                  | 自适应图谱抽取初始预算和每批模型调用 chunk 数                               |

Docker Compose 会在 API 容器内使用服务名覆盖基础设施地址：

```text
DATABASE_URL=postgresql+psycopg://postgres:postgres@postgres:5432/course_kg
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379/0
```

Embedding 模型与 Chat 模型支持独立端点配置：

```text
CHAT_BASE_URL=https://api.openai.com/v1
EMBEDDING_BASE_URL=https://api.openai.com/v1
```

如果你的 Embedding 提供商与 Chat 提供商不同（例如 embedding 用本地服务、chat 用云端 API），只需分别填写两个端点即可。系统不会将 Embedding 请求 fallback 到 Chat 端点。

如果宿主机可以访问模型供应商，但容器内到模型端点的网络不稳定，可以启用模型桥接。模型桥接只转发真实 OpenAI 兼容接口，不生成假响应，也不是 fallback。

## 运行

1. 配置 `.env`，至少提供真实模型接口：

```env
OPENAI_API_KEY=...
CHAT_BASE_URL=https://api.openai.com/v1
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-v4
CHAT_MODEL=qwen-plus
ENABLE_MODEL_FALLBACK=false
ENABLE_DATABASE_FALLBACK=false
```

2. 启动 Docker 栈：

```powershell
docker compose -f infra/docker-compose.yml up -d api web postgres redis qdrant
```

Windows 用户也可以直接双击 `start-app.bat` 一键启动后端、前端和基础设施容器；该脚本**不会**强制重建镜像，适合日常快速启动。

如果应用代码或依赖发生变更，需要重新构建本地镜像，可运行：

```powershell
docker compose -f infra/docker-compose.yml build api web
```

或在 Windows 下直接运行 `rebuild-images.bat`；如需强制无缓存重建，可执行 `rebuild-images.bat -NoCache`。

3. 打开 Web：

```text
http://127.0.0.1:3000
```

## 验证

后端单元测试：

```powershell
docker exec course-kg-api python -m pytest tests
```

前端检查：

```powershell
npm run typecheck --workspace web
npm run lint --workspace web
npm run test --workspace web
```

Docker smoke：

```powershell
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api
```

知识库质量门禁：

```powershell
docker exec course-kg-api python /app/scripts/quality_gate.py --course-name "知识库名称"
```

Evidence-first 检索对比评估：

```powershell
docker exec course-kg-api python /app/scripts/evaluate_evidence_first_retrieval.py --course-name "知识库名称"
```

质量决策评估：

```powershell
docker exec course-kg-api python /app/scripts/evaluate_quality_decisions.py --course-name "知识库名称"
```

重新解析单个知识库并清理过时派生数据：

```powershell
docker exec course-kg-api python /app/scripts/reingest_all_courses.py --course-name "知识库名称" --cleanup-stale
```

验收重点：

| 检查项       | 期望                                                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| 健康状态     | `/api/health` 返回可用服务状态                                                                                                       |
| 运行时配置   | `/api/settings/runtime-check` 没有阻断项                                                                                             |
| 模型降级     | `ENABLE_MODEL_FALLBACK=false`，模型不可用时快速失败                                                                                  |
| 数据库降级   | `ENABLE_DATABASE_FALLBACK=false`，数据库不可用时快速失败                                                                             |
| 向量健康     | Qdrant 向量数量与 active chunks 对齐，没有零向量                                                                                     |
| 检索质量     | child recall、parent context、rerank 和 citations 字段完整                                                                           |
| 图谱质量     | 节点数达到保留下限，边数近线性增长，社区字段、中心性和权重非空；增量更新后图谱结构稳定                                               |
| 质量系统     | quality_profile 已生成，chunk/concept/relation 的策略决策记录可观测，无大量 discard 误杀                                             |
| 检索分层     | 不同 query type 命中对应 layer，Redis 缓存命中/失效行为正确                                                                          |
| Agentic 闭环 | Perception、RetrievalPlanner、EvidenceEvaluator 节点在 trace 中可观测；后生成 Reflection 默认关闭；无 fallback 时 LLM 错误不静默吞掉 |
| 跨语言检索   | 中英混合查询能命中对侧语言材料，DocumentGrader 桥接通道生效                                                                          |
| 日志可观测   | ingestion logs 暴露进度、retry、失败原因和 terminal event                                                                            |

## 核心创新点

SymboGraph 在通用 GraphRAG 方向上的核心创新可概括为以下七点：

**1. 四级自适应质量架构**
区别于传统系统的单一阈值过滤，SymboGraph 建立了信号-策略-画像-裁判四级质量体系。Chunk 不再只有"保留/丢弃"两种命运，而是被路由到 `discard`、`summary_only`、`evidence_only`、`retrieval_candidate`、`graph_candidate`、`embed_only` 六种下游路径；Concept 和 Relation 同样经过差异化策略过滤。领域质量画像让每个知识库拥有自适应的质量基线，而非依赖全局固定阈值。

**2. 概念向量质心化与动态稀疏构图**
概念向量由其证据 chunk 向量求质心生成，而非由 LLM 提取的概念名直接嵌入，从根本上消除了概念漂移。动态 KNN + 语义阈值 + 互近邻/反向入边配额稀疏构图算法基于证据量 *m*`<sub>`i`</sub>` 和章节覆盖 *r*`<sub>`i`</sub>` 做候选边收发限制，保证边数随节点数近线性增长，天然抑制 Hubness Problem。

**3. Evidence-first Agentic RAG**
问答链路不是简单的"检索→生成"，而是 Perception → RetrievalPlanner → BaseRetrieval → EvidenceAnchorSelector → EvidenceChainPlanner → ControlledGraphEnhancer → EvidenceAssembler → DocumentGrader → EvidenceEvaluator → Generation 的完整 Agent 工作流。前置 `EvidenceEvaluator` 赋予系统"知道自己不知道"的能力，在生成前拦截低质检索。

**4. 跨语言鲁棒检索的三重机制**
LLM 显式翻译扩展生成双语子查询、Embedding Similarity 桥接跨越语言壁垒、DocumentGrader 的 *s*`<sub>`embedding`</sub>` ≥ 0.45 跨语言桥接通道豁免词面误杀——三重机制共同构建了一个不依赖单一多语言 Embedding 模型对齐质量的鲁棒检索系统。

**5. Small-to-Big 上下文装配与父子块解耦**
检索阶段只让最小粒度单元（child chunks）进入 dense/BM25/recall/rerank，避免 parent 和 child 在候选池中竞争；生成阶段再通过 `parent_chunk_id` 装配完整 parent context。这彻底解耦了"召回单元"与"生成单元"，同时兼顾精度与上下文完整性。

**6. 图论算法对冲 LLM 随机性**
Louvain 社区发现、谱聚类、连通分量消融、多维中心性（degree / PageRank / betweenness / closeness）和 Dijkstra 隐式关系发现共同构成对传统 LLM 抽取噪声的系统性对冲。图谱不是 LLM 输出的被动容器，而是经过严格图论清洗后的稀疏知识骨架。

**7. TPE 自动超参优化与增量更新**
传统 GraphRAG 系统使用固定超参（阈值、权重）处理所有知识库，无法适应不同领域的语义密度和概念分布差异。SymboGraph 引入 Optuna TPE 自动寻优：基于 probe chunks 预抽取、LLM 成对裁判和代理目标函数，在 30 轮 trial 中自动搜索最优的 Dijkstra 阈值、关系置信度和图算法权重。最优参数版本化持久化，失败时自动回退。同时支持增量图谱更新，仅对变更文档局部重算，避免全量重建的冗余成本。

## 版本库规则

不进入版本库：

- `.env`、本地密钥、Authorization header 或 provider 响应。
- `data/`、`output/`、`models/`、`comparative_experiment/` 运行数据。
- `node_modules/`、`.next/`、`dist/`、`build/`、coverage、Playwright 报告。
- `.db`、`.sqlite*`、`__pycache__/`、`*.pyc`、`*.tsbuildinfo`、日志和临时文件。

应进入版本库：

- `apps/api`、`apps/web`、`packages/shared`、`scripts`、`infra`。
- README、`.env.example`、Docker 配置、测试、schema 和类型契约。

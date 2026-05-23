**English** | [中文](./README.md)

<p align="center">
  <img src="./assets/diagraph-logo.svg" alt="SymboGraph logo" width="132" height="132">
</p>

<h1 align="center">SymboGraph</h1>

**SymboGraph is a new-generation Neuro-Symbolic Agentic RAG system developed for serious enterprise and academic scenarios. this project is developed based on GraphRAG, but it distinguishes itself from the "semantic fragmentation" of traditional RAG and the "high cost and hallucination accumulation" of bulky GraphRAG architectures.**

The system parses PDFs, slides, documents, web pages, Notebooks, images, and Markdown into searchable text chunks, Qdrant dense vectors, PostgreSQL sparse knowledge graphs, and citation-backed question-answering results. Whether your materials are in Chinese or English, the system retrieves them uniformly; all data stays local without uploading to third parties.

As a general GraphRAG platform, SymboGraph's knowledge-base concept is not limited to any single document type—you can use it for course materials, research literature, technical manuals, legal contracts, or any text collection that requires structured decomposition and semantic linking.

## At A Glance

| Area                   | Implementation                                                                                                                                                                         |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Runtime                | Docker Compose, full-stack containers                                                                                                                                                  |
| Backend                | FastAPI, Pydantic, SQLAlchemy, NetworkX, LangGraph                                                                                                                                     |
| Frontend               | Next.js 16.2.4, React 19, TypeScript, TanStack Query, ECharts                                                                                                                          |
| Database               | PostgreSQL 16 for knowledge bases, file versions, chunks, graphs, QA sessions, and traces                                                                                              |
| Vector Store           | Qdrant 1.17.1, collection `knowledge_chunks`                                                                                                                                           |
| Cache And Coordination | Redis 7                                                                                                                                                                                |
| Model API              | OpenAI-compatible Embedding / Chat API, with independent endpoint configuration                                                                                                        |
| Retrieval              | Evidence-first retrieval: dense + BM25 + rerank base recall, evidence anchors, controlled graph navigation, then parent context assembly                                               |
| Graph                  | LLM candidates, chunk-vector semantic graph, graph algorithms for sparse construction, deduplication, communities, centrality, and hidden links; adaptive best-first chunk selection; TPE hyperparameter auto-optimization; supports incremental and full rebuild |
| Quality System         | Signal-policy-profile-judge four-tier quality architecture: adaptive tiered filtering and routing for chunks, concepts, and relations                                                  |
| QA                     | Agentic RAG: Perception → Planning → Retrieval → EvidenceEvaluator → Generation, with cross-lingual retrieval and pre-generation evidence assessment                                   |

## Technology Stack

| Layer                  | Technology                                                          | Role                                                                                                       |
| ---------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Frontend               | Next.js 16.2.4, React 19, TypeScript, TanStack Query, ECharts       | Knowledge-base management, upload and ingestion UI, search, QA, graph browsing, runtime settings           |
| API                    | FastAPI, Pydantic, SQLAlchemy, LangGraph                            | REST / SSE APIs, typed validation, transaction orchestration, ingestion, retrieval, and QA orchestration   |
| Graph Algorithms       | NetworkX, NumPy, SciPy                                              | Sparse construction, connected components, Louvain, spectral clustering, centrality, Dijkstra hidden links |
| Quality System         | Signal engineering, rule policies, domain profiles, LLM-as-judge    | Chunk/Concept/Relation tiered filtering, adaptive domain baselines, cached judge                           |
| Database               | PostgreSQL 16                                                       | Knowledge bases, file versions, chunks, graphs, QA sessions, traces, and compensation records              |
| Vector Search          | Qdrant 1.17.1                                                       | Parent / child chunk vectors, dense recall, vector health checks                                           |
| Lexical Search         | PostgreSQL text data, BM25                                          | Child chunk lexical recall and hybrid fusion                                                               |
| Cache And Coordination | Redis 7                                                             | Runtime cache, task coordination, service dependency                                                       |
| Parsing                | PyMuPDF, PPTX / DOCX / Markdown / HTML / Notebook parsers, OCR path | Convert heterogeneous documents into structured sections and text                                          |
| Model API              | OpenAI-compatible Embedding / Chat API                              | Embeddings, summaries, keywords, entity candidates, relation candidates, answer generation                 |
| Reranking              | Lightweight reranker, optional Cross-Encoder                        | Reorder fused candidates by relevance                                                                      |
| Hyperparameter Optimization | Optuna TPE (Tree-structured Parzen Estimator)                  | Auto-search optimal thresholds and weights for graph algorithms, persist best params                       |
| Deployment             | Docker Compose                                                      | Fixed service boundaries, dependency versions, local persistence                                           |
| Testing                | pytest, Vitest, Next build, Docker smoke                            | Behavioral regression, frontend/backend contracts, no-fallback quality gates                               |

## Core Capabilities

| Capability                   | Description                                                                                                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Multi-format parsing         | Supports PDF, PPT/PPTX, DOCX, Markdown, TXT, Notebook, HTML, and image materials                                                                 |
| Parent-child chunking        | Parent chunks keep full context; child chunks drive precise recall, reranking, and evidence citation                                             |
| Semantic chunking            | Long text is split by structure, semantic boundaries, sentence boundaries, and length limits; embedding similarity can assist boundary selection |
| Context-enriched vectors     | Embedding input includes file metadata, chapter, parent summary, neighboring child summaries, keywords, table markers, and formula markers       |
| Hybrid retrieval             | Qdrant child dense recall is fused with child BM25 recall before reranking                                                                       |
| Cross-lingual retrieval      | LLM translates queries into bilingual sub-queries; DocumentGrader uses embedding similarity to bridge language barriers                          |
| Graph enhancement            | Graph relations must link back to evidence chunks; the graph expands retrieval signals instead of replacing evidence                             |
| Graph-theoretic construction | Sparse graphs, communities, centrality, Dijkstra, and relation completion reduce noise and preserve key structure                                |
| Adaptive quality system      | Signal-policy-profile-judge tiered filtering with domain-aware baselines for chunk, concept, and relation quality                                |
| Observable QA                | Retrieval audits, model-call audits, agent traces, citations, and failure reasons are stored                                                     |
| Runtime checks               | Health checks, runtime checks, fallback state, Qdrant status, and model endpoint status are exposed                                              |
| Auto HPO                     | When `ENABLE_AUTO_HPO=true`, automatically runs TPE optimization before graph rebuild to find optimal thresholds and weight combinations     |
| Incremental graph update     | Recomputes only subgraphs tied to changed documents, avoiding unnecessary full rebuilds                                                          |

## System Architecture

```mermaid
flowchart TB
    USER["User Browser"] --> WEB["Next.js Web<br/>course-kg-web"]
    WEB -->|"HTTP / SSE"| API["FastAPI<br/>course-kg-api"]

    subgraph APP["Application Layer"]
        API --> INGEST["Ingestion Pipeline<br/>parse -> parent/child chunk -> quality routing -> augment -> vector upsert"]
        API --> RETRIEVAL["Retrieval Pipeline<br/>dense/BM25 -> fusion -> rerank -> parent context"]
        API --> GRAPH["Graph Pipeline<br/>LLM candidates -> quality filter -> vector similarity graph -> sparse graph -> communities/centrality/inference"]
        API --> QA["Agentic QA<br/>Perception -> Planning -> Retrieval -> EvidenceEvaluator -> Generation"]
        API -.-> QUALITY["Quality System<br/>internal library invoked by pipelines"]
    end

    subgraph STORE["Storage And Runtime"]
        PG[("PostgreSQL<br/>metadata, chunks, sparse graph, audit records, quality profiles")]
        QD[("Qdrant<br/>chunk vectors and similarity recall")]
        RD[("Redis<br/>runtime cache, embedding / retrieval cache, judge cache, distributed locks, Celery broker")]
        FS["data/<br/>knowledge-base files, parser artifacts, local persistence"]
    end

    subgraph MODEL["Model API"]
        EMB["OpenAI-compatible Embedding API<br/>independent endpoint config"]
        CHAT["OpenAI-compatible Chat API<br/>independent endpoint config"]
        MB["Model Bridge<br/>host model bridge (optional)"]
    end

    API --> PG
    API --> QD
    API --> RD
    API --> FS
    API --> EMB
    API --> CHAT
    API --> MB
```

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Web
    participant API
    participant Files as File Storage
    participant DB as PostgreSQL
    participant Vector as Qdrant
    participant Model as Model API

    User->>Web: Upload or select knowledge-base files
    Web->>API: Create ingestion batch
    API->>DB: Create batch, job, document/version state
    API->>Files: Store original file and parser artifacts
    API->>API: Parse chapters, pages, tables, formulas, and Notebook cells
    API->>API: Create parent/child chunks (with inline quality signal extraction and discard)
    API->>API: Adaptive chunking: dynamically select chunk_size / overlap / strategy per section
    API->>Model: Generate parent chunk summaries and keywords (ChatProvider)
    API->>API: Generate context-enriched text (contextual_embedding_text)
    API->>Model: Generate embeddings for all chunks (EmbeddingProvider)
    API->>Vector: Upsert active chunk vectors (with zero-vector check and compensation)
    API->>DB: Activate document version and chunks
    API->>Model: Extract entity/relation candidates (adaptive best-first chunk subset)
    API->>DB: Store concepts, relations, and evidence (with internal quality filtering)
    API->>API: Auto hyperparameter optimization (optional): TPE iterative search
    alt Full rebuild
        API->>API: Run sparse graph construction, communities, centrality, and inference
        API->>DB: Store graph-algorithm fields (centrality, communities, ranks)
        API->>API: Generate community summaries
    else Incremental update
        API->>DB: Recompute only subgraphs tied to changed documents
    end
    API-->>Web: Stream logs, progress, retry, and failure state over SSE
```

Ingestion uses explicit batch / job state and file-level locks. A knowledge base keeps at most one non-terminal ingestion batch at a time. PostgreSQL is the source of truth for lifecycle state; Qdrant and Redis are derived or runtime stores. Failures record compensation or actionable error context instead of silently degrading.

## Ingestion, Chunking, And Vectors

### Hierarchical Chunking

1. Parsers convert source files into `ParsedSection` objects while preserving chapter, page, source type, table, formula, Notebook cell, and image OCR metadata.
2. Each structured section creates a parent chunk that preserves the full section, page span, or natural semantic segment.
3. Parent chunks are split into child chunks for precise recall, reranking, and evidence localization.
4. Markdown and Notebook files prefer heading and cell hierarchy; ordinary long text uses semantic boundaries, sentence boundaries, and safe length limits.
5. When `SEMANTIC_CHUNKING_ENABLED=true` and text length reaches `SEMANTIC_CHUNKING_MIN_LENGTH`, embedding similarity can assist chunk boundary selection.

> **Design Intent (Why we do this)**: Fixed-size chunking leads to severe context fragmentation. Using a parent-child hierarchy and semantic chunking ensures the model leverages the high precision of child chunks during retrieval, while accessing the full context of parent chunks during generation. This completely decouples the "retrieval unit" from the "generation unit".

### Adaptive Chunking Profile Selection

The hierarchical strategy above does not use globally fixed `chunk_size` and `chunk_overlap`. For each `ParsedSection`, the system dynamically analyzes its content features before chunking and selects the optimal parameter combination from a candidate configuration space. This process emits a `chunk_adaptive` SSE event.

**Document Feature Vector**: A 12-dimensional normalized feature vector is extracted for each section:

| Feature | Formula | Description |
|---------|---------|-------------|
| Text length | $\min(1, L_{\text{tokens}} / 1800)$ | Normalized token count |
| Mean semantic-unit length | $\min(1, \bar{l}_{\text{unit}} / 80)$ | Average token length per semantic unit |
| Length coefficient of variation | $\min(1, \sigma_l / \max(\bar{l}_{\text{unit}}, 1))$ | Dispersion of semantic-unit lengths |
| Definition density | From semantic-density signals | Ratio of definitional statements |
| Entity density | From semantic-density signals | Ratio of named entities |
| Term density | From semantic-density signals | Ratio of domain terms |
| Unique-token ratio | From semantic-density signals | Lexical richness after deduplication |
| Formula signal | $\min\bigl(1, \frac{12 N_{\text{formula}}}{L_{\text{tokens}}} + 0.35 \cdot \mathbf{1}_{\text{has\_formula}}\bigr)$ | Formula density |
| Table signal | $\mathbf{1}_{\text{has\_table}}$ | Whether the section contains tables |
| Code-marker ratio | Code lines / total lines | Fraction of lines with `def`/`class`/`import` etc. |
| Symbol ratio | $\min(1, \frac{4 N_{\text{symbol}}}{L_{\text{chars}}})$ | Fraction of non-alphanumeric symbols |
| Structural noise | $\max(S_{\text{structural}}, 20 R_{\text{mojibake}})$ | Structural anomalies and mojibake risk |

**Spectral Document Shape**: The section is split into semantic units; each unit produces a feature vector. A covariance matrix is built from these vectors and eigendecomposed:

$$
\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge \lambda_4,\qquad
\tilde{\lambda}_k = \frac{\lambda_k}{\sum_{j=1}^4 \lambda_j}
$$

The spectral gap $\rho = \tilde{\lambda}_1 - \tilde{\lambda}_2$ measures semantic consistency; the semantic curvature $\kappa$ measures feature drift between adjacent semantic units.

**Candidate Configuration Space** (chunk_size, overlap, strategy):

| chunk_size | overlap | strategy |
|-----------|---------|----------|
| 512 | 64 | `sentence_aware` |
| 640 | 96 | `sentence_aware` |
| 800 | 120 | `semantic_or_sentence` |
| 960 | 144 | `semantic_or_sentence` |
| 700 | 160 | `recursive_structure_preserving` |

**Scoring Mechanism**: A composite score is computed for each candidate.

Complexity (structural disorder of the content):

$$
\gamma = \min\!\Bigl(1,\; 0.22 \cdot c_v + 0.18 \cdot f + 0.16 \cdot t + 0.18 \cdot r_{\text{code}} + 0.16 \cdot r_{\text{sym}} + 0.10 \cdot \kappa\Bigr)
$$

Density (knowledge richness):

$$
\delta = \min\!\Bigl(1,\; 0.35 \cdot d_{\text{term}} + 0.30 \cdot d_{\text{entity}} + 0.20 \cdot d_{\text{def}} + 0.15 \cdot r_{\text{unique}}\Bigr)
$$

Target size and overlap (driven by complexity and density):

$$
S^* = 920 - 380\,\gamma + 160\,\delta + 100\,\rho,\qquad
O^* = 80 + 130\,\gamma + 40\,\kappa
$$

Fitness scores:

$$
\phi_{\text{size}} = 1 - \min\!\Bigl(1, \frac{|S - S^*|}{700}\Bigr),\qquad
\phi_{\text{overlap}} = 1 - \min\!\Bigl(1, \frac{|O - O^*|}{220}\Bigr)
$$

Strategy bonus:

$$
\beta =
\begin{cases}
0.08 \cdot \max(\kappa, \delta) & \text{if strategy} = \text{`semantic\_or\_sentence'} \\
0.10 \cdot r_{\text{code}} + 0.05 \cdot f & \text{if strategy} = \text{`recursive\_structure\_preserving'} \\
0 & \text{otherwise}
\end{cases}
$$

Composite score:

$$
\mathcal{F} = \max(0, \min(1,\; 0.54\,\phi_{\text{size}} + 0.26\,\phi_{\text{overlap}} + 0.14\,\delta - 0.10\,\nu + \beta))
$$

where $\nu$ is structural noise. The candidate with the highest $\mathcal{F}$ is selected as the chunking profile for the current section and persisted into chunk metadata.

### Context-Enriched Embeddings

Child vectors are not built from child text alone. `contextual_embedding_text()` builds context-enriched input:

```text
file metadata
chapter, page, and source type
child chunk content
parent summary or parent content
neighboring child summaries
keywords
table, formula, and content-kind markers
```

Parent chunks keep their own text, summary, and keywords. Child chunks inherit parent semantic summaries and neighboring context, reducing context loss in fine-grained chunks. The current embedding text version is `contextual_enriched_v3`.

> **Design Intent (Why we do this)**: Isolated short text chunks easily suffer from semantic ambiguity when embedded alone (e.g., "this method", "the next step"). Forcing the injection of parent summaries and neighboring context before embedding acts as "contextual retrieval" at ingestion time, significantly improving recall accuracy in the dense retrieval stage.

### Deduplication And Idempotency

Ingestion detects duplicates by knowledge base, normalized title, and checksum. Unchanged files are skipped with `unchanged_checksum`; duplicate copies with the same normalized title and checksum are skipped with `duplicate_document`, avoiding duplicate chunks and vectors. Forced reingestion regenerates document versions, chunks, Qdrant vectors, and graph candidates.

## Quality System

SymboGraph incorporates a four-tier quality architecture—Signals, Policies, Profiles, and Judge—for differentiated tiered filtering and adaptive routing of chunks, concepts, and relations. The quality system is not a simple pass/fail binary; it computes multidimensional signals for each object, outputs structured decisions, and allows downstream pipelines to take different actions based on the decision.

```mermaid
flowchart LR
    RAW["Raw Object<br/>chunk / concept / relation"] --> SIG["Signal Layer<br/>TextQuality · StructuralRole · SemanticDensity · DomainSpecificity · EvidenceGrounding"]
    SIG --> POL["Policy Layer<br/>ChunkPolicy · ConceptPolicy · RelationPolicy"]
    POL --> ROUTE["Routing Decision<br/>discard · summary_only · evidence_only · retrieval_candidate · graph_candidate · accept · candidate_only"]
    PROF["Profile Layer<br/>stratified sampling · positive/negative examples · domain term baselines"] --> POL
    JUD["Judge Layer<br/>LLM-as-judge · Redis cache · defer fallback"] --> POL
```

### 1. Signal Layer (Quality Signals)

The signal layer extracts quantifiable quality metrics from raw text and metadata:

- **TextQuality**: length, normalized length, mojibake ratio, control character count, repeated line ratio, TOC similarity
- **StructuralRole**: structural labels (chapter/page/filename), container hints, TOC pages, Notebook output
- **SemanticDensity**: unique token ratio, definition score, entity density, term density, formula/table markers
- **DomainSpecificity**: genericity score, specificity score, local IDF
- **EvidenceGrounding**: text span, chunk anchor, document anchor, endpoint match, support count

For concepts, the signal layer also includes **ModelJudgment** (LLM judge verdict, score, reasons).

### 2. Policy Layer (Quality Policies)

The policy layer maps signals into discrete routing decisions:

**ChunkQualityPolicy** decision space:

| Action                | Meaning                                        | Downstream Impact                                           |
| --------------------- | ---------------------------------------------- | ----------------------------------------------------------- |
| `discard`             | Mechanical noise, drop immediately             | No embedding, no retrieval, no graph                        |
| `summary_only`        | TOC page or structural label                   | Summary only, no retrieval or graph                         |
| `evidence_only`       | Too short or Notebook output                   | Embeddable and retrievable, but no summary, no graph        |
| `retrieval_candidate` | Ordinary content chunk                         | Embed, retrieve, summarize, no graph                        |
| `graph_candidate`     | High semantic density (definition/entity/term) | Embed, retrieve, summarize, participate in graph extraction |
| `embed_only`          | Code block without domain context              | Embed only, no retrieval or graph                           |

Chunk quality score formula:

$$
S_{\text{chunk}} = 0.30 \cdot \min\Bigl(1, \frac{L_{\text{norm}}}{600}\Bigr) + 0.25 \cdot D_{\text{term}} + 0.20 \cdot R_{\text{unique}} + 0.15 \cdot D_{\text{def}} + 0.05 \cdot \mathbf{1}_{\text{formula}} + 0.05 \cdot \mathbf{1}_{\text{table}} - 0.35 \cdot \mathbf{1}_{\text{toc}} - 0.40 \cdot \min\Bigl(1, 20 \cdot R_{\text{mojibake}}\Bigr)
$$

Where *L*`<sub>`norm`</sub>` is normalized length, *D*`<sub>`term`</sub>` is term density, *R*`<sub>`unique`</sub>` is unique token ratio, *D*`<sub>`def`</sub>` is definition score, and *R*`<sub>`mojibake`</sub>` is mojibake ratio.

**ConceptQualityPolicy** decision space is `accept` / `reject`:

$$
S_{\text{concept}} = \max\Bigl(S_{\text{specificity}},\; 0.35 D_{\text{def}} + 0.25 D_{\text{term}} + 0.20 D_{\text{entity}}\Bigr) - 0.35 S_{\text{structural}} - 0.25 G_{\text{genericity}}
$$

Admission requires no hard-rejection reasons (too short, mojibake, path/filename, structural container, low specificity, insufficient evidence) and score *S*`<sub>`concept`</sub>` ≥ 0.45.

**RelationQualityPolicy** decision space is `accept` / `candidate_only`:

$$
S_{\text{relation}} = 0.40 \cdot c + 0.25 \cdot \mathbf{1}_{\text{src}} + 0.25 \cdot \mathbf{1}_{\text{tgt}} + 0.10 \cdot \min\Bigl(1, \frac{n_{\text{support}}}{3}\Bigr)
$$

Where *c* is LLM confidence, **1**`<sub>`src`</sub>` / **1**`<sub>`tgt`</sub>` indicate whether the source/target concept appears in the evidence text. `inferred` or `related_to` relations are forced to `candidate_only`.

> **Design Intent (Why we do this)**: Traditional RAG/GraphRAG systems often apply only coarse-grained filtering before graph construction, allowing TOC pages, garbled text, and repeated extraction noise to pollute the vector store and knowledge graph. SymboGraph's tiered quality routing sends different content types to their proper destinations—noise is discarded, structural labels are summary-only, high-semantic-density blocks join the graph, and ordinary blocks handle retrieval—guaranteeing downstream quality from the data source.

### 3. Profile Layer (Domain Quality Profile)

The profile layer builds an adaptive quality baseline for each knowledge base:

1. **Stratified sampling**: samples by `(content_kind, chapter)`, extracting short/medium/long examples from each stratum to ensure coverage
2. **Positive examples**: chunks with high definition scores or high term density, serving as domain "good content" exemplars
3. **Negative examples**: TOC pages, garbled pages, and high-structural-score chunks, serving as domain "noise" exemplars
4. **Domain terms**: top-40 most frequent long tokens from the sample, used for subsequent concept specificity calculations
5. **Relation schema hints**: 13 predefined allowed relation types (`is_a`, `part_of`, `prerequisite_of`, `used_for`, `causes`, `derives_from`, `compares_with`, `example_of`, `defined_by`, `formula_of`, `solves`, `implemented_by`, `related_to`)

Profile data is stored in the `quality_profiles` table, versioned, and integrity-checked via SHA256 hash. Profiles are referenced during graph construction and LLM judging, giving quality decisions domain context.

### 4. Judge Layer (Quality Judge)

The judge layer is an optional LLM-as-judge enhancement:

- Receives policy-layer candidates plus the domain profile, and asks the LLM to output `accept` / `reject` / `candidate_only` / `defer`
- Cache key binds `(course_id, profile_version, target_type, model, candidate_hash)`; cache hits in Redis return the cached result directly
- When the LLM is unavailable, falls back to `defer`, fully returning control to the rule-based policy layer, ensuring system availability

> **Design Intent (Why we do this)**: The rule policy layer is fast and stable but lacks flexibility for complex domain-boundary cases. The LLM judge acts as a "slow thinking" supplement, intervening only when the rule layer cannot decide; Redis caching prevents repeated calls. This "rules first, LLM second, cache fallback" three-tier architecture balances latency, cost, and accuracy.

## Graph Construction

SymboGraph graph construction is not a single KNN or LLM extraction procedure. It is an evidence-first, multi-stage pipeline: the LLM discovers candidate entities and explicit semantic relations, quality gates enforce factual constraints, vector similarity provides controlled semantic candidates, classic graph algorithms analyze structure, and LLM evidence completion plus Dijkstra inference repair only local structure within audited boundaries. PostgreSQL is the source of truth for concepts, relations, and lifecycle state; Qdrant provides chunk vectors and similarity signals, but does not decide factual relations by itself.

This update introduces three major capabilities: **adaptive best-first chunk selection**, **TPE automatic hyperparameter optimization**, and **incremental graph updates**:
- **Adaptive Best-First**: Dynamically selects the most information-gain chunks for LLM extraction based on six coverage signals (document, chapter, section, content kind, embedding cluster proxy, and low-frequency terms), avoiding indiscriminate calls to all chunks.
- **TPE Hyperparameter Optimization**: Optionally runs Auto HPO before full rebuild, using Optuna's `TPESampler` over 30 trials to automatically search for the optimal Dijkstra semantic threshold, relation confidence threshold, and graph algorithm weight combinations.
- **Incremental Update**: Only recomputes subgraphs tied to changed documents, preserving concepts and relations from unchanged documents, significantly reducing update costs.

```mermaid
flowchart TB
    CHUNK["Active child chunks<br/>passed ChunkQualityPolicy"] --> VEC["Qdrant chunk vectors<br/>derived vector signals"]
    CHUNK --> LLM["LLM extraction<br/>candidate entities + explicit relations"]

    LLM --> MERGE["Concept merge<br/>canonical name, aliases, dedupe, chapter refs"]
    MERGE --> CONCEPT_GATE["ConceptQualityPolicy<br/>entity quality filter"]
    CONCEPT_GATE --> UPSERT["Upsert graph candidates<br/>concepts, relations, evidence with hard gate"]

    VEC --> CENTROID["Concept vector centroid<br/>evidence chunk centroid + L2 normalization"]
    UPSERT --> CENTROID
    CENTROID --> SPARSE["Semantic sparse candidates<br/>dynamic KNN + semantic threshold + mutual/inbound quota"]
    SPARSE --> CANDIDATE["GraphRelationCandidate<br/>candidate_only, not factual edges by default"]

    UPSERT --> EXPLICIT["Verified explicit relations<br/>evidence-supported factual edges"]
    CANDIDATE --> GRAPH["Verified graph + candidates<br/>input for centrality/community analysis"]
    EXPLICIT --> GRAPH

    UPSERT --> HPO["Auto HPO (optional)<br/>TPE from probe chunks"]
    GRAPH --> ENRICH["enrich_course_graph<br/>build_sparse_edges → analyze → complete → dijkstra → final analyze"]
    HPO --> ENRICH
    ENRICH --> COMMUNITY["Community summaries<br/>rebuild_graph_community_summaries"]
    ENRICH --> PG["PostgreSQL graph tables<br/>concepts, verified relations, candidates, metrics"]

    PG --> RETRIEVAL["Evidence-first retrieval<br/>expands only along auditable evidence"]
    PG --> UI["Community-aware graph UI"]
```

### 0. Adaptive Best-First Chunk Selection

Before LLM extraction, the system does not blindly call the model on all active chunks. Instead, it builds an **adaptive extraction plan** that prioritizes chunks by information gain:

$$
\Delta_{\text{cov}}(c) = \sum_{d \in \text{new}(c)} w_d + \sum_{h \in \text{new}(c)} w_h + \sum_{s \in \text{new}(c)} w_s + \sum_{k \in \text{new}(c)} w_k + \sum_{e \in \text{new}(c)} w_e + \sum_{t \in \text{new}(c)} w_t
$$

Where $\text{new}(c)$ denotes newly covered dimensions when chunk $c$ is added. The coverage dimensions and weights are:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Document coverage | $0.18$ | First time a document is covered |
| Chapter coverage | $0.16$ | First time a chapter is covered |
| Section coverage | $0.10$ | First time a section is covered |
| Content kind coverage | $0.08$ | First time a content kind is covered |
| Embedding cluster proxy | $0.12$ | First time a cluster is covered |
| Low-frequency terms | $\min(0.18, 0.015 \cdot n_{\text{rare}})$ | Weighted by number of new rare terms |
| Graph gap fill | $0.08$ | Bonus when the current graph lacks concepts linked to this chunk |

Selection uses a **greedy best-first** strategy: at each step, pick the unselected chunk with maximum $\Delta_{\text{cov}}$. Stopping criteria:
- Cumulative coverage exceeds threshold (default $0.85$)
- Marginal gain $\Delta_{\text{cov}} < 0.03$
- Soft-start budget reached (`GRAPH_EXTRACTION_SOFT_START_BUDGET`, default 120)
- All chunks selected

Selected chunks enter LLM graph extraction; remaining chunks stay in the vector store and retrieval system but do not consume model call budget.

### 1. Entities And Evidence

Each concept stores a canonical name, aliases, chapter references, importance, evidence chunk count, and quality audit fields. Concept vectors are not generated from names. They are centroids of supporting evidence chunk vectors, followed by L2 normalization:

$$
\bar{\mathbf{x}}_i = \frac{1}{|C_i|}\sum_{c \in C_i}\mathbf{z}_c,\qquad
\mathbf{x}_i = \frac{\bar{\mathbf{x}}_i}{\lVert \bar{\mathbf{x}}_i\rVert_2}
$$

*C*`<sub>`i`</sub>` is the set of active child chunks supporting concept *i*, and $\mathbf{z}_c$ is the chunk embedding. The purpose is to keep concept vectors faithful to the local course material instead of the LLM's generic pre-training semantics for the concept name.

### 2. Explicit LLM Relations And Quality Gates

LLM-extracted relations are not inserted into the graph directly. The system checks the relation type allowlist, endpoint existence, self-loops, confidence, support count, evidence chunk, whether endpoint names can be matched in the evidence text, and whether the relation source is allowed. Only relations that pass these quality gates enter the verified graph and become factual edges for centrality, community detection, and retrieval path planning.

The quality gate owns factuality: the LLM proposes possible relations, while the gate verifies whether each relation is supported by course material. This boundary is what prevents silent graph pollution.

### 3. Dynamic KNN + Semantic Threshold + Mutual/Inbound-Quota Sparse Graph

This is not standard Radius-NN graph construction, where neighbors are admitted solely by a fixed radius or distance threshold. The current implementation first generates candidates with dynamic KNN and a semantic similarity threshold, then keeps mutual nearest neighbors or candidates accepted by the reverse inbound quota. This layer exists to screen semantic candidate edges and control graph scale; it does not make final factual-relation decisions.

Each concept dynamically chooses outgoing candidates from its evidence volume:

$$
K_i = \mathrm{clamp}\bigl(4 + \lfloor \log_2(1 + m_i) \rfloor,\, 4,\, 12\bigr)
$$

Each concept dynamically limits accepted reverse inbound candidates from chapter coverage:

$$
B_i = \mathrm{clamp}\bigl(2 + \lfloor \log_2(1 + r_i) \rfloor,\, 2,\, 8\bigr)
$$

*m*`<sub>`i`</sub>` is evidence chunk count and *r*`<sub>`i`</sub>` is chapter reference count. The system keeps mutual nearest neighbors, candidates accepted by the reverse inbound quota *B*`<sub>`i`</sub>`, and high-confidence explicit LLM relations, keeping edge count close to linear in node count.

Pure semantic candidate edges are marked with `relation_source="semantic_sparse"` and `candidate_only=true`. They can be used as repair hints, audit objects, or later validation material, but they do not enter the centrality and community graph by default. This prevents similarity noise from being amplified into factual structure.

### 4. Edge Weights, Structure Analysis, And Pruning

Edge weight combines evidence support, relation confidence, semantic similarity, co-occurrence strength, and chapter-structure consistency:

$$
w_{ij}=
0.30\,s_{ij}^{\mathrm{evidence}}
+0.25\,c_{ij}^{\mathrm{relation}}
+0.20\,s_{ij}^{\mathrm{sem}}
+0.15\,s_{ij}^{\mathrm{cooccur}}
+0.10\,s_{ij}^{\mathrm{structure}}
$$

The final *w*`<sub>`ij`</sub>` is clipped to [0,1]. The verified graph stage runs:

- Connected-component analysis: identifies isolated structures, noise nodes, and major knowledge clusters.
- Louvain community detection: primary community labels and frontend color groups.
- Spectral clustering: secondary partitions for large components and large communities.
- Centrality: degree, weighted degree, PageRank, betweenness, closeness, and a combined `centrality_score`.
- Graph simplification: keeps central nodes, community representatives, bridge edges, and high-evidence concepts.

Centrality is not computed on the pure semantic candidate graph. It is computed on the evidence-gated verified graph; `graph_rank_score` also mixes concept importance and evidence count so a single topology signal cannot dominate ranking.

### 5. LLM Relation Completion And Dijkstra Inference

After structure analysis, the system selects local neighborhoods around high-`graph_rank_score` nodes, collects related evidence snippets, and asks the LLM to complete only relations supported by those snippets. Completion results still return through the same `RelationQualityPolicy + hard gate`; they cannot bypass validation.

Dijkstra searches 2-3 hop hidden structural hints on a non-negative cost graph:

$$
\mathrm{cost}_{ij}=\frac{1}{0.05+w_{ij}}
$$

If endpoint semantic similarity is high and path cost is low, the system may write an inferred edge with `relation_source="dijkstra_inferred"`. Inferred edges are navigation, audit, and candidate-completion hints; they are not unconditional facts. Answers must still return to original chunk evidence.

### 6. TPE Automatic Hyperparameter Optimization (Auto HPO)

When `ENABLE_AUTO_HPO=true`, the system automatically runs hyperparameter optimization before a **full rebuild**. HPO does not trial directly on the full graph (too expensive); instead, it builds a surrogate evaluation from a small set of **probe chunks** (default 5):

**Phase 1 — Candidate Generation and Feature Extraction**:
Generate candidate parameter sets from multiple seeds (conservative, aggressive, balanced, and random interpolations). For each parameter set, build a mock graph from pre-extracted probe payloads and extract features:
- Graph scale features: node count, edge count, connected component count, average degree
- Structural quality features: community modularity, clustering coefficient, centrality distribution entropy
- Semantic features: Dijkstra inferred edge ratio, LLM explicit relation ratio, semantic sparse edge ratio
- Hard constraint checks: excessive isolated nodes, edge count explosion, community degradation

**Phase 2 — Pairwise Judge**:
Use LLM as a judge to perform pairwise comparisons (A vs B) of candidate parameters, outputting `winner`, `confidence`, and `reasons`. After collecting at least `HPO_JUDGE_MIN_LABELS` (default 6) valid labels, train a **judge-learned surrogate objective function**:

$$
\mathcal{L}(\mathbf{f}, \mathbf{w}, b) = \sum_{(i,j) \in \mathcal{P}} \mathbb{1}[\hat{y}_{ij} = y_{ij}] \cdot \max(0, \; |s_i - s_j| - \epsilon)
$$

Where $\mathcal{P}$ is the set of judge-labeled candidate pairs, $\mathbf{f}$ is the feature vector, $\mathbf{w}$ is the learned weight vector, and $s_i = \mathbf{w}^\top \mathbf{f}_i + b$ is the candidate score. The surrogate objective maps high-dimensional features to a scalar quality score, so subsequent TPE only needs to optimize a scalar target.

**Phase 3 — TPE Iterative Optimization**:
Use Optuna's `TPESampler` (Tree-structured Parzen Estimator) to maximize the surrogate objective over 30 trials:

$$
\theta^* = \arg\max_{\theta \in \Theta} \; g_{\text{surrogate}}(\theta; \mathcal{D}_{\text{probe}})
$$

Where $\theta$ is an 11-dimensional hyperparameter vector:

| Hyperparameter | Search Range | Default | Description |
|----------------|--------------|---------|-------------|
| `min_relation_confidence` | $[0.50, 0.85]$ | $0.62$ | Minimum relation confidence |
| `min_accepted_relation_weight` | $[0.45, 0.78]$ | $0.56$ | Minimum accepted relation weight |
| `dijkstra_semantic_threshold` | $[0.65, 0.88]$ | $0.74$ | Dijkstra inference semantic threshold |
| `w_pagerank` | $[0.05, 0.50]$ | $0.20$ | PageRank weight |
| `w_betweenness` | $[0.05, 0.50]$ | $0.20$ | Betweenness weight |
| `w_degree` | $[0.05, 0.50]$ | $0.25$ | Degree weight |
| `w_weighted_degree` | $[0.05, 0.50]$ | $0.25$ | Weighted degree weight |
| `w_closeness` | $[0.05, 0.30]$ | $0.10$ | Closeness weight |
| `w_centrality` | $[0.10, 0.80]$ | $0.50$ | Combined centrality weight |
| `w_llm_importance` | $[0.05, 0.60]$ | $0.25$ | LLM importance weight |
| `w_evidence` | $[0.05, 0.60]$ | $0.25$ | Evidence weight |

The optimal parameters are persisted to the `course_model_hyperparameters` table and injected into `GraphHyperparameters` during full rebuild. If HPO fails, the system falls back to the last successful parameters or defaults.

### 7. Incremental Graph Updates

For partial knowledge-base changes (e.g., modifying or deleting a few documents), the system supports **incremental graph updates**, avoiding unnecessary full rebuilds:

1. **Change Detection**: Compare changed document chunks with current graph evidence to identify affected concepts and relations.
2. **Local Cleanup**: Call `delete_document_graph_incremental` to remove concepts, aliases, and relations supported only by changed documents; concepts and relations supported by other documents remain unaffected.
3. **Local Re-extraction**: Run adaptive best-first selection and LLM extraction on changed documents' active chunks.
4. **Local Recomputation**: When running graph algorithms (communities, centrality), only recompute affected subgraphs; unaffected nodes and edges retain their original metrics.
5. **Transaction Consistency**: All cleanup and re-extraction operations are performed within an explicit transaction; on failure, rollback ensures the graph never ends up in a half-deleted state.

Incremental update time complexity is proportional to the number of changed documents, not the total chunk count. When too many documents change or for initial construction, the system automatically falls back to full rebuild.

### 8. Collaboration Boundaries And Caveats

The architecture does not let any single algorithm decide the graph alone. Its signals constrain each other:

- The LLM discovers candidate semantics, but must obey evidence-first constraints.
- Quality gates prevent hallucinated, weakly supported, or disallowed relation types from entering the verified graph.
- Dynamic KNN sparse construction controls semantic candidate scale and reduces hubness and hairball risk.
- Graph algorithms provide communities, centrality, bridge structure, and pruning, but do not replace evidence.
- LLM completion and Dijkstra inference only provide local repair and navigation hints; they cannot bypass validation.

The limitations and course-material constraints are explicit:

- The graph represents knowledge supported by local course evidence chunks; it does not guarantee coverage of general encyclopedic knowledge.
- If course materials have chaotic chapters, OCR errors, overly short chunks, duplicate files, or poor concept extraction, graph quality will degrade accordingly.
- Dynamic KNN still favors locally dense semantic regions. Cross-chapter bridge concepts may be missed and need explicit LLM relations, evidence completion, or retrieval-path correction.
- Centrality can amplify bad edges that already entered the verified graph, so relation gates, candidate-edge isolation, and regression tests must stay strict.
- `semantic_sparse` and `dijkstra_inferred` edges must not be treated as answer evidence by themselves; user-visible answers must cite original chunks.
- Changing the embedding model, chunking strategy, or relation extraction prompt requires recalibrating similarity thresholds, quality gates, and graph quality metrics.

The frontend colors graph nodes by Louvain community, sizes nodes by centrality and graph rank, and renders inferred edges as dashed lines. Users can filter communities and open key entity details quickly, but the graph view should be understood as a course-evidence navigation graph, not an unsupported authoritative ontology.

## Retrieval And QA

SymboGraph's QA pipeline uses a **Perception → Retrieval Planning → Base Retrieval → Evidence Navigation → EvidenceEvaluator → Generation** evidence-first agent architecture orchestrated by LangGraph. Every node writes to `agent_trace_events`, and the frontend renders the live trace via SSE.

```mermaid
flowchart LR
    Q["Question"] --> PER["Perception<br/>intent · entity extraction · graph concept matching"]
    PER -->|"greeting / clarify"| AG["AnswerGenerator"]
    PER -->|"needs retrieval"| PLAN["RetrievalPlanner<br/>evidence-first params · cross-lingual translation"]
    PLAN --> BASE["BaseRetrieval<br/>dense + BM25 + fusion + rerank"]
    BASE --> ANCHOR["EvidenceAnchorSelector<br/>reliable chunk / concept anchors"]
    ANCHOR --> PATH["EvidenceChainPlanner<br/>verified edges / community routing"]
    PATH --> GRAPH["ControlledGraphEnhancer<br/>collect only planned-path evidence"]
    GRAPH --> ASSEMBLE["EvidenceAssembler<br/>base + anchor + graph evidence"]
    ASSEMBLE --> GRADE["DocumentGrader<br/>0.4·overlap + 0.6·embedding_sim"]
    GRADE --> EVAL["EvidenceEvaluator<br/>pre-generation sufficiency check"]
    EVAL -->|"insufficient + retry<2"| PLAN
    EVAL -->|"sufficient / insufficient+retry≥2"| CS["ContextSynthesizer"]
    CS --> AG
    AG --> CC["CitationChecker"]
    CC --> CV["CitationVerifier"]
    CV --> REFL["Reflection<br/>post-generation (default off)"]
    REFL --> AC["AnswerCorrector"]
    AC --> CS
```

### Perception

The Perception node understands user intent, extracts entities, and matches them against the knowledge-base graph:

1. **Fast-path**: greetings route to `direct_answer`; empty or anaphoric queries route to `clarify`.
2. **LLM perception**: calls ChatProvider to classify intent (`definition` / `comparison` / `analysis` / `application` / `procedure`), extract entities, and generate sub-queries.
3. **Graph concept matching**: matches extracted entities against `concepts` and `concept_aliases`, retrieving matched concept communities and one-hop neighbors.

Perception outputs:

- `intent`: question type
- `entities` / `matched_concepts`: extracted entities and graph matches
- `perceived_communities`: relevant community IDs
- `suggested_strategy`: recommended evidence-first route (`base_retrieval`, `evidence_chain`, `community`)
- `needs_graph`: whether graph enhancement is needed

### RetrievalPlanner

The planning layer configures evidence-first retrieval based on Perception output and performs cross-lingual query translation:

**Strategy selection:**

| Intent                             | Condition                  | Evidence-first params                                                                 |
| ---------------------------------- | -------------------------- | ------------------------------------------------------------------------------------- |
| `definition` / `formula`           | `needs_graph=false`        | Base retrieval and evidence evaluation only                                           |
| `comparison` or `needs_graph=true` | —                          | Enable verified-edge path planning after base recall                                  |
| `application` / `procedure`        | matched concepts exist     | Allow controlled evidence-chain planning up to 3 hops                                 |
| `analysis`                         | communities or broad query | Use community summaries only as routing hints; final answers still cite source chunks |

**Cross-lingual query expansion:**

The system detects query language (Chinese / English) and uses LLM to translate to the opposite language:

$$
Q_{\mathrm{bilingual}} = \{q_{\mathrm{original}},\; q_{\mathrm{translated}}\} \cup Q_{\mathrm{sub}}
$$

After deduplication, all sub-queries enter BaseRetrieval. This allows a Chinese query like "最大流" to also match English knowledge-base materials via the translated sub-query "max flow".

> **Design Intent (Why we do this)**: Multilingual embedding models often struggle with cross-lingual alignment. Explicitly translating queries and including bilingual sub-queries allows the retrieval engine to probe the document store in multiple linguistic forms simultaneously. This is a much more robust engineering solution than relying solely on the embedding model's internal alignment.

### Evidence-first Retrieval Execution

Execution always retrieves text evidence first, then uses the graph for navigation:

| Stage                  | Backend/node                                               | Description                                                          |
| ---------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------- |
| Base recall            | `hybrid_search_chunks` / `hybrid_search_chunks_with_audit` | Dense + BM25 hybrid recall, fusion, and reranking                    |
| Anchor selection       | `select_evidence_anchors`                                  | Select reliable anchor chunks / anchor concepts from base recall     |
| Path planning          | `plan_evidence_chains`                                     | Use verified graph edges only; community summaries are routing hints |
| Controlled enhancement | `controlled_graph_enhancement`                             | Collect evidence chunks only along planned paths; no neighbor flood  |
| Evidence assembly      | `assemble_evidence_documents`                              | Merge base evidence, anchor evidence, and graph-path evidence        |

All strategies follow the **Small-to-Big** principle: only the finest-grained units enter recall and reranking (child chunks, or parent chunks that have no children and thus represent the finest granularity themselves); parent context is assembled later via `parent_chunk_id` where available.

### DocumentGrader

Grades recalled documents for admission, fusing lexical overlap and vector semantic similarity:

$$
\mathrm{grade\_score} = 0.40 \cdot r_{\mathrm{overlap}} + 0.60 \cdot s_{\mathrm{embedding}}
$$

Where:

- *r*`<sub>`overlap`</sub>` = |*T*`<sub>`q`</sub>` ∩ *T*`<sub>`d`</sub>`| / |*T*`<sub>`q`</sub>`|, with *T*`<sub>`q`</sub>` the query term set and *T*`<sub>`d`</sub>` the document title+snippet+content term set
- *s*`<sub>`embedding`</sub>` is the cosine similarity between query and document vectors; when the raw vector is unavailable, it falls back to the dense score recorded at retrieval time

Admission rules (pass if any holds):

$$
\begin{cases}
\mathrm{grade\_score} \ge 0.35 & \text{(primary gate)} \\
s_{\mathrm{embedding}} \ge 0.45 & \text{(cross-lingual bridge gate)} \\
r_{\mathrm{overlap}} \ge 0.25 \;\land\; \mathrm{original\_score} \ge 0.3 & \text{(auxiliary gate)}
\end{cases}
$$

The cross-lingual bridge gate solves a critical problem: a Chinese query "最大流" and English material "max flow" share weak overlap in the `text-embedding-v4` vector space, but LLM-translated sub-queries can recall relevant chunks via dense search. In such cases *r*`<sub>`overlap`</sub>` may be near zero while *s*`<sub>`embedding`</sub>` remains high; the bridge gate prevents these valid cross-lingual results from being killed by monolingual term matching.

> **Design Intent (Why we do this)**: This is a funnel specifically designed to break the "cross-lingual wall". A Chinese query and English material often share zero literal overlap but high semantic relevance. The *s*`<sub>`embedding`</sub>` ≥ 0.45 cross-lingual bridge gate acts as an exemption channel, elegantly preventing purely lexical (BM25) mismatch from killing valid cross-lingual results.

### EvidenceEvaluator

**Before answer generation**, the EvidenceEvaluator assesses whether retrieved evidence is sufficient. This is SymboGraph's **pre-generation reflection** mechanism:

For each graded document, extract `grade_score` and compute:

$$
\bar{g} = \frac{1}{n}\sum_{i=1}^{n} g_i,\qquad g_{\max} = \max_i g_i
$$

Intent-dependent minimum evidence thresholds:

$$
\begin{cases}
(n_{\min}, \bar{g}_{\min}) = (1,\, 0.25) & \text{if intent} \in \{\text{definition},\, \text{procedure}\} \\
(n_{\min}, \bar{g}_{\min}) = (2,\, 0.20) & \text{if intent} \in \{\text{comparison},\, \text{analysis}\} \\
(n_{\min}, \bar{g}_{\min}) = (1,\, 0.20) & \text{otherwise}
\end{cases}
$$

Sufficiency condition:

$$
\mathrm{sufficient} \;\Leftrightarrow\; g_{\max} \ge 0.35 \;\land\; n \ge n_{\min} \;\land\; \bar{g} \ge \bar{g}_{\min}
$$

If only an anchor exists but quantity/score is marginal, the run is marked `marginal` and generation proceeds. If evidence is insufficient and `retry_count < 2`, the flow routes back to `RetrievalPlanner` with doubled `top_k`. If `retry_count >= 2`, `low_evidence=true` is set and generation proceeds with a disclaimer in the prompt and no forced citations.

> **Design Intent (Why we do this)**: This breaks the flawed traditional RAG paradigm of "generate answers no matter what garbage was retrieved". As a defensive assessment layer, it gives the system the ability to "know what it doesn't know". Intercepting low-quality retrievals and failing gracefully is crucial for reliability in professional domain QA.

### Post-Generation Loop (Default Off)

`ENABLE_POST_GENERATION_REFLECTION=false` by default. When enabled, post-generation nodes execute:

- **CitationVerifier**: samples high-importance claims for NLI verification.
- **Reflection**: LLM evaluates the answer for hallucination, insufficient coverage, or contradiction, returning `has_issue` / `issue_type` / `suggestion`.
- **AnswerCorrector**: adjusts strategy based on reflection results (expand top_k, rewrite query, or regenerate from high-confidence documents).

These nodes are observable in traces but do not participate in the main loop by default, avoiding extra latency and model call costs. The pre-generation `EvidenceEvaluator` already covers most insufficient-evidence scenarios.

### Evidence-first Graph Navigation

Every question starts with base recall; only comparison, derivation, procedure, or broad analysis questions enable graph navigation after evidence anchors are selected. `semantic_sparse`, `dijkstra_inferred`, `candidate_only`, and relations without evidence chunks do not participate in default path planning.

Cacheable retrieval results written to Redis must bind keys to course, query, filters, model, embedding text version, and relevant config. Cache hits still carry audit metadata.

### Small-To-Big Retrieval

The main retrieval path sends only the finest-grained units through recall and reranking, then attaches parent context:

```text
finest-grained dense recall + finest-grained BM25 recall
-> weighted fusion
-> rerank
-> load parent_chunk_id (if available)
-> finest-grained evidence + parent context (if available) + citations
```

This avoids both coarse recall from overly large chunks and missing context from tiny chunks. Retrieval results carry `retrieval_granularity=child_with_parent_context`, dense score, BM25 score, fused score, rerank score, graph boost, and model audit fields.

## Technical Advantages

| Advantage                  | Detail                                                                                                                                             |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Evidence-first             | Answers, relations, and graph expansion return to real chunks and parent context                                                                   |
| Context and precision      | Child chunks provide precise recall; parent chunks provide complete explanation context                                                            |
| Controlled graph structure | Dynamic KNN, semantic thresholding, and reverse inbound quotas cap edge growth, while components and communities reduce noise                     |
| Adaptive quality system    | Signal-policy-profile-judge four-tier architecture with differentiated tiered routing for chunks, concepts, and relations                          |
| Domain quality profiles    | Auto-built knowledge-base specificity baselines let quality judgments adapt to different domains                                                   |
| Document-aware             | Preserves chapters, pages, formulas, tables, Notebook cells, and source types                                                                      |
| Auditable                  | Stores batch/job/log state, model calls, retrieval scores, fallback state, and citations                                                           |
| Recoverable                | PostgreSQL stores lifecycle state; Qdrant / Redis can be repaired from durable records                                                             |
| No silent degradation      | Missing models, database, or Qdrant fail fast with actionable error context                                                                        |
| Extensible                 | Reranking, semantic chunking, graph enhancement, and model endpoints are isolated by configuration and service layers                              |
| Clear agent architecture   | Perception-Planning-Retrieval-EvidenceEvaluator-Generation separation; each stage independently observable and tunable                             |
| Cross-lingual robustness   | LLM translation query expansion + embedding similarity bridge + cross-lingual admission gate mitigates monolingual embedding alignment limitations |

## Data Model

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

| Table                                               | Purpose                                                                                                          |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `courses`                                           | Knowledge-base workspace                                                                                         |
| `documents` / `document_versions`                   | File metadata, versions, and parser artifact paths                                                               |
| `chunks`                                            | Parent/child text chunks, summaries, keywords, embedding text version, and evidence text                         |
| `concepts`                                          | Concepts, chapter references, evidence counts, communities, centrality, and graph rank                           |
| `concept_aliases`                                   | Concept aliases and normalized aliases                                                                           |
| `concept_relations`                                 | Sparse edges, relation types, evidence chunks, weights, semantic similarity, support count, and inference source |
| `quality_profiles`                                  | Domain quality profiles (versioned, stratified sampling, positive/negative examples, term baselines)             |
| `ingestion_batches` / `ingestion_jobs`              | Batch ingestion and single-file jobs                                                                             |
| `ingestion_logs` / `ingestion_compensation_logs`    | Event streams and cross-store compensation records                                                               |
| `qa_sessions` / `agent_runs` / `agent_trace_events` | QA sessions, agent runs, and observable traces                                                                   |
| `course_model_hyperparameters`                      | Course-level HPO hyperparameter records (versioned, auditable, fallback)                                         |
| `graph_extraction_runs` / `graph_extraction_chunk_tasks` | Adaptive graph extraction run state and per-chunk task tracking                                             |
| `graph_hpo_judge_samples` / `graph_hpo_objective_models` | HPO judge samples and surrogate objective model persistence                                                      |
| `entity_mentions` / `entity_merge_candidates`         | Entity mention records and LLM-verified merge candidates                                                         |
| `graph_relation_candidates`                           | Candidate relations from semantic sparse graph (not factual edges)                                               |
| `graph_community_summaries`                           | Community summary text and community-level aggregate metrics                                                     |

## Configuration

Copy the configuration template:

```powershell
Copy-Item .env.example .env
```

Common variables:

| Variable                                                                                                     | Description                                                                                       |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `API_HOST_PORT` / `WEB_HOST_PORT`                                                                            | Host ports                                                                                        |
| `DATABASE_URL`                                                                                               | PostgreSQL connection URL                                                                         |
| `ENABLE_DATABASE_FALLBACK`                                                                                   | Database fallback switch, default `false`                                                         |
| `QDRANT_URL` / `QDRANT_COLLECTION`                                                                           | Qdrant URL and collection name                                                                    |
| `REDIS_URL`                                                                                                  | Redis URL                                                                                         |
| `COURSE_NAME`                                                                                                | Default knowledge-base name                                                                       |
| `DATA_ROOT`                                                                                                  | Local data root                                                                                   |
| `OPENAI_API_KEY` / `CHAT_BASE_URL`                                                                           | OpenAI-compatible chat / graph extraction model endpoint                                          |
| `CHAT_RESOLVE_IP`                                                                                            | Target IP when chat model-domain resolution must be pinned                                        |
| `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL`                                                                   | OpenAI-compatible embedding model endpoint, independent from the chat endpoint                    |
| `EMBEDDING_RESOLVE_IP`                                                                                       | Target IP when embedding model-domain resolution must be pinned                                   |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` / `EMBEDDING_BATCH_SIZE`                                          | Embedding model, dimensions, and batch size                                                       |
| `CHAT_MODEL`                                                                                                 | Chat and graph extraction model                                                                   |
| `GRAPH_EXTRACTION_SOFT_START_BUDGET` / `GRAPH_EXTRACTION_CONCURRENCY` / `GRAPH_EXTRACTION_RESUME_BATCH_SIZE` | Adaptive graph extraction initial budget, concurrent model calls, and model-call chunk batch size |
| `ENABLE_MODEL_FALLBACK`                                                                                      | Model fallback switch, default `false`                                                            |
| `RERANKER_ENABLED` / `RERANKER_MODEL` / `RERANKER_MAX_LENGTH`                                                | Cross-Encoder reranker settings                                                                   |
| `SEMANTIC_CHUNKING_ENABLED` / `SEMANTIC_CHUNKING_MIN_LENGTH`                                                 | Semantic chunking switch and minimum text length                                                  |
| `RETRIEVAL_LAYER_ENABLED`                                                                                    | Retrieval layer switch, default `true`                                                            |
| `RETRIEVAL_CACHE_TTL_SECONDS`                                                                                | Redis retrieval cache TTL, default `300`                                                          |
| `ENABLE_AGENTIC_REFLECTION`                                                                                  | Agentic reflection and correction master switch, default `true`                                   |
| `ENABLE_POST_GENERATION_REFLECTION`                                                                          | Post-generation reflection switch (CitationVerifier/Reflection/AnswerCorrector), default `false`  |
| `CITATION_VERIFICATION_SAMPLE_MAX`                                                                           | Citation verification sample size per answer, default `3`                                         |
| `REFLECTION_MAX_RETRIES`                                                                                     | Max reflection-triggered correction retries, default `2`                                          |
| `MODEL_BRIDGE_ENABLED` / `MODEL_BRIDGE_PORT`                                                                 | Host model-bridge switch and port                                                                 |
| `ENABLE_AUTO_HPO`                                                                                            | Auto-run TPE hyperparameter optimization before graph rebuild, default `false`                    |
| `HPO_JUDGE_MAX_CANDIDATES` / `HPO_JUDGE_MAX_PAIRS` / `HPO_JUDGE_MIN_LABELS`                                  | HPO judge candidate count, pairwise comparison count, minimum valid labels                        |
| `HPO_JUDGE_MAX_TOKENS_PER_PAIR` / `HPO_JUDGE_CONCURRENCY`                                                    | Max tokens per judge pair and concurrency                                                         |
| `GRAPH_EXTRACTION_SOFT_START_BUDGET` / `GRAPH_EXTRACTION_RESUME_BATCH_SIZE`                                  | Adaptive graph extraction initial budget and per-batch model-call chunk count                     |

Docker Compose overrides infrastructure URLs inside the API container:

```text
DATABASE_URL=postgresql+psycopg://postgres:postgres@postgres:5432/course_kg
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379/0
```

Embedding and Chat models support independent endpoint configuration:

```text
CHAT_BASE_URL=https://api.openai.com/v1
EMBEDDING_BASE_URL=https://api.openai.com/v1
```

If your embedding provider differs from your chat provider (e.g., embedding served locally while chat uses a cloud API), simply fill in both endpoints separately. The system will not fallback embedding requests to the chat endpoint.

If the host can reach a model provider but container networking to that provider is unstable, enable the model bridge. The bridge forwards the real OpenAI-compatible endpoint only; it does not generate fake responses and is not a fallback path.

## Running

1. Configure `.env` with a real model endpoint:

```env
OPENAI_API_KEY=...
CHAT_BASE_URL=https://api.openai.com/v1
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-v4
CHAT_MODEL=qwen-plus
ENABLE_MODEL_FALLBACK=false
ENABLE_DATABASE_FALLBACK=false
```

2. Start the Docker stack:

```powershell
docker compose -f infra/docker-compose.yml up -d api web postgres redis qdrant
```

Windows users can also double-click `start-app.bat` to launch the backend, frontend, and infrastructure containers. This script **does not** force an image rebuild, making it suitable for daily quick starts.

If the application code or dependencies have changed and you need to rebuild the local images, run:

```powershell
docker compose -f infra/docker-compose.yml build api web
```

Or on Windows simply run `rebuild-images.bat`. To force a rebuild without cache, use `rebuild-images.bat -NoCache`.

3. Open the web app:

```text
http://127.0.0.1:3000
```

## Validation

Backend tests:

```powershell
docker exec course-kg-api python -m pytest tests
```

Frontend checks:

```powershell
npm run typecheck --workspace web
npm run lint --workspace web
npm run test --workspace web
```

Docker smoke:

```powershell
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api
```

Knowledge-base quality gate:

```powershell
docker exec course-kg-api python /app/scripts/quality_gate.py --course-name "Knowledge Base Name"
```

Evidence-first retrieval comparison:

```powershell
docker exec course-kg-api python /app/scripts/evaluate_evidence_first_retrieval.py --course-name "Knowledge Base Name"
```

Quality decision evaluation:

```powershell
docker exec course-kg-api python /app/scripts/evaluate_quality_decisions.py --course-name "Knowledge Base Name"
```

Reingest one knowledge base and clean stale derived data:

```powershell
docker exec course-kg-api python /app/scripts/reingest_all_courses.py --course-name "Knowledge Base Name" --cleanup-stale
```

Validation focus:

| Check                   | Expected                                                                                                                                                                                 |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Health                  | `/api/health` returns available service status                                                                                                                                           |
| Runtime configuration   | `/api/settings/runtime-check` has no blocking issue                                                                                                                                      |
| Model fallback          | `ENABLE_MODEL_FALLBACK=false`; model outages fail fast                                                                                                                                   |
| Database fallback       | `ENABLE_DATABASE_FALLBACK=false`; database outages fail fast                                                                                                                             |
| Vector health           | Qdrant vector count matches active chunks and no zero vectors exist                                                                                                                      |
| Retrieval quality       | Child recall, parent context, rerank, and citation fields are complete                                                                                                                   |
| Graph quality           | Node count meets the retention floor, edge growth is near-linear, and community, centrality, and weight fields are populated; graph remains stable after incremental updates             |
| Quality system          | quality_profile generated, chunk/concept/relation policy decisions observable, no mass discard false positives                                                                           |
| Layered retrieval       | Different query types hit the correct layer; Redis cache hit/miss behaves correctly                                                                                                      |
| Agentic loop            | Perception, RetrievalPlanner, EvidenceEvaluator nodes are observable in traces; post-generation Reflection is off by default; LLM errors are not silently swallowed when fallback is off |
| Cross-lingual retrieval | Mixed Chinese-English queries hit materials in the opposite language; DocumentGrader bridge gate is active                                                                               |
| Log observability       | Ingestion logs expose progress, retry, failure reason, and terminal event                                                                                                                |

## Core Innovations

SymboGraph's core innovations in the general GraphRAG direction can be summarized in seven points:

**1. Four-Tier Adaptive Quality Architecture**
Unlike traditional systems with single-threshold filtering, SymboGraph establishes a signal-policy-profile-judge four-tier quality system. Chunks are no longer limited to "keep/discard" binary fates; instead, they are routed to one of six downstream paths (`discard`, `summary_only`, `evidence_only`, `retrieval_candidate`, `graph_candidate`, `embed_only`). Concepts and relations undergo differentiated policy filtering as well. Domain quality profiles give each knowledge base an adaptive quality baseline rather than relying on global fixed thresholds.

**2. Concept Vector Centroidization and Dynamic Sparse Graph Construction**
Concept vectors are generated as centroids of their supporting chunk vectors, not by embedding the LLM-extracted concept name directly, fundamentally eliminating concept drift. The dynamic KNN + semantic threshold + mutual/inbound-quota sparse graph algorithm applies candidate send/receive limits based on evidence volume *m*`<sub>`i`</sub>` and chapter coverage *r*`<sub>`i`</sub>`, guaranteeing near-linear edge growth with node count and naturally suppressing the Hubness Problem.

**3. Evidence-first Agentic RAG**
The QA pipeline is not a simple "retrieve then generate" but a full Agent workflow: Perception → RetrievalPlanner → BaseRetrieval → EvidenceAnchorSelector → EvidenceChainPlanner → ControlledGraphEnhancer → EvidenceAssembler → DocumentGrader → EvidenceEvaluator → Generation. The pre-generation `EvidenceEvaluator` gives the system the ability to "know what it doesn't know", intercepting low-quality retrievals before generation.

**4. Triple-Mechanism Cross-lingual Robust Retrieval**
LLM explicit translation expansion produces bilingual sub-queries, embedding similarity bridges language barriers, and the DocumentGrader *s*`<sub>`embedding`</sub>` ≥ 0.45 cross-lingual bridge gate exempts lexical false kills—three mechanisms together build a robust retrieval system that does not rely on the alignment quality of a single multilingual embedding model.

**5. Small-to-Big Context Assembly with Parent-Child Decoupling**
At retrieval time, only the finest-grained units (child chunks) enter dense/BM25/recall/rerank, preventing parents and children from competing in the candidate pool; at generation time, full parent context is assembled via `parent_chunk_id`. This completely decouples the "recall unit" from the "generation unit", achieving both precision and contextual completeness.

**6. Graph-Theoretic Algorithms Hedging LLM Stochasticity**
Louvain community detection, spectral clustering, connected-component ablation, multidimensional centrality (degree / PageRank / betweenness / closeness), and Dijkstra hidden-link discovery together form a systematic hedge against LLM extraction noise. The graph is not a passive container for LLM output but a sparse knowledge skeleton rigorously cleaned by graph theory.

**7. TPE Automatic Hyperparameter Optimization and Incremental Updates**
Traditional GraphRAG systems use fixed hyperparameters (thresholds, weights) for all knowledge bases, unable to adapt to different domains' semantic density and concept distribution. SymboGraph introduces Optuna TPE auto-optimization: based on probe chunk pre-extraction, LLM pairwise judging, and a surrogate objective function, it searches for the optimal Dijkstra threshold, relation confidence, and graph algorithm weights over 30 trials. The best parameters are versioned and persisted, with automatic fallback on failure. Incremental graph updates are also supported, recomputing only subgraphs tied to changed documents and avoiding the redundant cost of full rebuilds.

## Version Control Rules

Excluded from Git:

- `.env`, local secrets, Authorization headers, and provider responses.
- `data/`, `output/`, `models/`, and `comparative_experiment/` runtime data.
- `node_modules/`, `.next/`, `dist/`, `build/`, coverage, and Playwright reports.
- `.db`, `.sqlite*`, `__pycache__/`, `*.pyc`, `*.tsbuildinfo`, logs, and temporary files.

Tracked in Git:

- `apps/api`, `apps/web`, `packages/shared`, `scripts`, and `infra`.
- README files, `.env.example`, Docker configuration, tests, schemas, and shared type contracts.

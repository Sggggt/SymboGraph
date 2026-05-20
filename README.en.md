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
| Graph                  | LLM candidates, chunk-vector semantic graph, graph algorithms for sparse construction, deduplication, communities, centrality, and hidden links; supports incremental and full rebuild |
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
        API --> QUALITY["Quality System<br/>signal extraction -> policy decision -> domain profile -> LLM judge"]
    end

    subgraph STORE["Storage And Runtime"]
        PG[("PostgreSQL<br/>metadata, chunks, sparse graph, audit records, quality profiles")]
        QD[("Qdrant<br/>chunk vectors and similarity recall")]
        RD[("Redis<br/>runtime cache, embedding / retrieval cache, judge cache, task coordination")]
        FS["data/<br/>knowledge-base files, parser artifacts, local persistence"]
    end

    subgraph MODEL["Model API"]
        EMB["OpenAI-compatible Embedding API<br/>independent endpoint config"]
        CHAT["OpenAI-compatible Chat API<br/>independent endpoint config"]
    end

    API --> PG
    API --> QD
    API --> RD
    API --> FS
    API --> EMB
    API --> CHAT
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
    API->>API: Create parent/child chunks and context-enriched text
    API->>API: Quality signal extraction and policy routing (discard/summary/retrieval/graph)
    API->>Model: Generate summaries, keywords, and embeddings
    API->>Vector: Upsert active chunk vectors
    API->>DB: Activate document version and chunks
    API->>Model: Extract entity/relation candidates
    API->>API: Quality filtering (concept accept/reject, relation accept/candidate)
    API->>API: Run sparse graph construction, communities, centrality, and inference
    API->>DB: Store concepts, relations, evidence, and graph-algorithm fields
    API->>DB: Incremental update: recompute only subgraphs tied to changed documents
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

SymboGraph builds graphs with an evidence-first policy: the LLM produces candidate entities and explicit relations, while chunk vectors and graph algorithms decide the final structure. PostgreSQL is the source of truth for the sparse graph; Qdrant provides chunk vectors and similarity signals. All candidates are filtered by the quality system's concept and relation policies before persistence.

```mermaid
flowchart LR
    CHUNK["Active child chunks<br/>(passed ChunkQualityPolicy)"] --> VEC["Qdrant chunk vectors"]
    CHUNK --> LLM["LLM entity/relation candidates"]
    LLM --> QUAL["ConceptQualityPolicy<br/>RelationQualityPolicy<br/>quality filter"]
    QUAL --> ENTITY["Concept merge<br/>alias and duplicate reduction"]
    VEC --> CENTROID["Concept vector centroid"]
    ENTITY --> SPARSE["Dynamic KNN + semantic threshold<br/>mutual / inbound quota sparse graph"]
    CENTROID --> SPARSE
    SPARSE --> CC["Connected components<br/>noise ablation"]
    CC --> COMM["Louvain + spectral communities"]
    COMM --> CENTRAL["Centrality ranking"]
    CENTRAL --> DIJK["Dijkstra hidden links"]
    DIJK --> COMPLETE["Traversal + prompt relation completion"]
    COMPLETE --> PG["PostgreSQL graph tables"]
    PG --> UI["Community-aware graph UI"]
```

### 1. Entities And Evidence

Each concept stores a canonical name, aliases, chapter references, importance, and evidence chunk count. Concept vectors are not generated from names. They are centroids of supporting chunk vectors:

$$
\mathbf{v}_e = \frac{1}{|C_e|}\sum_{c \in C_e}\mathbf{v}_c
$$

*C*`<sub>`e`</sub>` is the set of active child chunks supporting entity *e*. The centroid is normalized before semantic graph construction.

> **Design Intent (Why we do this)**: Traditional GraphRAG directly embeds the extracted concept name, which biases the vector space toward the LLM's generic pre-training data. Calculating the centroid of all supporting underlying child chunk vectors ensures the graph remains perfectly faithful to the specific local context of the knowledge base, eliminating concept drift.

### 2. Dynamic KNN + Semantic Threshold + Mutual/Inbound-Quota Sparse Graph

This is not standard Radius-NN graph construction, where neighbors are admitted solely by a fixed radius or distance threshold. The current implementation first generates candidates with dynamic KNN and a semantic similarity threshold, then keeps mutual nearest neighbors or candidates accepted by the reverse inbound quota.

Each concept dynamically chooses outgoing candidates from its evidence volume:

$$
K_i = \mathrm{clamp}\bigl(4 + \lfloor \log_2(1 + m_i) \rfloor,\, 4,\, 12\bigr)
$$

Each concept dynamically limits accepted reverse inbound candidates from chapter coverage:

$$
B_i = \mathrm{clamp}\bigl(2 + \lfloor \log_2(1 + r_i) \rfloor,\, 2,\, 8\bigr)
$$

*m*`<sub>`i`</sub>` is evidence chunk count and *r*`<sub>`i`</sub>` is chapter reference count. The system keeps mutual nearest neighbors, candidates accepted by the reverse inbound quota *B*`<sub>`i`</sub>`, and high-confidence explicit LLM relations, keeping edge count close to linear in node count.

> **Design Intent (Why we do this)**: If high-frequency words (e.g., "algorithm", "data") accept edges without limits, the graph quickly collapses into a useless giant hub (the Hubness Problem). A dynamic sparse algorithm based on evidence volume, chapter coverage, semantic thresholding, and reverse inbound quotas mathematically squeezes out low-quality edges, guaranteeing the graph remains clear, sparse, and focused.

### 3. Edge Weights And Graph Algorithms

Edge weight combines LLM confidence, semantic similarity, evidence support, and structural consistency:

$$
w_{ij}=0.45\,c_{ij}^{\mathrm{llm}}+0.30\,s_{ij}^{\mathrm{sem}}+0.15\,s_{ij}^{\mathrm{evidence}}+0.10\,s_{ij}^{\mathrm{structure}}
$$

When no explicit LLM relation exists, *c*`<sub>`ij`</sub><sup>`llm`</sup>`=0. The final *w*`<sub>`ij`</sub>` is clipped to [0,1]. The graph stage runs:

- Connected-component ablation: removes isolated, low-evidence, low-importance noise while preserving enough knowledge-base nodes.
- Louvain community detection: primary community labels and frontend color groups.
- Spectral clustering: secondary partitions for large components and large communities.
- Centrality: degree, weighted degree, PageRank, betweenness, closeness, and a combined `centrality_score`.
- Graph simplification: keeps central nodes, community representatives, bridge edges, and high-evidence concepts.

> **Design Intent (Why we do this)**: LLM-extracted graphs are often noisy and fragmented. Introducing classic graph algorithms like connected components, Louvain community detection, and centrality is the most effective way to hedge against LLM hallucinations. This multi-dimensional graph ablation retains only high-value core structures, solving the notorious "hairball" visualization problem in large-scale graphs.

### 4. Hidden Links And Relation Completion

Dijkstra searches 2-3 hop hidden relations on a non-negative cost graph:

$$
\mathrm{cost}_{ij}=\frac{1}{0.05+w_{ij}}
$$

If endpoint semantic similarity is high and path cost is low, the system writes a `relates_to` edge with `relation_source="dijkstra_inferred"` and uses the path score to repair weak existing weights. The system then extracts evidence snippets from two-hop neighborhoods around high-centrality nodes and asks the LLM to complete only evidence-supported relations.

The frontend colors graph nodes by Louvain community, sizes nodes by centrality and graph rank, and renders inferred edges as dashed lines. Users can filter communities and open key entity details quickly.

> **Design Intent (Why we do this)**: True knowledge often spans chapters (e.g., A belongs to B, B contains C, so A relates to C, even if unstated). Using Dijkstra's algorithm to efficiently find structural holes and then using the LLM to verify these specific 2-hop evidence snippets enables automated, highly precise ontology expansion that surpasses traditional rule-based extraction.

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
    Chunk ||--o{ Chunk : children
    Course ||--o{ Concept : has
    Concept ||--o{ ConceptAlias : aliases
    Concept ||--o{ ConceptRelation : source
    Concept ||--o{ ConceptRelation : target
    Course ||--o{ IngestionBatch : batches
    IngestionBatch ||--o{ IngestionJob : jobs
    Course ||--o{ QualityProfile : profiles
    Course ||--o{ QASession : sessions
    QASession ||--o{ AgentRun : runs
    AgentRun ||--o{ AgentTraceEvent : traces
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

SymboGraph's core innovations in the general GraphRAG direction can be summarized in six points:

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

## Version Control Rules

Excluded from Git:

- `.env`, local secrets, Authorization headers, and provider responses.
- `data/`, `output/`, `models/`, and `comparative_experiment/` runtime data.
- `node_modules/`, `.next/`, `dist/`, `build/`, coverage, and Playwright reports.
- `.db`, `.sqlite*`, `__pycache__/`, `*.pyc`, `*.tsbuildinfo`, logs, and temporary files.

Tracked in Git:

- `apps/api`, `apps/web`, `packages/shared`, `scripts`, and `infra`.
- README files, `.env.example`, Docker configuration, tests, schemas, and shared type contracts.

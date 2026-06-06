# SymboGraph Backend API

A high-performance, asynchronous FastAPI backend service orchestrating the SymboGraph local course knowledge graph infrastructure. It manages data ingestion, semantic parsing, hybrid vector/lexical retrieval, graph extraction, and automated hyperparameter optimization (HPO).

---

## 🛠️ Architecture & Services

The backend enforces strict ACID constraints and coordinates three persistent stores:

1. **PostgreSQL (v16)**: The single source of truth for structural course metadata, strategy profiles, ingestion batches, active document chunks, concept nodes, and relationship matrices. Enforces transaction atomicity and relational integrity.
2. **Qdrant (v1.17)**: High-speed vector store storing dense contextualized embeddings (1024-dimension `text-embedding-v4`) for semantic hybrid search.
3. **Redis (v7)**: Shared runtime cache, distributed locks (`_RedisDistributedLock`), and message broker facilitating high-throughput operations.

---

## 🚀 Core Pipelines

### 1. Ingestion, Parsing & Chunking
- **Multiformat Parsers**: Custom parsers for `.pdf`, `.ipynb`, `.md`, `.docx`, `.pptx`, `.html`.
- **Adaptive Semantic Chunking**: Preserves structural hierarchy, LaTeX formulas, and table coordinates. Dynamically balances chunk boundaries based on semantic coherence (F1 score).
- **Contextualized Embeddings**: Enhances dense vectors by prefixing parent-chapter context to prevent loss of local semantics in downstream RAG.
- **Chunk Version Control**: Tracks committed knowledge-base versions with `courses.current_chunk_version` and `chunks.chunk_version`; selected-file parses reuse the current version, while full reparse advances only after at least one file succeeds.
- **Phase-aware Cancellation**: Parse-phase cancellation rolls back parse writes, while graph-phase cancellation restores the pre-graph snapshot without discarding committed parse results.

### 2. Evidence-First Retrieval
- **Hybrid Search**: Combines Qdrant dense vector cosine similarity with lexical full-text matching, resolved using Reciprocal Rank Fusion (RRF).
- **Dijkstra Concept Pathing**: Traces semantic connection paths between multi-hop query concepts using graph algorithm weights.
- **Citation Grounding**: Every answer is strictly back-referenced to source document slices to ensure evidence-first accuracy and zero hallucination.

### 3. Adaptive Graph Extraction & Auto HPO
- **Adaptive Extraction Plan**: Automatically budgets API token consumption by selecting critical chunks (`adaptive_best_first` strategy) for LLM relationship extraction.
- **Bounded Model Calls**: Applies `GRAPH_EXTRACTION_MAX_MODEL_CALLS_PER_RUN`, `MODEL_REQUEST_CONCURRENCY`, and `MODEL_REQUEST_TIMEOUT_SECONDS` to avoid unbounded graph extraction fan-out.
- **Graph Enrichment**: Calculates PageRank, betweenness centrality, and Louvain community partitions to summarize corpus topology.
- **TPE Auto HPO**: Runs an Optuna-powered Tree-structured Parzen Estimator (TPE) over a surrogate LLM-pairwise-judge learned objective to optimize Dijkstra distance cutoffs and relation confidence weights.

### 4. Knowledge-Base Strategy Profiles
- **Default Compatibility Layer**: New and legacy courses bind to the built-in course Profile by default, preserving existing RAG, graph extraction, retrieval, and agent behavior.
- **Persistent Profiles**: Profiles are stored in PostgreSQL table `strategy_profiles` and linked from `courses.active_profile_id`; Profile state is not stored in `.env`.
- **Profile-Aware Boundaries**: Prompt builders, entity/relation normalization, chunk labels, embedding metadata, query classification, agent routes, graph judge hints, and quality policy hints read the active Profile at pipeline boundaries.
- **Non-Destructive Switching**: Profile changes only affect new tasks. Existing chunks, graph objects, vectors, and sessions remain untouched; graph extraction runs record `strategy_profile_id` and `strategy_profile_hash`.
- **Deletion Semantics**: The built-in default Profile cannot be deleted. Deleting any custom Profile soft-deletes it and automatically rebinds any affected courses to the default Profile in one transaction.
- **Profile Assistant**: `/settings/profile-assistant/stream` streams natural-language guidance plus a validated Profile JSON draft and caches assistant session state in Redis.

---

## 📂 Codebase Directory Layout

```
apps/api/
├── app/
│   ├── core/           # App settings, DB connections, Redis distributed lock
│   ├── models/         # SQLAlchemy ORM declarative database models
│   ├── schemas/        # Strong contract validation using Pydantic v2
│   ├── services/       # Core business logic: concept_graph, ingestion, strategy profiles, HPO engine
│   ├── main.py         # FastAPI app registration and API router entry
│   └── api.py          # Endpoints for ingestion, RAG chat, concept card catalog
├── tests/              # Extensive pytest suite covering logic, DB integrations
├── pyproject.toml      # Modern uv package manager configuration
└── Dockerfile          # Production-grade Docker container configuration
```

---

## 💻 Running & Testing

### Running within Docker (Recommended)
By default, all dependent services run via Docker Compose:
```bash
# Build and start the stack
docker compose -f infra/docker-compose.yml up --build -d
```
The API is exposed at `http://127.0.0.1:8000/api`.

### Running Integration & Unit Tests
Integrations with PostgreSQL, Redis, and Qdrant must be validated inside the container environment to ensure correct schema and connection behaviors:
```bash
# Run the complete test suite inside the container
docker exec course-kg-api python -m pytest tests

# Run Profile-specific regression tests
docker exec course-kg-api python -m pytest tests/test_strategy_profiles.py
```

### Running Locally for Active Debugging
Ensure your local PostgreSQL, Redis, and Qdrant instances are forwarded or running on `localhost`:
```bash
# Sync local virtualenv using uv
uv sync

# Launch FastAPI reload server
uv run uvicorn app.main:app --reload --port 8000
```

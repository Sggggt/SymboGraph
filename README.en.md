[**English**](./README.en.md) | [中文](./README.md)

<p align="center">
  <img src="./assets/diagraph-logo.svg" alt="SymboGraph logo" width="132" height="132">
</p>

<h1 align="center">SymboGraph</h1>

SymboGraph is a local, general-purpose intelligent knowledge base. The active path treats traceable evidence as the source of truth: files become `EvidenceAtom` records, those atoms form an `EvidenceGraph` with observed relations only, and the system derives graph-grounded chunk candidates, `QualityDecision` records, `ActiveChunk` records, retrieval traces, citation verification, answer sessions, and policy rewards.

> **Migration note**: product code and the frontend context have moved to `KnowledgeBase` / `partition` / evidence-first semantics. The default PostgreSQL database name has moved to `symbograph`. Docker Compose service names, container names, image names, and queue names still use historical `course-kg-*` infrastructure names. Those names will be handled in a separate infrastructure change and do not define product semantics.

## Table of Contents

- [Quick Overview](#quick-overview)
- [Technology Stack](#technology-stack)
- [Core Capabilities](#core-capabilities)
- [Main Path](#main-path)
- [Run](#run)
- [Environment Parameters](#environment-parameters)
- [Repository Layout](#repository-layout)
- [Tests and Acceptance](#tests-and-acceptance)
- [Product Boundaries](#product-boundaries)

## Quick Overview

| Dimension | Current design |
| --- | --- |
| Product role | Local general-purpose knowledge base for evidence retrieval, grounded QA, citation verification, and policy feedback |
| Source of truth | PostgreSQL records for document versions, source spans, evidence atoms, active chunks, retrieval traces, and answer sessions |
| Graph substrate | Evidence graph with observed edges only: adjacency, containment, layout continuity, citation dependency, semantic similarity, modality links, discourse shifts |
| Chunking | Evidence atoms enter the graph first; graph features, community boundaries, quality gates, and policy state decide active chunks |
| Retrieval QA | Dense/lexical evidence recall, community/graph expansion, rerank, context assembly, citation verification |
| Policy | `policy_states` plus contextual bandit manage operating points; HPO is only an experiment/offline replay baseline |
| Default runtime | Docker Compose starts API, worker, web, PostgreSQL, Redis, and Qdrant |

## Technology Stack

| Layer | Technology | Responsibility |
| --- | --- | --- |
| API | FastAPI, SQLAlchemy, Alembic, Pydantic | Ingestion orchestration, evidence graph, chunk quality, retrieval QA, runtime settings |
| Worker | Celery, Redis broker | File parsing, long tasks, graph construction, vector writes, recoverable batches |
| Web | Next.js 16.2.4, React 19, TypeScript, TanStack Query | Knowledge base UI, upload, evidence graph, retrieval, QA, settings |
| Metadata | PostgreSQL | Lifecycle state, evidence facts, policy state, traces, citation verification |
| Vector | Qdrant | Active chunk vectors and derived retrieval indexes |
| Runtime | Redis | Cache, runtime settings broadcast, worker coordination |
| Models | OpenAI-compatible chat / embedding endpoints | Grounded answer generation, measurement prompts, embeddings |

## Core Capabilities

| Capability | Description |
| --- | --- |
| Evidence-grade ingestion | Conservatively parses headings, paragraphs, list items, table blocks, code blocks, formulas, captions, page blocks, and similar atoms |
| Auditable graph | `EvidenceEdge` stores observed relations only and does not treat LLM-generated durable ontology as truth |
| Graph-grounded chunking | `ChunkCandidate` stores atom ids, span union, graph features, cost estimate, and generator version |
| Quality decisions | `QualityDecision` carries gate, reward, and feedback with hard constraints and diagnostics |
| Active chunks | Each active chunk stores source span union, graph state hash, quality decision, policy state, and community ids |
| Retrieval traces | `RetrievalTrace` records candidates, expansion, rerank, cache, and risk audit details |
| Citation verification | Answer citations must resolve to an `active_chunk_id`, `evidence_atom_id`, or source span |
| Policy feedback | `RewardEvent` traces back to retrieval, QA, chunks, and policy state for contextual bandit updates |
| Hot settings | `.env` updates publish Redis runtime settings versions; API and workers refresh at task boundaries |

## Main Path

```text
file parsing
-> evidence atoms
-> evidence graph
-> graph-grounded chunk candidates
-> quality gate
-> active chunks
-> retrieval / QA / citation verification
-> reward events
-> contextual bandit policy update
```

```mermaid
flowchart LR
    F["File parsing"] --> A["EvidenceAtom"]
    A --> G["EvidenceGraph"]
    G --> C["ChunkCandidate"]
    C --> Q["QualityDecision"]
    Q --> X["ActiveChunk"]
    X --> R["RetrievalTrace"]
    R --> S["AnswerSession / CitationVerification"]
    S --> E["RewardEvent"]
    E --> P["PolicyState"]
    P --> C
```

Algorithm details, constraint formulas, policy operating points, and quality decision fields live in [docs/technical-spec.md](./docs/technical-spec.md). The README keeps only engineering entry points and runtime boundaries.

## Run

Default runtime is Docker Compose:

```powershell
docker compose -f infra/docker-compose.yml up --build
```

Default services:

| Service | Default address | Description |
| --- | --- | --- |
| Web | `http://127.0.0.1:3000` | Next.js UI |
| API | `http://127.0.0.1:8000/api` | FastAPI |
| PostgreSQL | `127.0.0.1:5432` | Metadata source of truth |
| Redis | `127.0.0.1:6379` | Cache, broker, broadcast |
| Qdrant | `http://127.0.0.1:6333` | Active chunk vectors |

Compose services are still named `course-kg-api`, `course-kg-worker`, `course-kg-web`, `course-kg-postgres`, `course-kg-redis`, and `course-kg-qdrant`. Treat these as historical infrastructure names, not product terminology.

## Environment Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `APP_NAME` | `KnowledgeBase Knowledge Base API` | API display name |
| `APP_ENV` | `development` | Runtime environment label |
| `APP_PORT` | `8000` | API process listen port |
| `API_IMAGE` | `course-kg-api:local` | API/worker Docker image name; historical infrastructure name, not product terminology |
| `WEB_IMAGE` | `course-kg-web:local` | Web Docker image name; historical infrastructure name, not product terminology |
| `INGESTION_TASK_QUEUE` | `course-kg-main-ingestion` | Celery ingestion queue name; historical infrastructure name, not product terminology |
| `API_HOST_PORT` | `8000` | API host port mapping |
| `WEB_HOST_PORT` | `3000` | Web host port mapping |
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/symbograph` | PostgreSQL connection string; default database name is `symbograph` |
| `ENABLE_DATABASE_FALLBACK` | `false` | Database fallback is disabled on the active path |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `knowledge_chunks` | Active chunk vector collection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis endpoint |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Frontend origins allowed by the API |
| `API_KEYS` | empty | Optional comma-separated API key list |
| `KNOWLEDGE_BASE_NAME` | `Sample KnowledgeBase` | Default knowledge base name |
| `DATA_ROOT` | `./data` | Local data root |
| `STORAGE_ROOT` | `./data/Sample KnowledgeBase/storage` (commented in the example) | Optional source-file storage root override |
| `INGESTION_ROOT` | `./data/Sample KnowledgeBase/ingestion` (commented in the example) | Optional ingestion staging root override |
| `MODEL_BRIDGE_ENABLED` | `true` | Whether to enable the local model bridge |
| `MODEL_BRIDGE_PORT` | `8765` | Local model bridge port |
| `OPENAI_API_KEY` | empty | OpenAI-compatible API key |
| `CHAT_BASE_URL` | empty | OpenAI-compatible chat endpoint |
| `CHAT_RESOLVE_IP` | empty | Optional DNS resolve override IP for the chat endpoint |
| `CHAT_MODEL` | `qwen-plus` | Chat model name |
| `EMBEDDING_BASE_URL` | empty | Embedding endpoint; never falls back to the chat endpoint |
| `EMBEDDING_RESOLVE_IP` | empty | Optional DNS resolve override IP for the embedding endpoint |
| `EMBEDDING_API_KEY` | empty | Embedding endpoint API key |
| `EMBEDDING_MODEL` | `text-embedding-v4` | Embedding model name |
| `EMBEDDING_BATCH_SIZE` | `10` | Embedding batch size |
| `EMBEDDING_DIMENSIONS` | `1024` | Embedding dimensions |
| `WORKER_CONCURRENCY` | `3` | Worker concurrency |
| `INGESTION_FILE_CONCURRENCY` | `3` | File ingestion concurrency limit |
| `MODEL_REQUEST_CONCURRENCY` | `3` | Model request concurrency limit |
| `MODEL_REQUEST_TIMEOUT_SECONDS` | `240` | Model request timeout in seconds |
| `CHUNK_TOKEN_BUDGET` | `2400` | Active chunk token budget |
| `ENABLE_GRAPH_COMMUNITY_SUMMARIES` | `true` | Whether to generate community summary views |
| `SIGNAL_EXTRACTION_MAX_MODEL_BATCHES` | `4` | Signal extraction model batch limit |
| `SIGNAL_EXTRACTION_MAX_CANDIDATES_PER_BATCH` | `40` | Signal candidates per batch |
| `SIGNAL_EXTRACTION_MAX_TOKENS_PER_BATCH` | `6000` | Signal measurement token budget per batch |
| `SIGNAL_CANDIDATE_KEEP_THRESHOLD` | `0.62` | Signal candidate keep threshold |
| `COMMUNITY_LOUVAIN_RESOLUTION` | `1.0` | Louvain community resolution |
| `COMMUNITY_MIN_MODULARITY_WARN` | `0.18` | Modularity warning threshold |
| `GRAPH_OVERVIEW_MAX_NODES` | `260` | Graph overview node limit |
| `GRAPH_OVERVIEW_MAX_EDGES` | `800` | Graph overview edge limit |
| `ENABLE_MODEL_FALLBACK` | `false` | Model fallback is disabled on the active path |
| `RERANKER_ENABLED` | `false` | Whether to enable reranker |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `RERANKER_MAX_LENGTH` | `512` | Reranker maximum input length |
| `RERANKER_DEVICE` | `cpu` | Reranker runtime device |
| `HF_HUB_OFFLINE` | `1` | HuggingFace Hub offline mode |
| `SEMANTIC_CHUNKING_ENABLED` | `false` | Whether to enable semantic chunk candidates |
| `SEMANTIC_CHUNKING_MIN_LENGTH` | `2000` | Minimum text length for semantic chunking |
| `RETRIEVAL_LAYER_ENABLED` | `true` | Whether to enable layered retrieval |
| `RETRIEVAL_CACHE_TTL_SECONDS` | `120` | Retrieval cache TTL in seconds |
| `ENABLE_AGENTIC_REFLECTION` | `true` | Whether to enable agent reflection |
| `ENABLE_POST_GENERATION_REFLECTION` | `false` | Whether to enable post-generation reflection |
| `CITATION_VERIFICATION_SAMPLE_MAX` | `3` | Citation verification sample limit |
| `REFLECTION_MAX_RETRIES` | `2` | Reflection retry limit |

The full parameter list is the root `.env.example`; `apps/api/.env.example` is the API local-runtime subset. Use `SIGNAL_EXTRACTION_*` and `SIGNAL_CANDIDATE_KEEP_THRESHOLD` in `.env`. The old `CONCEPT_PROJECTION_*` / `CONCEPT_KEEP_POSTERIOR_THRESHOLD` names are deprecated and are no longer active configuration entries.

## Repository Layout

```text
apps/api        FastAPI, SQLAlchemy, Alembic, evidence graph, retrieval, QA, runtime settings
apps/web        Next.js 16.2.4, React 19, local knowledge base UI
apps/worker     Celery worker and watcher, reusing apps/api service logic
packages/shared Shared TypeScript contracts for frontend and backend-facing API payloads
infra           Default Docker Compose runtime stack
scripts         Smoke, quality gate, re-embedding, reparse, policy evaluation, maintenance scripts
docs            Architecture plans, technical specification, todo notes
local_light_tests Ignored lightweight real-corpus acceptance scripts
output          Test reports, screenshots, benchmark logs, smoke output
data            Local runtime data, never committed
```

`docs/todo.md` is for follow-up engineering tasks. `local_light_tests/` is for temporary real-corpus sampling and does not replace formal regression tests.

## Tests and Acceptance

Common checks:

```powershell
python -m py_compile apps/api/app/core/config.py
python -m pytest tests
npm run typecheck --workspace web
npm run lint --workspace web
npm run test --workspace web
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api
```

Backend unit tests should run from `apps/api`. Changes touching PostgreSQL, Qdrant, Redis, model endpoints, ingestion, retrieval, or QA should be accepted inside the Docker stack and should write reports to `output/`.

## Product Boundaries

- SymboGraph is a general-purpose local knowledge base; domain terms such as course, chapter, homework, or exam are not system truths.
- Evidence-first is mandatory. Retrieval, QA, quality decisions, community summaries, and rewards must trace back to document versions, source spans, evidence atoms, or active chunks.
- The LLM is a measurement component and grounded answer generator, not the default ontology builder.
- Community summaries are derived views. They do not replace citations and are not a source of truth.
- Fallback is disabled by default. Critical dependency failures should fail fast with actionable context.
- PostgreSQL is the durable source of truth. Qdrant and Redis are derived/runtime stores and must be repairable from persistent records.

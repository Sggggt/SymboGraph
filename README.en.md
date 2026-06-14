**English** | [中文](./README.md)

<p align="center">
  <img src="./assets/diagraph-logo.svg" alt="SymboGraph logo" width="132" height="132">
</p>

<h1 align="center">SymboGraph</h1>

## Overview

SymboGraph is a local, general-purpose intelligent knowledge base. It uses Four-Layer Context Graph RAG: fixed token chunks provide stable index and citation addresses, the Chunk Structure Graph preserves the source map, the Chunk Relation Graph stores reproducible low-level relations, Mid/Coarse Concept Graphs provide grounded concept navigation, and QA answers are generated from context packages with citation verification.

## Repository Layout

| Path | Responsibility |
| --- | --- |
| `apps/api` | FastAPI backend for ingestion, parsing, chunks, four-layer graphs, retrieval, QA, settings, and maintenance. |
| `apps/web` | Next.js 16.2.4 frontend for the knowledge base UI, graphs, retrieval traces, context packages, QA, and settings. |
| `apps/worker` | Celery worker and file watcher for long-running jobs, reusing API service logic. |
| `packages/shared` | Shared frontend/backend types and contracts. |
| `infra` | Default Docker Compose runtime. |
| `scripts` | Rebuild, reconciliation, diagnostics, retrieval evaluation, quality checks, and Docker smoke. |
| `docs` | Technical white paper, engineering checklist, and acceptance material. |
| `output` | Generated diagnostics, benchmarks, smoke output, screenshots, and acceptance reports. Not committed. |

## Product Positioning

SymboGraph is built for local document libraries, course material, technical docs, and research collections that need strict citation, traceable graph retrieval, and inspectable context packaging. PostgreSQL is the source of truth; Qdrant, BM25, and Redis are derived state. Answer facts come only from context packages and raw chunk citation spans.

## Stack

| Area | Technology |
| --- | --- |
| Backend | Python 3.13, FastAPI, SQLAlchemy, Alembic, Pydantic |
| Storage | PostgreSQL 16, Qdrant 1.17.1, Redis 7 |
| Jobs | Celery, Redis broker |
| Retrieval | Dense embedding, BM25, chunk relation graph, RQ-KMeans, layered retrieval |
| Models | OpenAI-compatible chat and embedding endpoints |
| Rerank | Optional Cross-Encoder reranker |
| Frontend | Next.js 16.2.4, React 19.2.4, TypeScript, TanStack Query, Tailwind CSS, ECharts |
| Operations | Docker Compose, Python maintenance scripts, pytest, Vitest, ESLint |

## Pipeline

```text
source files
-> parser and layout extractor
-> fixed token chunks
-> chunk structure graph
-> contextual embedding and BM25
-> chunk relation graph and fine clusters
-> mid concept graph
-> coarse concept graph
-> conversation state and query intent
-> Layered P&E Agent typed strategy
-> layered graph retrieval
-> context package
-> grounded answer and citation verification
-> reward and policy update
```

## Environment Configuration

Create `.env` from the template:

```powershell
Copy-Item .env.example .env
```

Configure chat and embedding endpoints:

```env
OPENAI_API_KEY=...
CHAT_BASE_URL=https://your-chat-endpoint/v1
CHAT_MODEL=your-chat-model

EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=https://your-embedding-endpoint/v1
EMBEDDING_MODEL=your-embedding-model
EMBEDDING_DIMENSIONS=1024
```

Keep fallback disabled for the active runtime:

```env
ENABLE_MODEL_FALLBACK=false
ENABLE_DATABASE_FALLBACK=false
```

Cross-Encoder reranking is optional and disabled by default:

```env
RERANKER_ENABLED=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_MAX_LENGTH=512
RERANKER_DEVICE=cpu
HF_HUB_OFFLINE=1
```

The Docker image installs the API `[rerank]` extra. The Hugging Face cache is mounted at `data/models/huggingface`; pass `PRELOAD_RERANK_MODEL=true` during image build to bake the model into the image.

## Quick Start

```powershell
docker compose -f infra/docker-compose.yml up -d --build
```

Open:

```text
Web: http://127.0.0.1:3000
API: http://127.0.0.1:8000/api
Health: http://127.0.0.1:8000/api/health
```

## Parameters

| Category | Variables |
| --- | --- |
| App | `APP_NAME`, `APP_ENV`, `APP_PORT`, `API_IMAGE`, `WEB_IMAGE` |
| Ports | `API_HOST_PORT`, `WEB_HOST_PORT` |
| Task queue | `INGESTION_TASK_QUEUE` |
| Infrastructure | `DATABASE_URL`, `QDRANT_URL`, `QDRANT_COLLECTION`, `REDIS_URL`, `CORS_ORIGINS`, `API_KEYS` |
| Data roots | `KNOWLEDGE_BASE_NAME`, `DATA_ROOT`, `STORAGE_ROOT`, `INGESTION_ROOT` |
| Model bridge | `MODEL_BRIDGE_ENABLED`, `MODEL_BRIDGE_PORT` |
| Chat | `OPENAI_API_KEY`, `CHAT_BASE_URL`, `CHAT_RESOLVE_IP`, `CHAT_MODEL` |
| Embedding | `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_RESOLVE_IP`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `EMBEDDING_BATCH_SIZE` |
| Concurrency | `WORKER_CONCURRENCY`, `WORKER_MAX_TASKS_PER_CHILD`, `MODEL_REQUEST_CONCURRENCY`, `MODEL_REQUEST_TIMEOUT_SECONDS` |
| Resource guards | `INGESTION_MEMORY_SOFT_LIMIT_RATIO`, `INGESTION_MEMORY_HARD_LIMIT_RATIO`, `INGESTION_MEMORY_CRITICAL_LIMIT_RATIO` |
| Chunks and context | `FIXED_CHUNK_SIZE_TOKENS`, `FIXED_CHUNK_OVERLAP_TOKENS`, `CONTEXT_PACKAGE_TOKEN_BUDGET` |
| Mid concepts | `MID_CONCEPT_EXTRACTION_MAX_MODEL_BATCHES`, `MID_CONCEPT_EXTRACTION_MAX_CANDIDATES_PER_BATCH`, `MID_CONCEPT_EXTRACTION_MAX_TOKENS_PER_BATCH`, `MID_CONCEPT_CANDIDATE_KEEP_THRESHOLD` |
| RQ-KMeans | `RQ_KMEANS_LEVELS`, `RQ_KMEANS_MAX_K`, `RQ_RESIDUAL_TAU` |
| Agent envelope | `AGENT_COARSE_ACTIVATION_BUDGET`, `AGENT_COARSE_JUMP_BUDGET`, `AGENT_MID_ACTIVATION_BUDGET`, `AGENT_MID_EXPANSION_RADIUS_CAP`, `AGENT_FINE_CLUSTER_BUDGET`, `AGENT_CHUNK_CANDIDATE_BUDGET`, `AGENT_STRUCTURE_RESTORE_BUDGET`, `AGENT_PLANNING_ROUND_BUDGET`, `AGENT_MAX_TYPED_ACTIONS_PER_ROUND`, `AGENT_REPAIR_ROUND_BUDGET`, `AGENT_VERIFICATION_BUDGET` |
| Rerank | `RERANKER_ENABLED`, `RERANKER_MODEL`, `RERANKER_MAX_LENGTH`, `RERANKER_DEVICE`, `HF_HUB_OFFLINE` |
| Fallback | `ENABLE_MODEL_FALLBACK`, `ENABLE_DATABASE_FALLBACK` |

## Verification

```powershell
cd apps/api
python -m pytest tests

cd ../..
npm run typecheck --workspace web
npm run lint --workspace web
npm run test --workspace web
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api
```

## Operations

```powershell
python scripts/diagnose_context_graph.py
python scripts/evaluate_layered_retrieval.py
python scripts/check_context_package_quality.py
python scripts/reconcile_vector_records.py
python scripts/reconcile_bm25_records.py
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api --worker-container course-kg-worker
```

Write scripts default to dry-run or require explicit `--execute`. Generated reports go under `output/`.

## Documents

- [docs/technical-spec.md](./docs/technical-spec.md): Four-Layer Context Graph RAG technical white paper.
- [docs/todo.md](./docs/todo.md): engineering implementation checklist.
- [apps/api/README.md](./apps/api/README.md): API backend guide.
- [apps/web/README.md](./apps/web/README.md): Web frontend guide.
- [apps/worker/README.md](./apps/worker/README.md): Worker guide.
- [scripts/README.md](./scripts/README.md): maintenance scripts.
- [infra/README.md](./infra/README.md): Docker Compose runtime.

## Boundaries

- `chunks` are the primary unit for indexing, citation, retrieval, QA, and graph links.
- The structure graph preserves the source map and does not change fixed chunk boundaries.
- The chunk relation graph stores reproducible relations only.
- Mid/Coarse concepts require support chunks, support spans, fine clusters, or bridge chunks.
- QA uses context packages, not raw search results.
- Citations point to raw chunk spans.
- Qdrant, BM25, and Redis are derived state and must be rebuildable from PostgreSQL.
- Profile affects only prompts, UI labels, and conversation preferences.
- PostgreSQL, Qdrant, Redis, model endpoints, and no-fallback paths are verified inside the Docker stack.

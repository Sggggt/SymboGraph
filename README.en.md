**English** | [中文](./README.md)

<p align="center">
  <img src="./assets/diagraph-logo.svg" alt="SymboGraph logo" width="132" height="132">
</p>

<h1 align="center">SymboGraph</h1>

## Overview

SymboGraph is a local, general-purpose intelligent knowledge base. It uses Four-Layer Context Graph RAG: fixed token chunks provide stable index and citation addresses, the Chunk Structure Graph only preserves the source map and context-restoration paths, the Chunk Relation Graph stores reproducible low-level semantic relations and RQ chunk-pair evidence, the Mid Concept Graph is projected strictly from RQ L3 prefix packets, the Coarse Concept Graph is projected strictly from RQ L2 prefix packets, and QA answers are generated from context packages with citation verification.

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

SymboGraph is built for local document libraries, course material, technical docs, and research collections that need strict citation, traceable graph retrieval, and inspectable context packaging. PostgreSQL is the source of truth; Qdrant and Redis are active derived/runtime state. Answer facts come only from context packages and raw chunk citation spans.

## Stack

| Area | Technology |
| --- | --- |
| Backend | Python 3.13, FastAPI, SQLAlchemy, Alembic, Pydantic |
| Storage | PostgreSQL 16, Qdrant 1.17.1, Redis 7 |
| Jobs | Celery, Redis broker |
| Retrieval | Dense embedding, dense-only chunk relation graph, RQ membership, staged layered traversal |
| Models | OpenAI-compatible chat and embedding endpoints |
| Frontend | Next.js 16.2.4, React 19.2.4, TypeScript, TanStack Query, Tailwind CSS, ECharts |
| Operations | Docker Compose, Python maintenance scripts, pytest, Vitest, ESLint |

## Pipeline

```text
source files
-> parser and layout extractor
-> fixed token chunks
-> chunk structure graph
-> contextual embedding and vector index
-> optional automatic TPE operating point selection on chunk-version increments
-> independent chunk relation graph
-> RQ prefix address tree and fuzzy membership
-> RQ L3 mid concept graph
-> RQ L2 coarse concept graph
-> active context graph state
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

Configure chat, graph-build, and embedding endpoints:

```env
CHAT_API_KEY=...
CHAT_BASE_URL=https://your-chat-endpoint/v1
CHAT_MODEL=your-chat-model

GRAPH_API_KEY=...
GRAPH_BASE_URL=https://your-graph-endpoint/v1
GRAPH_MODEL=your-graph-model

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
| Model bridge | `MODEL_BRIDGE_ENABLED`, `MODEL_BRIDGE_PORT`, `MODEL_BRIDGE_ADMIN_TOKEN` |
| Chat | `CHAT_API_KEY`, `CHAT_BASE_URL`, `CHAT_RESOLVE_IP`, `CHAT_MODEL` |
| Graph build | `GRAPH_API_KEY`, `GRAPH_BASE_URL`, `GRAPH_RESOLVE_IP`, `GRAPH_MODEL` |
| Embedding | `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_RESOLVE_IP`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `EMBEDDING_BATCH_SIZE` |
| Concurrency | `WORKER_CONCURRENCY`, `WORKER_MAX_TASKS_PER_CHILD`, `MODEL_REQUEST_CONCURRENCY`, `MODEL_REQUEST_TIMEOUT_SECONDS` |
| Resource guards | `INGESTION_MEMORY_SOFT_LIMIT_RATIO`, `INGESTION_MEMORY_HARD_LIMIT_RATIO`, `INGESTION_MEMORY_CRITICAL_LIMIT_RATIO` |
| Chunks and context | `FIXED_CHUNK_SIZE_TOKENS`, `FIXED_CHUNK_OVERLAP_TOKENS`, `CONTEXT_PACKAGE_TOKEN_BUDGET` |
| Mid concepts | `MID_CONCEPT_EXTRACTION_MAX_MODEL_BATCHES`, `MID_CONCEPT_EXTRACTION_MAX_CANDIDATES_PER_BATCH`, `MID_CONCEPT_EXTRACTION_MAX_TOKENS_PER_BATCH`, `MID_CONCEPT_CANDIDATE_KEEP_THRESHOLD` |
| RQ-KMeans | `RQ_KMEANS_LEVELS`, `RQ_KMEANS_MAX_K`, `RQ_RESIDUAL_TAU` |
| Dense relation operating point | `DENSE_KNN_K_MIN`, `DENSE_KNN_K_MAX`, `DENSE_REVERSE_B_MIN_BASE`, `DENSE_REVERSE_B_MAX_BASE`, `DENSE_REVERSE_B_MIN_DOC`, `DENSE_REVERSE_B_MAX_DOC`, `DENSE_REVERSE_B_MIN_LANG`, `DENSE_REVERSE_B_MAX_LANG`, `DENSE_MIN_COSINE`, `DENSE_STRONG_COSINE`, `CROSS_DOC_OUT_QUOTA_MIN`, `CROSS_DOC_OUT_QUOTA_MAX`, `CROSS_DOC_MIN_COSINE`, `CROSS_LANGUAGE_OUT_QUOTA_MIN`, `CROSS_LANGUAGE_OUT_QUOTA_MAX`, `CROSS_LANGUAGE_MIN_COSINE` |
| Auto TPE operating point | `ENABLE_AUTO_TPE`, `TPE_TRIAL_BUDGET`, `TPE_STARTUP_RANDOM_TRIALS`, `TPE_GOOD_QUANTILE_GAMMA`, `TPE_PROBE_QUERY_BUDGET`, `TPE_TRIAL_TIMEOUT_SECONDS`, `TPE_CANDIDATE_POOL_SIZE` |
| Operating point hard gate | `OPERATING_POINT_HARD_GATE_MAX_EDGE_DENSITY`, `OPERATING_POINT_HARD_GATE_MAX_ISOLATED_RATIO`, `OPERATING_POINT_HARD_GATE_MAX_HUBNESS_RATIO`, `OPERATING_POINT_HARD_GATE_MIN_STRUCTURE_RECOVERY_RATE`, `OPERATING_POINT_HARD_GATE_MAX_CANDIDATE_LATENCY_P95_MS` |
| Agent traversal budgets | `AGENT_COARSE_INITIAL_BUDGET`, `AGENT_COARSE_TOP_K`, `AGENT_MID_PER_COARSE_BUDGET`, `AGENT_COARSE_DRILLDOWN_MID_INITIAL_BUDGET`, `AGENT_MID_INITIAL_BUDGET`, `AGENT_MID_TOP_K`, `AGENT_CHUNK_PER_MID_BUDGET`, `AGENT_CHUNK_INITIAL_BUDGET`, `AGENT_CHUNK_TOP_K`, `CANDIDATE_POOL_DEDUPE_BUDGET` |
| Agent path and context envelope | `AGENT_MAX_DEPTH_PER_LAYER`, `AGENT_MAX_LABELS_PER_NODE`, `AGENT_MAX_EDGE_REUSE`, `AGENT_MAX_CYCLE_REWARD_PER_PATH`, `AGENT_CYCLE_REWARD_DISTANCE_THRESHOLD`, `AGENT_PATH_DISTANCE_GREEN_THRESHOLD`, `AGENT_PATH_DISTANCE_GRAY_THRESHOLD`, `AGENT_PATH_DISTANCE_HARD_THRESHOLD`, `AGENT_STRUCTURE_RESTORE_PER_CHUNK_BUDGET`, `CONTEXT_PATH_SUMMARY_BUDGET` |
| Agent planning and verification envelope | `AGENT_PLANNING_ROUND_BUDGET`, `AGENT_MAX_TYPED_ACTIONS_PER_ROUND`, `AGENT_REPAIR_ROUND_BUDGET`, `AGENT_VERIFICATION_BUDGET` |
| Agent compatibility aliases | `AGENT_COARSE_TOTAL_BUDGET`, `AGENT_STRUCTURE_RESTORE_BUDGET` |
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
python scripts/evaluate_agent_trace.py
python scripts/check_technical_spec_compliance.py --knowledge-base-name Bayes
python scripts/reconcile_vector_records.py
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api
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
- The structure graph preserves the source map and context-restoration paths only; it does not create, retain, or weight chunk relation edges.
- The chunk relation graph stores content-semantic relations and allowed RQ chunk-pair evidence only.
- Mid concepts are a one-to-one projection of RQ L3 prefixes, Coarse concepts are a one-to-one projection of RQ L2 prefixes, and upper-layer edges must be projected from bottom chunk relation edge support.
- QA uses context packages, not raw search results.
- Citations point to raw chunk spans.
- Qdrant and Redis are active derived/runtime state and must be rebuildable or refreshable from PostgreSQL.
- Profile affects only prompts, UI labels, and conversation preferences.
- PostgreSQL, Qdrant, Redis, model endpoints, and no-fallback paths are verified inside the Docker stack.

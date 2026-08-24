**English** | [中文](./README.md)

<p align="center">
  <img src="./assets/diagraph-logo.svg" alt="SymboGraph logo" width="132" height="132">
</p>

<h1 align="center">SymboGraph</h1>

## Overview

SymboGraph is a local, general-purpose intelligent knowledge base. It uses Four-Layer Context Graph RAG: fixed token chunks provide stable index and citation addresses, the Chunk Structure Graph preserves source structure and context-restoration paths, and the Chunk Relation Graph stores reproducible low-level semantic relations and RQ chunk-pair evidence. RQ persists exactly one L1/L2/L3 primary address chain per chunk. Deterministic eligibility rules compress RQ L3 and L2 prefix packets into Mid and Coarse concepts while enforcing `Mid≤chunks` and `Coarse≤Mid`. QA answers are generated from context packages and must pass citation verification.

## Repository Layout

| Path | Responsibility |
| --- | --- |
| `apps/api` | FastAPI backend for ingestion, parsing, chunks, four-layer graphs, retrieval, QA, settings, and maintenance. |
| `apps/web` | Next.js 16.2.4 frontend for the knowledge base UI, graphs, retrieval traces, context packages, QA, and settings. |
| `apps/worker` | Celery worker and file watcher for long-running jobs, reusing API service logic. |
| `packages/shared` | Shared frontend/backend types and contracts. |
| `infra` | Default Docker Compose runtime. |
| `scripts` | Rebuild, reconciliation, diagnostics, retrieval evaluation, quality checks, and Docker smoke. |
| `docs` | Four-Layer Context Graph RAG technical white paper. |
| `output` | Ephemeral local diagnostics generated on demand; always ignored and safe to clear. |

## Product Positioning

SymboGraph is built for local document libraries, course material, technical docs, and research collections that need strict citation, traceable graph retrieval, and inspectable context packaging. PostgreSQL is the source of truth; Qdrant and Redis are active derived/runtime state. Answer facts come only from context packages and raw chunk citation spans.

## Stack

| Area | Technology |
| --- | --- |
| Backend | Python 3.13, FastAPI, SQLAlchemy, Alembic, Pydantic |
| Storage | PostgreSQL 16, Qdrant 1.17.1, Redis 7 |
| Jobs | Celery, Redis broker |
| Retrieval | Dense embedding, dense-only chunk relation graph, RQ membership, staged layered traversal |
| Models | Selectable OpenAI-compatible / Anthropic Messages chat and graph endpoints; OpenAI-compatible embeddings |
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
-> RQ prefix address tree and primary-chain confidence
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

Configure PostgreSQL plus the chat, graph-build, and embedding endpoints. Generate the database password locally and never commit it:

```env
POSTGRES_USER=symbograph
POSTGRES_PASSWORD=<local-random-password>
POSTGRES_DB=symbograph
DATABASE_URL=postgresql+psycopg://symbograph:<local-random-password>@localhost:5432/symbograph

CHAT_API_KEY=...
CHAT_API_PROTOCOL=openai
CHAT_BASE_URL=https://your-chat-endpoint/v1
CHAT_MODEL=your-chat-model

GRAPH_API_KEY=...
GRAPH_API_PROTOCOL=openai
GRAPH_BASE_URL=https://your-graph-endpoint/v1
GRAPH_MODEL=your-graph-model

EMBEDDING_API_KEY=...
EMBEDDING_API_PROTOCOL=openai
EMBEDDING_BASE_URL=https://your-embedding-endpoint/v1
EMBEDDING_MODEL=your-embedding-model
EMBEDDING_DIMENSIONS=1024
```

The three protocol settings are independent. `CHAT_API_PROTOCOL` and `GRAPH_API_PROTOCOL` may each use `openai` or `anthropic`; for `anthropic`, set the corresponding base to the provider root or a path prefix that does not end in `/v1` or `/v1/messages`, and the client appends `/v1/messages`. `openai` chat/graph append `/chat/completions`. `EMBEDDING_API_PROTOCOL` currently accepts only `openai` and appends `/embeddings`; Anthropic Messages has no embedding contract. The embedding protocol is `rebuild_required` and can become active only through candidate, shadow build, evaluation, and promotion.

Keep fallback disabled for the active runtime:

```env
ENABLE_MODEL_FALLBACK=false
ENABLE_DATABASE_FALLBACK=false
```

The settings page updates the repository-root `.env` directly. `hot_reloadable` fields are published to the active runtime; `rebuild_required` and `service_recreate_required` fields wait for explicit rebuild/promotion or service recreation. Saving the form never silently rewrites the active graph or container topology. Real endpoints, model names, and API keys belong only in the Git-ignored root `.env`; they must not enter logs, reports, or test fixtures. Checked-in examples use placeholders or synthetic `.invalid` values.

## Quick Start

```powershell
.\start-app.ps1
```

The launcher rebuilds API/Web with mutable local tags; worker reuses the API
image. An `.env` value such as `API_IMAGE=name@sha256:...` is a locked runtime
reference and cannot be a Docker build output tag. `rebuild-images.ps1`
explicitly overrides it for local builds:

```powershell
# Rebuild local images only
.\rebuild-images.ps1

# Existing digests are runtime-only and require skipping the build
.\start-app.ps1 -SkipBuild `
  -ApiImage "course-kg-api@sha256:<digest>" `
  -WebImage "course-kg-web@sha256:<digest>"
```

Open:

```text
Web: http://127.0.0.1:3000
API: http://127.0.0.1:8000/api
Readiness: http://127.0.0.1:8000/api/ready
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
| Chat | `CHAT_API_KEY`, `CHAT_API_PROTOCOL`, `CHAT_BASE_URL`, `CHAT_RESOLVE_IP`, `CHAT_MODEL` |
| Graph build | `GRAPH_API_KEY`, `GRAPH_API_PROTOCOL`, `GRAPH_BASE_URL`, `GRAPH_RESOLVE_IP`, `GRAPH_MODEL` |
| Embedding | `EMBEDDING_API_KEY`, `EMBEDDING_API_PROTOCOL` (`openai`), `EMBEDDING_BASE_URL`, `EMBEDDING_RESOLVE_IP`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `EMBEDDING_BATCH_SIZE` |
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
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api --execute
```

The first smoke command performs GET-only preflight and prints the exact KB,
query, `POST /search`, `POST /qa`, and impact plan. Only the second command
sends the write-capable POST requests.

Before committing or pushing, also run:

```powershell
git status --short
git diff --check
git ls-files -o --exclude-standard
git ls-files -ci --exclude-standard
```

Verify that `.env`, local databases, source documents, `output/`, browser traces, certificates, and private keys are not tracked. Review every commit being pushed, not only the current worktree diff. Test credentials must be visibly synthetic (`unit-test-*` and `.invalid`); never copy a real endpoint, bridge token, personal filename, or private-document hash into a fixture.

## Operations

```powershell
python scripts/diagnose_context_graph.py
python scripts/evaluate_layered_retrieval.py
python scripts/evaluate_layered_retrieval.py --query "<query>" --execute
python scripts/check_context_package_quality.py
python scripts/evaluate_agent_trace.py
python scripts/check_technical_spec_compliance.py --knowledge-base-name "<knowledge-base-name>"
python scripts/reconcile_vector_records.py
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api --execute
```

`evaluate_layered_retrieval.py` replays persisted traces by default; creating a
new retrieval requires an explicit query plus `--execute`. Write scripts
default to dry-run or require explicit `--execute`. Diagnostics go under the
Git-ignored `output/` directory and may be cleared after review.

## Documents

- [docs/technical-spec.md](./docs/technical-spec.md): Four-Layer Context Graph RAG technical white paper.
- [apps/api/README.md](./apps/api/README.md): API backend guide.
- [apps/web/README.md](./apps/web/README.md): Web frontend guide.
- [apps/worker/README.md](./apps/worker/README.md): Worker guide.
- [scripts/README.md](./scripts/README.md): maintenance scripts.
- [infra/README.md](./infra/README.md): Docker Compose runtime.

## Boundaries

- `chunks` are the primary unit for indexing, citation, retrieval, QA, and graph links.
- The structure graph preserves the source map and context-restoration paths only; it does not create, retain, or weight chunk relation edges.
- The chunk relation graph stores content-semantic relations and allowed RQ chunk-pair evidence only.
- RQ persists one three-level primary chain and its confidence per chunk. Deterministic eligibility rules compress Mid/Coarse concepts, and upper-layer edges must be projected from bottom chunk relation edge support.
- QA uses context packages, not raw search results.
- Citations point to raw chunk spans.
- Qdrant and Redis are active derived/runtime state and must be rebuildable or refreshable from PostgreSQL.
- Profile affects only prompts, UI labels, and conversation preferences.
- PostgreSQL, Qdrant, Redis, model endpoints, and no-fallback paths are verified inside the Docker stack.

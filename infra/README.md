# Infrastructure

The local stack is Docker-first and split into reusable infrastructure plus small project images.

## Services

- `api`: project FastAPI image, `course-kg-api:local`
- `worker`: project Celery worker image, `course-kg-api:local`
- `web`: project Next.js image, `course-kg-web:local`
- `postgres`: reusable `postgres:16`
- `redis`: reusable `redis:7`
- `qdrant`: reusable `qdrant/qdrant:v1.17.1`

PostgreSQL must stay on major version 16 because the existing data directory has `PG_VERSION=16`.
Qdrant is pinned to 1.17.1 to match the API client's generated models.

## Persistent Profile State

Knowledge-base Profiles are persisted in PostgreSQL, not in `.env`. The relevant records are:

- `strategy_profiles`: Profile name, library type, JSON strategy payload, hash, built-in flag, and active flag.
- `courses.active_profile_id`: the active Profile binding for each knowledge base.
- Graph extraction run metadata: records `strategy_profile_id` and `strategy_profile_hash` for freshness checks.

Redis is used only for runtime/cache state such as Profile Assistant sessions. Deleting Redis data may clear assistant conversation state, but it does not delete saved Profiles. Deleting or recreating PostgreSQL volumes removes saved Profile definitions along with the rest of application metadata.

## Validate Existing Images

If these reusable images already exist on your machine, validate them and skip rebuilding:

```powershell
docker run --rm postgres:16 postgres --version
docker run --rm redis:7 redis-server --version
docker image inspect qdrant/qdrant:v1.17.1
```

## Build Missing Images

Build only the images you do not already have:

```powershell
docker build -f apps/api/Dockerfile -t course-kg-api:local .
docker build -f apps/web/Dockerfile -t course-kg-web:local .
```

## Run

From the repository root:

```powershell
.\start-app.ps1
```

The API image includes the reranker Python extra in system Python. Enable it with `RERANKER_ENABLED=true`; no separate reranker container or virtual environment is used.

Direct Compose examples:

```powershell
docker compose -f infra/docker-compose.yml up -d postgres redis qdrant api worker web
```

The worker consumes `course-kg-main-ingestion` and uses `WORKER_CONCURRENCY` for Celery process concurrency. Keep this value aligned with API-side bounded concurrency settings such as `INGESTION_FILE_CONCURRENCY`, `MODEL_REQUEST_CONCURRENCY`, and `HPO_CONCURRENCY`.

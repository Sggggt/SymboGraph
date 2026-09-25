# SymboGraph

[中文](README.md)

SymboGraph is a local document knowledge base. It builds four connected layers: source structure, chunk relations with RQ membership, mid-level concepts, and coarse concepts. Before planning, the LLM may read a complete coarse-node title directory and then a few selected summaries, or plan directly. The architecture separates task intent from execution strategy. The LLM chooses a coarse, mid, or chunk entry layer. Queries without lexical terms use dense-only entry selection; hybrid plans combine dense, RQ-cluster, and BM25 rankings with per-layer LLM-selected weights. Graph traversal restores raw evidence, followed by deterministic source admission, one answer generation, and citation binding.

The active serving path now uses this architecture. Dense, RQ, and BM25 only nominate and fuse graph entries; final evidence must pass graph traversal, structure restoration, and source admission. See the [technical specification](docs/technical-spec.md) for the implementation contract and the [research](docs/reference/retrieval-research.md) for design tradeoffs.

The Web product exposes Overview, Upload, QA, Graph, and Settings. The standalone Search page has been removed while QA continues to use the graph retrieval path above. PostgreSQL persists and restores conversation history, including failed and cancelled turns, and answers retain GFM, code, and KaTeX rendering.

## Start

Use Docker Desktop with Linux containers and PowerShell. Keep any existing root configuration files:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
if (-not (Test-Path settings.json)) { Copy-Item settings.example.json settings.json }
```

Configure database and model connections in `.env`; keep non-secret product, retrieval, budget, and build settings in `settings.json`. Double-click `start-app.bat` for one-click startup. Its window remains open after either success or failure until a key is pressed, while the launched services continue in the background. From a terminal, run:

```powershell
.\start-app.ps1 -NoBrowser
```

Defaults: Web at http://127.0.0.1:3000, API at http://127.0.0.1:8000/api, readiness at `/api/ready`, OpenAPI at `/docs`.

The launcher builds the API image and starts the backend Compose services; Worker and Beat reuse that image. Web runs with host-native Node.js and the locked `node_modules`. Automation that must call the batch wrapper can set process-local `SYMBOGRAPH_NO_PAUSE=1`; interactive command-line use should call the `.ps1` entry directly. See [infrastructure](infra/README.md) for image overrides, Compose project identity, data volumes and model protocols.

## Repository

| Path | Purpose |
|---|---|
| [apps/api](apps/api/README.md) | FastAPI, lifecycle, parsing, graph construction, retrieval and QA |
| [apps/web](apps/web/README.md) | Next.js 16.2.4 UI and shared API contracts |
| [apps/worker](apps/worker/README.md) | Bounded background tasks using API services |
| `packages/shared` | Shared TypeScript contracts |
| [scripts](scripts/README.md) | Persistent maintenance, diagnostics and acceptance tools |
| [docs](docs/technical-spec.md) | White paper, protocol references and development guide |
| `output` | Entire directory ignored; remove temporary artifacts after verification |

PostgreSQL is the lifecycle/audit source of truth. Qdrant and Redis are reconstructible derived/runtime stores. Application documents are in a Docker named volume, not the repository's local `data/` directory.

The root `.env` and `settings.json` form one configuration authority with disjoint keys. `.env` owns secrets, connections, paths, ports, and process/service startup values; `settings.json` owns non-secret product, retrieval, budget, and build values. Runtime Settings writes each field only to its owning file and retains hot-reload, rebuild-required, and service-recreate lifecycles. Profile controls prompts and conversation preferences. Keep production fallbacks disabled, and do not commit credentials, deployment endpoints, private documents or generated reports.

Read the [development guide](docs/development.md), [technical specification](docs/technical-spec.md), and [script catalog](scripts/README.md). Local acceptance records are kept in Git-ignored working files and are not part of the public repository.

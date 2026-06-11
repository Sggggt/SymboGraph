# API Tests

`apps/api/tests` covers the active evidence-first backend: ingestion, evidence graph
construction, active chunks, quality decisions, retrieval traces, QA grounding,
runtime settings, and maintenance.

## Run

From `apps/api`:

```powershell
python -m pytest tests -q
```

Core evidence-first tests:

```powershell
python -m pytest tests/test_evidence_graph_pipeline.py tests/test_evidence_first_retrieval.py tests/test_quality_system.py -q
```

Inside the Docker API container:

```powershell
docker exec -w /app/apps/api course-kg-api python -m pytest tests/test_evidence_graph_pipeline.py tests/test_evidence_first_retrieval.py tests/test_quality_system.py -q
```

## Coverage Map

| File | Coverage |
| --- | --- |
| `test_evidence_graph_pipeline.py` | evidence atoms, edges, signal layer, active chunks, graph payload |
| `test_evidence_first_retrieval.py` | active chunk retrieval, signal expansion, retrieval audit |
| `test_query_evidence_graph.py` | query-scoped evidence graph payload |
| `test_quality_system.py` | quality gate, reward, feedback policies |
| `test_agent_graph.py` | QA agent, citations, no-fallback path |
| `test_ingestion_batches.py` | ingestion batches, cancellation, version behavior, worker coordination |
| `test_runtime_settings.py` | `.env` writes, Redis version broadcast, hot reload |
| `test_maintenance.py` | destructive cleanup and derived-state maintenance |

## Rules

- New behavior gets regression tests.
- Data, retrieval, embedding, evidence graph, chunking, quality, community, policy,
  or runtime settings changes need focused unit and integration coverage.
- Paths that require PostgreSQL, Qdrant, Redis, or real model endpoints should run
  in the Docker stack.
- Generated reports, logs, temporary corpus results, and smoke outputs go under
  repository-root `output/`.

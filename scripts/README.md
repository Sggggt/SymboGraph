# Scripts

`scripts` contains smoke checks, quality gates, evaluation tools, re-embedding
utilities, reingestion helpers, and maintenance entrypoints for the active
evidence-first architecture.

## Recommended Runtime

From the repository root:

```powershell
python scripts\docker_smoke.py --base-url http://127.0.0.1:8000/api --worker-container course-kg-worker
```

When a script requires container paths or Docker services:

```powershell
docker exec -w /app/apps/api course-kg-api python /app/scripts/quality_gate.py --help
```

## Script Map

| Script | Purpose |
| --- | --- |
| `docker_smoke.py` | Creates a temporary knowledge base and verifies upload, parsing, evidence graph, signal layer, search, QA, and citations |
| `quality_gate.py` | Checks active chunks, evidence atoms, vector records, Qdrant points, communities, and derived state invariants |
| `analyze_chunk_quality.py` | Reports active chunk length, content kind, duplicate text, evidence atom refs, and embedding version distribution |
| `evaluate_evidence_first_retrieval.py` | Evaluates evidence-first retrieval behavior |
| `evaluate_existing_quality.py` | Evaluates existing QA quality with a judge model |
| `evaluate_quality_decisions.py` | Exports quality decision and chunk decision diagnostics |
| `manage_migrations.py` | Runs Alembic current, heads, check, upgrade, downgrade, and revision inside the API container |
| `reembed_all_chunks.py` | Repairs missing or zero active chunk vectors, or refreshes all active chunk vectors with `--refresh-all` |
| `reembed_with_enhancement.py` | Compatibility entrypoint that delegates to `reembed_all_chunks.py` |
| `reingest_all_knowledge_bases.py` | Reingests stored source files through the active ingestion pipeline |

## Output

All smoke outputs, benchmark logs, evaluation reports, and temporary corpus
acceptance artifacts should be written to:

```text
output/
```

## Safety

- Data-writing scripts must print the target knowledge base, processed counts,
  and impact scope.
- Destructive scripts need an explicit flag or a dry-run mode.
- Scripts must not print API keys, Authorization headers, or raw provider
  responses that may contain credentials.
- Active paths must not use fake embeddings, zero vectors, or silent fallback.

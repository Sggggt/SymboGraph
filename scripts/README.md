# Operations Scripts

`scripts/` contains maintenance, diagnostics, rebuild, migration, runtime probe, and Docker smoke entry points for SymboGraph.

Rules for this directory:

- Do not add scripts that depend on personal absolute paths, private local datasets, or hidden local state.
- Do not hard-code private knowledge-base names, local document titles, private questions, API keys, tokens, or provider responses.
- Scripts that write data must require an explicit execution flag such as `--execute`.
- Destructive scripts must print targets and require an explicit confirmation flag.
- Reports must be written under `output/`, which is ignored by git.

## Script Index

| Script | Purpose |
| --- | --- |
| `rebuild_chunks.py` | Reparse source files into fixed token chunks, structure graph, embeddings, vector index, and four-layer graph state. |
| `rebuild_structure_graph.py` | Rebuild structure nodes, edges, mappings, and coordinates. |
| `rebuild_chunk_relation_graph.py` | Rebuild independent chunk relation graph and dependent graph states. |
| `rebuild_rq_membership_graph.py` | Rebuild active RQ address and membership graph state. |
| `rebuild_mid_concept_graph.py` | Rebuild RQ L3 projected mid concepts and dependent states. |
| `rebuild_coarse_concept_graph.py` | Rebuild RQ L2 projected coarse concepts and active context graph state. |
| `rebuild_context_graph_all.py` | Rebuild all active four-layer derived graph state from current chunks. |
| `destroy_legacy_derived_data.py` | Clean legacy derived state behind explicit destructive flags. |
| `cleanup_stale_data.py` | Clean stale vector and inactive chunk state behind explicit flags. |
| `reconcile_vector_records.py` | Reconcile PostgreSQL vector records with Qdrant points. |
| `diagnose_context_graph.py` | Emit counts, freshness, grounding, RQ diagnostics, and graph payload samples. |
| `evaluate_layered_retrieval.py` | Run user-supplied layered retrieval queries and emit diagnostics. |
| `evaluate_agent_trace.py` | Run user-supplied QA requests and verify Agent trace, citations, and non-degraded execution. |
| `check_context_package_quality.py` | Check context package closure, citation spans, graph paths, RQ metrics, and token-budget metadata. |
| `check_runtime_settings_contract.py` | Validate runtime settings lifecycle and hot-reload contract. |
| `runtime_hot_reload_probe.py` | Probe Redis runtime settings version publication and local singleton refresh. |
| `check_technical_spec_compliance.py` | Check implementation against technical-spec invariants. |
| `manage_migrations.py` | Run Alembic operations through the API Docker container. |
| `docker_smoke.py` | Run HTTP smoke checks against the Docker API. |

## Examples

Read-only diagnostics:

```powershell
python scripts/diagnose_context_graph.py
python scripts/evaluate_layered_retrieval.py --query "<query>"
python scripts/check_context_package_quality.py --query "<query>"
```

Runtime probe with write intent:

```powershell
python scripts/runtime_hot_reload_probe.py --execute
```

Destructive legacy cleanup:

```powershell
python scripts/destroy_legacy_derived_data.py --execute --confirm-destroy-legacy
```

Docker smoke:

```powershell
python scripts/docker_smoke.py --base-url http://127.0.0.1:8000/api
```

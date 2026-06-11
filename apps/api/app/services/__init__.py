"""Service module map for the Evidence Graph Policy Engine.

See apps/api/README.md for the architecture overview. Key modules:
- evidence_graph: evidence atoms, observed edges, chunk candidates, quality decisions, policy states, active chunks.
- evidence_signal_projection: evidence-bound signal layer and projection diagnostics.
- retrieval: evidence-first dense/lexical/graph/community retrieval and retrieval traces.
- agent_graph: grounded QA workflow, citation verification, reward events, policy updates.
- ingestion: file parsing orchestration, cancellation compensation, vector indexing.
- maintenance: stale-data cleanup, vector-store reconciliation, policy-state reconciliation.
- runtime_settings: shared .env updates, runtime version publishing, singleton cache refresh.
"""

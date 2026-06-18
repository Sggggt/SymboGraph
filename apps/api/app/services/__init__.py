"""Service module map for the Four-Layer Context Graph RAG runtime.

Active services:
- chunking: fixed-token chunking with protected table, formula, code, heading, and caption spans.
- context_graph: structure graph, chunk relation graph, RQ prefix clusters, mid/coarse concepts, layered retrieval, and context packages.
- retrieval: search/dashboard helpers backed by context_graph.
- agent_graph: grounded QA over context packages with citation verification and reward events.
- ingestion: parsing orchestration, versioning, cancellation boundaries, embedding, BM25, and graph rebuilds.
- maintenance: destructive cleanup, vector-store reconciliation, and context-graph policy state reconciliation.
- runtime_settings: shared .env updates, runtime version publishing, singleton cache refresh.
"""

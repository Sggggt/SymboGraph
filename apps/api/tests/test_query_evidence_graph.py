from app.services.evidence_graph_payload import get_query_evidence_graph_payload


def test_query_evidence_graph_uses_retrieved_chunks_and_document_versions(db_session, sample_knowledge_base, indexed_chunks):
    _document, chunks = indexed_chunks

    graph = get_query_evidence_graph_payload(db_session, sample_knowledge_base.id, [chunks[0].id])

    node_ids = {node["id"] for node in graph["nodes"]}
    assert graph["graph_type"] == "evidence"
    assert f"evidence_chunk:{chunks[0].id}" in node_ids
    assert f"document_version:{chunks[0].document_version_id}" in node_ids
    assert any(edge["category"] == "evidence" and edge["label"] == "from_version" for edge in graph["edges"])


def test_query_evidence_graph_ignores_chunks_outside_requested_scope(db_session, sample_knowledge_base, indexed_chunks):
    _document, chunks = indexed_chunks

    graph = get_query_evidence_graph_payload(db_session, sample_knowledge_base.id, [chunks[1].id], query="DAG")

    node_ids = {node["id"] for node in graph["nodes"]}
    assert f"evidence_chunk:{chunks[1].id}" in node_ids
    assert f"evidence_chunk:{chunks[0].id}" not in node_ids

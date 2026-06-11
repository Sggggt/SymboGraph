from __future__ import annotations


def _search_item(chunk, score: float = 0.9) -> dict:
    return {
        "chunk_id": chunk.id,
        "snippet": chunk.snippet,
        "score": score,
        "citations": [],
        "metadata": {"scores": {"fused": score}, "quality_action": "retrieval_candidate"},
        "content": chunk.content,
        "document_title": "doc",
        "source_path": "doc.md",
        "partition": chunk.partition,
        "source_type": chunk.source_type,
    }


def test_evidence_first_planner_uses_only_verified_edges(db_session, sample_knowledge_base, indexed_chunks):
    from app.models import (
        EvidenceAtom,
        EvidenceEdge,
        EvidenceGraphState,
        SignalEdge,
        SignalNode,
        SignalState,
    )
    from app.schemas import SearchFilters
    from app.services.retrieval import (
        apply_unified_retrieval_scores,
        assemble_evidence_documents,
        controlled_signal_enhancement,
        controlled_graph_enhancement,
        plan_evidence_chains,
        plan_signal_projection_paths,
        record_retrieval_trace,
        select_evidence_anchors,
    )

    document, chunks = indexed_chunks
    atom_a = db_session.get(EvidenceAtom, chunks[0].atom_ids_json[0])
    atom_b = db_session.get(EvidenceAtom, chunks[1].atom_ids_json[0])
    assert atom_a is not None
    assert atom_b is not None
    graph_state = EvidenceGraphState(
        knowledge_base_id=sample_knowledge_base.id,
        scope_type="global",
        state_hash="graph-state-test",
        atom_scope_hash="atom-scope-test",
        active_document_version_ids=[chunks[0].document_version_id, chunks[1].document_version_id],
        active_atom_ids=[atom_a.id, atom_b.id],
        state="active",
    )
    db_session.add(graph_state)
    db_session.flush()
    edge = EvidenceEdge(
        graph_state_id=graph_state.id,
        source_atom_id=atom_a.id,
        target_atom_id=atom_b.id,
        edge_type="SEMANTIC_SIMILAR",
        weight=0.9,
        confidence=0.9,
    )
    for chunk, atom in ((chunks[0], atom_a), (chunks[1], atom_b)):
        chunk.graph_state_hash = graph_state.state_hash
        chunk.atom_ids_json = [atom.id]
        chunk.source_span_union_json = {"spans": [{"evidence_atom_id": atom.id, "start": 0, "end": len(chunk.content)}]}
    db_session.add(edge)
    signal_state = SignalState(
        knowledge_base_id=sample_knowledge_base.id,
        evidence_graph_state_id=graph_state.id,
        signal_state_hash="signal-state-test",
        evidence_graph_state_hash=graph_state.state_hash,
        active_signal_scope_hash="active-signal-scope-test",
        status="active",
        eligible_atom_ids_json=[atom_a.id, atom_b.id],
        processed_atom_ids_json=[atom_a.id, atom_b.id],
    )
    db_session.add(signal_state)
    db_session.flush()
    node_a = SignalNode(
        signal_state_id=signal_state.id,
        knowledge_base_id=sample_knowledge_base.id,
        normalized_key="degree centrality",
        canonical_label="Degree Centrality",
        signal_type="definition_like_signal",
        support_atom_ids_json=[atom_a.id],
        support_active_chunk_ids_json=[chunks[0].id],
        source_span_union_json={"spans": [{"evidence_atom_id": atom_a.id, "start": 0, "end": 16}]},
    )
    node_b = SignalNode(
        signal_state_id=signal_state.id,
        knowledge_base_id=sample_knowledge_base.id,
        normalized_key="betweenness centrality",
        canonical_label="Betweenness Centrality",
        signal_type="named_surface",
        support_atom_ids_json=[atom_b.id],
        support_active_chunk_ids_json=[chunks[1].id],
        source_span_union_json={"spans": [{"evidence_atom_id": atom_b.id, "start": 0, "end": 22}]},
    )
    db_session.add_all([node_a, node_b])
    db_session.flush()
    signal_edge = SignalEdge(
        signal_state_id=signal_state.id,
        knowledge_base_id=sample_knowledge_base.id,
        source_signal_id=node_a.id,
        target_signal_id=node_b.id,
        edge_type="co_supported_by_atom",
        confidence=0.8,
        support_atom_ids_json=[atom_a.id, atom_b.id],
        support_active_chunk_ids_json=[chunks[0].id, chunks[1].id],
        source_span_union_json={"spans": [{"evidence_atom_id": atom_a.id}, {"evidence_atom_id": atom_b.id}]},
    )
    db_session.add(signal_edge)
    chunks[0].metadata_json = {
        **(chunks[0].metadata_json or {}),
        "active_chunk_id": chunks[0].id,
        "evidence_atom_ids": [atom_a.id],
        "graph_state_hash": graph_state.state_hash,
        "signal_state_hash": signal_state.signal_state_hash,
        "signal_node_ids": [node_a.id],
    }
    chunks[1].metadata_json = {
        **(chunks[1].metadata_json or {}),
        "active_chunk_id": chunks[1].id,
        "evidence_atom_ids": [atom_b.id],
        "graph_state_hash": graph_state.state_hash,
        "signal_state_hash": signal_state.signal_state_hash,
        "signal_node_ids": [node_b.id],
    }
    db_session.commit()

    base = _search_item(chunks[0])
    base["metadata"].update(
        {
            "evidence_atom_ids": [atom_a.id],
            "active_chunk_id": chunks[0].id,
            "graph_state_hash": graph_state.state_hash,
            "signal_state_hash": signal_state.signal_state_hash,
            "signal_node_ids": [node_a.id],
        }
    )
    anchors, anchor_audit = select_evidence_anchors(db_session, sample_knowledge_base.id, [base])
    assert anchor_audit["anchor_count"] == 1
    assert anchors[0]["metadata"]["graph_verified"] is True
    assert anchor_audit["traceable_anchor_signals"] == 1

    paths, path_audit = plan_evidence_chains(db_session, sample_knowledge_base.id, anchors, query_type="comparison")
    assert path_audit["planned_paths"] == 1
    assert paths[0]["evidence_edge_ids"] == [edge.id]

    enhanced, graph_audit = controlled_graph_enhancement(
        db_session,
        sample_knowledge_base.id,
        "compare centrality",
        filters=SearchFilters(),
        base_chunk_ids={chunks[0].id},
        paths=paths,
    )
    assert graph_audit["graph_enhanced_chunks"] == 1
    assert enhanced[0]["metadata"]["evidence_role"] == "evidence_neighbor"
    assert enhanced[0]["metadata"]["graph_verified"] is True

    signal_paths, signal_audit = plan_signal_projection_paths(
        db_session,
        sample_knowledge_base.id,
        anchors,
        query="degree centrality",
        signal_state=signal_state,
    )
    assert signal_audit["signal_state_hash"] == signal_state.signal_state_hash
    assert signal_audit["planned_paths"] == 1
    signal_enhanced, signal_result_audit = controlled_signal_enhancement(
        db_session,
        sample_knowledge_base.id,
        "degree centrality",
        filters=SearchFilters(),
        base_chunk_ids={chunks[0].id},
        paths=signal_paths,
    )
    assert signal_result_audit["signal_enhanced_chunks"] == 1
    assert signal_enhanced[0]["metadata"]["evidence_role"] == "signal_neighbor"
    assert signal_enhanced[0]["metadata"]["signal_state_hash"] == signal_state.signal_state_hash

    assembled, assembly_audit = assemble_evidence_documents([_search_item(chunks[0])], anchors, enhanced, top_k=3, signal_results=signal_enhanced)
    assert assembly_audit["graph_documents"] == 1
    assert assembly_audit["signal_documents"] == 1
    assert {item["chunk_id"] for item in assembled} >= {chunks[0].id, chunks[1].id}
    unified, unified_audit = apply_unified_retrieval_scores(assembled, query_type="comparison")
    assert unified_audit["protocol_version"] == "retrieval_unified_score_v1"
    assert all("unified" in item["metadata"]["scores"] for item in unified)
    assert any(item["metadata"]["scores"].get("graph_path") for item in unified)
    assert any(item["metadata"]["scores"].get("signal_projection") for item in unified)
    assert all("uncertainty" in item["metadata"]["scores"] for item in unified)

    trace = record_retrieval_trace(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        query="degree centrality",
        filters=SearchFilters(),
        retrieval_mode="evidence_first_v1",
        results=assembled,
        audit={"signal": signal_audit, "signal_state_hash": signal_state.signal_state_hash},
    )
    assert trace.diagnostics_json["signal_state_hash"] == signal_state.signal_state_hash
    assert node_a.id in trace.diagnostics_json["retrieval_signal_node_ids"]

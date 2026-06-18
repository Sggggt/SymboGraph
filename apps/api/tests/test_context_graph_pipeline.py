from __future__ import annotations

import json

import pytest


def test_structure_edges_are_rejected_from_active_relation_graph():
    from app.services.context_graph import relation_edge_source_algorithm

    with pytest.raises(RuntimeError, match="Structure-derived relation edges are not allowed"):
        relation_edge_source_algorithm("same_page_region")


def test_entry_seed_calibration_prevents_zero_distance_route_seeds():
    from app.services.context_graph import calibrated_entry_seed_strength, distance_from_strength

    assert calibrated_entry_seed_strength(1.0, "bm25_entry") == pytest.approx(0.94)
    assert calibrated_entry_seed_strength(1.0, "mid_drilldown_entry") == pytest.approx(0.82)
    assert calibrated_entry_seed_strength(1.0, "coarse_to_mid_drilldown_entry") == pytest.approx(0.72)
    assert distance_from_strength(calibrated_entry_seed_strength(1.0, "mid_drilldown_entry")) > 0.0


@pytest.mark.asyncio
async def test_context_graph_pipeline_builds_all_layers(db_session, populated_context_graph):
    from sqlalchemy import func, select

    from app.models import (
        BM25Record,
        Chunk,
        ChunkContextText,
        ChunkRelationEdge,
        ChunkStructureMapping,
        ChunkStructureNode,
        CoarseConcept,
        CoarseConceptState,
        ContextGraphState,
        RQPrefix,
        RQPrefixMembership,
        MidConcept,
        MidConceptState,
        VectorRecord,
    )
    from app.services.context_graph import graph_layer_payload

    kb = populated_context_graph["knowledge_base"]
    assert db_session.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == kb.id, Chunk.state == "active")) >= 3
    assert db_session.scalar(select(func.count(ChunkStructureNode.id)).where(ChunkStructureNode.knowledge_base_id == kb.id)) >= 3
    assert db_session.scalar(select(func.count(ChunkStructureMapping.id))) >= 3
    assert db_session.scalar(select(func.count(ChunkContextText.id))) >= 3
    assert db_session.scalar(select(func.count(VectorRecord.id)).where(VectorRecord.knowledge_base_id == kb.id, VectorRecord.vector_status == "ready")) >= 3
    assert db_session.scalar(select(func.count(BM25Record.id)).where(BM25Record.knowledge_base_id == kb.id, BM25Record.state == "ready")) >= 3
    assert db_session.scalar(select(func.count(ChunkRelationEdge.id)).where(ChunkRelationEdge.knowledge_base_id == kb.id)) >= 2
    assert db_session.scalar(select(func.count(RQPrefix.id)).where(RQPrefix.knowledge_base_id == kb.id)) >= 1
    chunks_with_rq = db_session.scalars(select(Chunk).where(Chunk.knowledge_base_id == kb.id, Chunk.rq_residual_norm.is_not(None))).all()
    assert chunks_with_rq
    assert all(chunk.rq_path for chunk in chunks_with_rq)
    rq_prefixes = db_session.scalars(select(RQPrefix).where(RQPrefix.knowledge_base_id == kb.id, RQPrefix.rq_level.is_not(None))).all()
    assert rq_prefixes
    rq_prefix_memberships = [
        row
        for row in db_session.scalars(select(RQPrefixMembership).where(RQPrefixMembership.residual_norm.is_not(None))).all()
        if row.rq_path
    ]
    assert rq_prefix_memberships
    assert all(row.rq_path for row in rq_prefix_memberships)
    rq_edges = list(
        db_session.scalars(
            select(ChunkRelationEdge).where(
                ChunkRelationEdge.knowledge_base_id == kb.id,
                ChunkRelationEdge.edge_type.in_(["rq_hierarchy_near", "rq_prefix_sibling", "rq_residual_near"]),
            )
        ).all()
    )
    edge_types = {edge.edge_type for edge in rq_edges}
    assert edge_types
    assert not any((edge.features_json or {}).get("fallback_pair") for edge in rq_edges)
    rq_edge = db_session.scalar(select(ChunkRelationEdge).where(ChunkRelationEdge.knowledge_base_id == kb.id, ChunkRelationEdge.edge_type.like("rq_%")))
    assert rq_edge is not None
    assert "lcp_depth" in (rq_edge.features_json or {})
    assert "residual_distance" in (rq_edge.features_json or {})
    assert rq_edge.source_algorithm == "rq_kmeans"
    assert rq_edge.protocol_version
    assert rq_edge.graph_state_hash
    assert all(prefix.parent_rq_prefix_id or int(prefix.rq_level or 0) == 1 for prefix in rq_prefixes)
    graph_payload = graph_layer_payload(db_session, kb.id, "chunk-relation")
    assert graph_payload["full_counts"]["nodes"] >= graph_payload["sampled_counts"]["nodes"]
    assert graph_payload["full_counts"]["edges"] >= graph_payload["sampled_counts"]["edges"]
    assert graph_payload["edge_counts"]["full"] == graph_payload["full_counts"]["edges"]
    assert graph_payload["grounding"]["mid_grounded_rate"] == 1.0
    assert graph_payload["grounding"]["coarse_grounded_rate"] == 1.0
    assert any(node.get("category") == "rq_prefix" for node in graph_payload["nodes"])
    assert any(edge.get("category") == "rq_membership" for edge in graph_payload["edges"])
    assert any(str(edge.get("category", "")).startswith("rq_") for edge in graph_payload["edges"])
    l3_prefix_count = db_session.scalar(
        select(func.count(RQPrefix.id)).where(RQPrefix.knowledge_base_id == kb.id, RQPrefix.rq_level == 3, RQPrefix.state == "active")
    )
    l2_prefix_count = db_session.scalar(
        select(func.count(RQPrefix.id)).where(RQPrefix.knowledge_base_id == kb.id, RQPrefix.rq_level == 2, RQPrefix.state == "active")
    )
    mid_count = db_session.scalar(select(func.count(MidConcept.id)).where(MidConcept.knowledge_base_id == kb.id, MidConcept.state == "active"))
    coarse_count = db_session.scalar(select(func.count(CoarseConcept.id)).where(CoarseConcept.knowledge_base_id == kb.id, CoarseConcept.state == "active"))
    assert mid_count == l3_prefix_count
    assert coarse_count == l2_prefix_count
    mid_state = db_session.scalar(select(MidConceptState).where(MidConceptState.knowledge_base_id == kb.id, MidConceptState.state == "active"))
    assert mid_state is not None
    assert (mid_state.stats_json or {})["projected_rq_l3_prefixes"] == l3_prefix_count
    assert (mid_state.stats_json or {})["rq_l3_to_mid_projection_coverage"] == 1.0
    coarse_state = db_session.scalar(select(CoarseConceptState).where(CoarseConceptState.knowledge_base_id == kb.id, CoarseConceptState.state == "active"))
    assert coarse_state is not None
    assert (coarse_state.stats_json or {})["projected_rq_l2_prefixes"] == l2_prefix_count
    assert (coarse_state.stats_json or {})["rq_l2_to_coarse_projection_coverage"] == 1.0
    mid_audit = [concept.llm_audit_json or {} for concept in db_session.scalars(select(MidConcept).where(MidConcept.knowledge_base_id == kb.id)).all()]
    coarse_audit = [concept.llm_audit_json or {} for concept in db_session.scalars(select(CoarseConcept).where(CoarseConcept.knowledge_base_id == kb.id)).all()]
    assert "rq_path" not in json.dumps(mid_audit + coarse_audit, sort_keys=True)
    state = db_session.scalar(select(ContextGraphState).where(ContextGraphState.knowledge_base_id == kb.id, ContextGraphState.state == "active"))
    assert state is not None
    assert state.chunk_relation_graph_hash
    assert state.mid_concept_hash
    assert state.coarse_concept_hash


@pytest.mark.asyncio
async def test_layered_retrieval_writes_trace_and_context_package(db_session, populated_context_graph):
    from sqlalchemy import func, select

    from app.models import ContextPackage, GraphRetrievalStep, RetrievalTrace
    from app.schemas import SearchFilters
    from app.services.context_graph import build_context_package, layered_search
    from app.services.retrieval import get_context_package

    kb = populated_context_graph["knowledge_base"]
    result = await layered_search(db_session, kb.id, "How does a Markov blanket affect conditional independence?", SearchFilters(), 4)
    assert result.results
    assert result.audit["retrieval_pipeline"] == "layered_context_graph"
    assert "query_rq_path" in result.audit
    assert result.audit["dominance_pruned_count"] >= 0
    assert result.audit["hard_stop_pruned_count"] >= 0
    assert result.audit["gray_zone_decision_count"] >= result.audit["hard_stop_pruned_count"]
    assert any((item.get("metadata") or {}).get("rq") for item in result.results)
    for item in result.results:
        traversal = (item.get("metadata") or {}).get("traversal") or {}
        assert traversal.get("distance_so_far") is not None
        assert float(traversal["distance_so_far"]) >= 0.0
        assert item["score"] <= 1.0
        if not traversal.get("path_edge_ids"):
            assert float(traversal["distance_so_far"]) > 0.0
            assert (item.get("metadata") or {}).get("entry_strengths")
    assert result.trace.id
    assert db_session.scalar(select(func.count(GraphRetrievalStep.id)).where(GraphRetrievalStep.retrieval_trace_id == result.trace.id)) >= 4
    assert (
        db_session.scalar(select(func.count(GraphRetrievalStep.id)).where(GraphRetrievalStep.retrieval_trace_id == result.trace.id, GraphRetrievalStep.layer == "fine"))
        == 0
    )
    seed_step = db_session.scalar(
        select(GraphRetrievalStep).where(GraphRetrievalStep.retrieval_trace_id == result.trace.id, GraphRetrievalStep.layer == "fine")
    )
    assert seed_step is None
    seed_step = db_session.scalar(
        select(GraphRetrievalStep).where(
            GraphRetrievalStep.retrieval_trace_id == result.trace.id,
            GraphRetrievalStep.layer == "chunk",
            GraphRetrievalStep.action == "select_seeds_from_mid_rq_membership",
        )
    )
    assert seed_step is not None
    assert "query_rq_path" in (seed_step.input_json or {})
    assert any("lcp_depth" in candidate for candidate in ((seed_step.output_json or {}).get("candidate_rq") or {}).values())
    package = build_context_package(db_session, knowledge_base_id=kb.id, query="Markov blanket", trace=result.trace, results=result.results)
    db_session.commit()
    structure_step = db_session.scalar(
        select(GraphRetrievalStep).where(GraphRetrievalStep.retrieval_trace_id == result.trace.id, GraphRetrievalStep.layer == "structure")
    )
    assert structure_step is not None
    assert (structure_step.output_json or {}).get("context_package_id") == package.id
    assert db_session.get(RetrievalTrace, result.trace.id) is not None
    assert db_session.get(ContextPackage, package.id) is not None
    assert package.hit_chunk_ids_json
    assert package.restored_chunk_ids_json
    assert package.parent_structure_node_ids_json
    assert package.citation_spans_json
    chunks = (package.package_json or {}).get("chunks") or []
    assert any(item.get("structure_path") for item in chunks)
    assert any(item.get("structure_nodes") for item in chunks)
    assert any(item.get("structure_path") for item in package.citation_spans_json)
    assert package.token_budget > 0
    package_payload = get_context_package(db_session, package.id)
    assert package_payload is not None
    assert package_payload["package_hash"]
    assert package_payload["contexts"]
    assert package_payload["citation_spans"]
    assert package_payload["graph_expansion_paths"]

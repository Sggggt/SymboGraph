from __future__ import annotations

import json

import pytest


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
        ContextGraphState,
        FineCluster,
        FineClusterEdge,
        FineClusterMembership,
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
    assert db_session.scalar(select(func.count(FineCluster.id)).where(FineCluster.knowledge_base_id == kb.id)) >= 1
    chunks_with_rq = db_session.scalars(select(Chunk).where(Chunk.knowledge_base_id == kb.id, Chunk.rq_residual_norm.is_not(None))).all()
    assert chunks_with_rq
    assert all(chunk.rq_path for chunk in chunks_with_rq)
    rq_clusters = db_session.scalars(select(FineCluster).where(FineCluster.knowledge_base_id == kb.id, FineCluster.rq_level.is_not(None))).all()
    assert rq_clusters
    rq_memberships = [
        row
        for row in db_session.scalars(select(FineClusterMembership).where(FineClusterMembership.residual_norm.is_not(None))).all()
        if row.rq_path
    ]
    assert rq_memberships
    assert all(row.rq_path for row in rq_memberships)
    edge_types = set(
        db_session.scalars(
            select(ChunkRelationEdge.edge_type).where(
                ChunkRelationEdge.knowledge_base_id == kb.id,
                ChunkRelationEdge.edge_type.in_(["rq_hierarchy_near", "rq_prefix_sibling", "rq_residual_near"]),
            )
        ).all()
    )
    assert {"rq_hierarchy_near", "rq_prefix_sibling", "rq_residual_near"}.issubset(edge_types)
    rq_edge = db_session.scalar(select(ChunkRelationEdge).where(ChunkRelationEdge.knowledge_base_id == kb.id, ChunkRelationEdge.edge_type.like("rq_%")))
    assert rq_edge is not None
    assert "lcp_depth" in (rq_edge.features_json or {})
    assert "residual_distance" in (rq_edge.features_json or {})
    assert rq_edge.source_algorithm == "rq_kmeans"
    assert rq_edge.protocol_version
    assert rq_edge.graph_state_hash
    rq_cluster_edge_types = set(
        db_session.scalars(
            select(FineClusterEdge.edge_type)
            .join(FineCluster, FineClusterEdge.source_cluster_id == FineCluster.id)
            .where(FineCluster.knowledge_base_id == kb.id, FineClusterEdge.edge_type.like("rq_%"))
        ).all()
    )
    assert {"rq_parent_child", "rq_sibling", "rq_centroid_near", "rq_overlap_bridge"}.issubset(rq_cluster_edge_types)
    graph_payload = graph_layer_payload(db_session, kb.id, "chunk-relation")
    assert graph_payload["full_counts"]["nodes"] >= graph_payload["sampled_counts"]["nodes"]
    assert graph_payload["full_counts"]["edges"] >= graph_payload["sampled_counts"]["edges"]
    assert graph_payload["edge_counts"]["full"] == graph_payload["full_counts"]["edges"]
    assert graph_payload["grounding"]["mid_grounded_rate"] == 1.0
    assert graph_payload["grounding"]["coarse_grounded_rate"] == 1.0
    assert any(node.get("category") == "rq_prefix" for node in graph_payload["nodes"])
    assert any(edge.get("category") == "rq_membership" for edge in graph_payload["edges"])
    assert any(str(edge.get("category", "")).startswith("rq_") for edge in graph_payload["edges"])
    assert db_session.scalar(select(func.count(MidConcept.id)).where(MidConcept.knowledge_base_id == kb.id)) >= 1
    mid_state = db_session.scalar(select(MidConceptState).where(MidConceptState.knowledge_base_id == kb.id, MidConceptState.state == "active"))
    assert mid_state is not None
    assert (mid_state.stats_json or {})["llm_batches"] <= 4
    assert (mid_state.stats_json or {})["selected_fine_clusters"] <= (mid_state.stats_json or {})["fine_cluster_candidates"]
    assert db_session.scalar(select(func.count(CoarseConcept.id)).where(CoarseConcept.knowledge_base_id == kb.id)) >= 1
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
    assert any((item.get("metadata") or {}).get("rq") for item in result.results)
    assert result.trace.id
    assert db_session.scalar(select(func.count(GraphRetrievalStep.id)).where(GraphRetrievalStep.retrieval_trace_id == result.trace.id)) >= 4
    fine_step = db_session.scalar(
        select(GraphRetrievalStep).where(GraphRetrievalStep.retrieval_trace_id == result.trace.id, GraphRetrievalStep.layer == "fine")
    )
    assert fine_step is not None
    assert "query_rq_path" in (fine_step.input_json or {})
    assert any("lcp_depth" in candidate for candidate in ((fine_step.output_json or {}).get("candidate_rq") or {}).values())
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

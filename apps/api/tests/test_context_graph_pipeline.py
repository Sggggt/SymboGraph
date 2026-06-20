from __future__ import annotations

from collections import Counter
import json
from types import SimpleNamespace

import pytest


def test_structure_edges_are_rejected_from_active_relation_graph():
    from app.services.context_graph import relation_edge_source_algorithm

    with pytest.raises(RuntimeError, match="Structure-derived relation edges are not allowed"):
        relation_edge_source_algorithm("same_page_region")


def test_entry_seed_calibration_prevents_zero_distance_route_seeds():
    from app.services.context_graph import calibrated_entry_seed_strength, distance_from_strength

    assert calibrated_entry_seed_strength(1.0, "dense_entry") == pytest.approx(0.97)
    assert calibrated_entry_seed_strength(1.0, "mid_drilldown_entry") == pytest.approx(0.82)
    assert calibrated_entry_seed_strength(1.0, "coarse_to_mid_drilldown_entry") == pytest.approx(0.72)
    assert distance_from_strength(calibrated_entry_seed_strength(1.0, "mid_drilldown_entry")) > 0.0


def test_concept_searchable_text_uses_successful_i18n_only():
    from app.services.context_graph import concept_searchable_text

    concept = SimpleNamespace(
        canonical_label="Bayesian Regression",
        definition="Regression with Bayesian posterior inference.",
        summary="Bayesian regression summary.",
        scope_note="Course-level model concept.",
        aliases_json=["Bayesian linear model"],
        display_terms_json=[],
        llm_audit_json={
            "concept_i18n": {
                "status": "ok",
                "label_i18n": {"zh": "贝叶斯回归", "en": "Bayesian Regression"},
                "definition_i18n": {"zh": "使用后验推断的回归模型。", "en": "Regression with posterior inference."},
                "summary_i18n": {"zh": "贝叶斯回归摘要。", "en": "Bayesian regression summary."},
                "scope_note_i18n": {"zh": "课程模型概念。", "en": "Course model concept."},
                "aliases_i18n": {"zh": ["贝叶斯线性模型"], "en": ["Bayesian linear model"]},
                "search_terms_i18n": {"zh": ["后验", "回归"], "en": ["posterior", "regression"]},
            }
        },
    )

    disabled_searchable = concept_searchable_text(concept, include_i18n=False)
    assert "贝叶斯回归" not in disabled_searchable
    assert "Bayesian Regression" in disabled_searchable

    searchable = concept_searchable_text(concept, include_i18n=True)
    assert "贝叶斯回归" in searchable
    assert "后验" in searchable

    concept.llm_audit_json["concept_i18n"]["status"] = "original_text_fallback"
    fallback_searchable = concept_searchable_text(concept, include_i18n=True)
    assert "贝叶斯回归" not in fallback_searchable
    assert "Bayesian Regression" in fallback_searchable


@pytest.mark.asyncio
async def test_context_graph_pipeline_builds_all_layers(db_session, populated_context_graph):
    from sqlalchemy import func, select

    from app.models import (
        Chunk,
        ChunkContextText,
        ChunkRelationEdge,
        ChunkRelationGraphState,
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
    relation_state = db_session.scalar(select(ChunkRelationGraphState).where(ChunkRelationGraphState.knowledge_base_id == kb.id, ChunkRelationGraphState.state == "active"))
    assert relation_state is not None
    assert relation_state.graph_operating_point_hash
    assert relation_state.edge_type_calibration_protocol_hash
    assert (relation_state.diagnostics_json or {}).get("rq_pair_edges_active") is False
    relation_edges = db_session.scalars(select(ChunkRelationEdge).where(ChunkRelationEdge.knowledge_base_id == kb.id)).all()
    assert relation_edges
    assert {edge.edge_type for edge in relation_edges}.issubset({"dense_semantic", "dense_cross_document_bridge", "dense_cross_language_bridge"})
    assert (relation_state.diagnostics_json or {}).get("accepted_edge_types") == dict(Counter(edge.edge_type for edge in relation_edges))
    assert not any(edge.edge_type.startswith("rq_") for edge in relation_edges)
    assert all(edge.source_algorithm == "dense_embedding" for edge in relation_edges)
    assert all(edge.protocol_version and edge.graph_state_hash and edge.edge_distance_protocol_hash for edge in relation_edges)
    assert all(edge.distance is not None and edge.raw_strength is not None for edge in relation_edges)
    assert all(edge.is_cross_document is not None and edge.is_cross_language is not None for edge in relation_edges)
    assert all(prefix.parent_rq_prefix_id or int(prefix.rq_level or 0) == 1 for prefix in rq_prefixes)
    graph_payload = graph_layer_payload(db_session, kb.id, "chunk-relation")
    assert graph_payload["full_counts"]["nodes"] >= graph_payload["sampled_counts"]["nodes"]
    assert graph_payload["full_counts"]["edges"] >= graph_payload["sampled_counts"]["edges"]
    assert graph_payload["edge_counts"]["full"] == graph_payload["full_counts"]["edges"]
    assert graph_payload["grounding"]["mid_grounded_rate"] == 1.0
    assert graph_payload["grounding"]["coarse_grounded_rate"] == 1.0
    assert any(node.get("category") == "rq_prefix" for node in graph_payload["nodes"])
    assert any(edge.get("category") == "rq_membership" for edge in graph_payload["edges"])
    assert not any(str(edge.get("category", "")).startswith("rq_") and edge.get("category") != "rq_membership" for edge in graph_payload["edges"])
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
    assert result.audit["stage_queue_count"] >= 0
    assert result.audit["mid_topk_selected"] >= 0
    assert result.audit["chunk_topk_selected"] >= 0
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
    assert result.trace.stage_queues_json
    assert result.trace.candidate_pools_json
    assert result.trace.topk_selection_json
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
    assert seed_step.action_type == "select_seeds_from_mid_rq_membership"
    assert seed_step.selected_topk_ids_json
    assert "query_rq_path" in (seed_step.input_json or {})
    assert (seed_step.output_json or {}).get("candidate_count") is not None
    package = build_context_package(db_session, knowledge_base_id=kb.id, query="Markov blanket", trace=result.trace, results=result.results)
    db_session.commit()
    structure_step = db_session.scalar(
        select(GraphRetrievalStep).where(GraphRetrievalStep.retrieval_trace_id == result.trace.id, GraphRetrievalStep.layer == "structure")
    )
    assert structure_step is not None
    assert structure_step.action_type == "restore_context_package"
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

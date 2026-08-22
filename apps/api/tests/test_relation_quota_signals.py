from __future__ import annotations

from types import SimpleNamespace

import pytest


def _add_document_chunks(
    db_session,
    knowledge_base_id: str,
    *,
    suffix: str,
    chunk_ids: list[str],
    language: str = "en",
    text: str | None = None,
):
    from app.models import Chunk, Document, DocumentVersion
    from app.services.chunking import text_hash

    document = Document(
        knowledge_base_id=knowledge_base_id,
        title=f"Quota fixture {suffix}",
        source_path=f"quota-{suffix}.md",
        source_type="markdown",
        language=language,
        checksum=f"quota-checksum-{suffix}",
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=document.checksum,
        storage_path=document.source_path,
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    chunk_text = text or ("evidence " * 300).strip()
    chunks = []
    for index, chunk_id in enumerate(chunk_ids):
        chunk = Chunk(
            id=chunk_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document.id,
            document_version_id=version.id,
            chunk_version=1,
            chunk_index=index,
            token_start=index * 400,
            token_end=index * 400 + 300,
            char_start=index * 4000,
            char_end=index * 4000 + len(chunk_text),
            text=chunk_text,
            text_hash=text_hash(chunk_text),
            section_path=None,
            state="active",
        )
        db_session.add(chunk)
        chunks.append(chunk)
    db_session.flush()
    return document, version, chunks


def _signal_inputs(chunks, documents, scored_by_source=None):
    return {
        "vectors": {str(chunk.id): [1.0, 0.0] for chunk in chunks},
        "scored_by_source": scored_by_source
        or {str(chunk.id): [] for chunk in chunks},
        "documents": {str(document.id): document for document in documents},
        "type_thresholds": {
            "dense_semantic": 0.3,
            "dense_cross_document_bridge": 0.36,
            "dense_cross_language_bridge": 0.34,
        },
    }


def _quota(value: float) -> int:
    from app.services.context_graph import relation_quota_card

    return int(
        relation_quota_card(
            lower=0,
            upper=10,
            signal=value,
            signal_role="fixture",
        )["value"]
    )


def test_structure_summary_is_exact_without_materializing_mapping_orm_rows(
    db_session,
    sample_knowledge_base,
):
    from app.models import ChunkStructureMapping, ChunkStructureNode
    from app.services.context_graph import _relation_quota_structure_summaries

    document, version, chunks = _add_document_chunks(
        db_session,
        sample_knowledge_base.id,
        suffix="bounded-structure-summary",
        chunk_ids=["quota-bounded-structure-summary"],
    )
    chunk = chunks[0]
    expected_maximum = 0.0
    for index in range(64):
        node_type = "paragraph" if index < 63 else "table"
        mapping_weight = (index + 1) / 64.0
        expected_maximum = max(expected_maximum, mapping_weight)
        node = ChunkStructureNode(
            knowledge_base_id=sample_knowledge_base.id,
            document_id=document.id,
            document_version_id=version.id,
            node_type=node_type,
            depth=1,
            title=f"node-{index}",
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            path=f"root/{index}",
        )
        db_session.add(node)
        db_session.flush()
        db_session.add(
            ChunkStructureMapping(
                chunk_id=chunk.id,
                structure_node_id=node.id,
                document_version_id=version.id,
                overlap_chars=0,
                overlap_tokens=0,
                coverage_ratio=0.0,
                span_overlap=0.0,
                mapping_weight=mapping_weight,
                mapping_role="context",
            )
        )
    db_session.flush()
    chunk_id = str(chunk.id)
    db_session.expunge_all()

    summaries = _relation_quota_structure_summaries(db_session, [chunk_id])

    assert summaries[chunk_id]["mapping_count"] == 64
    assert summaries[chunk_id]["mapping_strength"] == expected_maximum
    assert summaries[chunk_id]["mapped_node_types"] == {"paragraph", "table"}
    assert not any(
        isinstance(value, (ChunkStructureMapping, ChunkStructureNode))
        for value in db_session.identity_map.values()
    )


def test_text_hash_integrity_distinguishes_match_mismatch_and_missing_and_propagates(
    db_session,
    sample_knowledge_base,
):
    from app.services.chunking import CHUNK_TEXT_HASH_PROTOCOL_VERSION, text_hash
    from app.services.context_graph import (
        CHUNK_NODE_QUALITY_PROTOCOL_VERSION,
        RELATION_PROTOCOL_VERSION,
        RELATION_QUOTA_PROTOCOL_VERSION,
        dense_graph_operating_point,
        relation_edge_candidates,
        relation_quota_node_signals,
    )

    document, _version, chunks = _add_document_chunks(
        db_session,
        sample_knowledge_base.id,
        suffix="text-hash-integrity",
        chunk_ids=[
            "quota-hash-correct",
            "quota-hash-mismatch",
            "quota-hash-missing",
            "quota-hash-inactive-correct",
            "quota-hash-inactive-mismatch",
        ],
    )
    correct, mismatch, missing, inactive_correct, inactive_mismatch = chunks
    mismatch.text_hash = "f" * 64
    assert mismatch.text_hash != text_hash(mismatch.text)
    missing.text_hash = ""
    inactive_correct.state = "inactive"
    inactive_mismatch.state = "inactive"
    inactive_mismatch.text_hash = "e" * 64
    assert inactive_mismatch.text_hash != text_hash(inactive_mismatch.text)
    db_session.flush()

    vectors = {str(chunk.id): [1.0, 0.0] for chunk in chunks}
    signals, _diagnostics = relation_quota_node_signals(
        db_session,
        chunks,
        **_signal_inputs(chunks, [document]),
    )
    lifecycle_by_id = {
        chunk.id: signals[chunk.id]["node_quality"]["components"]["lifecycle_integrity"]
        for chunk in chunks
    }

    assert RELATION_PROTOCOL_VERSION == "dense_only_chunk_relation_graph_v9"
    assert CHUNK_NODE_QUALITY_PROTOCOL_VERSION == "chunk_node_quality_intrinsic_v2"
    assert RELATION_QUOTA_PROTOCOL_VERSION == "dynamic_knn_reverse_quota_signals_v3"
    assert lifecycle_by_id[correct.id]["value"] == 1.0
    assert lifecycle_by_id[correct.id]["diagnostics"]["text_hash_status"] == "match"
    assert lifecycle_by_id[correct.id]["diagnostics"]["text_hash_matches"] is True
    assert lifecycle_by_id[correct.id]["diagnostics"]["expected_text_hash"] == correct.text_hash
    assert lifecycle_by_id[mismatch.id]["value"] == 0.5
    assert lifecycle_by_id[mismatch.id]["diagnostics"]["text_hash_status"] == "mismatch"
    assert lifecycle_by_id[mismatch.id]["diagnostics"]["text_hash_matches"] is False
    assert lifecycle_by_id[mismatch.id]["diagnostics"]["stored_text_hash_present"] is True
    assert lifecycle_by_id[missing.id]["value"] == 0.5
    assert lifecycle_by_id[missing.id]["diagnostics"]["text_hash_status"] == "missing"
    assert lifecycle_by_id[missing.id]["diagnostics"]["text_hash_matches"] is False
    assert lifecycle_by_id[missing.id]["diagnostics"]["stored_text_hash_present"] is False
    assert lifecycle_by_id[inactive_correct.id]["value"] == 0.5
    assert lifecycle_by_id[inactive_correct.id]["diagnostics"]["state_is_active"] is False
    assert lifecycle_by_id[inactive_correct.id]["diagnostics"]["text_hash_status"] == "match"
    assert lifecycle_by_id[inactive_mismatch.id]["value"] == 0.0
    assert lifecycle_by_id[inactive_mismatch.id]["diagnostics"]["state_is_active"] is False
    assert lifecycle_by_id[inactive_mismatch.id]["diagnostics"]["text_hash_status"] == "mismatch"
    assert {
        lifecycle_by_id[chunk.id]["diagnostics"]["chunk_text_hash_protocol_version"]
        for chunk in chunks
    } == {CHUNK_TEXT_HASH_PROTOCOL_VERSION}

    correct_quality = signals[correct.id]["node_quality"]["value"]
    mismatch_quality = signals[mismatch.id]["node_quality"]["value"]
    missing_quality = signals[missing.id]["node_quality"]["value"]
    assert correct_quality > mismatch_quality == missing_quality
    for invalid in (mismatch, missing):
        assert signals[invalid.id]["out_evidence_mass"]["components"]["node_quality"]["value"] == signals[
            invalid.id
        ]["node_quality"]["value"]
        assert signals[invalid.id]["in_acceptance_capacity"]["components"]["node_quality"]["value"] == signals[
            invalid.id
        ]["node_quality"]["value"]
        assert signals[correct.id]["out_evidence_mass"]["value"] > signals[invalid.id]["out_evidence_mass"]["value"]
        assert signals[correct.id]["in_acceptance_capacity"]["value"] > signals[invalid.id]["in_acceptance_capacity"]["value"]

    operating_point = {
        **dense_graph_operating_point(),
        "dense_knn_k_min": 1,
        "dense_knn_k_max": 10,
        "dense_reverse_b_min_base": 1,
        "dense_reverse_b_max_base": 10,
        "dense_min_cosine": 0.1,
        "dense_strong_cosine": 0.9,
    }
    candidates, _candidate_diagnostics = relation_edge_candidates(
        db_session,
        chunks,
        vectors,
        operating_point,
    )
    candidate = next(
        item
        for item in candidates.values()
        if {item.source_chunk_id, item.target_chunk_id} == {correct.id, mismatch.id}
    )
    features = candidate.features_json
    directed_source_id = features["directed_source_chunk_id"]
    directed_target_id = features["directed_target_chunk_id"]
    quality_cards_by_id = {
        directed_source_id: features["source_node_quality_card"],
        directed_target_id: features["target_node_quality_card"],
    }
    expected_pair = round(
        (quality_cards_by_id[correct.id]["value"] + quality_cards_by_id[mismatch.id]["value"]) / 2.0,
        6,
    )

    assert quality_cards_by_id[correct.id]["components"]["lifecycle_integrity"]["diagnostics"][
        "text_hash_status"
    ] == "match"
    assert quality_cards_by_id[mismatch.id]["components"]["lifecycle_integrity"]["diagnostics"][
        "text_hash_status"
    ] == "mismatch"
    assert features["node_quality_pair"] == expected_pair
    assert features["raw_strength_components"]["node_quality_pair"] == expected_pair
    assert features["source_out_signal_card"]["components"]["node_quality"]["value"] == features[
        "source_node_quality_card"
    ]["value"]
    assert features["target_in_acceptance_signal_card"]["components"]["node_quality"]["value"] == features[
        "target_node_quality_card"
    ]["value"]
    assert features["source_out_quota_cards"]["base_dense_candidates"]["signal"] == features[
        "source_out_evidence_mass"
    ]
    assert features["target_inbound_quota_card"]["signal"] == features["target_in_acceptance_capacity"]


def test_structure_and_raw_span_evidence_raise_outgoing_quota_without_node_mass_reuse(
    db_session,
    sample_knowledge_base,
):
    from app.models import ChunkCoordinate, ChunkSpan, ChunkStructureMapping, ChunkStructureNode
    from app.services.context_graph import relation_quota_node_signals

    document, version, chunks = _add_document_chunks(
        db_session,
        sample_knowledge_base.id,
        suffix="structure",
        chunk_ids=["quota-rich-structure", "quota-poor-structure"],
    )
    rich, poor = chunks
    rich.previous_chunk_id = poor.id
    rich.next_chunk_id = poor.id
    db_session.add(
        ChunkSpan(
            chunk_id=rich.id,
            document_version_id=version.id,
            char_start=rich.char_start,
            char_end=rich.char_end,
            token_start=rich.token_start,
            token_end=rich.token_end,
            span_type="raw_text",
        )
    )
    db_session.add(
        ChunkCoordinate(
            chunk_id=rich.id,
            document_version_id=version.id,
            page_number=1,
            coordinate_system="parser_layout_v1",
            confidence=1.0,
        )
    )
    for index, node_type in enumerate(("heading", "paragraph", "table", "formula")):
        node = ChunkStructureNode(
            knowledge_base_id=sample_knowledge_base.id,
            document_id=document.id,
            document_version_id=version.id,
            node_type=node_type,
            depth=index,
            title=node_type,
            char_start=rich.char_start,
            char_end=rich.char_end,
            path=f"root/{node_type}",
        )
        db_session.add(node)
        db_session.flush()
        db_session.add(
            ChunkStructureMapping(
                chunk_id=rich.id,
                structure_node_id=node.id,
                document_version_id=version.id,
                overlap_chars=rich.char_end - rich.char_start,
                overlap_tokens=rich.token_end - rich.token_start,
                coverage_ratio=1.0,
                span_overlap=1.0,
                mapping_weight=1.0,
                mapping_role="primary",
            )
        )
    db_session.flush()

    scored = {
        str(rich.id): [(1.0, poor)],
        str(poor.id): [(1.0, rich)],
    }
    signals, diagnostics = relation_quota_node_signals(
        db_session,
        chunks,
        **_signal_inputs(chunks, [document], scored),
    )
    rich_signal = signals[str(rich.id)]
    poor_signal = signals[str(poor.id)]

    assert rich_signal["node_quality"]["value"] == poor_signal["node_quality"]["value"] == 1.0
    assert rich_signal["out_evidence_mass"]["components"]["structure_coverage"]["value"] == 1.0
    assert poor_signal["out_evidence_mass"]["components"]["structure_coverage"]["value"] == 0.0
    assert rich_signal["out_evidence_mass"]["components"]["span_citability"]["value"] == 1.0
    assert poor_signal["out_evidence_mass"]["components"]["span_citability"]["value"] == 0.55
    assert rich_signal["out_evidence_mass"]["value"] > poor_signal["out_evidence_mass"]["value"]
    assert _quota(rich_signal["out_evidence_mass"]["value"]) > _quota(
        poor_signal["out_evidence_mass"]["value"]
    )
    assert diagnostics["node_mass_reused_for_out_and_in"] is False
    assert diagnostics["quota_signals_used_as_query_relevance"] is False


def test_historical_rq_roles_are_diagnostics_only_for_bottom_edge_quota(
    db_session,
    sample_knowledge_base,
):
    from app.models import ChunkRelationGraphState, RQPrefix, RQPrefixMembership
    from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION, stable_hash
    from app.services.context_graph import (
        RELATION_PROTOCOL_VERSION,
        dense_graph_operating_point,
        edge_type_calibration_config,
        relation_quota_node_signals,
    )

    document, _version, chunks = _add_document_chunks(
        db_session,
        sample_knowledge_base.id,
        suffix="rq-roles",
        chunk_ids=["quota-rq-bridge", "quota-rq-boundary", "quota-rq-primary", "quota-rq-none"],
    )
    operating_point = dense_graph_operating_point()
    calibration_config = edge_type_calibration_config(operating_point)
    graph_state = ChunkRelationGraphState(
        knowledge_base_id=sample_knowledge_base.id,
        chunk_version=1,
        scope_hash=stable_hash([chunk.id for chunk in chunks]),
        state_hash=stable_hash("quota-rq-state"),
        graph_operating_point_hash=stable_hash(operating_point),
        graph_operating_point_json=operating_point,
        embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
        relation_protocol_version=RELATION_PROTOCOL_VERSION,
        active_chunk_ids_json=[chunk.id for chunk in chunks],
        diagnostics_json={
            "edge_type_calibration_config_hash": calibration_config["config_hash"],
        },
        state="active",
    )
    db_session.add(graph_state)
    db_session.flush()
    prefix = RQPrefix(
        graph_state_id=graph_state.id,
        knowledge_base_id=sample_knowledge_base.id,
        rq_prefix_key="rq:quota:0",
        label="quota role fixture",
        rq_level=0,
        rq_path_prefix=[0],
        support_chunk_ids_json=[chunk.id for chunk in chunks[:3]],
    )
    db_session.add(prefix)
    db_session.flush()
    roles = {
        "quota-rq-bridge": "bridge_member",
        "quota-rq-boundary": "boundary_member",
        "quota-rq-primary": "primary_member",
    }
    for chunk in chunks[:3]:
        db_session.add(
            RQPrefixMembership(
                rq_prefix_id=prefix.id,
                chunk_id=chunk.id,
                membership_score=1.0,
                membership_role=roles[chunk.id],
                membership_reason="counterexample_fixture",
                membership_entropy=0.0,
                rq_path=[0],
                rank=1,
            )
        )
    db_session.flush()

    signals, diagnostics = relation_quota_node_signals(
        db_session,
        chunks,
        **_signal_inputs(chunks, [document]),
    )
    assert len({card["out_evidence_mass"]["value"] for card in signals.values()}) == 1
    assert len(
        {
            card["in_acceptance_capacity"]["value"]
            for card in signals.values()
        }
    ) == 1
    assert all(
        "rq_membership_coverage"
        not in card["in_acceptance_capacity"]["components"]
        for card in signals.values()
    )
    assert all(
        card["in_acceptance_capacity"]["components"]["bridge_coverage"][
            "available"
        ]
        is False
        for card in signals.values()
    )
    assert diagnostics["rq_source_graph_state_ids"] == []
    assert diagnostics["historical_rq_state_ids"] == [graph_state.id]
    assert diagnostics["historical_rq_prior_used_for_quota"] is False
    assert diagnostics["historical_rq_prior_used_for_edge_gate"] is False
    assert diagnostics["historical_rq_prior_used_for_raw_strength"] is False


def test_pre_integrity_relation_state_is_rejected_as_rq_prior(
    db_session,
    sample_knowledge_base,
):
    from app.models import ChunkRelationGraphState, RQPrefix, RQPrefixMembership
    from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION, stable_hash
    from app.services.context_graph import (
        dense_graph_operating_point,
        relation_operating_point_prior,
        relation_quota_node_signals,
    )

    document, _version, chunks = _add_document_chunks(
        db_session,
        sample_knowledge_base.id,
        suffix="legacy-rq-prior",
        chunk_ids=["quota-legacy-prior", "quota-legacy-peer"],
    )
    legacy_operating_point = {
        **dense_graph_operating_point(),
        "raw_strength_protocol_version": "dense_relation_raw_strength_v2",
        "chunk_node_quality_protocol": "chunk_node_quality_intrinsic_v1",
        "out_evidence_mass_protocol": "relation_out_evidence_mass_v1",
        "in_acceptance_capacity_protocol": "relation_in_acceptance_capacity_v1",
        "relation_quota_protocol": "dynamic_knn_reverse_quota_signals_v2",
    }
    legacy_state = ChunkRelationGraphState(
        knowledge_base_id=sample_knowledge_base.id,
        chunk_version=1,
        scope_hash=stable_hash([chunk.id for chunk in chunks]),
        state_hash=stable_hash("legacy-rq-prior-state"),
        graph_operating_point_hash=stable_hash(legacy_operating_point),
        graph_operating_point_json=legacy_operating_point,
        embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
        relation_protocol_version="dense_only_chunk_relation_graph_v6",
        active_chunk_ids_json=[chunk.id for chunk in chunks],
        state="active",
    )
    db_session.add(legacy_state)
    db_session.flush()
    prefix = RQPrefix(
        graph_state_id=legacy_state.id,
        knowledge_base_id=sample_knowledge_base.id,
        rq_prefix_key="rq:legacy-prior:0",
        label="legacy prior",
        rq_level=0,
        rq_path_prefix=[0],
        support_chunk_ids_json=[chunks[0].id],
    )
    db_session.add(prefix)
    db_session.flush()
    db_session.add(
        RQPrefixMembership(
            rq_prefix_id=prefix.id,
            chunk_id=chunks[0].id,
            membership_score=1.0,
            membership_role="primary_member",
            membership_reason="legacy_protocol_fixture",
            membership_entropy=0.0,
            rq_path=[0],
            rank=1,
        )
    )
    db_session.flush()

    prior, prior_audit = relation_operating_point_prior(legacy_state)
    assert prior is None
    assert prior_audit["status"] == "rejected"
    assert prior_audit["reused"] is False
    assert "relation_protocol_version_mismatch" in prior_audit["reasons"]

    signals, diagnostics = relation_quota_node_signals(
        db_session,
        chunks,
        **_signal_inputs(chunks, [document]),
    )

    assert (
        "rq_membership_coverage"
        not in signals[chunks[0].id]["in_acceptance_capacity"]["components"]
    )
    assert diagnostics["rq_source_graph_state_ids"] == []
    assert diagnostics["historical_rq_state_ids"] == []
    assert diagnostics["historical_rq_prior_used_for_quota"] is False
    assert diagnostics["rq_rejected_incompatible_state_count"] == 1
    rejection = diagnostics["rq_rejected_incompatible_states"][0]
    assert rejection["state_id"] == legacy_state.id
    assert "relation_protocol_version_mismatch" in rejection["reasons"]
    assert "operating_point_chunk_node_quality_protocol_mismatch" in rejection["reasons"]


def test_hub_pressure_reduces_inbound_capacity_but_not_outgoing_evidence_mass(
    db_session,
    sample_knowledge_base,
):
    from app.services.context_graph import relation_quota_node_signals

    document, _version, chunks = _add_document_chunks(
        db_session,
        sample_knowledge_base.id,
        suffix="hub",
        chunk_ids=["quota-hub", "quota-free", "quota-source-a", "quota-source-b", "quota-source-c"],
    )
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    scored = {chunk.id: [] for chunk in chunks}
    for source_id in ("quota-source-a", "quota-source-b", "quota-source-c"):
        scored[source_id] = [(0.9, chunk_by_id["quota-hub"])]

    signals, _diagnostics = relation_quota_node_signals(
        db_session,
        chunks,
        **_signal_inputs(chunks, [document], scored),
    )
    hub = signals["quota-hub"]
    free = signals["quota-free"]
    hub_component = hub["in_acceptance_capacity"]["components"]["hub_headroom"]
    free_component = free["in_acceptance_capacity"]["components"]["hub_headroom"]

    assert hub_component["diagnostics"]["inbound_pressure"] == 3
    assert hub_component["value"] == 0.0
    assert free_component["value"] == 1.0
    assert hub["out_evidence_mass"]["value"] == free["out_evidence_mass"]["value"]
    assert _quota(free["in_acceptance_capacity"]["value"]) > _quota(
        hub["in_acceptance_capacity"]["value"]
    )


def test_relation_candidate_trace_carries_independent_signal_and_quota_cards(
    db_session,
    sample_knowledge_base,
):
    from app.services import context_graph

    document, _version, chunks = _add_document_chunks(
        db_session,
        sample_knowledge_base.id,
        suffix="candidate",
        chunk_ids=["quota-candidate-a", "quota-candidate-b"],
    )
    vectors = {chunk.id: [1.0, 0.0] for chunk in chunks}
    operating_point = {
        **context_graph.dense_graph_operating_point(),
        "dense_knn_k_min": 1,
        "dense_knn_k_max": 10,
        "dense_reverse_b_min_base": 1,
        "dense_reverse_b_max_base": 10,
        "dense_min_cosine": 0.1,
        "dense_strong_cosine": 0.9,
    }
    candidates, diagnostics = context_graph.relation_edge_candidates(
        db_session,
        chunks,
        vectors,
        operating_point,
    )
    candidate = next(iter(candidates.values()))
    features = candidate.features_json

    assert features["source_out_signal_card"]["protocol_hash"] == context_graph.out_evidence_mass_protocol_hash()
    assert features["target_in_acceptance_signal_card"]["protocol_hash"] == context_graph.in_acceptance_capacity_protocol_hash()
    assert features["source_out_quota_cards"]["base_dense_candidates"]["signal_role"] == "out_evidence_mass"
    assert features["target_inbound_quota_card"]["signal_role"] == "in_acceptance_capacity"
    assert features["node_quality_pair"] == pytest.approx(
        (
            features["source_node_quality_card"]["value"]
            + features["target_node_quality_card"]["value"]
        )
        / 2.0
    )
    assert diagnostics["relation_quota_protocol_hash"] == context_graph.relation_quota_protocol_hash()
    assert diagnostics["legacy_node_mass_active"] is False

    trace = context_graph.relation_edge_rank_trace_payload(
        SimpleNamespace(
            id="quota-edge",
            source_chunk_id=candidate.source_chunk_id,
            target_chunk_id=candidate.target_chunk_id,
            edge_type=candidate.edge_type,
            raw_strength=candidate.raw_strength,
            weight=candidate.calibrated_strength,
            distance=candidate.distance,
            features_json=features,
        )
    )
    assert trace["source_out_signal_card"]["card_hash"]
    assert trace["target_in_acceptance_signal_card"]["card_hash"]
    assert trace["relation_quota_protocol_hash"] == context_graph.relation_quota_protocol_hash()
    assert trace["quota_signal_scope_hash"] == diagnostics["relation_quota_signals"]["signal_scope_hash"]


def test_relation_graph_state_hashes_the_quota_signal_protocol(
    db_session,
    sample_knowledge_base,
):
    from app.core.config import get_settings
    from app.models import VectorRecord
    from app.services.chunking import (
        CONTEXTUAL_TEXT_HASH_PROTOCOL_VERSION,
        CURRENT_EMBEDDING_TEXT_VERSION,
        LOCAL_CONTEXT_HINT_PROTOCOL_VERSION,
        stable_hash,
    )
    from app.services.context_graph import (
        CHUNK_SCHEMA_VERSION,
        QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
        QDRANT_OUTBOX_PROTOCOL_VERSION,
        QDRANT_VECTOR_DISTANCE_METRIC,
        VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION,
        build_chunk_relation_graph,
        dense_graph_operating_point,
        qdrant_collection_identity_digest,
        qdrant_collection_name,
        relation_operating_point_prior,
        relation_quota_protocol_hash,
        vector_payload_hash,
    )
    from app.services.vector_shadow_lifecycle import (
        ensure_active_vector_runtime_target,
        vector_runtime_diagnostics,
    )
    _document, _version, chunks = _add_document_chunks(
        db_session,
        sample_knowledge_base.id,
        suffix="state",
        chunk_ids=["quota-state-a", "quota-state-b"],
    )
    vector_runtime_target = ensure_active_vector_runtime_target(
        db_session,
        sample_knowledge_base.id,
    )
    settings = get_settings()
    for index, chunk in enumerate(chunks):
        vector = [1.0, float(index) * 0.1] + [
            0.0
        ] * (settings.embedding_dimensions - 2)
        collection_name = qdrant_collection_name(
            embedding_model=settings.embedding_model,
            embedding_dimension=len(vector),
            embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
            chunk_schema_version=CHUNK_SCHEMA_VERSION,
        )
        collection_identity_digest = qdrant_collection_identity_digest(
            embedding_model=settings.embedding_model,
            embedding_dimension=len(vector),
            embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
            chunk_schema_version=CHUNK_SCHEMA_VERSION,
        )
        context_hash = stable_hash({"chunk_id": chunk.id, "context": "quota-state"})
        local_hint_hash = stable_hash({"chunk_id": chunk.id, "hint": "quota-state"})
        payload_hash = vector_payload_hash(
            vector=vector,
            chunk_id=chunk.id,
            embedding_model=settings.embedding_model,
            embedding_dimension=len(vector),
            vector_distance_metric=QDRANT_VECTOR_DISTANCE_METRIC,
            embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
            chunk_schema_version=CHUNK_SCHEMA_VERSION,
            context_hash_protocol_version=CONTEXTUAL_TEXT_HASH_PROTOCOL_VERSION,
            context_hash=context_hash,
            local_hint_protocol_version=LOCAL_CONTEXT_HINT_PROTOCOL_VERSION,
            local_hint_hash=local_hint_hash,
            collection_identity_protocol_version=QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
            collection_identity_digest=collection_identity_digest,
        )
        db_session.add(
            VectorRecord(
                knowledge_base_id=sample_knowledge_base.id,
                chunk_id=chunk.id,
                qdrant_point_id=f"quota-point-{index}",
                collection_name=collection_name,
                embedding_model=settings.embedding_model,
                embedding_dimension=len(vector),
                embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
                payload_hash=payload_hash,
                vector_status="ready",
                diagnostics_json={
                    **vector_runtime_diagnostics(vector_runtime_target),
                    "embedding_vector": vector,
                    "embedding_dimension": len(vector),
                    "vector_distance_metric": QDRANT_VECTOR_DISTANCE_METRIC,
                    "chunk_schema_version": CHUNK_SCHEMA_VERSION,
                    "context_hash_protocol_version": CONTEXTUAL_TEXT_HASH_PROTOCOL_VERSION,
                    "context_hash": context_hash,
                    "local_hint_protocol_version": LOCAL_CONTEXT_HINT_PROTOCOL_VERSION,
                    "local_hint_hash": local_hint_hash,
                    "collection_identity_protocol_version": QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
                    "collection_identity_digest": collection_identity_digest,
                    "vector_payload_hash_protocol": VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION,
                    "qdrant_write_protocol_version": QDRANT_OUTBOX_PROTOCOL_VERSION,
                    "qdrant_write_intent_id": f"quota-intent-{index}",
                },
            )
        )
    db_session.flush()

    graph_state = build_chunk_relation_graph(
        db_session,
        sample_knowledge_base.id,
        chunks,
        operating_point=dense_graph_operating_point(),
    )

    assert graph_state.graph_operating_point_json["relation_quota_protocol_hash"] == relation_quota_protocol_hash()
    assert graph_state.graph_operating_point_hash == stable_hash(graph_state.graph_operating_point_json)
    assert graph_state.diagnostics_json["relation_quota_protocol_hash"] == relation_quota_protocol_hash()
    assert graph_state.diagnostics_json["relation_quota_signals"]["signal_scope_hash"]
    prior, prior_audit = relation_operating_point_prior(graph_state)
    assert prior == graph_state.graph_operating_point_json
    assert prior_audit["status"] == "accepted"
    assert prior_audit["compatible"] is True
    assert prior_audit["reused"] is True


def test_quota_protocol_propagates_to_operating_point_tpe_cache_and_preflight(no_fallback_env):
    from app.services.auto_tpe import (
        _protocol_hash,
        _relation_quota_protocol_diagnostics,
        normalize_theta,
        preflight_theta,
    )
    from app.services.context_graph import (
        context_graph_cache_key_components,
        dense_graph_operating_point,
        in_acceptance_capacity_protocol_hash,
        out_evidence_mass_protocol_hash,
        relation_quota_protocol_hash,
        relation_quota_signal_config,
    )

    operating_point = dense_graph_operating_point()
    normalized = normalize_theta(operating_point)
    cache = context_graph_cache_key_components(
        knowledge_base_id="kb-quota-protocol",
        query="quota audit",
        filters={},
        context_state=None,
        retrieval_mode="layered_context_graph",
        conversation_state_scope_hash="a" * 64,
        profile_hash_value="profile-quota-test",
    )

    assert normalized["out_evidence_mass_protocol_hash"] == out_evidence_mass_protocol_hash()
    assert normalized["in_acceptance_capacity_protocol_hash"] == in_acceptance_capacity_protocol_hash()
    assert normalized["relation_quota_protocol_hash"] == relation_quota_protocol_hash()
    assert cache["out_evidence_mass_protocol_hash"] == out_evidence_mass_protocol_hash()
    assert cache["in_acceptance_capacity_protocol_hash"] == in_acceptance_capacity_protocol_hash()
    assert cache["relation_quota_protocol_hash"] == relation_quota_protocol_hash()
    assert _relation_quota_protocol_diagnostics()["relation_quota_protocol_hash"] == relation_quota_protocol_hash()
    assert _protocol_hash()

    invalid = {**operating_point, "relation_quota_protocol_hash": "stale-quota-protocol"}
    assert any(reason.startswith("invalid_relation_quota_protocol:") for reason in preflight_theta(invalid))
    with pytest.raises(ValueError, match="relation_quota_protocol_hash"):
        relation_quota_signal_config(invalid)

    legacy_pre_integrity_fix = {
        **operating_point,
        "chunk_node_quality_protocol": "chunk_node_quality_intrinsic_v1",
        "out_evidence_mass_protocol": "relation_out_evidence_mass_v1",
        "in_acceptance_capacity_protocol": "relation_in_acceptance_capacity_v1",
        "relation_quota_protocol": "dynamic_knn_reverse_quota_signals_v2",
    }
    assert any(
        reason.startswith("invalid_relation_quota_protocol:")
        for reason in preflight_theta(legacy_pre_integrity_fix)
    )
    with pytest.raises(ValueError, match="chunk_node_quality_protocol"):
        relation_quota_signal_config(legacy_pre_integrity_fix)


@pytest.mark.parametrize(
    ("lower", "upper", "signal", "expected"),
    [(0, 10, 0.0, 0), (0, 10, 1.0, 4), (3, 5, 1.0, 5), (3, 5, -1.0, 3)],
)
def test_relation_quota_card_has_bounded_logarithmic_semantics(lower, upper, signal, expected):
    from app.services.context_graph import relation_quota_card

    card = relation_quota_card(
        lower=lower,
        upper=upper,
        signal=signal,
        signal_role="fixture",
    )
    assert card["value"] == expected
    assert card["query_relevance"] is False
    assert card["protocol_hash"]


def test_relation_quota_card_rejects_invalid_bounds():
    from app.services.context_graph import relation_quota_card

    with pytest.raises(ValueError, match="quota bounds"):
        relation_quota_card(lower=2, upper=1, signal=0.5, signal_role="fixture")

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _add_equal_quality_chunks(db_session, knowledge_base_id: str, chunk_ids: list[str]):
    from app.models import Chunk, Document, DocumentVersion
    from app.services.chunking import text_hash

    document = Document(
        knowledge_base_id=knowledge_base_id,
        title="Rank score fixture",
        source_path="rank-score.md",
        source_type="markdown",
        language="en",
        checksum="rank-score-checksum",
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
    chunks = []
    text = "Equal quality relation candidate"
    for index, chunk_id in enumerate(chunk_ids):
        chunk = Chunk(
            id=chunk_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document.id,
            document_version_id=version.id,
            chunk_version=1,
            chunk_index=index,
            token_start=index * 4,
            token_end=(index + 1) * 4,
            char_start=index * 100,
            char_end=index * 100 + len(text),
            text=text,
            text_hash=text_hash(text),
            section_path="Rank score fixture",
            state="active",
        )
        db_session.add(chunk)
        chunks.append(chunk)
    db_session.flush()
    return chunks


def _operating_point():
    from app.services.context_graph import dense_graph_operating_point

    return {
        **dense_graph_operating_point(),
        "dense_knn_k_min": 10,
        "dense_knn_k_max": 10,
        "dense_reverse_b_min_base": 10,
        "dense_reverse_b_max_base": 10,
        "dense_reverse_b_min_doc": 10,
        "dense_reverse_b_max_doc": 10,
        "dense_reverse_b_min_lang": 10,
        "dense_reverse_b_max_lang": 10,
        "cross_doc_out_quota_min": 10,
        "cross_doc_out_quota_max": 10,
        "cross_language_out_quota_min": 10,
        "cross_language_out_quota_max": 10,
        "dense_min_cosine": 0.5,
        "cross_doc_min_cosine": 0.5,
        "cross_language_min_cosine": 0.5,
        "dense_strong_cosine": 0.95,
    }


def _directional_contribution(candidates, source_id: str, target_id: str):
    key = (*sorted((source_id, target_id)), "dense_semantic")
    candidate = candidates[key]
    return candidate, next(
        item
        for item in candidate.features_json["directional_contributions"]
        if item["source_chunk_id"] == source_id and item["target_chunk_id"] == target_id
    )


@pytest.mark.parametrize(
    ("rank", "candidate_count", "expected"),
    [
        (1, 1, 1.0),
        (1, 4, 1.0),
        (2, 4, 0.75),
        (4, 4, 0.25),
    ],
)
def test_channel_rank_score_boundaries(rank, candidate_count, expected):
    from app.services.context_graph import channel_rank_score

    assert channel_rank_score(rank=rank, candidate_count=candidate_count) == expected


@pytest.mark.parametrize(
    ("rank", "candidate_count"),
    [(1, 0), (0, 1), (2, 1)],
)
def test_channel_rank_score_rejects_invalid_coordinates(rank, candidate_count):
    from app.services.context_graph import channel_rank_score

    with pytest.raises(ValueError):
        channel_rank_score(rank=rank, candidate_count=candidate_count)


def test_same_cosine_with_different_channel_rank_changes_only_rank_strength_component(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from app.services import context_graph

    chunk_ids = ["chunk-a", "chunk-b", "chunk-c", "chunk-d", "chunk-x"]
    chunks = _add_equal_quality_chunks(db_session, sample_knowledge_base.id, chunk_ids)
    vector_identity = {chunk_id: float(index + 1) for index, chunk_id in enumerate(chunk_ids)}
    vectors = {chunk_id: [identity] for chunk_id, identity in vector_identity.items()}
    pair_scores = {
        frozenset((vector_identity["chunk-a"], vector_identity["chunk-x"])): 0.9,
        frozenset((vector_identity["chunk-a"], vector_identity["chunk-b"])): 0.8,
        frozenset((vector_identity["chunk-c"], vector_identity["chunk-d"])): 0.8,
        frozenset((vector_identity["chunk-c"], vector_identity["chunk-x"])): 0.6,
    }

    def fixture_cosine(left, right):
        return pair_scores.get(frozenset((float(left[0]), float(right[0]))), 0.1)

    monkeypatch.setattr(context_graph, "cosine_similarity", fixture_cosine)
    candidates, diagnostics = context_graph.relation_edge_candidates(
        db_session,
        chunks,
        vectors,
        _operating_point(),
    )

    low_candidate, low_rank = _directional_contribution(candidates, "chunk-a", "chunk-b")
    high_candidate, high_rank = _directional_contribution(candidates, "chunk-c", "chunk-d")

    assert low_rank["cosine"] == high_rank["cosine"] == 0.8
    assert low_rank["rank_score"] == 0.5
    assert high_rank["rank_score"] == 1.0
    assert low_rank["rank_score"] != low_rank["cosine"]
    assert high_rank["rank_score"] != high_rank["cosine"]
    assert low_rank["rank_components"] == [
        {
            "candidate_channel": "base_dense_candidates",
            "rank": 2,
            "ordinal": 2,
            "candidate_count": 2,
            "selected_limit": 10,
            "selected_count": 2,
            "rank_score": 0.5,
        }
    ]
    assert high_rank["rank_components"][0]["rank"] == 1
    assert high_rank["rank_components"][0]["candidate_count"] == 2

    low_components = low_rank["raw_strength_components"]
    high_components = high_rank["raw_strength_components"]
    assert low_components["semantic"] == high_components["semantic"]
    assert low_components["reciprocity"] == high_components["reciprocity"] == 1.0
    assert low_components["node_quality_pair"] == high_components["node_quality_pair"]
    assert low_components["computed_raw_strength"] == low_rank["raw_strength"]
    assert high_components["computed_raw_strength"] == high_rank["raw_strength"]
    assert high_rank["raw_strength"] - low_rank["raw_strength"] == pytest.approx(0.035, abs=1e-6)

    protocol_hash = context_graph.relation_rank_score_protocol_hash()
    assert low_candidate.features_json["rank_score_protocol_version"] == context_graph.RANK_SCORE_PROTOCOL_VERSION
    assert low_candidate.features_json["rank_score_protocol_hash"] == protocol_hash
    assert low_candidate.features_json["node_weight_used_as_query_relevance"] is False
    assert diagnostics["rank_score_protocol_hash"] == protocol_hash
    assert diagnostics["accepted_rank_score_distribution"] != diagnostics["accepted_cosine_distribution"]
    trace_payload = context_graph.relation_edge_rank_trace_payload(
        SimpleNamespace(
            id="edge-a-b",
            source_chunk_id=low_candidate.source_chunk_id,
            target_chunk_id=low_candidate.target_chunk_id,
            edge_type=low_candidate.edge_type,
            raw_strength=low_candidate.raw_strength,
            features_json=low_candidate.features_json,
        )
    )
    assert trace_payload["rank_score_protocol_hash"] == protocol_hash
    assert trace_payload["raw_strength_components"]["computed_raw_strength"] == low_candidate.raw_strength
    assert len(trace_payload["directional_contributions"]) == 2
    assert trace_payload["node_weight_used_as_query_relevance"] is False


def test_equal_cosine_ties_share_competition_rank_while_ordinal_is_deterministic(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from app.services import context_graph

    chunk_ids = ["chunk-a", "chunk-b", "chunk-c", "chunk-x"]
    chunks = _add_equal_quality_chunks(db_session, sample_knowledge_base.id, chunk_ids)
    vector_identity = {chunk_id: float(index + 1) for index, chunk_id in enumerate(chunk_ids)}
    vectors = {chunk_id: [identity] for chunk_id, identity in vector_identity.items()}
    pair_scores = {
        frozenset((vector_identity["chunk-a"], vector_identity["chunk-x"])): 0.9,
        frozenset((vector_identity["chunk-a"], vector_identity["chunk-b"])): 0.8,
        frozenset((vector_identity["chunk-a"], vector_identity["chunk-c"])): 0.8,
    }

    monkeypatch.setattr(
        context_graph,
        "cosine_similarity",
        lambda left, right: pair_scores.get(frozenset((float(left[0]), float(right[0]))), 0.1),
    )
    candidates, _diagnostics = context_graph.relation_edge_candidates(
        db_session,
        chunks,
        vectors,
        _operating_point(),
    )

    _candidate_b, contribution_b = _directional_contribution(candidates, "chunk-a", "chunk-b")
    _candidate_c, contribution_c = _directional_contribution(candidates, "chunk-a", "chunk-c")
    rank_b = contribution_b["rank_components"][0]
    rank_c = contribution_c["rank_components"][0]
    assert rank_b["candidate_count"] == rank_c["candidate_count"] == 3
    assert rank_b["rank"] == rank_c["rank"] == 2
    assert rank_b["rank_score"] == rank_c["rank_score"] == 0.666667
    assert rank_b["ordinal"] == 2
    assert rank_c["ordinal"] == 3


def test_rank_protocol_propagates_to_operating_point_tpe_and_cache_key(no_fallback_env):
    from app.services.auto_tpe import normalize_theta
    from app.services.context_graph import (
        RANK_SCORE_PROTOCOL_VERSION,
        RELATION_RAW_STRENGTH_PROTOCOL_VERSION,
        context_graph_cache_key_components,
        dense_graph_operating_point,
        relation_rank_score_protocol_hash,
        relation_raw_strength_protocol_hash,
    )

    operating_point = dense_graph_operating_point()
    normalized_theta = normalize_theta(operating_point)
    cache_components = context_graph_cache_key_components(
        knowledge_base_id="kb-rank-protocol",
        query="rank audit",
        filters={},
        context_state=None,
        retrieval_mode="layered_context_graph",
        conversation_state_scope_hash="a" * 64,
        profile_hash_value="profile-rank-test",
    )

    assert operating_point["rank_score_protocol_version"] == RANK_SCORE_PROTOCOL_VERSION
    assert operating_point["rank_score_protocol_hash"] == relation_rank_score_protocol_hash()
    assert operating_point["raw_strength_protocol_version"] == RELATION_RAW_STRENGTH_PROTOCOL_VERSION
    assert operating_point["raw_strength_protocol_hash"] == relation_raw_strength_protocol_hash()
    assert normalized_theta["rank_score_protocol_hash"] == relation_rank_score_protocol_hash()
    assert normalized_theta["raw_strength_protocol_hash"] == relation_raw_strength_protocol_hash()
    assert cache_components["relation_rank_score_protocol_hash"] == relation_rank_score_protocol_hash()
    assert cache_components["relation_raw_strength_protocol_hash"] == relation_raw_strength_protocol_hash()

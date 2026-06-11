from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import SearchFilters, SearchRequest


def _seed_active_chunk(
    db_session,
    sample_knowledge_base,
    document,
    version,
    *,
    content: str,
    snippet: str,
    partition: str,
    section: str,
    parent_chunk_id: str | None = None,
    is_parent: bool = False,
):
    from app.models import (
        ActiveChunk,
        ChunkCandidate,
        ChunkDecision,
        EvidenceAtom,
        EvidenceGraphState,
        PolicyState,
        QualityDecision,
    )
    from app.services.evidence_graph import stable_hash

    policy_state = PolicyState(
        knowledge_base_id=sample_knowledge_base.id,
        profile_objective_hash="test-profile",
        posterior_json={},
        constraints_json={},
        exploration_json={},
        reward_summary_json={},
        state_hash=stable_hash({"policy": document.id, "section": section, "content": content}),
    )
    db_session.add(policy_state)
    db_session.flush()
    atom_index = (
        db_session.query(EvidenceAtom)
        .filter(EvidenceAtom.document_version_id == version.id)
        .count()
    )
    atom = EvidenceAtom(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        atom_index=atom_index,
        atom_type="paragraph",
        text=content,
        text_hash=stable_hash({"text": content}),
        source_span_json={"spans": [{"start": 0, "end": len(content), "section": section}]},
        layout_json={},
        metadata_json={"section": section, "section_index": atom_index},
        state="active",
    )
    db_session.add(atom)
    db_session.flush()
    graph_state = EvidenceGraphState(
        knowledge_base_id=sample_knowledge_base.id,
        scope_type="global",
        state_hash=stable_hash({"graph": atom.id}),
        atom_scope_hash=stable_hash({"atoms": [atom.id]}),
        active_document_version_ids=[version.id],
        active_atom_ids=[atom.id],
        policy_state_id=policy_state.id,
        stats_json={},
        diagnostics_json={},
        state="active",
    )
    db_session.add(graph_state)
    db_session.flush()
    candidate = ChunkCandidate(
        graph_state_id=graph_state.id,
        generator_name="test_fixture",
        generator_version="test_v1",
        atom_ids_json=[atom.id],
        source_span_union_json={"spans": [atom.source_span_json["spans"][0]]},
        token_count=16,
        graph_features_json={},
        cost_json={},
        diagnostics_json={},
    )
    db_session.add(candidate)
    db_session.flush()
    quality = QualityDecision(
        candidate_id=candidate.id,
        policy_state_id=policy_state.id,
        decision_action="answer_candidate",
        gate_passed=True,
        confidence=1.0,
        diagnostics_json={},
        reward_features_json={},
        feedback_json={},
    )
    db_session.add(quality)
    db_session.flush()
    decision = ChunkDecision(
        knowledge_base_id=sample_knowledge_base.id,
        graph_state_id=graph_state.id,
        candidate_id=candidate.id,
        quality_decision_id=quality.id,
        policy_state_id=policy_state.id,
        action="activate",
    )
    db_session.add(decision)
    db_session.flush()
    chunk = ActiveChunk(
        knowledge_base_id=sample_knowledge_base.id,
        chunk_decision_id=decision.id,
        document_version_scope_hash=stable_hash({"version": version.id}),
        graph_state_hash=graph_state.state_hash,
        atom_ids_json=[atom.id],
        text=content,
        source_span_union_json={"spans": [atom.source_span_json["spans"][0]]},
        boundary_policy_version="test_v1",
        quality_decision_id=quality.id,
        policy_state_id=policy_state.id,
        community_ids_json=[],
        metadata_json={
            "document_id": document.id,
            "document_version_id": version.id,
            "chunk_version": version.version,
            "partition": partition,
            "section": section,
            "source_type": document.source_type,
            "snippet": snippet,
            "content_kind": "markdown",
            "is_parent": is_parent,
            "parent_chunk_id": parent_chunk_id,
        },
        state="active",
    )
    db_session.add(chunk)
    db_session.flush()
    return chunk


def test_top_k_validation():
    assert SearchRequest(query="centrality").top_k == 6
    with pytest.raises(ValidationError):
        SearchRequest(query="centrality", top_k=0)
    with pytest.raises(ValidationError):
        SearchRequest(query="centrality", top_k=51)


def test_query_embedding_cache_version_tracks_runtime_settings(monkeypatch):
    from types import SimpleNamespace

    from app.services import retrieval

    monkeypatch.setattr(retrieval, "current_runtime_settings_version", lambda: "runtime-a")
    first = retrieval.query_embedding_cache_version(
        SimpleNamespace(embedding_model="text-embedding-v4", embedding_dimensions=1024)
    )
    second = retrieval.query_embedding_cache_version(
        SimpleNamespace(embedding_model="text-embedding-v3", embedding_dimensions=1024)
    )
    monkeypatch.setattr(retrieval, "current_runtime_settings_version", lambda: "runtime-b")
    third = retrieval.query_embedding_cache_version(
        SimpleNamespace(embedding_model="text-embedding-v4", embedding_dimensions=1024)
    )

    assert first != second
    assert first != third


def test_lexical_search_filters(db_session, sample_knowledge_base, indexed_chunks):
    from app.services.retrieval import lexical_search_chunks

    results = lexical_search_chunks(db_session, sample_knowledge_base.id, "degree centrality", SearchFilters(partition="L3"), 5)
    assert results
    assert results[0]["chunk_id"] == indexed_chunks[1][0].id
    assert results[0]["citations"][0]["document_title"] == "Centrality sources"

    none = lexical_search_chunks(db_session, sample_knowledge_base.id, "degree centrality", SearchFilters(partition="L9"), 5)
    assert none == []


def test_force_reparse_batch_marks_existing_files_pending(db_session, sample_knowledge_base):
    from app.core.config import get_settings
    from app.models import Document, DocumentVersion
    from app.services.ingestion import create_uploaded_files_batch
    from app.services.retrieval import list_knowledge_base_files

    storage_root = get_settings().knowledge_base_paths_for_name(sample_knowledge_base.name)["storage_root"]
    source_path = storage_root / "note.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("# Note\n\nAlready parsed", encoding="utf-8")

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="note",
        source_path=str(source_path),
        source_type="markdown",
        tags=["Unit"],
        checksum="checksum",
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="checksum",
        storage_path=str(source_path),
        extracted_path=None,
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    _seed_active_chunk(
        db_session,
        sample_knowledge_base,
        document,
        version,
        content="Already parsed",
        snippet="Already parsed",
        partition="Unit",
        section="Note",
    )
    db_session.commit()

    before = next(item for item in list_knowledge_base_files(db_session, sample_knowledge_base.id) if item["source_path"] == str(source_path))
    assert before["status"] == "parsed"

    create_uploaded_files_batch(db_session, sample_knowledge_base.id, [source_path], force=True)

    after = next(item for item in list_knowledge_base_files(db_session, sample_knowledge_base.id) if item["source_path"] == str(source_path))
    assert after["status"] == "pending"
    assert after["job_state"] == "queued"


@pytest.mark.asyncio
async def test_hybrid_search_uses_weighted_fusion_and_rerank(db_session, sample_knowledge_base, indexed_chunks, monkeypatch):
    from app.services import retrieval
    from app.services.retrieval import hybrid_search_chunks

    _, chunks = indexed_chunks
    dense_payload = {
        "chunk_id": chunks[1].id,
        "snippet": chunks[1].snippet,
        "score": 0.95,
        "citations": [],
        "metadata": {"scores": {"dense": 0.95}},
        "content": chunks[1].content,
        "document_title": "Centrality sources",
        "source_path": "centrality.md",
        "partition": "L3",
        "source_type": "markdown",
    }

    async def fake_dense_search(db, knowledge_base_id, query, filters, top_k, model_audit=None):
        if model_audit is not None:
            model_audit.update({"embedding_provider": "openai_compatible", "embedding_external_called": True})
        return [dense_payload]

    def fake_rerank_or_return(query, candidates, top_k):
        for item in candidates:
            item.setdefault("metadata", {}).setdefault("scores", {})["rerank"] = item["score"]
        return sorted(candidates, key=lambda item: item["score"], reverse=True)[:top_k]

    monkeypatch.setattr(retrieval, "dense_search_chunks", fake_dense_search)
    monkeypatch.setattr(retrieval, "rerank_or_return", fake_rerank_or_return)

    results = await hybrid_search_chunks(db_session, sample_knowledge_base.id, "degree centrality", SearchFilters(), 2)
    result_ids = {item["chunk_id"] for item in results}
    assert chunks[0].id in result_ids
    assert chunks[1].id in result_ids
    assert any("fused" in item["metadata"]["scores"] for item in results)
    assert all("unified" in item["metadata"]["scores"] for item in results)
    assert all(item["metadata"]["scores"]["unified_protocol"] == "retrieval_unified_score_v1" for item in results)
    assert all(item["metadata"]["scores"]["fusion_ranking_protocol"] == "retrieval_unified_score_v1" for item in results)
    assert all("rerank" in item["metadata"]["scores"] for item in results)
    assert all("query_type" in item["metadata"]["scores"] for item in results)


@pytest.mark.asyncio
async def test_hybrid_search_can_skip_reranker(db_session, sample_knowledge_base, indexed_chunks, monkeypatch):
    from app.services import retrieval
    from app.services.retrieval import hybrid_search_chunks

    _, chunks = indexed_chunks
    dense_payload = {
        "chunk_id": chunks[1].id,
        "snippet": chunks[1].snippet,
        "score": 0.95,
        "citations": [],
        "metadata": {"scores": {"dense": 0.95}},
        "content": chunks[1].content,
        "document_title": "Centrality sources",
        "source_path": "centrality.md",
        "partition": "L3",
        "source_type": "markdown",
    }

    async def fake_dense_search(db, knowledge_base_id, query, filters, top_k, model_audit=None):
        if model_audit is not None:
            model_audit.update({"embedding_provider": "openai_compatible", "embedding_external_called": True})
        return [dense_payload]

    class FakeSettings:
        embedding_model = "unit-test-embedding"
        enable_model_fallback = False
        retrieval_recall_k_default = 64
        retrieval_recall_k_formula = 80
        reranker_enabled = False

    monkeypatch.setattr(retrieval, "dense_search_chunks", fake_dense_search)
    monkeypatch.setattr(retrieval, "get_settings", lambda: FakeSettings())

    results = await hybrid_search_chunks(db_session, sample_knowledge_base.id, "degree centrality", SearchFilters(), 2)
    assert results
    assert all("lightweight_rerank" in item["metadata"]["scores"] for item in results)
    assert all("unified" in item["metadata"]["scores"] for item in results)


def test_lightweight_rerank_handles_none_fused_score():
    from app.services.retrieval import lightweight_rerank

    candidates = [
        {
            "chunk_id": "chunk-1",
            "snippet": "conjugate prior beta binomial example",
            "content": "A Beta prior is conjugate to a Bernoulli likelihood.",
            "score": 0.42,
            "metadata": {"scores": {"fused": None, "dense": 0.42}},
        }
    ]

    results = lightweight_rerank("Define conjugate prior", candidates, top_k=1)

    assert len(results) == 1
    assert results[0]["metadata"]["scores"]["lightweight_rerank"] > 0


@pytest.mark.asyncio
async def test_search_chunks_with_audit_reports_real_query_embedding(db_session, sample_knowledge_base, indexed_chunks, monkeypatch):
    from app.services import retrieval
    from app.services.embeddings import EmbeddingCallResult

    _, chunks = indexed_chunks

    async def fake_embed(self, texts, text_type="document"):
        assert texts == ["degree centrality"]
        assert text_type == "query"
        return EmbeddingCallResult(vectors=[[0.1, 0.2, 0.3]], provider="openai_compatible", external_called=True)

    class FakeVectorStore:
        def __init__(self, knowledge_base_name):
            self.knowledge_base_name = knowledge_base_name

        def search(self, *, vector, limit, filters):
            assert vector == [0.1, 0.2, 0.3]
            return [{"id": chunks[0].id, "score": 0.9}]

        async def async_search(self, **kwargs):
            return self.search(**kwargs)

    def fake_rerank_or_return(query, candidates, top_k):
        return candidates[:top_k]

    monkeypatch.setattr(retrieval.EmbeddingProvider, "embed_texts_with_meta", fake_embed)
    monkeypatch.setattr(retrieval, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(retrieval, "rerank_or_return", fake_rerank_or_return)

    results, audit = await retrieval.search_chunks_with_audit(db_session, sample_knowledge_base.id, "degree centrality", SearchFilters(), 1)

    assert results
    assert audit["embedding_provider"] == "openai_compatible"
    assert audit["embedding_external_called"] is True
    assert audit["embedding_fallback_reason"] is None
    assert results[0]["metadata"]["model_audit"]["embedding_external_called"] is True


@pytest.mark.asyncio
async def test_search_ignores_zero_score_dense_index_and_uses_lexical(db_session, sample_knowledge_base, indexed_chunks, monkeypatch):
    from app.services import retrieval
    from app.services.embeddings import EmbeddingCallResult

    _, chunks = indexed_chunks

    async def fake_embed(self, texts, text_type="document"):
        return EmbeddingCallResult(vectors=[[0.1, 0.2, 0.3]], provider="openai_compatible", external_called=True)

    class ZeroVectorStore:
        def __init__(self, knowledge_base_name):
            self.knowledge_base_name = knowledge_base_name

        def search(self, *, vector, limit, filters):
            return [{"id": chunk.id, "score": 0.0} for chunk in chunks]

        async def async_search(self, **kwargs):
            return self.search(**kwargs)

    def fake_rerank_or_return(query, candidates, top_k):
        return candidates[:top_k]

    monkeypatch.setattr(retrieval.EmbeddingProvider, "embed_texts_with_meta", fake_embed)
    monkeypatch.setattr(retrieval, "VectorStore", ZeroVectorStore)
    monkeypatch.setattr(retrieval, "rerank_or_return", fake_rerank_or_return)

    results, audit = await retrieval.search_chunks_with_audit(db_session, sample_knowledge_base.id, "degree centrality", SearchFilters(), 2)

    assert results
    assert audit["vector_index_warning"] == "qdrant_returned_only_zero_scores"
    assert all(item["metadata"]["scores"]["dense"] is None for item in results)
    assert any("bm25" in item["metadata"]["scores"] for item in results)


@pytest.mark.asyncio
async def test_search_scores_include_primary_channels_for_dense_only_results(db_session, sample_knowledge_base, indexed_chunks, monkeypatch):
    from app.services import retrieval
    from app.services.embeddings import EmbeddingCallResult

    _, chunks = indexed_chunks

    async def fake_embed(self, texts, text_type="document"):
        return EmbeddingCallResult(vectors=[[0.1, 0.2, 0.3]], provider="openai_compatible", external_called=True)

    class DenseOnlyStore:
        def __init__(self, knowledge_base_name):
            self.knowledge_base_name = knowledge_base_name

        def search(self, *, vector, limit, filters):
            return [{"id": chunks[0].id, "score": 0.8}]

        async def async_search(self, **kwargs):
            return self.search(**kwargs)

    def fake_rerank_or_return(query, candidates, top_k):
        for item in candidates:
            item.setdefault("metadata", {}).setdefault("scores", {})["rerank"] = 0.5
        return candidates[:top_k]

    monkeypatch.setattr(retrieval.EmbeddingProvider, "embed_texts_with_meta", fake_embed)
    monkeypatch.setattr(retrieval, "VectorStore", DenseOnlyStore)
    monkeypatch.setattr(retrieval, "lexical_search_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval, "rerank_or_return", fake_rerank_or_return)

    results, audit = await retrieval.search_chunks_with_audit(db_session, sample_knowledge_base.id, "no lexical match", SearchFilters(), 1)

    scores = results[0]["metadata"]["scores"]
    assert audit["reranker_called"] is True
    assert scores["dense"] == 0.8
    assert scores["lexical"] is None
    assert scores["fused"] is None
    assert scores["rerank"] == 0.5
    assert scores["unified"] is not None


@pytest.mark.asyncio
async def test_search_expands_child_result_to_parent_context(db_session, sample_knowledge_base, monkeypatch):
    from app.models import Document, DocumentVersion
    from app.services import retrieval
    from app.services.embeddings import EmbeddingCallResult

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Hierarchy sources",
        source_path="hierarchy.md",
        source_type="markdown",
        tags=["L4"],
        checksum="hierarchy",
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="hierarchy",
        storage_path="hierarchy.md",
        extracted_path=None,
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    parent = _seed_active_chunk(
        db_session,
        sample_knowledge_base,
        document,
        version,
        content="Full parent section about graph cuts and max flow. It contains the broader explanation.",
        snippet="Full parent section about graph cuts and max flow.",
        partition="L4",
        section="Cuts",
        is_parent=True,
    )
    child = _seed_active_chunk(
        db_session,
        sample_knowledge_base,
        document,
        version,
        content="A cut separates source and sink.",
        snippet="A cut separates source and sink.",
        partition="L4",
        section="Cuts",
        parent_chunk_id=parent.id,
    )
    db_session.commit()

    async def fake_embed(self, texts, text_type="document"):
        return EmbeddingCallResult(vectors=[[0.1, 0.2, 0.3]], provider="openai_compatible", external_called=True)

    class ChildOnlyStore:
        def __init__(self, knowledge_base_name):
            self.knowledge_base_name = knowledge_base_name

        def search(self, *, vector, limit, filters):
            assert "is_parent" not in filters
            return [{"id": child.id, "score": 0.9}]

        async def async_search(self, **kwargs):
            return self.search(**kwargs)

    monkeypatch.setattr(retrieval.EmbeddingProvider, "embed_texts_with_meta", fake_embed)
    monkeypatch.setattr(retrieval, "VectorStore", ChildOnlyStore)
    monkeypatch.setattr(retrieval, "lexical_search_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval, "rerank_or_return", lambda query, candidates, top_k: candidates[:top_k])

    results, audit = await retrieval.search_chunks_with_audit(db_session, sample_knowledge_base.id, "source sink cut", SearchFilters(), 1)

    assert audit["embedding_external_called"] is True
    assert results[0]["chunk_id"] == child.id
    assert results[0]["child_content"] is None
    assert results[0]["content"] == child.content
    assert results[0]["metadata"]["parent_chunk_id"] == parent.id
    assert "parent_content" not in results[0]["metadata"]

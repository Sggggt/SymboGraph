from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import SearchFilters, SearchRequest


def test_top_k_validation():
    assert SearchRequest(query="centrality").top_k == 6
    with pytest.raises(ValidationError):
        SearchRequest(query="centrality", top_k=0)
    with pytest.raises(ValidationError):
        SearchRequest(query="centrality", top_k=51)


def test_lexical_search_filters(db_session, sample_course, indexed_chunks):
    from app.services.retrieval import lexical_search_chunks

    results = lexical_search_chunks(db_session, sample_course.id, "degree centrality", SearchFilters(chapter="L3"), 5)
    assert results
    assert results[0]["chunk_id"] == indexed_chunks[1][0].id
    assert results[0]["citations"][0]["document_title"] == "Centrality Notes"

    none = lexical_search_chunks(db_session, sample_course.id, "degree centrality", SearchFilters(chapter="L9"), 5)
    assert none == []


def test_force_reparse_batch_marks_existing_files_pending(db_session, sample_course):
    from app.core.config import get_settings
    from app.models import Chunk, Document, DocumentVersion
    from app.services.ingestion import create_uploaded_files_batch
    from app.services.retrieval import list_course_files

    storage_root = get_settings().course_paths_for_name(sample_course.name)["storage_root"]
    source_path = storage_root / "note.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("# Note\n\nAlready parsed", encoding="utf-8")

    document = Document(
        course_id=sample_course.id,
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
    db_session.add(
        Chunk(
            course_id=sample_course.id,
            document_id=document.id,
            document_version_id=version.id,
            content="Already parsed",
            snippet="Already parsed",
            chapter="Unit",
            section="Note",
            source_type="markdown",
            metadata_json={"content_kind": "markdown"},
            embedding_status="ready",
        )
    )
    db_session.commit()

    before = next(item for item in list_course_files(db_session, sample_course.id) if item["source_path"] == str(source_path))
    assert before["status"] == "parsed"

    create_uploaded_files_batch(db_session, sample_course.id, [source_path], force=True)

    after = next(item for item in list_course_files(db_session, sample_course.id) if item["source_path"] == str(source_path))
    assert after["status"] == "pending"
    assert after["job_state"] == "queued"


@pytest.mark.asyncio
async def test_hybrid_search_uses_weighted_fusion_and_rerank(db_session, sample_course, indexed_chunks, monkeypatch):
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
        "document_title": "Centrality Notes",
        "source_path": "centrality.md",
        "chapter": "L3",
        "source_type": "markdown",
    }

    async def fake_dense_search(db, course_id, query, filters, top_k, model_audit=None):
        if model_audit is not None:
            model_audit.update({"embedding_provider": "openai_compatible", "embedding_external_called": True})
        return [dense_payload]

    def fake_rerank_or_return(query, candidates, top_k):
        for item in candidates:
            item.setdefault("metadata", {}).setdefault("scores", {})["rerank"] = item["score"]
        return sorted(candidates, key=lambda item: item["score"], reverse=True)[:top_k]

    monkeypatch.setattr(retrieval, "dense_search_chunks", fake_dense_search)
    monkeypatch.setattr(retrieval, "rerank_or_return", fake_rerank_or_return)

    results = await hybrid_search_chunks(db_session, sample_course.id, "degree centrality", SearchFilters(), 2)
    result_ids = {item["chunk_id"] for item in results}
    assert chunks[0].id in result_ids
    assert chunks[1].id in result_ids
    assert any("fused" in item["metadata"]["scores"] for item in results)
    assert all("rerank" in item["metadata"]["scores"] for item in results)
    assert all("query_type" in item["metadata"]["scores"] for item in results)


@pytest.mark.asyncio
async def test_hybrid_search_can_skip_reranker(db_session, sample_course, indexed_chunks, monkeypatch):
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
        "document_title": "Centrality Notes",
        "source_path": "centrality.md",
        "chapter": "L3",
        "source_type": "markdown",
    }

    async def fake_dense_search(db, course_id, query, filters, top_k, model_audit=None):
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

    results = await hybrid_search_chunks(db_session, sample_course.id, "degree centrality", SearchFilters(), 2)
    assert results
    assert all("lightweight_rerank" in item["metadata"]["scores"] for item in results)


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
async def test_search_chunks_with_audit_reports_real_query_embedding(db_session, sample_course, indexed_chunks, monkeypatch):
    from app.services import retrieval
    from app.services.embeddings import EmbeddingCallResult

    _, chunks = indexed_chunks

    async def fake_embed(self, texts, text_type="document"):
        assert texts == ["degree centrality"]
        assert text_type == "query"
        return EmbeddingCallResult(vectors=[[0.1, 0.2, 0.3]], provider="openai_compatible", external_called=True)

    class FakeVectorStore:
        def __init__(self, course_name):
            self.course_name = course_name

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

    results, audit = await retrieval.search_chunks_with_audit(db_session, sample_course.id, "degree centrality", SearchFilters(), 1)

    assert results
    assert audit["embedding_provider"] == "openai_compatible"
    assert audit["embedding_external_called"] is True
    assert audit["embedding_fallback_reason"] is None
    assert results[0]["metadata"]["model_audit"]["embedding_external_called"] is True


@pytest.mark.asyncio
async def test_search_ignores_zero_score_dense_index_and_uses_lexical(db_session, sample_course, indexed_chunks, monkeypatch):
    from app.services import retrieval
    from app.services.embeddings import EmbeddingCallResult

    _, chunks = indexed_chunks

    async def fake_embed(self, texts, text_type="document"):
        return EmbeddingCallResult(vectors=[[0.1, 0.2, 0.3]], provider="openai_compatible", external_called=True)

    class ZeroVectorStore:
        def __init__(self, course_name):
            self.course_name = course_name

        def search(self, *, vector, limit, filters):
            return [{"id": chunk.id, "score": 0.0} for chunk in chunks]

        async def async_search(self, **kwargs):
            return self.search(**kwargs)

    def fake_rerank_or_return(query, candidates, top_k):
        return candidates[:top_k]

    monkeypatch.setattr(retrieval.EmbeddingProvider, "embed_texts_with_meta", fake_embed)
    monkeypatch.setattr(retrieval, "VectorStore", ZeroVectorStore)
    monkeypatch.setattr(retrieval, "rerank_or_return", fake_rerank_or_return)

    results, audit = await retrieval.search_chunks_with_audit(db_session, sample_course.id, "degree centrality", SearchFilters(), 2)

    assert results
    assert audit["vector_index_warning"] == "qdrant_returned_only_zero_scores"
    assert all(item["metadata"]["scores"]["dense"] is None for item in results)
    assert any("bm25" in item["metadata"]["scores"] for item in results)


@pytest.mark.asyncio
async def test_search_scores_include_primary_channels_for_dense_only_results(db_session, sample_course, indexed_chunks, monkeypatch):
    from app.services import retrieval
    from app.services.embeddings import EmbeddingCallResult

    _, chunks = indexed_chunks

    async def fake_embed(self, texts, text_type="document"):
        return EmbeddingCallResult(vectors=[[0.1, 0.2, 0.3]], provider="openai_compatible", external_called=True)

    class DenseOnlyStore:
        def __init__(self, course_name):
            self.course_name = course_name

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

    results, audit = await retrieval.search_chunks_with_audit(db_session, sample_course.id, "no lexical match", SearchFilters(), 1)

    scores = results[0]["metadata"]["scores"]
    assert audit["reranker_called"] is True
    assert scores["dense"] == 0.8
    assert scores["lexical"] is None
    assert scores["fused"] is None
    assert scores["rerank"] == 0.5


@pytest.mark.asyncio
async def test_search_expands_child_result_to_parent_context(db_session, sample_course, monkeypatch):
    from app.models import Chunk, Document, DocumentVersion
    from app.services import retrieval
    from app.services.embeddings import EmbeddingCallResult

    document = Document(
        course_id=sample_course.id,
        title="Hierarchy Notes",
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
    parent = Chunk(
        course_id=sample_course.id,
        document_id=document.id,
        document_version_id=version.id,
        content="Full parent section about graph cuts and max flow. It contains the broader explanation.",
        snippet="Full parent section about graph cuts and max flow.",
        chapter="L4",
        section="Cuts",
        source_type="markdown",
        metadata_json={"content_kind": "markdown", "is_parent": True},
        embedding_status="ready",
    )
    db_session.add(parent)
    db_session.flush()
    child = Chunk(
        course_id=sample_course.id,
        document_id=document.id,
        document_version_id=version.id,
        content="A cut separates source and sink.",
        snippet="A cut separates source and sink.",
        chapter="L4",
        section="Cuts",
        source_type="markdown",
        metadata_json={"content_kind": "markdown", "is_parent": False},
        parent_chunk_id=parent.id,
        embedding_status="ready",
    )
    db_session.add(child)
    db_session.commit()

    async def fake_embed(self, texts, text_type="document"):
        return EmbeddingCallResult(vectors=[[0.1, 0.2, 0.3]], provider="openai_compatible", external_called=True)

    class ChildOnlyStore:
        def __init__(self, course_name):
            self.course_name = course_name

        def search(self, *, vector, limit, filters):
            assert "is_parent" not in filters
            return [{"id": child.id, "score": 0.9}]

        async def async_search(self, **kwargs):
            return self.search(**kwargs)

    monkeypatch.setattr(retrieval.EmbeddingProvider, "embed_texts_with_meta", fake_embed)
    monkeypatch.setattr(retrieval, "VectorStore", ChildOnlyStore)
    monkeypatch.setattr(retrieval, "lexical_search_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval, "rerank_or_return", lambda query, candidates, top_k: candidates[:top_k])

    results, audit = await retrieval.search_chunks_with_audit(db_session, sample_course.id, "source sink cut", SearchFilters(), 1)

    assert audit["embedding_external_called"] is True
    assert results[0]["chunk_id"] == child.id
    assert results[0]["child_content"] == "A cut separates source and sink."
    assert results[0]["content"] == parent.content
    assert results[0]["metadata"]["parent_chunk_id"] == parent.id
    assert results[0]["metadata"]["parent_content"] == parent.content
    assert results[0]["metadata"]["retrieval_granularity"] == "child_with_parent_context"

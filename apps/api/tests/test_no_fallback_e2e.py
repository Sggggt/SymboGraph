from __future__ import annotations

import os
import uuid
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.no_fallback_e2e

if os.getenv("RUN_NO_FALLBACK_E2E") != "1":
    pytest.skip("Set RUN_NO_FALLBACK_E2E=1 to run the real no-fallback integration smoke test", allow_module_level=True)


@pytest.mark.asyncio
async def test_parse_search_and_qa_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_MODEL_FALLBACK", "false")
    monkeypatch.setenv("ENABLE_DATABASE_FALLBACK", "false")

    from app.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    assert settings.openai_api_key, "OPENAI_API_KEY is required for no-fallback E2E"
    assert settings.chat_base_url, "CHAT_BASE_URL is required for no-fallback E2E"
    assert settings.embedding_api_key, "EMBEDDING_API_KEY is required for no-fallback E2E"
    assert settings.embedding_base_url, "EMBEDDING_BASE_URL is required for no-fallback E2E"
    assert settings.enable_model_fallback is False
    assert settings.enable_database_fallback is False

    import app.models as models
    from app.db import Base, SessionLocal, engine, ensure_schema
    from app.schemas import AgentRequest, SearchFilters
    from app.services.agent_graph import run_agent
    from app.services.ingestion import create_knowledge_base_space, ingest_file
    from app.services.maintenance import delete_knowledge_base_data
    from app.services.retrieval import search_chunks_with_audit
    from app.services.vector_store import VectorStore

    Base.metadata.create_all(bind=engine)
    ensure_schema()

    knowledge_base_name = f"No Fallback E2E {uuid.uuid4()}"
    source = tmp_path / "centrality.md"
    source.write_text(
        "# Centrality\n\nDegree centrality counts incident edges in a graph. "
        "It is a local measure used in network analysis.",
        encoding="utf-8",
    )

    db = SessionLocal()
    KnowledgeBase = None
    try:
        try:
            qdrant_response = httpx.get(f"{settings.qdrant_url.rstrip('/')}/collections", timeout=5.0, trust_env=False)
            qdrant_response.raise_for_status()
        except Exception as exc:
            pytest.fail(f"Qdrant is unavailable at {settings.qdrant_url}; no-fallback E2E cannot run: {exc}")
        VectorStore(knowledge_base_name=knowledge_base_name)
        KnowledgeBase = create_knowledge_base_space(db, knowledge_base_name, "temporary no-fallback smoke test")
        result = await ingest_file(db, source, trigger_source="test", knowledge_base_id=KnowledgeBase.id, rebuild_graph=False)
        assert result["status"] == "completed"
        assert result["stats"]["embedding_provider"] == "openai_compatible"
        assert result["stats"]["embedding_external_called"] is True
        assert result["stats"]["embedding_fallback_reason"] is None

        chunks = db.query(models.ActiveChunk).filter(models.ActiveChunk.knowledge_base_id == KnowledgeBase.id, models.ActiveChunk.state == "active").all()
        chunk_ids = [chunk.id for chunk in chunks]
        assert chunk_ids
        atoms = db.query(models.EvidenceAtom).filter(models.EvidenceAtom.knowledge_base_id == KnowledgeBase.id, models.EvidenceAtom.state == "active").all()
        assert atoms
        vector_records = db.query(models.VectorRecord).filter(models.VectorRecord.knowledge_base_id == KnowledgeBase.id).all()
        assert {record.active_chunk_id for record in vector_records}.issubset(set(chunk_ids))
        assert vector_records

        search_results, search_audit = await search_chunks_with_audit(db, KnowledgeBase.id, "degree centrality", SearchFilters(), 3)
        assert search_results
        assert search_results[0]["citations"]
        traces = db.query(models.RetrievalTrace).filter(models.RetrievalTrace.knowledge_base_id == KnowledgeBase.id).all()
        assert traces
        assert search_audit["embedding_provider"] == "openai_compatible"
        assert search_audit["embedding_external_called"] is True
        assert search_audit["embedding_fallback_reason"] is None

        response = await run_agent(
            db,
            AgentRequest(question="What is degree centrality?", knowledge_base_id=KnowledgeBase.id, top_k=3),
        )
        assert response["answer"]
        assert response["citations"]
        assert response["trace"]
        answer_trace = next(item for item in response["trace"] if item["node"] == "answer_generator")
        assert "provider=openai_compatible_chat" in (answer_trace["output_summary"] or "")
        assert "fallback=None" in (answer_trace["output_summary"] or "")

        vector_index = settings.knowledge_base_paths_for_name(KnowledgeBase.name)["ingestion_root"] / "vector_index.json"
        assert not vector_index.exists()
    finally:
        if KnowledgeBase is not None:
            delete_knowledge_base_data(db, KnowledgeBase)
        db.close()
        get_settings.cache_clear()

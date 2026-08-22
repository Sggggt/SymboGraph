from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select


def _chunked_document(db_session, knowledge_base, *, suffix: str = "local-hints"):
    from app.models import Document, DocumentVersion
    from app.services.context_graph import write_chunks_and_structure
    from app.services.parsers import ParsedSection

    document = Document(
        knowledge_base_id=knowledge_base.id,
        title=f"Local hint protocol {suffix}",
        source_path=f"local-hint-{suffix}.md",
        source_type="markdown",
        checksum=f"document-{suffix}",
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=f"version-{suffix}",
        storage_path=document.source_path,
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    sections = [
        ParsedSection(
            title="Deterministic neighborhood",
            section="Local hints > Deterministic neighborhood",
            page_number=1,
            text=" ".join(f"protocol-token-{index}" for index in range(180)),
        )
    ]
    chunks = write_chunks_and_structure(
        db_session,
        knowledge_base=knowledge_base,
        document=document,
        version=version,
        sections=sections,
        chunk_version=1,
        chunk_size=24,
        chunk_overlap=6,
    )
    assert len(chunks) >= 4
    return document, version, chunks


def test_local_hints_are_deterministic_nonoverlapping_and_raw_span_backed(
    db_session,
    sample_knowledge_base,
):
    from app.services.chunking import (
        LOCAL_CONTEXT_HINT_NEIGHBOR_TOKEN_BUDGET,
        LOCAL_CONTEXT_HINT_PROTOCOL_VERSION,
        LOCAL_CONTEXT_HINT_TOTAL_TOKEN_BUDGET,
        normalize_text,
    )
    from app.services.context_graph import build_local_context_hints

    _, version, chunks = _chunked_document(db_session, sample_knowledge_base)
    first = build_local_context_hints(db_session, chunks)
    second = build_local_context_hints(db_session, list(reversed(chunks)))
    middle = chunks[len(chunks) // 2]
    hint = first[middle.id]

    assert hint.protocol_version == LOCAL_CONTEXT_HINT_PROTOCOL_VERSION
    assert hint.hint_hash == second[middle.id].hint_hash
    assert hint.text == second[middle.id].text
    assert [card["role"] for card in hint.source_cards] == ["previous", "next"]
    assert sum(card["token_count"] for card in hint.source_cards) <= LOCAL_CONTEXT_HINT_TOTAL_TOKEN_BUDGET

    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    for card in hint.source_cards:
        source = chunks_by_id[card["source_chunk_id"]]
        span_start, span_end = card["char_span"]
        local_start = span_start - source.char_start
        local_end = span_end - source.char_start
        assert card["document_version_id"] == version.id == middle.document_version_id
        assert card["token_count"] <= LOCAL_CONTEXT_HINT_NEIGHBOR_TOKEN_BUDGET
        assert normalize_text(source.text[local_start:local_end]) == card["excerpt"]
        if card["role"] == "previous":
            assert span_end <= middle.char_start
        else:
            assert span_start >= middle.char_end


def test_local_hint_generation_fails_closed_on_broken_structure_pointer(
    db_session,
    sample_knowledge_base,
):
    from app.services.context_graph import build_local_context_hints

    _, _, chunks = _chunked_document(db_session, sample_knowledge_base, suffix="broken-pointer")
    chunks[1].next_chunk_id = "missing-active-neighbor"
    db_session.flush()

    with pytest.raises(RuntimeError, match="pointer target is missing"):
        build_local_context_hints(db_session, chunks)

    chunks[1].next_chunk_id = chunks[0].id
    db_session.flush()
    with pytest.raises(RuntimeError, match="invalid document order"):
        build_local_context_hints(db_session, chunks)


@pytest.mark.asyncio
async def test_contextual_index_persists_hint_hashes_and_binds_cache_key(
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import ChunkContextText, VectorRecord
    from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION, LOCAL_CONTEXT_HINT_PROTOCOL_VERSION
    from app.services.context_graph import (
        build_local_context_hints,
        context_graph_cache_key,
        contextual_index_state_hash,
        write_contextual_indexes,
    )

    _, _, chunks = _chunked_document(db_session, sample_knowledge_base, suffix="index")
    hints = build_local_context_hints(db_session, chunks)
    stats = await write_contextual_indexes(
        db_session,
        knowledge_base=sample_knowledge_base,
        chunks=chunks,
        local_hints=hints,
    )
    middle = chunks[len(chunks) // 2]
    row = db_session.scalar(
        select(ChunkContextText).where(
            ChunkContextText.chunk_id == middle.id,
            ChunkContextText.embedding_text_version == CURRENT_EMBEDDING_TEXT_VERSION,
        )
    )
    vector_record = db_session.scalar(
        select(VectorRecord).where(
            VectorRecord.chunk_id == middle.id,
            VectorRecord.embedding_text_version == CURRENT_EMBEDDING_TEXT_VERSION,
        )
    )
    stored_point = fake_model_stack["VectorStore"].points[middle.id]

    assert row is not None and vector_record is not None
    assert "Local hint:\nPrevious context:" in row.contextual_text
    assert (row.metadata_json or {})["local_hint_protocol_version"] == LOCAL_CONTEXT_HINT_PROTOCOL_VERSION
    assert (row.metadata_json or {})["local_hint_hash"] == hints[middle.id].hint_hash
    assert (vector_record.diagnostics_json or {})["local_hint_hash"] == hints[middle.id].hint_hash
    assert stored_point["payload"]["local_hint_hash"] == hints[middle.id].hint_hash
    assert stored_point["payload"]["context_hash"] == row.context_hash
    assert stats["contextual_index_scope_chunks"] == len(chunks)
    assert stats["contextual_index_hash"] == contextual_index_state_hash(
        db_session,
        sample_knowledge_base.id,
        chunks,
    )

    state = SimpleNamespace(
        diagnostics_json={"contextual_index_hash": stats["contextual_index_hash"]},
        context_graph_hash="context-graph",
        chunk_scope_hash="chunk-scope",
        structure_graph_hash="structure",
        chunk_relation_graph_hash="relation",
        rq_membership_hash="rq",
        mid_concept_hash="mid",
        coarse_concept_hash="coarse",
        policy_state_hash="policy",
        prompt_protocol_hash="prompt",
    )
    first_key = context_graph_cache_key(
        knowledge_base_id=sample_knowledge_base.id,
        query="local hint cache propagation",
        filters={},
        context_state=state,
        retrieval_mode="search",
        conversation_state_scope_hash="a" * 64,
        profile_hash_value="profile",
    )
    state.diagnostics_json = {"contextual_index_hash": "changed-contextual-index"}
    second_key = context_graph_cache_key(
        knowledge_base_id=sample_knowledge_base.id,
        query="local hint cache propagation",
        filters={},
        context_state=state,
        retrieval_mode="search",
        conversation_state_scope_hash="a" * 64,
        profile_hash_value="profile",
    )
    assert first_key != second_key


@pytest.mark.asyncio
async def test_rebuild_repairs_stale_local_hint_context_before_relation_graph(
    db_session,
    populated_context_graph,
):
    from app.models import ChunkContextText, VectorRecord
    from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION
    from app.services.context_graph import (
        contextual_index_state_hash,
        freshness_payload,
        rebuild_context_graph,
    )

    knowledge_base = populated_context_graph["knowledge_base"]
    chunks = populated_context_graph["chunks"]
    previous_state = populated_context_graph["state"]
    middle = chunks[len(chunks) // 2]
    context_row = db_session.scalar(
        select(ChunkContextText).where(
            ChunkContextText.chunk_id == middle.id,
            ChunkContextText.embedding_text_version == CURRENT_EMBEDDING_TEXT_VERSION,
        )
    )
    vector_row = db_session.scalar(
        select(VectorRecord).where(
            VectorRecord.chunk_id == middle.id,
            VectorRecord.embedding_text_version == CURRENT_EMBEDDING_TEXT_VERSION,
        )
    )
    assert context_row is not None and vector_row is not None
    expected_hint_hash = (context_row.metadata_json or {})["local_hint_hash"]
    context_row.contextual_text = f"{context_row.contextual_text}\ncorrupt"
    context_row.context_hash = "corrupt-context-hash"
    context_row.metadata_json = {**dict(context_row.metadata_json or {}), "local_hint_hash": "corrupt-hint-hash"}
    vector_row.payload_hash = "corrupt-vector-payload-hash"
    vector_row.diagnostics_json = {
        **dict(vector_row.diagnostics_json or {}),
        "context_hash": "corrupt-context-hash",
        "local_hint_hash": "corrupt-hint-hash",
    }
    db_session.flush()

    state = await rebuild_context_graph(
        db_session,
        knowledge_base.id,
        emit_heartbeats=False,
    )
    db_session.refresh(context_row)
    db_session.refresh(vector_row)
    maintenance = (state.diagnostics_json or {})["contextual_index_maintenance"]

    assert maintenance["reindexed_chunks"] == 1
    assert middle.id in maintenance["stale_reasons_by_chunk"]
    assert "vector_payload_hash_mismatch" in maintenance["stale_reasons_by_chunk"][middle.id]
    assert (context_row.metadata_json or {})["local_hint_hash"] == expected_hint_hash
    assert (vector_row.diagnostics_json or {})["local_hint_hash"] == expected_hint_hash
    assert (vector_row.diagnostics_json or {})["context_hash"] == context_row.context_hash
    assert (state.diagnostics_json or {})["contextual_index_hash"] == contextual_index_state_hash(
        db_session,
        knowledge_base.id,
        chunks,
    )
    previous_freshness = freshness_payload(db_session, knowledge_base.id, previous_state)
    assert previous_freshness["is_stale"] is True
    assert "contextual_index_changed" in previous_freshness["stale_reasons"]
    assert freshness_payload(db_session, knowledge_base.id, state)["is_stale"] is False


@pytest.mark.asyncio
async def test_production_ingestion_passes_generated_hint_packets_to_index_writer(
    monkeypatch,
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.services import ingestion
    from app.services.chunking import LOCAL_CONTEXT_HINT_PROTOCOL_VERSION
    from app.core.config import get_settings

    storage_root = get_settings().knowledge_base_paths_for_name(
        sample_knowledge_base.name
    )["storage_root"]
    storage_root.mkdir(parents=True, exist_ok=True)
    source = storage_root / "production-local-hints.md"
    source.write_text(
        "# Production local hints\n\n" + " ".join(f"ingestion-token-{index}" for index in range(720)),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    real_write_contextual_indexes = ingestion.write_contextual_indexes

    async def capture_write_contextual_indexes(*args, **kwargs):
        captured["local_hints"] = kwargs.get("local_hints")
        return await real_write_contextual_indexes(*args, **kwargs)

    monkeypatch.setattr(ingestion, "write_contextual_indexes", capture_write_contextual_indexes)
    result = await ingestion.ingest_file(
        db_session,
        source,
        knowledge_base_id=sample_knowledge_base.id,
        rebuild_graph=False,
        target_version=1,
    )

    local_hints = captured.get("local_hints")
    assert result["status"] == "completed"
    assert result["chunk_count"] >= 2
    assert isinstance(local_hints, dict)
    assert len(local_hints) == result["chunk_count"]
    assert any(packet.text for packet in local_hints.values())
    assert all(packet.protocol_version == LOCAL_CONTEXT_HINT_PROTOCOL_VERSION for packet in local_hints.values())
    assert result["stats"]["local_hint_chunks"] >= 1

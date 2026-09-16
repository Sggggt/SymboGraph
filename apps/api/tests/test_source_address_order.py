from copy import deepcopy
from uuid import UUID

import pytest
from sqlalchemy import select, text

from app.models import ChunkStructureNode, ChunkStructureMapping, ChunkCoordinate, KnowledgeBase
from app.services.context_graph import chunk_source_span, ChunkSourceProvenanceError
from test_citation_provenance import _build_package
from test_answer_sources_postgres import postgres_source_root
from test_tpe_audit_postgres import postgres_tpe_scope


async def tied_source(db, kb, path):
    _, doc, version, chunks, *_ = await _build_package(db, kb, path)
    chunk = chunks[0]
    ids = [str(UUID(int=i + 10)) for i in range(9)]
    for node_id in reversed(ids):
        db.add(ChunkStructureNode(id=node_id, knowledge_base_id=kb.id, document_id=doc.id,
            document_version_id=version.id, node_type="paragraph", depth=100, title="Synthetic tied structure",
            char_start=chunk.char_start, char_end=chunk.char_end))
    db.flush()
    for node_id in reversed(ids):
        db.add(ChunkStructureMapping(chunk_id=chunk.id, structure_node_id=node_id, document_version_id=version.id, mapping_weight=1.0))
    db.flush()
    return chunk, ids


@pytest.mark.asyncio
async def test_new_source_address_is_stable_and_legacy_tied_selection_is_replayed(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    chunk, ids = await tied_source(db_session, sample_knowledge_base, tmp_path)
    span = chunk_source_span(db_session, chunk)
    assert span["contract_version"] == "raw_chunk_source_span_v3"
    assert span["structure_node_ids"] == ids[:8]
    old = {**deepcopy(span), "contract_version": "raw_chunk_source_span_v1", "structure_node_ids": list(reversed(ids[1:]))}
    assert chunk_source_span(db_session, chunk, replay_source_span=old)["structure_node_ids"] == old["structure_node_ids"]
    assert chunk_source_span(db_session, chunk)["structure_node_ids"] == ids[:8]
    row = db_session.scalar(select(ChunkStructureMapping).where(ChunkStructureMapping.structure_node_id == old["structure_node_ids"][0]))
    row.mapping_weight = 0.25
    db_session.flush()
    with pytest.raises(ChunkSourceProvenanceError, match="legacy_node_rank_invalid"):
        chunk_source_span(db_session, chunk, replay_source_span=old)


@pytest.mark.asyncio
async def test_legacy_coordinate_must_have_maximum_confidence(db_session, sample_knowledge_base, tmp_path, fake_model_stack):
    chunk, _ = await tied_source(db_session, sample_knowledge_base, tmp_path)
    alternative = ChunkCoordinate(id=str(UUID(int=1)), chunk_id=chunk.id, document_version_id=chunk.document_version_id,
        page_range_json={"start": 2, "end": 2}, bbox_json={"x0": 1.0}, confidence=1.0)
    db_session.add(alternative)
    db_session.flush()
    span = chunk_source_span(db_session, chunk)
    assert span["page_range"] == [2, 2]
    legacy = {**span, "contract_version": "raw_chunk_source_span_v1"}
    alternative.confidence = 0.2
    db_session.flush()
    with pytest.raises(ChunkSourceProvenanceError, match="legacy_coordinate_invalid"):
        chunk_source_span(db_session, chunk, replay_source_span=legacy)


@pytest.mark.asyncio
async def test_postgresql_custom_and_generic_plans_keep_source_address(postgres_tpe_scope, postgres_source_root):
    from app.db import SessionLocal
    with SessionLocal() as db:
        kb = db.get(KnowledgeBase, postgres_tpe_scope["knowledge_base_id"])
        kb.source_root = str(postgres_source_root)
        db.commit()
        chunk, ids = await tied_source(db, kb, postgres_source_root)
        for mode in ("force_custom_plan", "force_generic_plan"):
            db.execute(text("SET LOCAL plan_cache_mode=" + mode))
            for _ in range(7):
                assert chunk_source_span(db, chunk)["structure_node_ids"] == ids[:8]
        db.rollback()

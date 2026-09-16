import numpy as np
import pytest
from sqlalchemy import event, select
from sqlalchemy.orm import Load

from app.models import Chunk, Document, VectorRecord
from app.schemas import SearchFilters
from app.services import context_graph as cg, retrieval_corpus as corpus_module


@pytest.mark.asyncio
async def test_read_projection_preserves_scores_sources_filters_and_protocol_rejection(
    db_session, populated_context_graph, monkeypatch,
):
    kb_id = populated_context_graph["knowledge_base"].id
    record = db_session.scalar(select(VectorRecord).where(VectorRecord.knowledge_base_id == kb_id))
    query = list(record.diagnostics_json["embedding_vector"])
    selected_id = record.chunk_id
    selected = db_session.get(Chunk, selected_id)
    selected.metadata_json = {**selected.metadata_json, "partition": "unit-test-chunk-only"}
    db_session.flush()

    def read():
        db_session.expunge_all()
        dense = cg.dense_chunk_entries(db_session, kb_id, query)
        view = corpus_module.RetrievalCorpus.load(db_session, knowledge_base_id=kb_id, filters=SearchFilters())
        db_session.expunge_all()
        filtered = corpus_module.RetrievalCorpus.load(db_session, knowledge_base_id=kb_id,
            filters=SearchFilters(partition="unit-test-chunk-only"))
        return dense, view, filtered

    with monkeypatch.context() as patch:
        # Use the original full-row ORM query as the independent read reference.
        def full_row(*columns, **kwargs):
            return Load(columns[0].class_)
        patch.setattr(cg, "load_only", full_row)
        patch.setattr(corpus_module, "load_only", full_row)
        expected = read()

    statements = []
    def capture(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("SELECT") and "vector_records JOIN" in statement:
            statements.append(statement.split("FROM", 1)[0])
    event.listen(db_session.bind, "before_cursor_execute", capture)
    try:
        actual = read()
    finally:
        event.remove(db_session.bind, "before_cursor_execute", capture)
    assert actual[0] == expected[0]
    assert statements and all("chunks.metadata_json" not in statement for statement in statements)
    for left, right in zip(actual[1:], expected[1:]):
        assert left.sources == right.sources and left.scope_hash == right.scope_hash
        np.testing.assert_array_equal(left.vectors, right.vectors)
    assert set(actual[2].by_id) == {selected_id}

    # Narrow fields must retain the same inactive-source and vector gates.
    record = db_session.scalar(select(VectorRecord).where(VectorRecord.chunk_id == selected_id))
    doc = db_session.get(Document, db_session.get(Chunk, selected_id).document_id)
    doc.title = "Unit test changed source locator"
    db_session.flush()
    changed = corpus_module.RetrievalCorpus.load(db_session, knowledge_base_id=kb_id, filters=SearchFilters())
    assert changed.scope_hash != actual[1].scope_hash
    doc.is_active = False
    db_session.flush()
    with pytest.raises(cg.DenseChunkCandidateScopeError):
        cg.dense_chunk_entries(db_session, kb_id, query)
    doc.is_active = True
    record.diagnostics_json = {**record.diagnostics_json, "candidate_vector_schema_hash": "unit-test-invalid"}
    db_session.flush()
    with pytest.raises(ValueError, match="retrieval_corpus_vector_provenance_invalid"):
        corpus_module.RetrievalCorpus.load(db_session, knowledge_base_id=kb_id, filters=SearchFilters())

from __future__ import annotations

import pytest


def _add_vector_record(db_session, *, chunk, knowledge_base_id: str, vector: list[float], index: int) -> None:
    from app.models import VectorRecord
    from app.services.chunking import CONTEXTUAL_TEXT_HASH_PROTOCOL_VERSION
    from app.services.context_graph import (
        LOCAL_CONTEXT_HINT_PROTOCOL_VERSION,
        QDRANT_OUTBOX_PROTOCOL_VERSION,
        QDRANT_VECTOR_DISTANCE_METRIC,
        VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION,
        vector_payload_hash,
    )
    from app.services.vector_shadow_lifecycle import (
        ensure_active_vector_runtime_target,
        vector_runtime_diagnostics,
    )

    target = ensure_active_vector_runtime_target(db_session, knowledge_base_id)
    schema = target.schema
    context_hash = f"{index + 10_000:064x}"
    local_hint_hash = f"{index + 20_000:064x}"
    payload_hash = vector_payload_hash(
        vector=vector,
        chunk_id=chunk.id,
        embedding_model=schema.embedding_model,
        embedding_dimension=schema.embedding_dimension,
        vector_distance_metric=QDRANT_VECTOR_DISTANCE_METRIC,
        embedding_text_version=schema.embedding_text_version,
        chunk_schema_version=schema.chunk_schema_version,
        context_hash_protocol_version=CONTEXTUAL_TEXT_HASH_PROTOCOL_VERSION,
        context_hash=context_hash,
        local_hint_protocol_version=LOCAL_CONTEXT_HINT_PROTOCOL_VERSION,
        local_hint_hash=local_hint_hash,
        collection_identity_protocol_version=(
            schema.collection_identity_protocol_version
        ),
        collection_identity_digest=schema.collection_identity_digest,
    )
    db_session.add(
        VectorRecord(
            knowledge_base_id=knowledge_base_id,
            chunk_id=chunk.id,
            qdrant_point_id=chunk.id,
            collection_name=schema.collection_name,
            embedding_model=schema.embedding_model,
            embedding_dimension=schema.embedding_dimension,
            embedding_text_version=schema.embedding_text_version,
            chunk_schema_version=schema.chunk_schema_version,
            payload_hash=payload_hash,
            vector_status="ready",
            diagnostics_json={
                "embedding_vector": vector,
                "context_hash": context_hash,
                "context_hash_protocol_version": CONTEXTUAL_TEXT_HASH_PROTOCOL_VERSION,
                "local_hint_protocol_version": LOCAL_CONTEXT_HINT_PROTOCOL_VERSION,
                "local_hint_hash": local_hint_hash,
                "vector_payload_hash_protocol": VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION,
                "collection_identity_protocol_version": (
                    schema.collection_identity_protocol_version
                ),
                "collection_identity_digest": schema.collection_identity_digest,
                "embedding_dimension": schema.embedding_dimension,
                "vector_distance_metric": QDRANT_VECTOR_DISTANCE_METRIC,
                "chunk_schema_version": schema.chunk_schema_version,
                "qdrant_write_intent_id": f"unit-intent-{index}",
                "qdrant_write_protocol_version": QDRANT_OUTBOX_PROTOCOL_VERSION,
                **vector_runtime_diagnostics(target),
            },
        )
    )


def _add_chunk(
    db_session,
    *,
    knowledge_base_id: str,
    document_id: str,
    document_version_id: str,
    chunk_index: int,
    state: str,
):
    from app.models import Chunk
    from app.services.chunking import text_hash

    text = f"dense active-scope chunk {document_version_id} {chunk_index}"
    chunk = Chunk(
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        document_version_id=document_version_id,
        chunk_version=1,
        chunk_index=chunk_index,
        token_start=0,
        token_end=4,
        char_start=0,
        char_end=len(text),
        text=text,
        text_hash=text_hash(text),
        state=state,
    )
    db_session.add(chunk)
    db_session.flush()
    return chunk


def test_dense_entries_filter_inactive_attempt_vectors_before_top_80(
    db_session,
    sample_knowledge_base,
):
    from app.core.config import get_settings
    from app.models import Document, DocumentVersion
    from app.services.context_graph import dense_chunk_entries

    settings = get_settings()
    dimensions = int(settings.embedding_dimensions)
    query_vector = [1.0, *([0.0] * (dimensions - 1))]
    inactive_vector = list(query_vector)
    active_vector = [0.5, 0.5, *([0.0] * (dimensions - 2))]

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Dense active scope",
        source_path="dense-active-scope.md",
        source_type="markdown",
        checksum="document-checksum",
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    inactive_attempt = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="inactive-attempt",
        storage_path="dense-active-scope-old.md",
        is_active=False,
    )
    active_attempt = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="active-attempt",
        storage_path="dense-active-scope-current.md",
        is_active=True,
    )
    db_session.add_all([inactive_attempt, active_attempt])
    db_session.flush()

    for index in range(81):
        chunk = _add_chunk(
            db_session,
            knowledge_base_id=sample_knowledge_base.id,
            document_id=document.id,
            document_version_id=inactive_attempt.id,
            chunk_index=index,
            state="inactive",
        )
        _add_vector_record(
            db_session,
            chunk=chunk,
            knowledge_base_id=sample_knowledge_base.id,
            vector=inactive_vector,
            index=index,
        )

    active_chunk = _add_chunk(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=active_attempt.id,
        chunk_index=0,
        state="active",
    )
    _add_vector_record(
        db_session,
        chunk=active_chunk,
        knowledge_base_id=sample_knowledge_base.id,
        vector=active_vector,
        index=1000,
    )
    db_session.commit()

    entries = dense_chunk_entries(db_session, sample_knowledge_base.id, query_vector)

    assert set(entries) == {active_chunk.id}
    assert entries[active_chunk.id] == pytest.approx(2**-0.5)


@pytest.mark.parametrize(
    ("inactive_scope", "expected_reason"),
    [
        ("document_version", "document_version_inactive"),
        ("document", "document_inactive"),
    ],
)
def test_dense_entries_fail_closed_when_active_chunk_has_inactive_parent_facts(
    db_session,
    sample_knowledge_base,
    inactive_scope: str,
    expected_reason: str,
):
    from app.core.config import get_settings
    from app.models import Document, DocumentVersion
    from app.services.context_graph import (
        DenseChunkCandidateScopeError,
        dense_chunk_entries,
    )

    dimensions = int(get_settings().embedding_dimensions)
    vector = [1.0, *([0.0] * (dimensions - 1))]
    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Dirty dense scope",
        source_path=f"dirty-dense-scope-{inactive_scope}.md",
        source_type="markdown",
        checksum=f"dirty-document-{inactive_scope}",
        is_active=inactive_scope != "document",
    )
    db_session.add(document)
    db_session.flush()
    attempt = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=f"dirty-attempt-{inactive_scope}",
        storage_path=f"dirty-dense-scope-{inactive_scope}.md",
        is_active=inactive_scope != "document_version",
    )
    db_session.add(attempt)
    db_session.flush()
    chunk = _add_chunk(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=attempt.id,
        chunk_index=0,
        state="active",
    )
    _add_vector_record(
        db_session,
        chunk=chunk,
        knowledge_base_id=sample_knowledge_base.id,
        vector=vector,
        index=2000,
    )
    db_session.commit()

    with pytest.raises(DenseChunkCandidateScopeError) as exc_info:
        dense_chunk_entries(db_session, sample_knowledge_base.id, vector)

    message = str(exc_info.value)
    assert expected_reason in message
    assert chunk.id in message
    assert attempt.id in message

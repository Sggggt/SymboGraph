from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import select


def _document_with_chunk(db_session, knowledge_base_id: str, *, suffix: str, chunk_version: int = 1):
    from app.models import Chunk, ChunkSpan, Document, DocumentVersion
    from app.services.chunking import text_hash

    document = Document(
        knowledge_base_id=knowledge_base_id,
        title=f"Document {suffix}",
        source_path=f"document-{suffix}.md",
        source_type="markdown",
        checksum=f"document-checksum-{suffix}",
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=chunk_version,
        checksum=f"version-checksum-{suffix}",
        storage_path=document.source_path,
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    text = f"Grounded content for {suffix}."
    chunk = Chunk(
        knowledge_base_id=knowledge_base_id,
        document_id=document.id,
        document_version_id=version.id,
        chunk_version=chunk_version,
        chunk_index=0,
        token_start=0,
        token_end=5,
        char_start=0,
        char_end=len(text),
        text=text,
        text_hash=text_hash(text),
        section_path=f"Section {suffix}",
        metadata_json={
            "chunk_schema_version": "chunk_schema_v1",
            "tokenizer_version": "symbograph_regex_tokenizer_v1",
            "chunk_size": 512,
            "chunk_overlap": 80,
        },
        state="active",
    )
    db_session.add(chunk)
    db_session.flush()
    db_session.add(
        ChunkSpan(
            chunk_id=chunk.id,
            document_version_id=version.id,
            char_start=0,
            char_end=len(text),
            token_start=0,
            token_end=5,
            span_type="raw_text",
            section_path=f"Section {suffix}",
        )
    )
    db_session.flush()
    return document, version, chunk


def test_selected_reparse_can_keep_same_knowledge_base_version(db_session, sample_knowledge_base):
    from app.models import DocumentVersion

    document, previous, _ = _document_with_chunk(db_session, sample_knowledge_base.id, suffix="reparse")
    previous.is_active = False
    replacement = DocumentVersion(
        document_id=document.id,
        version=previous.version,
        checksum="replacement-checksum",
        storage_path=document.source_path,
        is_active=True,
    )
    db_session.add(replacement)
    db_session.commit()

    attempts = list(
        db_session.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document.id, DocumentVersion.version == previous.version)
            .order_by(DocumentVersion.created_at.asc())
        ).all()
    )
    assert len(attempts) == 2
    assert sum(1 for attempt in attempts if attempt.is_active) == 1
    assert next(attempt for attempt in attempts if attempt.is_active).id == replacement.id


def test_vector_identity_allows_model_and_dimension_shadow_coexistence(db_session, sample_knowledge_base):
    from app.models import VectorRecord

    _, _, chunk = _document_with_chunk(db_session, sample_knowledge_base.id, suffix="vectors")
    identities = (
        ("embedding-model-a", 8),
        ("embedding-model-b", 8),
        ("embedding-model-a", 16),
    )
    records = [
        VectorRecord(
            knowledge_base_id=sample_knowledge_base.id,
            chunk_id=chunk.id,
            qdrant_point_id=chunk.id,
            collection_name=f"collection-{model}-{dimension}",
            embedding_model=model,
            embedding_dimension=dimension,
            embedding_text_version="contextual_text_v1",
            payload_hash=f"payload-{model}-{dimension}",
            vector_status="ready",
            diagnostics_json={"embedding_vector": [0.1] * dimension},
        )
        for model, dimension in identities
    ]
    db_session.add_all(records)
    db_session.commit()

    persisted = list(db_session.scalars(select(VectorRecord).where(VectorRecord.chunk_id == chunk.id)).all())
    assert {(record.embedding_model, record.embedding_dimension) for record in persisted} == set(identities)


def test_vector_payload_hash_binds_complete_contextual_vector_identity():
    from app.services.context_graph import (
        QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
        QDRANT_VECTOR_DISTANCE_METRIC,
        qdrant_collection_identity_digest,
        vector_payload_hash,
        vector_payload_hash_v2,
    )

    base = {
        "vector": [0.1, 0.2, 0.3],
        "chunk_id": "chunk-a",
        "embedding_model": "embedding-a",
        "embedding_dimension": 3,
        "vector_distance_metric": QDRANT_VECTOR_DISTANCE_METRIC,
        "embedding_text_version": "contextual-text-a",
        "chunk_schema_version": "chunk-schema-a",
        "context_hash_protocol_version": "context-hash-protocol-a",
        "context_hash": "context-hash-a",
        "local_hint_protocol_version": "local-hint-protocol-a",
        "local_hint_hash": "local-hint-hash-a",
        "collection_identity_protocol_version": QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
    }
    base["collection_identity_digest"] = qdrant_collection_identity_digest(
        embedding_model=base["embedding_model"],
        embedding_dimension=base["embedding_dimension"],
        embedding_text_version=base["embedding_text_version"],
        chunk_schema_version=base["chunk_schema_version"],
    )
    baseline = vector_payload_hash(**base)
    assert baseline == "a31151709cac924d29ca13074d4f319271099ca928b6710d1464ad4b1be02f70"
    assert vector_payload_hash_v2(
        vector=base["vector"],
        chunk_id=base["chunk_id"],
        embedding_model=base["embedding_model"],
        embedding_dimension=base["embedding_dimension"],
        embedding_text_version=base["embedding_text_version"],
    ) == "2bf29a461bf3b6bff198cf26cfc2ab583807eb8e2f0f94318193644aa23a297e"

    float32_round_trip = [
        0.10000000149011612,
        0.20000000298023224,
        0.30000001192092896,
    ]
    assert vector_payload_hash(**{**base, "vector": float32_round_trip}) == baseline
    assert vector_payload_hash(**{**base, "vector": [-0.0, 0.2, 0.3]}) == vector_payload_hash(
        **{**base, "vector": [0.0, 0.2, 0.3]}
    )

    def with_collection_identity(**changes):
        changed = {**base, **changes}
        changed["collection_identity_digest"] = qdrant_collection_identity_digest(
            embedding_model=changed["embedding_model"],
            embedding_dimension=changed["embedding_dimension"],
            embedding_text_version=changed["embedding_text_version"],
            chunk_schema_version=changed["chunk_schema_version"],
        )
        return changed

    for changed in (
        {**base, "vector": [0.1, 0.2, 0.4]},
        {**base, "chunk_id": "chunk-b"},
        with_collection_identity(embedding_model="embedding-b"),
        with_collection_identity(embedding_text_version="contextual-text-b"),
        with_collection_identity(chunk_schema_version="chunk-schema-b"),
        {**base, "context_hash_protocol_version": "context-hash-protocol-b"},
        {**base, "context_hash": "context-hash-b"},
        {**base, "local_hint_protocol_version": "local-hint-protocol-b"},
        {**base, "local_hint_hash": "local-hint-hash-b"},
    ):
        assert vector_payload_hash(**changed) != baseline

    with pytest.raises(ValueError, match="cosine"):
        vector_payload_hash(**{**base, "vector_distance_metric": "dot"})
    with pytest.raises(ValueError, match="not active"):
        vector_payload_hash(
            **{**base, "collection_identity_protocol_version": "collection-identity-v4"}
        )
    with pytest.raises(ValueError, match="not canonical"):
        vector_payload_hash(**{**base, "collection_identity_digest": "0" * 64})

    with pytest.raises(ValueError, match="does not match"):
        vector_payload_hash(**with_collection_identity(embedding_dimension=4))

    for invalid_value in (float("nan"), float("inf"), float("-inf"), True, "0.2"):
        with pytest.raises(ValueError, match="finite real number"):
            vector_payload_hash(**{**base, "vector": [0.1, invalid_value, 0.3]})
    with pytest.raises(ValueError, match="fit finite IEEE-754 binary32"):
        vector_payload_hash(**{**base, "vector": [0.1, 3.5e38, 0.3]})


def test_qdrant_collection_identity_uses_frozen_canonical_digest():
    from app.services.context_graph import (
        QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
        qdrant_collection_identity_digest,
        qdrant_collection_name,
    )

    identity = {
        "embedding_model": "vendor/model:v1",
        "embedding_dimension": 8,
        "embedding_text_version": "contextual_text_v2",
        "chunk_schema_version": "chunk_schema_v1",
    }
    assert QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION == "qdrant_collection_identity_u64be_utf8_sha256_v2"
    assert qdrant_collection_identity_digest(**identity) == (
        "8872d1d9d1360c78965a26de50341d03b64ef4cb603cfa55bcd14a6933203290"
    )
    collection_name = qdrant_collection_name(**identity)
    assert collection_name.endswith("_8872d1d9d1360c78965a26de50341d03b64ef4cb603cfa55bcd14a6933203290")
    assert len(collection_name) <= 180
    for changed in (
        {**identity, "embedding_model": "vendor/model:v2"},
        {**identity, "embedding_dimension": 16},
        {**identity, "embedding_text_version": "contextual_text_v3"},
        {**identity, "chunk_schema_version": "chunk_schema_v2"},
    ):
        assert qdrant_collection_name(**changed) != collection_name


def test_qdrant_collection_identity_separates_sanitized_model_collision():
    from app.services.context_graph import qdrant_collection_name

    common = {
        "embedding_dimension": 8,
        "embedding_text_version": "contextual_text_v2",
        "chunk_schema_version": "chunk_schema_v1",
    }
    slash_model = qdrant_collection_name(embedding_model="vendor/model:v1", **common)
    question_model = qdrant_collection_name(embedding_model="vendor?model:v1", **common)

    assert slash_model.rsplit("_", 1)[0] == question_model.rsplit("_", 1)[0]
    assert slash_model != question_model


def test_qdrant_collection_identity_separates_differences_beyond_readable_prefix():
    from app.services.context_graph import qdrant_collection_name

    long_common_prefix = "vendor-" + ("a" * 256)
    common = {
        "embedding_dimension": 8,
        "embedding_text_version": "contextual_text_v2",
        "chunk_schema_version": "chunk_schema_v1",
    }
    left = qdrant_collection_name(embedding_model=f"{long_common_prefix}-left", **common)
    right = qdrant_collection_name(embedding_model=f"{long_common_prefix}-right", **common)

    assert left.rsplit("_", 1)[0] == right.rsplit("_", 1)[0]
    assert left != right
    assert len(left) <= 180
    assert len(right) <= 180


@pytest.mark.asyncio
async def test_contextual_index_write_is_idempotent_for_same_model(
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from sqlalchemy import func

    from app.models import ChunkContextText, VectorRecord
    from app.services.context_graph import write_contextual_indexes

    _, _, chunk = _document_with_chunk(db_session, sample_knowledge_base.id, suffix="idempotent-index")
    for _ in range(2):
        result = await write_contextual_indexes(
            db_session,
            knowledge_base=sample_knowledge_base,
            chunks=[chunk],
        )
        assert result["vectors"] == 1

    assert db_session.scalar(select(func.count(ChunkContextText.id)).where(ChunkContextText.chunk_id == chunk.id)) == 1
    assert db_session.scalar(select(func.count(VectorRecord.id)).where(VectorRecord.chunk_id == chunk.id)) == 1
    record = db_session.scalar(select(VectorRecord).where(VectorRecord.chunk_id == chunk.id))
    assert record.vector_status == "ready"
    assert record.payload_hash
    assert record.diagnostics_json["vector_payload_hash_protocol"] == "vector_payload_hash_v3"
    assert record.diagnostics_json["collection_identity_protocol_version"] == (
        "qdrant_collection_identity_u64be_utf8_sha256_v2"
    )
    assert record.diagnostics_json["vector_distance_metric"] == "cosine"
    assert record.diagnostics_json["embedding_dimension"] == record.embedding_dimension
    assert record.diagnostics_json["chunk_schema_version"]
    assert len(record.diagnostics_json["collection_identity_digest"]) == 64
    stored_point = fake_model_stack["VectorStore"].points[chunk.id]
    assert stored_point["payload"]["vector_payload_hash"] == record.payload_hash
    assert stored_point["payload"]["embedding_model"] == record.embedding_model
    assert stored_point["payload"]["collection_identity_digest"] == record.diagnostics_json[
        "collection_identity_digest"
    ]


@pytest.mark.asyncio
async def test_contextual_index_rejects_provider_dimension_mismatch_before_vector_side_effects(
    db_session,
    sample_knowledge_base,
    fake_model_stack,
    monkeypatch,
):
    from sqlalchemy import func, select

    from app.core.config import get_settings
    from app.models import IngestionCompensationLog, VectorRecord
    from app.services import context_graph

    _, _, chunk = _document_with_chunk(
        db_session,
        sample_knowledge_base.id,
        suffix="dimension-mismatch",
    )
    expected_dimension = int(get_settings().embedding_dimensions)

    class WrongDimensionEmbeddingProvider:
        async def embed_texts(self, texts, text_type="document"):
            return [[0.1] * (expected_dimension + 1) for _text in texts]

    monkeypatch.setattr(context_graph, "EmbeddingProvider", WrongDimensionEmbeddingProvider)

    with pytest.raises(RuntimeError, match="do not match EMBEDDING_DIMENSIONS"):
        await context_graph.write_contextual_indexes(
            db_session,
            knowledge_base=sample_knowledge_base,
            chunks=[chunk],
        )

    assert db_session.scalar(
        select(func.count(VectorRecord.id)).where(VectorRecord.chunk_id == chunk.id)
    ) == 0
    assert db_session.scalar(select(func.count(IngestionCompensationLog.id))) == 0
    assert fake_model_stack["VectorStore"].points == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf"), True, "0.1", None],
    ids=["nan", "positive-infinity", "negative-infinity", "bool", "numeric-string", "zero-vector"],
)
async def test_contextual_index_rejects_non_finite_or_coerced_vectors_before_side_effects(
    db_session,
    sample_knowledge_base,
    fake_model_stack,
    monkeypatch,
    invalid_value,
):
    from sqlalchemy import func, select

    from app.core.config import get_settings
    from app.models import IngestionCompensationLog, VectorRecord
    from app.services import context_graph

    _, _, chunk = _document_with_chunk(
        db_session,
        sample_knowledge_base.id,
        suffix=f"invalid-vector-{type(invalid_value).__name__}",
    )
    expected_dimension = int(get_settings().embedding_dimensions)

    class InvalidValueEmbeddingProvider:
        async def embed_texts(self, texts, text_type="document"):
            vector = [0.0] * expected_dimension if invalid_value is None else [0.1] * expected_dimension
            if invalid_value is not None:
                vector[0] = invalid_value
            return [list(vector) for _text in texts]

    monkeypatch.setattr(context_graph, "EmbeddingProvider", InvalidValueEmbeddingProvider)

    with pytest.raises(RuntimeError, match="invalid numeric values"):
        await context_graph.write_contextual_indexes(
            db_session,
            knowledge_base=sample_knowledge_base,
            chunks=[chunk],
        )

    assert db_session.scalar(
        select(func.count(VectorRecord.id)).where(VectorRecord.chunk_id == chunk.id)
    ) == 0
    assert db_session.scalar(select(func.count(IngestionCompensationLog.id))) == 0
    assert fake_model_stack["VectorStore"].points == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_count", [1, 2])
async def test_tiny_knowledge_base_preserves_l1_l2_l3_and_all_graph_layers(
    db_session,
    sample_knowledge_base,
    fake_model_stack,
    chunk_count,
):
    from sqlalchemy import func

    from app.models import CoarseConcept, MidConcept, RQPrefix
    from app.services.context_graph import (
        latest_coarse_state,
        latest_mid_state,
        latest_relation_state,
        rebuild_context_graph,
        write_contextual_indexes,
    )

    chunks = [
        _document_with_chunk(
            db_session,
            sample_knowledge_base.id,
            suffix=f"tiny-{chunk_count}-{index}",
        )[2]
        for index in range(chunk_count)
    ]
    await write_contextual_indexes(
        db_session,
        knowledge_base=sample_knowledge_base,
        chunks=chunks,
    )
    context_state = await rebuild_context_graph(
        db_session,
        sample_knowledge_base.id,
        emit_heartbeats=False,
    )
    db_session.flush()

    relation_state = latest_relation_state(db_session, sample_knowledge_base.id)
    mid_state = latest_mid_state(db_session, sample_knowledge_base.id)
    coarse_state = latest_coarse_state(db_session, sample_knowledge_base.id)
    prefixes = list(db_session.scalars(select(RQPrefix).where(RQPrefix.graph_state_id == relation_state.id)).all())
    mid_count = db_session.scalar(select(func.count(MidConcept.id)).where(MidConcept.concept_state_id == mid_state.id))
    coarse_count = db_session.scalar(
        select(func.count(CoarseConcept.id)).where(CoarseConcept.coarse_state_id == coarse_state.id)
    )

    assert {prefix.rq_level for prefix in prefixes} == {1, 2, 3}
    assert 1 <= mid_count <= chunk_count
    assert 1 <= coarse_count <= mid_count
    if chunk_count > 1:
        assert mid_count < sum(
            1 for prefix in prefixes if prefix.rq_level == 3
        )
    assert context_state.stats_json["chunks"] == chunk_count
    assert context_state.stats_json["mid_concepts"] == mid_count
    assert context_state.stats_json["coarse_concepts"] == coarse_count


def test_chunk_version_state_covers_all_active_documents(db_session, sample_knowledge_base):
    from app.models import Chunk, ChunkVersion
    from app.services.context_graph import active_chunks_query, compute_chunk_scope_hash, write_chunk_version_state

    _, _, first_chunk = _document_with_chunk(db_session, sample_knowledge_base.id, suffix="first")
    state = write_chunk_version_state(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        chunk_version=1,
        chunks=[first_chunk],
        chunk_size=512,
        chunk_overlap=80,
    )
    db_session.flush()
    first_scope_hash = state.stats_json["active_chunk_scope_hash"]
    assert state.stats_json["chunk_count"] == 1

    _, _, second_chunk = _document_with_chunk(db_session, sample_knowledge_base.id, suffix="second")
    updated = write_chunk_version_state(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        chunk_version=1,
        chunks=[second_chunk],
        chunk_size=512,
        chunk_overlap=80,
    )
    db_session.flush()

    active_chunks = list(db_session.scalars(active_chunks_query(sample_knowledge_base.id)).all())
    persisted = db_session.scalar(
        select(ChunkVersion).where(
            ChunkVersion.knowledge_base_id == sample_knowledge_base.id,
            ChunkVersion.chunk_version == 1,
        )
    )
    assert updated.id == state.id == persisted.id
    assert {chunk.id for chunk in active_chunks} == {first_chunk.id, second_chunk.id}
    assert persisted.stats_json["chunk_count"] == 2
    assert persisted.stats_json["active_chunk_versions"] == {"1": 2}
    assert persisted.stats_json["active_chunk_scope_hash"] == compute_chunk_scope_hash(active_chunks)
    assert persisted.stats_json["active_chunk_scope_hash"] != first_scope_hash
    assert persisted.diagnostics_json["scope"] == "knowledge_base_active_chunks"


def _scope_chunk(**overrides):
    values = {
        "knowledge_base_id": "kb-1",
        "id": "chunk-1",
        "document_id": "document-1",
        "document_version_id": "document-version-1",
        "chunk_version": 1,
        "chunk_index": 0,
        "char_start": 10,
        "char_end": 20,
        "token_start": 2,
        "token_end": 7,
        "section_path": "Root / Section",
        "page_start": 3,
        "page_end": 4,
        "text_hash": "a" * 64,
        "metadata_json": {
            "chunk_schema_version": "chunk_schema_v1",
            "tokenizer_version": "symbograph_regex_tokenizer_v1",
            "chunk_size": 512,
            "chunk_overlap": 80,
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_chunk_scope_hash_is_order_independent_and_binds_complete_address_and_protocol():
    from app.services.context_graph import compute_chunk_scope_hash

    first = _scope_chunk()
    second = _scope_chunk(id="chunk-2", chunk_index=1, char_start=18, char_end=28)
    baseline = compute_chunk_scope_hash([first, second])
    assert compute_chunk_scope_hash([second, first]) == baseline

    mutations = [
        {"chunk_index": 9},
        {"token_start": 3},
        {"token_end": 8},
        {"section_path": "Root / Different"},
        {"page_start": 5},
        {"page_end": 6},
        {
            "metadata_json": {
                **first.metadata_json,
                "chunk_schema_version": "chunk_schema_v2",
            }
        },
        {
            "metadata_json": {
                **first.metadata_json,
                "tokenizer_version": "tokenizer_v2",
            }
        },
        {"metadata_json": {**first.metadata_json, "chunk_size": 256}},
        {"metadata_json": {**first.metadata_json, "chunk_overlap": 32}},
    ]
    for mutation in mutations:
        assert compute_chunk_scope_hash([_scope_chunk(**mutation), second]) != baseline


def test_chunk_version_state_records_mixed_active_protocol_distribution(db_session, sample_knowledge_base):
    from app.services.context_graph import write_chunk_version_state

    _, _, first_chunk = _document_with_chunk(
        db_session,
        sample_knowledge_base.id,
        suffix="protocol-v1",
        chunk_version=1,
    )
    _, _, second_chunk = _document_with_chunk(
        db_session,
        sample_knowledge_base.id,
        suffix="protocol-v2",
        chunk_version=2,
    )
    second_chunk.metadata_json = {
        "chunk_schema_version": "chunk_schema_v2",
        "tokenizer_version": "tokenizer_v2",
        "chunk_size": 256,
        "chunk_overlap": 32,
    }
    state = write_chunk_version_state(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        chunk_version=2,
        chunks=[second_chunk],
        chunk_size=256,
        chunk_overlap=32,
        chunk_schema_version="chunk_schema_v2",
        tokenizer_version="tokenizer_v2",
    )
    db_session.flush()

    assert state.diagnostics_json["protocol_version"] == "chunk_version_active_scope_state_v2"
    assert state.diagnostics_json["active_protocol_mixed"] is True
    assert state.diagnostics_json["active_protocol_descriptor_count"] == 2
    assert state.diagnostics_json["active_protocol_missing_chunk_count"] == 0
    assert state.stats_json["active_chunk_versions"] == {"1": 1, "2": 1}
    assert len(state.stats_json["active_version_protocol_distribution"]) == 2
    assert state.tokenizer_version == "tokenizer_v2"
    assert state.diagnostics_json["target_build_descriptor"]["chunk_size"] == 256

    previous_hash = state.state_hash
    first_chunk.metadata_json = {
        **dict(first_chunk.metadata_json or {}),
        "chunk_overlap": 40,
    }
    refreshed = write_chunk_version_state(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        chunk_version=2,
        chunks=[second_chunk],
        chunk_size=256,
        chunk_overlap=32,
        chunk_schema_version="chunk_schema_v2",
        tokenizer_version="tokenizer_v2",
    )
    assert refreshed.state_hash != previous_hash

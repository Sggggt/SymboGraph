from __future__ import annotations

import struct
from contextlib import nullcontext
from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy import select


def _canonical_reconcile_collection_name() -> str:
    from app.services.chunking import CHUNK_SCHEMA_VERSION
    from app.services.context_graph import qdrant_collection_name

    return qdrant_collection_name(
        embedding_model="reconcile-embedding-v1",
        embedding_dimension=3,
        embedding_text_version="reconcile-text-v1",
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
    )


def _document_with_chunk(db_session, knowledge_base_id: str, suffix: str):
    from app.models import Chunk, Document, DocumentVersion
    from app.services.chunking import text_hash

    document = Document(
        knowledge_base_id=knowledge_base_id,
        title=f"Reconcile {suffix}",
        source_path=f"reconcile-{suffix}.md",
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
    content = f"Grounded reconciliation content for {suffix}."
    chunk = Chunk(
        knowledge_base_id=knowledge_base_id,
        document_id=document.id,
        document_version_id=version.id,
        chunk_version=1,
        chunk_index=0,
        token_start=0,
        token_end=5,
        char_start=0,
        char_end=len(content),
        text=content,
        text_hash=text_hash(content),
        section_path=f"Section {suffix}",
        state="active",
    )
    db_session.add(chunk)
    db_session.flush()
    return chunk


def _add_committed_vector(
    db_session,
    *,
    knowledge_base_id: str,
    chunk,
    collection_name: str,
    vector: list[float] | None = None,
):
    from app.models import IngestionCompensationLog, VectorRecord
    from app.services.context_graph import (
        QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
        QDRANT_VECTOR_DISTANCE_METRIC,
        VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION,
        qdrant_collection_identity_digest,
        qdrant_collection_name,
        vector_payload_hash,
    )
    from app.services.chunking import CHUNK_SCHEMA_VERSION
    from app.services.qdrant_outbox import (
        QDRANT_OUTBOX_PROTOCOL_VERSION,
        QDRANT_UPSERT_OPERATION,
        _outbox_payload_hash,
        _outbox_protocol_contract,
    )
    from app.services.vector_store import canonical_embedding_vector

    vector = canonical_embedding_vector(
        list([0.25, 0.5, 0.75] if vector is None else vector),
        source="vector reconcile fixture",
    )
    embedding_model = "reconcile-embedding-v1"
    embedding_text_version = "reconcile-text-v1"
    identity_digest = qdrant_collection_identity_digest(
        embedding_model=embedding_model,
        embedding_dimension=len(vector),
        embedding_text_version=embedding_text_version,
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
    )
    assert collection_name == qdrant_collection_name(
        embedding_model=embedding_model,
        embedding_dimension=len(vector),
        embedding_text_version=embedding_text_version,
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
    )
    context_hash = "reconcile-context-hash-v1"
    context_hash_protocol_version = "reconcile-context-hash-protocol-v1"
    local_hint_protocol_version = "reconcile-local-hint-protocol-v1"
    local_hint_hash = "reconcile-local-hint-hash-v1"
    payload_hash = vector_payload_hash(
        vector=vector,
        chunk_id=str(chunk.id),
        embedding_model=embedding_model,
        embedding_dimension=len(vector),
        vector_distance_metric=QDRANT_VECTOR_DISTANCE_METRIC,
        embedding_text_version=embedding_text_version,
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
        context_hash_protocol_version=context_hash_protocol_version,
        context_hash=context_hash,
        local_hint_protocol_version=local_hint_protocol_version,
        local_hint_hash=local_hint_hash,
        collection_identity_protocol_version=QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
        collection_identity_digest=identity_digest,
    )
    intent_id = str(uuid4())
    payload = {
        "knowledge_base_id": knowledge_base_id,
        "chunk_id": str(chunk.id),
        "embedding_model": embedding_model,
        "embedding_dimension": len(vector),
        "vector_distance_metric": QDRANT_VECTOR_DISTANCE_METRIC,
        "embedding_text_version": embedding_text_version,
        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
        "context_hash": context_hash,
        "context_hash_protocol_version": context_hash_protocol_version,
        "local_hint_protocol_version": local_hint_protocol_version,
        "local_hint_hash": local_hint_hash,
        "collection_identity_protocol_version": QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
        "collection_identity_digest": identity_digest,
        "vector_payload_hash": payload_hash,
        "vector_payload_hash_protocol": VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION,
        "qdrant_write_intent_id": intent_id,
        "qdrant_write_protocol_version": QDRANT_OUTBOX_PROTOCOL_VERSION,
    }
    point = {"id": str(chunk.id), "vector": vector, "payload": payload}
    target_points = [point]
    outbox_contract = _outbox_protocol_contract(QDRANT_OUTBOX_PROTOCOL_VERSION)
    owner = IngestionCompensationLog(
        id=intent_id,
        knowledge_base_id=knowledge_base_id,
        operation=QDRANT_UPSERT_OPERATION,
        target_ids_json=[str(chunk.id)],
        payload_json={
            "protocol_version": QDRANT_OUTBOX_PROTOCOL_VERSION,
            "envelope_schema_version": outbox_contract["envelope_schema_version"],
            "envelope_schema_hash": outbox_contract["envelope_schema_hash"],
            "canonical_bytes_version": outbox_contract["canonical_bytes_version"],
            "intent_id": intent_id,
            "collection_name": collection_name,
            "target_points": target_points,
            "before_points": [],
            "target_payload_hash": _outbox_payload_hash(
                QDRANT_OUTBOX_PROTOCOL_VERSION,
                target_points,
            ),
            "before_image_hash": _outbox_payload_hash(
                QDRANT_OUTBOX_PROTOCOL_VERSION,
                [],
            ),
        },
        status="committed",
    )
    record = VectorRecord(
        knowledge_base_id=knowledge_base_id,
        chunk_id=chunk.id,
        qdrant_point_id=chunk.id,
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_dimension=len(vector),
        embedding_text_version=embedding_text_version,
        payload_hash=payload_hash,
        vector_status="ready",
        diagnostics_json={
            "embedding_vector": vector,
            "context_hash": payload["context_hash"],
            "context_hash_protocol_version": payload[
                "context_hash_protocol_version"
            ],
            "local_hint_protocol_version": payload[
                "local_hint_protocol_version"
            ],
            "local_hint_hash": payload["local_hint_hash"],
            "vector_payload_hash_protocol": VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION,
            "collection_identity_protocol_version": QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
            "collection_identity_digest": identity_digest,
            "vector_distance_metric": QDRANT_VECTOR_DISTANCE_METRIC,
            "embedding_dimension": len(vector),
            "chunk_schema_version": CHUNK_SCHEMA_VERSION,
            "qdrant_write_intent_id": intent_id,
            "qdrant_write_protocol_version": QDRANT_OUTBOX_PROTOCOL_VERSION,
        },
    )
    db_session.add_all([owner, record])
    db_session.flush()
    return record, point


def test_reconcile_scopes_shared_collection_by_knowledge_base(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from app.models import KnowledgeBase
    from app.services import maintenance, qdrant_outbox

    other_kb = KnowledgeBase(
        name="Other reconciliation KB",
        description="shared collection scope",
        source_root="unit-tests-other-reconcile",
    )
    db_session.add(other_kb)
    db_session.flush()
    collection_name = _canonical_reconcile_collection_name()
    first_chunk = _document_with_chunk(
        db_session,
        sample_knowledge_base.id,
        f"first-{uuid4().hex}",
    )
    second_chunk = _document_with_chunk(
        db_session,
        other_kb.id,
        f"second-{uuid4().hex}",
    )
    first_record, first_point = _add_committed_vector(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        chunk=first_chunk,
        collection_name=collection_name,
    )
    second_record, second_point = _add_committed_vector(
        db_session,
        knowledge_base_id=other_kb.id,
        chunk=second_chunk,
        collection_name=collection_name,
    )
    db_session.commit()

    snapshots = {
        str(sample_knowledge_base.id): first_point,
        str(other_kb.id): second_point,
    }
    calls: list[tuple[str, tuple[str, ...]]] = []
    expected_collection_name = collection_name

    class FakeVectorStore:
        def __init__(self, _kb_name, *, collection_name, create_if_missing):
            assert collection_name == expected_collection_name
            assert create_if_missing is False

        def reconciliation_snapshot(
            self,
            knowledge_base_id,
            expected_point_ids,
            *,
            expected_vector_size,
        ):
            assert expected_vector_size == 3
            calls.append((str(knowledge_base_id), tuple(expected_point_ids)))
            point = snapshots[str(knowledge_base_id)]
            return {
                "points": [point],
                "scanned_ids": [point["id"]],
                "scan_truncated": False,
            }

    monkeypatch.setattr(maintenance, "VectorStore", FakeVectorStore)
    lock_calls: list[str] = []

    def fake_reconcile_lock(_db, knowledge_base_id):
        lock_calls.append(str(knowledge_base_id))
        return nullcontext()

    monkeypatch.setattr(
        qdrant_outbox,
        "qdrant_outbox_reconcile_lock",
        fake_reconcile_lock,
    )

    preview = maintenance.reconcile_vector_store_sync(db_session, dry_run=True)

    assert preview["checked_records"] == 2
    assert preview["checked_collections"] == 2
    assert preview["checked_collection_scopes"] == 2
    assert preview["checked_unique_collections"] == 1
    assert preview["proposed_records"] == 0
    assert preview["missing_points"] == 0
    assert preview["stale_points"] == 0
    assert set(calls) == {
        (str(sample_knowledge_base.id), (str(first_chunk.id),)),
        (str(other_kb.id), (str(second_chunk.id),)),
    }
    assert [item[0] for item in calls] == sorted(item[0] for item in calls)
    assert lock_calls == sorted(
        [str(sample_knowledge_base.id), str(other_kb.id)]
    )
    assert preview["processed_knowledge_bases"] == 2
    assert preview["processed_record_batches"] == 2
    assert first_record.vector_status == second_record.vector_status == "ready"

    calls.clear()
    lock_calls.clear()
    applied = maintenance.reconcile_vector_store_sync(db_session, dry_run=False)
    assert applied["marked_records"] == 0
    assert len(calls) == 2
    assert lock_calls == sorted(
        [str(sample_knowledge_base.id), str(other_kb.id)]
    )
    assert first_record.vector_status == second_record.vector_status == "ready"


def test_reconcile_marks_real_point_payload_identity_drift_stale(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from app.services import maintenance

    chunk = _document_with_chunk(
        db_session,
        sample_knowledge_base.id,
        f"payload-drift-{uuid4().hex}",
    )
    record, point = _add_committed_vector(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        chunk=chunk,
        collection_name=_canonical_reconcile_collection_name(),
    )
    db_session.commit()
    record.diagnostics_json = {
        **dict(record.diagnostics_json or {}),
        "context_hash": "postgres-context-hash-drift",
    }
    db_session.commit()
    tampered = {
        **point,
        "payload": {
            **point["payload"],
            "collection_identity_digest": "f" * 64,
            "vector_payload_hash": "e" * 64,
        },
    }

    class FakeVectorStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def reconciliation_snapshot(
            self,
            _knowledge_base_id,
            _expected_point_ids,
            *,
            expected_vector_size,
        ):
            assert expected_vector_size == 3
            return {
                "points": [tampered],
                "scanned_ids": [tampered["id"], "legacy-orphan-point"],
                "scan_truncated": True,
                "collection_schema_error": "vector size is 4, expected 3",
            }

    monkeypatch.setattr(maintenance, "VectorStore", FakeVectorStore)

    preview = maintenance.reconcile_vector_store_sync(
        db_session,
        sample_knowledge_base.id,
        dry_run=True,
    )
    reasons = preview["stale_reasons_by_record"][str(record.id)]
    assert "qdrant_collection_identity_digest_mismatch" in reasons
    assert "qdrant_vector_payload_hash_mismatch" in reasons
    assert "qdrant_context_hash_mismatch" in reasons
    assert "committed_owner_target_payload_hash_mismatch" in reasons
    assert "qdrant_collection_schema_mismatch" in reasons
    assert preview["identity_mismatch_points"] == 1
    assert preview["payload_mismatch_points"] == 1
    assert preview["owner_mismatch_points"] == 1
    assert preview["orphan_points"] == 1
    assert preview["scan_truncated_collection_scopes"] == 1
    assert preview["collection_schema_mismatch_scopes"] == 1
    assert preview["legacy_orphan_collection_inventory"]["scanned"] is False
    assert preview["proposed_records"] == 1
    assert record.vector_status == "ready"

    applied = maintenance.reconcile_vector_store_sync(
        db_session,
        sample_knowledge_base.id,
        dry_run=False,
    )
    assert applied["marked_records"] == 1
    assert db_session.scalar(select(type(record)).where(type(record).id == record.id)).vector_status == "stale"


def test_reconcile_detects_coordinated_diagnostics_and_qdrant_vector_tamper(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from app.services import maintenance

    chunk = _document_with_chunk(
        db_session,
        sample_knowledge_base.id,
        f"coordinated-vector-drift-{uuid4().hex}",
    )
    record, point = _add_committed_vector(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        chunk=chunk,
        collection_name=_canonical_reconcile_collection_name(),
    )
    db_session.commit()
    coordinated_vector = [0.75, 0.25, 0.5]
    coordinated_context_hash = "coordinated-context-hash-drift"
    record.diagnostics_json = {
        **dict(record.diagnostics_json or {}),
        "embedding_vector": coordinated_vector,
        "context_hash": coordinated_context_hash,
    }
    db_session.commit()
    tampered_point = {
        **point,
        "vector": coordinated_vector,
        "payload": {
            **point["payload"],
            "context_hash": coordinated_context_hash,
        },
    }

    class FakeVectorStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def reconciliation_snapshot(
            self,
            _knowledge_base_id,
            _expected_point_ids,
            *,
            expected_vector_size,
        ):
            assert expected_vector_size == 3
            return {
                "points": [tampered_point],
                "scanned_ids": [tampered_point["id"]],
                "scan_truncated": False,
                "collection_schema_error": None,
            }

    monkeypatch.setattr(maintenance, "VectorStore", FakeVectorStore)

    preview = maintenance.reconcile_vector_store_sync(
        db_session,
        sample_knowledge_base.id,
        dry_run=True,
    )

    reasons = preview["stale_reasons_by_record"][str(record.id)]
    assert "postgres_vector_payload_hash_recompute_mismatch" in reasons
    assert "qdrant_vector_payload_hash_recompute_mismatch" in reasons
    assert "committed_owner_target_vector_mismatch" in reasons
    assert "committed_owner_diagnostics_vector_mismatch" in reasons
    assert "committed_owner_target_payload_hash_mismatch" in reasons
    assert "qdrant_vector_mismatch" not in reasons
    assert preview["proposed_records"] == 1
    assert preview["vector_mismatch_points"] == 1
    assert record.vector_status == "ready"


def test_reconcile_accepts_qdrant_float32_round_trip_when_committed_target_matches(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from app.services import maintenance

    chunk = _document_with_chunk(
        db_session,
        sample_knowledge_base.id,
        f"float32-round-trip-{uuid4().hex}",
    )
    source_vector = [0.1, 0.2, 0.3]
    record, point = _add_committed_vector(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        chunk=chunk,
        collection_name=_canonical_reconcile_collection_name(),
        vector=source_vector,
    )
    db_session.commit()
    qdrant_vector = [
        struct.unpack("!f", struct.pack("!f", value))[0]
        for value in source_vector
    ]
    assert qdrant_vector != source_vector
    observed_point = {**point, "vector": qdrant_vector}

    class FakeVectorStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def reconciliation_snapshot(
            self,
            _knowledge_base_id,
            _expected_point_ids,
            *,
            expected_vector_size,
        ):
            assert expected_vector_size == 3
            return {
                "points": [observed_point],
                "scanned_ids": [observed_point["id"]],
                "scan_truncated": False,
                "collection_schema_error": None,
            }

    monkeypatch.setattr(maintenance, "VectorStore", FakeVectorStore)

    preview = maintenance.reconcile_vector_store_sync(
        db_session,
        sample_knowledge_base.id,
        dry_run=True,
    )

    assert str(record.id) not in preview["stale_reasons_by_record"]
    assert preview["proposed_records"] == 0
    assert preview["vector_mismatch_points"] == 0
    assert record.vector_status == "ready"


def test_vector_store_reconcile_reads_are_bounded():
    from app.services.vector_store import VectorStore

    class FakeClient:
        def __init__(self):
            self.retrieve_sizes: list[int] = []
            self.scroll_limits: list[int] = []

        def get_collections(self):
            return SimpleNamespace(collections=[SimpleNamespace(name="bounded")])

        def retrieve(self, *, ids, **_kwargs):
            self.retrieve_sizes.append(len(ids))
            return [
                SimpleNamespace(id=point_id, vector=[1.0], payload={"knowledge_base_id": "kb"})
                for point_id in ids
            ]

        def scroll(self, *, limit, offset, **_kwargs):
            self.scroll_limits.append(limit)
            start = int(offset or 0)
            points = [SimpleNamespace(id=f"point-{index}") for index in range(start, start + limit)]
            return points, start + limit

    store = object.__new__(VectorStore)
    store.collection = "bounded"
    store.client = FakeClient()
    store.settings = SimpleNamespace(enable_model_fallback=False)

    points = store.get_points_batched([f"expected-{index}" for index in range(513)])
    inventory = store.list_ids_bounded("kb", max_points=3, page_size=2)

    assert len(points) == 513
    assert store.client.retrieve_sizes == [256, 256, 1]
    assert inventory == {
        "ids": ["point-0", "point-1", "point-2"],
        "truncated": True,
        "scanned_points": 3,
        "max_points": 3,
    }
    assert store.client.scroll_limits == [2, 2]


def test_vector_store_reconcile_reports_collection_schema_mismatch_without_creating():
    from app.services.vector_store import VectorStore

    class FakeClient:
        def get_collections(self):
            return SimpleNamespace(collections=[SimpleNamespace(name="wrong-schema")])

        def get_collection(self, *, collection_name):
            assert collection_name == "wrong-schema"
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=4, distance="Dot")
                    )
                )
            )

        def retrieve(self, *, ids, **_kwargs):
            return [
                SimpleNamespace(id=point_id, vector=[1.0] * 4, payload={})
                for point_id in ids
            ]

        def scroll(self, **_kwargs):
            return [SimpleNamespace(id="point-1")], None

    store = object.__new__(VectorStore)
    store.collection = "wrong-schema"
    store.client = FakeClient()
    store.settings = SimpleNamespace(enable_model_fallback=False)

    snapshot = store.reconciliation_snapshot(
        "kb",
        ["point-1"],
        expected_vector_size=3,
    )

    assert len(snapshot["points"]) == 1
    assert "vector size is 4" in snapshot["collection_schema_error"]
    assert snapshot["scan_truncated"] is False


def test_reconcile_postgres_batches_and_diagnostic_samples_are_bounded(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from app.services import maintenance

    collection_name = _canonical_reconcile_collection_name()
    records = []
    for index in range(257):
        chunk = _document_with_chunk(
            db_session,
            sample_knowledge_base.id,
            f"bounded-{index}-{uuid4().hex}",
        )
        record, _point = _add_committed_vector(
            db_session,
            knowledge_base_id=sample_knowledge_base.id,
            chunk=chunk,
            collection_name=collection_name,
        )
        records.append(record)
    db_session.commit()

    retrieval_batches: list[int] = []
    orphan_inventory = [f"orphan-{index:04d}" for index in range(300)]

    class FakeVectorStore:
        def __init__(self, _kb_name, *, collection_name, create_if_missing):
            assert collection_name == _canonical_reconcile_collection_name()
            assert create_if_missing is False

        def reconciliation_snapshot(
            self,
            _knowledge_base_id,
            expected_point_ids,
            *,
            expected_vector_size,
        ):
            assert expected_vector_size == 3
            retrieval_batches.append(len(expected_point_ids))
            return {
                "points": [],
                "scanned_ids": orphan_inventory,
                "scan_truncated": False,
                "collection_schema_error": None,
            }

        def get_points_batched(self, expected_point_ids, *, batch_size):
            assert batch_size == 256
            retrieval_batches.append(len(expected_point_ids))
            return []

    monkeypatch.setattr(maintenance, "VectorStore", FakeVectorStore)

    preview = maintenance.reconcile_vector_store_sync(
        db_session,
        sample_knowledge_base.id,
        dry_run=True,
        batch_size=10_000,
    )

    assert preview["checked_records"] == 257
    assert preview["processed_record_batches"] == 2
    assert preview["max_record_batch_size"] == 256
    assert preview["max_chunk_lookup_ids"] <= 256
    assert preview["max_owner_lookup_ids"] <= 256
    assert preview["max_inventory_lookup_ids"] <= 256
    assert retrieval_batches == [256, 1]
    assert preview["missing_points"] == 257
    assert preview["orphan_points"] == 300
    assert preview["records_requiring_repair"] == 257
    assert preview["proposed_records"] == 257
    assert preview["marked_records"] == 0
    assert len(preview["stale_reasons_by_record"]) == 64
    assert preview["sampled_stale_reason_values"] == 64
    assert preview["omitted_stale_reason_records"] == 193
    assert preview["sampled_orphan_point_ids"] == 64
    assert preview["omitted_orphan_point_ids"] == 236
    assert preview["diagnostics_truncated"] is True
    assert all(record.vector_status == "ready" for record in records)

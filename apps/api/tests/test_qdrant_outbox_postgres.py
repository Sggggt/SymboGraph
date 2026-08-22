from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest


class MemoryVectorStore:
    def __init__(self) -> None:
        self.points: dict[str, dict[str, Any]] = {}

    def get_points(self, ids: list[str]) -> list[dict[str, Any]]:
        return [dict(self.points[point_id]) for point_id in ids if point_id in self.points]

    def get_points_batched(self, ids: list[str], **_kwargs) -> list[dict[str, Any]]:
        return self.get_points(ids)

    def list_collection_names_bounded(self, **_kwargs) -> dict[str, Any]:
        names = ["qdrant_outbox-delete-timeout"] if self.points else []
        return {
            "collection_names": names,
            "collection_count": len(names),
            "truncated": False,
            "max_collections": 512,
            "complete_backend_inventory": True,
        }

    def list_owned_ids_complete(
        self,
        knowledge_base_id: str,
        **_kwargs,
    ) -> dict[str, Any]:
        ids = sorted(
            point_id
            for point_id, point in self.points.items()
            if (point.get("payload") or {}).get("knowledge_base_id")
            == knowledge_base_id
        )
        return {
            "ids": ids,
            "point_count": len(ids),
            "page_count": 1 if ids else 0,
            "truncated": False,
            "max_points": 1_000_000,
        }

    async def async_upsert(self, points: list[dict[str, Any]]) -> None:
        self.upsert(points)

    def upsert(self, points: list[dict[str, Any]]) -> None:
        for point in points:
            self.points[str(point["id"])] = dict(point)

    def delete(self, ids: list[str]) -> None:
        for point_id in ids:
            self.points.pop(point_id, None)

    def delete_if_payload_matches(self, expected_points: list[dict[str, Any]]) -> None:
        matching_ids = [
            str(point["id"])
            for point in expected_points
            if str(point["id"]) in self.points
            and self.points[str(point["id"])].get("payload") == point.get("payload")
        ]
        self.delete(matching_ids)


class AmbiguousTimeoutVectorStore(MemoryVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.release_late_write = asyncio.Event()
        self.late_write: asyncio.Task[None] | None = None

    async def async_upsert(self, points: list[dict[str, Any]]) -> None:
        captured = [
            {
                "id": str(point["id"]),
                "vector": list(point.get("vector") or []),
                "payload": dict(point.get("payload") or {}),
            }
            for point in points
        ]

        async def complete_later() -> None:
            await self.release_late_write.wait()
            self.upsert(captured)

        self.late_write = asyncio.create_task(complete_later())
        raise TimeoutError("Qdrant accepted the write but the client timed out")


class AmbiguousRestoreVectorStore(MemoryVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_async_upsert = False
        self.release_late_restore = asyncio.Event()
        self.late_restore: asyncio.Task[None] | None = None

    async def async_upsert(self, points: list[dict[str, Any]]) -> None:
        if not self.fail_next_async_upsert:
            self.upsert(points)
            return
        self.fail_next_async_upsert = False
        captured = [
            {
                "id": str(point["id"]),
                "vector": list(point.get("vector") or []),
                "payload": dict(point.get("payload") or {}),
            }
            for point in points
        ]

        async def complete_later() -> None:
            await self.release_late_restore.wait()
            self.upsert(captured)

        self.late_restore = asyncio.create_task(complete_later())
        raise TimeoutError("Qdrant accepted the restore but the client timed out")


class AmbiguousDeleteVectorStore(MemoryVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_delete = False
        self.release_late_delete = asyncio.Event()
        self.late_delete: asyncio.Task[None] | None = None
        self.delete_calls: list[list[str]] = []

    def delete(self, ids: list[str]) -> None:
        self.delete_calls.append(list(ids))
        if not self.fail_next_delete:
            super().delete(ids)
            return
        self.fail_next_delete = False
        captured = list(ids)

        async def complete_later() -> None:
            await self.release_late_delete.wait()
            MemoryVectorStore.delete(self, captured)

        self.late_delete = asyncio.create_task(complete_later())
        raise TimeoutError("Qdrant accepted the delete but the client timed out")

    def delete_if_payload_matches(self, expected_points: list[dict[str, Any]]) -> None:
        captured = [
            {
                "id": str(point["id"]),
                "vector": list(point.get("vector") or []),
                "payload": dict(point.get("payload") or {}),
            }
            for point in expected_points
        ]
        self.delete_calls.append([str(point["id"]) for point in captured])

        def apply_if_unchanged() -> None:
            for point in captured:
                point_id = str(point["id"])
                current = self.points.get(point_id)
                if current is not None and current.get("payload") == point.get("payload"):
                    self.points.pop(point_id, None)

        if not self.fail_next_delete:
            apply_if_unchanged()
            return
        self.fail_next_delete = False

        async def complete_later() -> None:
            await self.release_late_delete.wait()
            apply_if_unchanged()

        self.late_delete = asyncio.create_task(complete_later())
        raise TimeoutError("Qdrant accepted the delete but the conditional client timed out")


def _point(point_id: str, value: float) -> dict[str, Any]:
    from app.services.context_graph import (
        QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
        QDRANT_VECTOR_DISTANCE_METRIC,
        VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION,
        qdrant_collection_identity_digest,
        vector_payload_hash,
    )
    from app.services.vector_store import canonical_embedding_vector

    vector = canonical_embedding_vector(
        [value, 1.0 - value],
        source=f"test point {point_id}",
    )
    identity = _outbox_vector_identity()
    identity_digest = qdrant_collection_identity_digest(**identity)
    context_hash_protocol_version = "vector_schema-context-hash-protocol-v1"
    context_hash = f"vector_schema-context-hash-{point_id}"
    local_hint_protocol_version = "vector_schema-local-hint-protocol-v1"
    local_hint_hash = f"vector_schema-local-hint-hash-{point_id}"
    return {
        "id": point_id,
        "vector": vector,
        "payload": {
            "payload_hash": f"hash-{value}",
            "embedding_model": identity["embedding_model"],
            "embedding_dimension": identity["embedding_dimension"],
            "vector_distance_metric": QDRANT_VECTOR_DISTANCE_METRIC,
            "embedding_text_version": identity["embedding_text_version"],
            "chunk_schema_version": identity["chunk_schema_version"],
            "context_hash_protocol_version": context_hash_protocol_version,
            "context_hash": context_hash,
            "local_hint_protocol_version": local_hint_protocol_version,
            "local_hint_hash": local_hint_hash,
            "collection_identity_protocol_version": QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
            "collection_identity_digest": identity_digest,
            "vector_payload_hash_protocol": VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION,
            "vector_payload_hash": vector_payload_hash(
                vector=vector,
                chunk_id=point_id,
                embedding_model=identity["embedding_model"],
                embedding_dimension=identity["embedding_dimension"],
                vector_distance_metric=QDRANT_VECTOR_DISTANCE_METRIC,
                embedding_text_version=identity["embedding_text_version"],
                chunk_schema_version=identity["chunk_schema_version"],
                context_hash_protocol_version=context_hash_protocol_version,
                context_hash=context_hash,
                local_hint_protocol_version=local_hint_protocol_version,
                local_hint_hash=local_hint_hash,
                collection_identity_protocol_version=(
                    QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION
                ),
                collection_identity_digest=identity_digest,
            ),
        },
    }


def _outbox_vector_identity() -> dict[str, Any]:
    return {
        "embedding_model": "vector_schema/outbox-v2-postgres-test",
        "embedding_dimension": 2,
        "embedding_text_version": "contextual_text_v2",
        "chunk_schema_version": "fixed_token_chunk_v1",
    }


def _outbox_collection_name() -> str:
    from app.services.context_graph import qdrant_collection_name

    return qdrant_collection_name(**_outbox_vector_identity())


def _outbox_envelope_contract(qdrant_outbox: Any) -> dict[str, str]:
    contract = qdrant_outbox._outbox_protocol_contract(
        qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION
    )
    return {
        "envelope_schema_version": contract["envelope_schema_version"],
        "envelope_schema_hash": contract["envelope_schema_hash"],
        "canonical_bytes_version": contract["canonical_bytes_version"],
    }


@pytest.mark.asyncio
async def test_postgres_outbox_survives_rollback_and_reconciles_crash_timeout_retry(monkeypatch):
    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import IngestionCompensationLog, KnowledgeBase
    from app.services import qdrant_outbox
    from app.services import vector_store as vector_store_module

    setup = SessionLocal()
    if setup.get_bind().dialect.name != "postgresql":
        setup.close()
        pytest.skip("requires the Docker PostgreSQL runtime")
    knowledge_base = KnowledgeBase(
        name=f"qdrant_outbox-{uuid4()}",
        description="durable Qdrant outbox integration fixture",
        source_root=f"qdrant_outbox/{uuid4()}",
    )
    setup.add(knowledge_base)
    setup.commit()
    knowledge_base_id = knowledge_base.id
    knowledge_base_name = knowledge_base.name
    setup.close()

    store = MemoryVectorStore()
    monkeypatch.setattr(vector_store_module, "VectorStore", lambda *_args, **_kwargs: store)
    main = SessionLocal()
    try:
        assert main.get(KnowledgeBase, knowledge_base_id) is not None
        result = await qdrant_outbox.execute_qdrant_upsert_batches(
            main,
            store=store,
            knowledge_base_id=knowledge_base_id,
            job_id=None,
            collection_name=_outbox_collection_name(),
            points=[_point("rollback-orphan", 0.73)],
            batch_size=10,
        )
        intent_id = result["intent_ids"][0]

        audit = SessionLocal()
        durable_row = audit.get(IngestionCompensationLog, intent_id)
        assert durable_row is not None
        assert durable_row.status == "external_applied"
        assert durable_row.payload_json["target_points"][0]["payload"]["qdrant_write_intent_id"] == intent_id
        audit.close()

        main.rollback()
        audit = SessionLocal()
        assert audit.get(IngestionCompensationLog, intent_id).status == "compensation_pending"
        audit.close()
        assert "rollback-orphan" in store.points

        reconcile_db = SessionLocal()
        with qdrant_outbox.qdrant_outbox_reconcile_lock(reconcile_db, knowledge_base_id):
            preview = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=True,
            )
            assert preview["eligible_intents"] == 1
            assert preview["actions"][0]["action"] == "compensate_uncommitted_target"
            assert "rollback-orphan" in store.points
            applied = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
            )
            assert applied["compensated"] == 1
        reconcile_db.close()
        assert "rollback-orphan" not in store.points

        crash_intent_id = str(uuid4())
        crash_target = _point("crash-orphan", 0.61)
        crash_target["payload"].update(
            {
                "knowledge_base_id": knowledge_base_id,
                "chunk_id": "crash-orphan",
                "qdrant_write_intent_id": crash_intent_id,
                "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
            }
        )
        crash_db = SessionLocal()
        assert qdrant_outbox._persist_intent(
            crash_db,
            intent_id=crash_intent_id,
            knowledge_base_id=knowledge_base_id,
            job_id=None,
            collection_name=_outbox_collection_name(),
            target_points=[crash_target],
            before_points=[],
        ) is True
        crash_db.close()
        store.upsert([crash_target])

        reconcile_db = SessionLocal()
        with qdrant_outbox.qdrant_outbox_reconcile_lock(reconcile_db, knowledge_base_id):
            protected = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
            )
            assert protected["skipped_uncertainty_window"] == 1
            assert "crash-orphan" in store.points
            forced = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert forced["skipped_uncertainty_window"] == 1
            assert "crash-orphan" in store.points

            expiry_db = SessionLocal()
            crash_row = expiry_db.get(IngestionCompensationLog, crash_intent_id)
            crash_payload = dict(crash_row.payload_json or {})
            crash_payload["lease_expires_at"] = "2000-01-01T00:00:00"
            crash_row.payload_json = crash_payload
            expiry_db.commit()
            expiry_db.close()
            monkeypatch.setattr(qdrant_outbox, "QDRANT_OUTBOX_UNCERTAINTY_CONFIRM_SECONDS", 0)

            observed = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert observed["observed_uncertain"] == 1
            assert "crash-orphan" in store.points
            compensated_pending_verification = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert compensated_pending_verification["verification_pending"] == 1, compensated_pending_verification["actions"]
            assert "crash-orphan" not in store.points
            verified = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert verified["verification_pending"] == 1
            assert verified["actions"][0]["action"] == "watch_verified_compensation_postcondition"
            retry = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert retry["checked_intents"] == 1
            assert retry["verification_pending"] == 1
        reconcile_db.close()
        assert "crash-orphan" not in store.points

        final_audit = SessionLocal()
        rows = list(
            final_audit.scalars(
                select(IngestionCompensationLog).where(
                    IngestionCompensationLog.knowledge_base_id == knowledge_base_id
                )
            ).all()
        )
        assert {row.status for row in rows} == {
            "compensated",
            "compensation_verify_pending",
        }
        final_audit.close()
    finally:
        main.close()
        cleanup = SessionLocal()
        cleanup_kb = cleanup.get(KnowledgeBase, knowledge_base_id)
        if cleanup_kb is not None:
            cleanup.delete(cleanup_kb)
            cleanup.commit()
        cleanup.close()


@pytest.mark.asyncio
async def test_real_qdrant_orphan_is_deleted_from_durable_postgres_intent():
    from app.core.config import get_settings
    from app.db import SessionLocal
    from app.models import IngestionCompensationLog, KnowledgeBase
    from app.services import qdrant_outbox
    from app.services.vector_store import VectorStore

    setup = SessionLocal()
    if setup.get_bind().dialect.name != "postgresql":
        setup.close()
        pytest.skip("requires the Docker PostgreSQL runtime")
    settings = get_settings()
    knowledge_base = KnowledgeBase(
        name=f"qdrant_outbox-qdrant-{uuid4()}",
        description="real Qdrant outbox integration fixture",
        source_root=f"qdrant_outbox-qdrant/{uuid4()}",
    )
    setup.add(knowledge_base)
    setup.commit()
    knowledge_base_id = knowledge_base.id
    knowledge_base_name = knowledge_base.name
    setup.close()

    collection_name = _outbox_collection_name()
    point_id = str(uuid4())
    store = VectorStore(knowledge_base_name, collection_name=collection_name)
    main = SessionLocal()
    intent_id: str | None = None
    try:
        assert main.get(KnowledgeBase, knowledge_base_id) is not None
        result = await qdrant_outbox.execute_qdrant_upsert_batches(
            main,
            store=store,
            knowledge_base_id=knowledge_base_id,
            job_id=None,
            collection_name=collection_name,
            points=[_point(point_id, 1.0)],
            batch_size=1,
        )
        intent_id = result["intent_ids"][0]
        assert [point["id"] for point in store.get_points([point_id])] == [point_id]
        main.rollback()

        audit = SessionLocal()
        assert audit.get(IngestionCompensationLog, intent_id).status == "compensation_pending"
        audit.close()

        reconcile_db = SessionLocal()
        with qdrant_outbox.qdrant_outbox_reconcile_lock(reconcile_db, knowledge_base_id):
            stats = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
            )
            assert stats["compensated"] == 1
        reconcile_db.close()
        assert store.get_points([point_id]) == []
    finally:
        main.close()
        if store.client is not None:
            collections = {item.name for item in store.client.get_collections().collections}
            if collection_name in collections:
                store.client.delete_collection(collection_name=collection_name)
        cleanup = SessionLocal()
        cleanup_kb = cleanup.get(KnowledgeBase, knowledge_base_id)
        if cleanup_kb is not None:
            cleanup.delete(cleanup_kb)
            cleanup.commit()
        cleanup.close()


@pytest.mark.asyncio
async def test_postgres_nested_savepoint_does_not_finalize_before_outer_rollback():
    from app.db import SessionLocal
    from app.models import IngestionCompensationLog, KnowledgeBase
    from app.services import qdrant_outbox

    setup = SessionLocal()
    if setup.get_bind().dialect.name != "postgresql":
        setup.close()
        pytest.skip("requires the Docker PostgreSQL runtime")
    knowledge_base = KnowledgeBase(
        name=f"qdrant_outbox-savepoint-{uuid4()}",
        source_root=f"qdrant_outbox-savepoint/{uuid4()}",
    )
    setup.add(knowledge_base)
    setup.commit()
    knowledge_base_id = knowledge_base.id
    setup.close()

    main = SessionLocal()
    try:
        assert main.get(KnowledgeBase, knowledge_base_id) is not None
        nested = main.begin_nested()
        result = await qdrant_outbox.execute_qdrant_upsert_batches(
            main,
            store=MemoryVectorStore(),
            knowledge_base_id=knowledge_base_id,
            job_id=None,
            collection_name=_outbox_collection_name(),
            points=[_point("savepoint-orphan", 0.42)],
            batch_size=1,
        )
        intent_id = result["intent_ids"][0]
        nested.commit()

        audit = SessionLocal()
        assert audit.get(IngestionCompensationLog, intent_id).status == "external_applied"
        audit.close()

        main.rollback()
        audit = SessionLocal()
        assert audit.get(IngestionCompensationLog, intent_id).status == "compensation_pending"
        audit.close()
    finally:
        main.close()
        cleanup = SessionLocal()
        cleanup_kb = cleanup.get(KnowledgeBase, knowledge_base_id)
        if cleanup_kb is not None:
            cleanup.delete(cleanup_kb)
            cleanup.commit()
        cleanup.close()


def test_terminal_outbox_state_cannot_be_downgraded_by_late_callback():
    from app.db import SessionLocal
    from app.models import IngestionCompensationLog, KnowledgeBase
    from app.services import qdrant_outbox

    setup = SessionLocal()
    if setup.get_bind().dialect.name != "postgresql":
        setup.close()
        pytest.skip("requires the Docker PostgreSQL runtime")
    knowledge_base = KnowledgeBase(
        name=f"qdrant_outbox-terminal-fence-{uuid4()}",
        source_root=f"qdrant_outbox-terminal-fence/{uuid4()}",
    )
    setup.add(knowledge_base)
    setup.commit()
    knowledge_base_id = knowledge_base.id
    intent_id = str(uuid4())
    target = _point(str(uuid4()), 0.5)
    target["payload"].update(
        {
            "knowledge_base_id": knowledge_base_id,
            "chunk_id": str(target["id"]),
            "qdrant_write_intent_id": intent_id,
            "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
        }
    )
    assert qdrant_outbox._persist_intent(
        setup,
        intent_id=intent_id,
        knowledge_base_id=knowledge_base_id,
        job_id=None,
        collection_name=_outbox_collection_name(),
        target_points=[target],
        before_points=[],
    )
    qdrant_outbox._transition_intent(
        setup,
        intent_id=intent_id,
        status="external_applied",
        details={"reason": "test_external_applied"},
    )
    qdrant_outbox._transition_intent(
        setup,
        intent_id=intent_id,
        status="committed",
        details={"reason": "test_root_commit"},
    )
    with pytest.raises(qdrant_outbox.QdrantOutboxError, match="committed -> compensation_pending"):
        qdrant_outbox._transition_intent(
            setup,
            intent_id=intent_id,
            status="compensation_pending",
            details={"reason": "late_root_rollback_callback"},
        )

    audit = SessionLocal()
    assert audit.get(IngestionCompensationLog, intent_id).status == "committed"
    audit.close()
    setup.delete(setup.get(KnowledgeBase, knowledge_base_id))
    setup.commit()
    setup.close()


def test_v1_reconcile_failed_pending_history_resumes_durable_watch(monkeypatch):
    from app.db import SessionLocal
    from app.models import IngestionCompensationLog, KnowledgeBase
    from app.services import qdrant_outbox
    from app.services import vector_store as vector_store_module

    setup = SessionLocal()
    if setup.get_bind().dialect.name != "postgresql":
        setup.close()
        pytest.skip("requires the Docker PostgreSQL runtime")
    knowledge_base = KnowledgeBase(
        name=f"qdrant_outbox-v1-watch-{uuid4()}",
        source_root=f"qdrant_outbox-v1-watch/{uuid4()}",
    )
    setup.add(knowledge_base)
    setup.flush()
    knowledge_base_id = knowledge_base.id
    knowledge_base_name = knowledge_base.name
    intent_id = str(uuid4())
    point_id = str(uuid4())
    point = _point(point_id, 0.37)
    point["payload"].update(
        {
            "knowledge_base_id": knowledge_base_id,
            "chunk_id": point_id,
            "qdrant_write_intent_id": intent_id,
            "qdrant_write_protocol_version": "qdrant_side_effect_outbox_v1",
        }
    )
    payload = {
        "protocol_version": "qdrant_side_effect_outbox_v1",
        "intent_id": intent_id,
        "collection_name": "qdrant_outbox-v1-watch",
        "target_points": [point],
        "before_points": [],
        "target_payload_hash": qdrant_outbox._canonical_hash([point]),
        "before_image_hash": qdrant_outbox._canonical_hash([]),
        "lease_expires_at": "2000-01-01T00:00:00",
        "state_history": [
            {"status": "pending", "at": "legacy"},
            {"status": "reconcile_failed", "at": "legacy"},
        ],
    }
    setup.add(
        IngestionCompensationLog(
            id=intent_id,
            knowledge_base_id=knowledge_base_id,
            operation=qdrant_outbox.QDRANT_UPSERT_OPERATION,
            target_ids_json=[point_id],
            payload_json=payload,
            status="reconcile_failed",
        )
    )
    setup.commit()
    setup.close()

    store = MemoryVectorStore()
    store.upsert([point])
    monkeypatch.setattr(vector_store_module, "VectorStore", lambda *_args, **_kwargs: store)
    monkeypatch.setattr(qdrant_outbox, "QDRANT_OUTBOX_LEASE_SECONDS", 0)
    monkeypatch.setattr(qdrant_outbox, "QDRANT_OUTBOX_UNCERTAINTY_CONFIRM_SECONDS", 0)
    try:
        reconcile_db = SessionLocal()
        with qdrant_outbox.qdrant_outbox_reconcile_lock(reconcile_db, knowledge_base_id):
            resumed = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert resumed["actions"][0]["action"] == "resume_unresolved_transport_watch"
            observed = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert observed["observed_uncertain"] == 1
            compensated = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert compensated["verification_pending"] == 1
            watching = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert watching["verification_pending"] == 1
        reconcile_db.close()
        assert store.get_points([point_id]) == []
        audit = SessionLocal()
        assert audit.get(IngestionCompensationLog, intent_id).status == "compensation_verify_pending"
        audit.close()
    finally:
        cleanup = SessionLocal()
        cleanup_kb = cleanup.get(KnowledgeBase, knowledge_base_id)
        if cleanup_kb is not None:
            cleanup.delete(cleanup_kb)
            cleanup.commit()
        cleanup.close()


@pytest.mark.asyncio
async def test_compensation_restore_timeout_is_prelogged_and_cannot_overwrite_new_owner(monkeypatch):
    from app.db import SessionLocal
    from app.models import (
        Chunk,
        Document,
        DocumentVersion,
        IngestionCompensationLog,
        KnowledgeBase,
        VectorRecord,
    )
    from app.services import qdrant_outbox
    from app.services import vector_store as vector_store_module

    setup = SessionLocal()
    if setup.get_bind().dialect.name != "postgresql":
        setup.close()
        pytest.skip("requires the Docker PostgreSQL runtime")
    knowledge_base = KnowledgeBase(
        name=f"qdrant_outbox-restore-timeout-{uuid4()}",
        source_root=f"qdrant_outbox-restore-timeout/{uuid4()}",
    )
    setup.add(knowledge_base)
    setup.flush()
    knowledge_base_id = knowledge_base.id
    knowledge_base_name = knowledge_base.name
    point_id = str(uuid4())
    document = Document(
        knowledge_base_id=knowledge_base_id,
        title="Qdrant outbox restore timeout",
        source_path=f"qdrant_outbox/restore-timeout-{uuid4()}.txt",
        source_type="txt",
        checksum="e" * 64,
    )
    setup.add(document)
    setup.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="e" * 64,
        storage_path=document.source_path,
    )
    setup.add(version)
    setup.flush()
    chunk = Chunk(
        id=point_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document.id,
        document_version_id=version.id,
        chunk_version=1,
        chunk_index=0,
        token_start=0,
        token_end=2,
        char_start=0,
        char_end=15,
        text="restore timeout",
        text_hash="f" * 64,
        state="active",
    )
    setup.add(chunk)
    setup.commit()
    setup.close()

    def owner_point(owner_id: str, marker: str, value: float) -> dict[str, Any]:
        _ = marker
        point = _point(point_id, value)
        point["payload"].update(
            {
                "knowledge_base_id": knowledge_base_id,
                "chunk_id": point_id,
                "qdrant_write_intent_id": owner_id,
                "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
            }
        )
        return point

    store = AmbiguousRestoreVectorStore()
    monkeypatch.setattr(vector_store_module, "VectorStore", lambda *_args, **_kwargs: store)
    monkeypatch.setattr(qdrant_outbox, "QDRANT_OUTBOX_LEASE_SECONDS", 0)
    monkeypatch.setattr(qdrant_outbox, "QDRANT_OUTBOX_UNCERTAINTY_CONFIRM_SECONDS", 0)
    b_intent_id = str(uuid4())
    a_intent_id = str(uuid4())
    c_intent_id = str(uuid4())
    b_point = owner_point(b_intent_id, "b", 0.2)
    a_point = owner_point(a_intent_id, "a", 0.5)
    c_point = owner_point(c_intent_id, "c", 0.9)
    try:
        owner_db = SessionLocal()
        assert qdrant_outbox._persist_intent(
            owner_db,
            intent_id=b_intent_id,
            knowledge_base_id=knowledge_base_id,
            job_id=None,
            collection_name=_outbox_collection_name(),
            target_points=[b_point],
            before_points=[],
        )
        qdrant_outbox._transition_intent(
            owner_db,
            intent_id=b_intent_id,
            status="external_applied",
            details={"reason": "owner_b_applied"},
        )
        owner_db.add(
            VectorRecord(
                knowledge_base_id=knowledge_base_id,
                chunk_id=point_id,
                qdrant_point_id=point_id,
                collection_name=_outbox_collection_name(),
                embedding_model=_outbox_vector_identity()["embedding_model"],
                embedding_dimension=2,
                embedding_text_version=_outbox_vector_identity()["embedding_text_version"],
                payload_hash=b_point["payload"]["vector_payload_hash"],
                vector_status="ready",
                diagnostics_json={
                    "qdrant_write_intent_id": b_intent_id,
                    "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
                },
            )
        )
        owner_db.commit()
        qdrant_outbox._transition_intent(
            owner_db,
            intent_id=b_intent_id,
            status="committed",
            details={"reason": "owner_b_committed"},
        )

        assert qdrant_outbox._persist_intent(
            owner_db,
            intent_id=a_intent_id,
            knowledge_base_id=knowledge_base_id,
            job_id=None,
            collection_name=_outbox_collection_name(),
            target_points=[a_point],
            before_points=[b_point],
        )
        qdrant_outbox._transition_intent(
            owner_db,
            intent_id=a_intent_id,
            status="external_applied",
            details={"reason": "owner_a_applied"},
        )
        owner_db.close()
        store.upsert([a_point])

        handle = qdrant_outbox.QdrantOutboxHandle(
            id=a_intent_id,
            knowledge_base_id=knowledge_base_id,
            job_id=None,
            collection_name=_outbox_collection_name(),
            target_ids=(point_id,),
            target_points=(a_point,),
            before_points=(b_point,),
            payload_hash=qdrant_outbox._canonical_hash([a_point]),
            durable=True,
        )
        store.fail_next_async_upsert = True
        compensate_db = SessionLocal()
        with pytest.raises(TimeoutError, match="accepted the restore"):
            await qdrant_outbox.compensate_qdrant_handle(
                compensate_db,
                store=store,
                handle=handle,
                reason="rollback_after_later_batch_failure",
            )
        compensate_db.close()
        assert store.late_restore is not None

        audit = SessionLocal()
        a_row = audit.get(IngestionCompensationLog, a_intent_id)
        assert a_row.status == "external_outcome_unknown"
        assert a_row.payload_json["requires_uncertainty_watch"] is True
        assert a_row.payload_json["qdrant_mutation_attempt"]["state"] == "pending"
        audit.close()

        c_db = SessionLocal()
        c_db.add(
            IngestionCompensationLog(
                id=c_intent_id,
                job_id=None,
                knowledge_base_id=knowledge_base_id,
                operation=qdrant_outbox.QDRANT_UPSERT_OPERATION,
                target_ids_json=[point_id],
                payload_json={
                    "protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
                    **_outbox_envelope_contract(qdrant_outbox),
                    "intent_id": c_intent_id,
                    "collection_name": _outbox_collection_name(),
                    "target_points": [c_point],
                    "before_points": [a_point],
                    "target_payload_hash": qdrant_outbox._canonical_hash([c_point]),
                    "before_image_hash": qdrant_outbox._canonical_hash([a_point]),
                    "requires_uncertainty_watch": False,
                    "state_history": [
                        {"status": "committed", "at": "historical-overlap-fixture"}
                    ],
                },
                status="committed",
            )
        )
        record = c_db.query(VectorRecord).filter_by(
            knowledge_base_id=knowledge_base_id,
            qdrant_point_id=point_id,
        ).one()
        record.payload_hash = c_point["payload"]["vector_payload_hash"]
        record.diagnostics_json = {
            "qdrant_write_intent_id": c_intent_id,
            "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
        }
        c_db.commit()
        c_db.close()
        store.upsert([c_point])

        watch_db = SessionLocal()
        with qdrant_outbox.qdrant_outbox_reconcile_lock(watch_db, knowledge_base_id):
            for _ in range(3):
                stats = qdrant_outbox.reconcile_qdrant_outbox_sync(
                    watch_db,
                    knowledge_base_id=knowledge_base_id,
                    knowledge_base_name=knowledge_base_name,
                    dry_run=False,
                    include_unexpired=True,
                )
            assert stats["verification_pending"] == 1
        watch_db.close()
        assert store.points[point_id] == c_point

        store.release_late_restore.set()
        await store.late_restore
        assert store.points[point_id] == b_point

        repair_db = SessionLocal()
        with qdrant_outbox.qdrant_outbox_reconcile_lock(repair_db, knowledge_base_id):
            repaired = qdrant_outbox.reconcile_qdrant_outbox_sync(
                repair_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert repaired["verification_pending"] == 1
            verified = qdrant_outbox.reconcile_qdrant_outbox_sync(
                repair_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert verified["verification_pending"] == 1
        repair_db.close()
        assert store.points[point_id] == c_point

        audit = SessionLocal()
        a_row = audit.get(IngestionCompensationLog, a_intent_id)
        assert a_row.status == "compensation_verify_pending"
        assert a_row.payload_json["requires_uncertainty_watch"] is True
        assert a_row.payload_json["qdrant_mutation_attempt_total_count"] >= 2
        assert audit.get(IngestionCompensationLog, c_intent_id).status == "committed"
        audit.close()
    finally:
        if store.late_restore is not None and not store.late_restore.done():
            store.late_restore.cancel()
        cleanup = SessionLocal()
        cleanup_kb = cleanup.get(KnowledgeBase, knowledge_base_id)
        if cleanup_kb is not None:
            cleanup.delete(cleanup_kb)
            cleanup.commit()
        cleanup.close()


@pytest.mark.asyncio
async def test_ambiguous_timeout_requires_stable_observation_and_preserves_newer_owner(monkeypatch):
    from app.db import SessionLocal
    from app.models import (
        Chunk,
        Document,
        DocumentVersion,
        IngestionCompensationLog,
        KnowledgeBase,
        VectorRecord,
    )
    from app.services import qdrant_outbox
    from app.services import vector_store as vector_store_module

    setup = SessionLocal()
    if setup.get_bind().dialect.name != "postgresql":
        setup.close()
        pytest.skip("requires the Docker PostgreSQL runtime")
    knowledge_base = KnowledgeBase(
        name=f"qdrant_outbox-uncertain-{uuid4()}",
        source_root=f"qdrant_outbox-uncertain/{uuid4()}",
    )
    setup.add(knowledge_base)
    setup.commit()
    knowledge_base_id = knowledge_base.id
    knowledge_base_name = knowledge_base.name
    setup.close()

    store = AmbiguousTimeoutVectorStore()
    monkeypatch.setattr(vector_store_module, "VectorStore", lambda *_args, **_kwargs: store)
    monkeypatch.setattr(qdrant_outbox, "QDRANT_OUTBOX_LEASE_SECONDS", 0)
    monkeypatch.setattr(qdrant_outbox, "QDRANT_OUTBOX_UNCERTAINTY_CONFIRM_SECONDS", 0)
    main = SessionLocal()
    late_point_id = str(uuid4())
    newer_point_id = str(uuid4())
    try:
        with pytest.raises(TimeoutError, match="accepted"):
            await qdrant_outbox.execute_qdrant_upsert_batches(
                main,
                store=store,
                knowledge_base_id=knowledge_base_id,
                job_id=None,
                collection_name=_outbox_collection_name(),
                points=[_point(late_point_id, 0.2), _point(newer_point_id, 0.3)],
                batch_size=2,
            )
        audit = SessionLocal()
        row = audit.query(IngestionCompensationLog).filter_by(knowledge_base_id=knowledge_base_id).one()
        intent_id = row.id
        ambiguous_newer_target = next(
            dict(point)
            for point in list((row.payload_json or {}).get("target_points") or [])
            if str(point.get("id")) == newer_point_id
        )
        assert row.status == "external_outcome_unknown"
        audit.close()

        assert store.late_write is not None
        reconcile_db = SessionLocal()
        with qdrant_outbox.qdrant_outbox_reconcile_lock(reconcile_db, knowledge_base_id):
            first_observation = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert first_observation["observed_uncertain"] == 1
            compensation = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert compensation["verification_pending"] == 1
            verification_watch = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert verification_watch["verification_pending"] == 1
            assert verification_watch["actions"][0]["action"] == "watch_verified_compensation_postcondition"
        reconcile_db.close()

        audit = SessionLocal()
        assert audit.get(IngestionCompensationLog, intent_id).status == "compensation_verify_pending"
        audit.close()

        newer_intent_id = str(uuid4())
        newer_point = _point(newer_point_id, 0.99)
        newer_point["payload"].update(
            {
                "knowledge_base_id": knowledge_base_id,
                "chunk_id": newer_point_id,
                "qdrant_write_intent_id": newer_intent_id,
                "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
            }
        )
        owner_db = SessionLocal()
        owner_db.add(
            IngestionCompensationLog(
                id=newer_intent_id,
                job_id=None,
                knowledge_base_id=knowledge_base_id,
                operation=qdrant_outbox.QDRANT_UPSERT_OPERATION,
                target_ids_json=[newer_point_id],
                payload_json={
                    "protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
                    **_outbox_envelope_contract(qdrant_outbox),
                    "intent_id": newer_intent_id,
                    "collection_name": _outbox_collection_name(),
                    "target_points": [newer_point],
                    "before_points": [],
                    "target_payload_hash": qdrant_outbox._canonical_hash([newer_point]),
                    "before_image_hash": qdrant_outbox._canonical_hash([]),
                    "requires_uncertainty_watch": False,
                    "state_history": [
                        {"status": "committed", "at": "historical-overlap-fixture"}
                    ],
                },
                status="committed",
            )
        )
        document = Document(
            knowledge_base_id=knowledge_base_id,
            title="Qdrant outbox newer owner",
            source_path=f"qdrant_outbox/newer-owner-{uuid4()}.txt",
            source_type="txt",
            checksum="c" * 64,
        )
        owner_db.add(document)
        owner_db.flush()
        document_version = DocumentVersion(
            document_id=document.id,
            version=1,
            checksum="c" * 64,
            storage_path=document.source_path,
        )
        owner_db.add(document_version)
        owner_db.flush()
        chunk = Chunk(
            id=newer_point_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document.id,
            document_version_id=document_version.id,
            chunk_version=1,
            chunk_index=0,
            token_start=0,
            token_end=2,
            char_start=0,
            char_end=11,
            text="newer owner",
            text_hash="d" * 64,
            state="active",
        )
        owner_db.add(chunk)
        owner_db.flush()
        owner_db.add(
            VectorRecord(
                knowledge_base_id=knowledge_base_id,
                chunk_id=chunk.id,
                qdrant_point_id=newer_point_id,
                collection_name=_outbox_collection_name(),
                embedding_model=_outbox_vector_identity()["embedding_model"],
                embedding_dimension=2,
                embedding_text_version=_outbox_vector_identity()["embedding_text_version"],
                payload_hash=newer_point["payload"]["vector_payload_hash"],
                vector_status="ready",
                diagnostics_json={
                    "qdrant_write_intent_id": newer_intent_id,
                    "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
                },
            )
        )
        owner_db.commit()
        owner_db.close()
        store.upsert([newer_point])

        capture_db = SessionLocal()
        with qdrant_outbox.qdrant_outbox_reconcile_lock(capture_db, knowledge_base_id):
            captured = qdrant_outbox.reconcile_qdrant_outbox_sync(
                capture_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert captured["verification_pending"] == 1
        capture_db.close()
        assert store.points[newer_point_id] == newer_point

        store.release_late_write.set()
        await store.late_write
        assert set(store.points) == {late_point_id, newer_point_id}
        assert store.points[newer_point_id]["payload"]["qdrant_write_intent_id"] == intent_id

        repair_db = SessionLocal()
        with qdrant_outbox.qdrant_outbox_reconcile_lock(repair_db, knowledge_base_id):
            repaired = qdrant_outbox.reconcile_qdrant_outbox_sync(
                repair_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert repaired["verification_pending"] == 1
            verified_again = qdrant_outbox.reconcile_qdrant_outbox_sync(
                repair_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert verified_again["verification_pending"] == 1
        repair_db.close()

        audit = SessionLocal()
        assert audit.get(IngestionCompensationLog, intent_id).status == "compensation_verify_pending"
        assert audit.get(IngestionCompensationLog, newer_intent_id).status == "committed"
        audit.close()
        assert set(store.points) == {newer_point_id}
        assert store.points[newer_point_id] == newer_point

        # If PostgreSQL later removes the ready VectorRecord, the historical
        # before-image is no longer authoritative. A replayed late A write must
        # be deleted instead of resurrecting the retired B point.
        retire_db = SessionLocal()
        retired_record = retire_db.query(VectorRecord).filter_by(
            knowledge_base_id=knowledge_base_id,
            qdrant_point_id=newer_point_id,
        ).one()
        retire_db.delete(retired_record)
        retire_db.commit()
        retire_db.close()
        store.delete([newer_point_id])
        store.upsert([ambiguous_newer_target])
        assert store.points[newer_point_id]["payload"]["qdrant_write_intent_id"] == intent_id

        absence_db = SessionLocal()
        with qdrant_outbox.qdrant_outbox_reconcile_lock(absence_db, knowledge_base_id):
            removed_retired_point = qdrant_outbox.reconcile_qdrant_outbox_sync(
                absence_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert removed_retired_point["verification_pending"] == 1
            verified_absent = qdrant_outbox.reconcile_qdrant_outbox_sync(
                absence_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=False,
                include_unexpired=True,
            )
            assert verified_absent["verification_pending"] == 1
        absence_db.close()
        assert store.points == {}

        audit = SessionLocal()
        assert audit.get(IngestionCompensationLog, intent_id).status == "compensation_verify_pending"
        assert audit.get(IngestionCompensationLog, newer_intent_id).status == "committed"
        audit.close()
    finally:
        main.close()
        if store.late_write is not None and not store.late_write.done():
            store.late_write.cancel()
        cleanup = SessionLocal()
        cleanup_kb = cleanup.get(KnowledgeBase, knowledge_base_id)
        if cleanup_kb is not None:
            cleanup.delete(cleanup_kb)
            cleanup.commit()
        cleanup.close()


@pytest.mark.asyncio
async def test_knowledge_base_delete_timeout_keeps_tombstone_and_blocks_ingestion(monkeypatch):
    from app.db import SessionLocal
    from app.models import (
        Chunk,
        Document,
        DocumentVersion,
        IngestionCompensationLog,
        KnowledgeBase,
        VectorRecord,
    )
    from app.services import maintenance, qdrant_outbox, storage_maintenance
    from app.services.ingestion_resource_lock import (
        IngestionResourceBusyError,
        knowledge_base_ingestion_resource_lock,
    )

    setup = SessionLocal()
    if setup.get_bind().dialect.name != "postgresql":
        setup.close()
        pytest.skip("requires the Docker PostgreSQL runtime")
    knowledge_base = KnowledgeBase(
        name=f"qdrant_outbox-delete-timeout-{uuid4()}",
        source_root=f"qdrant_outbox-delete-timeout/{uuid4()}",
    )
    setup.add(knowledge_base)
    setup.flush()
    knowledge_base_id = knowledge_base.id
    point_id = str(uuid4())
    document = Document(
        knowledge_base_id=knowledge_base_id,
        title="Qdrant outbox delete timeout",
        source_path=f"qdrant_outbox/delete-timeout-{uuid4()}.txt",
        source_type="txt",
        checksum="1" * 64,
    )
    setup.add(document)
    setup.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="1" * 64,
        storage_path=document.source_path,
    )
    setup.add(version)
    setup.flush()
    chunk = Chunk(
        id=point_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document.id,
        document_version_id=version.id,
        chunk_version=1,
        chunk_index=0,
        token_start=0,
        token_end=2,
        char_start=0,
        char_end=14,
        text="delete timeout",
        text_hash="2" * 64,
        state="active",
    )
    setup.add(chunk)
    setup.flush()
    setup.add(
        VectorRecord(
            knowledge_base_id=knowledge_base_id,
            chunk_id=point_id,
            qdrant_point_id=point_id,
            collection_name="qdrant_outbox-delete-timeout",
            embedding_model="qdrant_outbox-test",
            embedding_dimension=2,
            embedding_text_version="qdrant_outbox-test-v1",
            payload_hash="3" * 64,
            vector_status="ready",
            diagnostics_json={},
        )
    )
    setup.commit()
    setup.close()

    point = _point(point_id, 0.44)
    point["payload"].update(
        {"knowledge_base_id": knowledge_base_id, "chunk_id": point_id}
    )
    store = AmbiguousDeleteVectorStore()
    store.upsert([point])
    store.fail_next_delete = True
    monkeypatch.setattr(maintenance, "VectorStore", lambda *_args, **_kwargs: store)
    monkeypatch.setattr(
        storage_maintenance,
        "VectorStore",
        lambda *_args, **_kwargs: store,
    )
    try:
        delete_db = SessionLocal()
        with pytest.raises(TimeoutError, match="accepted the delete"):
            maintenance.delete_knowledge_base_data(
                delete_db,
                delete_db.get(KnowledgeBase, knowledge_base_id),
            )
        delete_db.close()
        assert store.late_delete is not None

        audit = SessionLocal()
        assert audit.get(KnowledgeBase, knowledge_base_id) is not None
        tombstones = audit.query(IngestionCompensationLog).filter_by(
            knowledge_base_id=knowledge_base_id,
            operation=qdrant_outbox.QDRANT_DELETE_OPERATION,
        ).all()
        assert len(tombstones) == 1
        assert tombstones[0].status == "external_outcome_unknown"
        assert tombstones[0].payload_json["mutation_attempt"]["state"] == "pending"
        audit.close()

        ingest_db = SessionLocal()
        with pytest.raises(IngestionResourceBusyError) as exc_info:
            async with knowledge_base_ingestion_resource_lock(
                ingest_db,
                knowledge_base_id,
                operation="ingest_file",
                timeout_seconds=0.1,
            ):
                pass
        ingest_db.close()
        assert exc_info.value.diagnostics["reason"] == "knowledge_base_delete_pending"

        foreign_point = _point(point_id, 0.91)
        foreign_point["payload"].update(
            {
                "knowledge_base_id": "qdrant_outbox-foreign-kb",
                "chunk_id": point_id,
                "qdrant_write_intent_id": "qdrant_outbox-foreign-owner",
            }
        )
        store.upsert([foreign_point])
        store.release_late_delete.set()
        await store.late_delete
        assert store.get_points([point_id]) == [foreign_point]
        assert store.delete_calls == [[point_id]]

        retry_db = SessionLocal()
        stats = maintenance.delete_knowledge_base_data(
            retry_db,
            retry_db.get(KnowledgeBase, knowledge_base_id),
        )
        retry_db.close()
        assert stats["qdrant_points"] == 1
        assert store.delete_calls == [[point_id], [point_id]]
        assert store.get_points([point_id]) == [foreign_point]

        audit = SessionLocal()
        assert audit.get(KnowledgeBase, knowledge_base_id) is None
        assert audit.query(IngestionCompensationLog).filter_by(
            knowledge_base_id=knowledge_base_id,
            operation=qdrant_outbox.QDRANT_DELETE_OPERATION,
        ).count() == 0
        audit.close()
    finally:
        if store.late_delete is not None and not store.late_delete.done():
            store.late_delete.cancel()
        cleanup = SessionLocal()
        cleanup_kb = cleanup.get(KnowledgeBase, knowledge_base_id)
        if cleanup_kb is not None:
            cleanup.delete(cleanup_kb)
            cleanup.commit()
        cleanup.close()


@pytest.mark.asyncio
async def test_cross_kb_timeout_reservation_blocks_new_point_owner():
    from app.db import SessionLocal
    from app.models import IngestionCompensationLog, KnowledgeBase
    from app.services import qdrant_outbox

    setup = SessionLocal()
    if setup.get_bind().dialect.name != "postgresql":
        setup.close()
        pytest.skip("requires the Docker PostgreSQL runtime")
    kb_a = KnowledgeBase(
        name=f"qdrant_outbox-reservation-a-{uuid4()}",
        description="qdrant_outbox",
        source_root=f"qdrant_outbox/reservation-a-{uuid4()}",
    )
    kb_b = KnowledgeBase(
        name=f"qdrant_outbox-reservation-b-{uuid4()}",
        description="qdrant_outbox",
        source_root=f"qdrant_outbox/reservation-b-{uuid4()}",
    )
    setup.add_all([kb_a, kb_b])
    setup.commit()
    kb_a_id = kb_a.id
    kb_b_id = kb_b.id
    setup.close()

    collection_name = _outbox_collection_name()
    point_id = f"shared-{uuid4()}"
    store = AmbiguousTimeoutVectorStore()
    try:
        first_db = SessionLocal()
        with pytest.raises(TimeoutError, match="accepted the write"):
            await qdrant_outbox.execute_qdrant_upsert_batches(
                first_db,
                store=store,
                knowledge_base_id=kb_a_id,
                job_id=None,
                collection_name=collection_name,
                points=[_point(point_id, 0.2)],
                batch_size=1,
            )
        first_db.close()

        audit = SessionLocal()
        first_intent = audit.query(IngestionCompensationLog).filter_by(
            knowledge_base_id=kb_a_id,
            operation=qdrant_outbox.QDRANT_UPSERT_OPERATION,
        ).one()
        assert first_intent.status == "external_outcome_unknown"
        assert first_intent.payload_json["requires_uncertainty_watch"] is True
        audit.close()

        same_kb_db = SessionLocal()
        with pytest.raises(
            qdrant_outbox.QdrantOutboxError,
            match="unresolved durable reservation",
        ):
            await qdrant_outbox.execute_qdrant_upsert_batches(
                same_kb_db,
                store=store,
                knowledge_base_id=kb_a_id,
                job_id=None,
                collection_name=collection_name,
                points=[_point(point_id, 0.7)],
                batch_size=1,
            )
        same_kb_db.rollback()
        same_kb_db.close()

        second_db = SessionLocal()
        with pytest.raises(
            qdrant_outbox.QdrantOutboxError,
            match="unresolved durable reservation",
        ):
            await qdrant_outbox.execute_qdrant_upsert_batches(
                second_db,
                store=store,
                knowledge_base_id=kb_b_id,
                job_id=None,
                collection_name=collection_name,
                points=[_point(point_id, 0.9)],
                batch_size=1,
            )
        second_db.rollback()
        second_db.close()
        assert point_id not in store.points

        store.release_late_write.set()
        assert store.late_write is not None
        await store.late_write
        assert store.points[point_id]["payload"]["knowledge_base_id"] == kb_a_id
    finally:
        if store.late_write is not None and not store.late_write.done():
            store.late_write.cancel()
        cleanup = SessionLocal()
        for knowledge_base_id in (kb_a_id, kb_b_id):
            knowledge_base = cleanup.get(KnowledgeBase, knowledge_base_id)
            if knowledge_base is not None:
                cleanup.delete(knowledge_base)
        cleanup.commit()
        cleanup.close()


@pytest.mark.asyncio
async def test_legacy_failed_delete_reserves_point_across_knowledge_bases():
    from app.db import SessionLocal
    from app.models import IngestionCompensationLog, KnowledgeBase
    from app.services import qdrant_outbox

    setup = SessionLocal()
    if setup.get_bind().dialect.name != "postgresql":
        setup.close()
        pytest.skip("requires the Docker PostgreSQL runtime")
    kb_a = KnowledgeBase(
        name=f"qdrant_outbox-legacy-delete-a-{uuid4()}",
        description="qdrant_outbox legacy delete owner",
        source_root=f"qdrant_outbox/legacy-delete-a-{uuid4()}",
    )
    kb_b = KnowledgeBase(
        name=f"qdrant_outbox-legacy-delete-b-{uuid4()}",
        description="qdrant_outbox prospective new owner",
        source_root=f"qdrant_outbox/legacy-delete-b-{uuid4()}",
    )
    setup.add_all([kb_a, kb_b])
    setup.flush()
    kb_a_id = kb_a.id
    kb_b_id = kb_b.id
    collection_name = _outbox_collection_name()
    point_id = f"legacy-delete-{uuid4()}"
    setup.add(
        IngestionCompensationLog(
            knowledge_base_id=kb_a_id,
            operation=qdrant_outbox.LEGACY_QDRANT_DELETE_OPERATION,
            target_ids_json=[point_id],
            payload_json={
                "collection_name": collection_name,
                "maintenance_operation": "cleanup_stale_data",
            },
            status="failed",
            error_message="TimeoutError: legacy PointIdsList delete outcome unknown",
        )
    )
    setup.commit()
    setup.close()

    store = MemoryVectorStore()
    try:
        attempt = SessionLocal()
        with pytest.raises(
            qdrant_outbox.QdrantOutboxError,
            match="unresolved durable reservation",
        ):
            await qdrant_outbox.execute_qdrant_upsert_batches(
                attempt,
                store=store,
                knowledge_base_id=kb_b_id,
                job_id=None,
                collection_name=collection_name,
                points=[_point(point_id, 0.9)],
                batch_size=1,
            )
        attempt.rollback()
        attempt.close()
        assert point_id not in store.points

        inspect_db = SessionLocal()
        stats = qdrant_outbox.reconcile_qdrant_outbox_sync(
            inspect_db,
            knowledge_base_id=kb_a_id,
            knowledge_base_name="legacy-owner",
            dry_run=True,
        )
        inspect_db.close()
        assert stats["checked_upsert_intents"] == 0
        assert stats["checked_delete_intents"] == 1
        assert stats["delete_recovery_actions"][0]["legacy_unfenced_delete"] is True
        assert stats["delete_recovery_actions"][0]["retryable"] is False
    finally:
        cleanup = SessionLocal()
        for knowledge_base_id in (kb_a_id, kb_b_id):
            knowledge_base = cleanup.get(KnowledgeBase, knowledge_base_id)
            if knowledge_base is not None:
                cleanup.delete(knowledge_base)
        cleanup.commit()
        cleanup.close()


def test_legacy_cross_kb_late_write_reconcile_restores_global_postgresql_owner(monkeypatch):
    from app.db import SessionLocal
    from app.models import (
        Chunk,
        Document,
        DocumentVersion,
        IngestionCompensationLog,
        KnowledgeBase,
        VectorRecord,
    )
    from app.services import qdrant_outbox
    from app.services import vector_store as vector_store_module

    setup = SessionLocal()
    if setup.get_bind().dialect.name != "postgresql":
        setup.close()
        pytest.skip("requires the Docker PostgreSQL runtime")
    kb_a = KnowledgeBase(
        name=f"qdrant_outbox-legacy-a-{uuid4()}",
        description="qdrant_outbox",
        source_root=f"qdrant_outbox/legacy-a-{uuid4()}",
    )
    kb_b = KnowledgeBase(
        name=f"qdrant_outbox-owner-b-{uuid4()}",
        description="qdrant_outbox",
        source_root=f"qdrant_outbox/owner-b-{uuid4()}",
    )
    setup.add_all([kb_a, kb_b])
    setup.flush()
    kb_a_id = kb_a.id
    kb_a_name = kb_a.name
    kb_b_id = kb_b.id
    point_id = str(uuid4())
    document = Document(
        knowledge_base_id=kb_b_id,
        title="Qdrant outbox cross-KB owner",
        source_path=f"qdrant_outbox/cross-kb-{uuid4()}.txt",
        source_type="txt",
        checksum="4" * 64,
    )
    setup.add(document)
    setup.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="4" * 64,
        storage_path=document.source_path,
    )
    setup.add(version)
    setup.flush()
    setup.add(
        Chunk(
            id=point_id,
            knowledge_base_id=kb_b_id,
            document_id=document.id,
            document_version_id=version.id,
            chunk_version=1,
            chunk_index=0,
            token_start=0,
            token_end=2,
            char_start=0,
            char_end=14,
            text="cross kb owner",
            text_hash="5" * 64,
            state="active",
        )
    )
    setup.commit()
    setup.close()

    collection_name = _outbox_collection_name()
    owner_intent_id = str(uuid4())
    stale_intent_id = str(uuid4())

    def owned_point(
        *,
        knowledge_base_id: str,
        intent_id: str,
        marker: str,
        value: float,
    ) -> dict[str, Any]:
        _ = marker
        point = _point(point_id, value)
        point["payload"].update(
            {
                "knowledge_base_id": knowledge_base_id,
                "chunk_id": point_id,
                "qdrant_write_intent_id": intent_id,
                "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
            }
        )
        return point

    owner_point = owned_point(
        knowledge_base_id=kb_b_id,
        intent_id=owner_intent_id,
        marker="6",
        value=0.8,
    )
    stale_point = owned_point(
        knowledge_base_id=kb_a_id,
        intent_id=stale_intent_id,
        marker="7",
        value=0.2,
    )
    store = MemoryVectorStore()
    monkeypatch.setattr(vector_store_module, "VectorStore", lambda *_args, **_kwargs: store)
    try:
        owner_db = SessionLocal()
        assert qdrant_outbox._persist_intent(
            owner_db,
            intent_id=owner_intent_id,
            knowledge_base_id=kb_b_id,
            job_id=None,
            collection_name=collection_name,
            target_points=[owner_point],
            before_points=[],
        )
        qdrant_outbox._transition_intent(
            owner_db,
            intent_id=owner_intent_id,
            status="external_applied",
            details={"reason": "cross_kb_owner_applied"},
        )
        owner_db.add(
            VectorRecord(
                knowledge_base_id=kb_b_id,
                chunk_id=point_id,
                qdrant_point_id=point_id,
                collection_name=collection_name,
                embedding_model=_outbox_vector_identity()["embedding_model"],
                embedding_dimension=2,
                embedding_text_version=_outbox_vector_identity()["embedding_text_version"],
                payload_hash=owner_point["payload"]["vector_payload_hash"],
                vector_status="ready",
                diagnostics_json={
                    "qdrant_write_intent_id": owner_intent_id,
                    "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
                },
            )
        )
        owner_db.commit()
        qdrant_outbox._transition_intent(
            owner_db,
            intent_id=owner_intent_id,
            status="committed",
            details={"reason": "cross_kb_owner_committed"},
        )
        owner_db.close()

        observation = qdrant_outbox._point_observation(
            {point_id: stale_point},
            [point_id],
        )
        stale_payload = {
            "protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
            **_outbox_envelope_contract(qdrant_outbox),
            "intent_id": stale_intent_id,
            "collection_name": collection_name,
            "target_points": [stale_point],
            "before_points": [],
            "target_payload_hash": qdrant_outbox._canonical_hash([stale_point]),
            "before_image_hash": qdrant_outbox._canonical_hash([]),
            "requires_uncertainty_watch": True,
            "lease_expires_at": "2000-01-01T00:00:00",
            "confirmation_not_before": "2000-01-01T00:00:00",
            "uncertainty_observation": observation,
            "state_history": [
                {"status": "uncertainty_observed", "at": "2000-01-01T00:00:00"}
            ],
        }
        stale_db = SessionLocal()
        stale_db.add(
            IngestionCompensationLog(
                id=stale_intent_id,
                job_id=None,
                knowledge_base_id=kb_a_id,
                operation=qdrant_outbox.QDRANT_UPSERT_OPERATION,
                target_ids_json=[point_id],
                payload_json=stale_payload,
                status="uncertainty_observed",
            )
        )
        stale_db.commit()
        stale_db.close()
        store.upsert([stale_point])

        reconcile_db = SessionLocal()
        with qdrant_outbox.qdrant_outbox_reconcile_lock(reconcile_db, kb_a_id):
            result = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=kb_a_id,
                knowledge_base_name=kb_a_name,
                dry_run=False,
                include_unexpired=True,
            )
        reconcile_db.close()

        assert result["verification_pending"] == 1
        assert result["failed"] == 0
        assert store.points[point_id] == owner_point
        audit = SessionLocal()
        stale_row = audit.get(IngestionCompensationLog, stale_intent_id)
        assert stale_row.status == "compensation_verify_pending"
        protected = stale_row.payload_json["uncertainty_protected_superseders"]
        assert protected[0]["knowledge_base_id"] == kb_b_id
        assert protected[0]["owner_intent_id"] == owner_intent_id
        audit.close()
    finally:
        cleanup = SessionLocal()
        for knowledge_base_id in (kb_a_id, kb_b_id):
            knowledge_base = cleanup.get(KnowledgeBase, knowledge_base_id)
            if knowledge_base is not None:
                cleanup.delete(knowledge_base)
        cleanup.commit()
        cleanup.close()


def test_postgres_point_mutation_lock_serializes_same_collection_point():
    from app.db import SessionLocal
    from app.services.qdrant_outbox import QdrantOutboxError, qdrant_point_mutation_lock

    first_db = SessionLocal()
    second_db = SessionLocal()
    try:
        if first_db.get_bind().dialect.name != "postgresql":
            pytest.skip("requires the Docker PostgreSQL runtime")

        with qdrant_point_mutation_lock(
            first_db,
            collection_name="qdrant_outbox-shared-collection",
            point_ids=["shared-point"],
        ):
            with pytest.raises(QdrantOutboxError, match="point mutation is busy"):
                with qdrant_point_mutation_lock(
                    second_db,
                    collection_name="qdrant_outbox-shared-collection",
                    point_ids=["shared-point"],
                ):
                    pass

            with qdrant_point_mutation_lock(
                second_db,
                collection_name="qdrant_outbox-shared-collection",
                point_ids=["different-point"],
            ):
                pass

        with qdrant_point_mutation_lock(
            second_db,
            collection_name="qdrant_outbox-shared-collection",
            point_ids=["shared-point"],
        ):
            pass
    finally:
        first_db.close()
        second_db.close()


def test_real_qdrant_conditional_delete_is_owner_fenced():
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest

    from app.core.config import get_settings
    from app.services import qdrant_outbox
    from app.services.vector_store import VectorStore

    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, timeout=5.0)
    collection_name = f"qdrant_outbox-fenced-delete-{uuid4()}"
    point_id = str(uuid4())
    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=2, distance=rest.Distance.COSINE),
    )
    try:
        point = _point(point_id, 0.35)
        point["payload"].update(
            {
                "knowledge_base_id": "qdrant_outbox-fence-kb",
                "chunk_id": point_id,
                "qdrant_write_intent_id": "qdrant_outbox-old-owner",
                "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
                "vector_payload_hash": "8" * 64,
            }
        )
        client.upsert(
            collection_name=collection_name,
            points=[
                rest.PointStruct(
                    id=point_id,
                    vector=point["vector"],
                    payload=point["payload"],
                )
            ],
            wait=True,
        )
        store = VectorStore(
            "qdrant_outbox-fenced-delete",
            collection_name=collection_name,
            create_if_missing=False,
        )

        wrong_owner = {
            **point,
            "payload": {
                **point["payload"],
                "qdrant_write_intent_id": "qdrant_outbox-new-owner",
            },
        }
        store.delete_if_payload_matches([wrong_owner])
        assert store.get_points([point_id])[0]["payload"]["qdrant_write_intent_id"] == (
            "qdrant_outbox-old-owner"
        )

        store.delete_if_payload_matches([point])
        assert store.get_points([point_id]) == []
    finally:
        client.delete_collection(collection_name=collection_name)


def test_real_qdrant_dry_run_keeps_missing_collection_absent():
    from qdrant_client import QdrantClient

    from app.core.config import get_settings
    from app.db import SessionLocal
    from app.models import KnowledgeBase
    from app.services import qdrant_outbox

    setup = SessionLocal()
    if setup.get_bind().dialect.name != "postgresql":
        setup.close()
        pytest.skip("requires the Docker PostgreSQL/Qdrant runtime")
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, timeout=5.0)
    collection_name = _outbox_collection_name()
    knowledge_base = KnowledgeBase(
        name=f"qdrant_outbox-dryrun-{uuid4()}",
        source_root=f"qdrant_outbox-dryrun/{uuid4()}",
    )
    setup.add(knowledge_base)
    setup.commit()
    knowledge_base_id = knowledge_base.id
    knowledge_base_name = knowledge_base.name
    setup.close()

    try:
        intent_id = str(uuid4())
        target = _point(str(uuid4()), 0.4)
        target["payload"].update(
            {
                "knowledge_base_id": knowledge_base_id,
                "chunk_id": str(target["id"]),
                "qdrant_write_intent_id": intent_id,
                "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
            }
        )
        intent_db = SessionLocal()
        assert qdrant_outbox._persist_intent(
            intent_db,
            intent_id=intent_id,
            knowledge_base_id=knowledge_base_id,
            job_id=None,
            collection_name=collection_name,
            target_points=[target],
            before_points=[],
        )
        qdrant_outbox._transition_intent(
            intent_db,
            intent_id=intent_id,
            status="external_applied",
            details={"reason": "dry_run_missing_collection_probe"},
        )
        intent_db.close()
        assert collection_name not in {item.name for item in client.get_collections().collections}

        reconcile_db = SessionLocal()
        with qdrant_outbox.qdrant_outbox_reconcile_lock(reconcile_db, knowledge_base_id):
            stats = qdrant_outbox.reconcile_qdrant_outbox_sync(
                reconcile_db,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                dry_run=True,
                include_unexpired=True,
            )
        reconcile_db.close()
        assert stats["missing_collections"] == 1
        assert stats["actions"][0]["collection_missing"] is True
        assert collection_name not in {item.name for item in client.get_collections().collections}
    finally:
        if collection_name in {item.name for item in client.get_collections().collections}:
            client.delete_collection(collection_name=collection_name)
        cleanup = SessionLocal()
        cleanup_kb = cleanup.get(KnowledgeBase, knowledge_base_id)
        if cleanup_kb is not None:
            cleanup.delete(cleanup_kb)
            cleanup.commit()
        cleanup.close()


def test_vector_record_dry_run_keeps_missing_collection_absent():
    from qdrant_client import QdrantClient

    from app.core.config import get_settings
    from app.db import SessionLocal
    from app.models import Chunk, Document, DocumentVersion, KnowledgeBase, VectorRecord
    from app.services.maintenance import reconcile_vector_store_sync

    db = SessionLocal()
    if db.get_bind().dialect.name != "postgresql":
        db.close()
        pytest.skip("requires the Docker PostgreSQL/Qdrant runtime")
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, timeout=5.0)
    collection_name = f"qdrant_outbox_vector_dryrun_{uuid4().hex}"
    knowledge_base = KnowledgeBase(
        name=f"qdrant_outbox-vector-dryrun-{uuid4()}",
        source_root=f"qdrant_outbox-vector-dryrun/{uuid4()}",
    )
    db.add(knowledge_base)
    db.flush()
    document = Document(
        knowledge_base_id=knowledge_base.id,
        title="Qdrant outbox dry-run",
        source_path="qdrant_outbox/dry-run.txt",
        source_type="txt",
        checksum="a" * 64,
    )
    db.add(document)
    db.flush()
    document_version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="a" * 64,
        storage_path="qdrant_outbox/dry-run.txt",
    )
    db.add(document_version)
    db.flush()
    chunk = Chunk(
        knowledge_base_id=knowledge_base.id,
        document_id=document.id,
        document_version_id=document_version.id,
        chunk_version=1,
        chunk_index=0,
        token_start=0,
        token_end=2,
        char_start=0,
        char_end=7,
        text="dry run",
        text_hash="b" * 64,
        state="active",
    )
    db.add(chunk)
    db.flush()
    record = VectorRecord(
        knowledge_base_id=knowledge_base.id,
        chunk_id=chunk.id,
        qdrant_point_id=str(uuid4()),
        collection_name=collection_name,
        embedding_model="qdrant_outbox-test",
        embedding_dimension=2,
        embedding_text_version="qdrant_outbox-test-v1",
        payload_hash="c" * 64,
        vector_status="ready",
    )
    db.add(record)
    db.commit()
    knowledge_base_id = knowledge_base.id

    try:
        assert collection_name not in {item.name for item in client.get_collections().collections}
        stats = reconcile_vector_store_sync(db, knowledge_base_id, dry_run=True)
        assert stats["dry_run"] is True
        assert stats["checked_collections"] == 1
        assert stats["missing_points"] == 1
        assert record.vector_status == "ready"
        assert collection_name not in {item.name for item in client.get_collections().collections}
    finally:
        if collection_name in {item.name for item in client.get_collections().collections}:
            client.delete_collection(collection_name=collection_name)
        db.rollback()
        cleanup_kb = db.get(KnowledgeBase, knowledge_base_id)
        if cleanup_kb is not None:
            db.delete(cleanup_kb)
            db.commit()
        db.close()

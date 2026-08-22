from __future__ import annotations

import struct
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select


class MemoryVectorStore:
    def __init__(self, points: list[dict[str, Any]] | None = None, *, fail_on_calls: set[int] | None = None) -> None:
        self.points = {str(point["id"]): dict(point) for point in (points or [])}
        self.fail_on_calls = set(fail_on_calls or set())
        self.upsert_calls = 0
        self.delete_calls: list[list[str]] = []

    def get_points(self, ids: list[str]) -> list[dict[str, Any]]:
        return [dict(self.points[point_id]) for point_id in ids if point_id in self.points]

    async def async_upsert(self, points: list[dict[str, Any]]) -> None:
        self.upsert_calls += 1
        for point in points:
            self.points[str(point["id"])] = dict(point)
        if self.upsert_calls in self.fail_on_calls:
            raise RuntimeError(f"forced partial qdrant failure on call {self.upsert_calls}")

    def upsert(self, points: list[dict[str, Any]]) -> None:
        for point in points:
            self.points[str(point["id"])] = dict(point)

    def delete(self, ids: list[str]) -> None:
        self.delete_calls.append(list(ids))
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

    async def async_delete(self, ids: list[str]) -> None:
        self.delete(ids)


def _point(point_id: str, value: float, *, owner: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"payload_hash": f"hash-{value}"}
    if owner:
        payload["qdrant_write_intent_id"] = owner
    return {"id": point_id, "vector": [value, 1.0 - value], "payload": payload}


@pytest.mark.asyncio
async def test_partial_upsert_exception_remains_active_until_uncertainty_reconcile(db_session, sample_knowledge_base):
    from app.services.qdrant_outbox import (
        QDRANT_OUTBOX_TEST_HISTORY_KEY,
        execute_qdrant_upsert_batches,
    )

    before = _point("existing", 0.1)
    before["payload"].update(
        {"knowledge_base_id": sample_knowledge_base.id, "chunk_id": "existing"}
    )
    store = MemoryVectorStore([before], fail_on_calls={1})
    targets = [_point("existing", 0.9), _point("new", 0.8)]

    with pytest.raises(RuntimeError, match="forced partial qdrant failure"):
        await execute_qdrant_upsert_batches(
            db_session,
            store=store,
            knowledge_base_id=sample_knowledge_base.id,
            job_id=None,
            collection_name="unit-outbox",
            points=targets,
            batch_size=10,
        )

    assert set(store.points) == {"existing", "new"}
    history = db_session.info[QDRANT_OUTBOX_TEST_HISTORY_KEY]
    assert [item["status"] for item in history] == ["pending", "external_outcome_unknown"]
    intent_id = history[0]["intent_id"]
    assert all(point["payload"]["qdrant_write_intent_id"] == intent_id for point in store.points.values())


@pytest.mark.asyncio
async def test_later_batch_failure_compensates_already_applied_batches(db_session, sample_knowledge_base):
    from app.services.qdrant_outbox import (
        QDRANT_OUTBOX_TEST_HISTORY_KEY,
        execute_qdrant_upsert_batches,
    )

    store = MemoryVectorStore(fail_on_calls={2})
    with pytest.raises(RuntimeError, match="call 2"):
        await execute_qdrant_upsert_batches(
            db_session,
            store=store,
            knowledge_base_id=sample_knowledge_base.id,
            job_id=None,
            collection_name="unit-outbox",
            points=[_point("first", 0.2), _point("second", 0.3)],
            batch_size=1,
        )

    assert set(store.points) == {"second"}
    statuses = [item["status"] for item in db_session.info[QDRANT_OUTBOX_TEST_HISTORY_KEY]]
    assert statuses.count("pending") == 2
    assert statuses.count("compensated") == 1
    assert statuses.count("external_outcome_unknown") == 1
    assert "external_applied" in statuses


@pytest.mark.asyncio
async def test_transaction_commit_marks_all_external_writes_committed(db_session, sample_knowledge_base):
    from app.services.qdrant_outbox import (
        QDRANT_OUTBOX_TEST_HISTORY_KEY,
        execute_qdrant_upsert_batches,
    )

    db_session.scalar(select(type(sample_knowledge_base)).where(type(sample_knowledge_base).id == sample_knowledge_base.id))
    store = MemoryVectorStore()
    result = await execute_qdrant_upsert_batches(
        db_session,
        store=store,
        knowledge_base_id=sample_knowledge_base.id,
        job_id=None,
        collection_name="unit-outbox",
        points=[_point("first", 0.2), _point("second", 0.3)],
        batch_size=1,
    )
    db_session.commit()

    assert len(result["intent_ids"]) == 2
    assert result["batches"] == 2
    assert result["durable"] is False
    history = db_session.info[QDRANT_OUTBOX_TEST_HISTORY_KEY]
    assert [item["status"] for item in history].count("committed") == 2
    assert all(
        point["payload"]["qdrant_write_intent_id"] == result["point_intent_ids"][point_id]
        for point_id, point in store.points.items()
    )


@pytest.mark.asyncio
async def test_transaction_rollback_leaves_compensation_pending_for_reconcile(db_session, sample_knowledge_base):
    from app.services.qdrant_outbox import (
        QDRANT_OUTBOX_TEST_HISTORY_KEY,
        execute_qdrant_upsert_batches,
    )

    db_session.scalar(select(type(sample_knowledge_base)).where(type(sample_knowledge_base).id == sample_knowledge_base.id))
    store = MemoryVectorStore()
    await execute_qdrant_upsert_batches(
        db_session,
        store=store,
        knowledge_base_id=sample_knowledge_base.id,
        job_id=None,
        collection_name="unit-outbox",
        points=[_point("orphan", 0.7)],
        batch_size=1,
    )
    db_session.rollback()

    statuses = [item["status"] for item in db_session.info[QDRANT_OUTBOX_TEST_HISTORY_KEY]]
    assert statuses[-1] == "compensation_pending"
    assert "orphan" in store.points


@pytest.mark.asyncio
async def test_nested_commit_keeps_handle_until_root_transaction_rolls_back(db_session, sample_knowledge_base):
    from app.services.qdrant_outbox import (
        QDRANT_OUTBOX_SESSION_KEY,
        QDRANT_OUTBOX_TEST_HISTORY_KEY,
        execute_qdrant_upsert_batches,
    )

    db_session.scalar(select(type(sample_knowledge_base)).where(type(sample_knowledge_base).id == sample_knowledge_base.id))
    nested = db_session.begin_nested()
    await execute_qdrant_upsert_batches(
        db_session,
        store=MemoryVectorStore(),
        knowledge_base_id=sample_knowledge_base.id,
        job_id=None,
        collection_name="unit-outbox",
        points=[_point("savepoint-orphan", 0.6)],
        batch_size=1,
    )

    nested.commit()
    after_savepoint = [item["status"] for item in db_session.info[QDRANT_OUTBOX_TEST_HISTORY_KEY]]
    assert "committed" not in after_savepoint
    assert len(db_session.info[QDRANT_OUTBOX_SESSION_KEY]) == 1

    db_session.rollback()
    after_root_rollback = [item["status"] for item in db_session.info[QDRANT_OUTBOX_TEST_HISTORY_KEY]]
    assert after_root_rollback[-1] == "compensation_pending"
    assert not db_session.info.get(QDRANT_OUTBOX_SESSION_KEY)


@pytest.mark.asyncio
async def test_nested_rollback_only_compensates_savepoint_handles(db_session, sample_knowledge_base):
    from app.services.qdrant_outbox import (
        QDRANT_OUTBOX_SESSION_KEY,
        QDRANT_OUTBOX_TEST_HISTORY_KEY,
        execute_qdrant_upsert_batches,
    )

    db_session.scalar(select(type(sample_knowledge_base)).where(type(sample_knowledge_base).id == sample_knowledge_base.id))
    nested = db_session.begin_nested()
    await execute_qdrant_upsert_batches(
        db_session,
        store=MemoryVectorStore(),
        knowledge_base_id=sample_knowledge_base.id,
        job_id=None,
        collection_name="unit-outbox",
        points=[_point("rolled-back-savepoint", 0.4)],
        batch_size=1,
    )

    nested.rollback()
    after_savepoint_rollback = [item["status"] for item in db_session.info[QDRANT_OUTBOX_TEST_HISTORY_KEY]]
    assert after_savepoint_rollback[-1] == "compensation_pending"
    assert not db_session.info.get(QDRANT_OUTBOX_SESSION_KEY)

    db_session.commit()
    final_statuses = [item["status"] for item in db_session.info[QDRANT_OUTBOX_TEST_HISTORY_KEY]]
    assert final_statuses[-1] == "compensation_pending"
    assert "committed" not in final_statuses


def test_v1_active_intent_envelope_remains_reconcilable_but_mixed_protocol_fails():
    from app.services import qdrant_outbox

    intent_id = "legacy-v1-intent"
    target = _point("legacy-point", 0.5)
    target["payload"].update(
        {
            "knowledge_base_id": "kb-legacy",
            "chunk_id": "legacy-point",
            "qdrant_write_intent_id": intent_id,
            "qdrant_write_protocol_version": "qdrant_side_effect_outbox_v1",
        }
    )
    payload = {
        "protocol_version": "qdrant_side_effect_outbox_v1",
        "intent_id": intent_id,
        "collection_name": "legacy-v1",
        "target_points": [target],
        "before_points": [],
        "target_payload_hash": qdrant_outbox._canonical_hash([target]),
        "before_image_hash": qdrant_outbox._canonical_hash([]),
        "state_history": [{"status": "pending"}, {"status": "external_applied"}],
    }
    row = SimpleNamespace(
        id=intent_id,
        payload_json=payload,
        target_ids_json=["legacy-point"],
        status="external_applied",
        knowledge_base_id="kb-legacy",
    )

    targets, before = qdrant_outbox._validated_reconcile_payload(row)
    assert targets == [target]
    assert before == []
    assert qdrant_outbox._requires_uncertainty_watch(row) is False

    ambiguous_payload = {
        **payload,
        "state_history": [{"status": "pending"}, {"status": "reconcile_failed"}],
    }
    assert qdrant_outbox._requires_uncertainty_watch(
        SimpleNamespace(status="reconcile_failed", payload_json=ambiguous_payload)
    ) is True

    tampered_target = {
        **target,
        "payload": {
            **target["payload"],
            "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
        },
    }
    tampered_payload = {
        **payload,
        "target_points": [tampered_target],
        "target_payload_hash": qdrant_outbox._canonical_hash([tampered_target]),
    }
    with pytest.raises(qdrant_outbox.QdrantOutboxError, match="invalid target point"):
        qdrant_outbox._validated_reconcile_payload(
            SimpleNamespace(
                id=intent_id,
                payload_json=tampered_payload,
                target_ids_json=["legacy-point"],
                knowledge_base_id="kb-legacy",
            )
        )

    wrong_kb_target = {
        **target,
        "payload": {**target["payload"], "knowledge_base_id": "kb-other"},
    }
    with pytest.raises(qdrant_outbox.QdrantOutboxError, match="escapes its knowledge-base"):
        qdrant_outbox._validated_reconcile_payload(
            SimpleNamespace(
                id=intent_id,
                knowledge_base_id="kb-legacy",
                payload_json={
                    **payload,
                    "target_points": [wrong_kb_target],
                    "target_payload_hash": qdrant_outbox._canonical_hash([wrong_kb_target]),
                },
                target_ids_json=["legacy-point"],
            )
        )

    extra_before = _point("out-of-scope-before", 0.2)
    extra_before["payload"].update(
        {"knowledge_base_id": "kb-legacy", "chunk_id": "out-of-scope-before"}
    )
    with pytest.raises(qdrant_outbox.QdrantOutboxError, match="before-image ids escape"):
        qdrant_outbox._validated_reconcile_payload(
            SimpleNamespace(
                id=intent_id,
                knowledge_base_id="kb-legacy",
                payload_json={
                    **payload,
                    "before_points": [extra_before],
                    "before_image_hash": qdrant_outbox._canonical_hash([extra_before]),
                },
                target_ids_json=["legacy-point"],
            )
        )


def test_authoritative_reconcile_never_mutates_cross_kb_current_point(db_session):
    from app.services import qdrant_outbox

    intent_id = "kb-a-intent"
    point_id = "shared-point-id"
    target = _point(point_id, 0.8, owner=intent_id)
    target["payload"].update(
        {
            "knowledge_base_id": "kb-a",
            "chunk_id": point_id,
            "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
        }
    )
    foreign = _point(point_id, 0.3, owner="kb-b-owner")
    foreign["payload"].update(
        {"knowledge_base_id": "kb-b", "chunk_id": point_id}
    )
    store = MemoryVectorStore([foreign])

    with pytest.raises(qdrant_outbox.QdrantOutboxError, match="ungrounded current"):
        qdrant_outbox._sync_compensate_intent(
            db=db_session,
            store=store,
            row=SimpleNamespace(
                id=intent_id,
                knowledge_base_id="kb-a",
                payload_json={"collection_name": "qdrant_outbox-shared-collection"},
            ),
            target_points=[target],
            before_points=[],
            protected_points=[],
            dry_run=False,
            authoritative=True,
        )

    assert store.delete_calls == []
    assert store.points[point_id] == foreign


def test_outbox_state_history_is_bounded_with_first_last_and_total_audit():
    from app.services import qdrant_outbox

    first = {"status": "pending", "at": "first"}
    payload = {"state_history": [first]}
    transition_count = qdrant_outbox.QDRANT_OUTBOX_STATE_HISTORY_LIMIT + 10
    for index in range(transition_count):
        qdrant_outbox._append_bounded_state_history(
            payload,
            {
                "status": "compensation_verify_pending",
                "at": f"watch-{index}",
                "details": {"reason": f"watch-cycle-{index}"},
            },
        )

    assert len(payload["state_history"]) == qdrant_outbox.QDRANT_OUTBOX_STATE_HISTORY_LIMIT
    assert payload["state_history_total_count"] == transition_count + 1
    assert payload["state_history_first_transition"] == first
    assert payload["last_transition"]["at"] == f"watch-{transition_count - 1}"
    assert payload["state_history_truncated_count"] == 11


def test_knowledge_base_delete_preserves_active_upsert_tombstone(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from app.models import IngestionCompensationLog
    from app.services import maintenance, qdrant_outbox

    intent = IngestionCompensationLog(
        knowledge_base_id=sample_knowledge_base.id,
        operation=qdrant_outbox.QDRANT_UPSERT_OPERATION,
        target_ids_json=["late-point"],
        payload_json={"requires_uncertainty_watch": True},
        status="external_outcome_unknown",
    )
    db_session.add(intent)
    db_session.commit()

    def forbidden_store(*_args, **_kwargs):
        raise AssertionError("Qdrant must not be touched while an upsert tombstone is active")

    monkeypatch.setattr(maintenance, "VectorStore", forbidden_store)
    with pytest.raises(maintenance.MaintenanceConflict, match="durable Qdrant upsert intents"):
        maintenance.delete_knowledge_base_data(db_session, sample_knowledge_base)

    assert db_session.get(type(sample_knowledge_base), sample_knowledge_base.id) is not None
    assert db_session.get(IngestionCompensationLog, intent.id) is not None


def test_qdrant_delete_attempt_is_persisted_before_external_mutation(
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.services import qdrant_outbox

    point_id = "delete-scope-point"
    current = _point(point_id, 0.4)
    current["payload"].update(
        {
            "knowledge_base_id": sample_knowledge_base.id,
            "chunk_id": point_id,
            "qdrant_write_intent_id": "delete-scope-owner",
            "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
        }
    )
    intent_id = qdrant_outbox.persist_qdrant_delete_attempt(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        collection_name="delete-scope",
        target_ids=[point_id],
        current_points=[current],
        reason="unit_test",
    )
    row = db_session.get(IngestionCompensationLog, intent_id)
    assert row is not None
    assert row.operation == qdrant_outbox.QDRANT_DELETE_OPERATION
    assert row.status == "pending"
    assert row.payload_json["mutation_attempt"]["state"] == "pending"
    assert row.payload_json["requires_uncertainty_watch"] is True

    qdrant_outbox.record_qdrant_delete_attempt_error(
        db_session,
        intent_id=intent_id,
        error=TimeoutError("accepted then timed out"),
    )
    db_session.expire_all()
    row = db_session.get(IngestionCompensationLog, intent_id)
    assert row.status == "external_outcome_unknown"


def test_qdrant_delete_applied_and_postgresql_commit_share_auditable_lifecycle(
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.services import qdrant_outbox

    point_id = "delete-lifecycle-point"
    current = _point(point_id, 0.6)
    current["payload"].update(
        {
            "knowledge_base_id": sample_knowledge_base.id,
            "chunk_id": point_id,
            "qdrant_write_intent_id": "delete-lifecycle-owner",
            "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
        }
    )
    intent_id = qdrant_outbox.persist_qdrant_delete_attempt(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        collection_name="delete-lifecycle",
        target_ids=[point_id],
        current_points=[current],
        reason="unit_test_lifecycle",
    )

    qdrant_outbox.record_qdrant_delete_attempt_applied(
        db_session,
        intent_id=intent_id,
    )
    row = db_session.get(IngestionCompensationLog, intent_id)
    assert row.status == "external_applied"
    assert row.payload_json["mutation_attempt"]["state"] == "transport_resolved"
    assert row.payload_json["requires_uncertainty_watch"] is False
    assert [item.id for item in qdrant_outbox.pending_qdrant_delete_intents(db_session)] == [
        intent_id
    ]

    qdrant_outbox.mark_qdrant_delete_attempts_committed(
        db_session,
        intent_ids=[intent_id],
    )
    db_session.commit()
    row = db_session.get(IngestionCompensationLog, intent_id)
    assert row.status == "committed"
    assert qdrant_outbox.pending_qdrant_delete_intents(db_session) == []


def test_pending_delete_replay_is_owner_fenced_and_preserves_newer_point(
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.services import qdrant_outbox

    point_id = "conditional-delete-replay"
    old_point = _point(point_id, 0.2, owner="old-owner")
    old_point["payload"].update(
        {
            "knowledge_base_id": sample_knowledge_base.id,
            "chunk_id": point_id,
            "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
        }
    )
    intent_id = qdrant_outbox.persist_qdrant_delete_attempt(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        collection_name="conditional-delete-replay",
        target_ids=[point_id],
        current_points=[old_point],
        reason="cleanup_stale_data",
    )
    row = db_session.get(IngestionCompensationLog, intent_id)

    newer_point = _point(point_id, 0.9, owner="new-owner")
    newer_point["payload"].update(
        {
            "knowledge_base_id": sample_knowledge_base.id,
            "chunk_id": point_id,
            "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
        }
    )
    store = MemoryVectorStore([newer_point])

    replayed = qdrant_outbox.replay_qdrant_delete_intent(
        db_session,
        store=store,
        row=row,
    )

    assert replayed["status"] == "external_applied"
    assert replayed["replayed"] is True
    assert store.points[point_id] == newer_point
    db_session.expire_all()
    assert db_session.get(IngestionCompensationLog, intent_id).status == "external_applied"


def test_legacy_delete_v1_is_not_replay_supported_and_requires_manual_resolution():
    from app.services import qdrant_outbox

    row = SimpleNamespace(
        id="legacy-delete-v1",
        knowledge_base_id="kb-legacy",
        operation=qdrant_outbox.QDRANT_DELETE_OPERATION,
        status="external_outcome_unknown",
        target_ids_json=["point-1"],
        payload_json={
            "protocol_version": "qdrant_side_effect_delete_outbox_v1",
            "intent_id": "legacy-delete-v1",
            "reason": "cleanup_stale_data",
            "collection_name": "legacy-collection",
            "target_ids": ["point-1"],
            "before_points": [],
            "before_points_hash": qdrant_outbox._canonical_hash([]),
        },
    )

    assert "qdrant_side_effect_delete_outbox_v1" not in (
        qdrant_outbox.QDRANT_DELETE_SUPPORTED_PROTOCOL_VERSIONS
    )
    diagnostics = qdrant_outbox.qdrant_delete_intent_recovery_diagnostics(row)
    assert diagnostics["reason"] == "qdrant_delete_manual_resolution_required"
    assert diagnostics["retryable"] is False
    assert diagnostics["delete_protocol_supported"] is False
    with pytest.raises(qdrant_outbox.QdrantOutboxError, match="unsupported protocol"):
        qdrant_outbox._validated_qdrant_delete_intent(row)


def test_reconcile_reports_cleanup_delete_recovery_without_replaying(
    db_session,
    sample_knowledge_base,
):
    from app.services import qdrant_outbox

    intent_id = qdrant_outbox.persist_qdrant_delete_attempt(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        collection_name="cleanup-recovery-report",
        target_ids=["missing-point"],
        current_points=[],
        reason="cleanup_stale_data",
    )

    stats = qdrant_outbox.reconcile_qdrant_outbox_sync(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        knowledge_base_name=sample_knowledge_base.name,
        dry_run=True,
    )

    assert stats["checked_intents"] == 1
    assert stats["checked_upsert_intents"] == 0
    assert stats["checked_delete_intents"] == 1
    assert stats["blocking_delete_intents"] == 1
    assert stats["actions"] == []
    assert stats["delete_recovery_actions"] == [
        {
            "delete_tombstone_id": intent_id,
            "delete_tombstone_status": "pending",
            "delete_tombstone_operation": qdrant_outbox.QDRANT_DELETE_OPERATION,
            "delete_source_operation": "cleanup_stale_data",
            "delete_protocol_version": qdrant_outbox.QDRANT_DELETE_PROTOCOL_VERSION,
            "delete_protocol_supported": True,
            "conditional_delete_replay_safe": True,
            "legacy_unfenced_delete": False,
            "delete_intent_validation_error": None,
            "collection_name": "cleanup-recovery-report",
            "target_count": 1,
            "reason": "stale_cleanup_delete_pending",
            "retryable": True,
            "retry_guidance": "retry_cleanup_stale_data_with_identical_scope",
        }
    ]


def test_cleanup_stale_data_persists_delete_intent_before_qdrant_mutation(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from app.models import (
        Chunk,
        Document,
        DocumentVersion,
        IngestionCompensationLog,
        VectorRecord,
    )
    from app.services import maintenance, qdrant_outbox

    sample_knowledge_base.current_chunk_version = 2
    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Qdrant outbox stale cleanup",
        source_path="qdrant_outbox/stale-cleanup.txt",
        source_type="txt",
        checksum="a" * 64,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum="a" * 64,
        storage_path=document.source_path,
        is_active=False,
    )
    db_session.add(version)
    db_session.flush()
    chunk = Chunk(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        chunk_version=1,
        chunk_index=0,
        token_start=0,
        token_end=2,
        char_start=0,
        char_end=13,
        text="stale cleanup",
        text_hash="b" * 64,
        state="inactive",
    )
    db_session.add(chunk)
    db_session.flush()
    record = VectorRecord(
        knowledge_base_id=sample_knowledge_base.id,
        chunk_id=chunk.id,
        qdrant_point_id=chunk.id,
        collection_name="qdrant_outbox-stale-cleanup",
        embedding_model="qdrant_outbox-test",
        embedding_dimension=2,
        embedding_text_version="qdrant_outbox-v1",
        payload_hash="c" * 64,
        vector_status="ready",
        diagnostics_json={},
    )
    db_session.add(record)
    db_session.commit()
    record_id = record.id

    point = _point(chunk.id, 0.4)
    point["payload"].update(
        {
            "knowledge_base_id": sample_knowledge_base.id,
            "chunk_id": chunk.id,
            "qdrant_write_intent_id": "stale-cleanup-owner",
            "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
        }
    )

    class InspectingStore(MemoryVectorStore):
        fail_once = True
        observed_intent_statuses: list[str] = []

        def delete(self, ids: list[str]) -> None:
            intent = db_session.scalar(
                select(IngestionCompensationLog).where(
                    IngestionCompensationLog.operation
                    == qdrant_outbox.QDRANT_DELETE_OPERATION,
                )
            )
            assert intent is not None
            assert intent.target_ids_json == [chunk.id]
            assert db_session.get(VectorRecord, record_id) is not None
            self.observed_intent_statuses.append(intent.status)
            if self.fail_once:
                self.fail_once = False
                raise TimeoutError("conditional cleanup delete timed out")
            super().delete(ids)

    store = InspectingStore([point])
    monkeypatch.setattr(maintenance, "VectorStore", lambda *_args, **_kwargs: store)

    with pytest.raises(TimeoutError, match="cleanup delete timed out"):
        maintenance.cleanup_stale_data(
            db_session,
            sample_knowledge_base.id,
            sample_knowledge_base.name,
            dry_run=False,
            delete_inactive_chunks=False,
        )
    db_session.expire_all()
    tombstone = db_session.scalar(
        select(IngestionCompensationLog).where(
            IngestionCompensationLog.operation == qdrant_outbox.QDRANT_DELETE_OPERATION
        )
    )
    assert tombstone.status == "external_outcome_unknown"

    result = maintenance.cleanup_stale_data(
        db_session,
        sample_knowledge_base.id,
        sample_knowledge_base.name,
        dry_run=False,
        delete_inactive_chunks=False,
    )

    assert result["stale_vector_records"] == 1
    assert store.delete_calls == [[chunk.id]]
    assert store.observed_intent_statuses == ["pending", "external_outcome_unknown"]
    assert db_session.scalar(
        select(VectorRecord.id).where(VectorRecord.id == record_id)
    ) is None
    intent = db_session.scalar(
        select(IngestionCompensationLog).where(
            IngestionCompensationLog.operation == qdrant_outbox.QDRANT_DELETE_OPERATION
        )
    )
    assert intent.status == "committed"
    assert intent.payload_json["reason"] == "cleanup_stale_data"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "delete_status",
    ["pending", "external_outcome_unknown", "external_applied"],
)
async def test_pending_delete_tombstone_blocks_new_ingestion_lock(
    db_session,
    sample_knowledge_base,
    delete_status,
):
    from app.models import IngestionCompensationLog
    from app.services import qdrant_outbox
    from app.services.ingestion_resource_lock import (
        IngestionResourceBusyError,
        knowledge_base_ingestion_resource_lock,
    )

    tombstone_id = f"delete-lock-{delete_status}"
    before_points: list[dict[str, Any]] = []
    tombstone = IngestionCompensationLog(
        id=tombstone_id,
        knowledge_base_id=sample_knowledge_base.id,
        operation=qdrant_outbox.QDRANT_DELETE_OPERATION,
        target_ids_json=["point"],
        payload_json={
            "protocol_version": qdrant_outbox.QDRANT_DELETE_PROTOCOL_VERSION,
            "intent_id": tombstone_id,
            "knowledge_base_id": sample_knowledge_base.id,
            "target_ids": ["point"],
            "before_points": before_points,
            "before_points_hash": qdrant_outbox._canonical_hash(before_points),
            "conditional_delete_protocol_version": (
                qdrant_outbox.QDRANT_CONDITIONAL_DELETE_PROTOCOL_VERSION
            ),
            "conditional_delete_replay_safe": True,
            "reason": "delete_knowledge_base_data",
            "collection_name": "delete-lock",
            "requires_uncertainty_watch": True,
        },
        status=delete_status,
    )
    db_session.add(tombstone)
    db_session.commit()

    with pytest.raises(IngestionResourceBusyError) as exc_info:
        async with knowledge_base_ingestion_resource_lock(
            db_session,
            sample_knowledge_base.id,
            operation="ingest_file",
            timeout_seconds=0.1,
        ):
            pass
    assert exc_info.value.diagnostics["reason"] == "knowledge_base_delete_pending"
    assert exc_info.value.diagnostics["delete_tombstone_id"] == tombstone.id
    assert (
        exc_info.value.diagnostics["retry_guidance"]
        == "finish_or_retry_knowledge_base_deletion"
    )


@pytest.mark.asyncio
async def test_cleanup_delete_tombstone_guides_matching_cleanup_retry(
    db_session,
    sample_knowledge_base,
):
    from app.services import qdrant_outbox
    from app.services.ingestion_resource_lock import (
        IngestionResourceBusyError,
        knowledge_base_ingestion_resource_lock,
    )

    intent_id = qdrant_outbox.persist_qdrant_delete_attempt(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        collection_name="cleanup-lock",
        target_ids=["point"],
        current_points=[],
        reason="cleanup_stale_data",
    )

    with pytest.raises(IngestionResourceBusyError) as exc_info:
        async with knowledge_base_ingestion_resource_lock(
            db_session,
            sample_knowledge_base.id,
            operation="ingest_file",
            timeout_seconds=0.1,
        ):
            pass

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["reason"] == "stale_cleanup_delete_pending"
    assert diagnostics["delete_tombstone_id"] == intent_id
    assert diagnostics["retryable"] is True
    assert diagnostics["retry_guidance"] == "retry_cleanup_stale_data_with_identical_scope"


@pytest.mark.asyncio
async def test_legacy_failed_delete_blocks_point_ingestion_and_reports_manual_recovery(
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.services import maintenance, qdrant_outbox
    from app.services.ingestion_resource_lock import (
        IngestionResourceBusyError,
        knowledge_base_ingestion_resource_lock,
    )

    point_id = "legacy-unfenced-delete-point"
    legacy = IngestionCompensationLog(
        knowledge_base_id=sample_knowledge_base.id,
        operation=qdrant_outbox.LEGACY_QDRANT_DELETE_OPERATION,
        target_ids_json=[point_id],
        payload_json={
            "collection_name": "legacy-unfenced-delete",
            "maintenance_operation": "cleanup_stale_data",
        },
        status="failed",
        error_message="TimeoutError: legacy delete outcome unknown",
    )
    db_session.add(legacy)
    db_session.commit()

    with pytest.raises(qdrant_outbox.QdrantOutboxError, match="unresolved durable reservation"):
        qdrant_outbox._assert_qdrant_point_scope_available(
            db_session,
            knowledge_base_id="different-knowledge-base",
            collection_name="legacy-unfenced-delete",
            point_ids=[point_id],
            requested_operation=qdrant_outbox.QDRANT_UPSERT_OPERATION,
        )

    stats = qdrant_outbox.reconcile_qdrant_outbox_sync(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        knowledge_base_name=sample_knowledge_base.name,
        dry_run=True,
    )
    assert stats["checked_delete_intents"] == 1
    recovery = stats["delete_recovery_actions"][0]
    assert recovery["delete_tombstone_id"] == legacy.id
    assert recovery["delete_tombstone_operation"] == "qdrant_delete"
    assert recovery["delete_source_operation"] == "cleanup_stale_data"
    assert recovery["legacy_unfenced_delete"] is True
    assert recovery["retryable"] is False
    assert recovery["reason"] == "legacy_qdrant_delete_manual_resolution_required"

    with pytest.raises(IngestionResourceBusyError) as exc_info:
        async with knowledge_base_ingestion_resource_lock(
            db_session,
            sample_knowledge_base.id,
            operation="ingest_file",
            timeout_seconds=0.1,
        ):
            pass
    assert (
        exc_info.value.diagnostics["reason"]
        == "legacy_qdrant_delete_manual_resolution_required"
    )

    with pytest.raises(maintenance.MaintenanceConflict, match="unfenced Qdrant delete intent"):
        maintenance.cleanup_stale_data(
            db_session,
            sample_knowledge_base.id,
            sample_knowledge_base.name,
            dry_run=True,
        )


def test_cleanup_dry_run_never_updates_or_rolls_back_caller_state(
    db_session,
    sample_knowledge_base,
):
    from sqlalchemy import event

    from app.models import IngestionCompensationLog
    from app.services import maintenance, qdrant_outbox

    tombstone_id = "dry-run-scope-mismatch"
    before_points: list[dict[str, Any]] = []
    tombstone = IngestionCompensationLog(
        id=tombstone_id,
        knowledge_base_id=sample_knowledge_base.id,
        operation=qdrant_outbox.QDRANT_DELETE_OPERATION,
        target_ids_json=["missing-point"],
        payload_json={
            "protocol_version": qdrant_outbox.QDRANT_DELETE_PROTOCOL_VERSION,
            "intent_id": tombstone_id,
            "knowledge_base_id": sample_knowledge_base.id,
            "reason": "cleanup_stale_data",
            "collection_name": "scope-mismatch",
            "target_ids": ["missing-point"],
            "before_points": before_points,
            "before_points_hash": qdrant_outbox._canonical_hash(before_points),
            "conditional_delete_protocol_version": (
                qdrant_outbox.QDRANT_CONDITIONAL_DELETE_PROTOCOL_VERSION
            ),
            "conditional_delete_replay_safe": True,
        },
        status="pending",
    )
    db_session.add(tombstone)
    db_session.commit()
    knowledge_base_id = sample_knowledge_base.id
    knowledge_base_name = sample_knowledge_base.name
    sample_knowledge_base.description = "caller-owned-uncommitted-change"

    statements: list[str] = []
    bind = db_session.get_bind()

    def capture_statement(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(str(statement))

    event.listen(bind, "before_cursor_execute", capture_statement)
    try:
        with pytest.raises(maintenance.MaintenanceConflict, match="scope no longer matches"):
            maintenance.cleanup_stale_data(
                db_session,
                knowledge_base_id,
                knowledge_base_name,
                dry_run=True,
                delete_inactive_chunks=False,
            )
    finally:
        event.remove(bind, "before_cursor_execute", capture_statement)

    assert not any(statement.lstrip().upper().startswith("UPDATE") for statement in statements)
    assert sample_knowledge_base.description == "caller-owned-uncommitted-change"
    assert sample_knowledge_base in db_session.dirty


def _strict_v2_point(
    point_id: str,
    vector: list[float],
    *,
    knowledge_base_id: str,
    owner_intent_id: str | None = None,
    embedding_model: str = "vector_schema/outbox-v2",
    embedding_text_version: str = "contextual_text_v2",
    chunk_schema_version: str = "fixed_token_chunk_v1",
) -> tuple[str, dict[str, Any]]:
    from app.services import qdrant_outbox
    from app.services.context_graph import (
        QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
        QDRANT_VECTOR_DISTANCE_METRIC,
        VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION,
        qdrant_collection_identity_digest,
        qdrant_collection_name,
        vector_payload_hash,
    )

    dimension = len(vector)
    identity = {
        "embedding_model": embedding_model,
        "embedding_dimension": dimension,
        "embedding_text_version": embedding_text_version,
        "chunk_schema_version": chunk_schema_version,
    }
    identity_digest = qdrant_collection_identity_digest(**identity)
    context_hash_protocol_version = "vector_schema-context-hash-protocol-v1"
    context_hash = f"vector_schema-context-hash-{point_id}"
    local_hint_protocol_version = "vector_schema-local-hint-protocol-v1"
    local_hint_hash = f"vector_schema-local-hint-hash-{point_id}"
    payload = {
        "knowledge_base_id": knowledge_base_id,
        "chunk_id": point_id,
        "embedding_model": embedding_model,
        "embedding_dimension": dimension,
        "vector_distance_metric": QDRANT_VECTOR_DISTANCE_METRIC,
        "embedding_text_version": embedding_text_version,
        "chunk_schema_version": chunk_schema_version,
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
            embedding_model=embedding_model,
            embedding_dimension=dimension,
            vector_distance_metric=QDRANT_VECTOR_DISTANCE_METRIC,
            embedding_text_version=embedding_text_version,
            chunk_schema_version=chunk_schema_version,
            context_hash_protocol_version=context_hash_protocol_version,
            context_hash=context_hash,
            local_hint_protocol_version=local_hint_protocol_version,
            local_hint_hash=local_hint_hash,
            collection_identity_protocol_version=QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
            collection_identity_digest=identity_digest,
        ),
        "nested_metadata": {"labels": ["one", "two"]},
    }
    if owner_intent_id is not None:
        payload.update(
            {
                "qdrant_write_intent_id": owner_intent_id,
                "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
            }
        )
    return qdrant_collection_name(**identity), {
        "id": point_id,
        "vector": list(vector),
        "payload": payload,
    }


def test_outbox_protocol_contracts_have_frozen_schema_and_canonical_byte_goldens():
    from app.services import qdrant_outbox

    assert qdrant_outbox.QDRANT_OUTBOX_V1_ENVELOPE_SCHEMA_HASH == (
        "910a87d94eefd2f81adf1f4ee69fea9202cbde0263a20ded1f195e4bcac9f666"
    )
    assert qdrant_outbox.QDRANT_OUTBOX_V2_ENVELOPE_SCHEMA_HASH == (
        "fa90a6e862a11d35bae9c554ff36dc6f8899f99e2727ae50e0212e5475d6b8ba"
    )

    v1_value = {"z": [0.25, "\u65e7"], "at": datetime(2020, 1, 2, 3, 4, 5)}
    assert qdrant_outbox._outbox_v1_canonical_bytes(v1_value).hex() == (
        "7b226174223a22323032302d30312d30322030333a30343a3035222c227a223a"
        "5b302e32352c22e697a7225d7d"
    )
    assert qdrant_outbox._outbox_payload_hash(
        qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_V1,
        v1_value,
    ) == "315cccf7640888917798b0763bdd57d8844bd872019b1ef1fdf29b473f1a2142"

    v2_value = {"z": [0.25, "\u65e7"], "a": {"flag": True, "none": None}}
    assert qdrant_outbox._outbox_v2_canonical_bytes(v2_value).hex() == (
        "7b2261223a7b22666c6167223a747275652c226e6f6e65223a6e756c6c7d2c"
        "227a223a5b302e32352c22e697a7225d7d"
    )
    assert qdrant_outbox._outbox_payload_hash(
        qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
        v2_value,
    ) == "11cae8b0c4ccbbd4d4066f6a41ef5fadf33d65d9f5925f12e05a4b10cb1f0633"
    with pytest.raises(qdrant_outbox.QdrantOutboxError, match="strict-JSON schema"):
        qdrant_outbox._outbox_v2_canonical_bytes(v1_value)


def test_v1_decoder_uses_historical_golden_and_is_independent_of_generic_hash(monkeypatch):
    from app.services import qdrant_outbox

    intent_id = "legacy-v1-intent"
    target = {
        "id": "legacy-point",
        "vector": [0.25, 0.75],
        "payload": {
            "knowledge_base_id": "kb-legacy",
            "chunk_id": "legacy-point",
            "qdrant_write_intent_id": intent_id,
            "qdrant_write_protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_V1,
        },
    }
    row = SimpleNamespace(
        id=intent_id,
        knowledge_base_id="kb-legacy",
        target_ids_json=["legacy-point"],
        payload_json={
            "protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_V1,
            "intent_id": intent_id,
            "collection_name": "historical-v1",
            "target_points": [target],
            "before_points": [],
            "target_payload_hash": "f79342e0213299fb4e53eb195df11e0588a99dbf0e204f0361cc25ab9b346b19",
            "before_image_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        },
    )
    monkeypatch.setattr(
        qdrant_outbox,
        "_canonical_hash",
        lambda _payload: (_ for _ in ()).throw(AssertionError("generic hash must not run")),
    )

    decoded_target, decoded_before = qdrant_outbox._validated_reconcile_payload(row)
    assert decoded_target == [target]
    assert decoded_before == []
    assert decoded_target[0] is not target
    assert decoded_target[0]["payload"] is not target["payload"]


@pytest.mark.parametrize(
    "missing_field",
    [
        "embedding_model",
        "embedding_dimension",
        "vector_distance_metric",
        "embedding_text_version",
        "chunk_schema_version",
        "context_hash_protocol_version",
        "context_hash",
        "local_hint_protocol_version",
        "local_hint_hash",
        "collection_identity_protocol_version",
        "collection_identity_digest",
        "vector_payload_hash_protocol",
        "vector_payload_hash",
    ],
)
def test_v2_prepare_rejects_every_missing_schema_card_field(missing_field):
    from app.services import qdrant_outbox

    intent_id = "strict-v2-missing-field"
    collection_name, point = _strict_v2_point(
        "point-a",
        [0.1, 0.9],
        knowledge_base_id="kb-a",
        owner_intent_id=intent_id,
    )
    point["payload"].pop(missing_field)
    with pytest.raises(qdrant_outbox.QdrantOutboxError):
        qdrant_outbox._prepare_qdrant_upsert_envelope(
            intent_id=intent_id,
            knowledge_base_id="kb-a",
            job_id=None,
            collection_name=collection_name,
            target_points=[point],
            before_points=[],
            strict_schema=True,
        )


def test_v2_prepare_rejects_forged_collection_identity_and_vector_hash():
    from app.services import qdrant_outbox

    intent_id = "strict-v2-forgery"
    collection_name, point = _strict_v2_point(
        "point-a",
        [0.1, 0.9],
        knowledge_base_id="kb-a",
        owner_intent_id=intent_id,
    )
    with pytest.raises(qdrant_outbox.QdrantOutboxError, match="collection name is not canonical"):
        qdrant_outbox._prepare_qdrant_upsert_envelope(
            intent_id=intent_id,
            knowledge_base_id="kb-a",
            job_id=None,
            collection_name=f"{collection_name}-forged",
            target_points=[point],
            before_points=[],
            strict_schema=True,
        )

    point["payload"]["vector_payload_hash"] = "0" * 64
    with pytest.raises(qdrant_outbox.QdrantOutboxError, match="vector payload hash is invalid"):
        qdrant_outbox._prepare_qdrant_upsert_envelope(
            intent_id=intent_id,
            knowledge_base_id="kb-a",
            job_id=None,
            collection_name=collection_name,
            target_points=[point],
            before_points=[],
            strict_schema=True,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "context_hash_protocol_version",
        "context_hash",
        "local_hint_protocol_version",
        "local_hint_hash",
    ],
)
def test_v2_prepare_rejects_contextual_identity_mutation_without_hash_update(field_name):
    from app.services import qdrant_outbox

    intent_id = "strict-v2-contextual-forgery"
    collection_name, point = _strict_v2_point(
        "point-a",
        [0.1, 0.9],
        knowledge_base_id="kb-a",
    )
    point["payload"][field_name] = f"{point['payload'][field_name]}-tampered"

    with pytest.raises(qdrant_outbox.QdrantOutboxError, match="vector payload hash is invalid"):
        qdrant_outbox._prepare_qdrant_upsert_envelope(
            intent_id=intent_id,
            knowledge_base_id="kb-a",
            job_id=None,
            collection_name=collection_name,
            target_points=[point],
            before_points=[],
            strict_schema=True,
        )


def test_v2_reconcile_decodes_historical_vector_hash_but_new_prepare_rejects_it():
    from app.services import qdrant_outbox
    from app.services.context_graph import (
        VECTOR_PAYLOAD_HASH_PROTOCOL_V2,
        vector_payload_hash_v2,
    )
    from app.services.vector_store import canonical_embedding_vector

    intent_id = "historical-vector-hash-v2"
    collection_name, point = _strict_v2_point(
        "point-a",
        [0.1, 0.9],
        knowledge_base_id="kb-a",
        owner_intent_id=intent_id,
    )
    point["vector"] = canonical_embedding_vector(
        point["vector"],
        source="historical v2 test vector",
    )
    point["payload"]["vector_payload_hash_protocol"] = VECTOR_PAYLOAD_HASH_PROTOCOL_V2
    point["payload"]["vector_payload_hash"] = vector_payload_hash_v2(
        vector=point["vector"],
        chunk_id=point["id"],
        embedding_model=point["payload"]["embedding_model"],
        embedding_dimension=point["payload"]["embedding_dimension"],
        embedding_text_version=point["payload"]["embedding_text_version"],
    )
    for field_name in (
        "context_hash_protocol_version",
        "context_hash",
        "local_hint_protocol_version",
        "local_hint_hash",
    ):
        point["payload"].pop(field_name)

    with pytest.raises(qdrant_outbox.QdrantOutboxError, match="invalid vector payload hash protocol"):
        qdrant_outbox._prepare_qdrant_upsert_envelope(
            intent_id=intent_id,
            knowledge_base_id="kb-a",
            job_id=None,
            collection_name=collection_name,
            target_points=[point],
            before_points=[],
            strict_schema=True,
        )

    contract = qdrant_outbox._outbox_protocol_contract(
        qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION
    )
    envelope = {
        "protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
        "envelope_schema_version": contract["envelope_schema_version"],
        "envelope_schema_hash": contract["envelope_schema_hash"],
        "canonical_bytes_version": contract["canonical_bytes_version"],
        "intent_id": intent_id,
        "collection_name": collection_name,
        "target_points": [point],
        "before_points": [],
        "target_payload_hash": qdrant_outbox._outbox_payload_hash(
            qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
            [point],
        ),
        "before_image_hash": qdrant_outbox._outbox_payload_hash(
            qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
            [],
        ),
    }
    decoded, before = qdrant_outbox._validated_reconcile_payload(
        SimpleNamespace(
            id=intent_id,
            knowledge_base_id="kb-a",
            payload_json=envelope,
            target_ids_json=["point-a"],
        )
    )
    assert decoded == [point]
    assert before == []


@pytest.mark.asyncio
async def test_invalid_durable_v2_target_fails_before_lock_read_intent_or_qdrant_write(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from app.services import qdrant_outbox

    class TrackingStore(MemoryVectorStore):
        def __init__(self):
            super().__init__()
            self.get_calls = 0

        def get_points(self, ids):
            self.get_calls += 1
            return super().get_points(ids)

    store = TrackingStore()
    monkeypatch.setattr(qdrant_outbox, "_dialect_name", lambda _db: "postgresql")
    with pytest.raises(qdrant_outbox.QdrantOutboxError, match="missing schema fields"):
        await qdrant_outbox.execute_qdrant_upsert_batches(
            db_session,
            store=store,
            knowledge_base_id=sample_knowledge_base.id,
            job_id=None,
            collection_name="forged-collection",
            points=[_point("invalid-durable-point", 0.1)],
            batch_size=1,
        )

    assert store.get_calls == 0
    assert store.upsert_calls == 0
    assert not db_session.info.get(qdrant_outbox.QDRANT_OUTBOX_TEST_HISTORY_KEY)
    assert not db_session.info.get(qdrant_outbox.QDRANT_OUTBOX_SESSION_KEY)


def test_v2_decoder_revalidates_schema_after_envelope_hash_is_recomputed():
    from app.services import qdrant_outbox

    intent_id = "strict-v2-decoder"
    collection_name, point = _strict_v2_point(
        "point-a",
        [0.1, 0.9],
        knowledge_base_id="kb-a",
        owner_intent_id=intent_id,
    )
    prepared = qdrant_outbox._prepare_qdrant_upsert_envelope(
        intent_id=intent_id,
        knowledge_base_id="kb-a",
        job_id=None,
        collection_name=collection_name,
        target_points=[point],
        before_points=[],
        strict_schema=True,
    )
    contract = qdrant_outbox._outbox_protocol_contract(
        qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION
    )
    payload = {
        "protocol_version": qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
        "intent_id": intent_id,
        "collection_name": collection_name,
        "target_points": list(prepared.target_points),
        "before_points": [],
        "target_payload_hash": prepared.target_payload_hash,
        "before_image_hash": prepared.before_image_hash,
        "envelope_schema_version": contract["envelope_schema_version"],
        "envelope_schema_hash": contract["envelope_schema_hash"],
        "canonical_bytes_version": contract["canonical_bytes_version"],
    }
    row = SimpleNamespace(
        id=intent_id,
        knowledge_base_id="kb-a",
        target_ids_json=["point-a"],
        payload_json=payload,
    )
    decoded, before = qdrant_outbox._validated_reconcile_payload(row)
    assert decoded == list(prepared.target_points)
    assert before == []

    missing_contract = dict(payload)
    missing_contract.pop("envelope_schema_hash")
    with pytest.raises(qdrant_outbox.QdrantOutboxError, match="envelope contract"):
        qdrant_outbox._validated_reconcile_payload(
            SimpleNamespace(
                id=intent_id,
                knowledge_base_id="kb-a",
                target_ids_json=["point-a"],
                payload_json=missing_contract,
            )
        )

    tampered = {
        **payload,
        "target_points": [
            {
                **prepared.target_points[0],
                "payload": {
                    **prepared.target_points[0]["payload"],
                    "collection_identity_digest": "0" * 64,
                },
            }
        ],
    }
    tampered["target_payload_hash"] = qdrant_outbox._outbox_payload_hash(
        qdrant_outbox.QDRANT_OUTBOX_PROTOCOL_VERSION,
        tampered["target_points"],
    )
    with pytest.raises(qdrant_outbox.QdrantOutboxError, match="identity digest is invalid"):
        qdrant_outbox._validated_reconcile_payload(
            SimpleNamespace(
                id=intent_id,
                knowledge_base_id="kb-a",
                target_ids_json=["point-a"],
                payload_json=tampered,
            )
        )


@pytest.mark.asyncio
async def test_v2_float64_is_canonicalized_once_before_intent_handle_and_qdrant(
    db_session,
    sample_knowledge_base,
):
    from app.services import qdrant_outbox

    collection_name, source = _strict_v2_point(
        "canonical-point",
        [0.1, 0.9],
        knowledge_base_id=sample_knowledge_base.id,
    )
    original = {
        "id": source["id"],
        "vector": list(source["vector"]),
        "payload": {
            **source["payload"],
            "nested_metadata": {"labels": list(source["payload"]["nested_metadata"]["labels"])},
        },
    }
    store = MemoryVectorStore()
    result = await qdrant_outbox.execute_qdrant_upsert_batches(
        db_session,
        store=store,
        knowledge_base_id=sample_knowledge_base.id,
        job_id=None,
        collection_name=collection_name,
        points=[source],
        batch_size=1,
    )

    canonical_vector = [
        struct.unpack(">f", struct.pack(">f", value))[0]
        for value in original["vector"]
    ]
    pending = db_session.info[qdrant_outbox.QDRANT_OUTBOX_TEST_HISTORY_KEY][0]
    handle = db_session.info[qdrant_outbox.QDRANT_OUTBOX_SESSION_KEY][0]
    persisted_target = pending["payload"]["target_points"][0]
    qdrant_target = store.points["canonical-point"]
    assert persisted_target["vector"] == canonical_vector
    assert handle.target_points[0] == persisted_target == qdrant_target
    assert handle.target_points[0] is persisted_target
    assert handle.payload_hash == pending["payload"]["target_payload_hash"]
    assert result["point_intent_ids"]["canonical-point"] == handle.id
    assert source == original

    source["vector"][0] = 0.7
    source["payload"]["nested_metadata"]["labels"].append("mutated")
    assert persisted_target["vector"] == canonical_vector
    assert persisted_target["payload"]["nested_metadata"]["labels"] == ["one", "two"]


def test_reconcile_active_upserts_use_bounded_pk_keyset_pages_and_action_samples(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    import inspect

    from app.models import IngestionCompensationLog
    from app.services import qdrant_outbox

    monkeypatch.setattr(qdrant_outbox, "QDRANT_OUTBOX_RECONCILE_PAGE_SIZE", 2)
    monkeypatch.setattr(
        qdrant_outbox,
        "QDRANT_OUTBOX_RECONCILE_ACTION_SAMPLE_LIMIT",
        2,
    )
    rows = [
        IngestionCompensationLog(
            id=f"00000000-0000-0000-0000-{index:012d}",
            knowledge_base_id=sample_knowledge_base.id,
            operation=qdrant_outbox.QDRANT_UPSERT_OPERATION,
            target_ids_json=[f"point-{index}"],
            payload_json={
                "protocol_version": "unsupported-test-protocol",
                "intent_id": f"00000000-0000-0000-0000-{index:012d}",
            },
            status="pending",
        )
        for index in range(1, 6)
    ]
    db_session.add_all(rows)
    db_session.commit()

    high_water_id, candidate_count = qdrant_outbox._active_upsert_intent_scan_summary(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
    )
    yielded = list(
        qdrant_outbox._iter_active_upsert_intents_keyset(
            db_session,
            knowledge_base_id=sample_knowledge_base.id,
            high_water_id=high_water_id,
        )
    )
    assert candidate_count == 5
    assert [item[1].id for item in yielded] == [row.id for row in rows]
    assert [item[2] for item in yielded] == [0, 0, 1, 1, 2]
    assert len({id(item[0]) for item in yielded}) == 3
    assert all(not item[0].in_transaction() for item in yielded)
    assert all(not item[0].identity_map for item in yielded)
    assert ".all()" not in inspect.getsource(
        qdrant_outbox._iter_active_upsert_intents_keyset
    )

    stats = qdrant_outbox.reconcile_qdrant_outbox_sync(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        knowledge_base_name=sample_knowledge_base.name,
        dry_run=True,
        include_unexpired=True,
    )
    assert (
        stats["scan_protocol_version"]
        == qdrant_outbox.QDRANT_OUTBOX_RECONCILE_SCAN_PROTOCOL_VERSION
    )
    assert stats["candidate_upsert_intents"] == 5
    assert stats["checked_upsert_intents"] == 5
    assert stats["pages_scanned"] == 3
    assert stats["failed"] == 5
    assert stats["action_count"] == 5
    assert len(stats["actions"]) == 2
    assert stats["actions_truncated_count"] == 3


def test_reconcile_keyset_excludes_intents_above_frozen_high_water(
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionCompensationLog
    from app.services import qdrant_outbox

    def intent(index: int) -> IngestionCompensationLog:
        intent_id = f"10000000-0000-0000-0000-{index:012d}"
        return IngestionCompensationLog(
            id=intent_id,
            knowledge_base_id=sample_knowledge_base.id,
            operation=qdrant_outbox.QDRANT_UPSERT_OPERATION,
            target_ids_json=[f"point-{index}"],
            payload_json={"protocol_version": "test", "intent_id": intent_id},
            status="pending",
        )

    initial = [intent(1), intent(2)]
    db_session.add_all(initial)
    db_session.commit()
    high_water_id, candidate_count = qdrant_outbox._active_upsert_intent_scan_summary(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
    )

    db_session.add(intent(9))
    db_session.commit()
    yielded = list(
        qdrant_outbox._iter_active_upsert_intents_keyset(
            db_session,
            knowledge_base_id=sample_knowledge_base.id,
            high_water_id=high_water_id,
        )
    )

    assert candidate_count == 2
    assert high_water_id == initial[-1].id
    assert [item[1].id for item in yielded] == [row.id for row in initial]


def test_reconcile_store_cache_closes_each_client_once_and_releases_references():
    from app.services import qdrant_outbox

    close_calls: list[str] = []
    client = SimpleNamespace(close=lambda: close_calls.append("closed"))
    stores = {
        "first": SimpleNamespace(client=client),
        "second": SimpleNamespace(client=client),
    }

    qdrant_outbox._close_reconcile_stores(stores)

    assert close_calls == ["closed"]
    assert stores == {}


@pytest.mark.asyncio
async def test_outbox_writer_splits_oversized_batch_at_frozen_intent_bound(
    db_session,
    sample_knowledge_base,
    monkeypatch,
):
    from app.services import qdrant_outbox

    monkeypatch.setattr(qdrant_outbox, "QDRANT_OUTBOX_MAX_TARGET_POINTS_PER_INTENT", 2)
    store = MemoryVectorStore()
    result = await qdrant_outbox.execute_qdrant_upsert_batches(
        db_session,
        store=store,
        knowledge_base_id=sample_knowledge_base.id,
        job_id=None,
        collection_name="non-durable-test-adapter",
        points=[_point(f"bounded-{index}", 0.1 * index) for index in range(1, 6)],
        batch_size=100,
    )

    assert result["batches"] == 3
    pending = [
        item
        for item in db_session.info[qdrant_outbox.QDRANT_OUTBOX_TEST_HISTORY_KEY]
        if item["status"] == "pending"
    ]
    assert [len(item["payload"]["target_points"]) for item in pending] == [2, 2, 1]

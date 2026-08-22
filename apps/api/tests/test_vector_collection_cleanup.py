from __future__ import annotations

import pytest
from sqlalchemy import func, select


def _chunk(db_session, knowledge_base_id: str):
    from app.models import Chunk, Document, DocumentVersion
    from app.services.chunking import text_hash

    document = Document(
        knowledge_base_id=knowledge_base_id,
        title="Obsolete vector collection",
        source_path="obsolete-vector.md",
        source_type="markdown",
        checksum="a" * 64,
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=document.checksum,
        storage_path=document.source_path,
        is_active=True,
    )
    db_session.add(version)
    db_session.flush()
    text = "obsolete vector derived bytes"
    chunk = Chunk(
        knowledge_base_id=knowledge_base_id,
        document_id=document.id,
        document_version_id=version.id,
        chunk_version=1,
        chunk_index=0,
        text=text,
        text_hash=text_hash(text),
        state="active",
    )
    db_session.add(chunk)
    db_session.flush()
    return chunk


class _CleanupStore:
    def __init__(self, *, exists: bool = True) -> None:
        self.exists = exists
        self.delete_calls: list[str] = []

    def collection_exists(self) -> bool:
        return self.exists

    def delete_collection_exact(self, collection_name: str):
        self.delete_calls.append(collection_name)
        existed = self.exists
        self.exists = False
        return {
            "collection_name": collection_name,
            "existed_before": existed,
            "deleted": existed,
            "verified_absent": True,
            "outcome": "applied_and_verified" if existed else "already_absent",
        }


@pytest.mark.parametrize(
    ("vector_status", "authoritative"),
    [
        ("ready", True),
        ("shadow_ready", True),
        ("rollback_retained", True),
        ("rolled_back_retained", True),
        ("stale", False),
    ],
)
def test_qdrant_outbox_retains_every_shadow_lifecycle_authoritative_status(
    db_session,
    sample_knowledge_base,
    vector_status: str,
    authoritative: bool,
) -> None:
    from app.models import IngestionCompensationLog, VectorRecord
    from app.services.qdrant_outbox import (
        QDRANT_UPSERT_OPERATION,
        qdrant_intent_has_committed_records,
    )

    chunk = _chunk(db_session, sample_knowledge_base.id)
    collection_name = f"outbox_status_{vector_status}"
    intent = IngestionCompensationLog(
        knowledge_base_id=sample_knowledge_base.id,
        operation=QDRANT_UPSERT_OPERATION,
        target_ids_json=[chunk.id],
        payload_json={"collection_name": collection_name},
        status="committed",
    )
    db_session.add(intent)
    db_session.flush()
    db_session.add(
        VectorRecord(
            knowledge_base_id=sample_knowledge_base.id,
            chunk_id=chunk.id,
            qdrant_point_id=chunk.id,
            collection_name=collection_name,
            embedding_model="outbox-status-model",
            embedding_dimension=8,
            embedding_text_version="outbox-status-text-v1",
            chunk_schema_version="chunk_schema_v1",
            payload_hash="c" * 64,
            vector_status=vector_status,
            diagnostics_json={"qdrant_write_intent_id": intent.id},
        )
    )
    db_session.flush()

    assert qdrant_intent_has_committed_records(db_session, intent) is authoritative


def test_collection_cleanup_dry_run_blocks_active_pointer_without_writes(
    db_session,
    sample_knowledge_base,
) -> None:
    from app.models import IngestionCompensationLog
    from app.services.vector_collection_cleanup import vector_collection_cleanup_plan
    from app.services.vector_shadow_lifecycle import ensure_active_vector_runtime_target

    target = ensure_active_vector_runtime_target(db_session, sample_knowledge_base.id)
    db_session.commit()
    store = _CleanupStore()
    before = db_session.scalar(select(func.count(IngestionCompensationLog.id)))

    plan = vector_collection_cleanup_plan(
        db_session,
        collection_name=target.schema.collection_name,
        vector_store=store,
    )

    assert plan["dry_run"] is True
    assert plan["allowed"] is False
    assert "active_vector_runtime_pointer_references_collection" in plan["blockers"]
    assert plan["deletion_inferred"] is False
    assert store.delete_calls == []
    assert db_session.scalar(select(func.count(IngestionCompensationLog.id))) == before


@pytest.mark.parametrize(
    ("vector_status", "allowed"),
    [
        ("ready", False),
        ("shadow_ready", True),
        ("rollback_retained", True),
        ("rolled_back_retained", True),
        ("stale", True),
    ],
)
def test_exact_cleanup_releases_retention_only_after_explicit_operator_gate(
    db_session,
    sample_knowledge_base,
    vector_status: str,
    allowed: bool,
) -> None:
    from app.models import VectorRecord
    from app.services.vector_collection_cleanup import vector_collection_cleanup_plan

    chunk = _chunk(db_session, sample_knowledge_base.id)
    collection_name = f"cleanup_status_{vector_status}"
    db_session.add(
        VectorRecord(
            knowledge_base_id=sample_knowledge_base.id,
            chunk_id=chunk.id,
            qdrant_point_id=chunk.id,
            collection_name=collection_name,
            embedding_model="cleanup-status-model",
            embedding_dimension=8,
            embedding_text_version="cleanup-status-text-v1",
            chunk_schema_version="chunk_schema_v1",
            payload_hash="d" * 64,
            vector_status=vector_status,
            diagnostics_json={},
        )
    )
    db_session.commit()

    plan = vector_collection_cleanup_plan(
        db_session,
        collection_name=collection_name,
        check_qdrant=False,
    )

    assert plan["allowed"] is allowed
    assert plan["protected_record_count"] == int(
        vector_status == "ready"
    )
    assert plan["releasable_retained_record_count"] == int(
        vector_status
        in {"shadow_ready", "rollback_retained", "rolled_back_retained"}
    )
    assert (
        "ready_vector_records_reference_collection" in plan["blockers"]
    ) is (not allowed)


def test_exact_collection_cleanup_commits_intent_then_is_restart_safe(
    db_session,
    sample_knowledge_base,
) -> None:
    from app.models import IngestionCompensationLog, VectorRecord
    from app.services.vector_collection_cleanup import (
        execute_vector_collection_cleanup,
        prepare_vector_collection_cleanup,
        vector_collection_cleanup_plan,
    )

    chunk = _chunk(db_session, sample_knowledge_base.id)
    collection_name = "obsolete_vector_collection_v1"
    record = VectorRecord(
        knowledge_base_id=sample_knowledge_base.id,
        chunk_id=chunk.id,
        qdrant_point_id=chunk.id,
        collection_name=collection_name,
        embedding_model="legacy-model",
        embedding_dimension=8,
        embedding_text_version="legacy-text-v1",
        chunk_schema_version="chunk_schema_v1",
        payload_hash="b" * 64,
        vector_status="rollback_retained",
        diagnostics_json={"legacy": True},
    )
    db_session.add(record)
    db_session.commit()
    store = _CleanupStore()
    plan = vector_collection_cleanup_plan(
        db_session,
        collection_name=collection_name,
        vector_store=store,
    )
    assert plan["allowed"] is True
    assert plan["record_status_counts"] == {"rollback_retained": 1}
    assert plan["authoritative_record_count"] == 1
    assert plan["protected_record_count"] == 0
    assert plan["releasable_retained_record_count"] == 1

    with pytest.raises(ValueError, match="exactly repeat"):
        prepare_vector_collection_cleanup(
            db_session,
            audit_knowledge_base_id=sample_knowledge_base.id,
            collection_name=collection_name,
            confirmed_collection_name="another_collection",
            allow_sqlite_test_adapter=True,
        )
    intent = prepare_vector_collection_cleanup(
        db_session,
        audit_knowledge_base_id=sample_knowledge_base.id,
        collection_name=collection_name,
        confirmed_collection_name=collection_name,
        allow_sqlite_test_adapter=True,
    )
    db_session.commit()
    intent_id = intent.id
    assert intent.status == "collection_delete_pending"

    result = execute_vector_collection_cleanup(
        db_session,
        intent_id=intent_id,
        vector_store=store,
        allow_sqlite_test_adapter=True,
    )
    db_session.commit()
    db_session.refresh(record)
    assert result["committed"] is True
    assert result["qdrant_result"]["verified_absent"] is True
    assert store.delete_calls == [collection_name]
    assert record.vector_status == "missing"
    committed_intent = db_session.get(IngestionCompensationLog, intent_id)
    assert committed_intent.status == "committed"
    assert (committed_intent.payload_json or {})["phase"] == "completed"

    replay = execute_vector_collection_cleanup(
        db_session,
        intent_id=intent_id,
        vector_store=store,
        allow_sqlite_test_adapter=True,
    )
    assert replay["idempotent_replay"] is True
    assert store.delete_calls == [collection_name]


def test_pending_exact_cleanup_blocks_new_candidate_for_same_collection(
    db_session,
    sample_knowledge_base,
) -> None:
    from app.services.vector_collection_cleanup import prepare_vector_collection_cleanup
    from app.services.vector_shadow_lifecycle import (
        frozen_vector_schema,
        stage_vector_runtime_candidate,
    )

    _chunk(db_session, sample_knowledge_base.id)
    schema = frozen_vector_schema(
        embedding_model="blocked-candidate-model",
        embedding_dimension=8,
    )
    prepare_vector_collection_cleanup(
        db_session,
        audit_knowledge_base_id=sample_knowledge_base.id,
        collection_name=schema.collection_name,
        confirmed_collection_name=schema.collection_name,
        allow_sqlite_test_adapter=True,
    )
    db_session.commit()

    with pytest.raises(RuntimeError, match="destructive cleanup intent"):
        stage_vector_runtime_candidate(
            db_session,
            knowledge_base_ids=[sample_knowledge_base.id],
            embedding_model=schema.embedding_model,
            embedding_dimension=schema.embedding_dimension,
            source="unit_test_cleanup_fence",
        )


def test_cleanup_retry_finalizes_record_only_drift_after_verified_external_apply(
    db_session,
    sample_knowledge_base,
) -> None:
    from app.models import VectorRecord
    from app.services.vector_collection_cleanup import (
        execute_vector_collection_cleanup,
        prepare_vector_collection_cleanup,
    )

    chunk = _chunk(db_session, sample_knowledge_base.id)
    collection_name = "externally_applied_cleanup_retry"
    record = VectorRecord(
        knowledge_base_id=sample_knowledge_base.id,
        chunk_id=chunk.id,
        qdrant_point_id=chunk.id,
        collection_name=collection_name,
        embedding_model="retained-retry-model",
        embedding_dimension=8,
        embedding_text_version="retained-retry-text-v1",
        chunk_schema_version="chunk_schema_v1",
        payload_hash="e" * 64,
        vector_status="rollback_retained",
        diagnostics_json={},
    )
    db_session.add(record)
    db_session.commit()
    intent = prepare_vector_collection_cleanup(
        db_session,
        audit_knowledge_base_id=sample_knowledge_base.id,
        collection_name=collection_name,
        confirmed_collection_name=collection_name,
        allow_sqlite_test_adapter=True,
    )
    db_session.commit()

    # Model the crash window: Qdrant applied the exact delete, then a read-only
    # reconcile changed only the retained record status before finalization.
    store = _CleanupStore(exists=False)
    record.vector_status = "stale"
    db_session.commit()
    result = execute_vector_collection_cleanup(
        db_session,
        intent_id=intent.id,
        vector_store=store,
        allow_sqlite_test_adapter=True,
    )
    db_session.commit()

    db_session.refresh(record)
    assert result["recovered_after_external_apply"] is True
    assert result["qdrant_result"]["outcome"] == "already_absent"
    assert record.vector_status == "missing"


def test_vector_store_exact_delete_verifies_uncertain_response_and_rejects_drift() -> None:
    from app.services.vector_store import VectorStore

    class Client:
        def __init__(self, *, remove_before_error: bool) -> None:
            self.exists = True
            self.remove_before_error = remove_before_error
            self.calls: list[str] = []

        def collection_exists(self, *, collection_name: str) -> bool:
            return self.exists

        def delete_collection(self, *, collection_name: str) -> None:
            self.calls.append(collection_name)
            if self.remove_before_error:
                self.exists = False
            raise TimeoutError("uncertain response")

    applied_client = Client(remove_before_error=True)
    applied_store = object.__new__(VectorStore)
    applied_store.collection = "old_collection"
    applied_store.client = applied_client
    result = applied_store.delete_collection_exact("old_collection")
    assert result["outcome"] == "applied_response_uncertain_absence_verified"
    assert result["verified_absent"] is True

    retained_client = Client(remove_before_error=False)
    retained_store = object.__new__(VectorStore)
    retained_store.collection = "old_collection"
    retained_store.client = retained_client
    with pytest.raises(TimeoutError, match="uncertain response"):
        retained_store.delete_collection_exact("old_collection")
    assert retained_client.exists is True

    with pytest.raises(ValueError, match="does not match"):
        applied_store.delete_collection_exact("different_collection")
    assert applied_client.calls == ["old_collection"]

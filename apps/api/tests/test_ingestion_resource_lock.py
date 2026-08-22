from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest


def test_ingestion_advisory_lock_key_is_stable_and_resource_scoped():
    from app.services.ingestion_resource_lock import advisory_lock_key, knowledge_base_resource_key

    first_resource = knowledge_base_resource_key("kb-1")
    second_resource = knowledge_base_resource_key("kb-2")
    assert first_resource == "knowledge_base:kb-1"
    assert advisory_lock_key(first_resource) == advisory_lock_key(first_resource)
    assert advisory_lock_key(first_resource) != advisory_lock_key(second_resource)


@pytest.mark.asyncio
async def test_sqlite_lock_adapter_fails_closed_outside_pytest(monkeypatch, db_session, sample_knowledge_base):
    from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    with pytest.raises(RuntimeError, match="production ingest/rebuild correctness requires PostgreSQL"):
        async with knowledge_base_ingestion_resource_lock(
            db_session,
            sample_knowledge_base.id,
            operation="non_test_sqlite",
        ):
            raise AssertionError("SQLite must not become the production correctness backend")


@pytest.mark.asyncio
async def test_sqlite_test_adapter_reports_contention_and_releases_after_cancellation(db_session, sample_knowledge_base):
    from app.services.ingestion_resource_lock import (
        IngestionResourceBusyError,
        knowledge_base_ingestion_resource_lock,
    )

    acquired = asyncio.Event()

    async def hold_until_cancelled():
        async with knowledge_base_ingestion_resource_lock(
            db_session,
            sample_knowledge_base.id,
            operation="cancelled_holder",
            timeout_seconds=0.2,
        ) as lease:
            assert lease.backend == "sqlite_test_adapter"
            acquired.set()
            await asyncio.Future()

    holder = asyncio.create_task(hold_until_cancelled())
    await acquired.wait()

    async def contend():
        async with knowledge_base_ingestion_resource_lock(
            db_session,
            sample_knowledge_base.id,
            operation="contender",
            timeout_seconds=0.02,
        ):
            raise AssertionError("contender must not enter while the resource is held")

    with pytest.raises(IngestionResourceBusyError) as busy:
        await asyncio.create_task(contend())
    assert busy.value.diagnostics["reason"] == "resource_lock_timeout"
    assert busy.value.diagnostics["resource_key"] == f"knowledge_base:{sample_knowledge_base.id}"
    assert busy.value.diagnostics["backend"] == "sqlite_test_adapter"
    assert busy.value.diagnostics["retryable"] is True
    assert busy.value.diagnostics["holders"]

    holder.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holder

    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation="after_cancel",
        timeout_seconds=0.05,
    ) as reacquired:
        assert reacquired.backend == "sqlite_test_adapter"


@pytest.mark.asyncio
async def test_knowledge_base_delete_recovery_requires_exact_qdrant_owner_set(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.services import qdrant_outbox
    from app.services.ingestion_resource_lock import (
        IngestionResourceBusyError,
        knowledge_base_delete_recovery_owner_token,
        knowledge_base_ingestion_resource_lock,
    )

    tombstone = SimpleNamespace(
        id="delete-intent-1",
        operation=qdrant_outbox.QDRANT_DELETE_OPERATION,
        status="external_applied",
    )

    def pending(_db, *, knowledge_base_id=None):
        assert knowledge_base_id == sample_knowledge_base.id
        return [tombstone]

    def diagnostics(row):
        assert row is tombstone
        return {
            "delete_source_operation": "delete_knowledge_base_data",
            "delete_protocol_version": qdrant_outbox.QDRANT_DELETE_PROTOCOL_VERSION,
            "collection_name": "unit-collection",
            "target_count": 1,
            "retryable": True,
            "delete_intent_validation_error": None,
            "reason": "knowledge_base_delete_pending",
            "retry_guidance": "finish_or_retry_knowledge_base_deletion",
        }

    monkeypatch.setattr(qdrant_outbox, "pending_qdrant_delete_intents", pending)
    monkeypatch.setattr(
        qdrant_outbox,
        "qdrant_delete_intent_recovery_diagnostics",
        diagnostics,
    )
    owner = knowledge_base_delete_recovery_owner_token(
        db_session,
        sample_knowledge_base.id,
    )
    assert owner and owner.startswith("qdrant-delete:")

    with pytest.raises(IngestionResourceBusyError) as wrong_owner:
        async with knowledge_base_ingestion_resource_lock(
            db_session,
            sample_knowledge_base.id,
            operation="delete_knowledge_base_data",
            batch_id="qdrant-delete:wrong",
        ):
            raise AssertionError("a stale KB-delete owner token must not enter")
    assert wrong_owner.value.diagnostics["requested_owner_matches"] is False

    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation="delete_knowledge_base_data",
        batch_id=owner,
    ) as lease:
        assert lease.batch_id == owner
        assert lease.operation == "delete_knowledge_base_data"


@pytest.mark.asyncio
async def test_uploaded_batch_computes_target_version_inside_resource_lock(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.services import ingestion

    batch = ingestion.create_uploaded_files_batch(db_session, sample_knowledge_base.id, [])
    original_target_chunk_version = ingestion.target_chunk_version
    observed = {"called": False}

    def guarded_target_chunk_version(*, current_version: int, active_max_version: int, full_reparse: bool) -> int:
        lease = ingestion.active_ingestion_resource_lease(sample_knowledge_base.id)
        assert lease is not None
        assert lease.operation == "uploaded_files_ingestion"
        observed["called"] = True
        return original_target_chunk_version(
            current_version=current_version,
            active_max_version=active_max_version,
            full_reparse=full_reparse,
        )

    monkeypatch.setattr(ingestion, "target_chunk_version", guarded_target_chunk_version)
    result = await ingestion.run_uploaded_files_ingestion(batch.id, [], execution_mode="inline")

    assert observed["called"] is True
    assert result["state"] == "completed"
    db_session.expire_all()
    persisted = db_session.get(ingestion.IngestionBatch, batch.id)
    lock_diagnostics = (persisted.stats or {})["ingestion_resource_lock"]
    assert lock_diagnostics["resource_scope"] == "knowledge_base"
    assert lock_diagnostics["operation"] == "uploaded_files_ingestion"
    assert lock_diagnostics["acquired"] is True


@pytest.mark.asyncio
async def test_uploaded_batch_bootstraps_per_kb_roots_inside_resource_lock(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.services import ingestion

    batch = ingestion.create_uploaded_files_batch(
        db_session,
        sample_knowledge_base.id,
        [],
    )
    observed: list[dict[str, object]] = []

    def record_storage_gate(
        knowledge_base_name=None,
        *,
        knowledge_base_id=None,
        knowledge_base_source_root=None,
        create_missing=False,
    ):
        lease = ingestion.active_ingestion_resource_lease(
            sample_knowledge_base.id
        )
        observed.append(
            {
                "knowledge_base_name": knowledge_base_name,
                "knowledge_base_id": knowledge_base_id,
                "knowledge_base_source_root": knowledge_base_source_root,
                "create_missing": create_missing,
                "lease_operation": lease.operation if lease else None,
            }
        )
        return {"capabilities": []}

    monkeypatch.setattr(
        ingestion,
        "ensure_knowledge_base_storage_durability_ready",
        record_storage_gate,
    )

    result = await ingestion.run_uploaded_files_ingestion(
        batch.id,
        [],
        execution_mode="inline",
    )

    assert result["state"] == "completed"
    assert observed == [
        {
            "knowledge_base_name": sample_knowledge_base.name,
            "knowledge_base_id": None,
            "knowledge_base_source_root": sample_knowledge_base.source_root,
            "create_missing": True,
            "lease_operation": "uploaded_files_ingestion",
        }
    ]


@pytest.mark.asyncio
async def test_active_batch_recovery_is_a_durable_same_kb_fence_with_exact_owner_reentry(
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionBatchRecovery
    from app.services import ingestion
    from app.services.ingestion_resource_lock import (
        INGESTION_BATCH_RECOVERY_LOCK_OPERATION,
        IngestionResourceBusyError,
        ingestion_batch_recovery_owner_token,
        knowledge_base_ingestion_resource_lock,
    )

    batch = ingestion.create_uploaded_files_batch(
        db_session,
        sample_knowledge_base.id,
        [],
    )
    recovery = ingestion._prepare_batch_recovery(
        db_session,
        batch=batch,
        knowledge_base=sample_knowledge_base,
        target_version=0,
        full_reparse=False,
    )
    exact_owner = ingestion_batch_recovery_owner_token(recovery)

    with pytest.raises(IngestionResourceBusyError) as blocked:
        async with knowledge_base_ingestion_resource_lock(
            db_session,
            sample_knowledge_base.id,
            operation="context_graph_rebuild_batch",
            batch_id="unrelated-batch",
        ):
            raise AssertionError("ordinary same-KB mutation must remain fenced")
    assert blocked.value.diagnostics["reason"] == "active_ingestion_batch_recovery_fence"
    assert blocked.value.diagnostics["active_recovery_count"] == 1

    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation="uploaded_files_ingestion",
        batch_id=batch.id,
    ) as same_batch:
        assert same_batch.batch_id == batch.id

    with pytest.raises(IngestionResourceBusyError):
        async with knowledge_base_ingestion_resource_lock(
            db_session,
            sample_knowledge_base.id,
            operation=INGESTION_BATCH_RECOVERY_LOCK_OPERATION,
            batch_id="ingestion-recovery:wrong",
        ):
            raise AssertionError("wrong recovery owner token must not enter")

    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation=INGESTION_BATCH_RECOVERY_LOCK_OPERATION,
        batch_id=exact_owner,
    ) as exact:
        assert exact.batch_id == exact_owner

    recovery = db_session.get(IngestionBatchRecovery, recovery.id)
    recovery.status = "manual_review"
    db_session.commit()
    with pytest.raises(IngestionResourceBusyError) as manual_block:
        async with knowledge_base_ingestion_resource_lock(
            db_session,
            sample_knowledge_base.id,
            operation="uploaded_files_ingestion",
            batch_id=batch.id,
        ):
            raise AssertionError("manual review must fence even the original ingest batch")
    assert manual_block.value.diagnostics["manual_review_required"] is True

    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation=INGESTION_BATCH_RECOVERY_LOCK_OPERATION,
        batch_id=exact_owner,
    ):
        pass

    recovery = db_session.get(IngestionBatchRecovery, recovery.id)
    recovery.status = "parse_compensated"
    recovery.completed_at = ingestion.datetime.utcnow()
    db_session.commit()
    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation="context_graph_rebuild_batch",
        batch_id="after-terminal-recovery",
    ):
        pass


@pytest.mark.asyncio
async def test_same_kb_uploaded_ingestion_and_graph_rebuild_do_not_overlap(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.services import ingestion

    ingest_batch = ingestion.create_uploaded_files_batch(db_session, sample_knowledge_base.id, [])
    rebuild_batch = ingestion.create_context_graph_rebuild_batch(db_session, sample_knowledge_base.id)
    ingest_entered = asyncio.Event()
    release_ingest = asyncio.Event()
    observed_order: list[str] = []
    active_count = 0
    max_active_count = 0

    async def fake_ingest_locked(*args, **kwargs):
        nonlocal active_count, max_active_count
        assert ingestion.active_ingestion_resource_lease(sample_knowledge_base.id) is not None
        active_count += 1
        max_active_count = max(max_active_count, active_count)
        observed_order.append("ingest_enter")
        ingest_entered.set()
        await release_ingest.wait()
        observed_order.append("ingest_exit")
        active_count -= 1
        return {"batch_id": ingest_batch.id, "state": "completed"}

    async def fake_rebuild_locked(*args, **kwargs):
        nonlocal active_count, max_active_count
        assert ingestion.active_ingestion_resource_lease(sample_knowledge_base.id) is not None
        active_count += 1
        max_active_count = max(max_active_count, active_count)
        observed_order.append("rebuild_enter")
        await asyncio.sleep(0)
        observed_order.append("rebuild_exit")
        active_count -= 1
        return {"batch_id": rebuild_batch.id, "state": "completed"}

    monkeypatch.setattr(ingestion, "_run_uploaded_files_ingestion_locked", fake_ingest_locked)
    monkeypatch.setattr(ingestion, "_run_context_graph_rebuild_batch_locked", fake_rebuild_locked)

    ingest_task = asyncio.create_task(ingestion.run_uploaded_files_ingestion(ingest_batch.id, []))
    await ingest_entered.wait()
    rebuild_task = asyncio.create_task(ingestion.run_context_graph_rebuild_batch(rebuild_batch.id))
    await asyncio.sleep(0.03)
    assert observed_order == ["ingest_enter"]
    db_session.expire_all()
    waiting_rebuild = db_session.get(ingestion.IngestionBatch, rebuild_batch.id)
    waiting_diagnostics = (waiting_rebuild.stats or {})["ingestion_resource_lock"]
    assert (waiting_rebuild.stats or {})["phase"] == "waiting_resource_lock"
    assert waiting_diagnostics["status"] == "waiting"
    assert waiting_diagnostics["acquired"] is False
    assert waiting_diagnostics["retry_guidance"] == "wait_for_active_ingest_or_graph_rebuild"

    release_ingest.set()
    await asyncio.gather(ingest_task, rebuild_task)
    assert observed_order == ["ingest_enter", "ingest_exit", "rebuild_enter", "rebuild_exit"]
    assert max_active_count == 1


@pytest.mark.asyncio
async def test_postgres_advisory_lock_serializes_independent_connections_and_releases():
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.orm import Session

    from app.services.ingestion_resource_lock import (
        INGESTION_RESOURCE_LOCK_PROTOCOL_VERSION,
        IngestionResourceBusyError,
        knowledge_base_ingestion_resource_lock,
    )

    database_url = os.getenv("DATABASE_URL", "")
    if not database_url.startswith("postgresql"):
        pytest.skip("PostgreSQL DATABASE_URL is required for the cross-process advisory-lock integration test")
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        engine.dispose()
        pytest.skip(f"PostgreSQL is unavailable for advisory-lock integration testing: {exc.__class__.__name__}")

    first_session = Session(bind=engine)
    second_session = Session(bind=engine)
    knowledge_base_id = f"lock-integration-{uuid4()}"
    try:
        async with knowledge_base_ingestion_resource_lock(
            first_session,
            knowledge_base_id,
            operation="postgres_holder",
            timeout_seconds=0.5,
            poll_seconds=0.01,
        ) as holder:
            assert holder.backend == "postgresql"
            assert holder.protocol_version == INGESTION_RESOURCE_LOCK_PROTOCOL_VERSION

            async def postgres_contender():
                async with knowledge_base_ingestion_resource_lock(
                    second_session,
                    knowledge_base_id,
                    operation="postgres_contender",
                    timeout_seconds=0.05,
                    poll_seconds=0.01,
                ):
                    raise AssertionError("independent connection must not acquire an already-held advisory lock")

            with pytest.raises(IngestionResourceBusyError) as busy:
                await asyncio.create_task(postgres_contender())
            assert busy.value.diagnostics["backend"] == "postgresql"
            assert busy.value.diagnostics["contention_count"] >= 1
            assert busy.value.diagnostics["holders"]

        async with knowledge_base_ingestion_resource_lock(
            second_session,
            knowledge_base_id,
            operation="postgres_after_release",
            timeout_seconds=0.2,
            poll_seconds=0.01,
        ) as reacquired:
            assert reacquired.backend == "postgresql"
    finally:
        first_session.close()
        second_session.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_postgres_durable_recovery_fence_and_advisory_lock_across_connections():
    import hashlib
    import json

    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.orm import Session

    from app.models import IngestionBatch, IngestionBatchRecovery, KnowledgeBase
    from app.services.ingestion_resource_lock import (
        INGESTION_BATCH_RECOVERY_LOCK_OPERATION,
        IngestionResourceBusyError,
        ingestion_batch_recovery_owner_token,
        knowledge_base_ingestion_resource_lock,
    )

    database_url = os.getenv("DATABASE_URL", "")
    if not database_url.startswith("postgresql"):
        pytest.skip("PostgreSQL DATABASE_URL is required for the durable fence integration test")
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            table_present = bool(
                connection.execute(
                    text("SELECT to_regclass('public.ingestion_batch_recoveries') IS NOT NULL")
                ).scalar_one()
            )
        if not table_present:
            pytest.skip("Ingestion fence migration 0031 is not installed")
    except SQLAlchemyError as exc:
        engine.dispose()
        pytest.skip(f"PostgreSQL is unavailable: {exc.__class__.__name__}")

    def canonical_hash(payload):
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    setup = Session(bind=engine)
    first = Session(bind=engine)
    second = Session(bind=engine)
    knowledge_base_id = str(uuid4())
    try:
        knowledge_base = KnowledgeBase(
            id=knowledge_base_id,
            name=f"ingestion_fence-fence-{uuid4()}",
            source_root=f"ingestion_fence-postgres-fence-{uuid4()}",
            current_chunk_version=0,
        )
        batch = IngestionBatch(
            id=str(uuid4()),
            knowledge_base_id=knowledge_base_id,
            source_root="ingestion_fence-postgres-fence",
            status="failed",
        )
        before_state = {
            "protocol_version": "ingestion_batch_before_scope_v1",
            "knowledge_base_id": knowledge_base_id,
            "v_before_batch": 0,
            "active_chunk_ids": [],
            "active_chunk_scope_hash": canonical_hash([]),
            "active_document_version_ids": [],
            "chunk_version_descriptors": [],
        }
        graph_before = {
            "protocol_version": "ingestion_graph_before_scope_v1",
            "knowledge_base_id": knowledge_base_id,
            "states": {},
            "chunk_version_descriptors": [],
            "vector_runtime_pointer": None,
        }
        recovery = IngestionBatchRecovery(
            id=str(uuid4()),
            batch_id=batch.id,
            knowledge_base_id=knowledge_base_id,
            status="prepared",
            v_before_batch=0,
            target_version=0,
            parse_committed=False,
            before_state_json=before_state,
            before_state_hash=canonical_hash(before_state),
            graph_before_state_json=graph_before,
            graph_before_state_hash=canonical_hash(graph_before),
        )
        setup.add_all([knowledge_base, batch])
        setup.flush()
        setup.add(recovery)
        setup.commit()
        exact_owner = ingestion_batch_recovery_owner_token(recovery)

        with pytest.raises(IngestionResourceBusyError) as durable_block:
            async with knowledge_base_ingestion_resource_lock(
                second,
                knowledge_base_id,
                operation="context_graph_rebuild_batch",
                batch_id="ordinary-contender",
                timeout_seconds=0.2,
                poll_seconds=0.01,
            ):
                raise AssertionError("durable recovery must fence without an advisory holder")
        assert durable_block.value.diagnostics["reason"] == "active_ingestion_batch_recovery_fence"
        second.rollback()

        async with knowledge_base_ingestion_resource_lock(
            first,
            knowledge_base_id,
            operation=INGESTION_BATCH_RECOVERY_LOCK_OPERATION,
            batch_id=exact_owner,
            timeout_seconds=0.2,
            poll_seconds=0.01,
        ):
            async def ordinary_contender():
                async with knowledge_base_ingestion_resource_lock(
                    second,
                    knowledge_base_id,
                    operation="context_graph_rebuild_batch",
                    batch_id="ordinary-contender",
                    timeout_seconds=0.05,
                    poll_seconds=0.01,
                ):
                    raise AssertionError("ordinary contender must not overlap exact recovery")

            with pytest.raises(IngestionResourceBusyError) as advisory_block:
                await asyncio.create_task(ordinary_contender())
            assert advisory_block.value.diagnostics["reason"] == "resource_lock_timeout"
        second.rollback()

        persisted = setup.get(IngestionBatchRecovery, recovery.id)
        persisted.status = "parse_compensated"
        setup.commit()
        async with knowledge_base_ingestion_resource_lock(
            second,
            knowledge_base_id,
            operation="context_graph_rebuild_batch",
            batch_id="after-terminal-recovery",
            timeout_seconds=0.2,
            poll_seconds=0.01,
        ):
            pass
    finally:
        first.close()
        second.close()
        try:
            setup.rollback()
            knowledge_base = setup.get(KnowledgeBase, knowledge_base_id)
            if knowledge_base is not None:
                setup.delete(knowledge_base)
                setup.commit()
        finally:
            setup.close()
            engine.dispose()

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import sys
import time
import weakref
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session


INGESTION_RESOURCE_LOCK_PROTOCOL_VERSION = "postgres_advisory_kb_v1"
SQLITE_TEST_LOCK_PROTOCOL_VERSION = "sqlite_in_process_test_adapter_v1"
DEFAULT_INGESTION_RESOURCE_LOCK_TIMEOUT_SECONDS = 30.0
DEFAULT_INGESTION_RESOURCE_LOCK_POLL_SECONDS = 0.1
INGESTION_BATCH_RECOVERY_LOCK_OPERATION = "ingestion_batch_recovery"
INGESTION_BATCH_RECOVERY_ACTIVE_STATUSES = frozenset(
    {
        "prepared",
        "parsing",
        "parse_compensation_pending",
        "parse_compensating",
        "graph_building",
        "graph_compensation_pending",
        "manual_review",
    }
)
INGESTION_BATCH_RECOVERY_SAME_BATCH_OPERATION_STATUSES = {
    "uploaded_files_ingestion": frozenset({"prepared", "parsing", "graph_building"}),
    "ingest_file": frozenset({"prepared", "parsing"}),
    "context_graph_rebuild": frozenset({"graph_building"}),
}


class IngestionResourceBusyError(RuntimeError):
    def __init__(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            "Knowledge-base ingest/rebuild resource is busy "
            f"after {float(diagnostics.get('wait_seconds') or 0.0):.3f}s "
            f"(resource={diagnostics.get('resource_key')}, operation={diagnostics.get('operation')}). "
            "Retry after the active ingest or graph rebuild completes."
        )


class IngestionResourceLockReleaseError(RuntimeError):
    def __init__(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            "Knowledge-base ingest/rebuild resource lock could not be cleanly released "
            f"(resource={diagnostics.get('resource_key')}, backend={diagnostics.get('backend')}). "
            "The dedicated database connection was invalidated; inspect database connectivity before retrying."
        )


@dataclass
class IngestionResourceLockLease:
    knowledge_base_id: str
    resource_key: str
    advisory_key: int
    backend: str
    operation: str
    batch_id: str | None
    wait_seconds: float
    contention_count: int
    acquired_at: str
    protocol_version: str
    _owner_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    _active: bool = field(default=True, repr=False)

    def diagnostics(self, *, reentrant: bool = False) -> dict[str, Any]:
        return {
            "acquired": self._active,
            "resource_scope": "knowledge_base",
            "knowledge_base_id": self.knowledge_base_id,
            "resource_key": self.resource_key,
            "advisory_key": self.advisory_key,
            "backend": self.backend,
            "operation": self.operation,
            "batch_id": self.batch_id,
            "wait_seconds": round(float(self.wait_seconds), 6),
            "contention_count": int(self.contention_count),
            "acquired_at": self.acquired_at,
            "protocol_version": self.protocol_version,
            "reentrant": reentrant,
            "owner_host": socket.gethostname(),
            "owner_pid": os.getpid(),
        }


_ACTIVE_LEASES: ContextVar[dict[str, IngestionResourceLockLease] | None] = ContextVar(
    "active_ingestion_resource_leases",
    default=None,
)
_SQLITE_TEST_LOCKS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = weakref.WeakKeyDictionary()
_SQLITE_TEST_OWNERS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, IngestionResourceLockLease]] = weakref.WeakKeyDictionary()


def knowledge_base_resource_key(knowledge_base_id: str) -> str:
    normalized_id = str(knowledge_base_id or "").strip()
    if not normalized_id:
        raise ValueError("knowledge_base_id is required for ingestion resource locking")
    return f"knowledge_base:{normalized_id}"


def advisory_lock_key(resource_key: str) -> int:
    digest = hashlib.sha256(
        f"symbograph:ingestion-resource-lock:v1:{resource_key}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


def ingestion_resource_lock_wait_diagnostics(
    db: Session,
    knowledge_base_id: str,
    *,
    operation: str,
    batch_id: str | None = None,
    timeout_seconds: float = DEFAULT_INGESTION_RESOURCE_LOCK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    resource_key = knowledge_base_resource_key(knowledge_base_id)
    dialect_name = str(db.get_bind().dialect.name)
    if dialect_name == "postgresql":
        backend = "postgresql"
    elif dialect_name == "sqlite":
        backend = "sqlite_test_adapter"
    else:
        backend = dialect_name
    return {
        "acquired": False,
        "status": "waiting",
        "resource_scope": "knowledge_base",
        "knowledge_base_id": knowledge_base_id,
        "resource_key": resource_key,
        "advisory_key": advisory_lock_key(resource_key),
        "backend": backend,
        "operation": operation,
        "batch_id": batch_id,
        "timeout_seconds": round(max(0.0, float(timeout_seconds)), 6),
        "wait_started_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": (
            INGESTION_RESOURCE_LOCK_PROTOCOL_VERSION
            if backend == "postgresql"
            else SQLITE_TEST_LOCK_PROTOCOL_VERSION
        ),
        "retry_guidance": "wait_for_active_ingest_or_graph_rebuild",
    }


def active_ingestion_resource_lease(knowledge_base_id: str) -> IngestionResourceLockLease | None:
    resource_key = knowledge_base_resource_key(knowledge_base_id)
    lease = (_ACTIVE_LEASES.get() or {}).get(resource_key)
    if lease is None or not lease._active:
        return None
    try:
        current_task = asyncio.current_task()
    except RuntimeError:
        # ``asyncio.to_thread`` copies ContextVars but has no running event
        # loop.  The copied lease capability authorizes the bounded,
        # sequential filesystem section; unrelated threads do not inherit it.
        current_task = None
    if current_task is not None and lease._owner_task is not current_task:
        return None
    return lease


def knowledge_base_delete_recovery_owner_token(
    db: Session,
    knowledge_base_id: str,
) -> str | None:
    """Bind a KB-delete lock owner to the complete active Qdrant tombstone set."""

    from app.services.qdrant_outbox import (
        pending_qdrant_delete_intents,
        qdrant_delete_intent_recovery_diagnostics,
    )

    tombstones = pending_qdrant_delete_intents(
        db,
        knowledge_base_id=knowledge_base_id,
    )
    cards = []
    for row in tombstones:
        recovery = qdrant_delete_intent_recovery_diagnostics(row)
        cards.append(
            {
                "id": str(row.id),
                "operation": str(row.operation),
                "status": str(row.status),
                "source_operation": recovery.get("delete_source_operation"),
                "protocol_version": recovery.get("delete_protocol_version"),
                "collection_name": recovery.get("collection_name"),
                "target_count": recovery.get("target_count"),
                "retryable": bool(recovery.get("retryable")),
                "validation_error": recovery.get("delete_intent_validation_error"),
            }
        )
    cards.sort(key=lambda card: (card["id"], card["operation"], card["status"]))
    from app.services.storage_maintenance import knowledge_base_delete_owner_token

    storage_owner = knowledge_base_delete_owner_token(db, knowledge_base_id)
    if storage_owner is None and not cards:
        return None
    if storage_owner is None:
        encoded = json.dumps(
            {
                "protocol_version": "knowledge_base_delete_qdrant_owner_set_v1",
                "knowledge_base_id": knowledge_base_id,
                "tombstones": cards,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"qdrant-delete:{hashlib.sha256(encoded).hexdigest()}"
    encoded = json.dumps(
        {
            "protocol_version": "knowledge_base_delete_storage_qdrant_owner_set_v2",
            "knowledge_base_id": knowledge_base_id,
            "storage_owner": storage_owner,
            "qdrant_tombstones": cards,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"storage-delete:{hashlib.sha256(encoded).hexdigest()}"


def ingestion_batch_recovery_owner_token(recovery: Any) -> str:
    """Return the stable exact-owner token for one durable batch recovery."""

    card = {
        "protocol_version": "ingestion_batch_recovery_lock_owner_v1",
        "recovery_id": str(recovery.id),
        "batch_id": str(recovery.batch_id),
        "knowledge_base_id": str(recovery.knowledge_base_id),
        "recovery_protocol_version": str(recovery.protocol_version),
        "v_before_batch": int(recovery.v_before_batch),
        "target_version": int(recovery.target_version),
        "full_reparse": bool(recovery.full_reparse),
        "before_state_hash": str(recovery.before_state_hash),
        "graph_before_state_hash": str(recovery.graph_before_state_hash),
    }
    encoded = json.dumps(
        card,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"ingestion-recovery:{hashlib.sha256(encoded).hexdigest()}"


def active_ingestion_batch_recoveries(
    db: Session,
    *,
    knowledge_base_id: str,
) -> list[Any]:
    from app.models import IngestionBatchRecovery

    return list(
        db.scalars(
            select(IngestionBatchRecovery)
            .where(
                IngestionBatchRecovery.knowledge_base_id == knowledge_base_id,
                IngestionBatchRecovery.status.in_(
                    sorted(INGESTION_BATCH_RECOVERY_ACTIVE_STATUSES)
                ),
            )
            .order_by(
                IngestionBatchRecovery.created_at.asc(),
                IngestionBatchRecovery.id.asc(),
            )
        ).all()
    )


def ingestion_batch_recovery_owner_token_for_batch(
    db: Session,
    *,
    knowledge_base_id: str,
    batch_id: str,
) -> str | None:
    matches = [
        row
        for row in active_ingestion_batch_recoveries(
            db,
            knowledge_base_id=knowledge_base_id,
        )
        if str(row.batch_id) == str(batch_id)
    ]
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple active ingestion batch recoveries claim one batch/knowledge base"
        )
    return ingestion_batch_recovery_owner_token(matches[0]) if matches else None


def ingestion_batch_recovery_fence_diagnostics(
    db: Session,
    *,
    knowledge_base_id: str,
    operation: str,
    owner_id: str | None,
) -> dict[str, Any] | None:
    """Return a durable recovery fence diagnostic, or ``None`` when allowed."""

    recoveries = active_ingestion_batch_recoveries(
        db,
        knowledge_base_id=knowledge_base_id,
    )
    if not recoveries:
        return None
    cards = [
        {
            "recovery_id": str(row.id),
            "batch_id": str(row.batch_id),
            "status": str(row.status),
            "parse_committed": bool(row.parse_committed),
            "owner_token": ingestion_batch_recovery_owner_token(row),
        }
        for row in recoveries
    ]
    allowed = False
    matched_rule = "active_recovery_blocks_unowned_mutation"
    if len(recoveries) == 1:
        recovery = recoveries[0]
        expected_owner = cards[0]["owner_token"]
        exact_recovery_owner = (
            operation == INGESTION_BATCH_RECOVERY_LOCK_OPERATION
            and owner_id == expected_owner
        )
        same_batch_resume = (
            operation in INGESTION_BATCH_RECOVERY_SAME_BATCH_OPERATION_STATUSES
            and owner_id == str(recovery.batch_id)
            and str(recovery.status)
            in INGESTION_BATCH_RECOVERY_SAME_BATCH_OPERATION_STATUSES[operation]
        )
        allowed = exact_recovery_owner or same_batch_resume
        if exact_recovery_owner:
            matched_rule = "exact_recovery_owner"
        elif same_batch_resume:
            matched_rule = "same_batch_active_execution_resume"
    if allowed:
        return None
    return {
        "reason": "active_ingestion_batch_recovery_fence",
        "retryable": False,
        "manual_review_required": any(
            card["status"] == "manual_review" for card in cards
        ),
        "knowledge_base_id": knowledge_base_id,
        "requested_operation": operation,
        "requested_owner_id": owner_id,
        "active_recovery_count": len(cards),
        "active_recoveries": cards,
        "matched_rule": matched_rule,
        "recovery_lock_operation": INGESTION_BATCH_RECOVERY_LOCK_OPERATION,
    }


def _ensure_no_pending_knowledge_base_delete(
    db: Session,
    lease: IngestionResourceLockLease,
    *,
    requested_operation: str | None = None,
    requested_batch_id: str | None = None,
) -> None:
    effective_operation = requested_operation or lease.operation
    effective_batch_id = (
        requested_batch_id if requested_operation is not None else lease.batch_id
    )
    from app.services.qdrant_outbox import (
        pending_qdrant_delete_intents,
        qdrant_delete_intent_recovery_diagnostics,
    )

    tombstones = pending_qdrant_delete_intents(
        db,
        knowledge_base_id=lease.knowledge_base_id,
    )
    if tombstones:
        recoveries = [qdrant_delete_intent_recovery_diagnostics(row) for row in tombstones]
        expected_owner = knowledge_base_delete_recovery_owner_token(
            db,
            lease.knowledge_base_id,
        )
        exact_kb_delete_recovery = (
            effective_operation == "delete_knowledge_base_data"
            and effective_batch_id == expected_owner
            and all(
                bool(item.get("retryable"))
                and item.get("delete_source_operation") == "delete_knowledge_base_data"
                for item in recoveries
            )
        )
        if not exact_kb_delete_recovery:
            diagnostics = lease.diagnostics()
            diagnostics.update(recoveries[0])
            diagnostics.update(
                {
                    "acquired": False,
                    "delete_tombstone_count": len(tombstones),
                    "delete_tombstone_ids_sample": [
                        str(row.id) for row in tombstones[:64]
                    ],
                    "delete_tombstone_ids_truncated_count": max(
                        0, len(tombstones) - 64
                    ),
                    "requested_owner_matches": effective_batch_id == expected_owner,
                }
            )
            raise IngestionResourceBusyError(diagnostics)

    from app.services.storage_maintenance import (
        knowledge_base_delete_fence_diagnostics,
    )

    storage_delete_fence = knowledge_base_delete_fence_diagnostics(
        db,
        lease.knowledge_base_id,
    )
    if storage_delete_fence is not None:
        expected_owner = knowledge_base_delete_recovery_owner_token(
            db,
            lease.knowledge_base_id,
        )
        exact_storage_delete_recovery = (
            effective_operation == "delete_knowledge_base_data"
            and effective_batch_id == expected_owner
            and bool(storage_delete_fence.get("retryable"))
        )
        if not exact_storage_delete_recovery:
            diagnostics = lease.diagnostics()
            diagnostics.update(storage_delete_fence)
            diagnostics.update(
                {
                    "acquired": False,
                    "requested_owner_matches": effective_batch_id == expected_owner,
                }
            )
            raise IngestionResourceBusyError(diagnostics)

    # A durable single-source unlink intent survives process death while the
    # advisory lock does not.  Make that intent a central KB fence so no
    # ingest/rebuild can mutate the frozen document/scope before exact-owner
    # recovery completes.
    from app.services.ingestion import source_file_delete_resource_fence_diagnostics

    source_delete_fence = source_file_delete_resource_fence_diagnostics(
        db,
        knowledge_base_id=lease.knowledge_base_id,
        operation=effective_operation,
        owner_intent_id=effective_batch_id,
    )
    if source_delete_fence is not None:
        diagnostics = lease.diagnostics()
        diagnostics.update(source_delete_fence)
        diagnostics["acquired"] = False
        raise IngestionResourceBusyError(diagnostics)

    batch_recovery_fence = ingestion_batch_recovery_fence_diagnostics(
        db,
        knowledge_base_id=lease.knowledge_base_id,
        operation=effective_operation,
        owner_id=effective_batch_id,
    )
    if batch_recovery_fence is not None:
        diagnostics = lease.diagnostics()
        diagnostics.update(batch_recovery_fence)
        diagnostics["acquired"] = False
        raise IngestionResourceBusyError(diagnostics)


def _busy_diagnostics(
    *,
    knowledge_base_id: str,
    resource_key: str,
    lock_key: int,
    backend: str,
    operation: str,
    batch_id: str | None,
    timeout_seconds: float,
    wait_seconds: float,
    contention_count: int,
    holders: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "acquired": False,
        "reason": "resource_lock_timeout",
        "resource_scope": "knowledge_base",
        "knowledge_base_id": knowledge_base_id,
        "resource_key": resource_key,
        "advisory_key": lock_key,
        "backend": backend,
        "operation": operation,
        "batch_id": batch_id,
        "timeout_seconds": round(float(timeout_seconds), 6),
        "wait_seconds": round(float(wait_seconds), 6),
        "contention_count": int(contention_count),
        "holders": holders or [],
        "protocol_version": (
            INGESTION_RESOURCE_LOCK_PROTOCOL_VERSION
            if backend == "postgresql"
            else SQLITE_TEST_LOCK_PROTOCOL_VERSION
        ),
        "retryable": True,
        "retry_guidance": "retry_after_active_ingest_or_graph_rebuild_completes",
    }


def _postgres_lock_holders(connection, lock_key: int) -> list[dict[str, Any]]:
    unsigned_key = lock_key & ((1 << 64) - 1)
    class_id = unsigned_key >> 32
    object_id = unsigned_key & ((1 << 32) - 1)
    try:
        rows = connection.execute(
            text(
                """
                SELECT a.pid, a.application_name, a.client_addr, a.state,
                       a.wait_event_type, a.wait_event, a.query_start
                FROM pg_locks AS l
                JOIN pg_stat_activity AS a ON a.pid = l.pid
                WHERE l.locktype = 'advisory'
                  AND l.granted IS TRUE
                  AND l.classid = :class_id
                  AND l.objid = :object_id
                  AND l.objsubid = 1
                ORDER BY a.query_start ASC
                """
            ),
            {"class_id": class_id, "object_id": object_id},
        ).mappings().all()
    except Exception as exc:
        return [{"owner_lookup_error": exc.__class__.__name__}]
    return [
        {
            "pid": int(row["pid"]),
            "application_name": str(row["application_name"] or ""),
            "client_addr": str(row["client_addr"] or ""),
            "state": str(row["state"] or ""),
            "wait_event_type": str(row["wait_event_type"] or ""),
            "wait_event": str(row["wait_event"] or ""),
            "query_start": row["query_start"].isoformat() if row["query_start"] else None,
        }
        for row in rows
    ]


@asynccontextmanager
async def _postgres_resource_lock(
    db: Session,
    *,
    knowledge_base_id: str,
    resource_key: str,
    lock_key: int,
    operation: str,
    batch_id: str | None,
    timeout_seconds: float,
    poll_seconds: float,
) -> AsyncIterator[IngestionResourceLockLease]:
    bind = db.get_bind()
    engine = bind if isinstance(bind, Engine) else bind.engine
    connection = engine.connect()
    acquired = False
    lease: IngestionResourceLockLease | None = None
    started = time.monotonic()
    contention_count = 0
    try:
        while True:
            acquired = bool(
                connection.execute(
                    text("SELECT pg_try_advisory_lock(:lock_key)"),
                    {"lock_key": lock_key},
                ).scalar_one()
            )
            # Session-level advisory locks survive commit.  End the implicit
            # SQLAlchemy transaction immediately so a long parse/model call
            # does not leave an idle-in-transaction connection behind.
            connection.commit()
            if acquired:
                break
            contention_count += 1
            elapsed = time.monotonic() - started
            if elapsed >= timeout_seconds:
                raise IngestionResourceBusyError(
                    _busy_diagnostics(
                        knowledge_base_id=knowledge_base_id,
                        resource_key=resource_key,
                        lock_key=lock_key,
                        backend="postgresql",
                        operation=operation,
                        batch_id=batch_id,
                        timeout_seconds=timeout_seconds,
                        wait_seconds=elapsed,
                        contention_count=contention_count,
                        holders=_postgres_lock_holders(connection, lock_key),
                    )
                )
            await asyncio.sleep(min(poll_seconds, max(0.0, timeout_seconds - elapsed)))

        lease = IngestionResourceLockLease(
            knowledge_base_id=knowledge_base_id,
            resource_key=resource_key,
            advisory_key=lock_key,
            backend="postgresql",
            operation=operation,
            batch_id=batch_id,
            wait_seconds=time.monotonic() - started,
            contention_count=contention_count,
            acquired_at=datetime.now(timezone.utc).isoformat(),
            protocol_version=INGESTION_RESOURCE_LOCK_PROTOCOL_VERSION,
            _owner_task=asyncio.current_task(),
        )
        active = dict(_ACTIVE_LEASES.get() or {})
        active[resource_key] = lease
        context_token = _ACTIVE_LEASES.set(active)
        try:
            yield lease
        finally:
            _ACTIVE_LEASES.reset(context_token)
            lease._active = False
    finally:
        release_error: Exception | None = None
        body_error = sys.exc_info()[1]
        if acquired:
            try:
                released = bool(
                    connection.execute(
                        text("SELECT pg_advisory_unlock(:lock_key)"),
                        {"lock_key": lock_key},
                    ).scalar_one()
                )
                connection.commit()
                if not released:
                    raise RuntimeError("PostgreSQL reported that the advisory lock was not held")
            except Exception as exc:
                release_error = exc
                connection.invalidate()
        connection.close()
        if release_error is not None:
            diagnostics = {
                "resource_key": resource_key,
                "knowledge_base_id": knowledge_base_id,
                "advisory_key": lock_key,
                "backend": "postgresql",
                "operation": operation,
                "batch_id": batch_id,
                "protocol_version": INGESTION_RESOURCE_LOCK_PROTOCOL_VERSION,
                "release_error": release_error.__class__.__name__,
            }
            if body_error is not None:
                body_error.add_note(IngestionResourceLockReleaseError(diagnostics).args[0])
            else:
                raise IngestionResourceLockReleaseError(diagnostics) from release_error


def _sqlite_test_adapter_allowed() -> bool:
    return bool(os.getenv("PYTEST_CURRENT_TEST"))


@asynccontextmanager
async def _sqlite_test_resource_lock(
    *,
    knowledge_base_id: str,
    resource_key: str,
    lock_key: int,
    operation: str,
    batch_id: str | None,
    timeout_seconds: float,
) -> AsyncIterator[IngestionResourceLockLease]:
    if not _sqlite_test_adapter_allowed():
        raise RuntimeError(
            "SQLite ingestion locking is available only as an explicit pytest adapter; "
            "production ingest/rebuild correctness requires PostgreSQL advisory locks"
        )
    loop = asyncio.get_running_loop()
    locks = _SQLITE_TEST_LOCKS.setdefault(loop, {})
    owners = _SQLITE_TEST_OWNERS.setdefault(loop, {})
    lock = locks.setdefault(resource_key, asyncio.Lock())
    started = time.monotonic()
    contention_count = 1 if lock.locked() else 0
    try:
        if lock.locked():
            await asyncio.wait_for(lock.acquire(), timeout=max(0.0, timeout_seconds))
        else:
            await lock.acquire()
    except TimeoutError as exc:
        elapsed = time.monotonic() - started
        holder = owners.get(resource_key)
        raise IngestionResourceBusyError(
            _busy_diagnostics(
                knowledge_base_id=knowledge_base_id,
                resource_key=resource_key,
                lock_key=lock_key,
                backend="sqlite_test_adapter",
                operation=operation,
                batch_id=batch_id,
                timeout_seconds=timeout_seconds,
                wait_seconds=elapsed,
                contention_count=contention_count,
                holders=[holder.diagnostics() if holder else {"owner": "unknown_test_task"}],
            )
        ) from exc

    lease = IngestionResourceLockLease(
        knowledge_base_id=knowledge_base_id,
        resource_key=resource_key,
        advisory_key=lock_key,
        backend="sqlite_test_adapter",
        operation=operation,
        batch_id=batch_id,
        wait_seconds=time.monotonic() - started,
        contention_count=contention_count,
        acquired_at=datetime.now(timezone.utc).isoformat(),
        protocol_version=SQLITE_TEST_LOCK_PROTOCOL_VERSION,
        _owner_task=asyncio.current_task(),
    )
    owners[resource_key] = lease
    active = dict(_ACTIVE_LEASES.get() or {})
    active[resource_key] = lease
    context_token = _ACTIVE_LEASES.set(active)
    try:
        yield lease
    finally:
        _ACTIVE_LEASES.reset(context_token)
        lease._active = False
        if owners.get(resource_key) is lease:
            owners.pop(resource_key, None)
        lock.release()


@asynccontextmanager
async def knowledge_base_ingestion_resource_lock(
    db: Session,
    knowledge_base_id: str,
    *,
    operation: str,
    batch_id: str | None = None,
    timeout_seconds: float = DEFAULT_INGESTION_RESOURCE_LOCK_TIMEOUT_SECONDS,
    poll_seconds: float = DEFAULT_INGESTION_RESOURCE_LOCK_POLL_SECONDS,
) -> AsyncIterator[IngestionResourceLockLease]:
    timeout_seconds = max(0.0, float(timeout_seconds))
    poll_seconds = max(0.001, float(poll_seconds))
    resource_key = knowledge_base_resource_key(knowledge_base_id)
    existing = active_ingestion_resource_lease(knowledge_base_id)
    if existing is not None:
        _ensure_no_pending_knowledge_base_delete(
            db,
            existing,
            requested_operation=operation,
            requested_batch_id=batch_id,
        )
        yield existing
        return

    lock_key = advisory_lock_key(resource_key)
    dialect_name = str(db.get_bind().dialect.name)
    if dialect_name == "postgresql":
        async with _postgres_resource_lock(
            db,
            knowledge_base_id=knowledge_base_id,
            resource_key=resource_key,
            lock_key=lock_key,
            operation=operation,
            batch_id=batch_id,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        ) as lease:
            _ensure_no_pending_knowledge_base_delete(db, lease)
            yield lease
        return
    if dialect_name == "sqlite":
        async with _sqlite_test_resource_lock(
            knowledge_base_id=knowledge_base_id,
            resource_key=resource_key,
            lock_key=lock_key,
            operation=operation,
            batch_id=batch_id,
            timeout_seconds=timeout_seconds,
        ) as lease:
            _ensure_no_pending_knowledge_base_delete(db, lease)
            yield lease
        return
    raise RuntimeError(
        f"Unsupported database backend for ingestion resource locking: {dialect_name}. "
        "Production ingest/rebuild correctness requires PostgreSQL advisory locks."
    )

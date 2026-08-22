from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import (
    Chunk,
    Document,
    DocumentVersion,
    IngestionCompensationLog,
    KnowledgeBase,
    StorageMaintenanceIntent,
    VectorRecord,
)
from app.services.storage import (
    SourceSnapshotError,
    VerifiedSourceIdentity,
    durable_delete_storage_tree,
    durable_rmdir,
    durable_unlink,
    inventory_storage_tree,
    verified_source_checksum,
)
from app.services.vector_store import (
    QDRANT_DESTRUCTIVE_MAX_COLLECTIONS,
    QDRANT_DESTRUCTIVE_MAX_OWNED_POINTS,
    VectorStore,
)


KNOWLEDGE_BASE_DELETE_OPERATION = "knowledge_base_delete_v2"
KNOWLEDGE_BASE_DELETE_PROTOCOL_VERSION = (
    "knowledge_base_delete_pg_tombstone_fs_qdrant_proof_v2"
)
KNOWLEDGE_BASE_DELETE_SCOPE_PREFIX = "knowledge_base:"
KNOWLEDGE_BASE_DELETE_QDRANT_REASON = "delete_knowledge_base_data"
STORAGE_MAINTENANCE_HISTORY_LIMIT = 64
STORAGE_MAINTENANCE_DB_BATCH_SIZE = 512
STORAGE_MAINTENANCE_POINT_LOOKUP_BATCH_SIZE = 1_000
SOURCE_SNAPSHOT_GC_OPERATION = "source_snapshot_gc_v1"
SOURCE_SNAPSHOT_GC_PROTOCOL_VERSION = (
    "source_snapshot_bounded_inventory_retention_gc_v1"
)
SOURCE_SNAPSHOT_GC_DEFAULT_RETENTION_SECONDS = 7 * 24 * 60 * 60
STORAGE_MAINTENANCE_RECOVERY_HEALTH_PROTOCOL_VERSION = (
    "storage_maintenance_recovery_health_v1"
)
_STORAGE_MAINTENANCE_RECOVERY_HEALTH_LOCK = threading.Lock()
_STORAGE_MAINTENANCE_RECOVERY_HEALTH: dict[str, Any] = {
    "protocol_version": STORAGE_MAINTENANCE_RECOVERY_HEALTH_PROTOCOL_VERSION,
    "status": "not_run",
    "last_run_at": None,
    "selected": 0,
    "completed": 0,
    "pending": 0,
    "cache_pending": 0,
    "external_pending": 0,
    "manual_review": 0,
    "failed": 0,
    "retryable": True,
}


class StorageMaintenanceError(RuntimeError):
    pass


class StorageMaintenanceIntegrityError(StorageMaintenanceError):
    pass


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _payload_hash(payload: dict[str, Any]) -> str:
    return _canonical_hash(
        {key: value for key, value in payload.items() if key != "payload_hash"}
    )


def _append_history(
    payload: dict[str, Any],
    *,
    status: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = dict(payload)
    history = list(updated.get("state_history") or [])
    history.append(
        {
            "status": status,
            "at": datetime.utcnow().isoformat(),
            "details": details or {},
        }
    )
    if len(history) > STORAGE_MAINTENANCE_HISTORY_LIMIT:
        history = history[-STORAGE_MAINTENANCE_HISTORY_LIMIT:]
    updated["state_history"] = history
    updated["state_history_total_count"] = int(
        updated.get("state_history_total_count") or 0
    ) + 1
    updated["payload_hash"] = _payload_hash(updated)
    return updated


def _validate_delete_intent(
    row: StorageMaintenanceIntent,
    *,
    knowledge_base_id: str,
) -> dict[str, Any]:
    payload = dict(row.payload_json or {})
    required = {
        "protocol_version",
        "intent_id",
        "knowledge_base_id",
        "knowledge_base_name",
        "scope_key",
        "phase",
        "filesystem_inventory",
        "qdrant_inventory",
        "database_counts",
        "state_history",
        "state_history_total_count",
        "payload_hash",
    }
    if set(payload) != required:
        raise StorageMaintenanceIntegrityError(
            "Knowledge-base delete tombstone has an invalid frozen schema"
        )
    if (
        row.operation != KNOWLEDGE_BASE_DELETE_OPERATION
        or row.protocol_version != KNOWLEDGE_BASE_DELETE_PROTOCOL_VERSION
        or payload["protocol_version"] != KNOWLEDGE_BASE_DELETE_PROTOCOL_VERSION
        or str(row.id) != str(payload["intent_id"])
        or str(row.knowledge_base_id) != str(knowledge_base_id)
        or str(payload["knowledge_base_id"]) != str(knowledge_base_id)
        or str(row.scope_key) != str(payload["scope_key"])
        or row.inventory_hash
        != _canonical_hash(
            {
                "filesystem": payload["filesystem_inventory"],
                "qdrant": payload["qdrant_inventory"],
            }
        )
        or payload.get("payload_hash") != _payload_hash(payload)
    ):
        raise StorageMaintenanceIntegrityError(
            "Knowledge-base delete tombstone identity/hash validation failed"
        )
    return payload


def active_knowledge_base_delete_intent(
    db: Session,
    knowledge_base_id: str,
) -> StorageMaintenanceIntent | None:
    return db.scalar(
        select(StorageMaintenanceIntent)
        .where(
            StorageMaintenanceIntent.operation == KNOWLEDGE_BASE_DELETE_OPERATION,
            StorageMaintenanceIntent.scope_key
            == f"{KNOWLEDGE_BASE_DELETE_SCOPE_PREFIX}{knowledge_base_id}",
            StorageMaintenanceIntent.status != "completed",
        )
        .order_by(StorageMaintenanceIntent.created_at.asc())
        .limit(1)
    )


def validated_knowledge_base_delete_intent(
    db: Session,
    *,
    knowledge_base_id: str,
    intent_id: str,
) -> tuple[StorageMaintenanceIntent, dict[str, Any]]:
    """Load one exact immutable delete tombstone by primary key and validate it.

    Recovery callers must bind the primary key instead of selecting a latest
    completed row: repeatable same-scope maintenance deliberately retains every
    completed audit tombstone.
    """

    row = db.get(StorageMaintenanceIntent, intent_id)
    if row is None:
        raise StorageMaintenanceIntegrityError(
            "The exact knowledge-base delete tombstone does not exist"
        )
    payload = _validate_delete_intent(
        row,
        knowledge_base_id=knowledge_base_id,
    )
    return row, payload


def inventory_knowledge_base_qdrant_scope(
    db: Session,
    knowledge_base: KnowledgeBase,
) -> dict[str, Any]:
    """Return the complete bounded Qdrant ownership inventory without writes."""

    with db.no_autoflush:
        return _collection_inventory(db, knowledge_base)


def knowledge_base_delete_fence_diagnostics(
    db: Session,
    knowledge_base_id: str,
) -> dict[str, Any] | None:
    row = active_knowledge_base_delete_intent(db, knowledge_base_id)
    if row is None:
        return None
    try:
        payload = _validate_delete_intent(row, knowledge_base_id=knowledge_base_id)
        validation_error = None
    except StorageMaintenanceIntegrityError as exc:
        payload = {}
        validation_error = exc.__class__.__name__
    return {
        "protocol_version": KNOWLEDGE_BASE_DELETE_PROTOCOL_VERSION,
        "intent_id": str(row.id),
        "knowledge_base_id": str(row.knowledge_base_id),
        "status": str(row.status),
        "phase": payload.get("phase"),
        "inventory_hash": str(row.inventory_hash),
        "retryable": row.status
        in {
            "intent_committed",
            "external_deleting",
            "external_applied",
            "facts_deleted",
            "cache_invalidation_pending",
        }
        and validation_error is None,
        "validation_error": validation_error,
        "reason": "knowledge_base_delete_pending",
    }


def knowledge_base_delete_owner_token(
    db: Session,
    knowledge_base_id: str,
) -> str | None:
    row = active_knowledge_base_delete_intent(db, knowledge_base_id)
    if row is None:
        return None
    payload = _validate_delete_intent(row, knowledge_base_id=knowledge_base_id)
    return (
        f"storage-delete:{row.id}:"
        f"{_canonical_hash({'inventory_hash': row.inventory_hash, 'phase': payload['phase'], 'status': row.status})}"
    )


def _collection_inventory(
    db: Session,
    knowledge_base: KnowledgeBase,
    *,
    include_pending_delete_targets: bool = True,
) -> dict[str, Any]:
    from app.services.qdrant_outbox import (
        pending_qdrant_delete_intents,
        qdrant_delete_intent_recovery_diagnostics,
    )

    probe = VectorStore(
        knowledge_base.name,
        knowledge_base_id=knowledge_base.id,
    )
    collection_inventory = probe.list_collection_names_bounded()
    if collection_inventory.get("truncated"):
        raise StorageMaintenanceIntegrityError(
            "Qdrant collection inventory exceeded its hard bound"
        )
    if not collection_inventory.get("complete_backend_inventory"):
        raise StorageMaintenanceIntegrityError(
            "Full knowledge-base deletion requires a complete Qdrant backend inventory"
        )

    pg_collection_rows = list(
        db.scalars(
            select(VectorRecord.collection_name)
            .where(VectorRecord.knowledge_base_id == knowledge_base.id)
            .distinct()
            .order_by(VectorRecord.collection_name.asc())
            .limit(QDRANT_DESTRUCTIVE_MAX_COLLECTIONS + 1)
        ).all()
    )
    if len(pg_collection_rows) > QDRANT_DESTRUCTIVE_MAX_COLLECTIONS:
        raise StorageMaintenanceIntegrityError(
            "PostgreSQL vector collection inventory exceeded its hard bound"
        )
    pg_collections = {str(item) for item in pg_collection_rows}
    old_intents_by_collection: dict[str, IngestionCompensationLog] = {}
    pending_rows = (
        pending_qdrant_delete_intents(
            db,
            knowledge_base_id=knowledge_base.id,
        )
        if include_pending_delete_targets
        else []
    )
    for row in pending_rows:
        recovery = qdrant_delete_intent_recovery_diagnostics(row)
        collection_name = str(recovery.get("collection_name") or "")
        if (
            recovery.get("delete_source_operation")
            != KNOWLEDGE_BASE_DELETE_QDRANT_REASON
            or not collection_name
            or collection_name in old_intents_by_collection
            or not recovery.get("retryable")
        ):
            raise StorageMaintenanceIntegrityError(
                "An active Qdrant delete intent is not an exact retryable "
                "knowledge-base-delete scope"
            )
        old_intents_by_collection[collection_name] = row

    collection_names = sorted(
        set(collection_inventory.get("collection_names") or [])
        | pg_collections
        | set(old_intents_by_collection)
    )
    if len(collection_names) > QDRANT_DESTRUCTIVE_MAX_COLLECTIONS:
        raise StorageMaintenanceIntegrityError(
            "Combined Qdrant/PostgreSQL collection inventory exceeded its hard bound"
        )
    cards: list[dict[str, Any]] = []
    all_pairs: list[list[str]] = []
    for collection_name in collection_names:
        store = VectorStore(
            knowledge_base.name,
            collection_name=collection_name,
            knowledge_base_id=knowledge_base.id,
        )
        remaining_point_budget = (
            QDRANT_DESTRUCTIVE_MAX_OWNED_POINTS - len(all_pairs)
        )
        observed = store.list_owned_ids_complete(
            knowledge_base.id,
            max_points=max(1, remaining_point_budget),
        )
        if observed.get("truncated"):
            raise StorageMaintenanceIntegrityError(
                f"Qdrant owned inventory exceeded its hard bound: {collection_name}"
            )
        ids = set(str(item) for item in observed.get("ids") or [])
        resumed = old_intents_by_collection.get(collection_name)
        if resumed is not None:
            ids.update(str(item) for item in (resumed.target_ids_json or []))
        sorted_ids = sorted(ids)
        if not sorted_ids:
            continue
        if (
            remaining_point_budget <= 0
            or len(sorted_ids) > remaining_point_budget
        ):
            raise StorageMaintenanceIntegrityError(
                "Combined Qdrant owned inventory exceeded its hard bound"
            )
        postgresql_point_ids: set[str] = set()
        for offset in range(
            0,
            len(sorted_ids),
            STORAGE_MAINTENANCE_POINT_LOOKUP_BATCH_SIZE,
        ):
            lookup_ids = sorted_ids[
                offset : offset + STORAGE_MAINTENANCE_POINT_LOOKUP_BATCH_SIZE
            ]
            postgresql_point_ids.update(
                str(item)
                for item in db.scalars(
                    select(VectorRecord.qdrant_point_id).where(
                        VectorRecord.knowledge_base_id == knowledge_base.id,
                        VectorRecord.collection_name == collection_name,
                        VectorRecord.qdrant_point_id.in_(lookup_ids),
                    )
                ).all()
            )
        ids_hash = _canonical_hash(sorted_ids)
        cards.append(
            {
                "collection_name": collection_name,
                "point_count": len(sorted_ids),
                "point_ids": sorted_ids,
                "point_ids_hash": ids_hash,
                "orphan_point_count": len(ids - postgresql_point_ids),
            }
        )
        all_pairs.extend([[collection_name, point_id] for point_id in sorted_ids])
    cards.sort(key=lambda item: str(item["collection_name"]))
    all_pairs.sort()
    return {
        "protocol_version": "qdrant_payload_owned_inventory_v1",
        "collection_scan": collection_inventory,
        "collections": cards,
        "owned_point_count": len(all_pairs),
        "owned_scope_hash": _canonical_hash(all_pairs),
    }


def _observed_qdrant_scope(
    db: Session,
    knowledge_base: KnowledgeBase,
    *,
    include_pending_delete_targets: bool,
) -> dict[str, list[str]]:
    inventory = _collection_inventory(
        db,
        knowledge_base,
        include_pending_delete_targets=include_pending_delete_targets,
    )
    return {
        str(card["collection_name"]): list(card["point_ids"])
        for card in inventory["collections"]
    }


def prepare_knowledge_base_delete_intent(
    db: Session,
    knowledge_base: KnowledgeBase,
) -> tuple[StorageMaintenanceIntent, dict[str, Any]]:
    existing = active_knowledge_base_delete_intent(db, knowledge_base.id)
    if existing is not None:
        payload = _validate_delete_intent(
            existing,
            knowledge_base_id=knowledge_base.id,
        )
        if knowledge_base.lifecycle_status not in {
            "deleting",
            "delete_manual_review",
        }:
            raise StorageMaintenanceIntegrityError(
                "Delete tombstone exists without its KnowledgeBase lifecycle fence"
            )
        return existing, payload

    settings = get_settings()
    paths = settings.knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )
    filesystem_inventory = inventory_storage_tree(
        paths["knowledge_base_root"],
        authorized_parent=Path(settings.data_root),
    )
    qdrant_inventory = _collection_inventory(db, knowledge_base)
    database_counts = {
        "documents": int(
            db.scalar(
                select(func.count(Document.id)).where(
                    Document.knowledge_base_id == knowledge_base.id
                )
            )
            or 0
        ),
        "chunks": int(
            db.scalar(
                select(func.count(Chunk.id)).where(
                    Chunk.knowledge_base_id == knowledge_base.id
                )
            )
            or 0
        ),
        "vector_records": int(
            db.scalar(
                select(func.count(VectorRecord.id)).where(
                    VectorRecord.knowledge_base_id == knowledge_base.id
                )
            )
            or 0
        ),
    }
    intent_id = hashlib.sha256(
        f"{knowledge_base.id}:{datetime.utcnow().isoformat()}".encode("utf-8")
    ).hexdigest()[:32]
    # Preserve UUID-shaped identifiers for existing operational tooling.
    intent_id = (
        f"{intent_id[:8]}-{intent_id[8:12]}-{intent_id[12:16]}-"
        f"{intent_id[16:20]}-{intent_id[20:32]}"
    )
    scope_key = f"{KNOWLEDGE_BASE_DELETE_SCOPE_PREFIX}{knowledge_base.id}"
    inventory_hash = _canonical_hash(
        {
            "filesystem": filesystem_inventory,
            "qdrant": qdrant_inventory,
        }
    )
    payload = {
        "protocol_version": KNOWLEDGE_BASE_DELETE_PROTOCOL_VERSION,
        "intent_id": intent_id,
        "knowledge_base_id": knowledge_base.id,
        "knowledge_base_name": knowledge_base.name,
        "scope_key": scope_key,
        "phase": "intent_committed",
        "filesystem_inventory": filesystem_inventory,
        "qdrant_inventory": qdrant_inventory,
        "database_counts": database_counts,
        "state_history": [
            {
                "status": "intent_committed",
                "at": datetime.utcnow().isoformat(),
                "details": {},
            }
        ],
        "state_history_total_count": 1,
    }
    payload["payload_hash"] = _payload_hash(payload)
    row = StorageMaintenanceIntent(
        id=intent_id,
        knowledge_base_id=knowledge_base.id,
        knowledge_base_name=knowledge_base.name,
        operation=KNOWLEDGE_BASE_DELETE_OPERATION,
        protocol_version=KNOWLEDGE_BASE_DELETE_PROTOCOL_VERSION,
        scope_key=scope_key,
        target_root=str(paths["knowledge_base_root"]),
        inventory_hash=inventory_hash,
        payload_json=payload,
        status="intent_committed",
    )
    knowledge_base.lifecycle_status = "deleting"
    db.add(row)
    db.commit()
    return row, payload


def transition_delete_intent(
    db: Session,
    row: StorageMaintenanceIntent,
    payload: dict[str, Any],
    *,
    status: str,
    phase: str,
    details: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    updated = dict(payload)
    updated["phase"] = phase
    updated = _append_history(updated, status=status, details=details)
    row.payload_json = updated
    row.status = status
    row.error_message = error_message
    db.commit()
    return updated


def apply_qdrant_delete_inventory(
    db: Session,
    *,
    knowledge_base: KnowledgeBase,
    row: StorageMaintenanceIntent,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    from app.services.qdrant_outbox import (
        pending_qdrant_delete_intents,
        persist_qdrant_delete_attempt,
        qdrant_delete_intent_recovery_diagnostics,
        qdrant_point_mutation_lock,
        record_qdrant_delete_attempt_applied,
        record_qdrant_delete_attempt_error,
        replay_qdrant_delete_intent,
    )

    expected_scope = {
        str(card["collection_name"]): list(card["point_ids"])
        for card in payload["qdrant_inventory"]["collections"]
    }
    observed_scope = _observed_qdrant_scope(
        db,
        knowledge_base,
        include_pending_delete_targets=True,
    )
    if observed_scope != expected_scope:
        raise StorageMaintenanceIntegrityError(
            "Qdrant owned scope changed after the durable delete inventory was committed"
        )
    payload = transition_delete_intent(
        db,
        row,
        payload,
        status="external_deleting",
        phase="qdrant_deleting",
    )
    existing_by_collection: dict[str, IngestionCompensationLog] = {}
    for existing in pending_qdrant_delete_intents(
        db,
        knowledge_base_id=knowledge_base.id,
    ):
        recovery = qdrant_delete_intent_recovery_diagnostics(existing)
        collection_name = str(recovery.get("collection_name") or "")
        if (
            recovery.get("delete_source_operation")
            != KNOWLEDGE_BASE_DELETE_QDRANT_REASON
            or collection_name not in expected_scope
            or sorted(str(item) for item in (existing.target_ids_json or []))
            != sorted(expected_scope[collection_name])
            or collection_name in existing_by_collection
            or not recovery.get("retryable")
        ):
            raise StorageMaintenanceIntegrityError(
                "Qdrant delete recovery scope no longer matches the KB tombstone"
            )
        existing_by_collection[collection_name] = existing

    applied_intent_ids: list[str] = []
    for collection_name, point_ids in sorted(expected_scope.items()):
        store = VectorStore(
            knowledge_base.name,
            collection_name=collection_name,
            knowledge_base_id=knowledge_base.id,
        )
        existing = existing_by_collection.get(collection_name)
        if existing is not None:
            replay_qdrant_delete_intent(db, store=store, row=existing)
            applied_intent_ids.append(str(existing.id))
            continue
        current_points = store.get_points_batched(point_ids)
        if sorted(str(item.get("id")) for item in current_points) != sorted(point_ids):
            raise StorageMaintenanceIntegrityError(
                "A frozen Qdrant target disappeared before its durable delete intent"
            )
        intent_id = persist_qdrant_delete_attempt(
            db,
            knowledge_base_id=knowledge_base.id,
            collection_name=collection_name,
            target_ids=point_ids,
            current_points=current_points,
            reason=KNOWLEDGE_BASE_DELETE_QDRANT_REASON,
        )
        try:
            with qdrant_point_mutation_lock(
                db,
                collection_name=collection_name,
                point_ids=point_ids,
            ):
                store.delete_if_payload_matches(current_points)
                record_qdrant_delete_attempt_applied(
                    db,
                    intent_id=intent_id,
                )
        except Exception as exc:
            record_qdrant_delete_attempt_error(
                db,
                intent_id=intent_id,
                error=exc,
            )
            raise
        applied_intent_ids.append(intent_id)

    remaining = _observed_qdrant_scope(
        db,
        knowledge_base,
        include_pending_delete_targets=False,
    )
    if any(remaining.values()):
        raise StorageMaintenanceIntegrityError(
            "Qdrant knowledge-base delete postcondition still has owned points"
        )
    return payload, applied_intent_ids


def apply_filesystem_delete_inventory(
    db: Session,
    *,
    row: StorageMaintenanceIntent,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = get_settings()
    try:
        result = durable_delete_storage_tree(
            payload["filesystem_inventory"],
            authorized_parent=Path(settings.data_root),
        )
    except SourceSnapshotError as exc:
        raise StorageMaintenanceIntegrityError(str(exc)) from exc
    payload = transition_delete_intent(
        db,
        row,
        payload,
        status="external_applied",
        phase="external_applied",
        details={"filesystem_delete": result},
    )
    return payload, result


def mark_delete_intent_failure(
    db: Session,
    *,
    row_id: str,
    error: BaseException,
) -> None:
    db.rollback()
    row = db.get(StorageMaintenanceIntent, row_id)
    if row is None:
        raise StorageMaintenanceIntegrityError(
            "Knowledge-base delete failed and its durable tombstone disappeared"
        )
    payload = _validate_delete_intent(
        row,
        knowledge_base_id=row.knowledge_base_id,
    )
    manual_review = isinstance(error, StorageMaintenanceIntegrityError)
    payload = _append_history(
        payload,
        status="manual_review" if manual_review else "external_deleting",
        details={"failure_type": error.__class__.__name__},
    )
    row.payload_json = payload
    row.status = "manual_review" if manual_review else "external_deleting"
    row.error_message = error.__class__.__name__
    knowledge_base = db.get(KnowledgeBase, row.knowledge_base_id)
    if knowledge_base is not None:
        knowledge_base.lifecycle_status = (
            "delete_manual_review" if manual_review else "deleting"
        )
    db.commit()


def _nested_string_values(
    value: Any,
    *,
    max_nodes: int = 100_000,
) -> tuple[set[str], int]:
    values: set[str] = set()
    pending = [value]
    visited = 0
    while pending:
        item = pending.pop()
        visited += 1
        if visited > max_nodes:
            raise StorageMaintenanceIntegrityError(
                "Snapshot reference payload scan exceeded its hard node bound"
            )
        if isinstance(item, str):
            values.add(item)
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    return values, visited


def build_source_snapshot_gc_inventory(
    db: Session,
    knowledge_base: KnowledgeBase,
    *,
    retention_seconds: int = SOURCE_SNAPSHOT_GC_DEFAULT_RETENTION_SECONDS,
    max_entries: int = 100_000,
) -> dict[str, Any]:
    if type(retention_seconds) is not int or retention_seconds < 0:
        raise ValueError("Snapshot GC retention_seconds must be a nonnegative integer")
    settings = get_settings()
    paths = settings.knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )
    snapshot_root = paths["ingestion_root"] / "source_snapshots"
    tree = inventory_storage_tree(
        snapshot_root,
        authorized_parent=paths["ingestion_root"],
        max_entries=max_entries,
    )

    active_paths: set[str] = set()
    active_checksums: set[str] = set()
    audit_paths: set[str] = set()
    audit_checksums: set[str] = set()
    version_cursor: str | None = None
    version_count = 0
    while True:
        version_query = (
            select(
                DocumentVersion.id,
                DocumentVersion.storage_path,
                DocumentVersion.checksum,
                DocumentVersion.is_active,
            )
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(Document.knowledge_base_id == knowledge_base.id)
            .order_by(DocumentVersion.id.asc())
            .limit(STORAGE_MAINTENANCE_DB_BATCH_SIZE)
        )
        if version_cursor is not None:
            version_query = version_query.where(
                DocumentVersion.id > version_cursor
            )
        version_rows = list(db.execute(version_query).all())
        if not version_rows:
            break
        version_count += len(version_rows)
        if version_count > max_entries:
            raise StorageMaintenanceIntegrityError(
                "Snapshot GC document-version references exceeded their hard bound"
            )
        for version_id, storage_path, checksum, is_active in version_rows:
            target_paths = active_paths if is_active else audit_paths
            target_checksums = (
                active_checksums if is_active else audit_checksums
            )
            target_paths.add(str(Path(str(storage_path)).absolute()))
            target_checksums.add(str(checksum))
            version_cursor = str(version_id)

    pending_values: set[str] = set()
    audit_values: set[str] = set()
    terminal_intent_statuses = {
        "completed",
        "committed",
        "rolled_back",
        "compensated",
        "cancelled",
    }
    intent_cursor: str | None = None
    intent_count = 0
    remaining_payload_nodes = min(max_entries * 16, 1_000_000)
    while True:
        intent_query = (
            select(
                IngestionCompensationLog.id,
                IngestionCompensationLog.payload_json,
                IngestionCompensationLog.status,
            )
            .where(
                IngestionCompensationLog.knowledge_base_id
                == knowledge_base.id
            )
            .order_by(IngestionCompensationLog.id.asc())
            .limit(STORAGE_MAINTENANCE_DB_BATCH_SIZE)
        )
        if intent_cursor is not None:
            intent_query = intent_query.where(
                IngestionCompensationLog.id > intent_cursor
            )
        intent_rows = list(db.execute(intent_query).all())
        if not intent_rows:
            break
        intent_count += len(intent_rows)
        if intent_count > max_entries:
            raise StorageMaintenanceIntegrityError(
                "Snapshot GC compensation references exceeded their hard bound"
            )
        for intent_id, payload_json, intent_status in intent_rows:
            values, visited_nodes = _nested_string_values(
                payload_json or {},
                max_nodes=remaining_payload_nodes,
            )
            remaining_payload_nodes -= visited_nodes
            if intent_status in terminal_intent_statuses:
                audit_values.update(values)
            else:
                pending_values.update(values)
            intent_cursor = str(intent_id)

    # `datetime.utcnow()` is naive and `.timestamp()` interprets naive values
    # in the host timezone.  Retention must be independent of deployment
    # timezone, so anchor the cutoff directly to the Unix clock.
    cutoff_ns = time.time_ns() - (retention_seconds * 1_000_000_000)
    files: list[dict[str, Any]] = []
    counts = {
        "active_reference": 0,
        "pending_intent": 0,
        "audit_retained": 0,
        "orphan": 0,
        "gc_candidate": 0,
    }
    for card in tree.get("entries") or []:
        if card.get("kind") != "file":
            continue
        absolute = str(
            (snapshot_root / str(card["relative_path"])).absolute()
        )
        checksum = str(card["checksum"])
        if absolute in active_paths or checksum in active_checksums:
            classification = "active_reference"
        elif absolute in pending_values or checksum in pending_values:
            classification = "pending_intent"
        elif (
            absolute in audit_paths
            or checksum in audit_checksums
            or absolute in audit_values
            or checksum in audit_values
        ):
            classification = "audit_retained"
        else:
            classification = "orphan"
        gc_candidate = (
            classification == "orphan"
            and int(card["mtime_ns"]) <= cutoff_ns
        )
        counts[classification] += 1
        if gc_candidate:
            counts["gc_candidate"] += 1
        files.append(
            {
                **card,
                "classification": classification,
                "gc_candidate": gc_candidate,
            }
        )
    files.sort(key=lambda item: str(item["relative_path"]))
    inventory = {
        "protocol_version": SOURCE_SNAPSHOT_GC_PROTOCOL_VERSION,
        "knowledge_base_id": knowledge_base.id,
        "knowledge_base_name": knowledge_base.name,
        "snapshot_root": str(snapshot_root),
        "retention_seconds": retention_seconds,
        "cutoff_ns": cutoff_ns,
        "tree_inventory_hash": tree["inventory_hash"],
        "tree_root_identity": tree.get("root_identity"),
        "tree_exists": bool(tree["exists"]),
        "files": files,
        "counts": counts,
        "truncated": False,
        "max_entries": max_entries,
    }
    inventory["inventory_hash"] = _canonical_hash(
        {
            key: value
            for key, value in inventory.items()
            if key not in {"cutoff_ns", "inventory_hash"}
        }
    )
    return inventory


def _validate_snapshot_gc_payload(
    row: StorageMaintenanceIntent,
) -> dict[str, Any]:
    payload = dict(row.payload_json or {})
    if (
        row.operation != SOURCE_SNAPSHOT_GC_OPERATION
        or row.protocol_version != SOURCE_SNAPSHOT_GC_PROTOCOL_VERSION
        or payload.get("protocol_version") != SOURCE_SNAPSHOT_GC_PROTOCOL_VERSION
        or payload.get("intent_id") != row.id
        or payload.get("knowledge_base_id") != row.knowledge_base_id
        or payload.get("inventory_hash") != row.inventory_hash
        or payload.get("payload_hash") != _payload_hash(payload)
    ):
        raise StorageMaintenanceIntegrityError(
            "Snapshot GC durable intent identity/hash validation failed"
        )
    return payload


def _snapshot_gc_expected_identity(
    card: dict[str, Any],
    snapshot_root: Path,
) -> VerifiedSourceIdentity:
    root_stat = snapshot_root.lstat()
    return VerifiedSourceIdentity(
        protocol_version=str(card["final_open_protocol_version"]),
        root_device_id=int(root_stat.st_dev),
        root_inode=int(root_stat.st_ino),
        device_id=int(card["device_id"]),
        inode=int(card["inode"]),
        size_bytes=int(card["size_bytes"]),
        mtime_ns=int(card["mtime_ns"]),
        ctime_ns=int(card["ctime_ns"]),
        link_count=int(card["link_count"]),
    )


def _apply_snapshot_gc_files(payload: dict[str, Any]) -> dict[str, Any]:
    snapshot_root = Path(str(payload["inventory"]["snapshot_root"]))
    deleted_files = 0
    parent_candidates: dict[str, dict[str, Any]] = {}
    directory_cards = {
        str(card["relative_path"]): card
        for card in payload["tree_entries"]
        if card.get("kind") == "directory"
    }
    for card in payload["targets"]:
        target = snapshot_root / str(card["relative_path"])
        if not target.exists() and not target.is_symlink():
            continue
        expected = _snapshot_gc_expected_identity(card, snapshot_root)
        before = target.lstat()
        if (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
            int(before.st_nlink),
        ) != (
            expected.device_id,
            expected.inode,
            expected.size_bytes,
            expected.mtime_ns,
            expected.ctime_ns,
            expected.link_count,
        ):
            raise StorageMaintenanceIntegrityError(
                f"Snapshot GC target identity drifted: {card['relative_path']}"
            )
        mode = stat.S_IMODE(before.st_mode)
        if not mode & stat.S_IWUSR:
            os.chmod(target, mode | stat.S_IWUSR)
        checksum, observed = verified_source_checksum(target, snapshot_root)
        if (
            checksum != card["checksum"]
            or observed.device_id != expected.device_id
            or observed.inode != expected.inode
            or observed.size_bytes != expected.size_bytes
            or observed.mtime_ns != expected.mtime_ns
            or observed.link_count != expected.link_count
        ):
            raise StorageMaintenanceIntegrityError(
                f"Snapshot GC target changed during controlled unprotect: {card['relative_path']}"
            )
        durable_unlink(target, expected_identity=observed)
        deleted_files += 1
        cursor = Path(str(card["relative_path"])).parent
        while cursor != Path("."):
            directory_card = directory_cards.get(cursor.as_posix())
            if directory_card is not None:
                parent_candidates[cursor.as_posix()] = directory_card
            cursor = cursor.parent

    deleted_directories = 0
    for relative, card in sorted(
        parent_candidates.items(),
        key=lambda item: (-len(Path(item[0]).parts), item[0]),
    ):
        directory = snapshot_root / relative
        if not directory.exists():
            continue
        try:
            with os.scandir(directory) as iterator:
                empty = next(iterator, None) is None
        except OSError as exc:
            raise StorageMaintenanceIntegrityError(
                f"Snapshot GC could not inspect a candidate directory: {exc.__class__.__name__}"
            ) from exc
        if not empty:
            continue
        durable_rmdir(
            directory,
            expected_device_id=int(card["device_id"]),
            expected_inode=int(card["inode"]),
        )
        deleted_directories += 1
    for card in payload["targets"]:
        target = snapshot_root / str(card["relative_path"])
        if target.exists() or target.is_symlink():
            raise StorageMaintenanceIntegrityError(
                f"Snapshot GC postcondition failed: {card['relative_path']}"
            )
    return {
        "deleted_files": deleted_files,
        "deleted_directories": deleted_directories,
    }


def run_source_snapshot_gc(
    db: Session,
    knowledge_base: KnowledgeBase,
    *,
    execute: bool = False,
    confirm_knowledge_base_id: str | None = None,
    confirm_inventory_hash: str | None = None,
    retention_seconds: int = SOURCE_SNAPSHOT_GC_DEFAULT_RETENTION_SECONDS,
    max_entries: int = 100_000,
) -> dict[str, Any]:
    inventory = build_source_snapshot_gc_inventory(
        db,
        knowledge_base,
        retention_seconds=retention_seconds,
        max_entries=max_entries,
    )
    if not execute:
        return {
            "dry_run": True,
            "executed": False,
            "inventory": inventory,
        }
    from app.services.ingestion_resource_lock import (
        active_ingestion_resource_lease,
    )

    lease = active_ingestion_resource_lease(knowledge_base.id)
    if lease is None or lease.operation != SOURCE_SNAPSHOT_GC_OPERATION:
        raise StorageMaintenanceIntegrityError(
            "Snapshot GC execute requires the knowledge-base resource lock"
        )
    if confirm_knowledge_base_id != knowledge_base.id:
        raise StorageMaintenanceIntegrityError(
            "Snapshot GC requires exact --confirm-knowledge-base-id"
        )
    if confirm_inventory_hash != inventory["inventory_hash"]:
        existing = db.scalar(
            select(StorageMaintenanceIntent).where(
                StorageMaintenanceIntent.operation
                == SOURCE_SNAPSHOT_GC_OPERATION,
                StorageMaintenanceIntent.scope_key
                == f"snapshot_gc:{knowledge_base.id}:{confirm_inventory_hash}",
                StorageMaintenanceIntent.status != "completed",
            )
        )
        if existing is None:
            raise StorageMaintenanceIntegrityError(
                "Snapshot GC inventory changed; rerun dry-run and confirm the new exact hash"
            )
        payload = _validate_snapshot_gc_payload(existing)
        row = existing
    else:
        targets = [
            dict(card)
            for card in inventory["files"]
            if card["gc_candidate"]
        ]
        scope_key = (
            f"snapshot_gc:{knowledge_base.id}:{inventory['inventory_hash']}"
        )
        row = db.scalar(
            select(StorageMaintenanceIntent).where(
                StorageMaintenanceIntent.operation
                == SOURCE_SNAPSHOT_GC_OPERATION,
                StorageMaintenanceIntent.scope_key == scope_key,
            )
        )
        if row is None:
            tree = inventory_storage_tree(
                Path(inventory["snapshot_root"]),
                authorized_parent=Path(inventory["snapshot_root"]).parent,
                max_entries=max_entries,
            )
            if tree["inventory_hash"] != inventory["tree_inventory_hash"]:
                raise StorageMaintenanceIntegrityError(
                    "Snapshot tree changed between inventory and durable intent commit"
                )
            payload = {
                "protocol_version": SOURCE_SNAPSHOT_GC_PROTOCOL_VERSION,
                "intent_id": hashlib.sha256(scope_key.encode("utf-8")).hexdigest()[:32],
                "knowledge_base_id": knowledge_base.id,
                "knowledge_base_name": knowledge_base.name,
                "inventory_hash": inventory["inventory_hash"],
                "inventory": inventory,
                "targets": targets,
                "tree_entries": tree["entries"],
                "phase": "intent_committed",
                "state_history": [],
                "state_history_total_count": 0,
            }
            intent_id = payload["intent_id"]
            intent_id = (
                f"{intent_id[:8]}-{intent_id[8:12]}-{intent_id[12:16]}-"
                f"{intent_id[16:20]}-{intent_id[20:32]}"
            )
            payload["intent_id"] = intent_id
            payload = _append_history(
                payload,
                status="intent_committed",
            )
            row = StorageMaintenanceIntent(
                id=intent_id,
                knowledge_base_id=knowledge_base.id,
                knowledge_base_name=knowledge_base.name,
                operation=SOURCE_SNAPSHOT_GC_OPERATION,
                protocol_version=SOURCE_SNAPSHOT_GC_PROTOCOL_VERSION,
                scope_key=scope_key,
                target_root=inventory["snapshot_root"],
                inventory_hash=inventory["inventory_hash"],
                payload_json=payload,
                status="intent_committed",
            )
            db.add(row)
            db.commit()
        else:
            payload = _validate_snapshot_gc_payload(row)

    try:
        payload["phase"] = "external_deleting"
        payload = _append_history(payload, status="external_deleting")
        row.payload_json = payload
        row.status = "external_deleting"
        db.commit()
        result = _apply_snapshot_gc_files(payload)
        payload["phase"] = "completed"
        payload = _append_history(
            payload,
            status="completed",
            details=result,
        )
        row.payload_json = payload
        row.status = "completed"
        row.error_message = None
        db.commit()
    except Exception as exc:
        db.rollback()
        row = db.get(StorageMaintenanceIntent, row.id)
        if row is not None:
            payload = _validate_snapshot_gc_payload(row)
            payload = _append_history(
                payload,
                status="manual_review"
                if isinstance(exc, StorageMaintenanceIntegrityError)
                else "external_deleting",
                details={"failure_type": exc.__class__.__name__},
            )
            row.payload_json = payload
            row.status = (
                "manual_review"
                if isinstance(exc, StorageMaintenanceIntegrityError)
                else "external_deleting"
            )
            row.error_message = exc.__class__.__name__
            db.commit()
        raise
    return {
        "dry_run": False,
        "executed": True,
        "intent_id": row.id,
        "inventory_hash": row.inventory_hash,
        **result,
    }


def applied_delete_intent_ids(
    db: Session,
    knowledge_base_id: str,
) -> list[str]:
    from app.services.qdrant_outbox import (
        pending_qdrant_delete_intents,
        qdrant_delete_intent_recovery_diagnostics,
    )

    intent_ids: list[str] = []
    for row in pending_qdrant_delete_intents(
        db,
        knowledge_base_id=knowledge_base_id,
    ):
        recovery = qdrant_delete_intent_recovery_diagnostics(row)
        if (
            recovery.get("delete_source_operation")
            != KNOWLEDGE_BASE_DELETE_QDRANT_REASON
            or row.status != "external_applied"
        ):
            raise StorageMaintenanceIntegrityError(
                "Knowledge-base facts cannot be deleted before every Qdrant "
                "delete tombstone is externally applied"
            )
        intent_ids.append(str(row.id))
    return sorted(intent_ids)


def finalize_knowledge_base_facts(
    db: Session,
    *,
    knowledge_base: KnowledgeBase,
    row: StorageMaintenanceIntent,
    payload: dict[str, Any],
    qdrant_delete_intent_ids: list[str],
) -> dict[str, Any]:
    from app.services.qdrant_outbox import mark_qdrant_delete_attempts_committed

    if payload.get("phase") != "external_applied":
        raise StorageMaintenanceIntegrityError(
            "Knowledge-base facts require a proven external_applied phase"
        )
    root = Path(str(payload["filesystem_inventory"]["root_path"]))
    if root.exists() or root.is_symlink():
        raise StorageMaintenanceIntegrityError(
            "Knowledge-base facts cannot be deleted while its storage root remains"
        )
    current = _observed_qdrant_scope(
        db,
        knowledge_base,
        include_pending_delete_targets=False,
    )
    if any(current.values()):
        raise StorageMaintenanceIntegrityError(
            "Knowledge-base facts cannot be deleted while Qdrant owned points remain"
        )

    mark_qdrant_delete_attempts_committed(
        db,
        intent_ids=qdrant_delete_intent_ids,
    )
    updated = dict(payload)
    updated["phase"] = "facts_deleted"
    updated = _append_history(
        updated,
        status="cache_invalidation_pending",
        details={
            "qdrant_delete_intent_count": len(qdrant_delete_intent_ids),
            "postgresql_cascade_committed": True,
        },
    )
    row.payload_json = updated
    row.status = "cache_invalidation_pending"
    row.error_message = None
    db.delete(knowledge_base)
    db.commit()
    return updated


def complete_knowledge_base_delete_cache_invalidation(
    db: Session,
    *,
    row: StorageMaintenanceIntent,
    payload: dict[str, Any],
) -> dict[str, Any]:
    from app.services.cache_manager import get_cache_manager

    get_cache_manager().invalidate_knowledge_base(
        row.knowledge_base_id,
        strict=True,
    )
    completed = dict(payload)
    completed["phase"] = "completed"
    completed = _append_history(
        completed,
        status="completed",
        details={"cache_invalidated": True},
    )
    row.payload_json = completed
    row.status = "completed"
    row.error_message = None
    db.commit()
    return completed


def storage_maintenance_recovery_health() -> dict[str, Any]:
    """Return the last bounded recovery result without querying external stores."""

    with _STORAGE_MAINTENANCE_RECOVERY_HEALTH_LOCK:
        return dict(_STORAGE_MAINTENANCE_RECOVERY_HEALTH)


def _publish_storage_maintenance_recovery_health(
    summary: dict[str, Any],
) -> dict[str, Any]:
    health = {
        "protocol_version": STORAGE_MAINTENANCE_RECOVERY_HEALTH_PROTOCOL_VERSION,
        "status": (
            "degraded"
            if int(summary.get("pending") or 0)
            or int(summary.get("manual_review") or 0)
            or int(summary.get("failed") or 0)
            else "healthy"
        ),
        "last_run_at": datetime.utcnow().isoformat(),
        "selected": int(summary.get("selected") or 0),
        "completed": int(summary.get("completed") or 0),
        "pending": int(summary.get("pending") or 0),
        "cache_pending": int(summary.get("cache_pending") or 0),
        "external_pending": int(summary.get("external_pending") or 0),
        "manual_review": int(summary.get("manual_review") or 0),
        "failed": int(summary.get("failed") or 0),
        "retryable": int(summary.get("manual_review") or 0) == 0,
    }
    with _STORAGE_MAINTENANCE_RECOVERY_HEALTH_LOCK:
        _STORAGE_MAINTENANCE_RECOVERY_HEALTH.clear()
        _STORAGE_MAINTENANCE_RECOVERY_HEALTH.update(health)
    return dict(health)


def _mark_cache_invalidation_recovery_failure(
    db: Session,
    *,
    row_id: str,
    error: BaseException,
) -> str:
    db.rollback()
    row = db.scalar(
        select(StorageMaintenanceIntent)
        .where(StorageMaintenanceIntent.id == row_id)
        .with_for_update()
    )
    if row is None:
        raise StorageMaintenanceIntegrityError(
            "Storage maintenance recovery tombstone disappeared"
        ) from error
    try:
        payload = _validate_delete_intent(
            row,
            knowledge_base_id=row.knowledge_base_id,
        )
    except StorageMaintenanceIntegrityError:
        # A corrupt tombstone cannot safely be rewritten because doing so would
        # erase the evidence needed for manual diagnosis.
        row.status = "manual_review"
        row.error_message = "StorageMaintenanceIntegrityError"
        db.commit()
        return "manual_review"
    manual_review = isinstance(error, StorageMaintenanceIntegrityError)
    next_status = "manual_review" if manual_review else "cache_invalidation_pending"
    row.payload_json = _append_history(
        payload,
        status=next_status,
        details={"failure_type": error.__class__.__name__},
    )
    row.status = next_status
    row.error_message = error.__class__.__name__
    db.commit()
    return next_status


def _reconcile_pending_storage_maintenance_sync() -> dict[str, Any]:
    """Finish cache invalidation after a committed full-KB fact deletion.

    The tombstone intentionally has no KnowledgeBase foreign key, so this
    recovery path remains callable after the cascade removed the KB row.
    """

    from app.db import SessionLocal

    with SessionLocal() as discovery:
        bind = discovery.get_bind()
        if str(bind.dialect.name) == "sqlite" and not inspect(bind).has_table(
            StorageMaintenanceIntent.__tablename__
        ):
            summary = {
                "selected": 0,
                "completed": 0,
                "pending": 0,
                "cache_pending": 0,
                "external_pending": 0,
                "manual_review": 0,
                "failed": 0,
                "results": [],
            }
            return {
                **summary,
                **_publish_storage_maintenance_recovery_health(summary),
            }
        row_ids = list(
            discovery.scalars(
                select(StorageMaintenanceIntent.id)
                .where(
                    StorageMaintenanceIntent.operation
                    == KNOWLEDGE_BASE_DELETE_OPERATION,
                    StorageMaintenanceIntent.status.in_(
                        ["facts_deleted", "cache_invalidation_pending"]
                    ),
                )
                .order_by(StorageMaintenanceIntent.created_at.asc())
            ).all()
        )

    results: list[dict[str, Any]] = []
    for row_id in row_ids:
        with SessionLocal() as db:
            try:
                row = db.scalar(
                    select(StorageMaintenanceIntent)
                    .where(StorageMaintenanceIntent.id == row_id)
                    .with_for_update()
                )
                if row is None or row.status == "completed":
                    continue
                payload = _validate_delete_intent(
                    row,
                    knowledge_base_id=row.knowledge_base_id,
                )
                if payload.get("phase") != "facts_deleted":
                    raise StorageMaintenanceIntegrityError(
                        "Cache invalidation recovery requires the proven facts_deleted phase"
                    )
                if db.get(KnowledgeBase, row.knowledge_base_id) is not None:
                    raise StorageMaintenanceIntegrityError(
                        "Cache invalidation recovery found undeleted KnowledgeBase facts"
                    )
                target_root = Path(str(payload["filesystem_inventory"]["root_path"]))
                if target_root.exists() or target_root.is_symlink():
                    raise StorageMaintenanceIntegrityError(
                        "Cache invalidation recovery found a remaining storage root"
                    )
                complete_knowledge_base_delete_cache_invalidation(
                    db,
                    row=row,
                    payload=payload,
                )
                results.append(
                    {
                        "intent_id": str(row_id),
                        "knowledge_base_id": str(row.knowledge_base_id),
                        "status": "completed",
                    }
                )
            except Exception as exc:
                status = _mark_cache_invalidation_recovery_failure(
                    db,
                    row_id=str(row_id),
                    error=exc,
                )
                results.append(
                    {
                        "intent_id": str(row_id),
                        "status": status,
                        "error_type": exc.__class__.__name__,
                    }
                )

    with SessionLocal() as final:
        cache_pending = int(
            final.scalar(
                select(func.count(StorageMaintenanceIntent.id)).where(
                    StorageMaintenanceIntent.operation
                    == KNOWLEDGE_BASE_DELETE_OPERATION,
                    StorageMaintenanceIntent.status.in_(
                        ["facts_deleted", "cache_invalidation_pending"]
                    ),
                )
            )
            or 0
        )
        external_pending = int(
            final.scalar(
                select(func.count(StorageMaintenanceIntent.id)).where(
                    StorageMaintenanceIntent.operation
                    == KNOWLEDGE_BASE_DELETE_OPERATION,
                    StorageMaintenanceIntent.status.in_(
                        [
                            "intent_committed",
                            "external_deleting",
                            "external_applied",
                        ]
                    ),
                )
            )
            or 0
        )
        manual_review = int(
            final.scalar(
                select(func.count(StorageMaintenanceIntent.id)).where(
                    StorageMaintenanceIntent.operation
                    == KNOWLEDGE_BASE_DELETE_OPERATION,
                    StorageMaintenanceIntent.status == "manual_review",
                )
            )
            or 0
        )
    summary = {
        "selected": len(row_ids),
        "completed": sum(
            1 for item in results if item.get("status") == "completed"
        ),
        "pending": cache_pending + external_pending,
        "cache_pending": cache_pending,
        "external_pending": external_pending,
        "manual_review": manual_review,
        "failed": sum(
            1
            for item in results
            if item.get("status") == "cache_invalidation_pending"
        ),
        "results": results,
    }
    return {
        **summary,
        **_publish_storage_maintenance_recovery_health(summary),
    }


async def reconcile_pending_storage_maintenance_startup() -> dict[str, Any]:
    from app.services.storage import run_bounded_source_io

    return await run_bounded_source_io(
        _reconcile_pending_storage_maintenance_sync
    )

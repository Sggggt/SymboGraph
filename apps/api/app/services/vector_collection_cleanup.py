from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session

from app.models import (
    IngestionCompensationLog,
    KnowledgeBaseVectorRuntimeState,
    VectorRecord,
    VectorShadowBuild,
)
from app.services.qdrant_outbox import (
    QDRANT_AUTHORITATIVE_VECTOR_STATUSES,
    QDRANT_OUTBOX_ACTIVE_STATES,
    QDRANT_UPSERT_OPERATION,
)
from app.services.vector_store import VectorStore


VECTOR_COLLECTION_CLEANUP_PROTOCOL_VERSION = "vector_collection_exact_cleanup_v1"
VECTOR_COLLECTION_CLEANUP_OPERATION = "qdrant_collection_exact_delete"
VECTOR_COLLECTION_CLEANUP_PENDING_STATUS = "collection_delete_pending"
VECTOR_COLLECTION_CLEANUP_MAX_INTENT_SCAN = 4096
VECTOR_COLLECTION_CLEANUP_MAX_REFERENCE_SCAN = 4096
VECTOR_COLLECTION_LIVE_BUILD_STATUSES = {
    "staged",
    "building",
    "shadow_ready",
    "evaluating",
    "evaluation_passed",
    "promotion_blocked",
    "promotion_pending",
}
# Retained rollback/shadow vectors remain authoritative for automatic
# outbox/reconcile behavior, but an operator's exact-name destructive intent is
# the explicit protocol for relinquishing that retention.  A live shadow build
# is blocked separately; therefore only a serving ``ready`` record is always
# protected even if its active pointer is corrupt or missing.
VECTOR_COLLECTION_CLEANUP_PROTECTED_VECTOR_STATUSES = {"ready"}
VECTOR_COLLECTION_CLEANUP_RELEASABLE_RETAINED_STATUSES = {
    "shadow_ready",
    "rollback_retained",
    "rolled_back_retained",
}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _validate_exact_collection_name(collection_name: str) -> str:
    exact = str(collection_name or "").strip()
    if (
        not exact
        or len(exact) > 255
        or re.fullmatch(r"[A-Za-z0-9_-]+", exact) is None
    ):
        raise ValueError(
            "collection_name must be one exact Qdrant collection name without wildcards"
        )
    return exact


def vector_collection_lifecycle_lock(db: Session, collection_name: str) -> None:
    """Serialize stage/cleanup decisions for one exact collection identity."""

    exact = _validate_exact_collection_name(collection_name)
    dialect = db.get_bind().dialect.name
    if dialect == "sqlite":
        return
    if dialect != "postgresql":
        raise RuntimeError("Vector collection lifecycle locking requires PostgreSQL")
    lock_key = int.from_bytes(
        hashlib.sha256(
            f"vector-collection-lifecycle:{exact}".encode("utf-8")
        ).digest()[:8],
        "big",
        signed=True,
    )
    db.execute(text("SELECT pg_advisory_xact_lock(:lock_key)"), {"lock_key": lock_key})


def _pending_cleanup_intents(
    db: Session,
    collection_name: str,
    *,
    for_update: bool = False,
) -> list[IngestionCompensationLog]:
    statement = (
        select(IngestionCompensationLog)
        .where(
            IngestionCompensationLog.operation
            == VECTOR_COLLECTION_CLEANUP_OPERATION,
            IngestionCompensationLog.status
            == VECTOR_COLLECTION_CLEANUP_PENDING_STATUS,
            IngestionCompensationLog.payload_json["collection_name"].as_string()
            == collection_name,
        )
        .order_by(IngestionCompensationLog.created_at.asc())
        .limit(3)
    )
    if for_update:
        statement = statement.with_for_update()
    return list(db.scalars(statement).all())


def assert_vector_collection_not_pending_cleanup(
    db: Session,
    collection_name: str,
) -> None:
    exact = _validate_exact_collection_name(collection_name)
    vector_collection_lifecycle_lock(db, exact)
    pending = _pending_cleanup_intents(db, exact, for_update=True)
    if pending:
        raise RuntimeError(
            f"Vector collection {exact!r} has a durable destructive cleanup intent; "
            "finish or explicitly resolve it before staging a rebuild"
        )


def vector_collection_cleanup_plan(
    db: Session,
    *,
    collection_name: str,
    vector_store: Any | None = None,
    check_qdrant: bool = True,
) -> dict[str, Any]:
    """Return a read-only exact-target inventory; never infer orphan deletion."""

    exact = _validate_exact_collection_name(collection_name)
    pointers = list(
        db.scalars(
            select(KnowledgeBaseVectorRuntimeState)
            .where(KnowledgeBaseVectorRuntimeState.collection_name == exact)
            .order_by(KnowledgeBaseVectorRuntimeState.knowledge_base_id.asc())
            .limit(VECTOR_COLLECTION_CLEANUP_MAX_REFERENCE_SCAN + 1)
        ).all()
    )
    rollback_pointers = list(
        db.scalars(
            select(KnowledgeBaseVectorRuntimeState)
            .where(
                KnowledgeBaseVectorRuntimeState.previous_state_json["vector_schema"]
                ["collection_name"]
                .as_string()
                == exact
            )
            .order_by(KnowledgeBaseVectorRuntimeState.knowledge_base_id.asc())
            .limit(VECTOR_COLLECTION_CLEANUP_MAX_REFERENCE_SCAN + 1)
        ).all()
    )
    live_builds = list(
        db.scalars(
            select(VectorShadowBuild)
            .where(
                VectorShadowBuild.collection_name == exact,
                VectorShadowBuild.status.in_(VECTOR_COLLECTION_LIVE_BUILD_STATUSES),
            )
            .order_by(VectorShadowBuild.knowledge_base_id.asc(), VectorShadowBuild.id.asc())
            .limit(VECTOR_COLLECTION_CLEANUP_MAX_REFERENCE_SCAN + 1)
        ).all()
    )
    if (
        len(pointers) > VECTOR_COLLECTION_CLEANUP_MAX_REFERENCE_SCAN
        or len(rollback_pointers) > VECTOR_COLLECTION_CLEANUP_MAX_REFERENCE_SCAN
        or len(live_builds) > VECTOR_COLLECTION_CLEANUP_MAX_REFERENCE_SCAN
    ):
        raise RuntimeError("Vector collection cleanup refused an unbounded reference scan")
    record_owner_rows = list(
        db.execute(
            select(
                VectorRecord.knowledge_base_id,
                VectorRecord.vector_status,
                func.count(VectorRecord.id),
            )
            .where(VectorRecord.collection_name == exact)
            .group_by(VectorRecord.knowledge_base_id, VectorRecord.vector_status)
            .order_by(
                VectorRecord.knowledge_base_id.asc(),
                VectorRecord.vector_status.asc(),
            )
            .limit(VECTOR_COLLECTION_CLEANUP_MAX_REFERENCE_SCAN + 1)
        ).all()
    )
    if len(record_owner_rows) > VECTOR_COLLECTION_CLEANUP_MAX_REFERENCE_SCAN:
        raise RuntimeError("Vector collection cleanup refused an unbounded record-owner scan")
    record_status_counts: dict[str, int] = {}
    record_owner_cards: list[dict[str, Any]] = []
    for knowledge_base_id, status, count in record_owner_rows:
        normalized_status = str(status)
        normalized_count = int(count)
        record_status_counts[normalized_status] = (
            record_status_counts.get(normalized_status, 0) + normalized_count
        )
        record_owner_cards.append(
            {
                "knowledge_base_id": str(knowledge_base_id),
                "vector_status": normalized_status,
                "record_count": normalized_count,
            }
        )
    authoritative_record_count = sum(
        count
        for status, count in record_status_counts.items()
        if status in QDRANT_AUTHORITATIVE_VECTOR_STATUSES
    )
    protected_record_count = sum(
        count
        for status, count in record_status_counts.items()
        if status in VECTOR_COLLECTION_CLEANUP_PROTECTED_VECTOR_STATUSES
    )
    releasable_retained_record_count = sum(
        count
        for status, count in record_status_counts.items()
        if status in VECTOR_COLLECTION_CLEANUP_RELEASABLE_RETAINED_STATUSES
    )
    active_outbox_rows = list(
        db.scalars(
            select(IngestionCompensationLog)
            .where(
                IngestionCompensationLog.operation == QDRANT_UPSERT_OPERATION,
                IngestionCompensationLog.status.in_(QDRANT_OUTBOX_ACTIVE_STATES),
                IngestionCompensationLog.payload_json["collection_name"].as_string()
                == exact,
            )
            .order_by(IngestionCompensationLog.created_at.asc())
            .limit(VECTOR_COLLECTION_CLEANUP_MAX_INTENT_SCAN + 1)
        ).all()
    )
    if len(active_outbox_rows) > VECTOR_COLLECTION_CLEANUP_MAX_INTENT_SCAN:
        raise RuntimeError(
            "Vector collection cleanup refused an unbounded active outbox scan"
        )
    matching_outbox = active_outbox_rows
    pending_cleanup = _pending_cleanup_intents(db, exact)
    blockers: list[str] = []
    if pointers:
        blockers.append("active_vector_runtime_pointer_references_collection")
    if live_builds:
        blockers.append("live_vector_shadow_build_references_collection")
    if protected_record_count:
        blockers.append("ready_vector_records_reference_collection")
    if matching_outbox:
        blockers.append("active_qdrant_outbox_intent_references_collection")

    qdrant_exists: bool | None = None
    if check_qdrant:
        store = vector_store or VectorStore(collection_name=exact)
        qdrant_exists = bool(store.collection_exists())
    rollback_pointer_cards = [
        {
            "pointer_id": pointer.id,
            "knowledge_base_id": str(pointer.knowledge_base_id),
            "active_candidate_id": pointer.runtime_settings_candidate_id,
            "active_pointer_state_hash": pointer.state_hash,
            "rollback_recovery_will_be_relinquished": True,
        }
        for pointer in rollback_pointers
    ]
    fact_card = {
        "protocol_version": VECTOR_COLLECTION_CLEANUP_PROTOCOL_VERSION,
        "collection_name": exact,
        "active_pointer_ids": sorted(pointer.id for pointer in pointers),
        "rollback_pointer_ids": sorted(pointer.id for pointer in rollback_pointers),
        "rollback_pointer_cards": rollback_pointer_cards,
        "live_build_ids": sorted(build.id for build in live_builds),
        "record_status_counts": record_status_counts,
        "record_owner_cards": record_owner_cards,
        "active_outbox_intent_ids": sorted(row.id for row in matching_outbox),
    }
    safety_fact_card = {
        "protocol_version": VECTOR_COLLECTION_CLEANUP_PROTOCOL_VERSION,
        "collection_name": exact,
        "active_pointer_ids": fact_card["active_pointer_ids"],
        "rollback_pointer_cards": rollback_pointer_cards,
        "live_build_ids": fact_card["live_build_ids"],
        "active_outbox_intent_ids": fact_card["active_outbox_intent_ids"],
        "protected_record_count": protected_record_count,
    }
    return {
        **fact_card,
        "plan_hash": _canonical_hash(fact_card),
        "safety_fact_hash": _canonical_hash(safety_fact_card),
        "dry_run": True,
        "allowed": not blockers,
        "blockers": sorted(blockers),
        "qdrant_collection_exists": qdrant_exists,
        "record_count": sum(record_status_counts.values()),
        "authoritative_record_count": authoritative_record_count,
        "protected_record_count": protected_record_count,
        "releasable_retained_record_count": releasable_retained_record_count,
        "releasable_retained_statuses": sorted(
            VECTOR_COLLECTION_CLEANUP_RELEASABLE_RETAINED_STATUSES
        ),
        "active_pointer_knowledge_base_ids": sorted(
            str(pointer.knowledge_base_id) for pointer in pointers
        ),
        "rollback_pointer_cards": rollback_pointer_cards,
        "live_build_cards": [
            {
                "id": build.id,
                "knowledge_base_id": build.knowledge_base_id,
                "status": build.status,
                "candidate_vector_schema_hash": build.candidate_vector_schema_hash,
            }
            for build in live_builds
        ],
        "pending_cleanup_intent_ids": [row.id for row in pending_cleanup],
        "deletion_inferred": False,
        "recoverability": "derived_collection_rebuild_required_after_delete",
    }


def prepare_vector_collection_cleanup(
    db: Session,
    *,
    audit_knowledge_base_id: str,
    collection_name: str,
    confirmed_collection_name: str,
    allow_sqlite_test_adapter: bool = False,
) -> IngestionCompensationLog:
    """Persist the exact destructive intent; caller must commit before Qdrant I/O."""

    exact = _validate_exact_collection_name(collection_name)
    if _validate_exact_collection_name(confirmed_collection_name) != exact:
        raise ValueError("Destructive confirmation must exactly repeat collection_name")
    dialect = db.get_bind().dialect.name
    if dialect != "postgresql" and not (
        dialect == "sqlite" and allow_sqlite_test_adapter
    ):
        raise RuntimeError("Vector collection cleanup execution requires PostgreSQL")
    vector_collection_lifecycle_lock(db, exact)
    existing = _pending_cleanup_intents(db, exact, for_update=True)
    if len(existing) > 1:
        raise RuntimeError("Multiple pending cleanup intents exist for one collection")
    if existing:
        return existing[0]
    plan = vector_collection_cleanup_plan(
        db,
        collection_name=exact,
        check_qdrant=False,
    )
    if not plan["allowed"]:
        raise RuntimeError(
            "Exact collection cleanup is blocked: " + ", ".join(plan["blockers"])
        )
    created_at = datetime.utcnow().isoformat()
    payload = {
        "protocol_version": VECTOR_COLLECTION_CLEANUP_PROTOCOL_VERSION,
        "collection_name": exact,
        "confirmed_collection_name": exact,
        "plan_hash": plan["plan_hash"],
        "safety_fact_hash": plan["safety_fact_hash"],
        "record_status_counts": plan["record_status_counts"],
        "phase": "database_intent_pending",
        "created_at": created_at,
        "recovery": "retry_exact_collection_delete_then_finalize_postgresql",
    }
    intent = IngestionCompensationLog(
        knowledge_base_id=audit_knowledge_base_id,
        operation=VECTOR_COLLECTION_CLEANUP_OPERATION,
        target_ids_json=[exact],
        payload_json={**payload, "payload_hash": _canonical_hash(payload)},
        status=VECTOR_COLLECTION_CLEANUP_PENDING_STATUS,
    )
    db.add(intent)
    db.flush()
    return intent


def execute_vector_collection_cleanup(
    db: Session,
    *,
    intent_id: str,
    vector_store: Any | None = None,
    allow_sqlite_test_adapter: bool = False,
) -> dict[str, Any]:
    """Execute/retry one previously committed exact collection delete intent."""

    dialect = db.get_bind().dialect.name
    if dialect != "postgresql" and not (
        dialect == "sqlite" and allow_sqlite_test_adapter
    ):
        raise RuntimeError("Vector collection cleanup execution requires PostgreSQL")
    # Resolve the collection before taking locks, then acquire locks in the
    # same collection -> row order used by preparation/staging.  Re-read the
    # intent under FOR UPDATE after the advisory lock so concurrent retries
    # cannot create an advisory-lock/row-lock inversion.
    intent_preview = db.get(IngestionCompensationLog, intent_id)
    if intent_preview is None:
        raise ValueError(f"Unknown vector collection cleanup intent: {intent_id}")
    if intent_preview.operation != VECTOR_COLLECTION_CLEANUP_OPERATION:
        raise RuntimeError("Intent is not an exact vector collection cleanup")
    preview_collection = _validate_exact_collection_name(
        (intent_preview.payload_json or {}).get("collection_name")
    )
    vector_collection_lifecycle_lock(db, preview_collection)
    intent = db.scalar(
        select(IngestionCompensationLog)
        .where(IngestionCompensationLog.id == intent_id)
        .with_for_update()
    )
    if intent is None:
        raise RuntimeError("Vector collection cleanup intent disappeared after locking")
    if intent.operation != VECTOR_COLLECTION_CLEANUP_OPERATION:
        raise RuntimeError("Intent operation drifted after locking")
    if intent.status == "committed":
        return {
            "intent_id": intent.id,
            "collection_name": (intent.payload_json or {}).get("collection_name"),
            "committed": True,
            "idempotent_replay": True,
        }
    if intent.status != VECTOR_COLLECTION_CLEANUP_PENDING_STATUS:
        raise RuntimeError(f"Cleanup intent {intent.id} is {intent.status}")
    payload = dict(intent.payload_json or {})
    stored_payload_hash = str(payload.pop("payload_hash", ""))
    if (
        payload.get("protocol_version")
        != VECTOR_COLLECTION_CLEANUP_PROTOCOL_VERSION
        or stored_payload_hash != _canonical_hash(payload)
    ):
        raise RuntimeError("Vector collection cleanup intent payload hash is invalid")
    exact = _validate_exact_collection_name(payload.get("collection_name"))
    if exact != preview_collection:
        raise RuntimeError("Vector collection cleanup target drifted before locking")
    if payload.get("confirmed_collection_name") != exact:
        raise RuntimeError("Vector collection cleanup intent confirmation drifted")
    plan = vector_collection_cleanup_plan(
        db,
        collection_name=exact,
        check_qdrant=False,
    )
    if not plan["allowed"]:
        raise RuntimeError(
            "Exact collection cleanup became blocked: " + ", ".join(plan["blockers"])
        )

    store = vector_store or VectorStore(collection_name=exact)
    plan_hash_changed = plan["plan_hash"] != payload.get("plan_hash")
    recovered_after_external_apply = False
    if plan_hash_changed:
        # A crash after Qdrant applied the delete can allow read-only reconcile
        # to change retained record statuses before PostgreSQL finalization.
        # Permit that record-only drift only after an exact absence probe and
        # only when all pointer/build/outbox/protected-record safety facts are
        # unchanged. Never use this path while the collection still exists.
        if bool(store.collection_exists()):
            raise RuntimeError(
                "Vector collection cleanup PostgreSQL facts changed before external delete"
            )
        if plan["safety_fact_hash"] != payload.get("safety_fact_hash"):
            raise RuntimeError(
                "Vector collection cleanup safety facts changed after external delete"
            )
        recovered_after_external_apply = True
    deletion = store.delete_collection_exact(exact)
    if deletion.get("verified_absent") is not True:
        raise RuntimeError("Qdrant collection deletion did not verify exact absence")
    result = db.execute(
        update(VectorRecord)
        .where(VectorRecord.collection_name == exact)
        .values(vector_status="missing")
    )
    completed_at = datetime.utcnow().isoformat()
    completed_payload = {
        **payload,
        "phase": "completed",
        "completed_at": completed_at,
        "qdrant_result": deletion,
        "vector_records_marked_missing": int(result.rowcount or 0),
        "recoverability": "Qdrant is derived; rebuild the exact canonical vector schema",
        "recovered_after_external_apply": recovered_after_external_apply,
    }
    intent.payload_json = {
        **completed_payload,
        "payload_hash": _canonical_hash(completed_payload),
    }
    intent.status = "committed"
    intent.error_message = None
    db.flush()
    return {
        "intent_id": intent.id,
        "collection_name": exact,
        "committed": True,
        "idempotent_replay": False,
        "qdrant_result": deletion,
        "vector_records_marked_missing": int(result.rowcount or 0),
        "recovered_after_external_apply": recovered_after_external_apply,
    }

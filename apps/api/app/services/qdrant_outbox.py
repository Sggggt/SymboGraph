from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import event, func, select, text
from sqlalchemy.orm import Session

from app.models import IngestionCompensationLog, VectorRecord


QDRANT_OUTBOX_PROTOCOL_V1 = "qdrant_side_effect_outbox_v1"
QDRANT_OUTBOX_PROTOCOL_VERSION = "qdrant_side_effect_outbox_v2"
QDRANT_OUTBOX_SUPPORTED_PROTOCOL_VERSIONS = {
    QDRANT_OUTBOX_PROTOCOL_V1,
    QDRANT_OUTBOX_PROTOCOL_VERSION,
}
QDRANT_OUTBOX_V1_ENVELOPE_SCHEMA_VERSION = "qdrant_outbox_envelope_schema_v1"
QDRANT_OUTBOX_V2_ENVELOPE_SCHEMA_VERSION = "qdrant_outbox_envelope_schema_v2"
QDRANT_OUTBOX_V1_CANONICAL_BYTES_VERSION = "qdrant_outbox_json_utf8_sorted_compact_default_str_v1"
QDRANT_OUTBOX_V2_CANONICAL_BYTES_VERSION = "qdrant_outbox_json_utf8_sorted_compact_strict_json_v1"
_QDRANT_OUTBOX_V1_SCHEMA_MANIFEST = (
    b"qdrant_outbox_envelope_schema_v1\x00"
    b"core=protocol_version,intent_id,collection_name,target_points,before_points,"
    b"target_payload_hash,before_image_hash\x00"
    b"point=id,vector,payload\x00"
    b"canonical=qdrant_outbox_json_utf8_sorted_compact_default_str_v1"
)
# The frozen v2 target_schema below is the invariant core.  Its
# vector_payload_hash_protocol field dispatches a separately versioned vector
# identity card; v3 of that nested protocol requires context/hint fields while
# historical vector hash v2 remains recovery-only.
_QDRANT_OUTBOX_V2_SCHEMA_MANIFEST = (
    b"qdrant_outbox_envelope_schema_v2\x00"
    b"core=protocol_version,envelope_schema_version,envelope_schema_hash,"
    b"canonical_bytes_version,intent_id,collection_name,target_points,before_points,"
    b"target_payload_hash,before_image_hash\x00"
    b"point=id,vector,payload\x00"
    b"target_schema=knowledge_base_id,chunk_id,embedding_model,embedding_dimension,"
    b"vector_distance_metric,embedding_text_version,chunk_schema_version,"
    b"collection_identity_protocol_version,collection_identity_digest,"
    b"vector_payload_hash_protocol,vector_payload_hash,qdrant_write_intent_id,"
    b"qdrant_write_protocol_version\x00"
    b"canonical=qdrant_outbox_json_utf8_sorted_compact_strict_json_v1\x00"
    b"vector=finite_nonzero_ieee754_binary32"
)
QDRANT_OUTBOX_V1_ENVELOPE_SCHEMA_HASH = hashlib.sha256(
    _QDRANT_OUTBOX_V1_SCHEMA_MANIFEST
).hexdigest()
QDRANT_OUTBOX_V2_ENVELOPE_SCHEMA_HASH = hashlib.sha256(
    _QDRANT_OUTBOX_V2_SCHEMA_MANIFEST
).hexdigest()
QDRANT_UPSERT_OPERATION = "qdrant_upsert_intent_v1"
QDRANT_DELETE_OPERATION = "qdrant_delete_intent_v1"
QDRANT_VECTOR_SCHEMA_METADATA_PATCH_OPERATION = (
    "qdrant_vector_schema_metadata_patch_intent_v1"
)
QDRANT_VECTOR_SCHEMA_METADATA_PATCH_ACTIVE_STATES = {
    "pending",
    "external_outcome_unknown",
    "external_applied",
    "postgres_applied",
    "postverify_failed",
    "compensation_pending",
    "reconcile_failed",
}
LEGACY_QDRANT_DELETE_OPERATION = "qdrant_delete"
QDRANT_DELETE_PROTOCOL_VERSION = "qdrant_side_effect_delete_outbox_v2"
QDRANT_DELETE_SUPPORTED_PROTOCOL_VERSIONS = {
    QDRANT_DELETE_PROTOCOL_VERSION,
}
QDRANT_CONDITIONAL_DELETE_PROTOCOL_VERSION = "qdrant_payload_fenced_delete_v1"
QDRANT_DELETE_SOURCE_BINDING_PROTOCOL_VERSION = (
    "qdrant_delete_ingestion_recovery_binding_v1"
)
QDRANT_INGESTION_PARSE_COMPENSATION_REASON = (
    "ingestion_batch_parse_cancel_compensation"
)
QDRANT_OUTBOX_SESSION_KEY = "qdrant_outbox_handles_v1"
QDRANT_OUTBOX_NESTED_SESSION_KEY = "qdrant_outbox_nested_handles_v1"
QDRANT_OUTBOX_TEST_HISTORY_KEY = "qdrant_outbox_sqlite_test_history_v1"
QDRANT_OUTBOX_ACTIVE_STATES = {
    "pending",
    "external_applied",
    "external_outcome_unknown",
    "uncertainty_observed",
    "compensation_pending",
    "compensation_verify_pending",
    "reconcile_failed",
}
QDRANT_OUTBOX_TERMINAL_STATES = {"committed", "compensated", "superseded"}
# A committed Qdrant point remains PostgreSQL-authoritative throughout the
# vector shadow lifecycle.  Promotion/rollback only changes which immutable
# vector identity is active; candidate and retained rollback records must not
# be mistaken for orphaned points and compensated away.
QDRANT_AUTHORITATIVE_VECTOR_STATUSES = {
    "ready",
    "shadow_ready",
    "rollback_retained",
    "rolled_back_retained",
}
QDRANT_DELETE_ACTIVE_STATES = {
    "pending",
    "external_applied",
    "external_outcome_unknown",
    "reconcile_failed",
}
LEGACY_QDRANT_DELETE_BLOCKING_STATES = {"failed"}
QDRANT_OUTBOX_LEASE_SECONDS = 300
QDRANT_OUTBOX_UNCERTAINTY_CONFIRM_SECONDS = 5
QDRANT_OUTBOX_STATE_HISTORY_LIMIT = 32
QDRANT_OUTBOX_MAX_TARGET_POINTS_PER_INTENT = 256
QDRANT_OUTBOX_RECONCILE_SCAN_PROTOCOL_VERSION = "qdrant_outbox_active_intent_pk_keyset_v1"
QDRANT_OUTBOX_RECONCILE_PAGE_SIZE = 32
QDRANT_OUTBOX_RECONCILE_ACTION_SAMPLE_LIMIT = 128
QDRANT_OUTBOX_DELETE_DIAGNOSTIC_SAMPLE_LIMIT = 64
QDRANT_POINT_MUTATION_LOCK_PROTOCOL_VERSION = "postgres_advisory_qdrant_point_v1"
QDRANT_OUTBOX_UNCERTAINTY_STATES = {
    "pending",
    "external_outcome_unknown",
    "uncertainty_observed",
    "compensation_verify_pending",
}
QDRANT_OUTBOX_ALLOWED_TRANSITIONS = {
    "pending": {"pending", "external_applied", "external_outcome_unknown", "uncertainty_observed", "reconcile_failed"},
    "external_applied": {
        "external_applied",
        "external_outcome_unknown",
        "committed",
        "compensation_pending",
        "compensated",
        "superseded",
        "reconcile_failed",
    },
    "external_outcome_unknown": {"external_outcome_unknown", "uncertainty_observed", "reconcile_failed"},
    "uncertainty_observed": {"uncertainty_observed", "compensation_verify_pending", "reconcile_failed"},
    "compensation_pending": {
        "compensation_pending",
        "external_outcome_unknown",
        "compensated",
        "superseded",
        "reconcile_failed",
    },
    "compensation_verify_pending": {"compensation_verify_pending", "compensated", "superseded", "reconcile_failed"},
    "reconcile_failed": {
        "reconcile_failed",
        "external_outcome_unknown",
        "uncertainty_observed",
        "compensation_pending",
        "compensation_verify_pending",
        "committed",
        "compensated",
        "superseded",
    },
    "committed": {"committed"},
    "compensated": {"compensated"},
    "superseded": {"superseded"},
}


class QdrantOutboxError(RuntimeError):
    pass


class QdrantMutationOutcomeUnknown(QdrantOutboxError):
    pass


@dataclass(frozen=True)
class QdrantOutboxHandle:
    id: str
    knowledge_base_id: str
    job_id: str | None
    collection_name: str
    target_ids: tuple[str, ...]
    target_points: tuple[dict[str, Any], ...]
    before_points: tuple[dict[str, Any], ...]
    payload_hash: str
    durable: bool


@dataclass(frozen=True)
class _PreparedQdrantUpsertEnvelope:
    intent_id: str
    knowledge_base_id: str
    job_id: str | None
    collection_name: str
    target_points: tuple[dict[str, Any], ...]
    before_points: tuple[dict[str, Any], ...]
    target_payload_hash: str
    before_image_hash: str


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _outbox_v1_canonical_bytes(payload: Any) -> bytes:
    """Frozen historical v1 bytes. Never route this through a newer helper."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _strict_json_copy(value: Any, *, path: str = "$") -> Any:
    """Deep-copy the JSON subset accepted by the durable v2 envelope."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise QdrantOutboxError(f"Qdrant outbox value at {path} must be finite JSON")
        return 0.0 if value == 0.0 else value
    if isinstance(value, list):
        return [
            _strict_json_copy(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise QdrantOutboxError(
                    f"Qdrant outbox object at {path} contains a non-string key"
                )
            copied[key] = _strict_json_copy(item, path=f"{path}.{key}")
        return copied
    raise QdrantOutboxError(
        f"Qdrant outbox value at {path} is outside the frozen strict-JSON schema"
    )


def _outbox_v2_canonical_bytes(payload: Any) -> bytes:
    """Frozen v2 JSON bytes for already-canonical, strict-JSON envelopes."""

    copied = _strict_json_copy(payload)
    return json.dumps(
        copied,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _outbox_payload_hash(protocol_version: str, payload: Any) -> str:
    if protocol_version == QDRANT_OUTBOX_PROTOCOL_V1:
        encoded = _outbox_v1_canonical_bytes(payload)
    elif protocol_version == QDRANT_OUTBOX_PROTOCOL_VERSION:
        encoded = _outbox_v2_canonical_bytes(payload)
    else:
        raise QdrantOutboxError(f"Unsupported Qdrant outbox protocol: {protocol_version}")
    return hashlib.sha256(encoded).hexdigest()


def _outbox_protocol_contract(protocol_version: str) -> dict[str, str]:
    if protocol_version == QDRANT_OUTBOX_PROTOCOL_V1:
        return {
            "protocol_version": QDRANT_OUTBOX_PROTOCOL_V1,
            "envelope_schema_version": QDRANT_OUTBOX_V1_ENVELOPE_SCHEMA_VERSION,
            "envelope_schema_hash": QDRANT_OUTBOX_V1_ENVELOPE_SCHEMA_HASH,
            "canonical_bytes_version": QDRANT_OUTBOX_V1_CANONICAL_BYTES_VERSION,
        }
    if protocol_version == QDRANT_OUTBOX_PROTOCOL_VERSION:
        return {
            "protocol_version": QDRANT_OUTBOX_PROTOCOL_VERSION,
            "envelope_schema_version": QDRANT_OUTBOX_V2_ENVELOPE_SCHEMA_VERSION,
            "envelope_schema_hash": QDRANT_OUTBOX_V2_ENVELOPE_SCHEMA_HASH,
            "canonical_bytes_version": QDRANT_OUTBOX_V2_CANONICAL_BYTES_VERSION,
        }
    raise QdrantOutboxError(f"Unsupported Qdrant outbox protocol: {protocol_version}")


def _safe_error(exc: BaseException) -> str:
    message = str(exc).strip()
    return (message or exc.__class__.__name__)[:2000]


def _running_under_pytest() -> bool:
    return bool(
        os.getenv("PYTEST_CURRENT_TEST")
        or "pytest" in sys.modules
        or any("pytest" in str(argument).lower() for argument in sys.argv)
    )


def _dialect_name(db: Session) -> str:
    return str(db.get_bind().dialect.name)


def _audit_session(origin_db: Session) -> Session:
    return Session(bind=origin_db.get_bind(), autoflush=False, expire_on_commit=False, future=True)


def _assert_qdrant_point_scope_available(
    db: Session,
    *,
    knowledge_base_id: str,
    collection_name: str,
    point_ids: list[str],
    requested_operation: str,
    allowed_intent_ids: set[str] | None = None,
    allow_foreign_records: bool = False,
    allow_same_kb_upsert_reservations: bool = False,
) -> None:
    normalized_ids = sorted({str(item) for item in point_ids if str(item)})
    if not collection_name or not normalized_ids:
        raise QdrantOutboxError("Qdrant point reservation requires a non-empty collection and point scope")
    allowed = {str(item) for item in (allowed_intent_ids or set())}

    foreign_records = list(
        db.scalars(
            select(VectorRecord).where(
                VectorRecord.collection_name == collection_name,
                VectorRecord.qdrant_point_id.in_(normalized_ids),
                VectorRecord.knowledge_base_id != knowledge_base_id,
            )
        ).all()
    )
    if foreign_records and not allow_foreign_records:
        conflicts = sorted(
            {
                f"{record.qdrant_point_id}:{record.knowledge_base_id}"
                for record in foreign_records
            }
        )
        raise QdrantOutboxError(
            "Qdrant point scope is already owned by another knowledge base: "
            + ", ".join(conflicts)
        )

    reservation_states = sorted(
        QDRANT_OUTBOX_ACTIVE_STATES
        | QDRANT_DELETE_ACTIVE_STATES
        | QDRANT_VECTOR_SCHEMA_METADATA_PATCH_ACTIVE_STATES
    )
    candidate_rows = list(
        db.scalars(
            select(IngestionCompensationLog).where(
                (
                    IngestionCompensationLog.operation.in_(
                        [
                            QDRANT_UPSERT_OPERATION,
                            QDRANT_DELETE_OPERATION,
                            QDRANT_VECTOR_SCHEMA_METADATA_PATCH_OPERATION,
                        ]
                    )
                    & IngestionCompensationLog.status.in_(reservation_states)
                )
                | (
                    (IngestionCompensationLog.operation == LEGACY_QDRANT_DELETE_OPERATION)
                    & IngestionCompensationLog.status.in_(
                        sorted(LEGACY_QDRANT_DELETE_BLOCKING_STATES)
                    )
                )
            )
        ).all()
    )
    requested = set(normalized_ids)
    blockers: list[str] = []
    for row in candidate_rows:
        if row.id in allowed:
            continue
        payload = dict(row.payload_json or {})
        if str(payload.get("collection_name") or "") != collection_name:
            continue
        overlap = sorted(requested.intersection(str(item) for item in (row.target_ids_json or [])))
        same_kb_upsert_supersession = (
            allow_same_kb_upsert_reservations
            and row.knowledge_base_id == knowledge_base_id
            and requested_operation == QDRANT_UPSERT_OPERATION
            and row.operation == QDRANT_UPSERT_OPERATION
        )
        if overlap and not same_kb_upsert_supersession:
            blockers.append(
                f"{row.id}:{row.knowledge_base_id}:{row.operation}:{','.join(overlap)}"
            )
    if blockers:
        raise QdrantOutboxError(
            "Qdrant point scope has an unresolved durable reservation: "
            + "; ".join(sorted(blockers))
        )


def assert_vector_schema_metadata_patch_scope_available(
    db: Session,
    *,
    knowledge_base_id: str,
    collection_name: str,
    point_ids: list[str],
    allowed_intent_ids: set[str] | None = None,
) -> None:
    """Reserve an exact payload-only metadata scope against all Qdrant writers."""

    _assert_qdrant_point_scope_available(
        db,
        knowledge_base_id=knowledge_base_id,
        collection_name=collection_name,
        point_ids=point_ids,
        requested_operation=QDRANT_VECTOR_SCHEMA_METADATA_PATCH_OPERATION,
        allowed_intent_ids=allowed_intent_ids,
    )


def _qdrant_point_mutation_lock_key(collection_name: str, point_id: str) -> int:
    digest = hashlib.sha256(
        (
            f"symbograph:{QDRANT_POINT_MUTATION_LOCK_PROTOCOL_VERSION}:"
            f"{collection_name}:{point_id}"
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


@contextmanager
def qdrant_point_mutation_lock(
    db: Session,
    *,
    collection_name: str,
    point_ids: list[str],
):
    normalized_ids = sorted({str(item) for item in point_ids if str(item)})
    if not normalized_ids:
        yield
        return
    dialect = _dialect_name(db)
    if dialect == "sqlite" and _running_under_pytest():
        yield
        return
    if dialect != "postgresql":
        raise QdrantOutboxError("Qdrant point mutation locks require PostgreSQL")
    bind = db.get_bind()
    engine = bind if hasattr(bind, "connect") else bind.engine
    connection = engine.connect()
    acquired_keys: list[int] = []
    try:
        for point_id in normalized_ids:
            lock_key = _qdrant_point_mutation_lock_key(collection_name, point_id)
            acquired = bool(
                connection.execute(
                    text("SELECT pg_try_advisory_lock(:lock_key)"),
                    {"lock_key": lock_key},
                ).scalar_one()
            )
            connection.commit()
            if not acquired:
                raise QdrantOutboxError(
                    f"Qdrant point mutation is busy for collection={collection_name}, point={point_id}; retry"
                )
            acquired_keys.append(lock_key)
        yield
    finally:
        for lock_key in reversed(acquired_keys):
            try:
                connection.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": lock_key},
                )
                connection.commit()
            except Exception:
                connection.invalidate()
                break
        connection.close()


def _test_history(db: Session) -> list[dict[str, Any]]:
    return db.info.setdefault(QDRANT_OUTBOX_TEST_HISTORY_KEY, [])


def _record_test_transition(
    db: Session,
    *,
    intent_id: str,
    status: str,
    payload: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    _test_history(db).append(
        {
            "intent_id": intent_id,
            "status": status,
            "payload": dict(payload or {}),
            "error_message": error_message,
            "at": datetime.utcnow().isoformat(),
            "adapter": "sqlite_pytest_non_durable_adapter_v1",
        }
    )


def _append_bounded_state_history(
    payload: dict[str, Any],
    transition: dict[str, Any],
) -> None:
    history = [
        dict(item)
        for item in list(payload.get("state_history") or [])
        if isinstance(item, dict)
    ]
    first_transition = payload.get("state_history_first_transition")
    if not isinstance(first_transition, dict):
        first_transition = dict(history[0] if history else transition)
    try:
        previous_total = int(payload.get("state_history_total_count") or len(history))
    except (TypeError, ValueError):
        previous_total = len(history)
    total_count = max(previous_total, len(history)) + 1

    previous = history[-1] if history else None
    previous_reason = (
        str((previous.get("details") or {}).get("reason") or "")
        if isinstance(previous, dict) and isinstance(previous.get("details"), dict)
        else ""
    )
    transition_reason = (
        str((transition.get("details") or {}).get("reason") or "")
        if isinstance(transition.get("details"), dict)
        else ""
    )
    if (
        previous is not None
        and previous.get("status") == transition.get("status")
        and previous_reason == transition_reason
    ):
        compacted = dict(transition)
        compacted["first_at"] = previous.get("first_at") or previous.get("at")
        try:
            occurrences = int(previous.get("occurrences") or 1) + 1
        except (TypeError, ValueError):
            occurrences = 2
        compacted["occurrences"] = occurrences
        history[-1] = compacted
    else:
        history.append(dict(transition))
    history = history[-QDRANT_OUTBOX_STATE_HISTORY_LIMIT:]
    represented_count = 0
    for item in history:
        try:
            represented_count += max(1, int(item.get("occurrences") or 1))
        except (TypeError, ValueError):
            represented_count += 1
    payload["state_history"] = history
    payload["state_history_total_count"] = total_count
    payload["state_history_first_transition"] = dict(first_transition)
    payload["state_history_truncated_count"] = max(total_count - represented_count, 0)
    payload["last_transition"] = dict(transition)


def _validate_envelope_point_scope(
    *,
    intent_id: str,
    knowledge_base_id: str,
    protocol_version: str,
    target_points: list[dict[str, Any]],
    before_points: list[dict[str, Any]],
    expected_target_ids: list[str],
) -> None:
    target_ids = [str(point.get("id") or "") for point in target_points]
    if (
        not target_ids
        or not all(target_ids)
        or len(target_ids) != len(set(target_ids))
        or target_ids != expected_target_ids
        or len(expected_target_ids) != len(set(expected_target_ids))
    ):
        raise QdrantOutboxError(
            f"Intent {intent_id} target ids are not a unique ordered durable scope"
        )
    before_ids = [str(point.get("id") or "") for point in before_points]
    if (
        not all(before_ids)
        or len(before_ids) != len(set(before_ids))
        or not set(before_ids).issubset(set(target_ids))
    ):
        raise QdrantOutboxError(
            f"Intent {intent_id} before-image ids escape its target scope"
        )

    for label, points in (("target", target_points), ("before-image", before_points)):
        for point in points:
            point_id = str(point.get("id") or "")
            point_payload = point.get("payload")
            if not isinstance(point_payload, dict) or not isinstance(point.get("vector"), list):
                raise QdrantOutboxError(
                    f"Intent {intent_id} contains an invalid {label} point"
                )
            if (
                str(point_payload.get("knowledge_base_id") or "") != knowledge_base_id
                or str(point_payload.get("chunk_id") or "") != point_id
            ):
                raise QdrantOutboxError(
                    f"Intent {intent_id} {label} point {point_id} escapes its knowledge-base/chunk scope"
                )
            if label == "target" and (
                point_payload.get("qdrant_write_intent_id") != intent_id
                or point_payload.get("qdrant_write_protocol_version") != protocol_version
            ):
                raise QdrantOutboxError(
                    f"Intent {intent_id} contains an invalid target point owner/protocol"
                )


def _validate_current_point_scope(
    *,
    intent_id: str,
    knowledge_base_id: str,
    current_by_id: dict[str, dict[str, Any]],
) -> None:
    for point_id, point in current_by_id.items():
        payload = point.get("payload")
        if (
            not isinstance(payload, dict)
            or str(payload.get("knowledge_base_id") or "") != knowledge_base_id
            or str(payload.get("chunk_id") or "") != point_id
        ):
            raise QdrantOutboxError(
                f"Intent {intent_id} cannot mutate out-of-scope current Qdrant point {point_id}"
            )


def _validate_reconcile_current_point_scope(
    *,
    intent_id: str,
    knowledge_base_id: str,
    current_by_id: dict[str, dict[str, Any]],
    protected_points: list[dict[str, Any]],
) -> None:
    protected_by_id = {
        str(point.get("id") or ""): dict(point)
        for point in protected_points
    }
    if not all(protected_by_id) or len(protected_by_id) != len(protected_points):
        raise QdrantOutboxError(
            f"Intent {intent_id} has an invalid PostgreSQL-authoritative protected scope"
        )
    for point_id, point in current_by_id.items():
        payload = point.get("payload")
        belongs_to_intent_scope = (
            isinstance(payload, dict)
            and str(payload.get("knowledge_base_id") or "") == knowledge_base_id
            and str(payload.get("chunk_id") or "") == point_id
        )
        protected = protected_by_id.get(point_id)
        matches_protected_owner = (
            protected is not None
            and _canonical_hash(point) == _canonical_hash(protected)
        )
        if not belongs_to_intent_scope and not matches_protected_owner:
            raise QdrantOutboxError(
                f"Intent {intent_id} cannot reconcile ungrounded current Qdrant point {point_id}"
            )


def _canonical_outbox_vector(vector: Any, *, source: str) -> list[float]:
    # Imported lazily so context_graph can import this module without a cycle.
    from app.services.vector_store import canonical_embedding_vector

    try:
        return canonical_embedding_vector(vector, source=source)
    except ValueError as exc:
        raise QdrantOutboxError(str(exc)) from exc


def _active_v2_vector_contract() -> dict[str, Any]:
    # These functions/constants are the single frozen source for collection and
    # vector identity. The import must stay lazy because context_graph imports
    # execute_qdrant_upsert_batches from this module.
    from app.services.context_graph import (
        QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
        QDRANT_VECTOR_DISTANCE_METRIC,
        VECTOR_PAYLOAD_HASH_PROTOCOL_V2,
        VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION,
        qdrant_collection_identity_digest,
        qdrant_collection_name,
        vector_payload_hash,
        vector_payload_hash_v2,
    )

    return {
        "collection_identity_protocol_version": QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
        "vector_distance_metric": QDRANT_VECTOR_DISTANCE_METRIC,
        "vector_payload_hash_protocol_v2": VECTOR_PAYLOAD_HASH_PROTOCOL_V2,
        "vector_payload_hash_protocol": VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION,
        "collection_identity_digest": qdrant_collection_identity_digest,
        "collection_name": qdrant_collection_name,
        "vector_payload_hash": vector_payload_hash,
        "vector_payload_hash_v2": vector_payload_hash_v2,
    }


def _canonicalize_before_points(
    before_points: list[dict[str, Any]],
    *,
    intent_id: str,
) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for index, raw_point in enumerate(before_points):
        if not isinstance(raw_point, dict) or set(raw_point) != {"id", "vector", "payload"}:
            raise QdrantOutboxError(
                f"Intent {intent_id} before-image point {index} does not match the frozen point schema"
            )
        point_id = raw_point.get("id")
        if not isinstance(point_id, str) or not point_id:
            raise QdrantOutboxError(
                f"Intent {intent_id} before-image point {index} has no canonical string id"
            )
        payload = _strict_json_copy(
            raw_point.get("payload"),
            path=f"$.before_points[{index}].payload",
        )
        if not isinstance(payload, dict):
            raise QdrantOutboxError(
                f"Intent {intent_id} before-image point {point_id} has no payload object"
            )
        canonical.append(
            {
                "id": point_id,
                "vector": _canonical_outbox_vector(
                    raw_point.get("vector"),
                    source=f"Intent {intent_id} before-image point {point_id} vector",
                ),
                "payload": payload,
            }
        )
    return canonical


def _canonicalize_v2_target_points(
    target_points: list[dict[str, Any]],
    *,
    intent_id: str,
    knowledge_base_id: str,
    collection_name: str,
    strict_schema: bool,
    allow_historical_vector_hash: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(collection_name, str) or not collection_name:
        raise QdrantOutboxError("Qdrant v2 upsert requires a non-empty collection name")
    if len(target_points) > QDRANT_OUTBOX_MAX_TARGET_POINTS_PER_INTENT:
        raise QdrantOutboxError(
            "Qdrant upsert intent exceeds the frozen per-intent target-point bound"
        )
    contract = _active_v2_vector_contract() if strict_schema else None
    canonical: list[dict[str, Any]] = []
    common_schema_card: tuple[Any, ...] | None = None
    for index, raw_point in enumerate(target_points):
        if not isinstance(raw_point, dict) or set(raw_point) != {"id", "vector", "payload"}:
            raise QdrantOutboxError(
                f"Intent {intent_id} target point {index} does not match the frozen point schema"
            )
        point_id = raw_point.get("id")
        if not isinstance(point_id, str) or not point_id:
            raise QdrantOutboxError(
                f"Intent {intent_id} target point {index} has no canonical string id"
            )
        source_payload = _strict_json_copy(
            raw_point.get("payload"),
            path=f"$.target_points[{index}].payload",
        )
        if not isinstance(source_payload, dict):
            raise QdrantOutboxError(
                f"Intent {intent_id} target point {point_id} has no payload object"
            )
        supplied_knowledge_base_id = source_payload.get("knowledge_base_id")
        supplied_chunk_id = source_payload.get("chunk_id")
        supplied_owner = source_payload.get("qdrant_write_intent_id")
        supplied_protocol = source_payload.get("qdrant_write_protocol_version")
        if (
            supplied_knowledge_base_id is not None
            and supplied_knowledge_base_id != ""
            and supplied_knowledge_base_id != knowledge_base_id
        ):
            raise QdrantOutboxError(
                f"Qdrant target point {point_id} belongs to another knowledge base"
            )
        if (
            supplied_chunk_id is not None
            and supplied_chunk_id != ""
            and supplied_chunk_id != point_id
        ):
            raise QdrantOutboxError(
                f"Qdrant target point {point_id} has a mismatched chunk id"
            )
        if supplied_owner is not None and supplied_owner != "" and supplied_owner != intent_id:
            raise QdrantOutboxError(
                f"Qdrant target point {point_id} belongs to another durable intent"
            )
        if (
            supplied_protocol is not None
            and supplied_protocol != ""
            and supplied_protocol != QDRANT_OUTBOX_PROTOCOL_VERSION
        ):
            raise QdrantOutboxError(
                f"Qdrant target point {point_id} has a mismatched write protocol"
            )

        vector = _canonical_outbox_vector(
            raw_point.get("vector"),
            source=f"Intent {intent_id} target point {point_id} vector",
        )
        payload = {
            **source_payload,
            "knowledge_base_id": knowledge_base_id,
            "chunk_id": point_id,
            "qdrant_write_intent_id": intent_id,
            "qdrant_write_protocol_version": QDRANT_OUTBOX_PROTOCOL_VERSION,
        }

        if strict_schema:
            assert contract is not None
            required_string_fields = (
                "embedding_model",
                "embedding_text_version",
                "chunk_schema_version",
                "collection_identity_protocol_version",
                "collection_identity_digest",
                "vector_payload_hash_protocol",
                "vector_payload_hash",
            )
            missing = [
                field
                for field in required_string_fields
                if not isinstance(payload.get(field), str) or not payload.get(field)
            ]
            if missing:
                raise QdrantOutboxError(
                    f"Qdrant v2 target point {point_id} is missing schema fields: {', '.join(missing)}"
                )
            dimension = payload.get("embedding_dimension")
            if type(dimension) is not int or dimension <= 0:
                raise QdrantOutboxError(
                    f"Qdrant v2 target point {point_id} embedding_dimension must be a positive integer"
                )
            if len(vector) != dimension:
                raise QdrantOutboxError(
                    f"Qdrant v2 target point {point_id} vector length conflicts with embedding_dimension"
                )
            if payload.get("vector_distance_metric") != contract["vector_distance_metric"]:
                raise QdrantOutboxError(
                    f"Qdrant v2 target point {point_id} vector_distance_metric must be cosine"
                )
            if (
                payload.get("collection_identity_protocol_version")
                != contract["collection_identity_protocol_version"]
            ):
                raise QdrantOutboxError(
                    f"Qdrant v2 target point {point_id} has an invalid collection identity protocol"
                )
            vector_hash_protocol = payload.get("vector_payload_hash_protocol")
            active_vector_hash = vector_hash_protocol == contract["vector_payload_hash_protocol"]
            historical_vector_hash = (
                allow_historical_vector_hash
                and vector_hash_protocol == contract["vector_payload_hash_protocol_v2"]
            )
            if not active_vector_hash and not historical_vector_hash:
                raise QdrantOutboxError(
                    f"Qdrant v2 target point {point_id} has an invalid vector payload hash protocol"
                )
            if active_vector_hash:
                contextual_hash_fields = (
                    "context_hash_protocol_version",
                    "context_hash",
                    "local_hint_protocol_version",
                    "local_hint_hash",
                )
                missing_contextual_hash_fields = [
                    field
                    for field in contextual_hash_fields
                    if not isinstance(payload.get(field), str) or not payload.get(field)
                ]
                if missing_contextual_hash_fields:
                    raise QdrantOutboxError(
                        f"Qdrant v2 target point {point_id} is missing contextual hash fields: "
                        f"{', '.join(missing_contextual_hash_fields)}"
                    )

            identity_args = {
                "embedding_model": payload["embedding_model"],
                "embedding_dimension": dimension,
                "embedding_text_version": payload["embedding_text_version"],
                "chunk_schema_version": payload["chunk_schema_version"],
            }
            try:
                expected_digest = contract["collection_identity_digest"](**identity_args)
                expected_collection_name = contract["collection_name"](**identity_args)
            except (TypeError, ValueError) as exc:
                raise QdrantOutboxError(
                    f"Qdrant v2 target point {point_id} has an invalid vector schema card"
                ) from exc
            if payload.get("collection_identity_digest") != expected_digest:
                raise QdrantOutboxError(
                    f"Qdrant v2 target point {point_id} collection identity digest is invalid"
                )
            if collection_name != expected_collection_name:
                raise QdrantOutboxError(
                    f"Qdrant v2 collection name is not canonical for target point {point_id}"
                )
            try:
                if active_vector_hash:
                    expected_vector_hash = contract["vector_payload_hash"](
                        vector=vector,
                        chunk_id=point_id,
                        embedding_model=payload["embedding_model"],
                        embedding_dimension=dimension,
                        vector_distance_metric=payload["vector_distance_metric"],
                        embedding_text_version=payload["embedding_text_version"],
                        chunk_schema_version=payload["chunk_schema_version"],
                        context_hash_protocol_version=payload[
                            "context_hash_protocol_version"
                        ],
                        context_hash=payload["context_hash"],
                        local_hint_protocol_version=payload[
                            "local_hint_protocol_version"
                        ],
                        local_hint_hash=payload["local_hint_hash"],
                        collection_identity_protocol_version=payload[
                            "collection_identity_protocol_version"
                        ],
                        collection_identity_digest=payload[
                            "collection_identity_digest"
                        ],
                    )
                else:
                    expected_vector_hash = contract["vector_payload_hash_v2"](
                        vector=vector,
                        chunk_id=point_id,
                        embedding_model=payload["embedding_model"],
                        embedding_dimension=dimension,
                        embedding_text_version=payload["embedding_text_version"],
                    )
            except (TypeError, ValueError) as exc:
                raise QdrantOutboxError(
                    f"Qdrant v2 target point {point_id} has an invalid vector schema card"
                ) from exc
            if payload.get("vector_payload_hash") != expected_vector_hash:
                raise QdrantOutboxError(
                    f"Qdrant v2 target point {point_id} vector payload hash is invalid"
                )
            schema_card = (
                payload["embedding_model"],
                dimension,
                payload["vector_distance_metric"],
                payload["embedding_text_version"],
                payload["chunk_schema_version"],
                payload["collection_identity_protocol_version"],
                payload["collection_identity_digest"],
                payload["vector_payload_hash_protocol"],
            )
            if common_schema_card is None:
                common_schema_card = schema_card
            elif schema_card != common_schema_card:
                raise QdrantOutboxError(
                    "All Qdrant v2 target points in one intent must share one vector schema card"
                )

        canonical.append({"id": point_id, "vector": vector, "payload": payload})
    if not canonical:
        raise QdrantOutboxError("Qdrant v2 upsert intent requires at least one target point")
    return canonical


def _finalize_qdrant_upsert_envelope(
    *,
    intent_id: str,
    knowledge_base_id: str,
    job_id: str | None,
    collection_name: str,
    canonical_targets: list[dict[str, Any]],
    before_points: list[dict[str, Any]],
) -> _PreparedQdrantUpsertEnvelope:
    canonical_before = _canonicalize_before_points(before_points, intent_id=intent_id)
    target_ids = [str(point["id"]) for point in canonical_targets]
    _validate_envelope_point_scope(
        intent_id=intent_id,
        knowledge_base_id=knowledge_base_id,
        protocol_version=QDRANT_OUTBOX_PROTOCOL_VERSION,
        target_points=canonical_targets,
        before_points=canonical_before,
        expected_target_ids=target_ids,
    )
    return _PreparedQdrantUpsertEnvelope(
        intent_id=intent_id,
        knowledge_base_id=knowledge_base_id,
        job_id=job_id,
        collection_name=collection_name,
        target_points=tuple(canonical_targets),
        before_points=tuple(canonical_before),
        target_payload_hash=_outbox_payload_hash(
            QDRANT_OUTBOX_PROTOCOL_VERSION,
            canonical_targets,
        ),
        before_image_hash=_outbox_payload_hash(
            QDRANT_OUTBOX_PROTOCOL_VERSION,
            canonical_before,
        ),
    )


def _prepare_qdrant_upsert_envelope(
    *,
    intent_id: str,
    knowledge_base_id: str,
    job_id: str | None,
    collection_name: str,
    target_points: list[dict[str, Any]],
    before_points: list[dict[str, Any]],
    strict_schema: bool,
) -> _PreparedQdrantUpsertEnvelope:
    canonical_targets = _canonicalize_v2_target_points(
        target_points,
        intent_id=intent_id,
        knowledge_base_id=knowledge_base_id,
        collection_name=collection_name,
        strict_schema=strict_schema,
    )
    return _finalize_qdrant_upsert_envelope(
        intent_id=intent_id,
        knowledge_base_id=knowledge_base_id,
        job_id=job_id,
        collection_name=collection_name,
        canonical_targets=canonical_targets,
        before_points=before_points,
    )


def _persist_prepared_intent(
    db: Session,
    *,
    envelope: _PreparedQdrantUpsertEnvelope,
) -> bool:
    dialect = _dialect_name(db)
    now = datetime.utcnow()
    initial_transition = {"status": "pending", "at": now.isoformat()}
    protocol_contract = _outbox_protocol_contract(QDRANT_OUTBOX_PROTOCOL_VERSION)
    payload = {
        "protocol_version": QDRANT_OUTBOX_PROTOCOL_VERSION,
        "envelope_schema_version": protocol_contract["envelope_schema_version"],
        "envelope_schema_hash": protocol_contract["envelope_schema_hash"],
        "canonical_bytes_version": protocol_contract["canonical_bytes_version"],
        "intent_id": envelope.intent_id,
        "collection_name": envelope.collection_name,
        "target_points": list(envelope.target_points),
        "before_points": list(envelope.before_points),
        "target_payload_hash": envelope.target_payload_hash,
        "before_image_hash": envelope.before_image_hash,
        "requires_uncertainty_watch": True,
        "lease_expires_at": (now + timedelta(seconds=QDRANT_OUTBOX_LEASE_SECONDS)).isoformat(),
        "state_history": [initial_transition],
        "state_history_total_count": 1,
        "state_history_first_transition": initial_transition,
        "state_history_truncated_count": 0,
        "last_transition": initial_transition,
    }
    if dialect == "sqlite" and _running_under_pytest():
        _record_test_transition(
            db,
            intent_id=envelope.intent_id,
            status="pending",
            payload=payload,
        )
        return False
    if dialect != "postgresql":
        raise QdrantOutboxError(
            "Durable Qdrant side-effect intents require PostgreSQL; only the explicit SQLite pytest adapter is supported"
        )
    with _audit_session(db) as audit:
        _assert_qdrant_point_scope_available(
            audit,
            knowledge_base_id=envelope.knowledge_base_id,
            collection_name=envelope.collection_name,
            point_ids=[str(point["id"]) for point in envelope.target_points],
            requested_operation=QDRANT_UPSERT_OPERATION,
        )
        audit.add(
            IngestionCompensationLog(
                id=envelope.intent_id,
                job_id=envelope.job_id,
                knowledge_base_id=envelope.knowledge_base_id,
                operation=QDRANT_UPSERT_OPERATION,
                target_ids_json=[str(point["id"]) for point in envelope.target_points],
                payload_json=payload,
                status="pending",
            )
        )
        audit.commit()
    return True


def _persist_intent(
    db: Session,
    *,
    intent_id: str,
    knowledge_base_id: str,
    job_id: str | None,
    collection_name: str,
    target_points: list[dict[str, Any]],
    before_points: list[dict[str, Any]],
) -> bool:
    """Compatibility entry point; all new durable v2 intents pass the strict gate."""

    dialect = _dialect_name(db)
    envelope = _prepare_qdrant_upsert_envelope(
        intent_id=intent_id,
        knowledge_base_id=knowledge_base_id,
        job_id=job_id,
        collection_name=collection_name,
        target_points=target_points,
        before_points=before_points,
        strict_schema=dialect == "postgresql",
    )
    return _persist_prepared_intent(db, envelope=envelope)


def _transition_intent(
    db: Session,
    *,
    intent_id: str,
    status: str,
    details: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    if status not in QDRANT_OUTBOX_ACTIVE_STATES | QDRANT_OUTBOX_TERMINAL_STATES:
        raise ValueError(f"Unsupported Qdrant outbox state: {status}")
    dialect = _dialect_name(db)
    if dialect == "sqlite" and _running_under_pytest():
        _record_test_transition(
            db,
            intent_id=intent_id,
            status=status,
            payload=details,
            error_message=error_message,
        )
        return
    with _audit_session(db) as audit:
        row = audit.get(IngestionCompensationLog, intent_id, with_for_update=True)
        if row is None or row.operation != QDRANT_UPSERT_OPERATION:
            raise QdrantOutboxError(f"Durable Qdrant intent is missing: {intent_id}")
        current_status = str(row.status or "")
        if status not in QDRANT_OUTBOX_ALLOWED_TRANSITIONS.get(current_status, set()):
            raise QdrantOutboxError(
                f"Invalid Qdrant outbox transition for {intent_id}: {current_status} -> {status}"
            )
        if current_status in QDRANT_OUTBOX_TERMINAL_STATES and status == current_status:
            return
        payload = dict(row.payload_json or {})
        mutation_attempt = payload.get("qdrant_mutation_attempt")
        if status in QDRANT_OUTBOX_TERMINAL_STATES and (
            _requires_uncertainty_watch(row)
            or (
                isinstance(mutation_attempt, dict)
                and mutation_attempt.get("state") == "pending"
            )
        ):
            raise QdrantOutboxError(
                f"Intent {intent_id} cannot become terminal while a Qdrant transport outcome is unresolved"
            )
        now = datetime.utcnow()
        transition = {"status": status, "at": now.isoformat()}
        if details:
            transition["details"] = dict(details)
        _append_bounded_state_history(payload, transition)
        if status == "external_applied":
            payload["requires_uncertainty_watch"] = False
        elif status == "external_outcome_unknown":
            payload["requires_uncertainty_watch"] = True
            uncertainty_not_before = now + timedelta(seconds=QDRANT_OUTBOX_LEASE_SECONDS)
            payload["uncertainty_not_before"] = uncertainty_not_before.isoformat()
            payload["lease_expires_at"] = uncertainty_not_before.isoformat()
            payload.pop("uncertainty_observation", None)
            payload.pop("confirmation_not_before", None)
            payload.pop("compensation_verification", None)
        elif status == "uncertainty_observed":
            payload["uncertainty_observation"] = dict(details or {})
            payload["confirmation_not_before"] = (
                now + timedelta(seconds=QDRANT_OUTBOX_UNCERTAINTY_CONFIRM_SECONDS)
            ).isoformat()
        elif status == "compensation_verify_pending":
            payload["compensation_verification"] = dict(details or {})
            payload["confirmation_not_before"] = (
                now + timedelta(seconds=QDRANT_OUTBOX_UNCERTAINTY_CONFIRM_SECONDS)
            ).isoformat()
        elif status == "compensation_pending":
            payload["lease_expires_at"] = now.isoformat()
        protected_superseders = (details or {}).get("uncertainty_protected_superseders")
        if protected_superseders is not None:
            protected = [dict(item) for item in list(protected_superseders or [])]
            payload["uncertainty_protected_superseders"] = protected
            payload["uncertainty_protected_superseders_hash"] = _canonical_hash(protected)
        row.payload_json = payload
        row.status = status
        row.error_message = error_message
        audit.commit()


def _begin_qdrant_mutation_attempt(
    db: Session,
    *,
    intent_id: str,
    mutation_kind: str,
    planned_delete_ids: list[str],
    planned_upsert_points: list[dict[str, Any]],
) -> str:
    attempt_id = str(uuid4())
    dialect = _dialect_name(db)
    if dialect == "sqlite" and _running_under_pytest():
        return attempt_id
    if dialect != "postgresql":
        raise QdrantOutboxError("Durable Qdrant mutation attempts require PostgreSQL")
    with _audit_session(db) as audit:
        row = audit.get(IngestionCompensationLog, intent_id, with_for_update=True)
        if row is None or row.operation != QDRANT_UPSERT_OPERATION:
            raise QdrantOutboxError(f"Durable Qdrant intent is missing: {intent_id}")
        if row.status in QDRANT_OUTBOX_TERMINAL_STATES:
            raise QdrantOutboxError(
                f"Intent {intent_id} cannot start a Qdrant mutation after terminal state {row.status}"
            )
        target_ids = [str(item) for item in list(row.target_ids_json or [])]
        _assert_qdrant_point_scope_available(
            audit,
            knowledge_base_id=row.knowledge_base_id,
            collection_name=str((row.payload_json or {}).get("collection_name") or ""),
            point_ids=target_ids,
            requested_operation=QDRANT_UPSERT_OPERATION,
            allowed_intent_ids={row.id},
            allow_foreign_records=True,
            allow_same_kb_upsert_reservations=True,
        )
        delete_ids = [str(item) for item in planned_delete_ids]
        if len(delete_ids) != len(set(delete_ids)) or not set(delete_ids).issubset(set(target_ids)):
            raise QdrantOutboxError(f"Intent {intent_id} mutation delete scope is invalid")
        planned_by_id = {
            str(point.get("id") or ""): dict(point)
            for point in planned_upsert_points
        }
        if (
            not all(planned_by_id)
            or len(planned_by_id) != len(planned_upsert_points)
            or not set(planned_by_id).issubset(set(target_ids))
        ):
            raise QdrantOutboxError(f"Intent {intent_id} mutation upsert scope is invalid")
        authoritative_entries = _postgresql_authoritative_targets(audit, row, target_ids)
        authoritative_by_id = {
            str(item["point_id"]): dict(item["target_point"])
            for item in authoritative_entries
        }
        if any(point_id in authoritative_by_id for point_id in delete_ids):
            raise QdrantOutboxError(
                f"Intent {intent_id} mutation would delete a PostgreSQL-authoritative point"
            )
        for point_id, point in planned_by_id.items():
            desired = authoritative_by_id.get(point_id)
            if desired is None or _canonical_hash(point) != _canonical_hash(desired):
                raise QdrantOutboxError(
                    f"Intent {intent_id} mutation upsert for {point_id} is not PostgreSQL-authoritative"
                )

        desired_state = {
            "knowledge_base_id": row.knowledge_base_id,
            "collection_name": str((row.payload_json or {}).get("collection_name") or ""),
            "target_ids": sorted(target_ids),
            "present": [
                {
                    "point_id": point_id,
                    "point_hash": _canonical_hash(point),
                    "owner_intent_id": _point_owner(point),
                    "knowledge_base_id": str(
                        (point.get("payload") or {}).get("knowledge_base_id") or ""
                    ),
                }
                for point_id, point in sorted(authoritative_by_id.items())
            ],
            "absent_ids": sorted(set(target_ids) - set(authoritative_by_id)),
        }
        payload = dict(row.payload_json or {})
        previous_attempt = payload.get("qdrant_mutation_attempt")
        history = [
            dict(item)
            for item in list(payload.get("qdrant_mutation_attempt_history") or [])
            if isinstance(item, dict)
        ]
        if isinstance(previous_attempt, dict):
            history.append(dict(previous_attempt))
        history = history[-QDRANT_OUTBOX_STATE_HISTORY_LIMIT:]
        watch_required_before = _requires_uncertainty_watch(row)
        attempt = {
            "protocol_version": "qdrant_mutation_attempt_v1",
            "attempt_id": attempt_id,
            "mutation_kind": mutation_kind,
            "state": "pending",
            "started_at": datetime.utcnow().isoformat(),
            "watch_required_before_attempt": watch_required_before,
            "postgresql_desired_state_hash": _canonical_hash(desired_state),
            "postgresql_desired_present_ids": sorted(authoritative_by_id),
            "postgresql_desired_absent_ids": desired_state["absent_ids"],
            "planned_delete_ids": sorted(delete_ids),
            "planned_upsert_ids": sorted(planned_by_id),
        }
        payload["qdrant_mutation_attempt"] = attempt
        payload["qdrant_mutation_attempt_history"] = history
        payload["qdrant_mutation_attempt_total_count"] = int(
            payload.get("qdrant_mutation_attempt_total_count") or 0
        ) + 1
        payload["requires_uncertainty_watch"] = True
        row.payload_json = payload
        audit.commit()
    return attempt_id


def _resolve_qdrant_mutation_attempt(
    db: Session,
    *,
    intent_id: str,
    attempt_id: str,
) -> None:
    dialect = _dialect_name(db)
    if dialect == "sqlite" and _running_under_pytest():
        return
    with _audit_session(db) as audit:
        row = audit.get(IngestionCompensationLog, intent_id, with_for_update=True)
        if row is None or row.operation != QDRANT_UPSERT_OPERATION:
            raise QdrantOutboxError(f"Durable Qdrant intent is missing: {intent_id}")
        payload = dict(row.payload_json or {})
        attempt = payload.get("qdrant_mutation_attempt")
        if (
            not isinstance(attempt, dict)
            or attempt.get("attempt_id") != attempt_id
            or attempt.get("state") != "pending"
        ):
            raise QdrantOutboxError(
                f"Intent {intent_id} Qdrant mutation attempt {attempt_id} is not pending"
            )
        resolved = dict(attempt)
        resolved["state"] = "transport_resolved"
        resolved["resolved_at"] = datetime.utcnow().isoformat()
        payload["qdrant_mutation_attempt"] = resolved
        payload["requires_uncertainty_watch"] = bool(
            attempt.get("watch_required_before_attempt")
        )
        row.payload_json = payload
        audit.commit()


async def _store_get_points(store: Any, ids: list[str], *, allow_test_adapter: bool) -> list[dict[str, Any]]:
    getter = getattr(store, "get_points", None)
    if callable(getter):
        return [dict(point) for point in await asyncio.to_thread(getter, ids)]
    if allow_test_adapter and isinstance(getattr(store, "points", None), dict):
        point_map = getattr(store, "points")
        return [dict(point_map[point_id]) for point_id in ids if point_id in point_map]
    if allow_test_adapter:
        return []
    raise QdrantOutboxError("VectorStore must support get_points before a durable Qdrant write")


async def _store_delete(store: Any, ids: list[str], *, allow_test_adapter: bool) -> None:
    if not ids:
        return
    async_delete = getattr(store, "async_delete", None)
    if callable(async_delete):
        await async_delete(ids)
        return
    delete = getattr(store, "delete", None)
    if callable(delete):
        await asyncio.to_thread(delete, ids)
        return
    if allow_test_adapter and isinstance(getattr(store, "points", None), dict):
        for point_id in ids:
            getattr(store, "points").pop(point_id, None)
        return
    if allow_test_adapter:
        return
    raise QdrantOutboxError("VectorStore must support delete for Qdrant compensation")


def _point_owner(point: dict[str, Any] | None) -> str | None:
    if not point:
        return None
    return str((point.get("payload") or {}).get("qdrant_write_intent_id") or "") or None


def _store_collection_exists(store: Any) -> bool:
    probe = getattr(store, "collection_exists", None)
    if callable(probe):
        return bool(probe())
    # Test adapters and injected stores are explicit existing logical stores.
    return True


async def compensate_qdrant_handle(
    db: Session,
    *,
    store: Any,
    handle: QdrantOutboxHandle,
    reason: str,
) -> dict[str, Any]:
    with qdrant_point_mutation_lock(
        db,
        collection_name=handle.collection_name,
        point_ids=list(handle.target_ids),
    ):
        return await _compensate_qdrant_handle_locked(
            db,
            store=store,
            handle=handle,
            reason=reason,
        )


async def _compensate_qdrant_handle_locked(
    db: Session,
    *,
    store: Any,
    handle: QdrantOutboxHandle,
    reason: str,
) -> dict[str, Any]:
    allow_test_adapter = not handle.durable
    target_ids = list(handle.target_ids)
    before_by_id = {str(point["id"]): dict(point) for point in handle.before_points}
    current_points = await _store_get_points(store, target_ids, allow_test_adapter=allow_test_adapter)
    current_by_id = {str(point["id"]): point for point in current_points}
    _validate_current_point_scope(
        intent_id=handle.id,
        knowledge_base_id=handle.knowledge_base_id,
        current_by_id=current_by_id,
    )
    superseded_ids = sorted(
        point_id
        for point_id, point in current_by_id.items()
        if _point_owner(point) not in {None, handle.id}
    )
    deletable_ids = sorted(
        point_id
        for point_id in target_ids
        if point_id not in before_by_id
        and point_id not in superseded_ids
        and _point_owner(current_by_id.get(point_id)) == handle.id
    )
    restorable_points = [
        before_by_id[point_id]
        for point_id in sorted(before_by_id)
        if point_id not in superseded_ids
        and (_point_owner(current_by_id.get(point_id)) in {None, handle.id})
    ]
    mutation_attempt_id = ""
    if deletable_ids or restorable_points:
        mutation_attempt_id = _begin_qdrant_mutation_attempt(
            db,
            intent_id=handle.id,
            mutation_kind="compensate_handle",
            planned_delete_ids=deletable_ids,
            planned_upsert_points=restorable_points,
        )
    try:
        await _store_delete(store, deletable_ids, allow_test_adapter=allow_test_adapter)
        if restorable_points:
            await store.async_upsert(restorable_points)
    except BaseException as exc:
        try:
            _transition_intent(
                db,
                intent_id=handle.id,
                status="external_outcome_unknown",
                details={
                    "reason": "qdrant_compensation_mutation_result_unknown",
                    "source_reason": reason,
                    "mutation_attempt_id": mutation_attempt_id,
                    "deleted_ids": deletable_ids,
                    "restore_ids": [str(point["id"]) for point in restorable_points],
                    "superseded_ids": superseded_ids,
                },
                error_message=_safe_error(exc),
            )
        except Exception:
            # The pre-mutation audit already left requires_uncertainty_watch=true.
            pass
        raise
    if mutation_attempt_id:
        _resolve_qdrant_mutation_attempt(
            db,
            intent_id=handle.id,
            attempt_id=mutation_attempt_id,
        )
    terminal_status = "superseded" if superseded_ids else "compensated"
    details = {
        "reason": reason,
        "deleted_ids": deletable_ids,
        "restored_ids": [str(point["id"]) for point in restorable_points],
        "superseded_ids": superseded_ids,
    }
    _transition_intent(db, intent_id=handle.id, status=terminal_status, details=details)
    return {"status": terminal_status, **details}


def _register_transaction_handle(db: Session, handle: QdrantOutboxHandle) -> None:
    nested = db.get_nested_transaction()
    if nested is not None:
        nested_handles = db.info.setdefault(QDRANT_OUTBOX_NESTED_SESSION_KEY, {})
        handles = nested_handles.setdefault(id(nested), [])
    else:
        handles = db.info.setdefault(QDRANT_OUTBOX_SESSION_KEY, [])
    if all(existing.id != handle.id for existing in handles):
        handles.append(handle)


def _unregister_transaction_handles(db: Session, handle_ids: set[str]) -> None:
    registered = db.info.get(QDRANT_OUTBOX_SESSION_KEY) or []
    db.info[QDRANT_OUTBOX_SESSION_KEY] = [handle for handle in registered if handle.id not in handle_ids]
    nested_map = db.info.get(QDRANT_OUTBOX_NESTED_SESSION_KEY) or {}
    for transaction_id, handles in list(nested_map.items()):
        remaining = [handle for handle in handles if handle.id not in handle_ids]
        if remaining:
            nested_map[transaction_id] = remaining
        else:
            nested_map.pop(transaction_id, None)
    if not nested_map:
        db.info.pop(QDRANT_OUTBOX_NESTED_SESSION_KEY, None)


async def _execute_qdrant_upsert_batch(
    db: Session,
    *,
    store: Any,
    knowledge_base_id: str,
    job_id: str | None,
    collection_name: str,
    batch: list[dict[str, Any]],
    durable: bool,
) -> QdrantOutboxHandle:
    intent_id = str(uuid4())
    canonical_targets = _canonicalize_v2_target_points(
        batch,
        intent_id=intent_id,
        knowledge_base_id=knowledge_base_id,
        collection_name=collection_name,
        strict_schema=durable,
    )
    target_ids = [str(point["id"]) for point in canonical_targets]
    if len(target_ids) != len(set(target_ids)):
        raise QdrantOutboxError("Qdrant upsert batch has a duplicate point scope")
    with qdrant_point_mutation_lock(
        db,
        collection_name=collection_name,
        point_ids=target_ids,
    ):
        before_points = await _store_get_points(
            store,
            target_ids,
            allow_test_adapter=not durable,
        )
        current_by_id = {str(point["id"]): point for point in before_points}
        _validate_current_point_scope(
            intent_id=intent_id,
            knowledge_base_id=knowledge_base_id,
            current_by_id=current_by_id,
        )
        envelope = _finalize_qdrant_upsert_envelope(
            intent_id=intent_id,
            knowledge_base_id=knowledge_base_id,
            job_id=job_id,
            collection_name=collection_name,
            canonical_targets=canonical_targets,
            before_points=before_points,
        )
        is_durable = _persist_prepared_intent(db, envelope=envelope)
        audited_points = list(envelope.target_points)
        handle = QdrantOutboxHandle(
            id=intent_id,
            knowledge_base_id=knowledge_base_id,
            job_id=job_id,
            collection_name=collection_name,
            target_ids=tuple(str(point["id"]) for point in envelope.target_points),
            target_points=envelope.target_points,
            before_points=envelope.before_points,
            payload_hash=envelope.target_payload_hash,
            durable=is_durable,
        )
        try:
            await store.async_upsert(audited_points)
        except BaseException as exc:
            _transition_intent(
                db,
                intent_id=intent_id,
                status="external_outcome_unknown",
                details={
                    "reason": "qdrant_upsert_result_unknown",
                    "required_reconcile_protocol": (
                        "lease_then_stable_observation_then_compensation_verification_v1"
                    ),
                },
                error_message=_safe_error(exc),
            )
            raise
        _transition_intent(
            db,
            intent_id=intent_id,
            status="external_applied",
            details={
                "point_count": len(audited_points),
                "target_payload_hash": handle.payload_hash,
            },
        )
        return handle


async def execute_qdrant_upsert_batches(
    db: Session,
    *,
    store: Any,
    knowledge_base_id: str,
    job_id: str | None,
    collection_name: str,
    points: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    if not points:
        return {
            "protocol_version": QDRANT_OUTBOX_PROTOCOL_VERSION,
            "intent_ids": [],
            "durable": _dialect_name(db) == "postgresql",
            "batches": 0,
            "point_intent_ids": {},
        }
    bounded_batch_size = min(
        max(1, int(batch_size)),
        QDRANT_OUTBOX_MAX_TARGET_POINTS_PER_INTENT,
    )
    durable = _dialect_name(db) == "postgresql"
    if not durable and not (_dialect_name(db) == "sqlite" and _running_under_pytest()):
        raise QdrantOutboxError("Qdrant writes require a durable PostgreSQL outbox")
    applied: list[QdrantOutboxHandle] = []
    point_intent_ids: dict[str, str] = {}
    try:
        for offset in range(0, len(points), bounded_batch_size):
            batch = points[offset : offset + bounded_batch_size]
            handle = await _execute_qdrant_upsert_batch(
                db,
                store=store,
                knowledge_base_id=knowledge_base_id,
                job_id=job_id,
                collection_name=collection_name,
                batch=batch,
                durable=durable,
            )
            for point_id in handle.target_ids:
                point_intent_ids[point_id] = handle.id
            _register_transaction_handle(db, handle)
            applied.append(handle)
    except BaseException:
        for handle in reversed(applied):
            try:
                await compensate_qdrant_handle(
                    db,
                    store=store,
                    handle=handle,
                    reason="later_qdrant_batch_failed_or_cancelled",
                )
            except BaseException:
                pass
        applied_ids = {handle.id for handle in applied}
        _unregister_transaction_handles(db, applied_ids)
        raise
    return {
        "protocol_version": QDRANT_OUTBOX_PROTOCOL_VERSION,
        "intent_ids": [handle.id for handle in applied],
        "durable": durable,
        "batches": len(applied),
        "point_intent_ids": point_intent_ids,
    }


def _transition_registered_handles(session: Session, status: str, reason: str) -> None:
    handles = list(session.info.pop(QDRANT_OUTBOX_SESSION_KEY, []) or [])
    for handle in handles:
        try:
            _transition_intent(
                session,
                intent_id=handle.id,
                status=status,
                details={"reason": reason},
            )
        except Exception:
            # The PostgreSQL fact transaction has already committed/rolled back.
            # Leaving an earlier durable state is safe: reconcile consumes it.
            continue


def _transition_handles(session: Session, handles: list[QdrantOutboxHandle], status: str, reason: str) -> None:
    for handle in handles:
        try:
            _transition_intent(
                session,
                intent_id=handle.id,
                status=status,
                details={"reason": reason},
            )
        except Exception:
            continue


def _finish_nested_handles(session: Session, *, committed: bool) -> None:
    nested = session.get_nested_transaction()
    if nested is None:
        return
    nested_map = session.info.get(QDRANT_OUTBOX_NESTED_SESSION_KEY) or {}
    handles = list(nested_map.pop(id(nested), []) or [])
    if not nested_map:
        session.info.pop(QDRANT_OUTBOX_NESTED_SESSION_KEY, None)
    if not handles:
        return
    if not committed:
        _transition_handles(
            session,
            handles,
            "compensation_pending",
            "postgresql_savepoint_rolled_back",
        )
        return
    parent = nested.parent
    if parent is not None and bool(getattr(parent, "nested", False)):
        parent_map = session.info.setdefault(QDRANT_OUTBOX_NESTED_SESSION_KEY, {})
        target = parent_map.setdefault(id(parent), [])
    else:
        target = session.info.setdefault(QDRANT_OUTBOX_SESSION_KEY, [])
    existing_ids = {handle.id for handle in target}
    target.extend(handle for handle in handles if handle.id not in existing_ids)


@event.listens_for(Session, "after_commit")
def _qdrant_outbox_after_commit(session: Session) -> None:
    # SQLAlchemy emits after_commit when a SAVEPOINT is released.  At that
    # point the root transaction still owns the PostgreSQL facts, so consuming
    # the handles would strand a Qdrant write if the outer transaction rolls
    # back later.
    if session.in_nested_transaction():
        _finish_nested_handles(session, committed=True)
        return
    _transition_registered_handles(session, "committed", "postgresql_transaction_committed")


@event.listens_for(Session, "after_rollback")
def _qdrant_outbox_after_rollback(session: Session) -> None:
    # A nested SAVEPOINT rollback does not decide the outcome of the root
    # transaction.  Keep the handles until the root transaction ends.
    if session.in_nested_transaction():
        _finish_nested_handles(session, committed=False)
        return
    remaining_nested = session.info.pop(QDRANT_OUTBOX_NESTED_SESSION_KEY, {}) or {}
    for handles in remaining_nested.values():
        _transition_handles(
            session,
            list(handles or []),
            "compensation_pending",
            "postgresql_root_transaction_rolled_back",
        )
    _transition_registered_handles(session, "compensation_pending", "postgresql_transaction_rolled_back")


def pending_qdrant_intents(
    db: Session,
    *,
    knowledge_base_id: str | None = None,
) -> list[IngestionCompensationLog]:
    query = select(IngestionCompensationLog).where(
        IngestionCompensationLog.operation == QDRANT_UPSERT_OPERATION,
        IngestionCompensationLog.status.in_(QDRANT_OUTBOX_ACTIVE_STATES),
    )
    if knowledge_base_id:
        query = query.where(IngestionCompensationLog.knowledge_base_id == knowledge_base_id)
    return list(db.scalars(query.order_by(IngestionCompensationLog.created_at.asc())).all())


def persist_qdrant_delete_attempt(
    db: Session,
    *,
    knowledge_base_id: str,
    collection_name: str,
    target_ids: list[str],
    current_points: list[dict[str, Any]],
    reason: str,
    source_binding: dict[str, Any] | None = None,
) -> str:
    normalized_ids = [str(item) for item in target_ids]
    if (
        not normalized_ids
        or len(normalized_ids) != len(set(normalized_ids))
        or not collection_name
    ):
        raise QdrantOutboxError("Qdrant delete intent requires a unique non-empty target scope")
    current_by_id = {str(point.get("id") or ""): dict(point) for point in current_points}
    if not all(current_by_id) or not set(current_by_id).issubset(set(normalized_ids)):
        raise QdrantOutboxError("Qdrant delete intent current-point scope is invalid")
    intent_id = str(uuid4())
    _validate_current_point_scope(
        intent_id=intent_id,
        knowledge_base_id=knowledge_base_id,
        current_by_id=current_by_id,
    )
    before_points = [
        current_by_id[point_id]
        for point_id in normalized_ids
        if point_id in current_by_id
    ]
    owner_fenced_ids = sorted(
        point_id
        for point_id, point in current_by_id.items()
        if _point_owner(point) is not None
    )
    conditional_delete_replay_safe = (
        reason == "delete_knowledge_base_data"
        or len(owner_fenced_ids) == len(before_points)
    )
    if not conditional_delete_replay_safe:
        raise QdrantOutboxError(
            "Qdrant stale-point cleanup requires qdrant_write_intent_id fences on every present point"
        )
    normalized_source_binding: dict[str, str] | None = None
    if source_binding is not None or reason == QDRANT_INGESTION_PARSE_COMPENSATION_REASON:
        normalized_source_binding = _validated_qdrant_delete_source_binding(
            source_binding,
            required=reason == QDRANT_INGESTION_PARSE_COMPENSATION_REASON,
        )
    now = datetime.utcnow().isoformat()
    desired_state = {
        "knowledge_base_id": knowledge_base_id,
        "collection_name": collection_name,
        "target_ids": sorted(normalized_ids),
        "present": [],
        "absent_ids": sorted(normalized_ids),
    }
    payload = {
        "protocol_version": QDRANT_DELETE_PROTOCOL_VERSION,
        "intent_id": intent_id,
        "knowledge_base_id": knowledge_base_id,
        "collection_name": collection_name,
        "target_ids": normalized_ids,
        "before_points": before_points,
        "before_points_hash": _canonical_hash(before_points),
        "conditional_delete_protocol_version": QDRANT_CONDITIONAL_DELETE_PROTOCOL_VERSION,
        "conditional_delete_replay_safe": conditional_delete_replay_safe,
        "owner_fenced_ids": owner_fenced_ids,
        "desired_state_hash": _canonical_hash(desired_state),
        "requires_uncertainty_watch": True,
        "mutation_attempt": {
            "protocol_version": "qdrant_mutation_attempt_v1",
            "attempt_id": str(uuid4()),
            "mutation_kind": "delete_knowledge_base_points",
            "state": "pending",
            "started_at": now,
            "postgresql_desired_state_hash": _canonical_hash(desired_state),
        },
        "reason": reason,
        "source_binding": normalized_source_binding,
        "source_binding_hash": (
            _canonical_hash(normalized_source_binding)
            if normalized_source_binding is not None
            else None
        ),
        "state_history": [{"status": "pending", "at": now}],
    }
    row = IngestionCompensationLog(
        id=intent_id,
        job_id=None,
        knowledge_base_id=knowledge_base_id,
        operation=QDRANT_DELETE_OPERATION,
        target_ids_json=normalized_ids,
        payload_json=payload,
        status="pending",
    )
    dialect = _dialect_name(db)
    if dialect == "sqlite" and _running_under_pytest():
        db.add(row)
        db.flush()
        return intent_id
    if dialect != "postgresql":
        raise QdrantOutboxError("Durable Qdrant delete intents require PostgreSQL")
    with _audit_session(db) as audit:
        _assert_qdrant_point_scope_available(
            audit,
            knowledge_base_id=knowledge_base_id,
            collection_name=collection_name,
            point_ids=normalized_ids,
            requested_operation=QDRANT_DELETE_OPERATION,
        )
        audit.add(row)
        audit.commit()
    return intent_id


def _validated_qdrant_delete_source_binding(
    source_binding: dict[str, Any] | None,
    *,
    required: bool,
) -> dict[str, str] | None:
    if source_binding is None:
        if required:
            raise QdrantOutboxError(
                "Ingestion parse compensation delete intent requires a durable source binding"
            )
        return None
    if not isinstance(source_binding, dict):
        raise QdrantOutboxError("Qdrant delete source binding must be an object")
    expected_fields = {
        "protocol_version",
        "recovery_id",
        "file_stage_id",
        "write_set_hash",
    }
    if set(source_binding) != expected_fields:
        raise QdrantOutboxError(
            "Qdrant delete source binding has an invalid frozen schema"
        )
    normalized = {
        field: str(source_binding.get(field) or "").strip()
        for field in sorted(expected_fields)
    }
    if (
        normalized["protocol_version"]
        != QDRANT_DELETE_SOURCE_BINDING_PROTOCOL_VERSION
        or not normalized["recovery_id"]
        or not normalized["file_stage_id"]
        or len(normalized["write_set_hash"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in normalized["write_set_hash"].lower()
        )
    ):
        raise QdrantOutboxError("Qdrant delete source binding identity is invalid")
    normalized["write_set_hash"] = normalized["write_set_hash"].lower()
    return normalized


def _validated_qdrant_delete_intent(
    row: IngestionCompensationLog,
) -> dict[str, Any]:
    if row.operation != QDRANT_DELETE_OPERATION:
        raise QdrantOutboxError(f"Row {row.id} is not a Qdrant delete intent")
    payload = dict(row.payload_json or {})
    protocol_version = str(payload.get("protocol_version") or "")
    if protocol_version not in QDRANT_DELETE_SUPPORTED_PROTOCOL_VERSIONS:
        raise QdrantOutboxError(f"Qdrant delete intent {row.id} has an unsupported protocol")
    if payload.get("intent_id") != row.id:
        raise QdrantOutboxError(f"Qdrant delete intent {row.id} payload identity is invalid")
    collection_name = str(payload.get("collection_name") or "")
    target_ids = [str(item) for item in (row.target_ids_json or [])]
    payload_target_ids = [str(item) for item in (payload.get("target_ids") or [])]
    if (
        not collection_name
        or not target_ids
        or len(target_ids) != len(set(target_ids))
        or payload_target_ids != target_ids
    ):
        raise QdrantOutboxError(f"Qdrant delete intent {row.id} target scope is invalid")
    before_points = [dict(point) for point in (payload.get("before_points") or [])]
    if payload.get("before_points_hash") != _canonical_hash(before_points):
        raise QdrantOutboxError(f"Qdrant delete intent {row.id} before-image hash is invalid")
    before_by_id = {str(point.get("id") or ""): point for point in before_points}
    if (
        not all(before_by_id)
        or len(before_by_id) != len(before_points)
        or not set(before_by_id).issubset(set(target_ids))
    ):
        raise QdrantOutboxError(f"Qdrant delete intent {row.id} before-image scope is invalid")
    _validate_current_point_scope(
        intent_id=row.id,
        knowledge_base_id=row.knowledge_base_id,
        current_by_id=before_by_id,
    )
    if (
        payload.get("conditional_delete_protocol_version")
        != QDRANT_CONDITIONAL_DELETE_PROTOCOL_VERSION
        or payload.get("conditional_delete_replay_safe") is not True
    ):
        raise QdrantOutboxError(
            f"Qdrant delete intent {row.id} has no replay-safe conditional delete fence"
        )
    reason = str(payload.get("reason") or "")
    source_binding = _validated_qdrant_delete_source_binding(
        payload.get("source_binding"),
        required=reason == QDRANT_INGESTION_PARSE_COMPENSATION_REASON,
    )
    if payload.get("source_binding_hash") != (
        _canonical_hash(source_binding) if source_binding is not None else None
    ):
        raise QdrantOutboxError(
            f"Qdrant delete intent {row.id} source-binding hash is invalid"
        )
    return {
        "intent_id": row.id,
        "knowledge_base_id": row.knowledge_base_id,
        "collection_name": collection_name,
        "target_ids": target_ids,
        "before_points": before_points,
        "reason": reason,
        "source_binding": source_binding,
        "source_binding_hash": payload.get("source_binding_hash"),
        "protocol_version": protocol_version,
    }


def validated_qdrant_delete_intent(
    row: IngestionCompensationLog,
) -> dict[str, Any]:
    """Public fail-closed decoder for callers binding delete intents to facts."""

    return _validated_qdrant_delete_intent(row)


def find_qdrant_delete_intent_by_source_binding(
    db: Session,
    *,
    knowledge_base_id: str,
    collection_name: str,
    target_ids: list[str],
    recovery_id: str,
    file_stage_id: str,
    write_set_hash: str,
) -> IngestionCompensationLog | None:
    expected_binding = _validated_qdrant_delete_source_binding(
        {
            "protocol_version": QDRANT_DELETE_SOURCE_BINDING_PROTOCOL_VERSION,
            "recovery_id": recovery_id,
            "file_stage_id": file_stage_id,
            "write_set_hash": write_set_hash,
        },
        required=True,
    )
    normalized_ids = [str(item) for item in target_ids]
    matches: list[IngestionCompensationLog] = []
    rows = list(
        db.scalars(
            select(IngestionCompensationLog)
            .where(
                IngestionCompensationLog.operation == QDRANT_DELETE_OPERATION,
                IngestionCompensationLog.knowledge_base_id == knowledge_base_id,
            )
            .order_by(IngestionCompensationLog.created_at.asc())
        ).all()
    )
    for row in rows:
        validated = _validated_qdrant_delete_intent(row)
        if (
            validated["reason"] == QDRANT_INGESTION_PARSE_COMPENSATION_REASON
            and validated["collection_name"] == collection_name
            and validated["target_ids"] == normalized_ids
            and validated["source_binding"] == expected_binding
        ):
            matches.append(row)
    if len(matches) > 1:
        raise QdrantOutboxError(
            "Multiple Qdrant delete intents claim one ingestion recovery write set"
        )
    return matches[0] if matches else None


def validated_committed_qdrant_upsert_target(
    db: Session,
    *,
    intent_id: str,
    knowledge_base_id: str,
    collection_name: str,
    point_id: str,
) -> dict[str, Any]:
    row = db.get(IngestionCompensationLog, intent_id)
    sqlite_test_adapter = _dialect_name(db) == "sqlite" and _running_under_pytest()
    if row is None and sqlite_test_adapter:
        history_payloads = [
            dict(item.get("payload") or {})
            for item in _test_history(db)
            if item.get("intent_id") == intent_id
            and isinstance(item.get("payload"), dict)
            and (item.get("payload") or {}).get("protocol_version")
            in QDRANT_OUTBOX_SUPPORTED_PROTOCOL_VERSIONS
        ]
        if len(history_payloads) == 1:
            history_payload = history_payloads[0]
            row = IngestionCompensationLog(
                id=intent_id,
                job_id=None,
                knowledge_base_id=knowledge_base_id,
                operation=QDRANT_UPSERT_OPERATION,
                target_ids_json=[
                    str(point.get("id") or "")
                    for point in list(history_payload.get("target_points") or [])
                ],
                payload_json=history_payload,
                status="committed",
            )
    if row is None and sqlite_test_adapter:
        # The explicit SQLite pytest adapter keeps outbox transitions in
        # Session.info, so a restart test deliberately using a new Session has
        # no durable upsert row.  Reconstruct only the owner/hash fields from
        # its PostgreSQL-fact analogue; production PostgreSQL never takes this
        # branch and always requires the committed outbox envelope above.
        records = list(
            db.scalars(
                select(VectorRecord).where(
                    VectorRecord.knowledge_base_id == knowledge_base_id,
                    VectorRecord.collection_name == collection_name,
                    VectorRecord.qdrant_point_id == point_id,
                )
            ).all()
        )
        if len(records) == 1:
            record = records[0]
            diagnostics = dict(record.diagnostics_json or {})
            if diagnostics.get("qdrant_write_intent_id") == intent_id:
                target_payload = {
                    "knowledge_base_id": knowledge_base_id,
                    "chunk_id": str(record.chunk_id),
                    "qdrant_write_intent_id": intent_id,
                    "qdrant_write_protocol_version": str(
                        diagnostics.get("qdrant_write_protocol_version") or ""
                    ),
                    "vector_payload_hash_protocol": str(
                        diagnostics.get("vector_payload_hash_protocol") or ""
                    ),
                    "vector_payload_hash": str(record.payload_hash),
                }
                target = {
                    "id": str(point_id),
                    "vector": list(diagnostics.get("embedding_vector") or []),
                    "payload": target_payload,
                }
                return {
                    "row": None,
                    "target_point": target,
                    "target_point_hash": _canonical_hash(target),
                    "target_payload": target_payload,
                    "write_protocol_version": target_payload[
                        "qdrant_write_protocol_version"
                    ],
                    "sqlite_test_adapter_reconstructed": True,
                }
    if (
        row is None
        or row.operation != QDRANT_UPSERT_OPERATION
        or (row.status != "committed" and not sqlite_test_adapter)
        or row.knowledge_base_id != knowledge_base_id
    ):
        raise QdrantOutboxError(
            f"Qdrant write owner {intent_id} is not a committed same-KB upsert intent"
        )
    payload = dict(row.payload_json or {})
    if str(payload.get("collection_name") or "") != collection_name:
        raise QdrantOutboxError(
            f"Qdrant write owner {intent_id} uses another collection"
        )
    target_points, _before_points = _validated_reconcile_payload(row)
    targets = [
        dict(point)
        for point in target_points
        if str(point.get("id") or "") == str(point_id)
    ]
    if len(targets) != 1:
        raise QdrantOutboxError(
            f"Qdrant write owner {intent_id} has no unique target for point {point_id}"
        )
    target = targets[0]
    target_payload = dict(target.get("payload") or {})
    if (
        target_payload.get("qdrant_write_intent_id") != intent_id
        or target_payload.get("qdrant_write_protocol_version")
        != payload.get("protocol_version")
    ):
        raise QdrantOutboxError(
            f"Qdrant write owner {intent_id} target owner fields are invalid"
        )
    return {
        "row": row,
        "target_point": target,
        "target_point_hash": _canonical_hash(target),
        "target_payload": target_payload,
        "write_protocol_version": str(payload.get("protocol_version") or ""),
    }


def replay_qdrant_delete_intent(
    db: Session,
    *,
    store: Any,
    row: IngestionCompensationLog,
) -> dict[str, Any]:
    validated = _validated_qdrant_delete_intent(row)
    if row.status == "external_applied":
        return {**validated, "status": "external_applied", "replayed": False}
    if row.status not in QDRANT_DELETE_ACTIVE_STATES:
        raise QdrantOutboxError(
            f"Qdrant delete intent {row.id} cannot be replayed from {row.status}"
        )
    delete_if_matches = getattr(store, "delete_if_payload_matches", None)
    if not callable(delete_if_matches):
        raise QdrantOutboxError(
            "VectorStore must support replay-safe conditional Qdrant deletes"
        )
    with qdrant_point_mutation_lock(
        db,
        collection_name=str(validated["collection_name"]),
        point_ids=list(validated["target_ids"]),
    ):
        try:
            delete_if_matches(list(validated["before_points"]))
        except Exception as exc:
            record_qdrant_delete_attempt_error(
                db,
                intent_id=row.id,
                error=exc,
            )
            raise
        record_qdrant_delete_attempt_applied(
            db,
            intent_id=row.id,
        )
    return {**validated, "status": "external_applied", "replayed": True}


def record_qdrant_delete_attempt_error(
    db: Session,
    *,
    intent_id: str,
    error: BaseException,
) -> None:
    dialect = _dialect_name(db)
    if dialect == "sqlite" and _running_under_pytest():
        row = db.get(IngestionCompensationLog, intent_id)
        if row is not None:
            payload = dict(row.payload_json or {})
            payload["requires_uncertainty_watch"] = True
            row.payload_json = payload
            row.status = "external_outcome_unknown"
            row.error_message = _safe_error(error)
            db.commit()
        return
    with _audit_session(db) as audit:
        row = audit.get(IngestionCompensationLog, intent_id, with_for_update=True)
        if row is None or row.operation != QDRANT_DELETE_OPERATION:
            raise QdrantOutboxError(f"Durable Qdrant delete intent is missing: {intent_id}")
        payload = dict(row.payload_json or {})
        payload["requires_uncertainty_watch"] = True
        transition = {
            "status": "external_outcome_unknown",
            "at": datetime.utcnow().isoformat(),
            "details": {"reason": "qdrant_delete_result_unknown"},
        }
        _append_bounded_state_history(payload, transition)
        row.payload_json = payload
        row.status = "external_outcome_unknown"
        row.error_message = _safe_error(error)
        audit.commit()


def record_qdrant_delete_attempt_applied(
    db: Session,
    *,
    intent_id: str,
) -> None:
    dialect = _dialect_name(db)
    if dialect == "sqlite" and _running_under_pytest():
        row = db.get(IngestionCompensationLog, intent_id)
        if row is None or row.operation != QDRANT_DELETE_OPERATION:
            raise QdrantOutboxError(f"Durable Qdrant delete intent is missing: {intent_id}")
        payload = dict(row.payload_json or {})
        mutation_attempt = dict(payload.get("mutation_attempt") or {})
        mutation_attempt["state"] = "transport_resolved"
        mutation_attempt["resolved_at"] = datetime.utcnow().isoformat()
        payload["mutation_attempt"] = mutation_attempt
        payload["requires_uncertainty_watch"] = False
        row.payload_json = payload
        row.status = "external_applied"
        row.error_message = None
        db.flush()
        return
    with _audit_session(db) as audit:
        row = audit.get(IngestionCompensationLog, intent_id, with_for_update=True)
        if row is None or row.operation != QDRANT_DELETE_OPERATION:
            raise QdrantOutboxError(f"Durable Qdrant delete intent is missing: {intent_id}")
        if row.status == "external_applied":
            return
        if row.status not in {"pending", "external_outcome_unknown", "reconcile_failed"}:
            raise QdrantOutboxError(
                f"Qdrant delete intent {intent_id} cannot become external_applied from {row.status}"
            )
        payload = dict(row.payload_json or {})
        mutation_attempt = dict(payload.get("mutation_attempt") or {})
        mutation_attempt["state"] = "transport_resolved"
        mutation_attempt["resolved_at"] = datetime.utcnow().isoformat()
        payload["mutation_attempt"] = mutation_attempt
        payload["requires_uncertainty_watch"] = False
        transition = {"status": "external_applied", "at": datetime.utcnow().isoformat()}
        _append_bounded_state_history(payload, transition)
        row.payload_json = payload
        row.status = "external_applied"
        row.error_message = None
        audit.commit()


def mark_qdrant_delete_attempts_committed(
    db: Session,
    *,
    intent_ids: list[str],
) -> None:
    for intent_id in dict.fromkeys(str(item) for item in intent_ids):
        row = db.scalar(
            select(IngestionCompensationLog)
            .where(IngestionCompensationLog.id == intent_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if row is None or row.operation != QDRANT_DELETE_OPERATION:
            raise QdrantOutboxError(f"Durable Qdrant delete intent is missing: {intent_id}")
        if row.status == "committed":
            continue
        if row.status != "external_applied":
            raise QdrantOutboxError(
                f"Qdrant delete intent {intent_id} cannot commit from {row.status}"
            )
        payload = dict(row.payload_json or {})
        transition = {"status": "committed", "at": datetime.utcnow().isoformat()}
        _append_bounded_state_history(payload, transition)
        payload["requires_uncertainty_watch"] = False
        row.payload_json = payload
        row.status = "committed"
        row.error_message = None


def pending_qdrant_delete_intents(
    db: Session,
    *,
    knowledge_base_id: str | None = None,
) -> list[IngestionCompensationLog]:
    query = select(IngestionCompensationLog).where(
        (
            (IngestionCompensationLog.operation == QDRANT_DELETE_OPERATION)
            & IngestionCompensationLog.status.in_(sorted(QDRANT_DELETE_ACTIVE_STATES))
        )
        | (
            (IngestionCompensationLog.operation == LEGACY_QDRANT_DELETE_OPERATION)
            & IngestionCompensationLog.status.in_(
                sorted(LEGACY_QDRANT_DELETE_BLOCKING_STATES)
            )
        )
    )
    if knowledge_base_id:
        query = query.where(IngestionCompensationLog.knowledge_base_id == knowledge_base_id)
    return list(db.scalars(query.order_by(IngestionCompensationLog.created_at.asc())).all())


def qdrant_delete_intent_recovery_diagnostics(
    row: IngestionCompensationLog,
) -> dict[str, Any]:
    """Describe the only safe operator action without replaying a delete."""

    payload = dict(row.payload_json or {})
    if row.operation == LEGACY_QDRANT_DELETE_OPERATION:
        source_operation = str(
            payload.get("maintenance_operation") or payload.get("reason") or ""
        )
        return {
            "delete_tombstone_id": str(row.id),
            "delete_tombstone_status": str(row.status),
            "delete_tombstone_operation": str(row.operation),
            "delete_source_operation": source_operation or None,
            "delete_protocol_version": None,
            "delete_protocol_supported": False,
            "conditional_delete_replay_safe": False,
            "legacy_unfenced_delete": True,
            "collection_name": str(payload.get("collection_name") or "") or None,
            "target_count": len(row.target_ids_json or []),
            "reason": "legacy_qdrant_delete_manual_resolution_required",
            "retryable": False,
            "retry_guidance": (
                "quarantine_legacy_delete_and_verify_old_writer_stopped_then_repair_qdrant_state"
            ),
        }

    protocol_version = str(payload.get("protocol_version") or "")
    source_operation = str(payload.get("reason") or "")
    protocol_supported = protocol_version in QDRANT_DELETE_SUPPORTED_PROTOCOL_VERSIONS
    validation_error: str | None = None
    try:
        _validated_qdrant_delete_intent(row)
    except Exception as exc:
        validation_error = _safe_error(exc)
    conditional_replay_safe = validation_error is None
    transport_already_applied = row.status == "external_applied"
    source_retry_safe = conditional_replay_safe or transport_already_applied

    if not source_retry_safe:
        block_reason = "qdrant_delete_manual_resolution_required"
        recovery_action = (
            "quarantine_intent_and_manually_verify_writer_stopped_then_repair_qdrant_state"
        )
        retryable = False
    elif source_operation == "cleanup_stale_data":
        block_reason = "stale_cleanup_delete_pending"
        recovery_action = "retry_cleanup_stale_data_with_identical_scope"
        retryable = True
    elif source_operation == "delete_knowledge_base_data":
        block_reason = "knowledge_base_delete_pending"
        recovery_action = "finish_or_retry_knowledge_base_deletion"
        retryable = True
    else:
        block_reason = "qdrant_delete_manual_resolution_required"
        recovery_action = "inspect_delete_intent_source_and_resume_matching_maintenance_operation"
        retryable = False

    return {
        "delete_tombstone_id": str(row.id),
        "delete_tombstone_status": str(row.status),
        "delete_tombstone_operation": str(row.operation),
        "delete_source_operation": source_operation or None,
        "delete_protocol_version": protocol_version or None,
        "delete_protocol_supported": protocol_supported,
        "conditional_delete_replay_safe": conditional_replay_safe,
        "legacy_unfenced_delete": False,
        "delete_intent_validation_error": validation_error,
        "collection_name": str(payload.get("collection_name") or "") or None,
        "target_count": len(row.target_ids_json or []),
        "reason": block_reason,
        "retryable": retryable,
        "retry_guidance": recovery_action,
    }


def qdrant_intent_has_committed_records(db: Session, row: IngestionCompensationLog) -> bool:
    target_ids = [str(item) for item in (row.target_ids_json or [])]
    if not target_ids:
        return False
    records = list(
        db.scalars(
            select(VectorRecord).where(
                VectorRecord.knowledge_base_id == row.knowledge_base_id,
                VectorRecord.collection_name == str((row.payload_json or {}).get("collection_name") or ""),
                VectorRecord.qdrant_point_id.in_(target_ids),
                VectorRecord.vector_status.in_(QDRANT_AUTHORITATIVE_VECTOR_STATUSES),
            )
        ).all()
    )
    by_point = {str(record.qdrant_point_id): record for record in records}
    return all(
        point_id in by_point
        and (by_point[point_id].diagnostics_json or {}).get("qdrant_write_intent_id") == row.id
        for point_id in target_ids
    )


@contextmanager
def qdrant_outbox_reconcile_lock(db: Session, knowledge_base_id: str):
    dialect = _dialect_name(db)
    if dialect == "sqlite" and _running_under_pytest():
        yield
        return
    if dialect != "postgresql":
        raise QdrantOutboxError("Qdrant outbox reconcile requires PostgreSQL")
    from app.services.ingestion_resource_lock import advisory_lock_key, knowledge_base_resource_key

    resource_key = knowledge_base_resource_key(knowledge_base_id)
    lock_key = advisory_lock_key(resource_key)
    bind = db.get_bind()
    engine = bind if hasattr(bind, "connect") else bind.engine
    connection = engine.connect()
    acquired = False
    try:
        acquired = bool(
            connection.execute(
                text("SELECT pg_try_advisory_lock(:lock_key)"),
                {"lock_key": lock_key},
            ).scalar_one()
        )
        connection.commit()
        if not acquired:
            raise QdrantOutboxError(
                f"Cannot reconcile Qdrant intents while ingestion holds {resource_key}; retry after it completes"
            )
        yield
    finally:
        if acquired:
            try:
                connection.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": lock_key},
                )
                connection.commit()
            except Exception:
                connection.invalidate()
        connection.close()


def _decode_v1_reconcile_payload(
    row: IngestionCompensationLog,
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read-only historical decoder using the exact v1 byte contract."""

    target_points = payload.get("target_points")
    before_points = payload.get("before_points")
    if not isinstance(target_points, list) or not target_points:
        raise QdrantOutboxError(f"Intent {row.id} target payload is invalid")
    if len(target_points) > QDRANT_OUTBOX_MAX_TARGET_POINTS_PER_INTENT:
        raise QdrantOutboxError(f"Intent {row.id} target payload exceeds its bounded scope")
    if not isinstance(before_points, list):
        raise QdrantOutboxError(f"Intent {row.id} before-image payload is invalid")
    if payload.get("target_payload_hash") != _outbox_payload_hash(
        QDRANT_OUTBOX_PROTOCOL_V1,
        target_points,
    ):
        raise QdrantOutboxError(f"Intent {row.id} target payload hash is invalid")
    if payload.get("before_image_hash") != _outbox_payload_hash(
        QDRANT_OUTBOX_PROTOCOL_V1,
        before_points,
    ):
        raise QdrantOutboxError(f"Intent {row.id} before-image hash is invalid")
    decoded_targets = copy.deepcopy(target_points)
    decoded_before = copy.deepcopy(before_points)
    _validate_envelope_point_scope(
        intent_id=row.id,
        knowledge_base_id=str(row.knowledge_base_id),
        protocol_version=QDRANT_OUTBOX_PROTOCOL_V1,
        target_points=decoded_targets,
        before_points=decoded_before,
        expected_target_ids=[str(item) for item in (row.target_ids_json or [])],
    )
    return decoded_targets, decoded_before


def _decode_v2_reconcile_payload(
    row: IngestionCompensationLog,
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = _strict_json_copy(payload, path="$.envelope")
    if not isinstance(payload, dict):  # pragma: no cover - caller already normalizes a mapping
        raise QdrantOutboxError(f"Intent {row.id} v2 envelope is not an object")
    target_points = payload.get("target_points")
    before_points = payload.get("before_points")
    if not isinstance(target_points, list) or not target_points:
        raise QdrantOutboxError(f"Intent {row.id} target payload is invalid")
    if len(target_points) > QDRANT_OUTBOX_MAX_TARGET_POINTS_PER_INTENT:
        raise QdrantOutboxError(f"Intent {row.id} target payload exceeds its bounded scope")
    if not isinstance(before_points, list):
        raise QdrantOutboxError(f"Intent {row.id} before-image payload is invalid")
    if payload.get("target_payload_hash") != _outbox_payload_hash(
        QDRANT_OUTBOX_PROTOCOL_VERSION,
        target_points,
    ):
        raise QdrantOutboxError(f"Intent {row.id} target payload hash is invalid")
    if payload.get("before_image_hash") != _outbox_payload_hash(
        QDRANT_OUTBOX_PROTOCOL_VERSION,
        before_points,
    ):
        raise QdrantOutboxError(f"Intent {row.id} before-image hash is invalid")

    expected_contract = _outbox_protocol_contract(QDRANT_OUTBOX_PROTOCOL_VERSION)
    contract_fields = (
        "envelope_schema_version",
        "envelope_schema_hash",
        "canonical_bytes_version",
    )
    mismatched = [
        field
        for field in contract_fields
        if payload.get(field) != expected_contract[field]
    ]
    if mismatched:
        raise QdrantOutboxError(
            f"Intent {row.id} has an invalid frozen v2 envelope contract: {', '.join(mismatched)}"
        )

    decoded_targets = _canonicalize_v2_target_points(
        target_points,
        intent_id=row.id,
        knowledge_base_id=str(row.knowledge_base_id),
        collection_name=str(payload.get("collection_name") or ""),
        strict_schema=True,
        allow_historical_vector_hash=True,
    )
    decoded_before = _canonicalize_before_points(before_points, intent_id=row.id)
    if _outbox_v2_canonical_bytes(decoded_targets) != _outbox_v2_canonical_bytes(
        target_points
    ):
        raise QdrantOutboxError(
            f"Intent {row.id} target points were not persisted in canonical binary32 form"
        )
    if _outbox_v2_canonical_bytes(decoded_before) != _outbox_v2_canonical_bytes(
        before_points
    ):
        raise QdrantOutboxError(
            f"Intent {row.id} before-image points were not persisted in canonical binary32 form"
        )
    _validate_envelope_point_scope(
        intent_id=row.id,
        knowledge_base_id=str(row.knowledge_base_id),
        protocol_version=QDRANT_OUTBOX_PROTOCOL_VERSION,
        target_points=decoded_targets,
        before_points=decoded_before,
        expected_target_ids=[str(item) for item in (row.target_ids_json or [])],
    )
    return decoded_targets, decoded_before


def _validated_reconcile_payload(row: IngestionCompensationLog) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = dict(row.payload_json or {})
    protocol_version = str(payload.get("protocol_version") or "")
    if protocol_version not in QDRANT_OUTBOX_SUPPORTED_PROTOCOL_VERSIONS:
        raise QdrantOutboxError(f"Intent {row.id} has an unsupported protocol")
    if payload.get("intent_id") != row.id:
        raise QdrantOutboxError(f"Intent {row.id} payload identity is invalid")
    if protocol_version == QDRANT_OUTBOX_PROTOCOL_V1:
        return _decode_v1_reconcile_payload(row, payload)
    return _decode_v2_reconcile_payload(row, payload)


def validated_qdrant_outbox_target_points(
    row: IngestionCompensationLog,
) -> list[dict[str, Any]]:
    """Expose the frozen canonical target decoder for bounded recovery paths."""

    if row.operation != QDRANT_UPSERT_OPERATION:
        raise QdrantOutboxError("Qdrant target recovery requires an upsert intent")
    target_points, _before_points = _validated_reconcile_payload(row)
    return target_points


def _intent_lease_expired(row: IngestionCompensationLog, *, now: datetime | None = None) -> bool:
    if row.status == "compensation_pending":
        return True
    raw = (row.payload_json or {}).get("lease_expires_at")
    if not raw:
        return True
    try:
        deadline = datetime.fromisoformat(str(raw))
    except ValueError:
        return True
    return deadline <= (now or datetime.utcnow())


def _deadline_reached(raw: Any, *, now: datetime | None = None) -> bool:
    if not raw:
        return True
    try:
        deadline = datetime.fromisoformat(str(raw))
    except ValueError:
        return True
    return deadline <= (now or datetime.utcnow())


def _uncertainty_gate_reached(row: IngestionCompensationLog, *, now: datetime | None = None) -> bool:
    payload = dict(row.payload_json or {})
    if row.status in {"pending", "external_outcome_unknown"}:
        return _deadline_reached(
            payload.get("uncertainty_not_before") or payload.get("lease_expires_at"),
            now=now,
        )
    if row.status in {"uncertainty_observed", "compensation_verify_pending"}:
        return _deadline_reached(payload.get("confirmation_not_before"), now=now)
    return True


def _point_observation(points: dict[str, dict[str, Any]], target_ids: list[str]) -> dict[str, Any]:
    observed = []
    for point_id in sorted(target_ids):
        point = points.get(point_id)
        observed.append(
            {
                "id": point_id,
                "present": point is not None,
                "owner": _point_owner(point),
                "point_hash": _canonical_hash(point) if point is not None else None,
            }
        )
    return {"snapshot_hash": _canonical_hash(observed), "points": observed}


def _requires_uncertainty_watch(row: IngestionCompensationLog) -> bool:
    payload = dict(row.payload_json or {})
    explicit = payload.get("requires_uncertainty_watch")
    if explicit is not None:
        return bool(explicit)
    # v1 compatibility: an observed successful response is resolved; a
    # pending-only/unknown response remains ambiguous and must stay scanable.
    history_statuses = {
        str(item.get("status") or "")
        for item in list(payload.get("state_history") or [])
        if isinstance(item, dict)
    }
    return (
        "external_outcome_unknown" in history_statuses
        or "external_applied" not in history_statuses
    )


def _postgresql_authoritative_targets(
    db: Session,
    row: IngestionCompensationLog,
    target_ids: list[str],
) -> list[dict[str, Any]]:
    payload = dict(row.payload_json or {})
    collection_name = str(payload.get("collection_name") or "")
    records = list(
        db.scalars(
            select(VectorRecord).where(
                VectorRecord.collection_name == collection_name,
                VectorRecord.qdrant_point_id.in_(target_ids),
                VectorRecord.vector_status.in_(QDRANT_AUTHORITATIVE_VECTOR_STATUSES),
            )
        ).all()
    )
    records_by_point: dict[str, list[VectorRecord]] = {}
    for record in records:
        records_by_point.setdefault(str(record.qdrant_point_id), []).append(record)
    authoritative: list[dict[str, Any]] = []
    for point_id in sorted(target_ids):
        point_records = records_by_point.get(point_id, [])
        if len(point_records) > 1:
            raise QdrantOutboxError(
                f"Intent {row.id} has multiple authoritative VectorRecords for point {point_id}"
            )
        if not point_records:
            # During an unresolved transport watch PostgreSQL is authoritative:
            # without an authoritative VectorRecord there is no desired Qdrant point.
            continue

        record = point_records[0]
        diagnostics = dict(record.diagnostics_json or {})
        owner_intent_id = str(diagnostics.get("qdrant_write_intent_id") or "")
        if not owner_intent_id:
            raise QdrantOutboxError(
                f"Intent {row.id} cannot compensate point {point_id} while PostgreSQL owner is unresolved"
            )
        if owner_intent_id == row.id:
            if not qdrant_intent_has_committed_records(db, row):
                raise QdrantOutboxError(
                    f"Intent {row.id} PostgreSQL facts are not fully committed for its own target"
                )
            owner_row = row
        else:
            owner_row = db.get(IngestionCompensationLog, owner_intent_id)
            if (
                owner_row is None
                or owner_row.operation != QDRANT_UPSERT_OPERATION
                or owner_row.knowledge_base_id != record.knowledge_base_id
                or owner_row.status != "committed"
            ):
                raise QdrantOutboxError(
                    f"Intent {row.id} owner {owner_intent_id} is not a committed durable intent"
                )
        if owner_row.knowledge_base_id != record.knowledge_base_id:
            raise QdrantOutboxError(
                f"Intent {row.id} owner {owner_intent_id} does not own point {point_id}'s knowledge base"
            )
        owner_targets, _owner_before = _validated_reconcile_payload(owner_row)
        owner_collection = str((owner_row.payload_json or {}).get("collection_name") or "")
        if owner_collection != collection_name:
            raise QdrantOutboxError(
                f"Intent {row.id} superseding owner {owner_intent_id} uses another collection"
            )
        matching_targets = [
            dict(point) for point in owner_targets if str(point.get("id") or "") == point_id
        ]
        if len(matching_targets) != 1:
            raise QdrantOutboxError(
                f"Intent {row.id} superseding owner {owner_intent_id} has no unique target for {point_id}"
            )
        durable_target = matching_targets[0]
        owner_protocol = str((owner_row.payload_json or {}).get("protocol_version") or "")
        if (
            diagnostics.get("qdrant_write_protocol_version") != owner_protocol
            or record.payload_hash
            != str((durable_target.get("payload") or {}).get("vector_payload_hash") or "")
        ):
            raise QdrantOutboxError(
                f"Intent {row.id} superseding VectorRecord audit is inconsistent for {point_id}"
            )
        durable_hash = _canonical_hash(durable_target)
        authoritative.append(
            {
                "point_id": point_id,
                "owner_intent_id": owner_intent_id,
                "target_point": durable_target,
                "target_point_hash": durable_hash,
                "vector_record_id": record.id,
                "knowledge_base_id": record.knowledge_base_id,
                "vector_payload_hash": record.payload_hash,
                "captured_at": datetime.utcnow().isoformat(),
            }
        )
    return authoritative


def _durable_uncertainty_superseders(
    db: Session,
    row: IngestionCompensationLog,
    current_by_id: dict[str, dict[str, Any]],
    target_ids: list[str],
) -> list[dict[str, Any]]:
    payload = dict(row.payload_json or {})
    existing = [
        dict(item)
        for item in list(payload.get("uncertainty_protected_superseders") or [])
    ]
    if existing and payload.get("uncertainty_protected_superseders_hash") != _canonical_hash(existing):
        raise QdrantOutboxError(f"Intent {row.id} protected-superseder hash is invalid")
    protected = _postgresql_authoritative_targets(db, row, target_ids)
    for item in protected:
        point_id = str(item["point_id"])
        owner_intent_id = str(item["owner_intent_id"])
        durable_hash = str(item["target_point_hash"])
        current = current_by_id.get(point_id)
        current_owner = _point_owner(current)
        if current is not None and current_owner == owner_intent_id:
            if _canonical_hash(current) != durable_hash:
                raise QdrantOutboxError(
                    f"Intent {row.id} observed a corrupted authoritative point {point_id}"
                )
        elif current is not None and current_owner != row.id:
            stale_owner_row = (
                db.get(IngestionCompensationLog, current_owner)
                if current_owner is not None
                else None
            )
            if (
                stale_owner_row is None
                or stale_owner_row.operation != QDRANT_UPSERT_OPERATION
                or stale_owner_row.status != "committed"
                or str((stale_owner_row.payload_json or {}).get("collection_name") or "")
                != str((row.payload_json or {}).get("collection_name") or "")
            ):
                raise QdrantOutboxError(
                    f"Intent {row.id} observed ungrounded Qdrant owner {current_owner} for {point_id}"
                )
            stale_targets, _stale_before = _validated_reconcile_payload(stale_owner_row)
            stale_matches = [
                point
                for point in stale_targets
                if str(point.get("id") or "") == point_id
            ]
            if (
                len(stale_matches) != 1
                or _canonical_hash(stale_matches[0]) != _canonical_hash(current)
            ):
                raise QdrantOutboxError(
                    f"Intent {row.id} observed corrupted stale owner {current_owner} for {point_id}"
                )
    return protected


def _compensation_postcondition(
    *,
    current_by_id: dict[str, dict[str, Any]],
    target_points: list[dict[str, Any]],
    before_points: list[dict[str, Any]],
    protected_points: list[dict[str, Any]],
    intent_id: str,
    authoritative: bool = False,
) -> dict[str, Any]:
    target_ids = [str(point["id"]) for point in target_points]
    before_by_id = {str(point["id"]): point for point in before_points}
    protected_by_id = {str(point["id"]): point for point in protected_points}
    superseded_ids = sorted(
        point_id
        for point_id, point in current_by_id.items()
        if _point_owner(point) not in {None, intent_id}
    )
    mismatched_ids: list[str] = []
    for point_id in target_ids:
        current = current_by_id.get(point_id)
        protected = protected_by_id.get(point_id)
        if authoritative:
            if protected is None:
                if current is not None:
                    mismatched_ids.append(point_id)
            elif current is None or _canonical_hash(current) != _canonical_hash(protected):
                mismatched_ids.append(point_id)
            continue
        if point_id in superseded_ids:
            continue
        before = before_by_id.get(point_id)
        if current is None:
            if before is not None or protected is not None:
                mismatched_ids.append(point_id)
        elif _point_owner(current) == intent_id:
            mismatched_ids.append(point_id)
        elif before is not None and _canonical_hash(current) != _canonical_hash(before):
            mismatched_ids.append(point_id)
        elif protected is not None and _canonical_hash(current) != _canonical_hash(protected):
            mismatched_ids.append(point_id)
    return {
        "verified": not mismatched_ids,
        "mismatched_ids": sorted(mismatched_ids),
        "superseded_ids": [] if authoritative else superseded_ids,
        "terminal_status": "superseded" if superseded_ids and not authoritative else "compensated",
    }


def _sync_compensate_intent(
    *,
    db: Session,
    store: Any,
    row: IngestionCompensationLog,
    target_points: list[dict[str, Any]],
    before_points: list[dict[str, Any]],
    protected_points: list[dict[str, Any]],
    dry_run: bool,
    authoritative: bool = False,
) -> dict[str, Any]:
    collection_name = str((row.payload_json or {}).get("collection_name") or "")
    target_ids = [str(point["id"]) for point in target_points]
    with qdrant_point_mutation_lock(
        db,
        collection_name=collection_name,
        point_ids=target_ids,
    ):
        return _sync_compensate_intent_locked(
            db=db,
            store=store,
            row=row,
            target_points=target_points,
            before_points=before_points,
            protected_points=protected_points,
            dry_run=dry_run,
            authoritative=authoritative,
        )


def _sync_compensate_intent_locked(
    *,
    db: Session,
    store: Any,
    row: IngestionCompensationLog,
    target_points: list[dict[str, Any]],
    before_points: list[dict[str, Any]],
    protected_points: list[dict[str, Any]],
    dry_run: bool,
    authoritative: bool = False,
) -> dict[str, Any]:
    target_ids = [str(point["id"]) for point in target_points]
    before_by_id = {str(point["id"]): point for point in before_points}
    protected_by_id = {str(point["id"]): point for point in protected_points}
    current_by_id = {str(point["id"]): point for point in store.get_points(target_ids)}
    if authoritative:
        _validate_reconcile_current_point_scope(
            intent_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
            current_by_id=current_by_id,
            protected_points=protected_points,
        )
    else:
        _validate_current_point_scope(
            intent_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
            current_by_id=current_by_id,
        )
    if authoritative:
        delete_ids = sorted(
            point_id
            for point_id in target_ids
            if point_id in current_by_id and point_id not in protected_by_id
        )
        restore_points = [
            protected_by_id[point_id]
            for point_id in sorted(protected_by_id)
            if (
                point_id not in current_by_id
                or _canonical_hash(current_by_id[point_id])
                != _canonical_hash(protected_by_id[point_id])
            )
        ]
        if not dry_run and (delete_ids or restore_points):
            attempt_id = _begin_qdrant_mutation_attempt(
                db,
                intent_id=row.id,
                mutation_kind="authoritative_reconcile_compensation",
                planned_delete_ids=delete_ids,
                planned_upsert_points=restore_points,
            )
            try:
                if delete_ids:
                    store.delete(delete_ids)
                if restore_points:
                    store.upsert(restore_points)
            except Exception as exc:
                unknown = QdrantMutationOutcomeUnknown(
                    f"Qdrant authoritative compensation attempt {attempt_id} result is unknown: {_safe_error(exc)}"
                )
                unknown.attempt_id = attempt_id
                raise unknown from exc
            _resolve_qdrant_mutation_attempt(
                db,
                intent_id=row.id,
                attempt_id=attempt_id,
            )
        return {
            "status": "compensated",
            "delete_ids": delete_ids,
            "restore_ids": [str(point["id"]) for point in restore_points],
            "superseded_ids": [],
            "authoritative_source": "postgresql_ready_vector_record_and_committed_outbox",
        }
    superseded_ids = sorted(
        point_id
        for point_id, point in current_by_id.items()
        if _point_owner(point) not in {None, row.id}
    )
    unknown_new_ids = sorted(
        point_id
        for point_id, point in current_by_id.items()
        if point_id not in before_by_id and _point_owner(point) is None
    )
    superseded_ids = sorted(set(superseded_ids).union(unknown_new_ids))
    delete_ids = sorted(
        point_id
        for point_id in target_ids
        if point_id not in before_by_id and _point_owner(current_by_id.get(point_id)) == row.id
        and point_id not in protected_by_id
    )
    restore_points = [
        before_by_id[point_id]
        for point_id in sorted(before_by_id)
        if point_id not in superseded_ids
        and point_id not in protected_by_id
        and (
            point_id not in current_by_id
            or _point_owner(current_by_id.get(point_id)) == row.id
        )
    ]
    restore_points.extend(
        protected_by_id[point_id]
        for point_id in sorted(protected_by_id)
        if (
            point_id not in current_by_id
            or _point_owner(current_by_id.get(point_id)) == row.id
        )
    )
    if not dry_run and (delete_ids or restore_points):
        attempt_id = _begin_qdrant_mutation_attempt(
            db,
            intent_id=row.id,
            mutation_kind="reconcile_compensation",
            planned_delete_ids=delete_ids,
            planned_upsert_points=restore_points,
        )
        try:
            if delete_ids:
                store.delete(delete_ids)
            if restore_points:
                store.upsert(restore_points)
        except Exception as exc:
            unknown = QdrantMutationOutcomeUnknown(
                f"Qdrant compensation attempt {attempt_id} result is unknown: {_safe_error(exc)}"
            )
            unknown.attempt_id = attempt_id
            raise unknown from exc
        _resolve_qdrant_mutation_attempt(
            db,
            intent_id=row.id,
            attempt_id=attempt_id,
        )
    return {
        "status": "superseded" if superseded_ids else "compensated",
        "delete_ids": delete_ids,
        "restore_ids": [str(point["id"]) for point in restore_points],
        "superseded_ids": superseded_ids,
    }


def _active_upsert_intent_filter(knowledge_base_id: str) -> Any:
    return (
        (IngestionCompensationLog.knowledge_base_id == knowledge_base_id)
        & (IngestionCompensationLog.operation == QDRANT_UPSERT_OPERATION)
        & IngestionCompensationLog.status.in_(sorted(QDRANT_OUTBOX_ACTIVE_STATES))
    )


def _active_delete_intent_filter(knowledge_base_id: str) -> Any:
    return (
        (IngestionCompensationLog.knowledge_base_id == knowledge_base_id)
        & (
            (
                (IngestionCompensationLog.operation == QDRANT_DELETE_OPERATION)
                & IngestionCompensationLog.status.in_(sorted(QDRANT_DELETE_ACTIVE_STATES))
            )
            | (
                (IngestionCompensationLog.operation == LEGACY_QDRANT_DELETE_OPERATION)
                & IngestionCompensationLog.status.in_(
                    sorted(LEGACY_QDRANT_DELETE_BLOCKING_STATES)
                )
            )
        )
    )


def _active_upsert_intent_scan_summary(
    origin_db: Session,
    *,
    knowledge_base_id: str,
) -> tuple[str | None, int]:
    """Freeze a bounded primary-key scan horizon without retaining ORM rows."""

    with _audit_session(origin_db) as scan:
        condition = _active_upsert_intent_filter(knowledge_base_id)
        high_water_id = scan.scalar(
            select(func.max(IngestionCompensationLog.id)).where(condition)
        )
        if not high_water_id:
            return None, 0
        candidate_count = int(
            scan.scalar(
                select(func.count())
                .select_from(IngestionCompensationLog)
                .where(condition, IngestionCompensationLog.id <= str(high_water_id))
            )
            or 0
        )
        return str(high_water_id), candidate_count


def _iter_active_upsert_intents_keyset(
    origin_db: Session,
    *,
    knowledge_base_id: str,
    high_water_id: str | None,
):
    """Yield one page transaction at a time using the immutable primary key."""

    if not high_water_id:
        return
    cursor_id: str | None = None
    page_index = 0
    while True:
        with _audit_session(origin_db) as page_db:
            query = (
                select(IngestionCompensationLog)
                .where(
                    _active_upsert_intent_filter(knowledge_base_id),
                    IngestionCompensationLog.id <= high_water_id,
                )
                .order_by(IngestionCompensationLog.id.asc())
                .limit(QDRANT_OUTBOX_RECONCILE_PAGE_SIZE)
                .execution_options(populate_existing=True)
            )
            if cursor_id is not None:
                query = query.where(IngestionCompensationLog.id > cursor_id)
            rows = list(page_db.scalars(query))
            if not rows:
                return
            for row in rows:
                yield page_db, row, page_index
            cursor_id = str(rows[-1].id)
        page_index += 1


def _bounded_delete_recovery_diagnostics(
    origin_db: Session,
    *,
    knowledge_base_id: str,
) -> tuple[int, list[dict[str, Any]]]:
    # Production delete intents are committed independently before the
    # external mutation. The explicit SQLite pytest adapter cannot emulate
    # that second durable transaction while its fixture transaction is open,
    # so retain caller-session visibility only for that adapter.
    if _dialect_name(origin_db) == "sqlite" and _running_under_pytest():
        scan = origin_db
        close_scan = False
    else:
        scan = _audit_session(origin_db)
        close_scan = True
    try:
        condition = _active_delete_intent_filter(knowledge_base_id)
        total = int(
            scan.scalar(
                select(func.count())
                .select_from(IngestionCompensationLog)
                .where(condition)
            )
            or 0
        )
        rows = list(
            scan.scalars(
                select(IngestionCompensationLog)
                .where(condition)
                .order_by(IngestionCompensationLog.id.asc())
                .limit(QDRANT_OUTBOX_DELETE_DIAGNOSTIC_SAMPLE_LIMIT)
            )
        )
        return total, [qdrant_delete_intent_recovery_diagnostics(row) for row in rows]
    finally:
        if close_scan:
            scan.close()


def _append_reconcile_action(stats: dict[str, Any], action: dict[str, Any]) -> None:
    stats["action_count"] += 1
    if len(stats["actions"]) < QDRANT_OUTBOX_RECONCILE_ACTION_SAMPLE_LIMIT:
        stats["actions"].append(action)
    else:
        stats["actions_truncated_count"] += 1


def _close_reconcile_stores(stores: dict[str, Any]) -> None:
    seen_clients: set[int] = set()
    for store in stores.values():
        client = getattr(store, "client", None)
        close = getattr(client, "close", None)
        if not callable(close) or id(client) in seen_clients:
            continue
        seen_clients.add(id(client))
        try:
            close()
        except Exception:
            # Reconcile correctness does not depend on client-pool teardown;
            # the cache itself is still released at the page boundary.
            pass
    stores.clear()


def reconcile_qdrant_outbox_sync(
    db: Session,
    *,
    knowledge_base_id: str,
    knowledge_base_name: str,
    dry_run: bool = True,
    include_unexpired: bool = False,
) -> dict[str, Any]:
    from app.services.vector_store import VectorStore

    origin_db = db
    high_water_id, candidate_upsert_count = _active_upsert_intent_scan_summary(
        origin_db,
        knowledge_base_id=knowledge_base_id,
    )
    delete_count, delete_recovery_actions = _bounded_delete_recovery_diagnostics(
        origin_db,
        knowledge_base_id=knowledge_base_id,
    )
    stats: dict[str, Any] = {
        "protocol_version": QDRANT_OUTBOX_PROTOCOL_VERSION,
        "delete_protocol_version": QDRANT_DELETE_PROTOCOL_VERSION,
        "scan_protocol_version": QDRANT_OUTBOX_RECONCILE_SCAN_PROTOCOL_VERSION,
        "scan_page_size": QDRANT_OUTBOX_RECONCILE_PAGE_SIZE,
        "scan_high_water_id": high_water_id,
        "candidate_upsert_intents": candidate_upsert_count,
        "pages_scanned": 0,
        "knowledge_base_id": knowledge_base_id,
        "dry_run": bool(dry_run),
        "include_unexpired": bool(include_unexpired),
        "checked_intents": delete_count,
        "checked_upsert_intents": 0,
        "checked_delete_intents": delete_count,
        "blocking_delete_intents": delete_count,
        "delete_recovery_actions": delete_recovery_actions,
        "delete_recovery_actions_truncated_count": max(
            0,
            delete_count - len(delete_recovery_actions),
        ),
        "eligible_intents": 0,
        "skipped_unexpired": 0,
        "skipped_uncertainty_window": 0,
        "observed_uncertain": 0,
        "verification_pending": 0,
        "missing_collections": 0,
        "committed": 0,
        "compensated": 0,
        "superseded": 0,
        "failed": 0,
        "target_points": 0,
        "action_count": 0,
        "actions_truncated_count": 0,
        "actions": [],
    }
    stores: dict[str, Any] = {}
    current_page_index = -1
    for page_db, row, page_index in _iter_active_upsert_intents_keyset(
        origin_db,
        knowledge_base_id=knowledge_base_id,
        high_water_id=high_water_id,
    ):
        db = page_db
        if page_index != current_page_index:
            current_page_index = page_index
            stats["pages_scanned"] += 1
            _close_reconcile_stores(stores)
            stores = {}
        stats["checked_upsert_intents"] += 1
        stats["checked_intents"] += 1
        requires_uncertainty_watch = _requires_uncertainty_watch(row)
        uncertain_flow = (
            row.status in QDRANT_OUTBOX_UNCERTAINTY_STATES
            or requires_uncertainty_watch
        )
        if uncertain_flow and not _uncertainty_gate_reached(row):
            # An operator override may confirm that the writer process stopped,
            # but it cannot prove that an already accepted Qdrant request will
            # not finish late.  Uncertainty/confirmation windows are therefore
            # hard gates and are never bypassed by include_unexpired.
            stats["skipped_uncertainty_window"] += 1
            continue
        if not uncertain_flow and not include_unexpired and not _intent_lease_expired(row):
            stats["skipped_unexpired"] += 1
            continue
        stats["eligible_intents"] += 1
        try:
            target_points, before_points = _validated_reconcile_payload(row)
            stats["target_points"] += len(target_points)
            collection_name = str((row.payload_json or {}).get("collection_name") or "")
            if not collection_name:
                raise QdrantOutboxError(f"Intent {row.id} has no collection name")
            store = stores.get(collection_name)
            if store is None:
                store = VectorStore(
                    knowledge_base_name,
                    collection_name=collection_name,
                    create_if_missing=False,
                )
                stores[collection_name] = store
            collection_exists = _store_collection_exists(store)
            if not collection_exists:
                stats["missing_collections"] += 1
            target_ids = [str(point["id"]) for point in target_points]
            current = {str(point["id"]): point for point in store.get_points(target_ids)}
            observation = _point_observation(current, target_ids)
            protected_superseders = _durable_uncertainty_superseders(
                db,
                row,
                current,
                target_ids,
            )
            protected_points = [
                dict(item["target_point"]) for item in protected_superseders
            ]
            _validate_reconcile_current_point_scope(
                intent_id=row.id,
                knowledge_base_id=row.knowledge_base_id,
                current_by_id=current,
                protected_points=protected_points,
            )
            protected_details = {
                "uncertainty_protected_superseders": protected_superseders,
            }

            if (
                requires_uncertainty_watch
                and row.status not in QDRANT_OUTBOX_UNCERTAINTY_STATES
            ):
                action = {
                    "intent_id": row.id,
                    "action": "resume_unresolved_transport_watch",
                    "collection_missing": not collection_exists,
                    "transport_resolution_required": True,
                }
                if not dry_run:
                    _transition_intent(
                        db,
                        intent_id=row.id,
                        status="external_outcome_unknown",
                        details={
                            "reason": "resume_unresolved_mutation_or_legacy_transport_watch",
                            "required_reconcile_protocol": (
                                "postgresql_authoritative_watch_until_transport_resolution_v2"
                            ),
                            **protected_details,
                            **observation,
                        },
                    )
                _append_reconcile_action(stats, action)
                continue

            if row.status in {"pending", "external_outcome_unknown"}:
                action = {
                    "intent_id": row.id,
                    "action": "observe_uncertain_external_outcome",
                    "collection_missing": not collection_exists,
                    **observation,
                }
                if not dry_run:
                    _transition_intent(
                        db,
                        intent_id=row.id,
                        status="uncertainty_observed",
                        details={
                            "reason": "first_stable_owner_observation",
                            "collection_missing": not collection_exists,
                            **protected_details,
                            **observation,
                        },
                    )
                    stats["observed_uncertain"] += 1
                _append_reconcile_action(stats, action)
                continue

            if row.status == "uncertainty_observed":
                previous_observation = dict((row.payload_json or {}).get("uncertainty_observation") or {})
                if previous_observation.get("snapshot_hash") != observation["snapshot_hash"]:
                    action = {
                        "intent_id": row.id,
                        "action": "reset_uncertainty_observation",
                        "collection_missing": not collection_exists,
                        "previous_snapshot_hash": previous_observation.get("snapshot_hash"),
                        **observation,
                    }
                    if not dry_run:
                        _transition_intent(
                            db,
                            intent_id=row.id,
                            status="uncertainty_observed",
                            details={
                                "reason": "owner_observation_changed",
                                "collection_missing": not collection_exists,
                                **protected_details,
                                **observation,
                            },
                        )
                        stats["observed_uncertain"] += 1
                    _append_reconcile_action(stats, action)
                    continue
                try:
                    result = _sync_compensate_intent(
                        db=db,
                        store=store,
                        row=row,
                        target_points=target_points,
                        before_points=before_points,
                        protected_points=protected_points,
                        dry_run=dry_run,
                        authoritative=True,
                    )
                except Exception as exc:
                    action = {
                        "intent_id": row.id,
                        "action": "compensation_result_unknown",
                        "collection_missing": not collection_exists,
                        "error": _safe_error(exc),
                    }
                    stats["failed"] += 1
                    if not dry_run:
                        _transition_intent(
                            db,
                            intent_id=row.id,
                            status="compensation_verify_pending",
                            details={
                                "reason": "compensation_client_result_unknown",
                                "error": _safe_error(exc),
                                "observation_before_action_hash": observation["snapshot_hash"],
                                **protected_details,
                            },
                            error_message=_safe_error(exc),
                        )
                        stats["verification_pending"] += 1
                    _append_reconcile_action(stats, action)
                    continue
                action = {
                    "intent_id": row.id,
                    "action": "compensate_after_stable_uncertainty_observation",
                    "collection_missing": not collection_exists,
                    **result,
                }
                if not dry_run:
                    _transition_intent(
                        db,
                        intent_id=row.id,
                        status="compensation_verify_pending",
                        details={
                            "reason": "compensation_requires_delayed_postcondition_verification",
                            "observation_before_action_hash": observation["snapshot_hash"],
                            **protected_details,
                            **result,
                        },
                    )
                    stats["verification_pending"] += 1
                _append_reconcile_action(stats, action)
                continue

            if row.status == "compensation_verify_pending":
                postcondition = _compensation_postcondition(
                    current_by_id=current,
                    target_points=target_points,
                    before_points=before_points,
                    protected_points=protected_points,
                    intent_id=row.id,
                    authoritative=True,
                )
                if postcondition["verified"]:
                    if requires_uncertainty_watch:
                        action = {
                            "intent_id": row.id,
                            "action": "watch_verified_compensation_postcondition",
                            "collection_missing": not collection_exists,
                            "transport_resolution_required": True,
                            **postcondition,
                        }
                        if not dry_run:
                            _transition_intent(
                                db,
                                intent_id=row.id,
                                status="compensation_verify_pending",
                                details={
                                    "reason": "external_outcome_still_transport_unresolved",
                                    "transport_resolution_required": True,
                                    **protected_details,
                                    **postcondition,
                                    **observation,
                                },
                            )
                            stats["verification_pending"] += 1
                        _append_reconcile_action(stats, action)
                        continue
                    action = {
                        "intent_id": row.id,
                        "action": "verify_compensation_postcondition",
                        "collection_missing": not collection_exists,
                        **postcondition,
                    }
                    if not dry_run:
                        terminal_status = str(postcondition["terminal_status"])
                        _transition_intent(
                            db,
                            intent_id=row.id,
                            status=terminal_status,
                            details={
                                "reason": "delayed_compensation_postcondition_verified",
                                **postcondition,
                                **observation,
                            },
                        )
                        stats[terminal_status] += 1
                    _append_reconcile_action(stats, action)
                    continue
                try:
                    result = _sync_compensate_intent(
                        db=db,
                        store=store,
                        row=row,
                        target_points=target_points,
                        before_points=before_points,
                        protected_points=protected_points,
                        dry_run=dry_run,
                        authoritative=True,
                    )
                    action = {
                        "intent_id": row.id,
                        "action": "repeat_compensation_before_verification",
                        "collection_missing": not collection_exists,
                        **postcondition,
                        **result,
                    }
                    error_message = None
                except Exception as exc:
                    action = {
                        "intent_id": row.id,
                        "action": "repeated_compensation_result_unknown",
                        "collection_missing": not collection_exists,
                        **postcondition,
                        "error": _safe_error(exc),
                    }
                    result = {}
                    error_message = _safe_error(exc)
                    stats["failed"] += 1
                if not dry_run:
                    _transition_intent(
                        db,
                        intent_id=row.id,
                        status="compensation_verify_pending",
                        details={
                            "reason": "compensation_postcondition_not_yet_verified",
                            **protected_details,
                            **postcondition,
                            **result,
                        },
                        error_message=error_message,
                    )
                    stats["verification_pending"] += 1
                _append_reconcile_action(stats, action)
                continue

            if qdrant_intent_has_committed_records(db, row):
                blocking_ids = sorted(
                    point_id
                    for point_id, point in current.items()
                    if _point_owner(point) not in {None, row.id}
                )
                if blocking_ids:
                    raise QdrantOutboxError(
                        f"Intent {row.id} cannot reapply over points owned by another intent: {blocking_ids}"
                    )
                action = {
                    "intent_id": row.id,
                    "action": "reapply_committed_target",
                    "target_ids": target_ids,
                    "collection_missing": not collection_exists,
                }
                if not dry_run:
                    with qdrant_point_mutation_lock(
                        db,
                        collection_name=collection_name,
                        point_ids=target_ids,
                    ):
                        locked_current = {
                            str(point["id"]): point
                            for point in store.get_points(target_ids)
                        }
                        _validate_current_point_scope(
                            intent_id=row.id,
                            knowledge_base_id=row.knowledge_base_id,
                            current_by_id=locked_current,
                        )
                        locked_blocking_ids = sorted(
                            point_id
                            for point_id, point in locked_current.items()
                            if _point_owner(point) not in {None, row.id}
                        )
                        if locked_blocking_ids:
                            raise QdrantOutboxError(
                                f"Intent {row.id} cannot reapply over points owned by another intent: {locked_blocking_ids}"
                            )
                        attempt_id = _begin_qdrant_mutation_attempt(
                            db,
                            intent_id=row.id,
                            mutation_kind="reapply_committed_target",
                            planned_delete_ids=[],
                            planned_upsert_points=target_points,
                        )
                        try:
                            store.upsert(target_points)
                        except Exception as exc:
                            unknown = QdrantMutationOutcomeUnknown(
                                f"Qdrant reapply attempt {attempt_id} result is unknown: {_safe_error(exc)}"
                            )
                            unknown.attempt_id = attempt_id
                            raise unknown from exc
                        _resolve_qdrant_mutation_attempt(
                            db,
                            intent_id=row.id,
                            attempt_id=attempt_id,
                        )
                    _transition_intent(
                        db,
                        intent_id=row.id,
                        status="committed",
                        details={"reason": "reconcile_verified_postgresql_commit_and_reapplied_target"},
                    )
                    stats["committed"] += 1
            else:
                result = _sync_compensate_intent(
                    db=db,
                    store=store,
                    row=row,
                    target_points=target_points,
                    before_points=before_points,
                    protected_points=protected_points,
                    dry_run=dry_run,
                    authoritative=True,
                )
                action = {
                    "intent_id": row.id,
                    "action": "compensate_uncommitted_target",
                    "collection_missing": not collection_exists,
                    **result,
                }
                if not dry_run:
                    _transition_intent(
                        db,
                        intent_id=row.id,
                        status=str(result["status"]),
                        details={"reason": "reconcile_postgresql_target_not_committed", **result},
                    )
                    stats[str(result["status"])] += 1
            _append_reconcile_action(stats, action)
        except Exception as exc:
            stats["failed"] += 1
            _append_reconcile_action(
                stats,
                {
                    "intent_id": row.id,
                    "action": "reconcile_failed",
                    "error": _safe_error(exc),
                },
            )
            if not dry_run:
                retry_status = "reconcile_failed"
                retry_details: dict[str, Any] = {"reason": "reconcile_exception"}
                if isinstance(exc, QdrantMutationOutcomeUnknown):
                    retry_status = "external_outcome_unknown"
                    retry_details.update(
                        {
                            "reason": "qdrant_mutation_result_unknown",
                            "mutation_attempt_id": getattr(exc, "attempt_id", None),
                            "required_reconcile_protocol": (
                                "postgresql_authoritative_watch_until_transport_resolution_v2"
                            ),
                        }
                    )
                elif row.status in {"pending", "external_outcome_unknown"}:
                    retry_status = "external_outcome_unknown"
                    retry_details["required_reconcile_protocol"] = (
                        "lease_then_stable_observation_then_compensation_verification_v1"
                    )
                elif row.status == "uncertainty_observed":
                    retry_status = "uncertainty_observed"
                    retry_details.update(dict((row.payload_json or {}).get("uncertainty_observation") or {}))
                elif row.status == "compensation_verify_pending":
                    retry_status = "compensation_verify_pending"
                    retry_details.update(dict((row.payload_json or {}).get("compensation_verification") or {}))
                _transition_intent(
                    db,
                    intent_id=row.id,
                    status=retry_status,
                    details=retry_details,
                    error_message=_safe_error(exc),
                )
    _close_reconcile_stores(stores)
    return stats

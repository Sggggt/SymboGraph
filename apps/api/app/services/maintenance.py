from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Iterator, Sequence

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.orm import Session

from app.models import (
    Chunk,
    ChunkContextText,
    ChunkCoordinate,
    ChunkRelationEdge,
    ChunkSpan,
    ChunkStructureEdge,
    ChunkStructureMapping,
    ChunkStructureNode,
    ChunkVersion,
    CitationVerification,
    Document,
    DocumentVersion,
    IngestionBatch,
    IngestionCompensationLog,
    KnowledgeBase,
    PolicyState,
    RQPrefixMembership,
    VectorRecord,
)
from app.services.graph_state_hashes import canonical_policy_state_hash
from app.services.vector_store import VectorStore, canonical_embedding_vector


TERMINAL_BATCH_STATES = {"completed", "failed", "partial_failed", "skipped", "cancelled", "cancel_failed"}
VECTOR_RECONCILE_BATCH_SIZE = 256
VECTOR_RECONCILE_DIAGNOSTIC_SAMPLE_LIMIT = 64
KNOWLEDGE_BASE_DELETE_LOCK_OPERATION = "delete_knowledge_base_data"


class MaintenanceConflict(RuntimeError):
    pass


@dataclass
class VectorReconcileStats:
    knowledge_base_id: str | None
    dry_run: bool = False
    checked_records: int = 0
    checked_collections: int = 0
    checked_collection_scopes: int = 0
    checked_unique_collections: int = 0
    missing_points: int = 0
    stale_points: int = 0
    deleted_points: int = 0
    proposed_records: int = 0
    marked_records: int = 0
    checked_point_payloads: int = 0
    payload_mismatch_points: int = 0
    identity_mismatch_points: int = 0
    owner_mismatch_points: int = 0
    vector_mismatch_points: int = 0
    collection_schema_mismatch_scopes: int = 0
    orphan_points: int = 0
    records_requiring_repair: int = 0
    scan_truncated_collection_scopes: int = 0
    processed_knowledge_bases: int = 0
    processed_record_batches: int = 0
    flushed_record_batches: int = 0
    committed_transactions: int = 0
    max_record_batch_size: int = 0
    max_chunk_lookup_ids: int = 0
    max_owner_lookup_ids: int = 0
    max_inventory_lookup_ids: int = 0
    diagnostic_sample_limit: int = VECTOR_RECONCILE_DIAGNOSTIC_SAMPLE_LIMIT
    sampled_stale_reason_values: int = 0
    omitted_stale_reason_records: int = 0
    omitted_stale_reason_values: int = 0
    sampled_orphan_point_ids: int = 0
    omitted_orphan_point_ids: int = 0
    omitted_collection_schema_error_scopes: int = 0
    caller_transaction_preserved: bool = False
    durable_progress_recorded: bool = False
    progress_persistence: str = "per_knowledge_base_commit_without_durable_job_cursor"
    restart_protocol: str = "idempotent_stable_keyset_restart_v1"
    stale_reasons_by_record: dict[str, list[str]] = field(default_factory=dict)
    orphan_point_ids_by_scope: dict[str, list[str]] = field(default_factory=dict)
    collection_schema_errors_by_scope: dict[str, str] = field(default_factory=dict)
    legacy_orphan_collection_inventory: dict[str, Any] = field(
        default_factory=lambda: {
            "scanned": False,
            "reason": "collections_without_postgresql_vector_records_require_separate_bounded_inventory",
        }
    )

    def as_dict(self) -> dict:
        payload = self.__dict__.copy()
        payload["diagnostics_truncated"] = bool(
            self.omitted_stale_reason_records
            or self.omitted_stale_reason_values
            or self.omitted_orphan_point_ids
            or self.omitted_collection_schema_error_scopes
        )
        return payload


def _bounded_chunks(values: Sequence[str], *, batch_size: int) -> Iterator[list[str]]:
    for offset in range(0, len(values), batch_size):
        yield list(values[offset : offset + batch_size])


def _sample_stale_reasons(
    stats: VectorReconcileStats,
    *,
    record_id: str,
    reasons: list[str],
) -> None:
    remaining = stats.diagnostic_sample_limit - stats.sampled_stale_reason_values
    if remaining <= 0 or len(stats.stale_reasons_by_record) >= stats.diagnostic_sample_limit:
        stats.omitted_stale_reason_records += 1
        stats.omitted_stale_reason_values += len(reasons)
        return
    sampled = reasons[:remaining]
    stats.stale_reasons_by_record[record_id] = sampled
    stats.sampled_stale_reason_values += len(sampled)
    stats.omitted_stale_reason_values += len(reasons) - len(sampled)


def _sample_orphan_ids(
    stats: VectorReconcileStats,
    *,
    scope_key: str,
    orphan_ids: set[str],
) -> None:
    remaining = stats.diagnostic_sample_limit - stats.sampled_orphan_point_ids
    sampled = sorted(orphan_ids)[: max(0, remaining)]
    if sampled:
        stats.orphan_point_ids_by_scope[scope_key] = sampled
        stats.sampled_orphan_point_ids += len(sampled)
    stats.omitted_orphan_point_ids += len(orphan_ids) - len(sampled)


def _sample_collection_schema_error(
    stats: VectorReconcileStats,
    *,
    scope_key: str,
    error: str,
) -> None:
    if len(stats.collection_schema_errors_by_scope) >= stats.diagnostic_sample_limit:
        stats.omitted_collection_schema_error_scopes += 1
        return
    stats.collection_schema_errors_by_scope[scope_key] = error[:1000]


def _merge_vector_reconcile_stats(
    aggregate: VectorReconcileStats,
    partial: dict[str, Any],
) -> None:
    for field_name in (
        "checked_records",
        "checked_collections",
        "checked_collection_scopes",
        "missing_points",
        "stale_points",
        "deleted_points",
        "proposed_records",
        "marked_records",
        "checked_point_payloads",
        "payload_mismatch_points",
        "identity_mismatch_points",
        "owner_mismatch_points",
        "vector_mismatch_points",
        "collection_schema_mismatch_scopes",
        "orphan_points",
        "records_requiring_repair",
        "scan_truncated_collection_scopes",
        "processed_knowledge_bases",
        "processed_record_batches",
        "flushed_record_batches",
    ):
        setattr(
            aggregate,
            field_name,
            int(getattr(aggregate, field_name)) + int(partial.get(field_name, 0)),
        )
    for field_name in (
        "max_record_batch_size",
        "max_chunk_lookup_ids",
        "max_owner_lookup_ids",
        "max_inventory_lookup_ids",
    ):
        setattr(
            aggregate,
            field_name,
            max(int(getattr(aggregate, field_name)), int(partial.get(field_name, 0))),
        )
    for record_id, reasons in dict(
        partial.get("stale_reasons_by_record") or {}
    ).items():
        _sample_stale_reasons(
            aggregate,
            record_id=str(record_id),
            reasons=[str(reason) for reason in reasons],
        )
    aggregate.omitted_stale_reason_records += int(
        partial.get("omitted_stale_reason_records", 0)
    )
    aggregate.omitted_stale_reason_values += int(
        partial.get("omitted_stale_reason_values", 0)
    )
    for scope_key, point_ids in dict(
        partial.get("orphan_point_ids_by_scope") or {}
    ).items():
        _sample_orphan_ids(
            aggregate,
            scope_key=str(scope_key),
            orphan_ids={str(point_id) for point_id in point_ids},
        )
    aggregate.omitted_orphan_point_ids += int(
        partial.get("omitted_orphan_point_ids", 0)
    )
    for scope_key, error in dict(
        partial.get("collection_schema_errors_by_scope") or {}
    ).items():
        _sample_collection_schema_error(
            aggregate,
            scope_key=str(scope_key),
            error=str(error),
        )
    aggregate.omitted_collection_schema_error_scopes += int(
        partial.get("omitted_collection_schema_error_scopes", 0)
    )


def _normalized_vector(value: Any, *, expected_dimension: int) -> list[float] | None:
    try:
        normalized = canonical_embedding_vector(value, source="Reconciled vector")
    except ValueError:
        return None
    return normalized if len(normalized) == expected_dimension else None


def _vectors_match(
    expected: Any,
    observed: Any,
    *,
    expected_dimension: int,
) -> bool:
    expected_vector = _normalized_vector(
        expected,
        expected_dimension=expected_dimension,
    )
    observed_vector = _normalized_vector(
        observed,
        expected_dimension=expected_dimension,
    )
    if expected_vector is None or observed_vector is None:
        return False
    return expected_vector == observed_vector


def _vector_record_point_reasons(
    record: VectorRecord,
    point: dict[str, Any],
    *,
    committed_owner: IngestionCompensationLog | None,
) -> list[str]:
    from app.services.context_graph import (
        QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION,
        QDRANT_VECTOR_DISTANCE_METRIC,
        VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION,
        qdrant_collection_identity_digest,
        qdrant_collection_name,
        vector_payload_hash,
    )
    from app.services.qdrant_outbox import (
        QDRANT_UPSERT_OPERATION,
        _canonical_hash,
        _validated_reconcile_payload,
    )

    reasons: list[str] = []
    diagnostics = dict(record.diagnostics_json or {})
    expected_dimension = int(record.embedding_dimension or 0)
    payload = point.get("payload")
    if not isinstance(payload, dict):
        return ["qdrant_payload_missing"]
    point_id = str(point.get("id") or "")
    if point_id != str(record.qdrant_point_id):
        reasons.append("qdrant_point_id_mismatch")
    expected_payload_fields = {
        "knowledge_base_id": str(record.knowledge_base_id),
        "chunk_id": str(record.chunk_id),
        "embedding_model": str(record.embedding_model),
        "embedding_dimension": int(record.embedding_dimension or 0),
        "embedding_text_version": str(record.embedding_text_version),
        "chunk_schema_version": str(record.chunk_schema_version),
        "vector_payload_hash": str(record.payload_hash),
    }
    for key, expected in expected_payload_fields.items():
        if payload.get(key) != expected:
            reasons.append(f"qdrant_{key}_mismatch")

    # These contextual-index fields are persisted in both PostgreSQL
    # diagnostics and the Qdrant payload.  Compare every mirrored field rather
    # than relying only on the vector identity fields below; otherwise a
    # coordinated diagnostics/payload drift could leave the point marked
    # ready even though the committed outbox target still describes the
    # original contextual index.
    for key in (
        "context_hash",
        "context_hash_protocol_version",
        "local_hint_protocol_version",
        "local_hint_hash",
    ):
        diagnostics_has_value = key in diagnostics
        payload_has_value = key in payload
        if not diagnostics_has_value:
            reasons.append(f"postgres_{key}_missing")
        if not payload_has_value:
            reasons.append(f"qdrant_{key}_missing")
        if (
            diagnostics_has_value
            and payload_has_value
            and payload.get(key) != diagnostics.get(key)
        ):
            reasons.append(f"qdrant_{key}_mismatch")

    identity_protocol = str(
        diagnostics.get("collection_identity_protocol_version") or ""
    )
    identity_digest = str(diagnostics.get("collection_identity_digest") or "")
    vector_hash_protocol = str(diagnostics.get("vector_payload_hash_protocol") or "")
    vector_distance_metric = str(diagnostics.get("vector_distance_metric") or "")
    diagnostics_dimension = diagnostics.get("embedding_dimension")
    diagnostics_chunk_schema_version = str(diagnostics.get("chunk_schema_version") or "")
    chunk_schema_version = str(payload.get("chunk_schema_version") or "")
    if not identity_protocol:
        reasons.append("postgres_collection_identity_protocol_missing")
    elif payload.get("collection_identity_protocol_version") != identity_protocol:
        reasons.append("qdrant_collection_identity_protocol_mismatch")
    if not identity_digest:
        reasons.append("postgres_collection_identity_digest_missing")
    elif payload.get("collection_identity_digest") != identity_digest:
        reasons.append("qdrant_collection_identity_digest_mismatch")
    if not vector_hash_protocol:
        reasons.append("postgres_vector_payload_hash_protocol_missing")
    elif payload.get("vector_payload_hash_protocol") != vector_hash_protocol:
        reasons.append("qdrant_vector_payload_hash_protocol_mismatch")
    if not vector_distance_metric:
        reasons.append("postgres_vector_distance_metric_missing")
    elif payload.get("vector_distance_metric") != vector_distance_metric:
        reasons.append("qdrant_vector_distance_metric_mismatch")
    if identity_protocol != QDRANT_COLLECTION_IDENTITY_PROTOCOL_VERSION:
        reasons.append("postgres_collection_identity_protocol_not_active")
    if vector_hash_protocol != VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION:
        reasons.append("postgres_vector_payload_hash_protocol_not_active")
    if vector_distance_metric != QDRANT_VECTOR_DISTANCE_METRIC:
        reasons.append("postgres_vector_distance_metric_not_active")
    if type(diagnostics_dimension) is not int or diagnostics_dimension != expected_dimension:
        reasons.append("postgres_embedding_dimension_mismatch")
    if not diagnostics_chunk_schema_version:
        reasons.append("postgres_chunk_schema_version_missing")
    elif diagnostics_chunk_schema_version != str(record.chunk_schema_version):
        reasons.append("postgres_chunk_schema_version_direct_identity_mismatch")
    elif diagnostics_chunk_schema_version != chunk_schema_version:
        reasons.append("qdrant_chunk_schema_version_mismatch")
    if not chunk_schema_version:
        reasons.append("qdrant_chunk_schema_version_missing")
    else:
        try:
            expected_identity_digest = qdrant_collection_identity_digest(
                embedding_model=str(record.embedding_model),
                embedding_dimension=int(record.embedding_dimension or 0),
                embedding_text_version=str(record.embedding_text_version),
                chunk_schema_version=chunk_schema_version,
            )
            expected_collection_name = qdrant_collection_name(
                embedding_model=str(record.embedding_model),
                embedding_dimension=int(record.embedding_dimension or 0),
                embedding_text_version=str(record.embedding_text_version),
                chunk_schema_version=chunk_schema_version,
            )
        except (TypeError, ValueError, RuntimeError):
            reasons.append("postgres_collection_identity_inputs_invalid")
        else:
            if identity_digest != expected_identity_digest:
                reasons.append("postgres_collection_identity_digest_not_canonical")
            if str(record.collection_name) != expected_collection_name:
                reasons.append("postgres_collection_identity_name_not_canonical")

    owner_intent_id = str(diagnostics.get("qdrant_write_intent_id") or "")
    owner_protocol = str(diagnostics.get("qdrant_write_protocol_version") or "")
    if not owner_intent_id:
        reasons.append("postgres_committed_owner_missing")
    elif payload.get("qdrant_write_intent_id") != owner_intent_id:
        reasons.append("qdrant_committed_owner_mismatch")
    if not owner_protocol:
        reasons.append("postgres_committed_owner_protocol_missing")
    elif payload.get("qdrant_write_protocol_version") != owner_protocol:
        reasons.append("qdrant_committed_owner_protocol_mismatch")

    committed_target: dict[str, Any] | None = None
    if owner_intent_id:
        if committed_owner is None:
            reasons.append("committed_owner_intent_missing")
        elif (
            committed_owner.operation != QDRANT_UPSERT_OPERATION
            or committed_owner.status != "committed"
            or str(committed_owner.knowledge_base_id) != str(record.knowledge_base_id)
            or str((committed_owner.payload_json or {}).get("collection_name") or "")
            != str(record.collection_name)
        ):
            reasons.append("committed_owner_intent_scope_mismatch")
        else:
            try:
                target_points, _before_points = _validated_reconcile_payload(committed_owner)
            except Exception:
                reasons.append("committed_owner_hash_envelope_invalid")
            else:
                target = next(
                    (
                        candidate
                        for candidate in target_points
                        if str(candidate.get("id") or "") == str(record.qdrant_point_id)
                    ),
                    None,
                )
                if target is None:
                    reasons.append("committed_owner_target_missing")
                else:
                    committed_target = dict(target)
                    if _canonical_hash(target.get("payload") or {}) != _canonical_hash(payload):
                        reasons.append("committed_owner_target_payload_hash_mismatch")

    diagnostics_vector = _normalized_vector(
        diagnostics.get("embedding_vector"),
        expected_dimension=expected_dimension,
    )
    actual_vector = _normalized_vector(
        point.get("vector"),
        expected_dimension=expected_dimension,
    )
    if diagnostics_vector is None:
        reasons.append("postgres_embedding_vector_invalid")
    else:
        try:
            diagnostics_payload_hash = vector_payload_hash(
                vector=diagnostics_vector,
                chunk_id=str(record.chunk_id),
                embedding_model=str(record.embedding_model),
                embedding_dimension=expected_dimension,
                vector_distance_metric=diagnostics.get("vector_distance_metric"),
                embedding_text_version=str(record.embedding_text_version),
                chunk_schema_version=diagnostics.get("chunk_schema_version"),
                context_hash_protocol_version=diagnostics.get(
                    "context_hash_protocol_version"
                ),
                context_hash=diagnostics.get("context_hash"),
                local_hint_protocol_version=diagnostics.get(
                    "local_hint_protocol_version"
                ),
                local_hint_hash=diagnostics.get("local_hint_hash"),
                collection_identity_protocol_version=diagnostics.get(
                    "collection_identity_protocol_version"
                ),
                collection_identity_digest=diagnostics.get("collection_identity_digest"),
            )
        except (TypeError, ValueError):
            reasons.append("postgres_vector_payload_hash_recompute_failed")
        else:
            if diagnostics_payload_hash != str(record.payload_hash):
                reasons.append("postgres_vector_payload_hash_recompute_mismatch")

    actual_payload_hash: str | None = None
    if actual_vector is None:
        reasons.append("qdrant_vector_invalid")
    else:
        try:
            actual_payload_hash = vector_payload_hash(
                vector=actual_vector,
                chunk_id=payload.get("chunk_id"),
                embedding_model=payload.get("embedding_model"),
                embedding_dimension=payload.get("embedding_dimension"),
                vector_distance_metric=payload.get("vector_distance_metric"),
                embedding_text_version=payload.get("embedding_text_version"),
                chunk_schema_version=payload.get("chunk_schema_version"),
                context_hash_protocol_version=payload.get(
                    "context_hash_protocol_version"
                ),
                context_hash=payload.get("context_hash"),
                local_hint_protocol_version=payload.get(
                    "local_hint_protocol_version"
                ),
                local_hint_hash=payload.get("local_hint_hash"),
                collection_identity_protocol_version=payload.get(
                    "collection_identity_protocol_version"
                ),
                collection_identity_digest=payload.get("collection_identity_digest"),
            )
        except (TypeError, ValueError):
            reasons.append("qdrant_vector_payload_hash_recompute_failed")

    target_matches_actual = False
    if committed_target is not None:
        committed_target_vector = _normalized_vector(
            committed_target.get("vector"),
            expected_dimension=expected_dimension,
        )
        if committed_target_vector is None:
            reasons.append("committed_owner_target_vector_invalid")
        else:
            try:
                committed_target_payload = committed_target.get("payload")
                if not isinstance(committed_target_payload, dict):
                    raise ValueError("committed target payload is missing")
                committed_target_payload_hash = vector_payload_hash(
                    vector=committed_target_vector,
                    chunk_id=committed_target_payload.get("chunk_id"),
                    embedding_model=committed_target_payload.get("embedding_model"),
                    embedding_dimension=committed_target_payload.get(
                        "embedding_dimension"
                    ),
                    vector_distance_metric=committed_target_payload.get(
                        "vector_distance_metric"
                    ),
                    embedding_text_version=committed_target_payload.get(
                        "embedding_text_version"
                    ),
                    chunk_schema_version=committed_target_payload.get(
                        "chunk_schema_version"
                    ),
                    context_hash_protocol_version=committed_target_payload.get(
                        "context_hash_protocol_version"
                    ),
                    context_hash=committed_target_payload.get("context_hash"),
                    local_hint_protocol_version=committed_target_payload.get(
                        "local_hint_protocol_version"
                    ),
                    local_hint_hash=committed_target_payload.get("local_hint_hash"),
                    collection_identity_protocol_version=committed_target_payload.get(
                        "collection_identity_protocol_version"
                    ),
                    collection_identity_digest=committed_target_payload.get(
                        "collection_identity_digest"
                    ),
                )
            except (TypeError, ValueError):
                reasons.append(
                    "committed_owner_target_vector_payload_hash_recompute_failed"
                )
            else:
                if committed_target_payload_hash != str(record.payload_hash):
                    reasons.append(
                        "committed_owner_target_vector_payload_hash_recompute_mismatch"
                    )
                if (
                    not isinstance(committed_target_payload, dict)
                    or committed_target_payload.get("vector_payload_hash")
                    != committed_target_payload_hash
                ):
                    reasons.append(
                        "committed_owner_target_vector_payload_hash_field_mismatch"
                    )
        target_matches_actual = _vectors_match(
            committed_target.get("vector"),
            point.get("vector"),
            expected_dimension=expected_dimension,
        )
        if not target_matches_actual:
            reasons.append("committed_owner_target_vector_mismatch")
        if not _vectors_match(
            committed_target.get("vector"),
            diagnostics.get("embedding_vector"),
            expected_dimension=expected_dimension,
        ):
            reasons.append("committed_owner_diagnostics_vector_mismatch")
    if actual_payload_hash is not None and actual_payload_hash != str(record.payload_hash):
        reasons.append("qdrant_vector_payload_hash_recompute_mismatch")

    if not _vectors_match(
        diagnostics.get("embedding_vector"),
        point.get("vector"),
        expected_dimension=expected_dimension,
    ):
        reasons.append("qdrant_vector_mismatch")
    return sorted(set(reasons))


def active_batch_for_knowledge_base(db: Session, knowledge_base_id: str) -> IngestionBatch | None:
    return db.scalar(
        select(IngestionBatch)
        .where(IngestionBatch.knowledge_base_id == knowledge_base_id, IngestionBatch.status.notin_(TERMINAL_BATCH_STATES))
        .order_by(IngestionBatch.created_at.desc())
    )


def ensure_no_active_batch(db: Session, knowledge_base_id: str) -> None:
    if active_batch_for_knowledge_base(db, knowledge_base_id) is not None:
        raise MaintenanceConflict("Cannot run maintenance while an ingestion batch is active")


@contextmanager
def _destructive_maintenance_lock(
    lock_context: ContextManager[object],
    *,
    operation: str,
) -> Iterator[None]:
    """Map only lock-acquisition contention to a retryable maintenance conflict."""

    from app.services.ingestion_resource_lock import IngestionResourceBusyError
    from app.services.qdrant_outbox import QdrantOutboxError

    stack = ExitStack()
    try:
        stack.enter_context(lock_context)
    except (IngestionResourceBusyError, QdrantOutboxError) as exc:
        stack.close()
        raise MaintenanceConflict(
            f"Cannot {operation} while its ingestion/Qdrant maintenance lock is busy; "
            f"retry after the active operation completes. Lock diagnostic: {exc}"
        ) from exc
    with stack:
        yield


def delete_knowledge_base_data(db: Session, knowledge_base: KnowledgeBase) -> dict:
    from app.services.ingestion import source_file_delete_resource_fence_diagnostics
    from app.services.ingestion_resource_lock import active_ingestion_resource_lease
    from app.services.qdrant_outbox import qdrant_outbox_reconcile_lock

    source_delete_fence = source_file_delete_resource_fence_diagnostics(
        db,
        knowledge_base_id=knowledge_base.id,
        operation=KNOWLEDGE_BASE_DELETE_LOCK_OPERATION,
        owner_intent_id=None,
    )
    if source_delete_fence is not None:
        raise MaintenanceConflict(
            "Cannot delete a knowledge base while a durable source-file deletion "
            f"requires recovery: {source_delete_fence}"
        )

    active_lease = active_ingestion_resource_lease(knowledge_base.id)
    if active_lease is not None:
        if active_lease.operation != KNOWLEDGE_BASE_DELETE_LOCK_OPERATION:
            raise MaintenanceConflict(
                "Cannot delete a knowledge base from inside a different ingestion resource operation"
            )
        # The Qdrant reconcile lock uses the same PostgreSQL advisory key on a
        # dedicated connection.  The async route already owns that key, so a
        # second try-lock would deadlock/fail against itself; the outer lease is
        # the stronger shared resource fence for this call.
        return _delete_knowledge_base_data_locked(db, knowledge_base)

    with _destructive_maintenance_lock(
        qdrant_outbox_reconcile_lock(db, knowledge_base.id),
        operation="delete a knowledge base",
    ):
        return _delete_knowledge_base_data_locked(db, knowledge_base)


def _delete_knowledge_base_data_locked(db: Session, knowledge_base: KnowledgeBase) -> dict:
    from app.services.qdrant_outbox import pending_qdrant_intents
    from app.services.storage_maintenance import (
        StorageMaintenanceIntegrityError,
        active_knowledge_base_delete_intent,
        applied_delete_intent_ids,
        apply_filesystem_delete_inventory,
        apply_qdrant_delete_inventory,
        complete_knowledge_base_delete_cache_invalidation,
        finalize_knowledge_base_facts,
        mark_delete_intent_failure,
        prepare_knowledge_base_delete_intent,
    )
    from app.services.upload_replacement import pending_upload_replacement_blockers

    upload_blockers = pending_upload_replacement_blockers(db, knowledge_base.id)
    if upload_blockers:
        blocker_summary = ", ".join(
            f"{blocker.get('intent_id')}:{blocker.get('status')}" for blocker in upload_blockers[:8]
        )
        if len(upload_blockers) > 8:
            blocker_summary = f"{blocker_summary}, +{len(upload_blockers) - 8} more"
        raise MaintenanceConflict(
            "Cannot delete a knowledge base while durable upload replacement intents remain active; "
            f"blockers={blocker_summary}. Run upload replacement reconciliation before retrying"
        )
    ensure_no_active_batch(db, knowledge_base.id)
    unresolved_upserts = pending_qdrant_intents(db, knowledge_base_id=knowledge_base.id)
    if unresolved_upserts:
        raise MaintenanceConflict(
            "Cannot delete a knowledge base while durable Qdrant upsert intents remain active; "
            "run reconciliation until transport outcomes are resolved"
        )
    row = active_knowledge_base_delete_intent(db, knowledge_base.id)
    facts_deleted = False
    try:
        row, payload = prepare_knowledge_base_delete_intent(db, knowledge_base)
        if payload["phase"] in {"intent_committed", "qdrant_deleting"}:
            payload, qdrant_delete_ids = apply_qdrant_delete_inventory(
                db,
                knowledge_base=knowledge_base,
                row=row,
                payload=payload,
            )
        elif payload["phase"] == "external_applied":
            qdrant_delete_ids = applied_delete_intent_ids(
                db,
                knowledge_base.id,
            )
        else:
            raise StorageMaintenanceIntegrityError(
                f"Knowledge-base delete cannot resume from phase {payload['phase']!r}"
            )

        if payload["phase"] != "external_applied":
            payload, filesystem_result = apply_filesystem_delete_inventory(
                db,
                row=row,
                payload=payload,
            )
        else:
            filesystem_result = {
                "deleted_files": 0,
                "deleted_directories": 0,
                "already_absent": True,
            }
        qdrant_delete_ids = applied_delete_intent_ids(db, knowledge_base.id)
        stats = {
            **dict(payload["database_counts"]),
            "qdrant_points": int(
                payload["qdrant_inventory"]["owned_point_count"]
            ),
            "qdrant_orphan_points": sum(
                int(card["orphan_point_count"])
                for card in payload["qdrant_inventory"]["collections"]
            ),
            "qdrant_owned_scope_hash": payload["qdrant_inventory"][
                "owned_scope_hash"
            ],
            "filesystem_inventory_hash": payload["filesystem_inventory"][
                "inventory_hash"
            ],
            "filesystem_deleted_files": int(
                filesystem_result["deleted_files"]
            ),
            "filesystem_deleted_directories": int(
                filesystem_result["deleted_directories"]
            ),
            "storage_maintenance_intent_id": row.id,
        }
        payload = finalize_knowledge_base_facts(
            db,
            knowledge_base=knowledge_base,
            row=row,
            payload=payload,
            qdrant_delete_intent_ids=qdrant_delete_ids,
        )
        facts_deleted = True
    except Exception as exc:
        if row is not None and not facts_deleted:
            mark_delete_intent_failure(db, row_id=row.id, error=exc)
        if isinstance(exc, StorageMaintenanceIntegrityError):
            raise MaintenanceConflict(str(exc)) from exc
        raise

    try:
        complete_knowledge_base_delete_cache_invalidation(
            db,
            row=row,
            payload=payload,
        )
    except Exception as exc:
        db.rollback()
        pending = db.get(type(row), row.id)
        if pending is None:
            raise MaintenanceConflict(
                "Knowledge-base facts were deleted but the durable cache-retry tombstone disappeared"
            ) from exc
        pending.error_message = exc.__class__.__name__
        db.commit()
        raise MaintenanceConflict(
            "Knowledge-base facts and external data were deleted, but strict "
            "cache invalidation remains pending and will be retried"
        ) from exc
    return stats


@contextmanager
def cleanup_stale_data_lock(
    db: Session,
    knowledge_base_id: str,
) -> Iterator[None]:
    """Hold the same cross-process KB fence used by stale-data cleanup."""

    from app.services.qdrant_outbox import qdrant_outbox_reconcile_lock

    with qdrant_outbox_reconcile_lock(db, knowledge_base_id):
        yield


def cleanup_stale_data(
    db: Session,
    knowledge_base_id: str,
    knowledge_base_name: str,
    *,
    dry_run: bool = True,
    delete_inactive_chunks: bool = False,
    lock_already_held: bool = False,
    pre_mutation_check: Callable[[], None] | None = None,
) -> dict:
    lock_context = (
        nullcontext()
        if lock_already_held
        else cleanup_stale_data_lock(db, knowledge_base_id)
    )
    with lock_context:
        if not dry_run and pre_mutation_check is not None:
            pre_mutation_check()
        if dry_run:
            # A preview must neither flush unrelated caller changes nor issue
            # transient maintenance UPDATEs that need a broad Session rollback.
            with db.no_autoflush:
                return _cleanup_stale_data_locked(
                    db,
                    knowledge_base_id,
                    knowledge_base_name,
                    dry_run=True,
                    delete_inactive_chunks=delete_inactive_chunks,
                )
        return _cleanup_stale_data_locked(
            db,
            knowledge_base_id,
            knowledge_base_name,
            dry_run=dry_run,
            delete_inactive_chunks=delete_inactive_chunks,
        )


def _cleanup_stale_data_locked(
    db: Session,
    knowledge_base_id: str,
    knowledge_base_name: str,
    *,
    dry_run: bool = True,
    delete_inactive_chunks: bool = False,
) -> dict:
    from app.services.qdrant_outbox import (
        mark_qdrant_delete_attempts_committed,
        pending_qdrant_delete_intents,
        persist_qdrant_delete_attempt,
        qdrant_delete_intent_recovery_diagnostics,
        qdrant_point_mutation_lock,
        record_qdrant_delete_attempt_applied,
        record_qdrant_delete_attempt_error,
        replay_qdrant_delete_intent,
    )

    ensure_no_active_batch(db, knowledge_base_id)
    knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
    current_chunk_version = knowledge_base.current_chunk_version if knowledge_base else 0
    if not dry_run:
        db.execute(
            update(ChunkVersion)
            .where(ChunkVersion.knowledge_base_id == knowledge_base_id, ChunkVersion.chunk_version != current_chunk_version, ChunkVersion.state == "active")
            .values(state="inactive")
        )
        db.execute(
            update(ChunkVersion)
            .where(ChunkVersion.knowledge_base_id == knowledge_base_id, ChunkVersion.chunk_version == current_chunk_version, ChunkVersion.state != "active")
            .values(state="active")
        )
    inactive_chunks = db.scalars(select(Chunk.id).where(Chunk.knowledge_base_id == knowledge_base_id, Chunk.state != "active")).all()
    inactive_chunk_ids = set(inactive_chunks)
    inactive_document_version_ids = {
        row[0]
        for row in db.execute(
            select(DocumentVersion.id)
            .join(Document, DocumentVersion.document_id == Document.id)
            .where(Document.knowledge_base_id == knowledge_base_id, DocumentVersion.is_active.is_(False))
        ).all()
    }
    inactive_chunk_version_ids = {
        row[0]
        for row in db.execute(
            select(ChunkVersion.id).where(
                ChunkVersion.knowledge_base_id == knowledge_base_id,
                ChunkVersion.chunk_version != current_chunk_version,
            )
        ).all()
    }
    inactive_chunk_id_filter = inactive_chunk_ids or {"__none__"}
    inactive_document_version_filter = inactive_document_version_ids or {"__none__"}
    stale_records = db.scalars(select(VectorRecord).where(VectorRecord.knowledge_base_id == knowledge_base_id, VectorRecord.chunk_id.in_(inactive_chunk_ids or {"__none__"}))).all()
    points_by_collection: dict[str, list[str]] = defaultdict(list)
    for record in stale_records:
        points_by_collection[record.collection_name].append(record.qdrant_point_id)
    active_deletes = pending_qdrant_delete_intents(
        db,
        knowledge_base_id=knowledge_base_id,
    )
    resumed_by_collection: dict[str, IngestionCompensationLog] = {}
    for row in active_deletes:
        recovery = qdrant_delete_intent_recovery_diagnostics(row)
        if not recovery["retryable"]:
            raise MaintenanceConflict(
                "Cannot automatically resume an unfenced Qdrant delete intent; "
                f"intent={row.id}, recovery={recovery['retry_guidance']}"
            )
        payload = dict(row.payload_json or {})
        collection_name = str(payload.get("collection_name") or "")
        if (
            payload.get("reason") != "cleanup_stale_data"
            or collection_name not in points_by_collection
            or sorted(str(item) for item in (row.target_ids_json or []))
            != sorted(points_by_collection[collection_name])
            or collection_name in resumed_by_collection
        ):
            raise MaintenanceConflict(
                "Cannot resume stale-data cleanup because its durable Qdrant delete scope no longer matches"
            )
        resumed_by_collection[collection_name] = row
    if not dry_run:
        for collection_name, row in resumed_by_collection.items():
            if row.status == "external_applied":
                continue
            try:
                replay_qdrant_delete_intent(
                    db,
                    store=VectorStore(knowledge_base_name, collection_name=collection_name),
                    row=row,
                )
            except Exception as exc:
                raise MaintenanceConflict(
                    f"Cannot safely replay durable Qdrant stale cleanup intent {row.id}: {exc}"
                ) from exc
    elif any(row.status != "external_applied" for row in resumed_by_collection.values()):
        raise MaintenanceConflict(
            "Dry-run found a replayable Qdrant delete intent; execute stale cleanup to consume it"
        )
    inactive_counts = {
        "chunk_spans": db.scalar(select(func.count(ChunkSpan.id)).where(ChunkSpan.chunk_id.in_(inactive_chunk_id_filter))) or 0,
        "chunk_coordinates": db.scalar(select(func.count(ChunkCoordinate.id)).where(ChunkCoordinate.chunk_id.in_(inactive_chunk_id_filter))) or 0,
        "chunk_context_texts": db.scalar(select(func.count(ChunkContextText.id)).where(ChunkContextText.chunk_id.in_(inactive_chunk_id_filter))) or 0,
        "chunk_structure_mappings": db.scalar(select(func.count(ChunkStructureMapping.id)).where(ChunkStructureMapping.chunk_id.in_(inactive_chunk_id_filter))) or 0,
        "chunk_relation_edges": db.scalar(
            select(func.count(ChunkRelationEdge.id)).where(
                (ChunkRelationEdge.source_chunk_id.in_(inactive_chunk_id_filter)) | (ChunkRelationEdge.target_chunk_id.in_(inactive_chunk_id_filter))
            )
        )
        or 0,
        "rq_prefix_memberships": db.scalar(select(func.count(RQPrefixMembership.id)).where(RQPrefixMembership.chunk_id.in_(inactive_chunk_id_filter))) or 0,
        "chunk_structure_edges": db.scalar(select(func.count(ChunkStructureEdge.id)).where(ChunkStructureEdge.document_version_id.in_(inactive_document_version_filter))) or 0,
        "chunk_structure_nodes": db.scalar(select(func.count(ChunkStructureNode.id)).where(ChunkStructureNode.document_version_id.in_(inactive_document_version_filter))) or 0,
    }
    deleted_inactive = {
        "chunks": 0,
        "document_versions": 0,
        "chunk_versions": 0,
    }
    if not dry_run:
        delete_intent_ids: list[str] = []
        for collection_name, point_ids in points_by_collection.items():
            resumed = resumed_by_collection.get(collection_name)
            if resumed is not None:
                delete_intent_ids.append(resumed.id)
                continue
            store = VectorStore(knowledge_base_name, collection_name=collection_name)
            with qdrant_point_mutation_lock(
                db,
                collection_name=collection_name,
                point_ids=point_ids,
            ):
                current_points = store.get_points(point_ids)
                delete_intent_id = persist_qdrant_delete_attempt(
                    db,
                    knowledge_base_id=knowledge_base_id,
                    collection_name=collection_name,
                    target_ids=point_ids,
                    current_points=current_points,
                    reason="cleanup_stale_data",
                )
                try:
                    store.delete_if_payload_matches(current_points)
                except Exception as exc:
                    record_qdrant_delete_attempt_error(
                        db,
                        intent_id=delete_intent_id,
                        error=exc,
                    )
                    raise
                record_qdrant_delete_attempt_applied(
                    db,
                    intent_id=delete_intent_id,
                )
                delete_intent_ids.append(delete_intent_id)
        if stale_records:
            db.query(VectorRecord).filter(VectorRecord.id.in_([record.id for record in stale_records])).delete(synchronize_session=False)
        if delete_inactive_chunks and inactive_chunk_ids:
            db.execute(update(CitationVerification).where(CitationVerification.chunk_id.in_(inactive_chunk_id_filter)).values(chunk_id=None))
            db.query(ChunkRelationEdge).filter(
                (ChunkRelationEdge.source_chunk_id.in_(inactive_chunk_id_filter)) | (ChunkRelationEdge.target_chunk_id.in_(inactive_chunk_id_filter))
            ).delete(synchronize_session=False)
            db.query(RQPrefixMembership).filter(RQPrefixMembership.chunk_id.in_(inactive_chunk_id_filter)).delete(synchronize_session=False)
            db.query(ChunkStructureMapping).filter(ChunkStructureMapping.chunk_id.in_(inactive_chunk_id_filter)).delete(synchronize_session=False)
            db.query(ChunkContextText).filter(ChunkContextText.chunk_id.in_(inactive_chunk_id_filter)).delete(synchronize_session=False)
            db.query(ChunkCoordinate).filter(ChunkCoordinate.chunk_id.in_(inactive_chunk_id_filter)).delete(synchronize_session=False)
            db.query(ChunkSpan).filter(ChunkSpan.chunk_id.in_(inactive_chunk_id_filter)).delete(synchronize_session=False)
            deleted_inactive["chunks"] = db.query(Chunk).filter(Chunk.id.in_(inactive_chunk_id_filter)).delete(synchronize_session=False)
        if delete_inactive_chunks and inactive_document_version_ids:
            db.query(ChunkStructureEdge).filter(ChunkStructureEdge.document_version_id.in_(inactive_document_version_filter)).delete(synchronize_session=False)
            db.query(ChunkStructureNode).filter(ChunkStructureNode.document_version_id.in_(inactive_document_version_filter)).delete(synchronize_session=False)
            deleted_inactive["document_versions"] = db.query(DocumentVersion).filter(DocumentVersion.id.in_(inactive_document_version_filter)).delete(synchronize_session=False)
        if delete_inactive_chunks and inactive_chunk_version_ids:
            deleted_inactive["chunk_versions"] = db.query(ChunkVersion).filter(ChunkVersion.id.in_(inactive_chunk_version_ids)).delete(synchronize_session=False)
        mark_qdrant_delete_attempts_committed(
            db,
            intent_ids=delete_intent_ids,
        )
        db.commit()
    return {
        "inactive_chunks": len(inactive_chunk_ids),
        "inactive_document_versions": len(inactive_document_version_ids),
        "inactive_chunk_versions": len(inactive_chunk_version_ids),
        "stale_vector_records": len(stale_records),
        "stale_qdrant_points": sum(len(ids) for ids in points_by_collection.values()),
        "collections": sorted(points_by_collection),
        "delete_inactive_chunks": delete_inactive_chunks,
        "inactive_dependents": inactive_counts,
        "deleted_inactive": deleted_inactive,
        "applied": not dry_run,
    }


def reconcile_vector_store_sync(
    db: Session,
    knowledge_base_id: str | None = None,
    *,
    dry_run: bool = False,
    batch_size: int = VECTOR_RECONCILE_BATCH_SIZE,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    lock_already_held: bool = False,
    _dry_run_guarded: bool = False,
) -> dict:
    """Reconcile PostgreSQL vector state using bounded, deterministic keysets.

    A fresh session is owned and committed by this function for compatibility
    with the worker task.  If the caller already has a transaction, only
    bounded flushes are performed and the caller retains commit/rollback
    ownership.  Dry-run uses ``no_autoflush`` and never changes ORM state.
    """

    bounded_batch_size = max(1, min(int(batch_size), VECTOR_RECONCILE_BATCH_SIZE))
    if dry_run and not _dry_run_guarded:
        no_autoflush = getattr(db, "no_autoflush", nullcontext())
        with no_autoflush:
            return reconcile_vector_store_sync(
                db,
                knowledge_base_id,
                dry_run=True,
                batch_size=bounded_batch_size,
                progress_callback=progress_callback,
                lock_already_held=lock_already_held,
                _dry_run_guarded=True,
            )
    in_transaction = getattr(db, "in_transaction", None)
    owns_transaction = not bool(in_transaction()) if callable(in_transaction) else True

    if knowledge_base_id is None:
        from app.services.qdrant_outbox import qdrant_outbox_reconcile_lock

        aggregate = VectorReconcileStats(
            knowledge_base_id=None,
            dry_run=bool(dry_run),
            caller_transaction_preserved=not owns_transaction,
        )
        unique_collection_query = select(
            func.count(func.distinct(VectorRecord.collection_name))
        )
        aggregate.checked_unique_collections = int(
            db.scalar(unique_collection_query) or 0
        )
        knowledge_base_cursor: str | None = None
        try:
            while True:
                knowledge_base_query = select(KnowledgeBase.id).where(
                    select(VectorRecord.id)
                    .where(
                        VectorRecord.knowledge_base_id == KnowledgeBase.id
                    )
                    .exists()
                )
                if knowledge_base_cursor is not None:
                    knowledge_base_query = knowledge_base_query.where(
                        KnowledgeBase.id > knowledge_base_cursor
                    )
                row = db.execute(
                    knowledge_base_query.order_by(KnowledgeBase.id).limit(1)
                ).first()
                if row is None:
                    break
                scoped_knowledge_base_id = str(row[0])
                knowledge_base_cursor = scoped_knowledge_base_id
                with qdrant_outbox_reconcile_lock(db, scoped_knowledge_base_id):
                    try:
                        partial = reconcile_vector_store_sync(
                            db,
                            scoped_knowledge_base_id,
                            dry_run=dry_run,
                            batch_size=bounded_batch_size,
                            progress_callback=None,
                            lock_already_held=True,
                            _dry_run_guarded=_dry_run_guarded,
                        )
                        _merge_vector_reconcile_stats(aggregate, partial)
                        if not dry_run and owns_transaction:
                            db.commit()
                            aggregate.committed_transactions += 1
                    except Exception:
                        if not dry_run and owns_transaction:
                            db.rollback()
                        raise
                if progress_callback is not None:
                    progress_callback(
                        {
                            "knowledge_base_id": scoped_knowledge_base_id,
                            "processed_knowledge_bases": aggregate.processed_knowledge_bases,
                            "processed_record_batches": aggregate.processed_record_batches,
                            "checked_records": aggregate.checked_records,
                            "proposed_records": aggregate.proposed_records,
                            "marked_records": aggregate.marked_records,
                            "dry_run": aggregate.dry_run,
                        }
                    )
            if dry_run and owns_transaction:
                db.rollback()
            elif owns_transaction and aggregate.committed_transactions == 0:
                # The aggregate/count query opened a read transaction even
                # when no vector-bearing knowledge base existed.
                db.rollback()
        except Exception:
            if not dry_run and owns_transaction:
                db.rollback()
            raise
        return aggregate.as_dict()

    if not lock_already_held:
        from app.services.qdrant_outbox import qdrant_outbox_reconcile_lock

        with qdrant_outbox_reconcile_lock(db, knowledge_base_id):
            return reconcile_vector_store_sync(
                db,
                knowledge_base_id,
                dry_run=dry_run,
                batch_size=bounded_batch_size,
                progress_callback=progress_callback,
                lock_already_held=True,
                _dry_run_guarded=_dry_run_guarded,
            )

    stats = VectorReconcileStats(
        knowledge_base_id=knowledge_base_id,
        dry_run=bool(dry_run),
    )
    stats.caller_transaction_preserved = not owns_transaction
    no_autoflush = nullcontext()

    def _emit_progress(*, scoped_kb_id: str) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "knowledge_base_id": scoped_kb_id,
                "processed_record_batches": stats.processed_record_batches,
                "checked_records": stats.checked_records,
                "proposed_records": stats.proposed_records,
                "marked_records": stats.marked_records,
                "dry_run": stats.dry_run,
            }
        )

    try:
        with no_autoflush:
            unique_collection_query = select(
                func.count(func.distinct(VectorRecord.collection_name))
            )
            if knowledge_base_id:
                unique_collection_query = unique_collection_query.where(
                    VectorRecord.knowledge_base_id == knowledge_base_id
                )
            stats.checked_unique_collections = int(
                db.scalar(unique_collection_query) or 0
            )

            knowledge_base_cursor: str | None = None
            while True:
                knowledge_base_query = select(KnowledgeBase.id, KnowledgeBase.name)
                if knowledge_base_id:
                    if stats.processed_knowledge_bases:
                        break
                    knowledge_base_query = knowledge_base_query.where(
                        KnowledgeBase.id == knowledge_base_id
                    )
                elif knowledge_base_cursor is not None:
                    knowledge_base_query = knowledge_base_query.where(
                        KnowledgeBase.id > knowledge_base_cursor
                    )
                knowledge_base_row = db.execute(
                    knowledge_base_query.order_by(KnowledgeBase.id).limit(1)
                ).first()
                if knowledge_base_row is None:
                    break
                scoped_knowledge_base_id = str(knowledge_base_row[0])
                knowledge_base_name = str(knowledge_base_row[1])
                knowledge_base_cursor = scoped_knowledge_base_id
                stats.processed_knowledge_bases += 1

                cursor_collection: str | None = None
                cursor_record_id: str | None = None
                active_collection: str | None = None
                active_store: VectorStore | None = None
                active_scope_reasons: list[str] = []
                active_expected_vector_size: int | None = None
                active_scope_initialized = False

                while True:
                    record_query = select(VectorRecord).where(
                        VectorRecord.knowledge_base_id == scoped_knowledge_base_id
                    )
                    if cursor_collection is not None and cursor_record_id is not None:
                        record_query = record_query.where(
                            or_(
                                VectorRecord.collection_name > cursor_collection,
                                and_(
                                    VectorRecord.collection_name == cursor_collection,
                                    VectorRecord.id > cursor_record_id,
                                ),
                            )
                        )
                    records = list(
                        db.scalars(
                            record_query.order_by(
                                VectorRecord.collection_name,
                                VectorRecord.id,
                            ).limit(bounded_batch_size)
                        )
                    )
                    if not records:
                        break
                    stats.processed_record_batches += 1
                    stats.checked_records += len(records)
                    stats.max_record_batch_size = max(
                        stats.max_record_batch_size,
                        len(records),
                    )
                    marked_before_batch = stats.marked_records

                    segment_start = 0
                    while segment_start < len(records):
                        collection_name = str(records[segment_start].collection_name)
                        segment_end = segment_start + 1
                        while (
                            segment_end < len(records)
                            and str(records[segment_end].collection_name) == collection_name
                        ):
                            segment_end += 1
                        collection_records = records[segment_start:segment_end]

                        if active_collection != collection_name:
                            active_collection = collection_name
                            active_store = VectorStore(
                                knowledge_base_name,
                                collection_name=collection_name,
                                # Reconciliation is observation plus PostgreSQL
                                # status repair. Collection creation belongs to
                                # the production writer/outbox path.
                                create_if_missing=False,
                            )
                            scope_dimensions = {
                                int(value or 0)
                                for value in db.scalars(
                                    select(VectorRecord.embedding_dimension)
                                    .where(
                                        VectorRecord.knowledge_base_id
                                        == scoped_knowledge_base_id,
                                        VectorRecord.collection_name == collection_name,
                                    )
                                    .distinct()
                                    .order_by(VectorRecord.embedding_dimension)
                                    .limit(2)
                                )
                            }
                            active_expected_vector_size = (
                                next(iter(scope_dimensions))
                                if len(scope_dimensions) == 1
                                else None
                            )
                            active_scope_reasons = []
                            if len(scope_dimensions) != 1:
                                active_scope_reasons.append(
                                    "postgres_collection_scope_mixed_embedding_dimensions"
                                )
                            active_scope_initialized = False
                            stats.checked_collections += 1
                            stats.checked_collection_scopes += 1

                        expected_point_ids = sorted(
                            {
                                str(record.qdrant_point_id)
                                for record in collection_records
                            }
                        )
                        assert active_store is not None
                        if not active_scope_initialized:
                            snapshot = active_store.reconciliation_snapshot(
                                scoped_knowledge_base_id,
                                expected_point_ids,
                                expected_vector_size=active_expected_vector_size,
                            )
                            active_scope_initialized = True
                            schema_error = str(
                                snapshot.get("collection_schema_error") or ""
                            )
                            scope_key = (
                                f"{collection_name}:{scoped_knowledge_base_id}"
                            )
                            if schema_error:
                                active_scope_reasons.append(
                                    "qdrant_collection_schema_mismatch"
                                )
                                stats.collection_schema_mismatch_scopes += 1
                                _sample_collection_schema_error(
                                    stats,
                                    scope_key=scope_key,
                                    error=schema_error,
                                )
                            scanned_ids = {
                                str(point_id)
                                for point_id in snapshot.get("scanned_ids") or []
                                if str(point_id)
                            }
                            known_inventory_ids: set[str] = set()
                            for inventory_batch in _bounded_chunks(
                                sorted(scanned_ids),
                                batch_size=bounded_batch_size,
                            ):
                                stats.max_inventory_lookup_ids = max(
                                    stats.max_inventory_lookup_ids,
                                    len(inventory_batch),
                                )
                                known_inventory_ids.update(
                                    str(value)
                                    for value in db.scalars(
                                        select(VectorRecord.qdrant_point_id).where(
                                            VectorRecord.knowledge_base_id
                                            == scoped_knowledge_base_id,
                                            VectorRecord.collection_name
                                            == collection_name,
                                            VectorRecord.qdrant_point_id.in_(
                                                inventory_batch
                                            ),
                                        )
                                    )
                                )
                            orphan_ids = scanned_ids - known_inventory_ids
                            stats.orphan_points += len(orphan_ids)
                            stats.stale_points += len(orphan_ids)
                            if orphan_ids:
                                _sample_orphan_ids(
                                    stats,
                                    scope_key=scope_key,
                                    orphan_ids=orphan_ids,
                                )
                            if snapshot.get("scan_truncated"):
                                stats.scan_truncated_collection_scopes += 1
                            observed_points = list(snapshot.get("points") or [])
                        else:
                            observed_points = active_store.get_points_batched(
                                expected_point_ids,
                                batch_size=bounded_batch_size,
                            )

                        points_by_id = {
                            str(point.get("id") or ""): dict(point)
                            for point in observed_points
                            if str(point.get("id") or "")
                        }
                        missing = set(expected_point_ids) - set(points_by_id)
                        stats.missing_points += len(missing)

                        chunk_ids = list(
                            dict.fromkeys(
                                str(record.chunk_id)
                                for record in collection_records
                            )
                        )
                        stats.max_chunk_lookup_ids = max(
                            stats.max_chunk_lookup_ids,
                            len(chunk_ids),
                        )
                        active_chunk_ids = {
                            str(value)
                            for value in db.scalars(
                                select(Chunk.id).where(
                                    Chunk.id.in_(chunk_ids),
                                    Chunk.knowledge_base_id
                                    == scoped_knowledge_base_id,
                                    Chunk.state == "active",
                                )
                            )
                        }
                        owner_intent_ids = list(
                            dict.fromkeys(
                                owner_intent_id
                                for owner_intent_id in (
                                    str(
                                        (record.diagnostics_json or {}).get(
                                            "qdrant_write_intent_id"
                                        )
                                        or ""
                                    )
                                    for record in collection_records
                                )
                                if owner_intent_id
                            )
                        )
                        stats.max_owner_lookup_ids = max(
                            stats.max_owner_lookup_ids,
                            len(owner_intent_ids),
                        )
                        committed_owners = {
                            str(row.id): row
                            for row in db.scalars(
                                select(IngestionCompensationLog).where(
                                    IngestionCompensationLog.id.in_(
                                        owner_intent_ids or ["__none__"]
                                    )
                                )
                            )
                        }

                        stale_point_ids: set[str] = set()
                        payload_mismatch_ids: set[str] = set()
                        identity_mismatch_ids: set[str] = set()
                        owner_mismatch_ids: set[str] = set()
                        vector_mismatch_ids: set[str] = set()
                        for record in collection_records:
                            if record.vector_status in {"shadow_pending", "shadow_ready"}:
                                desired = "shadow_ready"
                            elif record.vector_status in {
                                "rollback_retained",
                                "rolled_back_retained",
                            }:
                                desired = record.vector_status
                            else:
                                desired = "ready"
                            reasons: list[str] = []
                            point_id = str(record.qdrant_point_id)
                            if point_id in missing:
                                desired = "missing"
                                reasons.append("qdrant_point_missing")
                            else:
                                stats.checked_point_payloads += 1
                                owner_intent_id = str(
                                    (record.diagnostics_json or {}).get(
                                        "qdrant_write_intent_id"
                                    )
                                    or ""
                                )
                                reasons.extend(
                                    _vector_record_point_reasons(
                                        record,
                                        points_by_id[point_id],
                                        committed_owner=committed_owners.get(
                                            owner_intent_id
                                        ),
                                    )
                                )
                            diagnostics = dict(record.diagnostics_json or {})
                            if desired == "shadow_ready":
                                required_shadow_bindings = {
                                    "runtime_settings_candidate_id",
                                    "runtime_settings_candidate_hash",
                                    "vector_shadow_build_id",
                                    "candidate_vector_schema_hash",
                                    "vector_shadow_writer_protocol_version",
                                }
                                missing_shadow_bindings = sorted(
                                    key
                                    for key in required_shadow_bindings
                                    if not diagnostics.get(key)
                                )
                                if missing_shadow_bindings:
                                    reasons.extend(
                                        f"shadow_binding_missing:{key}"
                                        for key in missing_shadow_bindings
                                    )
                            if str(record.chunk_id) not in active_chunk_ids:
                                reasons.append("chunk_not_active")
                            reasons.extend(active_scope_reasons)
                            reasons = sorted(set(reasons))
                            if desired != "missing" and reasons:
                                desired = "stale"
                            if desired != "ready":
                                stats.records_requiring_repair += 1
                                _sample_stale_reasons(
                                    stats,
                                    record_id=str(record.id),
                                    reasons=reasons,
                                )
                            if desired == "stale":
                                stale_point_ids.add(point_id)
                                if any("payload" in reason for reason in reasons):
                                    payload_mismatch_ids.add(point_id)
                                if any("identity" in reason for reason in reasons):
                                    identity_mismatch_ids.add(point_id)
                                if any("owner" in reason for reason in reasons):
                                    owner_mismatch_ids.add(point_id)
                                if any(
                                    reason
                                    in {
                                        "postgres_embedding_vector_invalid",
                                        "qdrant_vector_invalid",
                                        "qdrant_vector_mismatch",
                                    }
                                    or reason.startswith(
                                        "postgres_vector_payload_hash_recompute_"
                                    )
                                    or reason.startswith(
                                        "qdrant_vector_payload_hash_recompute_"
                                    )
                                    or reason.startswith(
                                        "committed_owner_target_vector_"
                                    )
                                    or reason
                                    == "committed_owner_diagnostics_vector_mismatch"
                                    for reason in reasons
                                ):
                                    vector_mismatch_ids.add(point_id)
                            if record.vector_status != desired:
                                stats.proposed_records += 1
                                if not dry_run:
                                    record.vector_status = desired
                                    stats.marked_records += 1
                        stats.stale_points += len(stale_point_ids)
                        stats.payload_mismatch_points += len(payload_mismatch_ids)
                        stats.identity_mismatch_points += len(identity_mismatch_ids)
                        stats.owner_mismatch_points += len(owner_mismatch_ids)
                        stats.vector_mismatch_points += len(vector_mismatch_ids)
                        segment_start = segment_end

                    if not dry_run and stats.marked_records > marked_before_batch:
                        db.flush()
                        stats.flushed_record_batches += 1
                    _emit_progress(scoped_kb_id=scoped_knowledge_base_id)
                    cursor_collection = str(records[-1].collection_name)
                    cursor_record_id = str(records[-1].id)

                if knowledge_base_id:
                    break

        if not dry_run and owns_transaction:
            db.commit()
            stats.committed_transactions = 1
        elif dry_run and owns_transaction:
            # Release the read transaction opened by the bounded scan without
            # touching a transaction that belonged to the caller.
            db.rollback()
    except Exception:
        if not dry_run and owns_transaction:
            rollback = getattr(db, "rollback", None)
            if callable(rollback):
                rollback()
        raise
    return stats.as_dict()


def reconcile_policy_state(db: Session, knowledge_base_id: str) -> int:
    from app.services.context_graph import agent_operating_envelope, agent_operating_envelope_state_hash, runtime_settings_state_hash

    latest = db.scalar(
        select(PolicyState)
        .where(PolicyState.knowledge_base_id == knowledge_base_id, PolicyState.policy_family == "context_graph_bandit")
        .order_by(PolicyState.created_at.desc())
    )
    arms = [
        "high_precision_direct_chunk",
        "structure_context_heavy",
        "rq_membership_expansion",
        "mid_concept_expansion",
        "coarse_to_mid_drilldown",
        "bridge_edge_exploration",
        "formula_table_closure",
        "cross_document_synthesis",
        "low_latency_minimal_context",
    ]
    weights = {arm: 1.0 for arm in arms}
    safe_arms = list(arms)
    reward_summary = {
        "origin": "seed",
        "previous_policy_state_hash": None,
        "safe_arms": safe_arms,
        "posterior": weights,
        "exploration_rate": 0.05,
        "drift_status": "normal",
        "policy_version": "context_graph_bandit_v1",
        "runtime_settings_hash": runtime_settings_state_hash(),
        "agent_operating_envelope_hash": agent_operating_envelope_state_hash(),
        "reward_history_tail": [],
    }
    constraints = {
        "fallback_disabled": True,
        "citation_verification_required": True,
        "agent_operating_envelope": agent_operating_envelope(),
        "runtime_settings_hash": runtime_settings_state_hash(),
        "planner_replacement": False,
        "gray_zone_decision_authority": False,
        "gray_zone_rule_inputs_modified": False,
        "gray_zone_model_call_count": 0,
    }
    exploration = {
        "epsilon": 0.05,
        "exploration_rate": 0.05,
        "safe_arms": safe_arms,
        "threshold_suggestions_runtime_lifecycle_accepted": False,
        "threshold_suggestions_applied": False,
        "gray_zone_decision_authority": False,
        "gray_zone_model_call_count": 0,
    }
    state_hash = canonical_policy_state_hash(
        policy_family="context_graph_bandit",
        policy_version="context_graph_bandit_v1",
        profile_objective_hash=None,
        weights=weights,
        constraints=constraints,
        exploration=exploration,
        reward_summary=reward_summary,
    )
    if latest and latest.state_hash == state_hash:
        return 0
    db.add(
        PolicyState(
            knowledge_base_id=knowledge_base_id,
            policy_family="context_graph_bandit",
            policy_version="context_graph_bandit_v1",
            weights_json=weights,
            constraints_json=constraints,
            exploration_json=exploration,
            reward_summary_json=reward_summary,
            state_hash=state_hash,
        )
    )
    return 1
